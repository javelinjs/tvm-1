/*!
 * Copyright (c) 2018 by Contributors
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/tvm.h>
#include <tuple>
#include <utility>
#include <vector>
#include <string>

#include "./alter_op_layout.h"

namespace tvm {
namespace relay {

namespace alter_op_layout {

// Make a transform CallNode
Expr TransformLayout(Expr raw, Layout src_layout, Layout dst_layout) {
  if (src_layout.Equals(dst_layout)) { return raw; }
  CHECK(src_layout.defined() && dst_layout.defined())
    << "Cannot insert layout transform because there are undefined layouts. src="
    << src_layout << ". dst=" << dst_layout;
  CHECK(src_layout.Convertible(dst_layout))
    << "Cannot insert layout transform because there are inconvertible layouts: "
    << src_layout << " v.s. " << dst_layout;
  static auto &transform_op = Op::Get("layout_transform");
  NodePtr<LayoutTransformAttrs> attrs = make_node<LayoutTransformAttrs>();
  attrs->src_layout = src_layout.name();
  attrs->dst_layout = dst_layout.name();
  Call transform = CallNode::make(transform_op, {raw}, Attrs{attrs});
  return transform;
}

// Memorize layout transform so we can reuse internal transformed nodes
class TransformMemorizerNode : public Node {
 public:
  // map from (Expr, src_layout, dst_layout) to transformed Expr
  using TransformKey = std::tuple<const Node*, std::string, std::string>;
  struct key_hash : public std::unary_function<TransformKey , std::size_t> {
    std::size_t operator()(const TransformKey& k) const {
      return dmlc::HashCombine<std::string>(dmlc::HashCombine<std::string>(
              std::hash<const Node*>()(std::get<0>(k)), std::get<1>(k)), (std::get<2>(k)));
    }
  };

  std::unordered_map<TransformKey, Expr, key_hash> memo;
  static constexpr const char *_type_key = "relay.alter_op_layout.TransformMemorizerNode";
  TVM_DECLARE_NODE_TYPE_INFO(TransformMemorizerNode, Node);
};

class TransformMemorizer : public NodeRef {
 public:
  TransformMemorizer() {}
  explicit TransformMemorizer(NodePtr<Node> n) : NodeRef(n) {}

  TransformMemorizerNode* operator->() {
    return static_cast<TransformMemorizerNode*>(node_.get());
  }

  // Transform layout with memorizer
  Expr Transform(Expr raw, const Layout& src_layout, const Layout& dst_layout) {
    if (src_layout.Equals(dst_layout)) { return raw; }

    std::tuple<const Node*, std::string, std::string> key =
        std::make_tuple<>(raw.get(), src_layout.name(), dst_layout.name());
    auto& memo = operator->()->memo;

    auto iter = memo.find(key);
    if (iter != memo.end()) {
      return iter->second;
    } else {
      Expr transform = TransformLayout(raw, src_layout, dst_layout);
      memo[key] = transform;
      return transform;
    }
  }

  using ContainerType = TransformMemorizerNode;
};


// TempExprNode during layout transform
// Instance of this expr will be Realized to normal expr ultimately
class LayoutAlternatedExprNode : public TempExprNode {
 public:
  Expr value;
  Layout old_layout;
  Layout new_layout;
  Type old_type;
  Type new_type;
  TransformMemorizer memorizer;

  Expr Realize() const final {
    // NOTE: use a copy to discard the "const" qualifier
    TransformMemorizer tmp_memorizer = memorizer;
    // fallback to old layout
    return tmp_memorizer.Transform(value, new_layout, old_layout);
  }

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("value", &value);
    v->Visit("old_layout", &old_layout);
    v->Visit("new_layout", &new_layout);
  }

  static constexpr const char *_type_key = "relay.alter_op_layout.LayoutAlternatedExprNode";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutAlternatedExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(LayoutAlternatedExpr, LayoutAlternatedExprNode, TempExpr);

// Call registered FInferCorrectLayout of an op.
// Parameters are the same as the parameters for FInferCorrectLayout
// Returns inferred_input_layout, inferred_output_layout, success
std::tuple<Array<Layout>, Array<Layout>, bool> CallInfer(
    const Call& call,
    const size_t num_output,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr> > &old_in_shapes) {
  static auto finfer_layout = Op::GetAttr<FInferCorrectLayout>("FInferCorrectLayout");

  Op op = Downcast<Op>(call->op);
  if (finfer_layout.count(op)) {
    Array<Array<Layout> > inferred_layouts;
    inferred_layouts = finfer_layout[op](call->attrs, new_in_layouts,
                                         old_in_layouts, old_in_shapes);
    CHECK_EQ(inferred_layouts.size(), 2)
      << "FInferCorrectLayout should return an array with size of 2";
    /*
    for (auto x : inferred_layouts) {
      for (auto y : x) {
        if (!y.defined()) {  // inference fails
          return std::make_tuple<>(old_in_layouts,
                                   Array<Layout>(num_output, Layout::Undef()), false);
        }
      }
    }
    */
    return std::make_tuple<>(inferred_layouts[0], inferred_layouts[1], true);
  } else {
    return std::make_tuple<>(old_in_layouts, Array<Layout>(num_output, Layout::Undef()), false);
  }
}

class ReplaceArgsMutator : private ExprMutator {
 public:
  ReplaceArgsMutator(Array<Expr> real_args, Array<Expr> fake_args)
    : real_args_(std::move(real_args)), fake_args_(std::move(fake_args)) {
    CHECK_EQ(real_args.size(), fake_args.size());
  }

  Expr Visit(const Expr& expr) { return ExprMutator::Mutate(expr); }

 private:
  Expr VisitExpr_(const VarNode* op) final {
    for (size_t i = 0; i < fake_args_.size(); ++i) {
      if (op->name_hint() == fake_args_[i].as<VarNode>()->name_hint()) {
        return real_args_[i];
      }
    }
    CHECK(false);
    return GetRef<Expr>(op);
  }

  Array<Expr> real_args_;
  Array<Expr> fake_args_;
};

class FixLayoutMutator : private ExprMutator {
 public:
  FixLayoutMutator(TransformMemorizer& memorizer,
                   Array<Expr> args,
                   Array<Layout> new_in_layouts,
                   Array<Layout> old_in_layouts)
  : memorizer_(memorizer),
    args_(std::move(args)),
    new_in_layouts_(std::move(new_in_layouts)),
    old_in_layouts_(std::move(old_in_layouts)) {}

  Expr Visit(const Expr& expr) { return ExprMutator::Mutate(expr); }

 private:
  // Post order traversal.
  Expr VisitExpr_(const CallNode* call) final {
    // TODO: call is tuple
    LOG(INFO) << "(1) mutate args and collect the input layouts & shapes";
    Array<Expr> new_args;
    Array<Layout> new_in_layouts;
    Array<Layout> old_in_layouts;
    Array<Array<IndexExpr> > old_in_shapes;
    for (const auto& arg : call->args) {
      Expr mutated_arg = ExprMutator::Mutate(arg);
      // TODO: arg is tuple
      const LayoutAlternatedExprNode* inp = mutated_arg.as<LayoutAlternatedExprNode>();
      CHECK(inp);

      new_args.push_back(inp->value);
      new_in_layouts.push_back(inp->new_layout);
      old_in_layouts.push_back(inp->old_layout);
      if (auto* arg_type = inp->value->checked_type().as<TensorTypeNode>()) {
        old_in_shapes.push_back(arg_type->shape);
      } else {
        // TODO: TupleNode
        old_in_shapes.push_back(Array<IndexExpr>(nullptr));
      }
    }

    LOG(INFO) << "(2) run layout inference" << " for " << GetRef<Expr>(call);
    Array<Layout> required_in_layouts, out_layouts;
    bool success = false;
    std::tie(required_in_layouts, out_layouts, success) =
      CallInfer(GetRef<Call>(call), 1, new_in_layouts, old_in_layouts, old_in_shapes);

    // TODO
    CHECK_EQ(out_layouts.size(), 1);

    CHECK_EQ(new_in_layouts.size(), required_in_layouts.size())
      << "The number of input nodes should keep the same during alter_op_layout";

    LOG(INFO) << "(3) generate layout transform";
    Array<Expr> transformed_args;
    for (size_t i = 0; i < new_args.size(); ++i) {
      LOG(INFO) << "arg[" << i << "]: new_in_layout=" << new_in_layouts[i] << " required_in_layout=" << required_in_layouts[i];
      transformed_args.push_back(
        memorizer_.Transform(new_args[i], new_in_layouts[i], required_in_layouts[i]));
    }
    Expr new_call = CallNode::make(call->op, transformed_args, call->attrs);

    LOG(INFO) << "(4) infer shape for newly created node." << new_call;
    new_call = InferType(new_call, Module());

    LOG(INFO) << "(5) pass (out_layout, shape) to next layer";
    auto ret = make_node<LayoutAlternatedExprNode>();
    ret->value = new_call;
    ret->old_layout = out_layouts[0];
    ret->new_layout = out_layouts[0];
    ret->memorizer = memorizer_;

    return Expr(ret);
  }

  Expr VisitExpr_(const VarNode* op) final {
    for (size_t i = 0; i < args_.size(); ++i) {
      if (op->name_hint() == args_[i].as<VarNode>()->name_hint()) {
        auto ret = make_node<LayoutAlternatedExprNode>();
        ret->value = InferType(GetRef<Expr>(op), Module());
        ret->old_layout = old_in_layouts_[i];
        ret->new_layout = new_in_layouts_[i];
        ret->memorizer = memorizer_;
        return Expr(ret);
      }
    }
    CHECK(false);
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    auto ret = make_node<LayoutAlternatedExprNode>();
    ret->value = InferType(GetRef<Expr>(op), Module());
    ret->old_layout = Layout::Undef();
    ret->new_layout = Layout::Undef();
    ret->memorizer = memorizer_;
    return Expr(ret);
  }

  Expr VisitExpr_(const GlobalVarNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const OpNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const TupleNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const FunctionNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const LetNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const IfNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const TupleGetItemNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const RefCreateNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const RefReadNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const RefWriteNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const ConstructorNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }
  Expr VisitExpr_(const MatchNode* op) final { LOG(FATAL) << "not supported " << GetRef<Expr>(op); }

  Array<Expr> args_;
  Array<Layout> new_in_layouts_;
  Array<Layout> old_in_layouts_;
  TransformMemorizer memorizer_;
};

// Call registered FTVMAlterOpLayout of an op
// Returns the altered expression
std::tuple<Call, Array<Layout>, bool> CallAlter(const Call& ref_call,
               const std::vector<LayoutAlternatedExpr>& new_args,
               TransformMemorizer& memorizer,
               Array<Layout> new_in_layouts,
               Array<Layout> old_in_layouts) {
  static auto falter_layout = Op::GetAttr<FTVMAlterOpLayout>("FTVMAlterOpLayout");
  Op op = Downcast<Op>(ref_call->op);

  Expr new_e;
  bool modified = false;
  Array<Expr> real_args;
  Array<Expr> fake_args;

  tvm::Array<tvm::Tensor> tinfos;
  LOG(INFO) << "CallAlter1" << new_args.size();
  for (size_t i = 0; i < new_args.size(); ++i) {
    // TODO: arg is tuple
    auto arg = new_args[i];
    LOG(INFO) << "arg[" << i << "] = " << arg;
    auto old_type = arg->old_type.as<TensorTypeNode>();
    auto new_type = arg->new_type.as<TensorTypeNode>();
    CHECK(old_type && new_type);
    tinfos.push_back(tvm::placeholder(old_type->shape, old_type->dtype));
    std::stringstream var_name;
    // TODO: flatten tuple node?
    var_name << op->name << "." << ref_call.hash() << "#" << i;
    fake_args.push_back(VarNode::make(var_name.str(),
                                      TensorTypeNode::make(new_type->shape,
                                                           new_type->dtype)));
    real_args.push_back(new_args[i]->value);
  }
  LOG(INFO) << "CallAlter2";

  if (falter_layout.count(op)) {
    CHECK_EQ(op->arguments.size(), ref_call->args.size());
    Expr altered_value = falter_layout[op](ref_call->attrs, fake_args, tinfos);
    if (altered_value.defined()) {
      new_e = altered_value;
      modified = true;
    }
  }

  if (!modified) {
    new_e = CallNode::make(ref_call->op, fake_args,
                           ref_call->attrs);
  }
  LOG(INFO) << "CallAlter3";

  // fix layout
  Array<Layout> out_layout;
  FixLayoutMutator layout_fixer(memorizer, fake_args, new_in_layouts, old_in_layouts);
  new_e = layout_fixer.Visit(new_e);
  // TODO: what if it is a tuple node?
  out_layout.push_back(new_e.as<LayoutAlternatedExprNode>()->new_layout);
  new_e = new_e.as<LayoutAlternatedExprNode>()->value;
  LOG(INFO) << "CallAlter4";

  CHECK(new_e->checked_type_.defined());
  auto ttype = new_e->checked_type_;
  LOG(INFO) << "CallAlter5";

  ReplaceArgsMutator args_replacer(real_args, fake_args);
  new_e = args_replacer.Visit(new_e);
  // since arg_replacer does not change the root node's type,
  // we can simply restore the backup type.
  new_e->checked_type_ = ttype;
  LOG(INFO) << "CallAlter6";

  const CallNode *new_call = new_e.as<CallNode>();
  CHECK(new_call) << "Can only replace the original operator with another call node";
  return std::make_tuple<>(GetRef<Call>(new_call), out_layout, modified);
}

LayoutAlternatedExpr GetLayoutAlternatedExpr(Expr arg, const TransformMemorizer& memorizer) {
  if (const LayoutAlternatedExprNode *inp = arg.as<LayoutAlternatedExprNode>()) {
    return GetRef<LayoutAlternatedExpr>(inp);
  }
  auto inode = make_node<LayoutAlternatedExprNode>();
  inode->value = arg;
  inode->old_type = arg->checked_type();
  inode->new_type = arg->checked_type();
  inode->memorizer = memorizer;
  return LayoutAlternatedExpr(inode);
}

Expr AlterOpLayoutRewrite(const Call &ref_call,
                          const Array<Expr> &new_args,
                          const NodeRef& ctx) {
  LOG(INFO) << "hello1 " << ref_call;
  std::vector<LayoutAlternatedExpr> inputs;
  Array<Array<IndexExpr> > input_shapes;

  const size_t num_output = ref_call->checked_type()->is_type<TupleTypeNode>() ?
                            ref_call->type_as<TupleTypeNode>()->fields.size() : 1;

  // NOTE: discard the "const" qualifier
  TransformMemorizer memorizer = Downcast<TransformMemorizer>(ctx);

  // fill incomplete state and flatten tuple
  // We always expect LayoutAlternatedExpr.
  // This is used to convert the normal Expr to LayoutAlternatedExpr.
  for (auto new_arg : new_args) {
    if (new_arg->is_type<TupleNode>()) {
      LOG(INFO) << "IS TUPLE NODE";
      Tuple tuple_new_arg = Downcast<Tuple>(new_arg);
      for (auto x : tuple_new_arg->fields) {
        CHECK(!x->is_type<TupleNode>()) << "AlterOpLayout pass does not support nested tuple";
        inputs.push_back(GetLayoutAlternatedExpr(x, memorizer));
      }
    } else {
      inputs.push_back(GetLayoutAlternatedExpr(new_arg, memorizer));
    }
  }
  LOG(INFO) << "hello2 " << ref_call;

  // old_in, new_in = state[inputs]
  Array<Layout> old_in, old_out, new_in, new_out;
  for (auto inp : inputs) {
    old_in.push_back(inp->old_layout);
    new_in.push_back(inp->new_layout);
  }

  for (auto arg : ref_call->args) {
    if (arg->is_type<TupleNode>()) {  // flatten tuple
      Tuple tuple_arg = Downcast<Tuple>(arg);
      for (auto x : tuple_arg->fields) {
        input_shapes.push_back(x->type_as<TensorTypeNode>()->shape);
      }
    } else {
      input_shapes.push_back(arg->type_as<TensorTypeNode>()->shape);
    }
  }
  LOG(INFO) << "hello3 " << ref_call;

  // old_in, old_out = op.infer(old_in)
  bool success = false;
  Array<Layout> inferred_old_layout;
  std::tie(inferred_old_layout, old_out, success) = CallInfer(ref_call, num_output,
                                                              Array<Layout>(nullptr),
                                                              old_in, input_shapes);
  LOG(INFO) << "hello4 " << ref_call;
  if (success) {
    CHECK_EQ(old_in.size(), inferred_old_layout.size());
    for (size_t i = 0; i < old_in.size(); ++i) {
      if (old_in[i].defined()) {
        CHECK(old_in[i].Equals(inferred_old_layout[i]))
          << ref_call->op << " inferred a mismatched layout with its input[" << i << "]."
          << " Required: " << old_in[i] << ", inferred: " << inferred_old_layout[i] << ".";

      } else {
        old_in.Set(i, inferred_old_layout[i]);
      }
    }
  }
  CHECK_EQ(old_in.size(), new_in.size());
  LOG(INFO) << "hello5 " << ref_call;

  // if new_in == 'undef':  new_in = old_in
  for (size_t i = 0; i < new_in.size(); ++i) {
    if (!new_in[i].defined()) {
      new_in.Set(i, old_in[i]);
    }
  }

  // new_op = alter(op)
  Call new_call;
  bool altered;
  std::tie(new_call, new_out, altered) = CallAlter(ref_call, inputs, memorizer, new_in, old_in);
  CHECK_EQ(new_out.size(), old_out.size());
  LOG(INFO) << "hello6 " << ref_call;

  // new_in2, new_out = op.infer(new_in)
  Expr ret = new_call;
  if (!new_call->op->is_type<OpNode>()) {
    return Expr(nullptr);
  }

  LOG(INFO) << "ret = " << ret;
  // state[node] = (old_out, new_out)
  // (handle tuple output)
  if (ref_call->checked_type()->is_type<TupleTypeNode>()) {
    Array<Expr> fields;
    for (size_t i = 0; i < new_out.size(); ++i) {
      auto rnode = make_node<LayoutAlternatedExprNode>();
      rnode->value = TupleGetItemNode::make(ret, i);
      rnode->old_layout = old_out[i];
      rnode->new_layout = new_out[i];
      rnode->old_type = ref_call->checked_type();
      rnode->new_type = ret->checked_type();
      rnode->memorizer = memorizer;
      fields.push_back(Expr(rnode));
    }
    return TupleNode::make(fields);
  } else {
    auto rnode = make_node<LayoutAlternatedExprNode>();
    CHECK_EQ(new_out.size(), 1);
    rnode->value = ret;
    rnode->old_layout = old_out[0];
    rnode->new_layout = new_out[0];
    rnode->old_type = ref_call->checked_type();
    rnode->new_type = ret->checked_type();
    rnode->memorizer = memorizer;
    return Expr(rnode);
  }
}

// Limiations:
// 1. the altered op should have the same number of arguments as the previous one
// 2. do not support nested tuple arguments
TVM_REGISTER_API("relay._ir_pass.AlterOpLayout")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  TransformMemorizer transformMemorizer(make_node<TransformMemorizerNode>());
  auto fcontext = [&](const Call& call) -> NodeRef{
    return transformMemorizer;
  };

  *ret = ForwardRewrite(args[0], AlterOpLayoutRewrite, fcontext);
});

}  // namespace alter_op_layout

}  // namespace relay
}  // namespace tvm
