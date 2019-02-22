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

// Tuple<Expr> -> Tuple<T> or Expr -> T
template<typename T>
Expr ConvertKeepTuple(const Expr& e, std::function<T (const Expr&)> converter) {
  if (e->is_type<TupleNode>()) {
    Tuple t = Downcast<Tuple>(e);
    Array<Expr> new_fields;
    for (Expr arg : t->fields) {
      T new_arg = converter(arg);
      new_fields.push_back(new_arg);
    }
    return TupleNode::make(new_fields);
  } else {
    return converter(e);
  }
}

// Array<TupleNode<LayoutAlternatedExpr>> -> Array<T> or
// Array<LayoutAlternatedExpr>            -> Array<T>
template<typename T>
Array<T> FlattenArray(const Array<Expr>& array,
                      std::function<T (const LayoutAlternatedExpr&)> converter) {
  Array<T> results;
  for (Expr e : array) {
    if (e->is_type<TupleNode>()) {
      Tuple t = Downcast<Tuple>(e);
      for (Expr field : t->fields) {
        const auto *inp = field.as<LayoutAlternatedExprNode>();
        CHECK(inp);
        results.push_back(converter(GetRef<LayoutAlternatedExpr>(inp)));
      }
    } else {
      const auto *inp = e.as<LayoutAlternatedExprNode>();
      CHECK(inp);
      results.push_back(converter(GetRef<LayoutAlternatedExpr>(inp)));
    }
  }
  return results;
}

// Call registered FInferCorrectLayout of an op.
// Parameters are the same as the parameters for FInferCorrectLayout
// Returns inferred_input_layout, inferred_output_layout, success
std::tuple<Array<Layout>, Array<Layout>, bool> CallInfer(
    const Call& call,
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
    return std::make_tuple<>(inferred_layouts[0], inferred_layouts[1], true);
  } else {
    return std::make_tuple<>(old_in_layouts, Array<Layout>(nullptr), false);
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
                   const Expr& input,
                   Array<Expr> args,
                   const Array<Expr>& args_replacement,
                   Array<Layout> new_in_layouts,
                   Array<Layout> old_in_layouts,
                   Array<Layout> old_out_layouts)
  : memorizer_(memorizer),
    input_(input),
    args_(std::move(args)),
    args_replacement_(args_replacement),
    new_in_layouts_(std::move(new_in_layouts)),
    old_in_layouts_(std::move(old_in_layouts)),
    old_out_layouts_(std::move(old_out_layouts)) {}

  Expr Visit() { return ExprMutator::Mutate(input_); }

 private:
  // Post order traversal.
  Expr VisitExpr_(const CallNode* call) final {
    // TODO: call is tuple
    LOG(INFO) << "(1) mutate args and collect the input layouts & shapes";
    Array<Expr> new_args;
    Array<Expr> new_alternated_expr_args;

    for (const auto& arg : call->args) {
      Expr mutated_arg = ExprMutator::Mutate(arg);

      LOG(INFO) << "haha1";
      new_args.push_back(ConvertKeepTuple<Expr>(mutated_arg, [](const Expr& single_arg) {
        auto inp = single_arg.as<LayoutAlternatedExprNode>();
        CHECK(inp);
        return inp->value;
      }));
      new_alternated_expr_args.push_back(mutated_arg);
    }

    LOG(INFO) << "haha2";
    Array<Layout> old_in_layouts = FlattenArray<Layout>(
      new_alternated_expr_args, [](const LayoutAlternatedExpr& e) -> Layout {
      return e->old_layout;
    });

    LOG(INFO) << "haha3";
    Array<Layout> new_in_layouts = FlattenArray<Layout>(
      new_alternated_expr_args, [](const LayoutAlternatedExpr& e) -> Layout {
      return e->new_layout;
    });

    Array<Array<IndexExpr> > old_in_shapes = FlattenArray<Array<IndexExpr> >(
      new_alternated_expr_args, [](const LayoutAlternatedExpr& e) -> Array<IndexExpr> {
      return e->old_type.defined() ?
             e->old_type.as<TensorTypeNode>()->shape :
             Array<IndexExpr>(nullptr);
    });

    LOG(INFO) << "(2) run layout inference" << " for " << GetRef<Expr>(call);
    Array<Layout> required_in_layouts, out_layouts;
    bool success_infer_layout = false;
    std::tie(required_in_layouts, out_layouts, success_infer_layout) =
      CallInfer(GetRef<Call>(call), new_in_layouts, old_in_layouts, old_in_shapes);

    CHECK_EQ(new_in_layouts.size(), required_in_layouts.size())
      << "The number of input nodes should keep the same during alter_op_layout";

    LOG(INFO) << "(3) generate layout transform";
    Array<Expr> transformed_args;
    size_t idx_flatten = 0;
    for (size_t i = 0; i < new_args.size(); ++i) {
//      LOG(INFO) << "arg[" << i << "]: new_in_layout=" << new_in_layouts[i] << " required_in_layout=" << required_in_layouts[i];
      Expr transformed_arg = ConvertKeepTuple<Expr>(new_args[i], [&](const Expr& arg) {
        Expr res = memorizer_.Transform(arg, new_in_layouts[idx_flatten],
                                        required_in_layouts[idx_flatten]);
        idx_flatten++;
        return res;
      });
      transformed_args.push_back(transformed_arg);
    }
    Expr new_call = CallNode::make(call->op, transformed_args, call->attrs);

    LOG(INFO) << "(4) infer shape for newly created node." << new_call;
    new_call = InferType(new_call, Module());
    const size_t num_output = new_call->checked_type()->is_type<TupleTypeNode>() ?
                              new_call->type_as<TupleTypeNode>()->fields.size() : 1;

    LOG(INFO) << "(5) pass (out_layout, shape) to next layer";
    if (!success_infer_layout) {
      out_layouts = Array<Layout>(num_output, Layout::Undef());
    }

    // root node, do fake_args replacement
    if (call == input_.get()) {
      LOG(INFO) << "root node, do fake_args replacement";
      // since arg_replacer does not change the root node's type,
      // we can simply backup the type and restore later.
      auto checked_type = new_call->checked_type_;
      ReplaceArgsMutator args_replacer(args_replacement_, args_);
      new_call = args_replacer.Visit(new_call);
      new_call->checked_type_ = checked_type;
    }

    if (new_call->checked_type()->is_type<TupleTypeNode>()) {
      Array<Expr> fields;
      for (size_t i = 0; i < num_output; ++i) {
        auto rnode = make_node<LayoutAlternatedExprNode>();
        rnode->value = TupleGetItemNode::make(new_call, i);
        rnode->new_layout = out_layouts[i];
        rnode->new_type = new_call->checked_type();
        if (call == input_.get()) {
          LOG(INFO) << "call == input_.get())";
          CHECK_EQ(num_output, old_out_layouts_.size());
          rnode->old_layout = old_out_layouts_[i];
          rnode->old_type = input_->checked_type();
        } else {
          rnode->old_layout = out_layouts[i];
          rnode->old_type = new_call->checked_type();
        }
        rnode->memorizer = memorizer_;
        fields.push_back(Expr(rnode));
      }
      return TupleNode::make(fields);
    } else {
      auto rnode = make_node<LayoutAlternatedExprNode>();
      CHECK_EQ(out_layouts.size(), 1);
      rnode->value = new_call;
      rnode->new_layout = out_layouts[0];
      rnode->new_type = new_call->checked_type();
      if (call == input_.get()) {
        LOG(INFO) << "call == input_.get())";
        CHECK_EQ(1, old_out_layouts_.size());
        rnode->old_layout = old_out_layouts_[0];
        rnode->old_type = input_->checked_type();
      } else {
        rnode->old_layout = out_layouts[0];
        rnode->old_type = new_call->checked_type();
      }
      rnode->memorizer = memorizer_;
      return Expr(rnode);
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    for (size_t i = 0; i < args_.size(); ++i) {
      if (op->name_hint() == args_[i].as<VarNode>()->name_hint()) {
        auto ret = make_node<LayoutAlternatedExprNode>();
        ret->value = InferType(GetRef<Expr>(op), Module());
        ret->old_layout = old_in_layouts_[i];
        ret->new_layout = new_in_layouts_[i];
        ret->old_type = args_replacement_[i]->checked_type();
        ret->new_type = args_replacement_[i]->checked_type();
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

  Expr input_;
  Array<Expr> args_;
  Array<Expr> args_replacement_;
  Array<Layout> new_in_layouts_;
  Array<Layout> old_in_layouts_;
  Array<Layout> old_out_layouts_;
  TransformMemorizer memorizer_;
};

// Call registered FTVMAlterOpLayout of an op
// Returns the altered expression
Expr CallAlter(const Call& ref_call,
               const Array<Expr>& new_args,
               TransformMemorizer& memorizer,
               Array<Layout> new_in_layouts,
               Array<Layout> old_in_layouts,
               Array<Layout> old_out_layouts) {
  static auto falter_layout = Op::GetAttr<FTVMAlterOpLayout>("FTVMAlterOpLayout");
  Op op = Downcast<Op>(ref_call->op);

  Expr new_e;
  bool modified = false;
  Array<Expr> real_args;
  Array<Expr> fake_args;

  for (size_t i = 0; i < new_args.size(); ++i) {
    std::stringstream var_name;
    var_name << op->name << "." << ref_call.hash() << "#" << i;
    Expr fake_arg = ConvertKeepTuple<Var>(new_args[i], [&var_name](const Expr& arg) {
      const auto* inp = arg.as<LayoutAlternatedExprNode>();
      CHECK(inp);
      auto new_type = inp->new_type.as<TensorTypeNode>();
      CHECK(new_type);
      return VarNode::make(var_name.str(), TensorTypeNode::make(new_type->shape,
                                                                new_type->dtype));
    });
    fake_args.push_back(fake_arg);

    Expr real_arg = ConvertKeepTuple<Expr>(new_args[i], [](const Expr& arg) {
      const auto* inp = arg.as<LayoutAlternatedExprNode>();
      CHECK(inp);
      return inp->value;
    });
    real_args.push_back(real_arg);
  }

  tvm::Array<tvm::Tensor> tinfos = FlattenArray<tvm::Tensor>(new_args, [](const Expr& arg) {
    const auto* inp = arg.as<LayoutAlternatedExprNode>();
    CHECK(inp);
    auto old_type = inp->old_type.as<TensorTypeNode>();
    CHECK(old_type);
    return tvm::placeholder(old_type->shape, old_type->dtype);
  });

  if (falter_layout.count(op)) {
    CHECK_EQ(op->arguments.size(), ref_call->args.size());
    Expr altered_value = falter_layout[op](ref_call->attrs, fake_args, tinfos);
    if (altered_value.defined()) {
      new_e = altered_value;
      CHECK(new_e.as<CallNode>())
        << "Can only replace the original operator with another call node";
      modified = true;
    }
  }

  if (!modified) {
    new_e = CallNode::make(ref_call->op, fake_args,
                           ref_call->attrs);
  }
  // this is for setting old_type
  new_e->checked_type_ = ref_call->checked_type();

  // fix layout
  // TODO: fake_args can contain tuple
  LOG(INFO) << "old_out_layouts = " << old_out_layouts;
  FixLayoutMutator layout_fixer(memorizer, new_e, fake_args, real_args,
                                new_in_layouts, old_in_layouts, old_out_layouts);

  return layout_fixer.Visit();
}

LayoutAlternatedExpr GetLayoutAlternatedExpr(Expr arg, const TransformMemorizer& memorizer) {
  if (const auto *inp = arg.as<LayoutAlternatedExprNode>()) {
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

  const size_t num_output = ref_call->checked_type()->is_type<TupleTypeNode>() ?
                            ref_call->type_as<TupleTypeNode>()->fields.size() : 1;

  // NOTE: discard the "const" qualifier
  TransformMemorizer memorizer = Downcast<TransformMemorizer>(ctx);

  // Fill incomplete state and flatten tuple
  // We always expect LayoutAlternatedExpr.
  // This is used to convert the normal Expr to LayoutAlternatedExpr.
  // old_in, new_in = state[inputs]
  // we keep tuple struct for arguments, but flatten for shapes and layouts.
  // because infer_layouts treats tuple input layouts as flatten ones.
  Array<Expr> inputs; // Array<LayoutAlternatedExpr> effectively
  for (auto new_arg : new_args) {
    Expr input = ConvertKeepTuple<LayoutAlternatedExpr>(
      new_arg, [&memorizer](const Expr& field) {
      return GetLayoutAlternatedExpr(field, memorizer);
    });
    inputs.push_back(input);
  }

  Array<Layout> old_in = FlattenArray<Layout>(
    inputs, [](const LayoutAlternatedExpr& e) -> Layout {
    return e->old_layout;
  });

  Array<Layout> new_in = FlattenArray<Layout>(
    inputs, [](const LayoutAlternatedExpr& e) -> Layout {
    return e->new_layout;
  });

  Array<Array<IndexExpr> > input_shapes = FlattenArray<Array<IndexExpr> >(
    inputs, [](const LayoutAlternatedExpr& e) -> Array<IndexExpr> {
    return e->old_type.as<TensorTypeNode>()->shape;
  });

  Array<Layout> old_out, new_out;

  // old_in, old_out = op.infer(old_in)
  bool success = false;
  Array<Layout> inferred_old_layout;
  std::tie(inferred_old_layout, old_out, success) = CallInfer(ref_call,
                                                              Array<Layout>(nullptr),
                                                              old_in, input_shapes);
  LOG(INFO) << "old_out = " << old_out;
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
  } else {
    old_out = Array<Layout>(num_output, Layout::Undef());
  }
  CHECK_EQ(old_in.size(), new_in.size());

  // if new_in == 'undef':  new_in = old_in
  for (size_t i = 0; i < new_in.size(); ++i) {
    if (!new_in[i].defined()) {
      new_in.Set(i, old_in[i]);
    }
  }

  return CallAlter(ref_call, inputs, memorizer, new_in, old_in, old_out);
  /*
  if (!new_call->op->is_type<OpNode>()) {
    return Expr(nullptr);
  }
   */
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
