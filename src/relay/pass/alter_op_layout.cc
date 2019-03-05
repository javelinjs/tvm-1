/*!
 * Copyright (c) 2018 by Contributors
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/tvm.h>
#include <tuple>
#include <utility>
#include <vector>
#include <string>

#include "./alter_op_layout.h"

namespace tvm {
namespace relay {

// Make a transform CallNode
Expr TransformLayout(Expr raw, const Layout &src_layout, const Layout& dst_layout) {
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

Expr TransformMemorizer::Transform(Expr raw, const Layout& src_layout, const Layout& dst_layout) {
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

// TempExprNode during layout transform
// Instance of this expr will be Realized to normal expr ultimately
class LayoutAlternatedExprNode : public TempExprNode {
 public:
  Expr value;
  Array<Layout> old_layouts;
  Array<Layout> new_layouts;
  Array<Type> old_types;
  Array<Type> new_types;
  TransformMemorizer memorizer;

  Expr Realize() const final {
    // NOTE: use a copy to discard the "const" qualifier
    TransformMemorizer tmp_memorizer = memorizer;
    // fallback to old layout
    if (value->is_type<TupleNode>()) {
      Tuple t = Downcast<Tuple>(value);
      Array<Expr> fields;
      for (size_t i = 0; i < t->fields.size(); ++i) {
        const auto *inp = t->fields[i].as<LayoutAlternatedExprNode>();
        if (inp) {
          fields.push_back(inp->Realize());
        } else {
          fields.push_back(tmp_memorizer.Transform(t->fields[i], new_layouts[i], old_layouts[i]));
        }
      }
      return TupleNode::make(fields);
    } else if (value->checked_type()->is_type<TupleTypeNode>()) {
      // do not do any transform for tuple-type call node.
      // FIXME: TupleGetItem and do transform(new_layouts[i], old_layouts[i])
      return value;
    } else {
      CHECK_EQ(old_layouts.size(), 1);
      CHECK_EQ(new_layouts.size(), 1);
      return tmp_memorizer.Transform(value, new_layouts[0], old_layouts[0]);
    }
  }

  inline void Check() const {
    size_t num_outputs = old_layouts.size();
    CHECK_EQ(new_layouts.size(), num_outputs);
    CHECK_EQ(old_types.size(), num_outputs);
    CHECK_EQ(new_types.size(), num_outputs);
  }

  Array<Array<IndexExpr> > GetOldShapes() const {
    Array<Array<IndexExpr> > shapes;
    for (const Type& type : old_types) {
      const auto* old_type = type.as<TensorTypeNode>();
      CHECK(old_type) << old_types;
      shapes.push_back(old_type->shape);
    }
    return shapes;
  }

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("value", &value);
  }

  static constexpr const char *_type_key = "relay.alter_op_layout.LayoutAlternatedExprNode";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutAlternatedExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(LayoutAlternatedExpr, LayoutAlternatedExprNode, TempExpr);

// Tuple<Expr> -> Tuple<T> or Expr -> T
template<typename T>
Expr MapTuple(const Expr& e, const std::function<T(const Expr&)>& mapper) {
  LOG(INFO) << "MapTuple";
  if (e->is_type<TupleNode>()) {
    Tuple t = Downcast<Tuple>(e);
    Array<Expr> new_fields;
    for (size_t i = 0; i < t->fields.size(); ++i) {
      T new_arg = MapTuple(t->fields[i], mapper);
      new_fields.push_back(new_arg);
    }
    return TupleNode::make(new_fields);
  } else {
    return mapper(e);
  }
}

// Array<TupleNode<Expr>> -> Array<T> or
// Array<Expr>            -> Array<T>
template<typename T>
Array<T> FlatMapTuple(const Array<Expr>& array,
                      std::function<T(const Expr&)> mapper) {
  LOG(INFO) << "MapFlatTuple";
  Array<T> results;
  for (Expr e : array) {
    if (e->is_type<TupleNode>()) {
      Tuple t = Downcast<Tuple>(e);
      for (const T& ret : FlatMapTuple(t->fields, mapper)) {
        results.push_back(ret);
      }
    } else {
      results.push_back(mapper(e));
    }
  }
  return results;
}

// Call registered FInferCorrectLayout of an op.
// Parameters are the same as the parameters for FInferCorrectLayout
// Returns provided_input_layouts, request_input_layouts, output_layouts
std::tuple<Array<Layout>, Array<Layout>, Array<Layout> > CallInfer(
    const Call& call, const Array<Expr>& args, bool infer_old) {
  Array<Layout> provided_old_layouts;
  Array<Layout> provided_new_layouts;
  // TODO: use type directly ?
  Array<Array<IndexExpr> > provided_old_shapes;
  for (const Expr& arg : args) {
    const auto* arg_info = arg.as<LayoutAlternatedExprNode>();
    CHECK(arg_info);
    arg_info->Check();
    for (const Layout& layout : arg_info->old_layouts) {
      provided_old_layouts.push_back(layout);
      if (infer_old) provided_new_layouts.push_back(layout);
    }

    for (auto shape : arg_info->GetOldShapes()) {
      provided_old_shapes.push_back(shape);
    }

    if (!infer_old) {
      for (const Layout& layout : arg_info->new_layouts) {
        provided_new_layouts.push_back(layout);
        LOG(INFO) << "!infer_old layout = " << layout;
      }
    }
  }

  static auto finfer_layout = Op::GetAttr<FInferCorrectLayout>("FInferCorrectLayout");
  Op op = Downcast<Op>(call->op);
  if (finfer_layout.count(op)) {
    Array<Array<Layout> > inferred_layouts;
    inferred_layouts = finfer_layout[op](call->attrs, provided_new_layouts,
                                         provided_old_layouts, provided_old_shapes);
    CHECK_EQ(inferred_layouts.size(), 2)
      << "FInferCorrectLayout should return an array with size of 2";
    return std::make_tuple<>(provided_new_layouts,
                             inferred_layouts[0],
                             inferred_layouts[1]);
  } else {
    return std::make_tuple<>(provided_new_layouts,
                             provided_old_layouts,
                             Array<Layout>(nullptr));
  }
}

class ReplaceArgsMutator : private ExprMutator {
 public:
  const Array<Expr> real_inputs_;
  const Array<Expr> fake_inputs_;

  Array<Expr> real_inputs_flatten_;
  Array<Expr> fake_inputs_flatten_;

  ReplaceArgsMutator() = delete;

  Expr Run(const Expr& expr) {
    auto type = expr->checked_type_;
    auto new_e = ExprMutator::Mutate(expr);
    new_e->checked_type_ = type;
    return new_e;
  }

  static ReplaceArgsMutator BuildInst(const Expr& expr,
                                      const Array<Expr>& inputs) {
    Array<Expr> fake_inputs;
    size_t idx = 0;
    for (const Expr& input : inputs) {
      Expr fake_input = MapTuple<Expr>(input, [&](const Expr& non_tuple) {
        std::stringstream var_name;
        var_name << expr.hash() << ".input#" << idx++;
        CHECK(non_tuple->checked_type_.defined());
        Expr fake_var = VarNode::make(var_name.str(), non_tuple->checked_type_);
        return fake_var;
      });
      fake_input = InferType(fake_input, Module());
      fake_inputs.push_back(fake_input);

      CHECK(input->checked_type_.defined());
    }

    return ReplaceArgsMutator(inputs, fake_inputs);
  }

 private:
  ReplaceArgsMutator(const Array<Expr>& real_inputs, const Array<Expr>& fake_inputs)
      : real_inputs_(real_inputs), fake_inputs_(fake_inputs) {
    real_inputs_flatten_ = FlatMapTuple<Expr>(real_inputs, [](const Expr& e) {
      return e;
    });
    fake_inputs_flatten_ = FlatMapTuple<Expr>(fake_inputs, [](const Expr& e) {
      return e;
    });
  }

  Expr VisitExpr_(const VarNode* op) final {
    for (size_t i = 0; i < fake_inputs_flatten_.size(); ++i) {
      if (op == fake_inputs_flatten_[i].get()) {
        return real_inputs_flatten_[i];
      }
    }
    CHECK(false) << GetRef<Expr>(op);
    return GetRef<Expr>(op);
  }
};

// Call registered FTVMAlterOpLayout of an op
// Returns the altered expression
Expr CallAlter(const Call& ref_call,
               const Array<Expr>& args,
               TransformMemorizer& memorizer) {
  static auto falter_layout = Op::GetAttr<FTVMAlterOpLayout>("FTVMAlterOpLayout");
  Op op = Downcast<Op>(ref_call->op);

  Expr new_e;
  if (falter_layout.count(op)) {
    Array<Layout> flatten_old_layouts;
    Array<Layout> flatten_new_layouts;
    Array<Type> flatten_old_types;
    Array<Type> flatten_new_types;

    Array<Expr> raw_args;
    tvm::Array<tvm::Tensor> tinfos;
    for (const Expr& arg : args) {
      // use old type
      const auto* arg_info = arg.as<LayoutAlternatedExprNode>();
      CHECK(arg_info);
      arg_info->Check();

      for (const auto& old_type : arg_info->old_types) {
        flatten_old_types.push_back(old_type);
        const auto* type = old_type.as<TensorTypeNode>();
        CHECK(type);
        tinfos.push_back(tvm::placeholder(type->shape, type->dtype));
      }
      for (const auto& new_type : arg_info->new_types) {
        flatten_new_types.push_back(new_type);
      }
      for (const auto& old_layout : arg_info->old_layouts) {
        flatten_old_layouts.push_back(old_layout);
      }
      for (const auto& new_layout : arg_info->new_layouts) {
        flatten_new_layouts.push_back(new_layout);
      }

      raw_args.push_back(arg_info->value);
    }

    ReplaceArgsMutator input_replacer = ReplaceArgsMutator::BuildInst(ref_call, raw_args);
    Expr altered_value = falter_layout[op](ref_call->attrs, input_replacer.fake_inputs_, tinfos);
    if (altered_value.defined()) {
      // get inputs' old/new layouts and types.
      size_t index = 0;
      Array<Expr> fake_input_infos = FlatMapTuple<Expr>(input_replacer.fake_inputs_,
                                                        [&](const Expr& non_tuple) {
        CHECK(non_tuple->checked_type_.defined());
        auto node = make_node<LayoutAlternatedExprNode>();
        node->value = non_tuple;
        node->new_layouts = Array<Layout>{flatten_new_layouts[index]};
        node->old_layouts = Array<Layout>{flatten_old_layouts[index]};
        node->new_types = Array<Type>{flatten_new_types[index]};
        node->old_types = Array<Type>{flatten_old_types[index]};
        node->memorizer = memorizer;
        index++;
        return LayoutAlternatedExpr(node);
      });

      AlterOpLayoutMutator layout_fixer(fake_input_infos, false, memorizer);
      new_e = layout_fixer.Run(altered_value);
      const auto* altered_ptr = new_e.as<LayoutAlternatedExprNode>();
      CHECK(altered_ptr);
      CHECK(altered_ptr->value->checked_type_.defined());
      LOG(INFO) << "layout fixer = " << altered_ptr->value;
      // replace fake inputs
      auto node = make_node<LayoutAlternatedExprNode>();
      node->value = input_replacer.Run(altered_ptr->value);
      node->new_layouts = altered_ptr->new_layouts;
      node->old_layouts = altered_ptr->old_layouts;
      node->new_types = altered_ptr->new_types;
      node->old_types = altered_ptr->old_types;
      node->memorizer = memorizer;
      CHECK(node->value->checked_type_.defined());
      LOG(INFO) << "final = " << node->value;
      new_e = LayoutAlternatedExpr(node);
    }
  }
  return new_e;
}

AlterOpLayoutMutator::AlterOpLayoutMutator(const Array<Expr>& inputs,
                                           const bool enable_alter,
                                           TransformMemorizer& memorizer)
  : enable_alter_(enable_alter), memorizer_(memorizer) {
  if (inputs.defined()) {
    for (const Expr &input : inputs) {
      const auto *temp = input.as<LayoutAlternatedExprNode>();
      CHECK(temp);
      cache_[temp->value.get()] = input;
    }
  }
}

Expr AlterOpLayoutMutator::VisitCache_(const Expr& e) {
  auto it = cache_.find(e.get());
  if (it != cache_.end()) {
    return it->second;
  } else {
    Expr mutated = ExprMutator::Mutate(e);
    cache_[e.get()] = mutated;
    return mutated;
  }
}

Expr AlterOpLayoutMutator::VisitExpr_(const CallNode* call) {
  LOG(INFO) << "(1) mutate args and collect the input layouts & shapes";
  Array<Expr> mutated_args;
  Array<Expr> raw_args;
  for (const auto& arg : call->args) {
    Expr mutated_arg = VisitCache_(arg);

    const auto* alternated_expr_arg = mutated_arg.as<LayoutAlternatedExprNode>();
    CHECK(alternated_expr_arg);
    alternated_expr_arg->Check();

    mutated_args.push_back(mutated_arg);
    raw_args.push_back(alternated_expr_arg->value);
  }

  LOG(INFO) << "(2) get old layout for " << GetRef<Expr>(call);
  LOG(INFO) << "raw_args = " << raw_args;
  Array<Layout> provided_old_layouts, request_old_layouts, out_old_layouts;
  std::tie(provided_old_layouts, request_old_layouts, out_old_layouts)
    = CallInfer(GetRef<Call>(call), mutated_args, true);
  LOG(INFO) << "provided_old_layouts = " << provided_old_layouts;
  for (auto l : provided_old_layouts) {
    LOG(INFO) << l;
  }
  LOG(INFO) << "request_old_layouts = " << request_old_layouts;
  for (auto l : request_old_layouts) {
    LOG(INFO) << l;
  }

  Expr new_call;
  if (enable_alter_) {
    // provided layouts can be undefined, we need to leverage current node to do reverse-inference.
    CHECK_EQ(provided_old_layouts.size(), request_old_layouts.size());
    for (size_t i = 0; i < provided_old_layouts.size(); ++i) {
//      if (!provided_old_layouts[i].defined()) {
//        provided_old_layouts.Set(i, request_old_layouts[i]);
//      } else {
      if (provided_old_layouts[i].defined()) {
        CHECK(provided_old_layouts[i].Equals(request_old_layouts[i]))
          << call->op << " inferred a mismatched layout with its input[" << i << "]"
          << ", provided: " << provided_old_layouts[i]
          << ", inferred: " << request_old_layouts[i] << ".";
      }
    }
    // update mutated_args's layouts
    Array<Expr> new_mutated_args;
    size_t idx_flatten = 0;
    for (const Expr& mutated_arg : mutated_args) {
      const auto* arg_ptr = mutated_arg.as<LayoutAlternatedExprNode>();
      CHECK(arg_ptr);
      auto node = make_node<LayoutAlternatedExprNode>();
      node->value = arg_ptr->value;
      node->new_layouts = arg_ptr->new_layouts;
      node->old_layouts = arg_ptr->old_layouts;
      node->new_types = arg_ptr->new_types;
      node->old_types = arg_ptr->old_types;
      node->memorizer = memorizer_;
      for (size_t i = 0; i < node->old_layouts.size(); ++i) {
        if (!node->old_layouts[i].defined()) {
          // var, const, etc.
          CHECK(!node->new_layouts[i].defined());
          node->old_layouts.Set(i, request_old_layouts[idx_flatten]);
          node->new_layouts.Set(i, request_old_layouts[idx_flatten]);
        }
        idx_flatten++;
      }
      auto new_mutated_arg = LayoutAlternatedExpr(node);
      cache_[node->value.get()] = new_mutated_arg;
      new_mutated_args.push_back(new_mutated_arg);
    }
    mutated_args = new_mutated_args;

    LOG(INFO) << "(3) call alter layout function for op " << GetRef<Expr>(call);
    new_call = CallAlter(GetRef<Call>(call), mutated_args, memorizer_);
  }

  if (new_call.defined()) {
    // Altered
    const auto* new_call_ptr = new_call.as<LayoutAlternatedExprNode>();
    CHECK(new_call_ptr);
    CHECK(new_call_ptr->value->checked_type_.defined());
    auto rnode = make_node<LayoutAlternatedExprNode>();
    rnode->value = new_call_ptr->value;
    LOG(INFO) << "out_old_layouts for " << rnode->value << " = " << out_old_layouts;
    rnode->new_layouts = new_call_ptr->new_layouts;
    if (!out_old_layouts.defined()) {
      out_old_layouts = Array<Layout>(rnode->new_layouts.size(), Layout::Undef());
    }
    rnode->old_layouts = out_old_layouts;
    rnode->new_types = new_call_ptr->new_types;
    rnode->old_types = call->checked_type()->is_type<TupleTypeNode>() ?
                       call->type_as<TupleTypeNode>()->fields :
                       Array<Type>{call->checked_type_};
    rnode->memorizer = memorizer_;
    return LayoutAlternatedExpr(rnode);
  } else {
    LOG(INFO) << "(4) get new layout for " << GetRef<Expr>(call);
    Array<Layout> provided_new_layouts, request_new_layouts, out_new_layouts;
    std::tie(provided_new_layouts, request_new_layouts, out_new_layouts)
      = CallInfer(GetRef<Call>(call), mutated_args, false);
    LOG(INFO) << "provided_new_layouts = " << provided_new_layouts;
    for (auto l : provided_new_layouts) LOG(INFO) << l;
    LOG(INFO) << "request_new_layouts = " << request_new_layouts;
    for (auto l : request_new_layouts) LOG(INFO) << l;
    CHECK_EQ(provided_new_layouts.size(), request_new_layouts.size());
    for (size_t i = 0; i < provided_new_layouts.size(); ++i) {
      if (!provided_new_layouts[i].defined()) {
        provided_new_layouts.Set(i, request_new_layouts[i]);
      }
    }

    LOG(INFO) << "(5) generate layout transform";
    ReplaceArgsMutator input_replacer =
      ReplaceArgsMutator::BuildInst(GetRef<Expr>(call), raw_args);

    Array<Expr> transformed_args;
    size_t idx_flatten = 0;
    for (const auto& fake_arg : input_replacer.fake_inputs_) {
      Expr transformed_arg = MapTuple<Expr>(fake_arg, [&](const Expr &non_tuple) {
        Expr res = memorizer_.Transform(non_tuple, provided_new_layouts[idx_flatten],
                                        request_new_layouts[idx_flatten]);
        idx_flatten++;
        return res;
      });
      transformed_args.push_back(transformed_arg);
    }
    // it is created by fake_args for *incremental* type inference.
    Expr fake_new_call = CallNode::make(call->op, transformed_args, call->attrs);
    new_call = input_replacer.Run(fake_new_call);

    LOG(INFO) << "(6) Infer type for new_call " << new_call;
    fake_new_call = InferType(fake_new_call, Module());
    new_call->checked_type_ = fake_new_call->checked_type();

    LOG(INFO) << "(7) Generate LayoutAlternatedExprNode";
    CHECK(new_call->checked_type_.defined());
    const size_t num_outputs = new_call->checked_type_->is_type<TupleTypeNode>() ?
                               new_call->type_as<TupleTypeNode>()->fields.size() : 1;
    if (!out_new_layouts.defined()) {
      out_old_layouts = Array<Layout>(num_outputs, Layout::Undef());
      out_new_layouts = Array<Layout>(num_outputs, Layout::Undef());
    }
    auto rnode = make_node<LayoutAlternatedExprNode>();
    rnode->value = new_call;
    rnode->new_layouts = out_new_layouts;
    rnode->new_types = new_call->checked_type()->is_type<TupleTypeNode>() ?
                       new_call->type_as<TupleTypeNode>()->fields :
                       Array<Type>{new_call->checked_type_};
    if (enable_alter_) {
      rnode->old_layouts = out_old_layouts;
      rnode->old_types = call->checked_type()->is_type<TupleTypeNode>() ?
                         call->type_as<TupleTypeNode>()->fields :
                         Array<Type>{call->checked_type_};
    } else {
      // for (root) altered call node, we don't know its old layouts/types here,
      // they are set in the `if (new_call.defined())` branch instead.
      // otherwise their old = new, since they are brand new nodes.
      rnode->old_layouts = rnode->new_layouts;
      rnode->old_types = rnode->new_types;
    }
    rnode->memorizer = memorizer_;
    return LayoutAlternatedExpr(rnode);
  }
}

Expr AlterOpLayoutMutator::VisitExpr_(const VarNode* op) {
  CHECK(op->checked_type_.defined()) << GetRef<Expr>(op);
  auto ret = make_node<LayoutAlternatedExprNode>();
  ret->value = GetRef<Expr>(op);
  ret->old_layouts = Array<Layout>{Layout::Undef()};
  ret->new_layouts = Array<Layout>{Layout::Undef()};
  ret->old_types = Array<Type>{op->checked_type()};
  ret->new_types = Array<Type>{op->checked_type()};
  ret->memorizer = memorizer_;
  return LayoutAlternatedExpr(ret);
}

Expr AlterOpLayoutMutator::VisitExpr_(const ConstantNode* op) {
  auto ret = make_node<LayoutAlternatedExprNode>();
  ret->value = op->checked_type_.defined() ? GetRef<Expr>(op) : InferType(GetRef<Expr>(op), Module());
  ret->old_layouts = Array<Layout>{Layout::Undef()};
  ret->new_layouts = Array<Layout>{Layout::Undef()};
  ret->old_types = Array<Type>{ret->value->checked_type()};
  ret->new_types = Array<Type>{ret->value->checked_type()};
  ret->memorizer = memorizer_;
  return LayoutAlternatedExpr(ret);
}

Expr AlterOpLayoutMutator::VisitExpr_(const TupleNode* op) {
  auto ret = make_node<LayoutAlternatedExprNode>();
  Array<Type> types;
  Array<Expr> new_fields;
  for (Expr field : op->fields) {
    Expr mutated_field = VisitCache_(field);
    const auto* field_ptr = mutated_field.as<LayoutAlternatedExprNode>();
    CHECK(field_ptr);
    field_ptr->Check();
    CHECK(field_ptr->value->checked_type_.defined());
    types.push_back(field_ptr->value->checked_type_);
    for (size_t i = 0; i < field_ptr->old_layouts.size(); ++i) {
      ret->old_layouts.push_back(field_ptr->old_layouts[i]);
      ret->new_layouts.push_back(field_ptr->new_layouts[i]);
      ret->old_types.push_back(field_ptr->old_types[i]);
      ret->new_types.push_back(field_ptr->new_types[i]);
    }
    new_fields.push_back(field_ptr->value);
  }
  ret->value = TupleNode::make(new_fields);
  ret->value->checked_type_ = TupleTypeNode::make(types);
  ret->memorizer = memorizer_;
  return LayoutAlternatedExpr(ret);
}

Expr AlterOpLayoutMutator::VisitExpr_(const TupleGetItemNode* op) {
  auto new_tuple = VisitCache_(op->tuple);
  const auto* ptr = new_tuple.as<LayoutAlternatedExprNode>();
  CHECK(ptr);
  ptr->Check();
  CHECK(ptr->value->checked_type_.defined());

  auto ret = make_node<LayoutAlternatedExprNode>();
  ret->old_layouts.push_back(ptr->old_layouts[0]);
  ret->new_layouts.push_back(ptr->new_layouts[0]);
  ret->old_types.push_back(ptr->old_types[0]);
  ret->new_types.push_back(ptr->new_types[0]);
  ret->memorizer = memorizer_;

  if (const auto* ptuple = ptr->value.as<TupleNode>()) {
    ret->value = ptuple->fields[op->index];
  } else {
    ret->value = TupleGetItemNode::make(ptr->value, op->index);
  }
  CHECK(ptr->value->checked_type()->is_type<TupleTypeNode>());
  ret->value->checked_type_ = ptr->value->type_as<TupleTypeNode>()->fields[op->index];
  return LayoutAlternatedExpr(ret);
}

Expr AlterOpLayoutMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<TypeVar> ty_params;
  for (auto ty_param : op->type_params) {
    TypeVar new_ty_param = Downcast<TypeVar>(VisitType(ty_param));
    ty_params.push_back(new_ty_param);
  }

  tvm::Array<Var> params;
  for (auto param : op->params) {
    const auto* ptr = this->Mutate(param).as<LayoutAlternatedExprNode>();
    CHECK(ptr);
    Var new_param = Downcast<Var>(ptr->Realize());
    params.push_back(new_param);
  }

  auto ret_type = this->VisitType(op->ret_type);
  const auto* body_ptr = this->Mutate(op->body).as<LayoutAlternatedExprNode>();
  CHECK(body_ptr);
  auto body = body_ptr->Realize();

  if (ty_params.same_as(op->type_params) &&
      params.same_as(op->params) &&
      ret_type.same_as(op->ret_type) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return FunctionNode::make(params, body, ret_type, ty_params, op->attrs);
  }
}

//  // NOTE: discard the "const" qualifier
//  TransformMemorizer memorizer = Downcast<TransformMemorizer>(ctx);

// Limiations:
// 1. the altered op should have the same number of arguments as the previous one
// 2. do not support nested tuple arguments
TVM_REGISTER_API("relay._ir_pass.AlterOpLayout")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  TransformMemorizer memorizer(make_node<TransformMemorizerNode>());
  LOG(INFO) << "haha0";
  AlterOpLayoutMutator mutator(Array<Expr>(nullptr), true, memorizer);
  LOG(INFO) << "haha";
  Expr mutated = mutator.Run(args[0]);
  LOG(INFO) << "haha2";
  const auto* mutated_ptr = mutated.as<LayoutAlternatedExprNode>();
  LOG(INFO) << "haha3";
  if (mutated_ptr) {
    *ret = mutated_ptr->Realize();
  } else {
    *ret = mutated;
  }
});

}  // namespace relay
}  // namespace tvm
