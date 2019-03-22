#include <utility>

/*!
 * Copyright (c) 2018 by Contributors
 * \file alter_operator.cc
 * \brief TODO: Canonicalize special operators to basic operators.
    This can simplify latter analysis. (e.g. Expand bias_add to expand_dims and broadcast_add.)
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/tvm.h>
#include "pattern_util.h"
#include "./layout_infer.h"

namespace tvm {
namespace relay {

template<typename T>
Array<T> FlatMapTupleType(const Array<Type>& types,
                          std::function<T(const Type&)> mapper) {
  Array<T> results;
  for (const Type& t : types) {
    if (t->is_type<TupleTypeNode>()) {
      const auto* tuple_type = t.as<TupleTypeNode>();
      for (const T& ret : FlatMapTupleType(tuple_type->fields, mapper)) {
        results.push_back(ret);
      }
    } else {
      results.push_back(mapper(t));
    }
  }
  return results;
}

class OpMutator : public ExprMutator {
 public:
  explicit OpMutator(Map<Expr, RelayBaseLayout> op_layouts) : op_layouts_(std::move(op_layouts)) {}

  // Automatic fold TupleGetItem.
  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr tuple = this->VisitCache_(op->tuple);
    if (const auto* ptuple = tuple.as<TupleNode>()) {
      return ptuple->fields[op->index];
    } else {
      if (tuple.same_as(op->tuple)) {
        return GetRef<Expr>(op);
      } else {
        return TupleGetItemNode::make(tuple, op->index);
      }
    }
  }

  Expr VisitExpr_(const CallNode* n) {
    static auto falter_layout = Op::GetAttr<FTVMAlterOperator>("FTVMAlterOperator");
    auto ref_call = GetRef<Call>(n);
    Op op = Downcast<Op>(ref_call->op);

    CHECK(op_layouts_.count(ref_call));
    RelayBaseLayout op_layout = op_layouts_[ref_call];

    Array<Expr> new_args;
    Array<RelayBaseLayout> arg_layouts;
    for (const Expr& arg: ref_call->args) {
      CHECK(op_layouts_.count(arg));
      arg_layouts.push_back(op_layouts_[ref_call]);
      new_args.push_back(VisitCache_(arg));
    }

    if (falter_layout.count(op)) {
      Array<Type> arg_types;
      for (const Expr& arg : new_args) {
        arg_types.push_back(arg->checked_type());
      }

      Array<tvm::Tensor> tinfos = FlatMapTupleType<tvm::Tensor>(arg_types, [&](const Type& type) -> tvm::Tensor {
        const auto* ttype = type.as<TensorTypeNode>();
        CHECK(ttype);
        return tvm::placeholder(ttype->shape, ttype->dtype);
      });

      Array<BijectiveLayout> in_layouts;
      Array<BijectiveLayout> out_layouts;
      for (const RelayBaseLayout& arg_layout : arg_layouts) {
        Array<Layout> flattened = arg_layout.Flatten();
        for (const Layout& layout : flattened) {
          // TODO: remove hard-coded NCHW
          in_layouts.push_back(BijectiveLayoutNode::make(layout, LayoutNode::make("NCHW")));
        }
      }
      for (const Layout& layout : op_layout.Flatten()) {
        // TODO: we don't provide suggestion for output layout at the moment
        out_layouts.push_back(BijectiveLayoutNode::make(layout, Layout::Undef()));
      }

      Expr altered_value = falter_layout[op](ref_call->attrs, ref_call->args, tinfos, in_layouts, out_layouts);
      if (altered_value.defined()) {
        return altered_value;
      }
    }
    // TODO: flatten tuple node
    return CallNode::make(ref_call->op, new_args, ref_call->attrs);;
  }
 private:
  Expr VisitCache_(const Expr& e) {
    auto it = cache_.find(e.get());
    if (it != cache_.end()) {
      return it->second;
    } else {
      Expr mutated = ExprMutator::Mutate(e);
      cache_[e.get()] = mutated;
      return mutated;
    }
  }

  std::unordered_map<const Node*, Expr> cache_;
  Map<Expr, RelayBaseLayout> op_layouts_;
};

Expr MutateOperators(const Expr& e, const Map<Expr, RelayBaseLayout>& layouts) {
  return OpMutator(layouts).Mutate(e);
}

TVM_REGISTER_API("relay._ir_pass.alter_operator")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  LayoutInferencer inferencer;
  auto layouts = inferencer.Infer(args[0]);
  *ret = MutateOperators(args[0], layouts);
});

}  // namespace relay
}  // namespace tvm

