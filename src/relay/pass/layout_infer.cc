/*!
 *  Copyright (c) 2018 by Contributors
 * \file layout_infer.cc
 * \brief Relay type inference and checking.
 *
 * TODO:
 * This file implements one of the most important passes to the
 * Relay IR. In order to do many transformations and generate the
 * most efficient code we need to obtain type information for the
 * IR.
 *
 * Like computation graphs the IR leaves most type information
 * implicit and relies performing analysis of the program to
 * generate this information.
 *
 * This pass given an expression `e` will infer a type `t` for
 * the expression simultaneous checking the property `e : t`
 * (i.e we can show e has type t).
 *
 * If we can not infer a type or there are conflicting typing
 * constraints we will trigger an error.
 */

#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/pass.h>
#include "./layout_infer.h"
#include "./pass_util.h"
#include "type_solver.h"
#include "../ir/type_functor.h"

#include "./alter_op_layout.h"

namespace tvm {
namespace relay {

Array<Layout> RelayBaseLayout::Flatten() const {
  const auto* rlayout = this->as<RelayLayoutNode>();
  if (rlayout) {
    return Array<Layout>{ rlayout->layout };
  }
  const auto* tlayout = this->as<RelayTupleLayoutNode>();
  CHECK(tlayout);
  Array<Layout> ret;
  for (const auto& l : tlayout->fields) {
    for (Layout layout : l.Flatten()) {
      ret.push_back(layout);
    }
  }
  return ret;
}

RelayLayout RelayLayoutNode::make(Layout layout) {
  NodePtr<RelayLayoutNode> n = make_node<RelayLayoutNode>();
  n->layout = std::move(layout);
  return RelayLayout(n);
}

RelayTupleLayout RelayTupleLayoutNode::make(tvm::Array<RelayBaseLayout> fields) {
  NodePtr<RelayTupleLayoutNode> n = make_node<RelayTupleLayoutNode>();
  n->fields = std::move(fields);
  return RelayTupleLayout(n);
}

RelayBaseLayout LayoutInferencer::VisitExpr_(const CallNode* call) {
  static auto finfer_layout = Op::GetAttr<FInferCorrectLayout>("FInferCorrectLayout");
  Op op = Downcast<Op>(call->op);

  Array<Layout> arg_flatten_layouts;
  Array<Array<IndexExpr> > arg_shapes;
  for (Expr arg : call->args) {
    RelayBaseLayout arg_layout = GetLayout(arg);
    for (Layout layout : arg_layout.Flatten()) {
      arg_flatten_layouts.push_back(layout);
    }
    const auto* ttype = arg->type_as<TensorTypeNode>();
    CHECK(ttype);
    arg_shapes.push_back(ttype->shape);
  }

  Array<Array<Layout> > inferred_layouts;
  if (finfer_layout.count(op)) {
    inferred_layouts = finfer_layout[op](call->attrs, arg_flatten_layouts,
                                         arg_flatten_layouts, arg_shapes);
  } else {
    // default layout
    const size_t num_outputs = call->checked_type_->is_type<TupleTypeNode>() ?
                               call->type_as<TupleTypeNode>()->fields.size() : 1;
    inferred_layouts.push_back(arg_flatten_layouts);
    inferred_layouts.push_back(Array<Layout>(num_outputs, Layout::Undef()));
  }

  CHECK(inferred_layouts.size() == 2); // (in_layouts, out_layouts)
  Array<Layout> output_layouts = inferred_layouts[1];
  CHECK(output_layouts.size() >= 1);
  if (output_layouts.size() == 1) {
    return RelayLayoutNode::make(output_layouts[0]);
  } else {
    Array<RelayBaseLayout> outs;
    for (const Layout& output_layout : output_layouts) {
      outs.push_back(RelayLayoutNode::make(output_layout));
    }
    return RelayTupleLayoutNode::make(outs);
  }
}

Map<Expr, RelayBaseLayout> LayoutInferencer::Infer(Expr expr) {
  GetLayout(expr);
  Map<Expr, RelayBaseLayout> outputs;
  for (auto it = layout_map_.begin(); it != layout_map_.end(); ++it) {
    outputs.Set(it->first, it->second);
  }
  return outputs;
}

Map<Expr, RelayBaseLayout> CollectLayout(const Expr& expr) {
  LayoutInferencer inferencer;
  return inferencer.Infer(expr);
}

TVM_REGISTER_API("relay._ir_pass.infer_layout")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = CollectLayout(args[0]);
});

}  // namespace relay
}  // namespace tvm
