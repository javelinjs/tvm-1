/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file layout_infer.cc
 * \brief Relay layout inference and checking.
 *
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
#include <tvm/data_layout.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/layout.h>
#include <utility>
#include "./pass_util.h"
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

class LayoutInferencer : private ExprFunctor<RelayLayout(const Expr&)> {
 public:
  explicit LayoutInferencer(const Map<std::string, RelayLayout> in_layouts)
    : modified_(false), timestamp_(0), var_layouts_(std::move(in_layouts)) {}

  // inference the type of expr.
  void Infer(const Module& mod) {
    auto funcs = mod->functions;
    for (auto pair : funcs) {
      auto global_var = pair.first;
      auto func = pair.second;
      auto func_layout = this->VisitExpr(func);
      while (modified_) {
        modified_ = false;
        timestamp_++;
        func_layout = this->VisitExpr(func);
      }
      layout_map_.Set(global_var, func_layout);
    }
  }

  Map<Expr, Array<Layout> > CollectLayoutInfo() {
    Map<Expr, Array<Layout> > map;
    for (auto& iter : layout_map_) {
      auto layout = iter.second;
      if (auto* tensor_layout = layout.as<TensorLayoutNode>()) {
        map.Set(iter.first, Array<Layout>({tensor_layout->layout}));
      } else if (auto* tuple_layout = layout.as<TupleLayoutNode>()) {
        CHECK(tuple_layout);
        map.Set(iter.first, tuple_layout->fields);
      } else {
        CHECK(false) << "TODO: FuncLayoutNode";
      }
    }
    return map;
  }

 private:
  bool modified_;
  int timestamp_;
  Map<std::string, RelayLayout> var_layouts_;
  Map<Expr, RelayLayout> layout_map_;
  std::unordered_map<Expr, int, NodeHash, NodeEqual> layout_timestamp_;

  RelayLayout GetLayout(const Expr& expr) {
    auto it = layout_map_.find(expr);
    if (it == layout_map_.end() || layout_timestamp_[expr] < timestamp_) {
      auto layout = this->VisitExpr(expr);
      SetLayout(expr, layout);
    }
    return layout_map_[expr];
  }

  void SetLayout(const Expr& expr, const RelayLayout& layout) {
    layout_timestamp_[expr] = timestamp_;
    if (!layout_map_.count(expr) || !layout_map_[expr].Equals(layout)) {
      layout_map_.Set(expr, layout);
      modified_ = true;
    }
  }

  void SetLayout(const LayoutReporter& reporter) {
    for (auto& it : reporter->results) {
      SetLayout(it.first, it.second);
    }
  }

  inline RelayLayout UndefLayout(const Expr& expr) {
    if (expr->checked_type()->IsInstance<TupleTypeNode>()) {
      const size_t num_outputs = expr->type_as<TupleTypeNode>()->fields.size();
      return TupleLayoutNode::make(Array<Layout>(num_outputs, Layout::Undef()));
    } else {
      return TensorLayoutNode::make(Layout::Undef());
    }
  }

  // Visitor Logic
  RelayLayout VisitExpr_(const VarNode* op) final {
    if (!layout_map_.count(GetRef<Var>(op))) {
      auto layout = var_layouts_.count(op->name_hint())
          ? var_layouts_[op->name_hint()]
          : UndefLayout(GetRef<Var>(op));
      SetLayout(GetRef<Var>(op), layout);
    }
    return layout_map_[GetRef<Var>(op)];
  }

  RelayLayout VisitExpr_(const GlobalVarNode* op) final {
    auto expr = GetRef<GlobalVar>(op);
    if (!layout_map_.count(expr)) {
      SetLayout(expr, UndefLayout(expr));
    }
    return layout_map_[expr];
  }

  RelayLayout VisitExpr_(const ConstantNode* op) final {
    auto expr = GetRef<Constant>(op);
    if (!layout_map_.count(expr)) {
      SetLayout(expr, UndefLayout(expr));
    }
    return layout_map_[expr];
  }

  RelayLayout VisitExpr_(const TupleNode* op) final {
    LOG(FATAL) << "TupleNode";
  }

  RelayLayout VisitExpr_(const TupleGetItemNode* op) final {
    LOG(FATAL) << "TupleGetItemNode";
  }

  RelayLayout VisitExpr_(const OpNode* op) final {
    LOG(FATAL) << "OpNode";
  }

  RelayLayout VisitExpr_(const LetNode* let) final {
    LOG(FATAL) << "LetNode";
  }

  RelayLayout VisitExpr_(const IfNode* ite) final {
    LOG(FATAL) << "IfNode";
  }

  RelayLayout VisitExpr_(const CallNode* call) final {
    auto node = GetRef<Call>(call);
    Array<RelayLayout> layouts;
    Array<Type> types;
    Array<Expr> nodes;

    for (auto arg : call->args) {
      auto arg_layout = GetLayout(arg);
      layouts.push_back(arg_layout);
      if (const auto* func_type = arg->checked_type().as<FuncTypeNode>()) {
        types.push_back(func_type->ret_type);
      } else {
        types.push_back(arg->checked_type());
      }
      nodes.push_back(arg);
    }

    auto curr_layout = layout_map_.count(node) ? layout_map_[node] : UndefLayout(node);
    SetLayout(node, curr_layout);
    layouts.push_back(curr_layout);
    types.push_back(call->checked_type());
    nodes.push_back(GetRef<Call>(call));

    static auto finfer_layout = Op::GetAttr<FInferLayout>("FInferLayout");

    if (const auto* opnode = call->op.as<OpNode>()) {
      Op op = Downcast<Op>(call->op);
      if (finfer_layout.count(op)) {
        auto reporter = LayoutReporterNode::make(nodes, layouts);
        bool infer_success = finfer_layout[op](layouts, types, call->args.size(), call->attrs, reporter);
        if (infer_success) {
          SetLayout(reporter);
        }
      }
    }

    CHECK(layout_map_.count(node));
    return layout_map_[node];
  }

  RelayLayout VisitExpr_(const FunctionNode* f) final {
    for (auto param : f->params) {
      GetLayout(param);
    }
    return GetLayout(f->body);
  }

  RelayLayout VisitExpr_(const MatchNode* op) final {
    LOG(FATAL) << "MatchNode";
  }

  RelayLayout VisitExpr_(const RefCreateNode* op) final {
    LOG(FATAL) << "RefCreateNode";
  }

  RelayLayout VisitExpr_(const RefReadNode* op) final {
    LOG(FATAL) << "RefReadNode";
  }

  RelayLayout VisitExpr_(const RefWriteNode* op) final {
    LOG(FATAL) << "RefWriteNode";
  }

  RelayLayout VisitExpr_(const ConstructorNode* c) final {
    LOG(FATAL) << "ConstructorNode";
  }
};

Map<Expr, Array<Layout> > CollectLayoutInfo(const Module& mod,
                                            const Map<std::string, RelayLayout>& in_layouts
                                                  = Map<std::string, RelayLayout>()) {
  LayoutInferencer inferencer(in_layouts);
  inferencer.Infer(mod);
  return inferencer.CollectLayoutInfo();
}

TVM_REGISTER_API("relay._analysis.CollectLayoutInfo")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() == 2) {
      *ret = CollectLayoutInfo(args[0].operator Module(), args[1].operator Map<std::string, RelayLayout>());
    } else {
      *ret = CollectLayoutInfo(args[0].operator Module());
    }
  });


}  // namespace relay
}  // namespace tvm
