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
#include <tvm/relay/pass.h>
#include <tvm/data_layout.h>
#include "./pass_util.h"
#include "type_solver.h"
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

/*! \brief Base type of the Relay Layout hierarchy. */
class BaseLayoutNode : public RelayNode {
public:
  static constexpr const char* _type_key = "relay.Layout";
  TVM_DECLARE_BASE_NODE_INFO(BaseLayoutNode, Node);
};

class BaseLayout : public NodeRef {
public:
  BaseLayout() {}
  explicit BaseLayout(NodePtr<tvm::Node> p) : NodeRef(p) {}
  using ContainerType = BaseLayoutNode;
};

class TensorLayout;
/*! \brief TensorType container node */
class TensorLayoutNode : public BaseLayoutNode {
public:
  Layout layout;

  void VisitAttrs(tvm::AttrVisitor *v) final {
    v->Visit("layout", &layout);
  }

  TVM_DLL static TensorLayout make(Layout layout);

  static constexpr const char* _type_key = "relay.TensorLayout";

  TVM_DECLARE_NODE_TYPE_INFO(TensorLayoutNode, BaseLayoutNode);
};

RELAY_DEFINE_NODE_REF(TensorLayout, TensorLayoutNode, BaseLayout);

TensorLayout TensorLayoutNode::make(Layout layout) {
  NodePtr<TensorLayoutNode> n = make_node<TensorLayoutNode>();
  n->layout = std::move(layout);
  return TensorLayout(n);
}

class TupleLayout;
/*!
 * \brief TupleType container.
 */
class TupleLayoutNode : public BaseLayoutNode {
public:
  /*! \brief The type of each field in the tuple. */
  tvm::Array<Layout> fields;

  TupleLayoutNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
  }

  TVM_DLL static TupleLayout make(tvm::Array<Layout> fields);

  static constexpr const char* _type_key = "relay.TupleLayout";
  TVM_DECLARE_NODE_TYPE_INFO(TupleLayoutNode, BaseLayoutNode);
};

RELAY_DEFINE_NODE_REF(TupleLayout, TupleLayoutNode, BaseLayout);

TupleLayout TupleLayoutNode::make(Array<Layout> fields) {
  NodePtr<TupleLayoutNode> n = make_node<TupleLayoutNode>();
  n->fields = std::move(fields);
  return TupleLayout(n);
}

class LayoutInferencer : private ExprFunctor<Type(const Expr&)> {
 public:
  // inference the type of expr.
  Expr Infer(Expr expr);

 private:
  // Visitor Logic
  Type VisitExpr_(const VarNode* op) final { }

  Type VisitExpr_(const GlobalVarNode* op) final {
    // module.lookup(var)
  }

  Type VisitExpr_(const ConstantNode* op) final { }

  Type VisitExpr_(const TupleNode* op) final { }

  Type VisitExpr_(const TupleGetItemNode* op) final { }

  Type VisitExpr_(const MatchNode* op) final { }

  Type VisitExpr_(const OpNode* op) final { }

  Type VisitExpr_(const LetNode* let) final { }

  Type VisitExpr_(const IfNode* ite) final { }

  Type VisitExpr_(const CallNode* call) final { }

  Type VisitExpr_(const FunctionNode* f) final { }

  Type VisitExpr_(const RefCreateNode* op) final { }

  Type VisitExpr_(const RefReadNode* op) final { }

  Type VisitExpr_(const RefWriteNode* op) final { }

  Type VisitExpr_(const ConstructorNode* c) final { }
};

Expr LayoutInferencer::Infer(Expr expr) {
}

Function InferLayout(const Expr& func) {
}

TVM_REGISTER_API("relay._ir_pass.infer_layout")
.set_body_typed<Expr(const Expr&)>([](const Expr& expr) {
  return InferLayout(expr);
});

}  // namespace relay
}  // namespace tvm
