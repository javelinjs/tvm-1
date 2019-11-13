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
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for relay ops.
 */
#ifndef TVM_RELAY_OP_OP_COMMON_H_
#define TVM_RELAY_OP_OP_COMMON_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/layout.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "type_relations.h"
#include "../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

inline bool UnaryInferLayout(const Array<RelayLayout>& layouts,
                             const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const LayoutReporter& reporter) {
  CHECK_EQ(layouts.size(), 2);
  const auto* pin = layouts[0].as<TensorLayoutNode>();
  const auto* pout = layouts[1].as<TensorLayoutNode>();
  CHECK(pin && pout);
  const auto in = pin->layout;
  const auto out = pout->layout;
  if ((in.defined() && out.defined()) || (!in.defined() && !out.defined())) {
    CHECK(in.Equals(out));
    return false;
  }
  const auto tmp = in.defined() ? in : out;
  reporter->Assign(0, TensorLayoutNode::make(tmp));
  reporter->Assign(1, TensorLayoutNode::make(tmp));
  return true;
}

inline bool BinaryBroadcastInferLayout(const Array<RelayLayout>& layouts,
                                       const Array<Type>& types,
                                       int num_inputs,
                                       const Attrs& attrs,
                                       const LayoutReporter& reporter) {
  CHECK_EQ(layouts.size(), 3);
  CHECK_EQ(types.size(), 3);

  const auto* pout = layouts[2].as<TensorLayoutNode>();
  CHECK(pout);
  const Layout out = pout->layout;

  const auto* plhs_type = types[0].as<TensorTypeNode>();
  const auto* prhs_type = types[1].as<TensorTypeNode>();
  const auto* pout_type = types[2].as<TensorTypeNode>();
  CHECK(plhs_type && prhs_type && pout_type);
  const Array<IndexExpr> lhs_shape = plhs_type->shape;
  const Array<IndexExpr> rhs_shape = prhs_type->shape;
  const Array<IndexExpr> out_shape = pout_type->shape;

  size_t large_idx = lhs_shape.size() >= rhs_shape.size() ? 0 : 1;
  size_t small_idx = 1 - large_idx;
  const auto* ptr_small_layout = layouts[small_idx].as<TensorLayoutNode>();
  const auto* ptr_large_layout = layouts[large_idx].as<TensorLayoutNode>();
  CHECK(ptr_small_layout && ptr_large_layout);
  if (lhs_shape.size() == rhs_shape.size() && !ptr_large_layout->layout.defined()) {
    std::swap(large_idx, small_idx);
    std::swap(ptr_large_layout, ptr_small_layout);
  }
  const Layout small_layout = ptr_small_layout->layout;
  const Layout large_layout = ptr_large_layout->layout;
  size_t small_shape_size = lhs_shape.size() >= rhs_shape.size() ? rhs_shape.size() : lhs_shape.size();

  if (!out.defined()) {
    if (large_layout.defined()) {
      reporter->Assign(small_idx, TensorLayoutNode::make(
          large_layout.SubLayout(out_shape.size()-small_shape_size, small_shape_size)));
      reporter->Assign(2, TensorLayoutNode::make(large_layout));
      return true;
    } else {
      return false;
    }
  } else {  // reverse infer
    reporter->Assign(large_idx, layouts[2]);
    reporter->Assign(small_idx, TensorLayoutNode::make(
        large_layout.SubLayout(out_shape.size()-small_shape_size, small_shape_size)));
    return true;
  }
}

/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * We make the decision to always only expose positional argument.
 * We will do rewrapping in the frontend to support language
 * sugars such as keyword arguments and default value.

 * \param OpName the name of registry.
 */
#define RELAY_REGISTER_UNARY_OP(OpName)                     \
  TVM_REGISTER_API("relay.op._make." OpName)                \
    .set_body_typed<Expr(Expr)>([](Expr data) {             \
        static const Op& op = Op::Get(OpName);              \
        return CallNode::make(op, {data}, Attrs(), {});     \
      });                                                   \
  RELAY_REGISTER_OP(OpName)                                 \
    .set_num_inputs(1)                                      \
    .add_argument("data", "Tensor", "The input tensor.")    \
    .add_type_rel("Identity", IdentityRel)                  \
    .set_attr<TOpPattern>("TOpPattern", kElemWise)          \
    .set_attr<TOpIsStateful>("TOpIsStateful", false)        \
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",   \
                                   ElemwiseArbitraryLayout) \
    .set_attr<FInferLayout>("FInferLayout",                 \
                            UnaryInferLayout)


/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * We make the decision to always only expose positional argument.
 * We will do rewrapping in the frontend to support language
 * sugars such as keyword arguments and default value.
 *
 * \param OpName the name of registry.
 */
#define RELAY_REGISTER_BINARY_OP(OpName)                          \
  TVM_REGISTER_API("relay.op._make." OpName)                      \
    .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {    \
        static const Op& op = Op::Get(OpName);                    \
        return CallNode::make(op, {lhs, rhs}, Attrs(), {});       \
      });                                                         \
  RELAY_REGISTER_OP(OpName)                                       \
    .set_num_inputs(2)                                            \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")  \
    .add_argument("rhs", "Tensor", "The right hand side tensor.") \
    .add_type_rel("Broadcast", BroadcastRel)                      \
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)               \
    .set_attr<TOpIsStateful>("TOpIsStateful", false)              \
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",         \
                                   BinaryBroadcastLayout)         \
    .set_attr<FInferLayout>("FInferLayout",                       \
                            BinaryBroadcastInferLayout)

// Comparisons
#define RELAY_REGISTER_CMP_OP(OpName)                             \
  TVM_REGISTER_API("relay.op._make." OpName)                      \
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {      \
    static const Op& op = Op::Get(OpName);                        \
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});           \
  });                                                             \
  RELAY_REGISTER_OP(OpName)                                       \
    .set_num_inputs(2)                                            \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")  \
    .add_argument("rhs", "Tensor", "The right hand side tensor.") \
    .add_type_rel("BroadcastComp", BroadcastCompRel)              \
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)               \
    .set_attr<TOpIsStateful>("TOpIsStateful", false)              \
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",         \
                                   BinaryBroadcastLayout)         \
    .set_attr<FInferLayout>("FInferLayout",                       \
                            BinaryBroadcastInferLayout)


/*! \brief A helper class for matching and rewriting operators. */
template<typename R>
class OpMatch {
 public:
  using MatchFunc =
      std::function<R(const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_args)>;

  /*! \brief Match an operator with the given name.
   *  \param op_name The name of the operator to match.
   *  \param func The function to execute when it matches.
   *  \return A self-reference for builder style API.
   */
  inline OpMatch& Match(const std::string& op_name, MatchFunc func) {
    auto op = Op::Get(op_name);
    match_map_.insert({op, func});
    return *this;
  }

  /*! \brief Rewrite a call operation based on the operator and the registered
   *  match functions.
   * \param call The call to rewrite.
   * \return The result of rewriting.
   */
  inline R operator()(const Call& call) {
    auto it = match_map_.find(Downcast<Op>(call->op));
    if (it != match_map_.end()) {
      return it->second(call->args, call->attrs, call->type_args);
    } else {
      if (default_ != nullptr) {
        return default_(call->args, call->attrs, call->type_args);
      } else {
        LOG(FATAL) << "unexpected operation " << call->op;
      }
    }
  }

 private:
  /*! \brief The match function map. */
  std::unordered_map<Op, MatchFunc, NodeHash, NodeEqual> match_map_;
  /*! \brief An optional default case. */
  MatchFunc default_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_OP_COMMON_H_
