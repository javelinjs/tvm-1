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
 * \file src/tvm/ir/type.cc
 * \brief The type system AST nodes of Relay.
 */
#include <tvm/relay/layout.h>

namespace tvm {
namespace relay {

TensorLayout TensorLayoutNode::make(Layout layout) {
  NodePtr<TensorLayoutNode> n = make_node<TensorLayoutNode>();
  n->layout = std::move(layout);
  return TensorLayout(n);
}

TupleLayout TupleLayoutNode::make(Array<Layout> fields) {
  NodePtr<TupleLayoutNode> n = make_node<TupleLayoutNode>();
  n->fields = std::move(fields);
  return TupleLayout(n);
}

LayoutReporter LayoutReporterNode::make(Array<Expr> args, Array<RelayLayout> args_layout) {
  NodePtr<LayoutReporterNode> n = make_node<LayoutReporterNode>();
  n->args = std::move(args);
  n->args_layout = std::move(args_layout);
  return LayoutReporter(n);
}

bool RelayLayout::Equals(const RelayLayout &rhs) const {
  const auto* lhs_tensor_layout = as<TensorLayoutNode>();
  const auto* rhs_tensor_layout = rhs.as<TensorLayoutNode>();
  if (lhs_tensor_layout && rhs_tensor_layout) {
    return lhs_tensor_layout->layout.Equals(rhs_tensor_layout->layout);
  }
  const auto* lhs_tuple_layout = as<TupleLayoutNode>();
  const auto* rhs_tuple_layout = rhs.as<TupleLayoutNode>();
  if (lhs_tuple_layout && rhs_tuple_layout) {
    const auto lhs_fields = lhs_tuple_layout->fields;
    const auto rhs_fields = rhs_tuple_layout->fields;
    if (lhs_fields.size() == rhs_fields.size()) {
      for (size_t i = 0; i < lhs_fields.size(); ++i) {
        if (!lhs_fields[i].Equals(rhs_fields[i])) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

void LayoutReporterNode::Assign(size_t index, const RelayLayout& layout) {
  CHECK_LT(index, this->args.size()) << "Index " << index << " out of bound. Size =  "
                                     << this->args.size();
  this->results.Set(this->args[index], layout);
}

}  // namespace relay
}  // namespace tvm
