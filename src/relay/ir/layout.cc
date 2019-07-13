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

LayoutReporter LayoutReporterNode::make(tvm::Array<Expr> node) {
  NodePtr<LayoutReporterNode> n = make_node<LayoutReporterNode>();
  n->node = std::move(node);
  return LayoutReporter(n);
}

void LayoutReporterNode::Assign(size_t index, const RelayLayout& layout) {
  CHECK_LT(index, this->node.size()) << "Index " << index << " out of bound. Size =  "
                                     << this->node.size();
  this->layout_map.Set(this->node[index], layout);
}

}  // namespace relay
}  // namespace tvm
