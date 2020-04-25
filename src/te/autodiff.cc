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
 * \file autodiff.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/autodiff.h>
#include <tvm/te/zero_elimination.h>
#include <tvm/tir/stmt_functor.h>
#include <memory>
#include <topi/transform.h>
#include "operation/op_util.h"

namespace tvm {
namespace te {

DifferentiationResult::DifferentiationResult(Array<Tensor> result,
                                             Map<Tensor, Tensor> adjoints,
                                             Map<Tensor, Map<Tensor, Tensor>> adjoint_summands)
   : DifferentiationResult(make_object<DifferentiationResultNode>(
       result, adjoints, adjoint_summands)) {}

TVM_REGISTER_NODE_TYPE(DifferentiationResultNode);

#define NOT_IMPLEMENTED \
  { LOG(FATAL) << "Derivative of this expr is not implemented: " << GetRef<PrimExpr>(op); throw; }

/*! \brief Differentiate an expression wrt a variable or a tensor element */
class JacobianMutator : public ExprMutator {
 public:
  /*!
   * \brief Differentiate wrt `input(indices)`.
   * \param input The input tensor.
   * \param indices The indices of the element with respect to which to differentiate.
   */
  explicit JacobianMutator(Tensor input, Array<PrimExpr> indices)
    : input_(input), indices_(indices) {}
  /*!
   * \brief Differentiate wrt the input variable.
   * \param input The input variable.
   */
  explicit JacobianMutator(Var input) : input_var_(input) {}

  virtual PrimExpr VisitExpr(PrimExpr e) {
    if (e.dtype().is_int() || e.dtype().is_uint()) {
      LOG(WARNING) << "For now we assume that the derivative of any integer expression is always 0."
                   << " e = " << e;
      return make_zero(e.dtype());
    } else {
      return ExprMutator::VisitExpr(e);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    if (input_var_.get() && input_var_.get() == op && op->dtype.is_float()) {
      return FloatImm(op->dtype, 1.0);
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LetNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const CallNode* op) {
    if (op->call_type == CallNode::CallType::Halide) {
      if (input_.get() && op->func.same_as(input_->op) &&
          op->value_index == input_->value_index) {
        // Tensor(indices)
        CHECK_EQ(indices_.size(), op->args.size());
        PrimExpr condition = const_true();
        for (size_t i = 0; i < input_.ndim(); ++i) {
          condition = AndNode::make(condition, EQNode::make(indices_[i], op->args[i]));
        }
        return CastNode::make(op->dtype, condition);
      } else {
        return make_zero(op->dtype);
      }
    }
    NOT_IMPLEMENTED
  }

  PrimExpr VisitExpr_(const AddNode* op) {
    return AddNode::make(VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const SubNode* op) {
    return SubNode::make(VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const MulNode* op) {
    return AddNode::make(
        MulNode::make(VisitExpr(op->a), op->b),
        MulNode::make(op->a, VisitExpr(op->b)));
  }

  PrimExpr VisitExpr_(const DivNode* op) {
    return DivNode::make(
        SubNode::make(
            MulNode::make(VisitExpr(op->a), op->b),
            MulNode::make(op->a, VisitExpr(op->b))),
        MulNode::make(op->b, op->b));
  }

  PrimExpr VisitExpr_(const ModNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const FloorDivNode* op) {
    return FloorDivNode::make(
        SubNode::make(
            MulNode::make(VisitExpr(op->a), op->b),
            MulNode::make(op->a, VisitExpr(op->b))),
        MulNode::make(op->b, op->b));
  }

  PrimExpr VisitExpr_(const FloorModNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const MinNode* op) {
    return SelectNode::make(LENode::make(op->a, op->b),
        VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const MaxNode* op) {
    return SelectNode::make(GENode::make(op->a, op->b),
        VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const EQNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const NENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LTNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const GTNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const GENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const AndNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const OrNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const ReduceNode* op) {
    // This case is relatively difficult because a reduction expression
    // may use an arbitrary combiner.
    // The resulting reduction expression will return a tuple containing
    // both derivatives and the original results (in exactly this order).

    // We have to clone the reduction axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    PrimExpr expr_with_new_axes = te::CloneReduction(GetRef<PrimExpr>(op));
    const ReduceNode* new_op = expr_with_new_axes.as<ReduceNode>();

    // New lhs and rhs variables of the new combiner consist of
    // variables representing derivatives (which are later derived from new_op->source)
    // followed by the original variables.
    Array<Var> new_lhs;
    for (const auto& var : new_op->combiner->lhs) {
      new_lhs.push_back(var.copy_with_suffix(".der"));
    }
    for (const auto& var : new_op->combiner->lhs) {
      new_lhs.push_back(var);
    }

    Array<Var> new_rhs;
    for (const auto& var : new_op->combiner->rhs) {
      new_rhs.push_back(var.copy_with_suffix(".der"));
    }
    for (const auto& var : new_op->combiner->rhs) {
      new_rhs.push_back(var);
    }

    // The new combiner result also consists of the resulting derivatives
    // followed by the original results.
    Array<PrimExpr> new_result;
    for (const auto& res : new_op->combiner->result) {
      // Each resulting derivative is computed as a sum of derivatives
      // wrt lhs and rhs multiplied by the derivatives of lhs and rhs
      PrimExpr new_res = make_zero(res.dtype());
      for (size_t i = 0; i < new_op->combiner->lhs.size(); ++i) {
        PrimExpr res_di = Derivative(res, new_op->combiner->lhs[i]);
        // new_lhs[i] is the derivative of lhs[i] (wrt our input tensor)
        new_res = AddNode::make(new_res, MulNode::make(new_lhs[i], res_di));
      }
      for (size_t i = 0; i < new_op->combiner->rhs.size(); ++i) {
        PrimExpr res_di = Derivative(res, new_op->combiner->rhs[i]);
        // new_rhs[i] is the derivative of rhs[i] (wrt our input tensor)
        new_res = AddNode::make(new_res, MulNode::make(new_rhs[i], res_di));
      }
      new_result.push_back(new_res);
    }
    // add original results
    for (const auto& res : new_op->combiner->result) {
      new_result.push_back(res);
    }

    // The identity is transformed in a similar way
    Array<PrimExpr> new_identity;
    for (const auto& id : new_op->combiner->identity_element) {
      new_identity.push_back(VisitExpr(id));
    }
    for (const auto& id : new_op->combiner->identity_element) {
      new_identity.push_back(id);
    }

    // Same as source
    Array<PrimExpr> new_source;
    for (const auto& src : new_op->source) {
      new_source.push_back(VisitExpr(src));
    }
    for (const auto& src : new_op->source) {
      new_source.push_back(src);
    }

    CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
    // Also simplify the resulting combiner
    // (mostly to get rid of unused components, e.g., the original expressions)
    return Simplify(
        ReduceNode::make(new_combiner, new_source, new_op->axis, new_op->condition, new_op->value_index));
  }

  PrimExpr VisitExpr_(const CastNode* op) {
    if (op->dtype.is_float()) {
      return CastNode::make(op->dtype, VisitExpr(op->value));
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const NotNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const SelectNode* op) {
    return SelectNode::make(op->condition,
        VisitExpr(op->true_value), VisitExpr(op->false_value));
  }

  PrimExpr VisitExpr_(const RampNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const BroadcastNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const ShuffleNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const IntImmNode* op) {
    return IntImm(op->dtype, 0);
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) {
    return FloatImm(op->dtype, 0);
  }

  PrimExpr VisitExpr_(const StringImmNode* op) NOT_IMPLEMENTED

 private:
  Tensor input_;
  Array<PrimExpr> indices_;
  Var input_var_;
};

PrimExpr Derivative(const PrimExpr& expr, const Var& var) {
  return JacobianMutator(var).VisitExpr(expr);
}

PrimExpr Jacobian(const PrimExpr& expr, const Tensor& input, const Array<PrimExpr>& indices) {
  LOG(INFO) << "body = " << expr;
  return JacobianMutator(input, indices).VisitExpr(expr);
}

Tensor Jacobian(const Tensor& output, const Tensor& input, bool optimize) {
  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op) << "Derivative of this op is not implemented: " << output->op;
  bool is_input_tensor = false;
  for (const Tensor& child : op->InputTensors()) {
    if (input == child) {
      is_input_tensor = true;
      break;
    }
  }
  CHECK(is_input_tensor) << "Jacobian is called on a pair of tensors such that the output "
                         << "does not depend on the input. This is probably a mistake.";

  // We have to clone the iteration axes because otherwise the original expression
  // cannot be used together with the derivative (it will lead to errors during lowering)
  Array<IterVar> new_axis;
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (IterVar iv : op->axis) {
    IterVar new_v =
      IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""),
            iv->iter_type, iv->thread_tag);
    new_axis.push_back(new_v);
    vmap[iv->var.get()] = new_v;
  }

  Array<PrimExpr> input_indices;
  size_t i = 0;
  for (PrimExpr ext : input->shape) {
    IterVar new_v = IterVarNode::make(Range(0, ext), Var("jac_i" + std::to_string(i++)),
        IterVarType::kDataPar);
    // Append jacobian iter to new_axis
    new_axis.push_back(new_v);
    // Differentiate wrt input[input_indices]
    input_indices.push_back(new_v);
  }

  // Compute Jacobian
  PrimExpr new_body = Jacobian(Substitute(op->body[output->value_index], vmap), input, input_indices);
  new_body = Simplify(new_body);
  LOG(INFO) << "body.der = " << new_body;

  int value_index = 0;
  Array<PrimExpr> new_bodies;

  // If this is a reduction then it may return a tuple and we have
  // to repeat the body several times
  if (const ReduceNode* red = new_body.as<ReduceNode>()) {
    value_index = red->value_index;
    for (size_t i = 0; i < red->source.size(); ++i) {
      new_bodies.push_back(
            ReduceNode::make(red->combiner, red->source, red->axis, red->condition, i));
    }
  } else {
    new_bodies.push_back(new_body);
  }

  auto new_op = ComputeOpNode::make(op->name + ".jacobian", op->tag, op->attrs, new_axis, new_bodies);

  // Jacobian shape = output.shape + input.shape
  Array<PrimExpr> new_shape = output->shape;
  for (const auto& e : input->shape) {
    new_shape.push_back(e);
  }

  Tensor tensor = TensorNode::make(new_shape, output->dtype, new_op, value_index);

  if (optimize) {
    tensor = OptimizeAndLiftNonzeronessConditions(tensor);
  }

  return tensor;
}

PrimExpr InlineThisCall(const PrimExpr& expr) {
  if (const CallNode* op = expr.as<CallNode>()) {
    if (op->call_type == CallNode::CallType::Halide) {
      if (const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>()) {
        Array<Var> tensor_axes;
        for (const auto& var : op_comp->axis) {
          tensor_axes.push_back(var->var);
        }

        Stmt inlined = Inline(EvaluateNode::make(expr), op->func, tensor_axes,
                              op_comp->body[op->value_index]);
        if (const EvaluateNode* ev = inlined.as<EvaluateNode>()) {
          // If it is a reduction, clone it
          return CloneReduction(ev->value);
        }
      }
    }
  }
  return expr;
}

// Implements InlineTensors by trying to inline every Call of the given Expr
class InlineTensorsMutator : public ExprMutator {
 public:
  explicit InlineTensorsMutator(const Array<Tensor>& inlineable, bool inline_reductions = false)
      : inline_reductions_(inline_reductions) {
    for (const Tensor& tensor : inlineable) {
      inlineable_.emplace(tensor->op.operator->(), tensor->value_index);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) {
    if (op->call_type == CallNode::CallType::Halide) {
      if (const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>()) {
        // Inline only if the array of inlineable tensors is empty or contains this tensor
        if (inlineable_.empty() || inlineable_.count({op_comp, op->value_index})) {
          // Inline only compute nodes that are not reductions (unless inline reductions is allowed)
          if (inline_reductions_ || !op_comp->body[0].as<ReduceNode>()) {
            // Inline this call and then try to perform further inlining
            return VisitExpr(InlineThisCall(GetRef<PrimExpr>(op)));
          }
        }
      }
    }

    // If we cannot inline this call, we should try to do inlining in its arguments
    return ExprMutator::VisitExpr_(op);
  }

 private:
  // Tensors which are allowed to be inlined, represented as pairs (op_node, value_index)
  std::set<std::pair<const OperationNode*, int>> inlineable_;
  bool inline_reductions_;
};

Tensor InlineTensors(const Tensor& tensor, const Array<Tensor>& inlineable,
                     bool inline_reductions) {
  const ComputeOpNode* op = tensor->op.as<ComputeOpNode>();
  CHECK(op);
  PrimExpr body = op->body[tensor->value_index];
  Array<IterVar> axis = op->axis;

  LOG(INFO) << "body = " << body;

  body = InlineTensorsMutator(inlineable, inline_reductions)(body);

  return TensorFromExpr(body, op->axis, op->name, op->tag, op->attrs, /*clone_axis=*/true);
}

// If expr is a Call node, perform inlining, otherwise do nothing
Tensor InlineThisCall(const Tensor& tensor) {
  const ComputeOpNode* op = tensor->op.as<ComputeOpNode>();
  CHECK(op);
  PrimExpr body = InlineThisCall(op->body[tensor->value_index]);
  return TensorFromExpr(body, op->axis, op->name, op->tag, op->attrs, /*clone_axis=*/true);
}

Tensor DiffBuildingBlock(const Tensor& output, const Tensor& input, const Tensor& head) {
  Tensor jac_output_input = Jacobian(output, input);
  Tensor result = topi::tensordot(head, jac_output_input, output->shape.size(),
                                  output->op->name + "." + input->op->name + ".grad");
  // TODO(sgrechanik-h): Here we inline only jac_output_input because otherwise there will be
  // performance problems. A better solution would be to inline smartly.
  result = InlineTensors(result, {jac_output_input}, /*inline_reductions=*/false);
  result = OptimizeAndLiftNonzeronessConditions(result);
  result = InlineThisCall(result);
  return result;
}

TVM_REGISTER_GLOBAL("tir.Jacobian")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() > 2) {
      *ret = Jacobian(args[0], args[1], args[2].operator bool());
    } else {
      *ret = Jacobian(args[0], args[1]);
    }
  });

TVM_REGISTER_GLOBAL("tir.DiffBuildingBlock")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = DiffBuildingBlock(args[0], args[1], args[2]);
  });

}  // namespace te
}  // namespace tvm
