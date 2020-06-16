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
 * \file jac_elimination.cc
 * \brief TODO
 */
#include <dmlc/optional.h>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/autodiff.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <memory>
#include <utility>

#include "../schedule/operation_inline.h"
#include "ad_util.h"

namespace tvm {
namespace te {

template <class K, class V>
Map<K, V> Merge(Map<K, V> original, const Map<K, V>& update) {
  for (const auto& p : update) {
    original.Set(p.first, p.second);
  }
  return std::move(original);
}

Map<Var, Range> IterVarsToMap(const Array<IterVar>& itervars) {
  Map<Var, Range> res;
  for (const IterVar& v : itervars) {
    res.Set(v->var, v->dom);
  }
  return res;
}

// Concatenate two arrays
template <class T>
Array<T> Concat(Array<T> a, const Array<T>& b) {
  for (const auto& x : b) {
    a.push_back(x);
  }
  return std::move(a);
}

template <typename ValueType>
inline bool is_const_value(const PrimExpr& e, ValueType value) {
  static_assert(std::is_integral<ValueType>::value,
                "Comparison to non-integer values is forbidden.");
  // This implementation was copy-pasted from HalideIR
  if (const tir::IntImmNode* i = e.as<tir::IntImmNode>()) {
    return i->value == value;
  } else if (const tir::FloatImmNode* i = e.as<tir::FloatImmNode>()) {
    return i->value == value;
  } else if (const tir::CastNode* c = e.as<tir::CastNode>()) {
    return is_const_value(c->value, value);
  } else if (const tir::BroadcastNode* b = e.as<tir::BroadcastNode>()) {
    return is_const_value(b->value, value);
  } else {
    return false;
  }
}

// Return true if this combiner is just a sum.
bool IsSumCombiner(const CommReducer& combiner, const Map<Var, Range>& vranges) {
  arith::Analyzer analyzer;
  analyzer.Bind(vranges);
  if (combiner->result.size() != 1) {
    return false;
  }

  if (!is_const_value(analyzer.Simplify(combiner->identity_element[0], 3), 0)) {
    return false;
  }

  PrimExpr combiner_result = analyzer.Simplify(combiner->result[0], 3);

  return tir::ExprDeepEqual()(combiner_result, combiner->lhs[0] + combiner->rhs[0]) ||
        tir::ExprDeepEqual()(combiner_result, combiner->rhs[0] + combiner->lhs[0]);
}

bool CanFactorZeroFromCombiner(const CommReducer& combiner, int value_index,
                               const Map<Var, Range>& vranges) {
  arith::Analyzer analyzer;
  analyzer.Bind(vranges);
  if (!is_const_value(analyzer.Simplify(combiner->identity_element[value_index], 3), 0)) {
    return false;
  }

  PrimExpr zero = make_zero(combiner->result[value_index].dtype());
  PrimExpr in = Substitute(combiner->result[value_index],
                           {{combiner->lhs[value_index], zero}, {combiner->rhs[value_index], zero}});
  in = analyzer.Simplify(in, 3);

  return is_const_value(in, 0);
}

struct NonzeronessConditionResult {
  PrimExpr cond;
  PrimExpr value;

  PrimExpr to_expr() const {
    return SelectNode::make(cond, value, make_zero(value.dtype()));
  }

  friend std::ostream& operator<<(std::ostream& os, const NonzeronessConditionResult& r) {
    return os << r.to_expr();
  }
};

// The implementation of NonzeronessCondition
// transform expression to cond ? value : 0
class NonzeronessConditionFunctor
    : public ExprFunctor<NonzeronessConditionResult(const PrimExpr&)> {
 public:
  NonzeronessConditionResult NonzeronessCondition(const PrimExpr& e) {
    if (e.dtype().is_bool()) {
      // Boolean expressions are non-zero whenever they are true themselves
      return {e, const_true()};
    } else {
      return VisitExpr(e);
    }
  }

  // Most of the cases are implemented using helpers below
  result_type VisitExpr_(const VarNode* op) final { return Default_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const IntImmNode* op) final { return Const_(op); }
  result_type VisitExpr_(const FloatImmNode* op) final { return Const_(op); }
  result_type VisitExpr_(const StringImmNode* op) final { return Default_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const AddNode* op) final { return BinOpAddLike_(op); }
  result_type VisitExpr_(const SubNode* op) final { return BinOpAddLike_(op); }
  result_type VisitExpr_(const MulNode* op) final { return BinOpMulLike_(op); }
  result_type VisitExpr_(const DivNode* op) final { return BinOpDivLike_(op); }
  result_type VisitExpr_(const ModNode* op) final { return BinOpDivLike_(op); }
  result_type VisitExpr_(const FloorDivNode* op) final { return BinOpDivLike_(op); }
  result_type VisitExpr_(const FloorModNode* op) final { return BinOpDivLike_(op); }
  result_type VisitExpr_(const MinNode* op) final { return BinOpAddLike_(op); }
  result_type VisitExpr_(const MaxNode* op) final { return BinOpAddLike_(op); }

  result_type VisitExpr_(const CastNode* op) final {
    auto nz_a = NonzeronessCondition(op->value);

    if (nz_a.value.same_as(op->value)) {
      return {nz_a.cond, GetRef<PrimExpr>(op)};
    } else {
      return {nz_a.cond, CastNode::make(op->dtype, nz_a.value)};
    }
  }

  result_type VisitExpr_(const SelectNode* op) final {
    PrimExpr cond = op->condition, true_val = op->true_value, false_val = op->false_value;
    auto nz_a = NonzeronessCondition(true_val);
    auto nz_b = NonzeronessCondition(false_val);

    // If the false part is zero, we can get rid of the select
    if (is_const_value(nz_b.value, 0)) {
      PrimExpr new_cond = analyzer_.Simplify(nz_a.cond && cond, 3);
      return {new_cond, nz_a.value};
    }

    // If the true part is zero, we can also get rid of the select
    if (is_const_value(nz_a.value, 0)) {
      PrimExpr new_cond = analyzer_.Simplify(nz_b.cond && !cond, 3);
      return {new_cond, nz_b.value};
    }

    // Otherwise we retain the select and combine the conditions into this
    PrimExpr new_cond = analyzer_.Simplify((cond && nz_a.cond) || (!cond && nz_b.cond), 3);
    if (nz_a.value.same_as(true_val) && nz_b.value.same_as(false_val)) {
      return {new_cond, GetRef<PrimExpr>(op)};
    } else {
      return {new_cond, SelectNode::make(cond, nz_a.value, nz_b.value)};
    }
  }

  result_type VisitExpr_(const CallNode* op) final {
    if (op->name == intrinsic::tvm_if_then_else) {
      PrimExpr cond = op->args[0], true_val = op->args[1], false_val = op->args[2];
      auto nz_a = NonzeronessCondition(true_val);
      auto nz_b = NonzeronessCondition(false_val);

      // We don't have as much freedom here as in the select case
      // since the `if` must be preserved in any case
      PrimExpr new_cond = analyzer_.Simplify((cond && nz_a.cond) || (!cond && nz_b.cond), 3);
      if (nz_a.value.same_as(true_val) && nz_b.value.same_as(false_val)) {
        return {new_cond, GetRef<PrimExpr>(op)};
      } else {
        return {new_cond, if_then_else(cond, nz_a.value, nz_b.value)};
      }
    } else {
      return Default_(GetRef<PrimExpr>(op));
    }
  }

  NonzeronessConditionResult Default_(const PrimExpr& e) {
    // This is always correct, so it's the default
    return {const_true(), e};
  }

  template <class TNode>
  NonzeronessConditionResult Const_(const TNode* op) {
    if (op->value == 0) {
      return {const_false(), GetRef<PrimExpr>(op)};
    } else {
      return {const_true(), GetRef<PrimExpr>(op)};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpAddLike_(const TNode* op) {
    auto nz_a = NonzeronessCondition(op->a);
    auto nz_b = NonzeronessCondition(op->b);

    // For addition and similar ops the result may be nonzero if either of the arguments is
    // nonzero, so we combine the conditions with Or.
    if (tir::ExprDeepEqual()(nz_a.cond, nz_b.cond)) {
      // If the conditions are the same, we don't need Or
      if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
        return {nz_a.cond, GetRef<PrimExpr>(op)};
      } else {
        return {nz_a.cond, TNode::make(nz_a.value, nz_b.value)};
      }
    } else {
      // Otherwise use Or
      PrimExpr new_cond = analyzer_.Simplify(nz_a.cond || nz_b.cond, 3);
      // A little optimization: if the combined condition is the same as one of the inner
      // conditions, we don't need to guard the inner value with a select, otherwise
      // we create a select in the `to_expr` call.
      PrimExpr new_a = tir::ExprDeepEqual()(nz_a.cond, new_cond) ? nz_a.value : nz_a.to_expr();
      PrimExpr new_b = tir::ExprDeepEqual()(nz_b.cond, new_cond) ? nz_b.value : nz_b.to_expr();
      PrimExpr new_expr = TNode::make(new_a, new_b);
      return {new_cond, new_expr};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpMulLike_(const TNode* op) {
    auto nz_a = NonzeronessCondition(op->a);
    auto nz_b = NonzeronessCondition(op->b);

    // For multiplication and similar ops the result may be nonzero if
    // both the arguments are nonzero, so we combine with And.
    PrimExpr new_cond = analyzer_.Simplify(nz_a.cond && nz_b.cond, 3);

    if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
      return {new_cond, GetRef<PrimExpr>(op)};
    } else {
      return {new_cond, TNode::make(nz_a.value, nz_b.value)};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpDivLike_(const TNode* op) {
    auto nz_a = NonzeronessCondition(op->a);

    // For Div we simply use the condition of the numerator.

    if (nz_a.value.same_as(op->a)) {
      return {nz_a.cond, GetRef<PrimExpr>(op)};
    } else {
      return {nz_a.cond, TNode::make(nz_a.value, op->b)};
    }
  }
 private:
  arith::Analyzer analyzer_;
};

inline NonzeronessConditionResult NonzeronessCondition(const PrimExpr& expr) {
  return NonzeronessConditionFunctor().NonzeronessCondition(expr);
}

Array<Var> IterVarsToVars(const Array<IterVar>& itervars) {
  Array<Var> res;
  for (const IterVar& v : itervars) {
    res.push_back(v->var);
  }
  return res;
}

struct FactorOutAtomicFormulasResult {
  std::vector<PrimExpr> atomic_formulas;
  PrimExpr rest;

  PrimExpr to_expr() const {
    PrimExpr res = rest;
    for (const PrimExpr& e : atomic_formulas) {
      res = AndNode::make(e, res);
    }
    return res;
  }

  Array<PrimExpr> to_array() const {
    Array<PrimExpr> res = atomic_formulas;
    res.push_back(rest);
    return res;
  }
};

// The implementation of FactorOutAtomicFormulas
class FactorOutAtomicFormulasFunctor
    : public ExprFunctor<FactorOutAtomicFormulasResult(const PrimExpr&)> {
 public:
  result_type Atomic_(const PrimExpr& e) {
    // For atomic expressions the result is the expr itself with True as the residual
    return {{e}, make_const(e.dtype(), 1)};
  }

  // This is basically the list of expression kinds that are considered atomic
  result_type VisitExpr_(const VarNode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const CallNode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const IntImmNode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const EQNode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const NENode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const LENode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const LTNode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const GENode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }
  result_type VisitExpr_(const GTNode* op) final { return Atomic_(GetRef<PrimExpr>(op)); }

  result_type VisitExpr_(const SelectNode* op) final {
    // Select can be rewritten through other logical ops
    PrimExpr expr = (op->condition && op->true_value) || (!op->condition && op->false_value);
    return VisitExpr(expr);
  }

  result_type VisitExpr_(const NotNode* op) final {
    // Not should be moved down
    if (const OrNode* or_expr = op->a.as<OrNode>()) {
      PrimExpr expr = !or_expr->a && !or_expr->b;
      return VisitExpr(expr);
    } else if (const AndNode* and_expr = op->a.as<AndNode>()) {
      PrimExpr expr = !and_expr->a || !and_expr->b;
      return VisitExpr(expr);
    } if (const SelectNode* sel_expr = op->a.as<SelectNode>()) {
      PrimExpr expr = ((!sel_expr->condition || !sel_expr->true_value) &&
                       (sel_expr->condition || !sel_expr->false_value));
      return VisitExpr(expr);
    }
    return Atomic_(GetRef<PrimExpr>(op));
  }

  result_type VisitExpr_(const AndNode* op) final {
    auto res_a = VisitExpr(op->a);
    auto res_b = VisitExpr(op->b);

    // For the And case we return the union of the sets of atomic formulas
    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> res_set;
    res_set.reserve(res_a.atomic_formulas.size() + res_b.atomic_formulas.size());
    std::copy(res_a.atomic_formulas.begin(),
              res_a.atomic_formulas.end(),
              std::inserter(res_set, res_set.end()));
    std::copy(res_b.atomic_formulas.begin(),
              res_b.atomic_formulas.end(),
              std::inserter(res_set, res_set.end()));

    std::vector<PrimExpr> res {res_set.begin(), res_set.end()};

    // And the residuals are combined with &&
    return {res, res_a.rest && res_b.rest};
  }

  result_type VisitExpr_(const MulNode* op) final {
    // Since we work with bools, for multiplication we do the same thing as for And
    PrimExpr e_and = op->a && op->b;
    return VisitExpr(e_and);
  }

  result_type VisitExpr_(const OrNode* op) final {
    auto res_a = VisitExpr(op->a);
    auto res_b = VisitExpr(op->b);

    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> res_a_set {
        res_a.atomic_formulas.begin(), res_a.atomic_formulas.end() };
    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> res_b_set {
        res_b.atomic_formulas.begin(), res_b.atomic_formulas.end() };

    // For the Or case we intersect the sets of atomic formulas
    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> res_set;
    res_set.reserve(std::min(res_a.atomic_formulas.size(), res_b.atomic_formulas.size()));
    for (const auto& res_b_formula : res_b_set) {
      if (res_a_set.count(res_b_formula)) {
        res_set.insert(res_b_formula);
      }
    }

    // Computing the residual is more complex: we have to compute the sets of atomic formulas
    // which are left behind, and then combine them with the residuals into the new residual.
    std::vector<PrimExpr> new_cond_a;
    new_cond_a.reserve(res_a.atomic_formulas.size() - res_set.size());
    for (const auto& formula : res_a_set) {
      if (!res_set.count(formula)) new_cond_a.emplace_back(formula);
    }

    std::vector<PrimExpr> new_cond_b;
    new_cond_b.reserve(res_b.atomic_formulas.size() - res_set.size());
    for (const auto& formula : res_b_set) {
      if (!res_set.count(formula)) new_cond_b.emplace_back(formula);
    }

    res_a.atomic_formulas = std::move(new_cond_a);
    res_b.atomic_formulas = std::move(new_cond_b);

    PrimExpr new_rest = res_a.to_expr() || res_b.to_expr();
    std::vector<PrimExpr> res {res_set.begin(), res_set.end()};

    return {res, new_rest};
  }
};

// Transform the given formula into a conjunction of atomic formulas (represented as an array)
// and a non-atomic residual. Atomic formulas are consts, calls, variables and comparisons (a <= b,
// etc), i.e. formulas which are not logical operators (||, &&, !) on the top level.
FactorOutAtomicFormulasResult FactorOutAtomicFormulas(const PrimExpr& e) {
  CHECK(e.dtype().is_bool());
  return FactorOutAtomicFormulasFunctor().VisitExpr(e);
}



// Combine all expressions from the container using &&.
template <class container>
PrimExpr All(const container& c) {
  PrimExpr res;
  for (const auto& e : c) {
    if (res.get()) {
      res = res && e;
    } else {
      res = e;
    }
  }
  if (res.get()) {
    return res;
  } else {
    return const_true();
  }
}

struct EliminateDivModResult {
  PrimExpr expr;
  Map<Var, PrimExpr> substitution;
  Array<Var> new_variables;
  Array<PrimExpr> conditions;
  Map<Var, Range> ranges;
};

// TODO: This is a duplicate of the same enum from canonical_simplify.cc, they should be merged
enum DivMode {
  /*! \brief Truncated division. */
  kTruncDiv,
  /*! \brief Floor division. */
  kFloorDiv
};

inline PrimExpr ModImpl(PrimExpr a, PrimExpr b, DivMode mode) {
  if (mode == kTruncDiv) {
    return truncmod(a, b);
  } else {
    CHECK_EQ(mode, kFloorDiv);
    return floormod(a, b);
  }
}

inline PrimExpr DivImpl(PrimExpr a, PrimExpr b, DivMode mode) {
  if (mode == kTruncDiv) {
    return truncdiv(a, b);
  } else {
    CHECK_EQ(mode, kFloorDiv);
    return floordiv(a, b);
  }
}

class EliminateDivModMutator : public ExprMutator {
 public:
  Map<Var, PrimExpr> substitution;
  Array<Var> new_variables;
  Array<PrimExpr> conditions;
  Map<Var, Range> ranges;

  explicit EliminateDivModMutator(Map<Var, Range> ranges)
      : ranges(std::move(ranges)) {}

  virtual PrimExpr VisitExpr_(const DivNode* op) {
    const IntImmNode* imm = op->b.as<IntImmNode>();
    if (imm && imm->value != 0) {
      if (imm->value < 0) {
        // x / -c == -(x/c) for truncated division
        return make_zero(op->dtype) - VisitExpr(truncdiv(op->a, make_const(op->dtype, -imm->value)));
      }

      // Try to find the already existing variables for this expression
      auto it = expr_to_vars_.find(std::make_tuple(kTruncDiv, op->a, imm->value));
      if (it != expr_to_vars_.end()) {
        return it->second.first;
      }

      // Otherwise recursively mutate the left hand side, and create new variables
      PrimExpr mutated_a = VisitExpr(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value, kTruncDiv)) {
        return var_pair_opt.value().first;
      } else {
        return truncdiv(mutated_a, op->b);
      }
    }

    return truncdiv(VisitExpr(op->a), VisitExpr(op->b));
  }

  virtual PrimExpr VisitExpr_(const ModNode* op) {
    const IntImmNode* imm = op->b.as<IntImmNode>();
    if (imm && imm->value != 0) {
      if (imm->value < 0) {
        // x % -c == x % c for truncated division
        return VisitExpr(truncmod(op->a, make_const(op->dtype, -imm->value)));
      }

      // Try to find the already existing variables for this expression
      auto it = expr_to_vars_.find(std::make_tuple(kTruncDiv, op->a, imm->value));
      if (it != expr_to_vars_.end()) {
        return it->second.second;
      }

      // Otherwise recursively mutate the left hand side, and create new variables
      PrimExpr mutated_a = VisitExpr(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value, kTruncDiv)) {
        return var_pair_opt.value().second;
      } else {
        return truncmod(mutated_a, op->b);
      }
    }

    return truncmod(VisitExpr(op->a), VisitExpr(op->b));
  }

  virtual PrimExpr VisitExpr_(const FloorDivNode* op) {
    const IntImmNode* imm = op->b.as<IntImmNode>();
    if (imm && imm->value != 0) {
      if (imm->value < 0) {
        // x / -c == (-x) / c for flooring division
        return VisitExpr(floordiv(make_zero(op->dtype) - op->a, make_const(op->dtype, -imm->value)));
      }

      // Try to find the already existing variables for this expression
      auto it = expr_to_vars_.find(std::make_tuple(kFloorDiv, op->a, imm->value));
      if (it != expr_to_vars_.end()) {
        return it->second.first;
      }

      // Otherwise recursively mutate the left hand side, and create new variables
      PrimExpr mutated_a = VisitExpr(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value, kFloorDiv)) {
        return var_pair_opt.value().first;
      } else {
        return floordiv(mutated_a, op->b);
      }
    }

    return floordiv(VisitExpr(op->a), VisitExpr(op->b));
  }

  virtual PrimExpr VisitExpr_(const FloorModNode* op) {
    const IntImmNode* imm = op->b.as<IntImmNode>();
    if (imm && imm->value != 0) {
      if (imm->value < 0) {
        // x % -c == -(-x % c) for flooring division
        return VisitExpr(make_zero(op->dtype) - floormod(make_zero(op->dtype) - op->a,
                                                         make_const(op->dtype, -imm->value)));
      }

      // Try to find the already existing variables for this expression
      auto it = expr_to_vars_.find(std::make_tuple(kFloorDiv, op->a, imm->value));
      if (it != expr_to_vars_.end()) {
        return it->second.second;
      }

      // Otherwise recursively mutate the left hand side, and create new variables
      PrimExpr mutated_a = VisitExpr(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value, kFloorDiv)) {
        return var_pair_opt.value().second;
      } else {
        return floormod(mutated_a, op->b);
      }
    }

    return floormod(VisitExpr(op->a), VisitExpr(op->b));
  }

 private:
  dmlc::optional<std::pair<Var, Var>> AddNewVarPair(const PrimExpr& e,
                                                    const PrimExpr& mut,
                                                    int64_t val,
                                                    DivMode mode) {
    using tresult = dmlc::optional<std::pair<Var, Var>>;

    // Try to find the variables using the mutated expressions
    if (!e.same_as(mut)) {
      auto it = expr_to_vars_.find(std::make_tuple(mode, mut, val));
      if (it != expr_to_vars_.end()) {
        return tresult(it->second);
      }
    }

    PrimExpr val_e = make_const(e.dtype(), val);
    idx_ += 1;

    // Convert `ranges` to IntSets
    std::unordered_map<const VarNode*, IntSet> var_intsets;
    for (const auto& p : ranges) {
      var_intsets[p.first.get()] = IntSet::range(p.second);
    }

    // Infer ranges for the expressions we want to replace with variables
    Range div_range = EvalSet(DivImpl(mut, val_e, mode), var_intsets).cover_range(Range());
    Range mod_range = EvalSet(ModImpl(mut, val_e, mode), var_intsets).cover_range(Range());

    // We don't want to add unbounded variables
    if (!div_range.get() || !mod_range.get()) {
      LOG(WARNING) << "EliminateDivMod: won't eliminate " << DivImpl(e, val_e, mode)
                   << "  because its bounds cannot be inferred";
      return tresult();
    }
    if (!mod_range.get()) {
      LOG(WARNING) << "EliminateDivMod: won't eliminate " << ModImpl(e, val_e, mode)
                   << "  because its bounds cannot be inferred";
      return tresult();
    }

    // Create new variables for the expressions
    auto div = Var((mode == kTruncDiv ? "tdiv" : "fdiv") + std::to_string(idx_), e.dtype());
    auto mod = Var((mode == kTruncDiv ? "tmod" : "fmod") + std::to_string(idx_), e.dtype());

    new_variables.push_back(div);
    new_variables.push_back(mod);

    // Note that we have to perform substitution to mut because mut may contain new variables
    substitution.Set(div, DivImpl(Substitute(mut, substitution), val_e, mode));
    substitution.Set(mod, ModImpl(Substitute(mut, substitution), val_e, mode));

    ranges.Set(div, div_range);
    ranges.Set(mod, mod_range);

    // This additional condition works as a definition for the new variables
    conditions.push_back(mut == div*val_e + mod);

    if (!analyzer_.CanProve(mod_range->extent <= val_e)) {
      // Since we use the C/C++ definition of mod, there may be multiple values of `mod`
      // satisfying the added condition if the expr `e` may change its sign, so we
      // have to add another condition.
      LOG(WARNING) << "EliminateDivMod: cannot fully eliminate div or mod because "
                   << ModImpl(e, val_e, mode) << "  probably may change its sign";
      conditions.push_back(SelectNode::make(e >= 0, mod >= 0, mod <= 0));
    }

    auto p = std::make_pair(div, mod);
    expr_to_vars_[std::make_tuple(mode, e, val)] = p;
    if (!e.same_as(mut)) {
      expr_to_vars_[std::make_tuple(mode, mut, val)] = p;
    }
    return tresult(p);
  }

  class TupleEqual_ {
   public:
    bool operator()(const std::tuple<DivMode, PrimExpr, int64_t>& lhs,
                    const std::tuple<DivMode, PrimExpr, int64_t>& rhs) const {
      // TODO: ExprDeepEqual or StruturalEqual?
      return std::get<0>(lhs) == std::get<0>(rhs) &&
          tir::ExprDeepEqual()(std::get<1>(lhs), std::get<1>(rhs)) &&
          std::get<2>(lhs) == std::get<2>(rhs);
    }
  };

  class TupleHasher_ {
   public:
    size_t operator()(const std::tuple<DivMode, PrimExpr, int64_t>& key) const {
      return ((std::hash<int>()(std::get<0>(key))
               ^ (StructuralHash()(std::get<1>(key)) << 1)) >> 1)
               ^ (std::hash<int64_t>()(std::get<2>(key)) << 1);
    }
  };

  // A counter for naming new variables
  int idx_{0};
  // A map from pairs of exprs and numbers (e, n) to pairs of new vars (div, mod)
  // such that `div = e / n` and `mod = e % n`
  std::unordered_map<std::tuple<DivMode, PrimExpr, int64_t>,
                     std::pair<Var, Var>,
                     TupleHasher_,
                     TupleEqual_> expr_to_vars_;
  arith::Analyzer analyzer_;
};

// Replace every subexpr of the form e/const and e % const with a new variable.
// Syntactically equal expressions will be mapped to the same variable.
EliminateDivModResult EliminateDivMod(const PrimExpr& expr, Map<Var, Range> ranges) {
  EliminateDivModResult res;
  EliminateDivModMutator mutator(ranges);
  res.expr = mutator(expr);
  res.conditions = std::move(mutator.conditions);
  res.new_variables = std::move(mutator.new_variables);
  res.substitution = std::move(mutator.substitution);
  res.ranges = std::move(mutator.ranges);
  return res;
}

arith::IntConstraintsTransform EliminateDivModFromDomainConditions(const arith::IntConstraints& domain) {
  auto elim_res = EliminateDivMod(All(domain->relations), domain->ranges);

  Map<Var, Range> new_vranges = elim_res.ranges;
  Array<Var> new_axis = Concat(domain->variables, elim_res.new_variables);
  PrimExpr new_cond = elim_res.expr && All(elim_res.conditions);

  arith::IntConstraints new_domain(new_axis,
                                   new_vranges,
                                   FactorOutAtomicFormulas(new_cond).to_array());

  Map<Var, PrimExpr> src_to_dst;
  Map<Var, PrimExpr> dst_to_src = elim_res.substitution;
  for (const Var& v : domain->variables) {
    src_to_dst.Set(v, v);
    dst_to_src.Set(v, v);
  }

  return arith::IntConstraintsTransform(domain, new_domain, src_to_dst, dst_to_src);
}

// Simplify an iteration domain.
inline arith::IntConstraintsTransform IdentityTransformation(const arith::IntConstraints& domain) {
  Map<Var, PrimExpr> identity_map;
  for (const Var& v : domain->variables) {
    identity_map.Set(v, v);
  }
  return arith::IntConstraintsTransform(domain, domain, identity_map, identity_map);
}

arith::IntConstraintsTransform SimplifyIterDomain(const arith::IntConstraints& domain,
                                                  bool eliminate_div_mod) {
  arith::IntConstraintsTransform transf = IdentityTransformation(domain);

  if (eliminate_div_mod) {
    transf = transf + EliminateDivModFromDomainConditions(transf->dst);
  }

  // TODO(sgrechanik-h): Repeating the following steps has a positive effect, however we probably
  // should find a better terminating criterion (like stop when the domain volume stops decreasing)
  // Also 2 steps seems to be slightly better than 3
  for (size_t i = 0; i < 2; ++i) {
    arith::IntConstraintsTransform tr = arith::SolveLinearEquations(transf->dst);
    transf = transf + tr;
    // TODO(sgrechanik-h): This helps for some artificial examples, however I'm not sure about
    // enabling it in general. The problem it solves is propagating equalities of outer vars.
    // tr = AddOuterVariablesIntoDomain(transf->dst);
    tr = arith::SolveInequalitiesDeskewRange(transf->dst);
    transf = transf + tr;
  }

  return transf;
}

// Given a map from vars to ranges create an array of itervars
Array<IterVar> IterVarsFromMap(const Array<Var>& vars, const Map<Var, Range>& vranges,
                               IterVarType iter_type = kDataPar, std::string thread_tag = "") {
  Array<IterVar> res;
  for (const Var& v : vars) {
    CHECK(vranges.count(v)) << "A range for the variable " << v
                            << " was not provided in map " << vranges;
    res.push_back(IterVarNode::make(vranges[v], v, iter_type, thread_tag));
  }
  return res;
}

// Use the condition of a reduction op to simplify its domain (axis)
PrimExpr SimplifyReductionDomain(const PrimExpr& expr, const Map<Var, Range>& outer_vranges) {
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    Array<Var> vars = IterVarsToVars(red->axis);
    Map<Var, Range> vranges = Merge(outer_vranges, IterVarsToMap(red->axis));
    Array<PrimExpr> relations = FactorOutAtomicFormulas(red->condition).to_array();

    arith::IntConstraints domain(vars, vranges, relations);
    auto res = SimplifyIterDomain(domain);

    Array<PrimExpr> new_source;
    for (const PrimExpr& src : red->source) {
      new_source.push_back(Substitute(src, res->src_to_dst));
    }

    Array<IterVar> new_axis =
        IterVarsFromMap(res->dst->variables, res->dst->ranges, kCommReduce);

    // Perform simplification mainly to remove a possibly empty reduction.
    arith::Analyzer analyzer;
    return analyzer.Simplify(
        ReduceNode::make(red->combiner,
                         new_source,
                         new_axis,
                         All(res->dst->relations),
                         red->value_index), 3);
  } else {
    return expr;
  }
}

// Extract from cond an implication of cond not containing vars
std::pair<PrimExpr, PrimExpr> ImplicationNotContainingVars(
    const PrimExpr& cond, const std::unordered_set<const VarNode*>& vars) {
  CHECK(cond.dtype().is_bool()) << "The type of cond must be bool";
  // TODO(sgrechanik-h): not
  if (const AndNode* op = cond.as<AndNode>()) {
    auto pair_a = ImplicationNotContainingVars(op->a, vars);
    auto pair_b = ImplicationNotContainingVars(op->b, vars);
    return {pair_a.first && pair_b.first,
            pair_a.second && pair_b.second};
  } else if (const OrNode* op = cond.as<OrNode>()) {
    auto pair_a = ImplicationNotContainingVars(op->a, vars);
    auto pair_b = ImplicationNotContainingVars(op->b, vars);
    return {pair_a.first || pair_b.first,
            (pair_a.first || pair_b.second) &&
            (pair_b.first || pair_a.second) &&
            (pair_a.second || pair_b.second)};
  } else if (!tir::ExprUseVar(cond, [&vars](const VarNode* var) { return vars.count(var); })) {
    return {cond, const_true()};
  } else {
    return {const_true(), cond};
  }
}

// Factor conditions out of a reduction by applying Fourier-Motzkin elimination and moving out
// (in)equalities which do not depend on the reduction variables.
std::pair<PrimExpr, PrimExpr> LiftConditionsThroughReduction(const PrimExpr& cond,
                                                             const Array<IterVar>& red_axis,
                                                             const Array<IterVar>& outer_axis) {
  // Factor out atomics so that we can consider this as a system of inequalities
  auto factoratomic_res = FactorOutAtomicFormulas(cond);
  Array<PrimExpr> atomics = factoratomic_res.atomic_formulas;
  const PrimExpr& rest = factoratomic_res.rest;

  Array<Var> allvars;
  for (const IterVar& v : red_axis) {
    allvars.push_back(v->var);
  }
  for (const IterVar& v : outer_axis) {
    allvars.push_back(v->var);
  }

  auto vranges = Merge(IterVarsToMap(red_axis), IterVarsToMap(outer_axis));
  // start from reduction vars, so that input vars don't depend on them
  arith::IntConstraints ineq_to_solve(allvars, vranges, atomics);
  auto res_ineq = arith::SolveLinearInequalities(ineq_to_solve);
  atomics = arith::as_conditions(res_ineq.first, res_ineq.second);

  // Append the rest part
  PrimExpr rewritten_cond = All(atomics) && rest;

  std::unordered_set<const VarNode*> vset;
  for (const IterVar& v : red_axis) {
    vset.insert(v->var.get());
  }

  // The outer (first) condition does not contain reduction vars,
  // the inner (second) condition is everything else
  auto res = ImplicationNotContainingVars(rewritten_cond, vset);
  LOG(INFO) << "LiftConditionsThroughReduction first = " << res.first;
  LOG(INFO) << "LiftConditionsThroughReduction second = " << res.second;
  return res;
}

// Convert an array of itervars to an array of inequalities
Array<PrimExpr> IterVarsToInequalities(const Array<IterVar>& itervars) {
  Array<PrimExpr> res;
  for (const IterVar& v : itervars) {
    res.push_back(GENode::make(v->var, v->dom->min));
    res.push_back(LTNode::make(v->var, v->dom->min + v->dom->extent));
  }
  return res;
}

class RemoveRedundantInequalitiesMutator : public ExprMutator {
 public:
  explicit RemoveRedundantInequalitiesMutator(Array<PrimExpr> known) {
    for (const PrimExpr& cond : known) {
      known_.push_back(analyzer_.Simplify(cond, 3));
    }
  }

  virtual PrimExpr VisitExpr_(const SelectNode* op) {
    bool has_side_effect = HasSideEffect(GetRef<PrimExpr>(op));
    PrimExpr new_cond = analyzer_.Simplify(VisitExpr(op->condition), 3);
    if (is_one(new_cond) && !has_side_effect) {
      return VisitExpr(op->true_value);
    } else if (is_zero(new_cond) && !has_side_effect) {
      return VisitExpr(op->false_value);
    } else {
      Array<PrimExpr> new_known = known_;
      for (const PrimExpr& atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
        new_known.push_back(atomic);
      }
      RemoveRedundantInequalitiesMutator new_mutator(new_known);
      // Note that we mutate only the true value with the new mutator
      // TODO(sgrechanik-h): Update known conditions for the false value as well
      return SelectNode::make(new_cond, new_mutator(op->true_value), VisitExpr(op->false_value));
    }
  }

  virtual PrimExpr VisitExpr_(const CallNode* op) {
    if (op->name == intrinsic::tvm_if_then_else) {
      PrimExpr new_cond = analyzer_.Simplify(VisitExpr(op->args[0]), 3);
      if (is_one(new_cond)) {
        return VisitExpr(op->args[1]);
      } else if (is_zero(new_cond)) {
        return VisitExpr(op->args[2]);
      } else {
        Array<PrimExpr> new_known = known_;
        for (const PrimExpr& atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
          new_known.push_back(atomic);
        }
        RemoveRedundantInequalitiesMutator new_mutator(new_known);
        // Note that we mutate only the true value with the new mutator
        // TODO(sgrechanik-h): Update known conditions for the false value as well
        return if_then_else(new_cond, new_mutator(op->args[1]), VisitExpr(op->args[2]));
      }
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  virtual PrimExpr VisitExpr_(const ReduceNode* op) {
    Array<PrimExpr> known_with_axes = known_;
    for (const PrimExpr& axis_cond : IterVarsToInequalities(op->axis)) {
      known_with_axes.push_back(axis_cond);
    }
    RemoveRedundantInequalitiesMutator mutator_with_axes(known_with_axes);

    PrimExpr new_cond = mutator_with_axes(op->condition);

    Array<PrimExpr> new_known = known_with_axes;
    for (const PrimExpr& atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
      new_known.push_back(atomic);
    }
    RemoveRedundantInequalitiesMutator new_mutator(new_known);

    Array<PrimExpr> new_source;
    for (const PrimExpr& src : op->source) {
      new_source.push_back(new_mutator(src));
    }

    return ReduceNode::make(op->combiner, new_source, op->axis, new_cond, op->value_index);
  }

  virtual PrimExpr VisitExpr_(const EQNode* op) { return MutateAtomic_(GetRef<PrimExpr>(op)); }
  virtual PrimExpr VisitExpr_(const NENode* op) { return MutateAtomic_(GetRef<PrimExpr>(op)); }
  virtual PrimExpr VisitExpr_(const LTNode* op) { return MutateAtomic_(GetRef<PrimExpr>(op)); }
  virtual PrimExpr VisitExpr_(const LENode* op) { return MutateAtomic_(GetRef<PrimExpr>(op)); }
  virtual PrimExpr VisitExpr_(const GTNode* op) { return MutateAtomic_(GetRef<PrimExpr>(op)); }
  virtual PrimExpr VisitExpr_(const GENode* op) { return MutateAtomic_(GetRef<PrimExpr>(op)); }

  virtual PrimExpr VisitExpr_(const AndNode* op) {
    return VisitExpr(op->a) && VisitExpr(op->b);
  }

 private:
  PrimExpr MutateAtomic_(const PrimExpr& e) {
    PrimExpr simplified = analyzer_.Simplify(e, 3);
    for (const PrimExpr& other : known_) {
      if (ExprDeepEqual()(simplified, other)) {
        return const_true();
      }
    }
    return simplified;
  }

  Array<PrimExpr> known_;
  arith::Analyzer analyzer_;
};

// Propagate information from conditions and remove redundant inequalities
// TODO(sgrechanik-h): This should be merged into standard simplifiers
PrimExpr RemoveRedundantInequalities(const PrimExpr& expr, const Array<PrimExpr>& known) {
  return RemoveRedundantInequalitiesMutator(known)(expr);
}

// Extract the given expr under the given condition as a separate tensor if the volume of the
// extracted tensor will be less than the volume of the outer_axis
PrimExpr TrySimplifyCompute(const PrimExpr& expr, const PrimExpr& cond,
                            const Array<Var>& outer_axis,
                            const Map<Var, Range>& vranges) {
  // solve cond, e.g., (jac_i0 == i) && (jac_i1 == j)
  arith::IntConstraints domain_to_solve(outer_axis, vranges,
                                        FactorOutAtomicFormulas(cond).to_array());
  LOG(INFO) << "domain to solve " << domain_to_solve;
  auto res = SimplifyIterDomain(domain_to_solve);
  LOG(INFO) << "solved domain " << res;

  arith::Analyzer analyzer;
  analyzer.Bind(res->dst->ranges);
  PrimExpr new_expr = analyzer.Simplify(Substitute(expr, res->src_to_dst), 3);
  // TODO: This is mostly done to simplify if_then_else which is not known by the Halide simplifier
  new_expr = RemoveRedundantInequalities(new_expr, res->dst->relations);

  // Keep only those variables of the new vars which are used in the new_expr
  Array<Var> used_res_variables;
  for (const Var& var : res->dst->variables) {
    if (ExprUseVar(new_expr, var)) {
      CHECK(res->dst->ranges.count(var)) << "Range of " << var << " cannot be inferred.";
      used_res_variables.push_back(var);
    }
  }

  // If the expression does not use vars then it is probably better to keep it inlined
  if (used_res_variables.empty()) {
    // We can return the new_expr here instead of the old expr because it doesn't use variables
    // otherwise we would need to replace the new vars or create a let-expression
    return new_expr;
  }

  // If it's already a call to a tensor then it will probably be useless to further simplify it.
  if (const CallNode* call = new_expr.as<CallNode>()) {
    if (call->call_type == CallNode::CallType::Halide) {
      return expr;
    }
  }

  // Compute volumes before and after
  PrimExpr old_volume = make_const(DataType::Int(64), 1);
  for (const Var& var : outer_axis) {
    CHECK(vranges.count(var)) << "Range of " << var << " was not provided.";
    old_volume = old_volume * vranges[var]->extent;
  }

  PrimExpr new_volume = make_const(DataType::Int(64), 1);
  for (const Var& var : used_res_variables) {
    new_volume = new_volume * res->dst->ranges[var]->extent;
  }

  // if we can prove that the old volume is not greater than the new volume then
  // prefer the old expression.
  arith::Analyzer ana_vranges;
  ana_vranges.Bind(vranges);
  if (ana_vranges.CanProve(old_volume <= new_volume)) {
    return expr;
  }

  Tensor tensor =
      TensorFromExpr(new_expr, IterVarsFromMap(used_res_variables, res->dst->ranges),
                     "extracted_tensor");

  Array<PrimExpr> args;
  for (const Var& var : used_res_variables) {
    args.push_back(res->dst_to_src[var]);
  }

  return CallNode::make(expr.dtype(), tensor->op->name, args,
                        CallNode::CallType::Halide, tensor->op, tensor->value_index);
}

class FreeVarsVisitor : public StmtExprVisitor {
 public:
  std::vector<Var> free_array;
  std::unordered_set<const VarNode*> bound;
  std::unordered_set<const VarNode*> free;

  void VisitExpr_(const VarNode* op) final {
    if (!bound.count(op) && !free.count(op)) {
      free.insert(op);
      free_array.push_back(GetRef<Var>(op));
    }
  }

  void VisitStmt_(const LetStmtNode* op) final {
    bound.insert(op->var.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    bound.insert(op->loop_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LetNode* op) final {
    bound.insert(op->var.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ReduceNode* op) final {
    for (const auto& iv : op->axis) {
      bound.insert(iv->var.get());
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const StoreNode* op) final {
    VisitExpr(op->buffer_var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    VisitExpr(op->buffer_var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const FreeNode* op) final {
    VisitExpr(op->buffer_var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LoadNode* op) final {
    VisitExpr(op->buffer_var);
    StmtExprVisitor::VisitExpr_(op);
  }
};

class ReductionAsTensorAccessMutator : public ExprMutator {
 public:
  explicit ReductionAsTensorAccessMutator(const Array<Var>& outer_axis,
                                          Map<Var, Range> vranges,
                                          std::string name = "extracted_reduction")
      : outer_axis_(outer_axis), vranges_(std::move(vranges)), name_(std::move(name)) {}

  PrimExpr VisitExpr_(const ReduceNode* op) final {
    ReductionAsTensorAccessMutator new_mutator(Concat(IterVarsToVars(op->axis), outer_axis_),
                                         Merge(vranges_, IterVarsToMap(op->axis)),
                                         name_);

    Array<PrimExpr> new_source;
    for (const PrimExpr& src : op->source) {
      new_source.push_back(new_mutator(src));
    }

    PrimExpr new_reduce =
        ReduceNode::make(op->combiner, new_source, op->axis, op->condition, op->value_index);

    FreeVarsVisitor fv_visitor;
    fv_visitor(new_reduce);

    // Vars of the tensor we are going to create for this reduction
    Array<Var> vars;
    for (const Var& v : outer_axis_) {
      // We take variables from the outer_axis_ which are also present in the new reduction
      if (fv_visitor.free.count(v.get())) {
        vars.push_back(v);
      }
    }

    auto new_axis_vmap_pair = CloneIterVars(IterVarsFromMap(vars, vranges_));
    Array<IterVar> new_axis = new_axis_vmap_pair.first;
    arith::Analyzer analyzer;
    analyzer.Bind(IterVarsToMap(new_axis));
    new_reduce = analyzer.Simplify(Substitute(new_reduce, new_axis_vmap_pair.second), 3);

    Tensor tensor = TensorFromExpr(new_reduce, new_axis, name_, tag_, attrs_);

    Array<PrimExpr> args;
    for (const Var& v : vars) {
      args.push_back(v);
    }

    return CallNode::make(op->dtype, tensor->op->name, args,
                          CallNode::CallType::Halide, tensor->op, tensor->value_index);
  }

 private:
  Array<Var> outer_axis_;
  Map<Var, Range> vranges_;
  std::string name_;
  std::string tag_;
  Map<String, ObjectRef> attrs_;
};

// Extract reductions as separate tensors.
inline PrimExpr ReductionAsTensorAccess(const PrimExpr& expr,
                                        const Array<Var>& outer_axis,
                                        const Map<Var, Range>& vranges) {
  return ReductionAsTensorAccessMutator(outer_axis, vranges)(expr);
}

PrimExpr LiftReductions(const PrimExpr& expr,
                        const Array<Var>& outer_axis,
                        const Map<Var, Range>& vranges) {
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    Array<Var> new_outer_axis = Concat(IterVarsToVars(red->axis), outer_axis);
    Map<Var, Range> new_vranges = Merge(vranges, IterVarsToMap(red->axis));
    Array<PrimExpr> new_source;
    for (const PrimExpr& src : red->source) {
      new_source.push_back(ReductionAsTensorAccess(src, new_outer_axis, new_vranges));
    }
    PrimExpr new_condition = ReductionAsTensorAccess(red->condition, new_outer_axis, new_vranges);

    return ReduceNode::make(red->combiner, new_source, red->axis,
                            new_condition, red->value_index);
  } else {
    return ReductionAsTensorAccess(expr, outer_axis, vranges);
  }
}

PrimExpr OptimizeAndLiftNonzeronessConditionsImpl(const PrimExpr& expr_orig,
                                                  const Array<IterVar>& axis,
                                                  const Map<Var, Range>& vranges) {
  PrimExpr result;
  Map<Var, Range> combined_vranges = Merge(vranges, IterVarsToMap(axis));
  arith::Analyzer analyzer;
  analyzer.Bind(combined_vranges);

  // Simplify the original expression first, mostly to simplify combiners
  PrimExpr expr = analyzer.Simplify(expr_orig, 3);
  LOG(INFO) << "expr (after simplification) " << expr;

  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    // TODO(sgrechanik-h): There are some other operations which behave like sum
    bool is_sum = IsSumCombiner(red->combiner, vranges);
    if (is_sum || CanFactorZeroFromCombiner(red->combiner, red->value_index, vranges)) {
      PrimExpr new_red = expr;

      // Here we simplify the reduction
      {
        PrimExpr cond = red->condition;
        Array<PrimExpr> source = red->source;

        // If it is a summation then we can lift nonzeroness conditions from the source
        // and add them to the reduction conditions
        if (is_sum) {
          auto nz = NonzeronessCondition(red->source[red->value_index]);
          cond = nz.cond && cond;
          source.Set(0, nz.value);
        }

        new_red = ReduceNode::make(red->combiner, source, red->axis, cond, red->value_index);
        new_red = SimplifyReductionDomain(new_red, combined_vranges);
        red = new_red.as<ReduceNode>();

        // If the reduction disappears completely then transform the result as a non-reduction
        if (!red) {
          return OptimizeAndLiftNonzeronessConditionsImpl(new_red, axis, vranges);
        }
      }

      PrimExpr new_outer_cond, new_reduce_cond;
      Array<PrimExpr> new_source = red->source;

      // Partially lift conditions from the reduce condition
      std::tie(new_outer_cond, new_reduce_cond) =
          LiftConditionsThroughReduction(red->condition, red->axis, axis);

      // If it's not sum then we haven't yet lifted nonzeroness cond from the source
      if (!is_sum) {
        PrimExpr outer_nz_cond, nz_cond, nz_source;
        auto nz = NonzeronessCondition(red->source[red->value_index]);
        // Append conditions from the reduction
        nz_cond = new_reduce_cond && nz.cond;
        nz_source = nz.value;
        std::tie(outer_nz_cond, nz_cond) =
            LiftConditionsThroughReduction(nz_cond, red->axis, axis);
        new_outer_cond = new_outer_cond && outer_nz_cond;
        new_source.Set(red->value_index,
                       SelectNode::make(nz_cond, nz_source, make_zero(nz_source.dtype())));
      }

      PrimExpr new_reduce = ReduceNode::make(red->combiner, new_source, red->axis,
                                             new_reduce_cond, red->value_index);
      new_reduce =
          TrySimplifyCompute(new_reduce, new_outer_cond, IterVarsToVars(axis), combined_vranges);
      result = SelectNode::make(new_outer_cond, new_reduce, make_zero(new_reduce.dtype()));
    } else {
      return SimplifyReductionDomain(expr, combined_vranges);
    }
  } else {
    auto nz = NonzeronessCondition(expr);
    LOG(INFO) << nz.cond << " ? " << nz.value << " : 0";
    PrimExpr new_expr =
        TrySimplifyCompute(nz.value, nz.cond, IterVarsToVars(axis), combined_vranges);
    result = SelectNode::make(nz.cond, new_expr, make_zero(new_expr.dtype()));
  }

  // Note that RemoveRedundantInequalities can sometimes propagate equalities which
  // other simplifiers cannot, like (i % 3) == 0.
  Array<PrimExpr> axis_conds = IterVarsToInequalities(axis);
  result = RemoveRedundantInequalities(result, axis_conds);

  // Currently in TVM reductions are only allowed at the top level of compute,
  // we need to extract intermediate inlined reduction as a separate stage (tensor).
  // Sometimes TrySimplifyCompute doesn't perform lift / extraction,
  // so there may be some non-top reductions left, take care of them.
  result = analyzer.Simplify(LiftReductions(result, IterVarsToVars(axis), combined_vranges), 3);
  return result;
}

Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor, const Map<Var, Range>& vranges) {
  auto transform_func = [&vranges](const PrimExpr& expr, const Array<IterVar>& axis) {
    return OptimizeAndLiftNonzeronessConditionsImpl(expr, axis, vranges);
  };
  return TransformTensorBody(tensor, transform_func);
}

}  // namespace te
}  // namespace tvm

