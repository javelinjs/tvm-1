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
 * \file zero_elimination.cc
 */

#include <tvm/te/zero_elimination.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/autodiff.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include "operation/op_util.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <tvm/tir/ir_pass.h>
#include <tvm/arith/pattern.h>

namespace tvm {
namespace te {

// Merge two maps, prefer the right one on conflict
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

// Convert an array of itervars to an array of vars
Array<Var> IterVarsToVars(const Array<IterVar>& itervars) {
  Array<Var> res;
  for (const IterVar& v : itervars) {
    res.push_back(v->var);
  }
  return res;
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

PrimExpr SuperSimplify(PrimExpr e, const Map<Var, Range>& vranges = Map<Var, Range>()) {
  // For some reason no simplifier can detect that there is only one value of the variable
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (const auto& var_range : vranges) {
    if (is_const_int(var_range.second->extent, 1)) {
      vmap[var_range.first.get()] = var_range.second->min;
    }
  }
  if (!vmap.empty()) {
    e = tir::Substitute(e, vmap);
  }

  arith::Analyzer an;
  for (const auto& var_range : vranges) {
    an.Bind(var_range.first, var_range.second);
  }

  // According to my experiments two best simplifications orders were can->rw and rw->can->rw,
  // but rw->can->rw is better for a couple of cases.
  // Note that we should end with rw because it factors multipliers out.
  PrimExpr res = e;
  res = an.rewrite_simplify(res);
  res = an.canonical_simplify(res);
  res = an.rewrite_simplify(res);

  return res;
}

// Given a map from vars to ranges create an array of itervars
Array<IterVar> IterVarsFromMap(const Array<Var>& vars, const Map<Var, Range>& vranges,
                               IterVarType iter_type, std::string thread_tag) {
  Array<IterVar> res;
  for (const Var& v : vars) {
    CHECK(vranges.count(v)) << "A range for the variable " << v
      << " was not provided in map " << vranges;
    res.push_back(IterVarNode::make(vranges[v], v, iter_type, thread_tag));
  }
  return res;
}

// Return true if this combiner is just a sum.
bool IsSumCombiner(const CommReducer& combiner, const Map<Var, Range>& vranges) {
  if (combiner->result.size() != 1) {
    return false;
  }

  if (!is_const_value(SuperSimplify(combiner->identity_element[0], vranges), 0)) {
    return false;
  }

  PrimExpr combiner_result = SuperSimplify(combiner->result[0], vranges);

  return Equal(combiner_result, combiner->lhs[0] + combiner->rhs[0]) ||
          Equal(combiner_result, combiner->rhs[0] + combiner->lhs[0]);
}

// Return true if zero may be factored out of a reduction with this combiner.
bool CanFactorZeroFromCombiner(const CommReducer& combiner, int value_index,
                               const Map<Var, Range>& vranges) {
  if (!is_const_value(SuperSimplify(combiner->identity_element[value_index], vranges), 0)) {
    return false;
  }

  PrimExpr zero = make_zero(combiner->result[value_index].dtype());
  PrimExpr in = tir::Substitute(combiner->result[value_index],
                       {{combiner->lhs[value_index], zero},
                        {combiner->rhs[value_index], zero}});
  in = SuperSimplify(in, vranges);

  return is_const_value(in, 0);
}

Operation ComputeOpFromExprs(const Array<PrimExpr>& exprs, const Array<IterVar>& axis,
                             const std::string& name, const std::string& tag,
                             const Map<std::string, ObjectRef>& attrs,
                             bool clone_axis) {
  if (clone_axis) {
    Array<IterVar> new_axis = axis;
    Map<Var, PrimExpr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(axis);
    Array<PrimExpr> new_exprs;
    for (const PrimExpr& e : exprs) {
      new_exprs.push_back(tir::Substitute(CloneReduction(e), vmap));
    }
    return ComputeOpFromExprs(new_exprs, new_axis, name, tag, attrs, false);
  }

  Array<PrimExpr> new_exprs;

  // If this is a reduction then we have to replicate it
  if (const ReduceNode* red = exprs[0].as<ReduceNode>()) {
    for (size_t i = 0; i < red->source.size(); ++i) {
      PrimExpr ith_red = ReduceNode::make(red->combiner, red->source, red->axis, red->condition, i);
      new_exprs.push_back(ith_red);
    }
  } else {
    new_exprs = exprs;
  }

  return ComputeOpNode::make(name, tag, attrs, axis, new_exprs);
}

Tensor TensorFromExpr(const PrimExpr& expr, const Array<IterVar>& axis,
                      const std::string& name, const std::string& tag,
                      const Map<std::string, ObjectRef>& attrs,
                      bool clone_axis) {
  int new_value_index = 0;
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    new_value_index = red->value_index;
  }
  return ComputeOpFromExprs({expr}, axis, name, tag, attrs, clone_axis).output(new_value_index);
}

// Convert a variable map into a sorted vector of pairs. Sorting is done with deep expr comparison.
template <typename T>
std::vector<std::pair<Var, T>> VarMapToVectorOfPairs(const Map<Var, T>& varmap) {
  using tpair = std::pair<Var, T>;
  std::vector<tpair> res;
  for (const tpair& pair : varmap) {
    res.push_back(pair);
  }
  std::sort(res.begin(), res.end(),
            [](const tpair& l, const tpair& r) { return Compare(l.first, r.first) < 0; });
  return res;
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
      PrimExpr new_cond = SuperSimplify(nz_a.cond && cond);
      return {new_cond, nz_a.value};
    }

    // If the true part is zero, we can also get rid of the select
    if (is_const_value(nz_a.value, 0)) {
      PrimExpr new_cond = SuperSimplify(nz_b.cond && !cond);
      return {new_cond, nz_b.value};
    }

    // Otherwise we retain the select and combine the conditions into this
    PrimExpr new_cond = SuperSimplify((cond && nz_a.cond) || (!cond && nz_b.cond));
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
      PrimExpr new_cond = SuperSimplify((cond && nz_a.cond) || (!cond && nz_b.cond));
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

    if (Equal(nz_a.cond, nz_b.cond)) {
      // If the conditions are the same, we don't need Or
      if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
        return {nz_a.cond, GetRef<PrimExpr>(op)};
      } else {
        return {nz_a.cond, TNode::make(nz_a.value, nz_b.value)};
      }
    } else {
      // Otherwise use Or
      PrimExpr new_cond = SuperSimplify(nz_a.cond || nz_b.cond);
      // A little optimization: if the combined condition is the same as one of the inner
      // conditions, we don't need to guard the inner value with a select, otherwise
      // we create a select in the `to_expr` call.
      PrimExpr new_a = Equal(nz_a.cond, new_cond) ? nz_a.value : nz_a.to_expr();
      PrimExpr new_b = Equal(nz_b.cond, new_cond) ? nz_b.value : nz_b.to_expr();
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

    PrimExpr new_cond = SuperSimplify(nz_a.cond && nz_b.cond);

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
};

struct ExprLess {
  bool operator()(const PrimExpr& l, const PrimExpr& r) const {
    return Compare(l, r) < 0;
  }
};

struct ExprEq {
  bool operator()(const PrimExpr& l, const PrimExpr& r) const {
    return Compare(l, r) == 0;
  }
};

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

// Transform the given formula into a conjunction of atomic formulas (represented as an array)
// and a non-atomic residual. Atomic formulas are consts, calls, variables and comparisons (a <= b,
// etc), i.e. formulas which are not logical operators (||, &&, !) on the top level.
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
    std::vector<PrimExpr> res;
    res.reserve(res_a.atomic_formulas.size() + res_b.atomic_formulas.size());
    std::set_union(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                   res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                   std::back_inserter(res),
                   ExprLess());

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

    // For the Or case we intersect the sets of atomic formulas
    std::vector<PrimExpr> res;
    res.reserve(std::min(res_a.atomic_formulas.size(), res_b.atomic_formulas.size()));
    std::set_intersection(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                          res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                          std::back_inserter(res),
                          ExprLess());

    // Computing the residual is more complex: we have to compute the sets of atomic formulas
    // which are left behind, and then combine them with the residuals into the new residual.

    std::vector<PrimExpr> new_cond_a;
    new_cond_a.reserve(res_a.atomic_formulas.size() - res.size());
    std::set_difference(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                        res.begin(), res.end(),
                        std::back_inserter(new_cond_a),
                        ExprLess());

    std::vector<PrimExpr> new_cond_b;
    new_cond_b.reserve(res_b.atomic_formulas.size() - res.size());
    std::set_difference(res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                        res.begin(), res.end(),
                        std::back_inserter(new_cond_b),
                        ExprLess());

    res_a.atomic_formulas = std::move(new_cond_a);
    res_b.atomic_formulas = std::move(new_cond_b);

    PrimExpr new_rest = res_a.to_expr() || res_b.to_expr();

    return {res, new_rest};
  }
};

class NormalizeComparisonsMutator : public ExprMutator {
 public:
  virtual PrimExpr VisitExpr_(const EQNode* op) { return Make<EQNode>(op->a, op->b); }
  virtual PrimExpr VisitExpr_(const NENode* op) { return Make<NENode>(op->a, op->b); }
  virtual PrimExpr VisitExpr_(const LTNode* op) { return Make<LTNode>(op->a, op->b); }
  virtual PrimExpr VisitExpr_(const LENode* op) { return Make<LENode>(op->a, op->b); }
  virtual PrimExpr VisitExpr_(const GTNode* op) { return Make<LTNode>(op->b, op->a); }
  virtual PrimExpr VisitExpr_(const GENode* op) { return Make<LENode>(op->b, op->a); }

 private:
  template <class TNode>
  PrimExpr Make(const PrimExpr& a, const PrimExpr& b) {
    // rewrite LT to LE for ints
    if (std::is_same<TNode, LTNode>::value && (a.dtype().is_int() || a.dtype().is_uint())) {
      return LENode::make(SuperSimplify(a - b + 1), make_zero(a.dtype()));
    }
    return TNode::make(SuperSimplify(a - b), make_zero(a.dtype()));
  }
};

int gcd(int a, int b) {
  if (a < b) std::swap(a, b);
  while (b != 0) {
      int64_t tmp = b;
      b = a % b;
      a = tmp;
  }
  return a;
}

int lcm(int a, int b) {
  return (a*b)/gcd(a, b);
}

std::tuple<int64_t, int64_t, int64_t> xgcd(int64_t a, int64_t b) {
  int64_t s = 0, old_s = 1;
  int64_t t = 1, old_t = 0;
  int64_t r = b, old_r = a;

  while (r != 0) {
    int64_t q = old_r / r;
    std::swap(r, old_r);
    r -= q * old_r;
    std::swap(s, old_s);
    s -= q * old_s;
    std::swap(t, old_t);
    t -= q * old_t;
  }

  CHECK_EQ(a % old_r, 0);
  CHECK_EQ(b % old_r, 0);
  CHECK(old_r == old_s*a + old_t*b);

  return std::make_tuple(old_r, old_s, old_t);
}

struct LinearSystem {
  // \alpha, \beta
  Array<Var> variables;
  // 1 <= \alpha <= N, etc.
  Map<Var, Range> ranges;
  // linear equalities or inequalities
  // e.g., A \alpha = \beta or A \alpha <= \beta
  Array<PrimExpr> relations;

  friend std::ostream& operator<<(std::ostream& os, const LinearSystem& le) {
    return os << "variables = " << le.variables
              << "\nvariables range = " << le.ranges
              << "\nrelations = " << le.relations;
  }
};

struct LinearSystemTransform {
  LinearSystem src;
  LinearSystem dst;
  Map<Var, PrimExpr> src_to_dst;
  Map<Var, PrimExpr> dst_to_src;
};

LinearSystemTransform ComposeLinearTransform(const LinearSystemTransform& base,
                                             const LinearSystemTransform& addition) {
  // TODO: CHECK(second.src.same_as(first.dst));
  Map<Var, PrimExpr> src_to_dst;
  Map<Var, PrimExpr> dst_to_src;
  for (auto p : addition.dst_to_src) {
    dst_to_src.Set(p.first, SuperSimplify(Substitute(p.second, base.dst_to_src),
                                          base.src.ranges));
  }
  for (auto p : base.src_to_dst) {
    src_to_dst.Set(p.first, SuperSimplify(Substitute(p.second, addition.src_to_dst),
                                          addition.dst.ranges));
  }

  LinearSystemTransform res;
  res.src = base.src;
  res.dst = addition.dst;
  res.src_to_dst = src_to_dst;
  res.dst_to_src = dst_to_src;
  return res;
}

LinearSystemTransform SolveSystemOfEquations(const LinearSystem& system_to_solve) {
  const Array<Var>& variables = system_to_solve.variables;
  const Map<Var, Range>& ranges = system_to_solve.ranges;
  const Array<PrimExpr>& conditions = system_to_solve.relations;
  // conditions - A \alpha = \beta
  // variables  - \alpha, \beta
  // ranges     - 1 <= \alpha <= N


  // Conditions we don't know what to do with
  std::vector<PrimExpr> rest;
  // Matrix represented as a vector of rows, each row is an array of coefficients
  std::vector<std::vector<int64_t>> matrix;
  // A column of right hand sides
  std::vector<PrimExpr> rhs;
  // A map from old vars to new vars represented as a matrix, each row of this matrix corresponds to
  // an old variable (from domain->variables) and represents a vector of coefficients
  std::vector<std::vector<int64_t>> old_to_new;
  // A map from new vars to old vars represented directly as an array of expressions
  std::vector<PrimExpr> new_to_old;

  size_t num_vars = variables.size();

  // Initialize the old_to_new matrix with the identity matrix
  for (size_t i = 0; i < num_vars; ++i) {
    old_to_new.emplace_back(num_vars);
    old_to_new.back()[i] = 1;
    new_to_old.push_back(variables[i]);
  }

   auto dumpall = [&]() {
     std::cout << "Matrix:\n";
     for (size_t i = 0; i < matrix.size(); ++i) {
       for (auto e : matrix[i]) {
         std::cout << e << "\t";
       }
       std::cout << "\t->\t" << rhs[i];
       std::cout << "\n";
     }
     std::cout << "old_to_new:\n";
     for (const auto& r : old_to_new) {
       for (auto e : r) {
         std::cout << e << "\t";
       }
       std::cout << "\n";
     }
     std::cout << "new_to_old:\n" << Array<PrimExpr>(new_to_old);
     std::cout << "\n" << std::endl;
   };

  LOG(INFO) << "variables = " << variables;

  // Transform formulas into rows of the matrix
  // matrix [\vec{\alpha}, \vec{\beta}]^T = rhs
  for (const PrimExpr& formula : conditions) {
    LOG(INFO) << "old condition = " << formula;
    if (const EQNode* eq = formula.as<EQNode>()) {
      // \beta + coeff * \alpha = 0
      // a-b = sum_{i=0}^{n-1} variables[i] * coeff[i] + coeff[n]
      Array<PrimExpr> coeffs = arith::DetectLinearEquation(SuperSimplify(eq->a - eq->b, ranges), variables);
      if (!coeffs.empty()) {
        std::vector<int64_t> row;
        for (size_t j = 0; j < coeffs.size() - 1; ++j) {
          PrimExpr c = coeffs[j];
          if (const IntImmNode* ic = c.as<IntImmNode>()) {
            row.push_back(ic->value);
          } else {
            // ignore some formulas that we cannot deal with.
            row.clear();
            break;
          }
        }

        if (!row.empty()) {
          matrix.push_back(row);
          rhs.push_back(-coeffs[coeffs.size() - 1]);
          continue;
        }
      }
    }

    // otherwise
    rest.push_back(formula);
  }
  dumpall();

  // Diagonalize the matrix
  // find out USV = A
  for (size_t index = 0; index < std::min(matrix.size(), num_vars); ++index) {
    // Here the matrix is partially diagonalized, that is matrix[i, j] is zero for all i, j
    // such that (i < index) or (j < index), unless (i == j).
    // That is, now we are diagonalizing the submatrix with i >= index and j >= index

    // Find a row with a nonzero element in the index-th column
    // (We also prefer rows where this element has minimal abs value)
    size_t best_i = index;
    for (size_t i = best_i; i < matrix.size(); ++i) {
      int64_t m_old = matrix[best_i][index];
      int64_t m_new = matrix[i][index];
      if (m_new != 0) {
        if (m_old == 0 || std::abs(m_new) < std::abs(m_old)) {
          best_i = i;
        }
      }
    }
    // Move the row we found to the index-th position
    std::swap(matrix[index], matrix[best_i]);
    std::swap(rhs[index], rhs[best_i]);

    // If the index-th diagonal element is still zero, try to find a column with nonzero index-th
    // element and move it to the index-th position
    if (matrix[index][index] == 0) {
      for (size_t j = index + 1; j < num_vars; ++j) {
        if (matrix[index][j] != 0) {
          for (size_t i = index; i < matrix.size(); ++i) {
            std::swap(matrix[i][index], matrix[i][j]);
          }
          // swapping columns corresponds to swapping the corresponding new variables
          std::swap(new_to_old[index], new_to_old[j]);
          for (size_t i = 0; i < old_to_new.size(); ++i) {
            std::swap(old_to_new[i][index], old_to_new[i][j]);
          }
          break;
        }
      }
    }

    // If the index-th diagonal element is still zero, then both the index-th row and the index-th
    // column are completely zero, and we don't need to do anything; just go to the next index
    if (matrix[index][index] == 0) {
      continue;
    }

    // Now the index-th diagonal element is non-zero and we can zero all the index-th column
    // below it by subtracting rows from each other
    for (auto i = index + 1; i < matrix.size(); ++i) {
      if (matrix[i][index] != 0) {
        int64_t g, a, b;
        // g = a*matrix[index][index] + b*matrix[i][index]
        if (matrix[i][index] % matrix[index][index] != 0) {
          std::tie(g, a, b) = xgcd(matrix[index][index], matrix[i][index]);
        } else {
          // Explicitly avoid changing the index-th row. This is important to avoid infinite
          // loop.
          g = matrix[index][index];
          a = 1;
          b = 0;
        }

        // Let m = matrix[index][index], n = matrix[i][index], then the following is true:
        //
        // [ a   n/g ][ m/g  n/g ] = [ 1  0 ]
        // [ b  -m/g ][ b    -a  ] = [ 0  1 ]
        //
        // Note that the two matrices are integer (since g = gcd(m, n)).
        // We will essentially multiply our matrix on the left by a dilated and transposed version
        // of the first of these two matrices. The second matrix is not needed here, however we will
        // use it while zeroing the index-th row.

        int64_t m_g = matrix[index][index] / g;
        int64_t n_g = matrix[i][index] / g;

        // Note that j is the index of the column, not the row
        for (size_t j = index; j < matrix[i].size(); ++j) {
          // Multiply index-th row by a and add the i-th row multiplied by b
          // This will make the index-th diagonal element equal to the gcd
          int64_t new_index_j = a*matrix[index][j] + b*matrix[i][j];
          // This transformation performs zeroing of matrix[i][index]
          int64_t new_i_j = n_g*matrix[index][j] - m_g*matrix[i][j];
          matrix[index][j] = new_index_j;
          matrix[i][j] = new_i_j;
        }
        // We have to do the same with rhs
        PrimExpr ea = make_const(rhs[index].dtype(), a);
        PrimExpr eb = make_const(rhs[i].dtype(), b);
        PrimExpr e_m_g = make_const(rhs[i].dtype(), m_g);
        PrimExpr e_n_g = make_const(rhs[index].dtype(), n_g);
        PrimExpr new_index_rhs = ea*rhs[index] + eb*rhs[i];
        PrimExpr new_i_rhs = e_n_g*rhs[index] - e_m_g*rhs[i];
        rhs[index] = new_index_rhs;
        rhs[i] = new_i_rhs;
      }
    }

    bool changed = false;

    // Now we have to zero the elements of the index-th row by manipulating columns.
    // This is more difficult because column manipulation corresponds to variable manipulation,
    // but the algorithm is essentially the same as before.
    for (size_t j = index + 1; j < num_vars; ++j) {
      if (matrix[index][j] != 0) {
        int64_t g, a, b;
        // g = a*matrix[index][index] + b*matrix[index][j]
        if (matrix[index][j] % matrix[index][index] != 0) {
          std::tie(g, a, b) = xgcd(matrix[index][index], matrix[index][j]);
          // During this phase we may disrupt the zeroness of the index-th column, so we will
          // have to take some action if this might have happened.
          changed = true;
        } else {
          // Explicitly avoid changing the index-th column. This is important to avoid infinite
          // loop. Note that here we don't have to set `changed` to true since we don't change the
          // index-th column.
          g = matrix[index][index];
          a = 1;
          b = 0;
        }

        // Let m = matrix[index][index], n = matrix[index][j], then the following is true:
        //
        // [ a   n/g ][ m/g  n/g ] = [ 1  0 ]
        // [ b  -m/g ][ b    -a  ] = [ 0  1 ]
        //
        // Now we are going to multiply our matrix on the right (to manipulate columns instead of
        // rows), we will also transform the old_to_new matrix the same way, and we will use the
        // second matrix to transform new_to_old.

        int64_t m_g = matrix[index][index] / g;
        int64_t n_g = matrix[index][j] / g;

        for (size_t i = index; i < matrix.size(); ++i) {
          int64_t new_i_index = a*matrix[i][index] + b*matrix[i][j];
          int64_t new_i_j = n_g*matrix[i][index] - m_g*matrix[i][j];
          matrix[i][index] = new_i_index;
          matrix[i][j] = new_i_j;
        }
        // We do exactly the same transformations with old_to_new
        for (size_t i = 0; i < old_to_new.size(); ++i) {
          int64_t new_i_index = a*old_to_new[i][index] + b*old_to_new[i][j];
          int64_t new_i_j = n_g*old_to_new[i][index] - m_g*old_to_new[i][j];
          old_to_new[i][index] = new_i_index;
          old_to_new[i][j] = new_i_j;
        }
        // And apply reverse transformations to new_to_old.
        PrimExpr ea = make_const(new_to_old[j].dtype(), a);
        PrimExpr eb = make_const(new_to_old[index].dtype(), b);
        PrimExpr e_m_g = make_const(new_to_old[index].dtype(), m_g);
        PrimExpr e_n_g = make_const(new_to_old[j].dtype(), n_g);
        PrimExpr new_index = e_m_g*new_to_old[index] + e_n_g*new_to_old[j];
        PrimExpr new_j = eb*new_to_old[index] - ea*new_to_old[j];
        new_to_old[index] = new_index;
        new_to_old[j] = new_j;
      }
    }

    if (changed) {
      // We might have changed the first column, so we have to zero it once more (or at least check
      // if it's zero), so just perform this iteration once more.
      index -= 1;
    }
  }

  // matrix old_to_new [\vec{\alpha}, \vec{\beta}]^T = \hat{rhs}
  // new_to_old = old_to_new [\vec{\alpha}, \vec{\beta}]^T
  // matrix is now diagonal (singular values)
  // i.e., A x = b
  // => S V^T x = U b
  // in which
  // S = matrix
  // x = [\vec{\alpha}, \vec{\beta}]^T
  // b = rhs
  // V^T = old_to_new
  // U b = \hat{rhs}
  LOG(INFO) << "After diag: ";
  dumpall();

  Array<Var> new_vars;
  Map<Var, PrimExpr> new_to_old_map;
  Array<PrimExpr> solution;
  Array<PrimExpr> new_conditions;

  // Simplify right hand sides
  for (PrimExpr r : rhs) {
    r = SuperSimplify(r, ranges);
  }

  // Create the conditions of the existence of a solution
  for (size_t j = 0; j < matrix.size(); ++j) {
    PrimExpr new_cond;
    if (j >= num_vars || matrix[j][j] == 0) {
      // The row of matrix is zero. A solution exists only if the rhs[j] is also zero
      new_cond = (rhs[j] == 0);
    } else {
      // The diagonal element is non-zero. A solution exists only if the diagonal element
      // is a divisor of the rhs[j]
      new_cond = (floormod(rhs[j], std::abs(matrix[j][j])) == 0);
    }
    new_cond = SuperSimplify(new_cond, ranges);
    if (is_const_int(new_cond, 0)) {
      return LinearSystemTransform(); // TODO: ZE_LOG_RES(EmptyDomainTransformation(domain));
    } else if (!is_const_int(new_cond, 1)) {
      new_conditions.push_back(new_cond);
    }
  }
  LOG(INFO) << "new conditions = " << new_conditions;

  // Now create new variables or directly solve the equations
  for (size_t j = 0; j < num_vars; ++j) {
    if (j >= matrix.size() || matrix[j][j] == 0) {
      // The j-th variable can take any integer value, create a tvm variable for it
      // x = (pseudo-inverse of A) b + K_{m, m-r} z_{m-r}
      //   = V_{m,m} (pseudo-inverse of S_{m,n}) U_{n,n} b_{n} + K_{m, m-r} z_{m-r}
      // in which K is the right m-r columns of V
      // thus z is variable essentially means the last m-r rows (elements) of V^T x (solution) are variables
      // because last m-r rows of (pseudo-inverse of S_{m,n}) U_{n,n} b_{n} are zeros
      // and V^T K 's first r rows are zeros, remaining m-r rows is diag(1, ... 1)
      PrimExpr to_old = SuperSimplify(new_to_old[j], ranges);
      std::string name_hint = "n" + std::to_string(new_vars.size());
      if (const VarNode* v_old = to_old.as<VarNode>()) {
        name_hint += "_" + v_old->name_hint;
      }
      Var v = Var(name_hint, new_to_old[j].dtype());
      solution.push_back(v);
      new_vars.push_back(v);
      new_to_old_map.Set(v, to_old);
    } else {
      // The j-th variable is just a single value, don't create a tvm variable
      if (matrix[j][j] >= 0) {
        PrimExpr a = make_const(rhs[j].dtype(), matrix[j][j]);
        // (pseudo-inverse of matrix) * rhs
        // solution = V^T x = S^{-1} U b
        solution.push_back(SuperSimplify(floordiv(rhs[j], a), ranges));
      } else {
        // This is required because some simplifiers have problems with dividing by negative numbers
        PrimExpr a = make_const(rhs[j].dtype(), -matrix[j][j]);
        solution.push_back(SuperSimplify(floordiv(-rhs[j], a), ranges));
      }
    }
  }

  LOG(INFO) << "new_vars = " << new_vars;
  LOG(INFO) << "solution = " << solution;
  dumpall();

  // Convert the old_to_new matrix to map
  Map<Var, PrimExpr> old_to_new_map;
  for (size_t i = 0; i < num_vars; ++i) {
    PrimExpr e = make_zero(variables[i].dtype());
    // old_to_new * old = V^T * new
    // solution = V^T * new
    // V * solution
    for (size_t j = 0; j < num_vars; ++j) {
      e = e + make_const(e.dtype(), old_to_new[i][j])*solution[j];
    }
    e = SuperSimplify(e);
    old_to_new_map.Set(variables[i], e);
  }

  LOG(INFO) << "old_to_new_map = " << old_to_new_map;

  // From now on we will use sorted domain variable ranges to increase determinism
  std::vector<std::pair<Var, Range>> sorted_domain_ranges = VarMapToVectorOfPairs(ranges);

  // The resulting ranges
  Map<Var, Range> new_ranges;

  // First of all, fill the new ranges with outer variable ranges
  std::unordered_set<const VarNode*> vset;
  for (const Var& v : variables) {
    vset.insert(v.get());
  }
  for (const auto& p : sorted_domain_ranges) {
    if (!vset.count(p.first.get())) {
      new_ranges.Set(p.first, p.second);
    }
  }

  LOG(INFO) << "new ranges = " << new_ranges;

  // Convert original ranges to IntSets
  std::unordered_map<const VarNode*, IntSet> var_intsets;
  for (const auto& p : sorted_domain_ranges) {
    var_intsets[p.first.get()] = IntSet::range(p.second);
  }

  // Infer ranges for the new variables and add them to the resulting ranges
  for (const auto& p : new_to_old_map) {
    Range range = EvalSet(p.second, var_intsets).cover_range(Range());
    if (range.defined()) {
      new_ranges.Set(p.first, range);
    }
  }
  LOG(INFO) << "new ranges = " << new_ranges;
  LOG(INFO) << "new_condition = " << new_conditions;

  // We have to transform ranges of the old variables into conditions over new variables because new
  // ranges are not enough usually.
  for (const auto& p : sorted_domain_ranges) {
    if (old_to_new_map.count(p.first)) {
      PrimExpr in_terms_of_new = old_to_new_map[p.first];
      LOG(INFO) << "old = " << p.first << " in_terms_of_new = " << in_terms_of_new;
      PrimExpr lower_cond = SuperSimplify(p.second->min <= in_terms_of_new, new_ranges);
      PrimExpr upper_cond = SuperSimplify(in_terms_of_new < p.second->min + p.second->extent, new_ranges);
      LOG(INFO) << "lower_cond = " << (p.second->min <= in_terms_of_new) << " ranges = " << new_ranges;
      if (!is_const_int(lower_cond, 1)) {
        new_conditions.push_back(lower_cond);
      }
      if (!is_const_int(upper_cond, 1)) {
        new_conditions.push_back(upper_cond);
      }
    }
  }
  LOG(INFO) << "new condidtions = " << new_conditions;

  // Add the rest conditions
  for (const PrimExpr& cond : rest) {
    LOG(INFO) << "rest cond = " << cond;
    new_conditions.push_back(Substitute(cond, old_to_new_map));
  }

  LOG(INFO) << "\nnew_vars = " << new_vars
            << "\nnew_conditions = " << new_conditions
            << "\nnew_ranges = " << new_ranges
            << "\nold_to_new_map = " << old_to_new_map
            << "\nnew_to_old_map = " << new_to_old_map;

//  Domain new_domain = DomainNode::make(new_vars, new_conditions, new_ranges);
//  ZE_LOG_RES(DomainTransformationNode::make(new_domain, domain,
//                                            new_to_old_map, old_to_new_map))

  LinearSystem new_le;
  new_le.variables = new_vars;
  new_le.ranges = new_ranges;
  new_le.relations = new_conditions;

  LinearSystemTransform transform;
  transform.src = system_to_solve;
  transform.dst = new_le;
  transform.src_to_dst = old_to_new_map;
  transform.dst_to_src = new_to_old_map;

  return transform;
}

VarBounds VarBounds::substitute(const Map<Var, PrimExpr>& subst) const {
  auto apply_fun = [&subst](const PrimExpr& e) { return Substitute(e, subst); };
  return {Substitute(coef, subst),
          UpdateArray(lower, apply_fun),
          UpdateArray(equal, apply_fun),
          UpdateArray(upper, apply_fun)};
}

Array<PrimExpr> SolveSystemOfInequalitiesResult::as_conditions() const {
  Array<PrimExpr> res;
  for (const Var& v : variables) {
    auto it = bounds.find(v.get());
    CHECK(it != bounds.end());
    const VarBounds& bnds = it->second;
    PrimExpr lhs = bnds.coef * v;
    for (const PrimExpr& rhs : bnds.equal) {
      res.push_back(EQNode::make(lhs, rhs));
    }
    for (const PrimExpr& rhs : bnds.lower) {
      res.push_back(GENode::make(lhs, rhs));
    }
    for (const PrimExpr& rhs : bnds.upper) {
      res.push_back(LENode::make(lhs, rhs));
    }
  }
  for (const PrimExpr& e : other_conditions) {
    res.push_back(e);
  }
  return res;
}

SolveSystemOfInequalitiesResult SolveSystemOfInequalities(
    const Array<PrimExpr>& inequalities,
    const Array<Var>& variables,
    const Map<Var, Range>& vranges) {


  LOG(INFO) << "solving inequalities " << inequalities;
  arith::Analyzer analyzer;
  for (auto kv : vranges) {
    analyzer.Bind(kv.first, kv.second);
  }
  SolveSystemOfInequalitiesResult res;
  res.variables = variables;

  // The algorithm consists in doing the following things for each variable v
  // - Take formulas from `current` and classify them according to polarity wrt v
  // - Combine each formula of positive polarity (wrt v) with each formula of negative polarity
  // - Put the resulting combinations into `new_current` along with unclassifiable formulas
  // - Replace `current` with `new_current` and move to the next variable

  // normalized inequality
  // current and new_current are sorted to enable some heuristics
  std::set<PrimExpr, ExprLess> current;
  std::set<PrimExpr, ExprLess> new_current;
  // A vector of pairs (c, e), c > 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_pos;
  // A vector of pairs (c, e), c < 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_neg;

  // formulas we don't know what to do with
  std::vector<PrimExpr> rest;

  auto debug_print = [&]() {
    std::cout << "Current:\n[";
    for (auto& ineq : current) {
      std::cout << ineq << ", ";
    }
    std::cout << "]\n";

    std::cout << "New Current:\n[";
    for (auto& ineq : current) {
      std::cout << ineq << ", ";
    }
    std::cout << "]\n";

    std::cout << "coef_pos:\n[";
    for (auto& coef : coef_pos) {
      std::cout << "(" << coef.first << ", " << coef.second << "), ";
    }
    std::cout << "]\n";

    std::cout << "coef_neg:\n[";
    for (auto& coef : coef_neg) {
      std::cout << "(" << coef.first << ", " << coef.second << "), ";
    }
    std::cout << "]\n";
  };

  auto add_to_new_current = [&new_current, &vranges, &analyzer] (const PrimExpr& new_ineq) {
    if (analyzer.CanProve(new_ineq)) {
      // redundant: follows from the vranges
      return;
    }
    LOG(INFO) << "add new inequality = " << new_ineq;
    if (const LENode* new_le = new_ineq.as<LENode>()) {
      // A heuristic: check if the new inequality is a consequence of one
      // of its future neighbors (in this case don't add it) or if a future neighbor is
      // a consequence of the new ineq (in which case remove the neighbor)
      auto it_neighbor = new_current.lower_bound(new_ineq);
      if (it_neighbor != new_current.begin()) {
        const LENode* le = std::prev(it_neighbor)->as<LENode>();
        if (le && analyzer.CanProve(new_le->a - le->a <= 0)) {
          return;
        } else if (le && analyzer.CanProve(le->a - new_le->a <= 0)) {
          new_current.erase(std::prev(it_neighbor));
        }
      }
      // Check the other neighbor
      if (it_neighbor != new_current.end()) {
        const LENode* le = it_neighbor->as<LENode>();
        if (le && analyzer.CanProve(new_le->a - le->a <= 0)) {
          return;
        } else if (le && analyzer.CanProve(le->a - new_le->a <= 0)) {
          it_neighbor = new_current.erase(it_neighbor);
        }
      }

      new_current.insert(it_neighbor, new_ineq);
    } else {
      new_current.insert(new_ineq);
    }
  };

  // Simplify each inequality into the form `expr <= 0` and add to new_current formulas
  for (const PrimExpr& ineq : inequalities) {
    add_to_new_current(NormalizeComparisonsMutator()(SuperSimplify(ineq, vranges)));
  }

  std::swap(current, new_current);

  for (const Var& v : variables) {
    CHECK(!res.bounds.count(v.get())) <<
      "Variable " << v << " appears more than one time in the `variables` which might be a bug";

    new_current.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (vranges.count(v)) {
      const Range& range = vranges[v];
      PrimExpr range_lbound = SuperSimplify(range->min, vranges);
      PrimExpr range_ubound = SuperSimplify(range->min + range->extent - 1, vranges);
      coef_neg.push_back({-1, range_lbound});
      coef_pos.push_back({1, -range_ubound});
    }

    // Take formulas from `current` and classify them according to polarity wrt v
    // and store to coef_pos and coef_neg respectively.
    LOG(INFO) << "current.size = " << current.size();
    for (const PrimExpr& ineq : current) {
      LOG(INFO) << "classify inequality " << ineq;
      if (const LENode* le = ineq.as<LENode>()) {
        Array<PrimExpr> coef = arith::DetectLinearEquation(le->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0) {
            // zero polarity, straight to new_current
            add_to_new_current(ineq);
          } else if (coef0 > 0) {
            coef_pos.push_back({coef0, coef[1]});
          } else if (coef0 < 0) {
            coef_neg.push_back({coef0, coef[1]});
          }
          continue;
        }
      } else if (const EQNode* eq = ineq.as<EQNode>()) {
        Array<PrimExpr> coef = arith::DetectLinearEquation(eq->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0) {
            // zero polarity, straight to new_current
            add_to_new_current(ineq);
          } else if (coef0 > 0) {
            // Equalities may be considered as pairs of two inequalities
            coef_pos.push_back({coef0, coef[1]});
            coef_neg.push_back({-coef0, -coef[1]});
          } else if (coef0 < 0) {
            coef_pos.push_back({-coef0, -coef[1]});
            coef_neg.push_back({coef0, coef[1]});
          }
          continue;
        }
      }

      // if nothing worked, put it in rest
      rest.push_back(ineq);
    }
    LOG(INFO) << "Debug print for " << v;
    debug_print();

    // Combine each positive inequality with each negative one (by adding them together)
    for (const auto& pos : coef_pos) {
      for (const auto& neg : coef_neg) {
        auto first_gcd = gcd(pos.first, -neg.first);
        PrimExpr c_pos = make_const(v.dtype(), neg.first/first_gcd);
        PrimExpr c_neg = make_const(v.dtype(), pos.first/first_gcd);
        PrimExpr new_lhs = c_neg*neg.second - c_pos*pos.second;
        PrimExpr new_ineq = LENode::make(new_lhs, make_zero(pos.second.dtype()));
        LOG(INFO) << "new_ineq = " << new_ineq;
        new_ineq = NormalizeComparisonsMutator()(SuperSimplify(new_ineq, vranges));
        add_to_new_current(new_ineq);
      }
    }
    LOG(INFO) << "After adding together ";
    debug_print();

    // Now we have to generate resulting (in)equalities for the variable v

    // Find the common denominator in a sense
    // We will generate formulas of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto& pos : coef_pos) {
      coef_lcm = lcm(coef_lcm, pos.first);
    }
    for (const auto& neg : coef_neg) {
      coef_lcm = lcm(coef_lcm, -neg.first);
    }

    // The resulting lower and upper bounds stored in sorted vectors
    std::vector<PrimExpr> upper_bounds;
    std::vector<PrimExpr> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto& pos : coef_pos) {
      PrimExpr bound = make_const(v.dtype(), -coef_lcm/pos.first)*pos.second;
      bound = SuperSimplify(bound, vranges);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &vranges, &analyzer](const PrimExpr& o)
                      { return analyzer.CanProve(o - bound <= 0); })) {
        continue;
      }
      // Erase all worse bounds
      upper_bounds.erase(
        std::remove_if(upper_bounds.begin(), upper_bounds.end(),
                       [&bound, &vranges, &analyzer](const PrimExpr& o)
                       { return analyzer.CanProve(o - bound >= 0); }),
        upper_bounds.end());
      // Add
      upper_bounds.push_back(bound);
    }
    for (const auto& neg : coef_neg) {
      PrimExpr bound = make_const(v.dtype(), -coef_lcm/neg.first)*neg.second;
      bound = SuperSimplify(bound, vranges);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &vranges, &analyzer](const PrimExpr& o)
                      { return analyzer.CanProve(o - bound >= 0); })) {
        continue;
      }
      // Erase all worse bounds
      lower_bounds.erase(
        std::remove_if(lower_bounds.begin(), lower_bounds.end(),
                       [&bound, &vranges, &analyzer](const PrimExpr& o)
                       { return analyzer.CanProve(o - bound <= 0); }),
        lower_bounds.end());
      // Add
      lower_bounds.push_back(bound);
    }

    // Sort the vectors and remove duplicates
    for (std::vector<PrimExpr>* bounds : {&upper_bounds, &lower_bounds}) {
      std::sort(bounds->begin(), bounds->end(), ExprLess());
      bounds->erase(std::unique(bounds->begin(), bounds->end(), ExprEq()), bounds->end());
    }

    // Bounds which are both lower and upper should go to equal...
    std::vector<PrimExpr> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    std::set_intersection(upper_bounds.begin(), upper_bounds.end(),
                          lower_bounds.begin(), lower_bounds.end(),
                          std::back_inserter(equal), ExprLess());

    // ...and be removed from upper bounds...
    std::vector<PrimExpr> new_upper;
    new_upper.reserve(upper_bounds.size() - equal.size());
    std::set_difference(upper_bounds.begin(), upper_bounds.end(),
                        equal.begin(), equal.end(),
                        std::back_inserter(new_upper), ExprLess());

    // ...and from lower bounds.
    std::vector<PrimExpr> new_lower;
    new_lower.reserve(lower_bounds.size() - equal.size());
    std::set_difference(lower_bounds.begin(), lower_bounds.end(),
                        equal.begin(), equal.end(),
                        std::back_inserter(new_lower), ExprLess());

    // Write it to the result.
    auto& bnds = res.bounds[v.get()];
    bnds.coef = make_const(v.dtype(), coef_lcm);
    bnds.equal = equal;
    bnds.lower = new_lower;
    bnds.upper = new_upper;
    LOG(INFO) << "Bound of " << v << " coef = " << bnds.coef
              << " EQUAL: " << bnds.equal
              << " LOWER: " << bnds.lower
              << " UPPER: " << bnds.upper;

    std::swap(current, new_current);
  }

  // Everything that is left goes to res.other_conditions
  for (const PrimExpr& e : current) {
    PrimExpr e_simp = SuperSimplify(e, vranges);
    if (is_const_int(e_simp, 0)) {
      // contradiction detected
      res.other_conditions = {const_false()};
      return res;
    } else if (is_const_int(e_simp, 1)) {
      continue;
    } else {
      res.other_conditions.push_back(e_simp);
    }
  }

  for (const PrimExpr& e : rest) {
    res.other_conditions.push_back(e);
  }

  return res;
}

// Deskew the given domain
LinearSystemTransform DeskewDomain(const LinearSystem& inequality) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the domain->variables but are in domain->ranges (jac_xxx)
  Map<Var, Range> res_ranges;

  /*
  // vars are variables from domain's variables followed by all the other variables from its ranges
  // TODO: we don't need this if eq.ranges only contains variables in eq.variables
  Array<Var> vars = equation.variables;
  for (const auto& pair : VarMapToVectorOfPairs(equation.ranges)) {
    bool already = false;
    for (const Var& v : vars) {
      already = already || v.same_as(pair.first);
    }
    if (!already) {
      vars.push_back(pair.first);
      // Also populate the resulting ranges with ranges of outer variables
      res_ranges.Set(pair.first, pair.second);
    }
  }
  LOG(INFO) << "res_ranges = " << res_ranges;
  */

  // we get a set of equality, lower, upper bound of each variable.
  auto solved_system = SolveSystemOfInequalities(inequality.relations, inequality.variables, inequality.ranges);

  arith::Analyzer analyzer;

  Map<Var, PrimExpr> res_old_to_new;
  Map<Var, PrimExpr> res_new_to_old;
  Array<Var> res_variables;
  Array<PrimExpr> res_relations;
  std::unordered_map<const VarNode*, IntSet> new_var_intsets;

  // this keeps being updated during determining the range of each variable.
  Map<Var, Range> vranges = inequality.ranges;

  // Initialize new_var_intsets with the old var intsets
  for (const auto& pair : inequality.ranges) {
    new_var_intsets[pair.first.get()] = IntSet::range(pair.second);
    analyzer.Bind(pair.first, pair.second);
  }

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = inequality.variables.rbegin(); it != inequality.variables.rend(); ++it) {
    const Var& var = *it;
    LOG(INFO) << "Processing variable " << var;
    auto& bnd = solved_system.bounds[var.get()];
    // Note that we replace old vars with new ones
    bnd = bnd.substitute(res_old_to_new);

    LOG(INFO) << "Coefficient " << bnd.coef;
    if (is_one(bnd.coef) && !bnd.equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity, so it must be
      // the simplest one.
      res_old_to_new.Set(var, bnd.equal[0]);
      LOG(INFO) << "Replaced with " << bnd.equal[0];
    } else {
      std::vector<PrimExpr> lowers(bnd.equal.begin(), bnd.equal.end());
      std::vector<PrimExpr> uppers(bnd.equal.begin(), bnd.equal.end());
      for (const auto& expr : bnd.lower) lowers.push_back(expr);
      for (const auto& expr : bnd.upper) uppers.push_back(expr);

      LOG(INFO) << "LowersUnsorted " << Array<PrimExpr>(lowers);
      LOG(INFO) << "UppersUnsorted " << Array<PrimExpr>(uppers);

      std::sort(lowers.begin(), lowers.end(), ExprLess());
      std::sort(uppers.begin(), uppers.end(), ExprLess());

      LOG(INFO) << "Lowers " << Array<PrimExpr>(lowers);
      LOG(INFO) << "Uppers " << Array<PrimExpr>(uppers);

      // Here we will try all pairs of lower and upper bounds and find the best pair, that is, the
      // pair with the minimal difference between the upper and the lower.
      // Note that the bounds are for v, not for v*coef, because we will need bounds for v anyway

      // The lower bound of the best pair so far
      PrimExpr best_lower = vranges[var]->min;
      // The difference between the upper and the lower of the best pair, maybe overapproximation
      PrimExpr best_diff_over = vranges[var]->extent - 1;

      LOG(INFO) << "Initial best low " << best_lower;
      LOG(INFO) << "Initial best diff_over " << best_diff_over;

      for (const PrimExpr& low : lowers) {
        for (const PrimExpr& upp : uppers) {
          LOG(INFO) << "Considering low " << low;
          LOG(INFO) << "Considering upp " << upp;
          PrimExpr diff_1 = SuperSimplify(floordiv(upp - low, bnd.coef), vranges);
          // Since diff may depend on some other variables, we compute its overapproximation
          PrimExpr diff_over_1 = SuperSimplify(EvalSet(diff_1, new_var_intsets).max(), vranges);

          // low is the lower bound for v*coef, but we need the lower bound for v.
          // We use rounding-up division to compute it. Since we want to use a single formula
          PrimExpr low_divided = SuperSimplify(floordiv(low + bnd.coef - 1, bnd.coef), vranges);

          LOG(INFO) << "Considering low_divided " << low_divided;
          LOG(INFO) << "Considering diff_1 " << diff_1;
          LOG(INFO) << "Considering diff_over_1 " << diff_over_1;

          // Compute another difference which may be more precise (or not).
          PrimExpr diff_2 = SuperSimplify(floordiv(upp, bnd.coef) - low_divided, vranges);
          PrimExpr diff_over_2 = SuperSimplify(EvalSet(diff_2, new_var_intsets).max(), vranges);

          LOG(INFO) << "Considering diff_2 " << diff_2;
          LOG(INFO) << "Considering diff_over_2 " << diff_over_2;

          if (analyzer.CanProve(diff_over_2 - diff_over_1 < 0)) {
            diff_over_1 = diff_over_2;
          }

          PrimExpr diff_over_1_is_better_expr;
          diff_over_1_is_better_expr = diff_over_1 - best_diff_over < 0;

          // If it is provable that the new one is strictly better than the current best one,
          // then replace it. Note that we are biased towards earlier pairs which should be simpler.
          if (analyzer.CanProve(diff_over_1_is_better_expr)) {
            LOG(INFO) << "Current best low " << low_divided;
            LOG(INFO) << "Current best diff " << diff_over_1;
            best_lower = low_divided;
            best_diff_over = diff_over_1;
          }
        }
      }
      LOG(INFO) << "Resulting best low " << best_lower;
      LOG(INFO) << "Resulting best diff_over " << best_diff_over;

      std::string suffix = Equal(best_lower, vranges[var]->min) ? "" : ".shifted";
      Var new_var = var.copy_with_suffix(suffix);

      PrimExpr diff = SuperSimplify(best_diff_over, vranges);

      if (is_const_int(diff, 0)) {
        // Don't create an itervar, just replace it everywhere with its min
        res_old_to_new.Set(var, best_lower);
        LOG(INFO) << "Replaced with " << best_lower;
      } else {
        // created new_var starts from 0
        res_old_to_new.Set(var, new_var + best_lower);
        // Note that we are substituting old with new, so best_lower contains new var,
        // that is we have to substitute new with old in best_lower here
        res_new_to_old.Set(new_var,
                           SuperSimplify(var - Substitute(best_lower, res_new_to_old), vranges));

        new_var_intsets[new_var.get()] = IntSet::interval(make_zero(new_var.dtype()), diff);

        // Add the new var to the resulting axis
        auto range = Range(make_zero(new_var.dtype()), SuperSimplify(diff + 1, vranges));
        res_variables.push_back(new_var);
        res_ranges.Set(new_var, range);
        vranges.Set(new_var, range);

        LOG(INFO) << "Replaced with " << (new_var + best_lower);
        LOG(INFO) << "New var range " << range;
      }
    }
  }

  // Add the original conditions (with variables substituted) to the resulting conditions
  for (const PrimExpr& old_cond : solved_system.as_conditions()) {
    PrimExpr new_cond = SuperSimplify(Substitute(old_cond, res_old_to_new), vranges);
    if (!is_const_int(new_cond, 1)) {
      // those not represented in vranges (res_ranges)
      res_relations.push_back(new_cond);
    }
  }
  LOG(INFO) << "res_conditions = " << res_relations;

  // Reverse the axis so that it matches the order of the original variables
  res_variables = Array<Var>(res_variables.rbegin(), res_variables.rend());

  LinearSystem new_inequality;
  new_inequality.variables = res_variables;
  new_inequality.ranges = res_ranges;
  new_inequality.relations = res_relations;
  LOG(INFO) << "result linear system:\n" << new_inequality;

  LinearSystemTransform transform;
  transform.src = inequality;
  transform.dst = new_inequality;
  transform.src_to_dst = res_old_to_new;
  transform.dst_to_src = res_new_to_old;

  return transform;
}

// Use the condition of a reduction op to simplify its domain (axis)
PrimExpr SimplifyReductionDomain(const PrimExpr& expr, const Map<Var, Range>& outer_vranges) {
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    Map<Var, Range> vranges = Merge(outer_vranges, IterVarsToMap(red->axis));
    Array<Var> reduce_vars;
    for (const IterVar& v : red->axis) {
      reduce_vars.push_back(v->var);
    }

    LinearSystem system_to_solve;
    system_to_solve.variables = reduce_vars;
    system_to_solve.relations = FactorOutAtomicFormulasFunctor()(red->condition).to_array();
    system_to_solve.ranges = Merge(outer_vranges, IterVarsToMap(red->axis));

    LOG(INFO) << "system to solve = " << system_to_solve;

    LinearSystemTransform solution_eq = SolveSystemOfEquations(system_to_solve);
    LinearSystemTransform res = DeskewDomain(solution_eq.dst);
    res = ComposeLinearTransform(solution_eq, res);

    Array<PrimExpr> new_source;
    for (const PrimExpr& src : red->source) {
      new_source.push_back(tir::Substitute(src, res.src_to_dst));
    }

    LOG(INFO) << "old source = " << red->source;
    LOG(INFO) << "new source = " << new_source;

    Array<IterVar> new_axis =
        IterVarsFromMap(res.dst.variables, res.dst.ranges, kCommReduce);

    // Perform simplification mainly to remove a possibly empty reduction.
    return SuperSimplify(ReduceNode::make(red->combiner, new_source, new_axis,
                                          All(res.dst.relations),
                                          red->value_index));
  } else {
    return expr;
  }
}

PrimExpr ExtractAsTensorMaybe(const PrimExpr& nonzero_value, const PrimExpr& nonzero_cond,
                              const Array<Var>& outer_axis,
                              const Map<Var, Range>& vrange) {
  LOG(INFO) << "cond = " << nonzero_cond;
  Array<PrimExpr> atomic_cond = FactorOutAtomicFormulasFunctor().VisitExpr(nonzero_cond).to_array();
  LOG(INFO) << "cond factor out atomic = " << atomic_cond;
  LOG(INFO) << "outer axis = " << outer_axis;
//  Domain domain = DomainNode::make(outer_axis, atomic_cond, vrange);

  LinearSystem system_to_solve;
  system_to_solve.variables = outer_axis;
  system_to_solve.ranges = vrange;
  // conditions leads to non-zero output of grad.
  system_to_solve.relations = atomic_cond;

  // TODO:
  // SimplifyDomain(domain); {
  LinearSystemTransform solution_eq = SolveSystemOfEquations(system_to_solve);
  LinearSystemTransform res = DeskewDomain(solution_eq.dst);
  res = ComposeLinearTransform(solution_eq, res);
  // }

  LOG(INFO) << "nonzero expr = " << nonzero_value;
  LOG(INFO) << "src_to_dst = " << res.src_to_dst;
  LOG(INFO) << "dst_to_src = " << res.dst_to_src;
  LOG(INFO) << "dst.ranges = " << res.dst.ranges;
  PrimExpr new_nonzero_val = SuperSimplify(Substitute(nonzero_value, res.src_to_dst), res.dst.ranges);
  LOG(INFO) << "new_nonzero_val = " << new_nonzero_val;

  // This is mostly done to simplify if_then_else which is not known by the Halide simplifier
// TODO: new_expr = RemoveRedundantInequalities(new_expr, res->new_domain->conditions);

  // Keep only those variables of the new vars which are used in the new_expr
  Array<Var> used_res_variables;
  for (const Var& var : res.dst.variables) {
    if (ExprUseVar(new_nonzero_val, var)) {
      used_res_variables.push_back(var);
    }
  }

  // If the expression does not use vars then it is probably better to keep it inlined
  if (used_res_variables.empty()) {
    // We can return the new_expr here instead of the old expr because it doesn't use variables
    // otherwise we would need to replace the new vars or create a let-expression
    return new_nonzero_val;
  }

  // If it's already a call to a tensor then extracting it will probably be useless
  if (const CallNode* call = new_nonzero_val.as<CallNode>()) {
    if (call->call_type == CallNode::CallType::Halide) {
      return nonzero_value;
    }
  }

  arith::Analyzer analyzer;
  // Compute volumes before and after
  PrimExpr old_volume = make_const(DataType::Int(64), 1);
  for (const Var& var : outer_axis) {
    old_volume = old_volume * vrange[var]->extent;
    analyzer.Bind(var, vrange[var]);
  }

  PrimExpr new_volume = make_const(DataType::Int(64), 1);
  for (const Var& var : used_res_variables) {
    new_volume = new_volume * res.dst.ranges[var]->extent;
  }

  // if we can prove that the old volume is not greater than the new volume then
  // prefer the old expression.
  LOG(INFO) << "old volume = " << old_volume;
  LOG(INFO) << "new volume = " << new_volume;
  if (analyzer.CanProve(old_volume <= new_volume)) {
    return nonzero_value;
  }

  // new_nonzero_val indicates how to compute the tensor (using new variables)
  Tensor tensor = TensorFromExpr(
      new_nonzero_val,
      IterVarsFromMap(used_res_variables, res.dst.ranges),
      "extracted_tensor", /*tag=*/"", /*attr=*/{}, /*clone_axis=*/true);

  // access the new tensor by old variables (so that we can inline?)
  Array<PrimExpr> args;
  for (const Var& var : used_res_variables) {
    args.push_back(res.dst_to_src[var]);
  }

  return CallNode::make(new_nonzero_val.dtype(), tensor->op->name, args,
                        CallNode::CallType::Halide, tensor->op, tensor->value_index);
}

// Extract from cond an implication of cond not containing vars
std::pair<PrimExpr, PrimExpr> ImplicationNotContainingVars(
    const PrimExpr& cond, const std::unordered_set<const VarNode*>& vars) {
  CHECK(cond.dtype().is_bool()) << "The type of cond must be bool";
  // TODO(sgrechanik-h): NotNode
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
  } else if (!ExprUseVar(cond, vars)) {
    return {cond, const_true()};
  } else {
    return {const_true(), cond};
  }
}

// Factor conditions out of a reduction by applying Fourier-Motzkin elimination and moving out
// (in)equalities which do not depend on the reduction variables.
std::pair<PrimExpr, PrimExpr> LiftConditionsThroughReduction(
    const PrimExpr& cond, const Array<IterVar>& red_axis, const Array<IterVar>& outer_axis) {
  // Factor out atomics so that we can consider this as a system of inequalities
  auto factoratomic_res = FactorOutAtomicFormulasFunctor()(cond);
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
  LOG(INFO) << "to solve inequality = " << atomics << " vars = " << allvars << " range = " << vranges;
  atomics = SolveSystemOfInequalities(atomics, allvars, vranges).as_conditions();
  LOG(INFO) << "solved atomics =  " << atomics;

  // Append the rest part
  PrimExpr rewritten_cond = All(atomics) && rest;

  std::unordered_set<const VarNode*> vset;
  for (const IterVar& v : red_axis) {
    vset.insert(v->var.get());
  }

  // The outer (first) condition does not contain reduction vars,
  // the inner (second) condition is everything else
  auto res = ImplicationNotContainingVars(rewritten_cond, vset);
  return res;
}

Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor) {
  const ComputeOpNode* op = tensor->op.as<ComputeOpNode>();
  CHECK(op);
  PrimExpr body = op->body[tensor->value_index];
  Array<IterVar> axis = op->axis;

  LOG(INFO) << "body = " << body;

  Map<Var, Range> vrange;
  Array<Var> outer_axis;
  for (const IterVar& iter : axis) {
    vrange.Set(iter->var, iter->dom);
    outer_axis.push_back(iter->var);
  }
  body = SuperSimplify(body, vrange);

  LOG(INFO) << "body (simplified) = " << body;

  PrimExpr new_body;
  if (const ReduceNode* red = body.as<ReduceNode>()) {
    // TODO(sgrechanik-h): There are some other operations which behave like sum
    bool is_sum = IsSumCombiner(red->combiner, vrange);
    if (is_sum || CanFactorZeroFromCombiner(red->combiner, red->value_index, vrange)) {
      PrimExpr new_red = body;

      // Here we simplify the reduction
      {
        PrimExpr cond = red->condition;
        Array<PrimExpr> source = red->source;

        // If it is a summation then we can lift nonzeroness conditions from the source
        // and add them to the reduction conditions
        if (is_sum) {
          // Lift condition out
          auto nz = NonzeronessConditionFunctor()(red->source[red->value_index]);
          LOG(INFO) << "nz = " << nz.to_expr() << " original cond = " << cond;
          cond = nz.cond && cond;
          source.Set(0, nz.value);
        }

        // condition
        new_red = ReduceNode::make(red->combiner, source, red->axis, cond, red->value_index);
        new_red = SimplifyReductionDomain(new_red, vrange);
        LOG(INFO) << "old reduction = " << body;
        LOG(INFO) << "old 2nd reduction = " << ReduceNode::make(red->combiner, source, red->axis, cond, red->value_index);
        LOG(INFO) << "new reduction = " << new_red;
        red = new_red.as<ReduceNode>();

        // If the reduction disappears completely then transform the result as a non-reduction
        CHECK(red);
//        if (!red) {
//          return OptimizeAndLiftNonzeronessConditionsImpl(new_red, axis, vrange);
//        }
        // fixme:
        new_body = new_red;
      }

      PrimExpr new_outer_cond, new_reduce_cond;
      Array<PrimExpr> new_source = red->source;

      // Partially lift conditions from the reduce condition
      std::tie(new_outer_cond, new_reduce_cond) =
        LiftConditionsThroughReduction(red->condition, red->axis, axis);

      // FIXME:
      CHECK(is_sum);
      // If it's not sum then we haven't yet lifted nonzeroness cond from the source
//      if (!is_sum) {
//        Expr outer_nz_cond, nz_cond, nz_source;
//        auto nz = NonzeronessCondition(red->source[red->value_index]);
//        // Append conditions from the reduction
//        nz_cond = new_reduce_cond && nz.cond;
//        nz_source = nz.value;
//        std::tie(outer_nz_cond, nz_cond) =
//          LiftConditionsThroughReduction(nz_cond, red->axis, axis);
//        new_outer_cond = new_outer_cond && outer_nz_cond;
//        new_source.Set(red->value_index, SelectElseZero(nz_cond, nz_source));
//      }
//
      LOG(INFO) << "outer_cond = " << new_outer_cond << " reduce_cond = " << new_reduce_cond;
      PrimExpr new_reduce = ReduceNode::make(
          red->combiner, new_source, red->axis,
          new_reduce_cond, red->value_index);
      // solve outer_cond and replace new_reduce with new varibles
      new_reduce = ExtractAsTensorMaybe(new_reduce, new_outer_cond,
                                        IterVarsToVars(axis),
                                        vrange);
      new_body = SelectNode::make(new_outer_cond, new_reduce, make_zero(new_reduce.dtype()));
    } else {
      LOG(FATAL) << "TODO: body is reduction (else).";
      // TODO: return ZE_LOG_RES(SimplifyReductionDomain(expr, combined_vranges));
    }


  } else {
    auto nz = NonzeronessConditionFunctor().VisitExpr(body);
    LOG(INFO) << "non zero condition = " << nz.to_expr();
    PrimExpr new_expr = ExtractAsTensorMaybe(nz.value, nz.cond, outer_axis, vrange);
    new_body = SelectNode::make(nz.cond, new_expr, make_zero(new_expr.dtype()));

    /* TODO: folowing I tried to replace 4 loop axis by 2 (i, j, jac_i, jac_j) -> (n0, n1) and eliminate SelectNode
     * (by replacing op->axis with test_axis), however it results,
     * [23:46:33] /home/ubuntu/tvm/src/te/zero_elimination.cc:1873: TEST src_to_dst = {j: (9 - n1), jac_i0: (9 - n0), i: (9 - n0), jac_i1: (9 - n1)}
       [23:46:33] /home/ubuntu/tvm/src/te/zero_elimination.cc:1874: TEST dst_to_src = {n1: (9 - jac_i1), n0: (9 - jac_i0)}
       [23:46:33] /home/ubuntu/tvm/src/te/zero_elimination.cc:1875: TEST dst.ranges = {n0: range(min=0, ext=10), n1: range(min=0, ext=10)}
// attr [compute(B.jacobian, 0x16cdc40)] realize_scope = ""
realize B.jacobian([0, 10], [0, 10]) {
  produce B.jacobian {
    for (n0, 0, 10) {
      for (n1, 0, 10) {
        B.jacobian(n0, n1) =1f
      }
    }
  }
}
     aka. it not only eliminate for loops, but also the tensor dimension.

    Array<PrimExpr> atomic_cond = FactorOutAtomicFormulasFunctor().VisitExpr(nz.cond).to_array();
    LinearSystem system_to_solve;
    system_to_solve.variables = outer_axis;
    system_to_solve.ranges = vrange;
    // conditions leads to non-zero output of grad.
    system_to_solve.relations = atomic_cond;

    LinearSystemTransform solution_eq = SolveSystemOfEquations(system_to_solve);
    LinearSystemTransform res = DeskewDomain(solution_eq.dst);
    res = ComposeLinearTransform(solution_eq, res);

    LOG(INFO) << "TEST src_to_dst = " << res.src_to_dst;
    LOG(INFO) << "TEST dst_to_src = " << res.dst_to_src;
    LOG(INFO) << "TEST dst.ranges = " << res.dst.ranges;
    PrimExpr new_nonzero_val = SuperSimplify(Substitute(nz.value, res.src_to_dst), res.dst.ranges);

    Array<IterVar> test_axis;
    for (const Var& var: res.dst.variables) {
      Range dom(0, res.dst.ranges[var]->extent);
      test_axis.push_back(IterVarNode::make(dom, var, IterVarType::kDataPar));
    }
    return TensorFromExpr(new_nonzero_val, test_axis, op->name, op->tag, op->attrs, true);
    */
  }

  if (new_body.same_as(body)) {
    return tensor;
  }
  return TensorFromExpr(new_body, op->axis, op->name, op->tag, op->attrs, /*clone_axis=*/true);
}

}  // namespace te
}  // namespace tvm
