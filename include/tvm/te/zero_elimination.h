#ifndef TVM_TE_ZERO_ELIMINATION_H
#define TVM_TE_ZERO_ELIMINATION_H

#include <tvm/te/tensor.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

/*!
 * \brief A struct representing a set of inequalities describing bounds of a variable.
 *
 *  Given a variable x, this struct represents the following (in)equalities:
 *  - `coef*x >= low` for each `low` in `lower`
 *  - `coef*x == eq` for each `eq` in `equal`
 *  - `coef*x <= upp` for each `upp` in `upper`
 *
 *  Note that every array is supposed to be sorted in the order of increasing expression
 *  complexity.
 */
struct VarBounds {
  PrimExpr coef;
  Array<PrimExpr> lower;
  Array<PrimExpr> equal;
  Array<PrimExpr> upper;

 /*!
  * \brief Perform substitution on all components of the struct.
  */
  VarBounds substitute(const Map<Var, PrimExpr>& subst) const;
};

/*!
 * \brief A struct representing a system of inequalities resulted from Fourier-Motzkin elimination.
 */
struct SolveSystemOfInequalitiesResult {
  Array<Var> variables;
  std::unordered_map<const VarNode*, VarBounds> bounds;
  Array<PrimExpr> other_conditions;

  /*!
   * \brief Combine the information into an array of (in)equalities.
   */
  Array<PrimExpr> as_conditions() const;
};

/*!
 * \brief Perform lifting of conditions of being possible to be non-zero together with
 *  applying some transformations like simplifying the reduction domain. Works only with
 *  this particular tensor's body, i.e. doesn't perform inlining.
 *
 * \param tensor The original tensor;
 * \param vranges Optional map from free variables to their value ranges.  // TODO
 * \return An optimized tensor.
 */
TVM_DLL Tensor OptimizeAndLiftNonzeronessConditions(
    const Tensor& tensor);

// Given a map from vars to ranges create an array of itervars
Array<IterVar> IterVarsFromMap(const Array<Var>& vars, const Map<Var, Range>& vranges,
                               IterVarType iter_type = kDataPar, std::string thread_tag = "");

Operation ComputeOpFromExprs(const Array<PrimExpr>& exprs, const Array<IterVar>& axis,
                             const std::string& name, const std::string& tag,
                             const Map<std::string, ObjectRef>& attrs,
                             bool clone_axis);

Tensor TensorFromExpr(const PrimExpr& expr, const Array<IterVar>& axis,
                      const std::string& name, const std::string& tag,
                      const Map<std::string, ObjectRef>& attrs,
                      bool clone_axis);

}  // namespace te
}  // namespace tvm

#endif //TVM_TE_ZERO_ELIMINATION_H
