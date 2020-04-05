# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import random
import numpy as np
import tvm
from tvm import te, arith, ir, tir


def run_expr(expr, vranges):
    def _compute_body(*us):
        vmap = {v: u + r.min for (v, r), u in zip(vranges.items(), us)}
        return tir.ir_pass.Substitute(expr, vmap)

    A = te.compute([r.extent.value for v, r in vranges.items()], _compute_body)
    args = [tvm.nd.empty(A.shape, A.dtype)]
    sch = te.create_schedule(A.op)
    mod = tvm.build(sch, [A])
    mod(*args)
    return args[0].asnumpy()


def check_bruteforce(bool_expr, vranges, cond=None):
    if cond is not None:
        bool_expr = te.any(tir.Not(cond), bool_expr)

    res = run_expr(bool_expr, vranges)
    if not np.all(res):
        indices = list(np.argwhere(res == 0)[0])
        counterex = [(str(v), i + r.min) for (v, r), i in zip(vranges.items(), indices)]
        counterex = sorted(counterex, key=lambda x: x[0])
        counterex = ", ".join([v + " = " + str(i) for v, i in counterex])
        raise AssertionError("Expression {}\nis not true on {}\n"
                             "Counterexample: {}"
                             .format(tir.ir_pass.CanonicalSimplify(bool_expr), vranges, counterex))


def check_solution(solution, vranges={}):
    def _check_forward(formula1, formula2, varmap, backvarmap):
        all_vranges = vranges.copy()
        all_vranges.update({v: r for v, r in formula1.ranges.items()})

        # Check that the transformation is injective
        cond_on_vars = tir.const(1, 'bool')
        for v in formula1.variables:
            # variable mapping is consistent
            v_back = tir.ir_pass.Simplify(tir.ir_pass.Substitute(varmap[v], backvarmap))
            cond_on_vars = te.all(cond_on_vars, v == v_back)
        # Also we have to check that the new relations are true when old relations are true
        cond_subst = tir.ir_pass.Substitute(
            te.all(tir.const(1, 'bool'), *formula2.relations), backvarmap)
        # We have to include relations from vranges too
        for v in formula2.variables:
            if v in formula2.ranges:
                r = formula2.ranges[v]
                range_cond = te.all(v >= r.min, v < r.min + r.extent)
                range_cond = tir.ir_pass.Substitute(range_cond, backvarmap)
                cond_subst = te.all(cond_subst, range_cond)
        cond_subst = tir.ir_pass.Simplify(cond_subst)
        check_bruteforce(te.all(cond_subst, cond_on_vars), all_vranges,
                         cond=te.all(tir.const(1, 'bool'), *formula1.relations))

    rels = solution.dst.relations
    if len(rels) == 1 and ir.structural_equal(rels[0], False):
        # not solvable, skip
        return
    _check_forward(solution.src, solution.dst,
                   solution.src_to_dst, solution.dst_to_src)
    _check_forward(solution.dst, solution.src,
                   solution.dst_to_src, solution.src_to_dst)


def test_solution_consistency():
    random.seed(0)

    def _check(num_vars, num_formulas, coef=(-5, 5), bounds=(-20, 20)):
        variables = [te.var("x" + str(i)) for i in range(num_vars)]

        relations = []
        for i in range(num_formulas):
            s1 = sum([v*random.randint(coef[0], coef[1]) for v in variables])
            s1 += random.randint(coef[0], coef[1])
            s2 = sum([v*random.randint(coef[0], coef[1]) for v in variables])
            s2 += random.randint(coef[0], coef[1])
            if random.random() < 0.7:
                op = tvm.tir.EQ
            else:
                # we also make sure it can correctly handle inequalities
                op = random.choice([tvm.tir.LE, tvm.tir.LT, tvm.tir.GE, tvm.tir.GT])
            relations.append(op(s1, s2))

        vranges = {v: tvm.ir.expr.Range(bounds[0], bounds[1] + 1) for v in variables}
        solution = arith.solve_linear_equations(relations, variables, vranges)

        check_solution(solution)

        # leaving some variables as parameters should also be ok
        for k in [1, 2]:
            if len(variables) > k:
                solution = arith.solve_linear_equations(relations, variables[:-k], vranges)
                param_ranges = {v: vranges[v] for v in variables[-k:]}
                check_solution(solution, param_ranges)

    for i in range(2):
        _check(num_vars=1, num_formulas=1)
    for i in range(2):
        _check(num_vars=1, num_formulas=2)

    for i in range(2):
        _check(num_vars=2, num_formulas=1)
    for i in range(2):
        _check(num_vars=2, num_formulas=2)
    for i in range(2):
        _check(num_vars=2, num_formulas=3)

    for i in range(3):
        _check(num_vars=3, num_formulas=3, coef=(-2, 2))
    for i in range(3):
        _check(num_vars=3, num_formulas=4, coef=(-2, 2))

    for i in range(3):
        _check(num_vars=4, num_formulas=3, coef=(-1, 1))

    for i in range(3):
        _check(num_vars=10, num_formulas=2, coef=(-1, 1), bounds=(0, 4))
    for i in range(3):
        _check(num_vars=10, num_formulas=3, coef=(0, 1), bounds=(0, 4))


def test_unique_solution():
    x, y = te.var("x"), te.var("y")
    ranges = {}

    solution = arith.solve_linear_equations([
        tvm.tir.EQ(x + y, 20),
        tvm.tir.EQ(x - y, 10),
    ], [x, y], ranges)
    assert list(solution.dst.variables) == []
    assert ir.structural_equal(solution.src_to_dst[x], 15)
    assert ir.structural_equal(solution.src_to_dst[y], 5)


def test_low_rank():
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    ranges = {}

    solution = arith.solve_linear_equations([
        tvm.tir.EQ(x + y + z, 15),
        tvm.tir.EQ(x + y, 10),
    ], [x, y, z], ranges)
    [n0] = solution.dst.variables
    assert ir.structural_equal(solution.src_to_dst[x], n0 + 10)
    assert ir.structural_equal(solution.src_to_dst[y], -n0)
    assert ir.structural_equal(solution.src_to_dst[z], 5)


def test_infer_range():
    x, y = te.var("x"), te.var("y")
    ranges = {
        x: tvm.ir.Range.make_by_min_extent(-5, 10),
        y: tvm.ir.Range.make_by_min_extent(0, 10),
    }

    solution = arith.solve_linear_equations([
        tvm.tir.EQ(x + y, 0),
    ], [x, y], ranges)
    [n0] = solution.dst.variables
    assert ir.structural_equal(solution.src_to_dst[x], n0)
    assert ir.structural_equal(solution.src_to_dst[y], -n0)
    # inferred from y's range
    assert ir.structural_equal(solution.dst.ranges[n0].min, -9)
    assert ir.structural_equal(solution.dst.ranges[n0].extent, 10)
    # additional inequality is added into the system for x
    [ineq] = solution.dst.relations
    assert isinstance(ineq, tvm.tir.LE)
    assert ir.structural_equal(ineq.a, -5)
    assert ir.structural_equal(ineq.b, n0)


def test_ill_formed():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_linear_equations([
        tvm.tir.EQ(x + y, 0),
        tvm.tir.EQ(x - y, 0),
        tvm.tir.EQ(x, 5),
    ], [x, y], {})
    assert list(solution.dst.variables) == []
    [rel] = solution.dst.relations
    assert ir.structural_equal(rel, False)
    assert len(solution.src_to_dst) == 0
    assert len(solution.dst_to_src) == 0


if __name__ == "__main__":
    test_unique_solution()
    test_low_rank()
    test_infer_range()
    test_ill_formed()
    test_solution_consistency()
