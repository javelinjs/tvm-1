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
"""Test infer layout pass"""
import tvm

from tvm import relay
from tvm.relay import transform, analysis
from tvm.relay.analysis import collect_layout
from tvm.relay.expr_functor import ExprVisitor


class AssertInferredLayout(ExprVisitor):
    def __init__(self, assert_func):
        super(AssertInferredLayout, self).__init__()
        self.assert_func = assert_func
        self.nodes = []

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)
        self.assert_func(call)
        self.nodes.append(call.op.name)

    def visit_var(self, var):
        self.assert_func(var)
        self.nodes.append(var.name_hint)


def print_dict(layouts):
    for k, v in layouts.items():
        print(k, "|:", v)
        print()


def run_infer_layout(expr, layouts={}):
    expr = relay.Function(analysis.free_vars(expr), expr)
    mod = relay.Module.from_expr(expr)
    mod = transform.InferType()(mod)
    f = mod["main"]
    layout_map = collect_layout(mod, layouts)
    return f, layout_map


def test_relu():
    x = relay.var('x', shape=(1, 64, 56, 56))
    x = relay.nn.relu(x)
    y = relay.var("y", shape=(1, 1, 56, 56))
    z = x + y
    f, layouts = run_infer_layout(z, layouts={"x": tvm.layout("NCHW")})
    # check results
    def assert_func(node):
        node_layout = layouts[node]
        assert len(node_layout) == 1
        assert node_layout[0][:] == "NCHW"
    checker = AssertInferredLayout(assert_func)
    checker.visit(f)
    assert set(checker.nodes) == set(["x", "y", "nn.relu", "add"]), checker.nodes


def test_conv2d_broadcast():
    x = relay.var("x", shape=(1, 64, 56, 56))
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

    bias = relay.var("bias", shape=(64, 1, 1))
    z = relay.add(y, bias)
    f, layouts = run_infer_layout(z, layouts={"x": "NCHW"})

    # check results
    def assert_func(node):
        node_layout = layouts[node]
        assert len(node_layout) == 1
        node_layout = node_layout[0][:]
        if isinstance(node, tvm.relay.expr.Var):
            if node.name_hint == "x":
                assert node_layout == "NCHW"
            elif node.name_hint == "weight":
                assert node_layout == "OIHW"
            elif node.name_hint == "bias":
                assert node_layout == "CHW"
        elif isinstance(node, tvm.relay.expr.Call):
            # conv2d
            assert node_layout == "NCHW"
    checker = AssertInferredLayout(assert_func)
    checker.visit(f)
    assert set(checker.nodes) == set(["x", "weight", "bias", "add", "nn.conv2d"]), checker.nodes


def test_reverse_infer():
    x = relay.var('x', shape=(1, 64, 56, 56))
    x = relay.nn.relu(x)
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
    f, layouts = run_infer_layout(y)

    # check results
    def assert_func(node):
        node_layout = layouts[node]
        assert len(node_layout) == 1
        node_layout = node_layout[0][:]
        if isinstance(node, tvm.relay.expr.Var):
            if node.name_hint == "x":
                assert node_layout == "NCHW"
            elif node.name_hint == "weight":
                assert node_layout == "OIHW"
        elif isinstance(node, tvm.relay.expr.Call):
            assert node_layout == "NCHW"
    checker = AssertInferredLayout(assert_func)
    checker.visit(f)
    assert set(checker.nodes) == set(["x", "weight", "nn.relu", "nn.conv2d"]), checker.nodes


def test_global_var_recursion():
    mod = relay.Module({})
    gv = relay.GlobalVar("main")
    x = relay.var('x', shape=[])
    tt = relay.scalar_type('float32')

    func = relay.Function([x], relay.Call(gv, [x]), tt)
    mod[gv] = func

    # ft = run_infer_type(gv, mod)
    # assert ft.checked_type == relay.FuncType([tt], tt)


if __name__ == "__main__":
    # test_relu()
    test_conv2d_broadcast()
    # test_reverse_infer()
