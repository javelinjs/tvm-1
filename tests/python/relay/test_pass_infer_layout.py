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

class NameInferredLayout(ExprVisitor):
    def __init__(self, layout_map):
        super(NameInferredLayout, self).__init__()
        self.layouts = layout_map
        self.named_layouts = {}

    def to_list(self, arr):
        l = []
        for i in range(len(arr)):
            if arr[i]:
                l.append(arr[i][:])
            else:
                l.append(arr[i])
        return l

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)
        if isinstance(call.op, tvm.relay.expr.GlobalVar):
            self.named_layouts[call.op.name_hint] = self.to_list(self.layouts[call])
        else:
            self.named_layouts[call.op.name] = self.to_list(self.layouts[call])

    def visit_var(self, var):
        self.named_layouts[var.name_hint] = self.to_list(self.layouts[var])


def run_infer_layout(mod, layouts={}):
    if not isinstance(mod, relay.Module):
        func = relay.Function(analysis.free_vars(mod), mod)
        mod = relay.Module.from_expr(func)
    mod = transform.InferType()(mod)
    layout_map = collect_layout(mod, layouts)
    namer = NameInferredLayout(layout_map)
    for func in mod.get_global_vars():
        namer.visit(mod[func])
    return namer.named_layouts


def test_relu():
    x = relay.var('x', shape=(1, 64, 56, 56))
    x = relay.nn.relu(x)
    y = relay.var("y", shape=(1, 1, 56, 56))
    z = x + y
    layouts = run_infer_layout(z, layouts={"x": "NCHW"})

    assert layouts["x"][0][:] == "NCHW"
    assert layouts["x"] == ["NCHW"]
    assert layouts["y"] == ["NCHW"]
    assert layouts["nn.relu"] == ["NCHW"]
    assert layouts["add"] == ["NCHW"]


def test_conv2d_broadcast():
    x = relay.var("x", shape=(1, 64, 56, 56))
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

    bias = relay.var("bias", shape=(64, 1, 1))
    z = relay.add(y, bias)
    layouts = run_infer_layout(z, layouts={"x": "NCHW"})

    assert layouts["x"] == ["NCHW"]
    assert layouts["weight"] == ["OIHW"]
    assert layouts["bias"] == ["CHW"]
    assert layouts["add"] == ["NCHW"]
    assert layouts["nn.conv2d"] == ["NCHW"]


def test_reverse_infer():
    x = relay.var('x', shape=(1, 64, 56, 56))
    x = relay.nn.relu(x)
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
    layouts = run_infer_layout(y)

    assert layouts["x"] == ["NCHW"]
    assert layouts["weight"] == ["OIHW"]
    assert layouts["nn.conv2d"] == ["NCHW"]
    assert layouts["nn.relu"] == ["NCHW"]


def test_global_var_recursion():
    mod = relay.Module({})
    gv = relay.GlobalVar("main")
    x = relay.var('x', shape=[])
    tt = relay.scalar_type('float32')

    func = relay.Function([x], relay.Call(gv, [x]), tt)
    mod[gv] = func
    layouts = run_infer_layout(mod)
    # TODO: should it be undef?
    assert layouts["main"] == [None]
    assert layouts["x"] == [None]


def test_multi_func():
    mod = relay.Module({})

    x = relay.var('x', shape=(1, 64, 56, 56))
    r = relay.nn.relu(x)
    func_relu = relay.Function([x], r)

    gv_relu = relay.GlobalVar("relu")
    mod[gv_relu] = func_relu

    y = relay.var("y", shape=(1, 1, 56, 56))
    z = gv_relu + y
    func = relay.Function([x, y], z, relay.TensorType((1, 64, 56, 56)))
    mod[relay.GlobalVar("main")] = func

    # print(mod.astext())
    layouts = run_infer_layout(mod, layouts={"x": "NCHW"})

    # first function
    assert layouts["x"] == ["NCHW"]
    assert layouts["nn.relu"] == ["NCHW"]
    # second function
    assert layouts["y"] == ["NCHW"]
    assert layouts["add"] == ["NCHW"]


if __name__ == "__main__":
    test_relu()
    test_conv2d_broadcast()
    test_reverse_infer()
    test_global_var_recursion()
    test_multi_func()
