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


def run_infer_layout(expr, in_layouts=[]):
    expr = relay.Function(analysis.free_vars(expr), expr)
    mod = relay.Module.from_expr(expr)
    mod = transform.InferType()(mod)
    f = mod["main"]
    f = f if isinstance(f, relay.Function) else f.body
    in_layouts_map = {}
    for i in range(len(in_layouts)):
        in_layouts_map[f.params[i]] = in_layouts[i]
    layout_map = collect_layout(f) #, in_layouts_map)
    return layout_map


def test_unary():
    x = relay.var('x', shape=(2, 2))
    y = relay.nn.relu(x)
    layouts = run_infer_layout(y, [tvm.layout("NCHW")])
    print(layouts)


def test_broadcast():
    x = relay.var('x', shape=(1, 64, 56, 56))
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

    bias = relay.var("bias", shape=(56,))
    z = relay.add(y, bias)
    z = relay.Function(analysis.free_vars(z), z)

    mod = relay.Module.from_expr(z)
    mod = transform.InferType()(mod)
    f = mod[mod.entry_func]
    f = f if isinstance(f, relay.Function) else f.body
    layout_map = collect_layout(f)
    print(f.params)
    print("bias layout = ", layout_map[f.params[2]][0])
    print("haha", layout_map)


def test_conv2d():
    x = relay.var('x', shape=(1, 64, 56, 56))
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
    y = relay.Function(analysis.free_vars(y), y)
    # ttype = relay.TensorType([], dtype='float32')
    # assert_has_type(func, relay.FuncType([ttype], ttype))
    mod = relay.Module.from_expr(y)
    mod = transform.InferType()(mod)
    f = mod[mod.entry_func]
    f = f if isinstance(f, relay.Function) else f.body

    layout_map = collect_layout(f)
    print(layout_map)
    print(f.params)
    print("x layout = ", layout_map[f.params[0]][0])


def test_reverse_infer():
    x = relay.var('x', shape=(1, 64, 56, 56))
    x = relay.nn.relu(x)
    weight = relay.var("weight")
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
    layouts = run_infer_layout(y)
    print("layouts", layouts)


if __name__ == "__main__":
    # test_unary()
    # test_broadcast()
    # test_conv2d()
    test_reverse_infer()
