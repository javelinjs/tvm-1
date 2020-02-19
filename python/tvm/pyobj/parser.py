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
"""Parse Python class
"""

import ast
import sys


class PyObjectMutator(ast.NodeTransformer):
    def __init__(self, cxx_cons, args):
        self.cxx_cons = cxx_cons
        self.args = args

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            # e.g.,
            # self.__init_handle_by_constructor__(_api_internal._Var, name)
            invoke_func = ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="__init_handle_by_constructor__",
                ctx=ast.Load()
            )
            args = [
                ast.Attribute(
                    value=ast.Name(id="_api_internal", ctx=ast.Load()),
                    attr=self.cxx_cons,
                    ctx=ast.Load()
                ),
                ast.Name(
                    id="name",
                    ctx=ast.Load()
                )
            ]
            call = ast.Call(func=invoke_func, args=args, keywords=[])
            node.body = [
                ast.Expr(value=call)
            ]
        return node


class PyObjectParser(ast.NodeVisitor):
    def __init__(self):
        self.cls_name = None
        self.cls_args = []
        self.fields = []
        self.base_cls = []
        self.namespace = ""

    def _get_cxx_type(self, name, annotation):
        # TODO
        _type_map = {"str": "std::string"}
        return _type_map[annotation.id]

    def visit_Module(self, node):
        assert len(node.body) == 1, "Only one-function source code will be fed to this parser!"
        self.visit(node.body[0])

    def visit_ClassDef(self, node):
        self.cls_name = node.name
        assert len(node.bases) == 1, "Only one base class is supported."
        self.base_cls.append(node.bases[0].id)

        for func in node.body:
            self.visit(func)

    def visit_Name(self, node):
        return node.id

    def visit_Attribute(self, node):
        name = self.visit(node.value)
        assert isinstance(node.attr, str)
        return name, node.attr

    def visit_Call(self, node):
        func = node.func
        func_name = self.visit(func)
        if func_name == ("self", "__init_handle_by_constructor__"):
            assert len(node.args) >= 1
            self.namespace, obj_name = self.visit(node.args[0])

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            for idx, arg in enumerate(node.args.args):
                _attr = "id" if sys.version_info[0] < 3 else "arg"  # To make py2 and 3 compatible
                arg_name = getattr(arg, _attr)
                if arg_name != "self":
                    arg_type = self._get_cxx_type(arg_name, arg.annotation)
                    self.cls_args.append((arg_name, arg_type))
                    # also add to fields
                    self.fields.append((arg_name, arg_type))
            for body in node.body:
                self.visit(body)

    def gen_arg_list(self, with_type=True):
        ret = "("
        for i in range(len(self.cls_args)):
            arg_name, arg_type = self.cls_args[i]
            if with_type:
                ret += f"{arg_type} "
            ret += arg_name
            ret += "" if i == len(self.cls_args) - 1 else ", "
        ret += ")"
        return ret

    def gen_header(self):
        # Node
        ret = f"class {self.cls_name}Node "
        if len(self.base_cls) > 0:
            ret += f": public {self.base_cls[0]}Node "
        ret += "{\n"
        ret += " public:\n"
        ret += f"  {self.cls_name}Node() {{}}\n"

        if len(self.cls_args) > 0:
            ret += f"  {self.cls_name}Node"
            ret += self.gen_arg_list()
            ret += ";\n"

        ret += "\n"
        for field, field_type in self.fields:
            ret += f"  {field_type} {field};\n"
        ret += "\n"

        # visitors
        ret += "  void VisitAttrs(AttrVisitor* v) {\n"
        for field, field_type in self.fields:
            ret += f"    v->Visit(\"{field}\", &{field});\n"
        ret += "  }\n\n"

        ret += f"  static constexpr const char* _type_key = \"{self.cls_name}\";\n"
        # TODO: decide whether FINAL or not
        ret += f"  TVM_DECLARE_FINAL_OBJECT_INFO({self.cls_name}Node, {self.base_cls[0]}Node);\n"
        ret += "};\n"

        # Ref
        ret += "\n"
        ret += f"class {self.cls_name} "
        if len(self.base_cls) > 0:
            ret += f": public {self.base_cls[0]} "
        ret += "{\n"
        ret += " public:\n"
        ret += f"  explicit {self.cls_name}(ObjectPtr<Object> n) : {self.base_cls[0]}(n) {{}}\n"
        if len(self.cls_args) > 0:
            ret += f"  TVM_DLL explicit {self.cls_name}"
            ret += self.gen_arg_list()
            ret += ";\n\n"

        ret += f"  const {self.cls_name}Node* operator->() const {{\n"
        ret += "    return get();\n"
        ret += "  }\n\n"

        ret += f"  const {self.cls_name}Node* get() const {{\n"
        ret += f"    return static_cast<const {self.cls_name}Node*>(data_.get());\n"
        ret += "  }\n\n"

        ret += f"  using ContainerType = {self.cls_name}Node;\n"
        ret += "};\n"

        return ret

    def gen_cc(self):
        ret = ""
        if len(self.cls_args) > 0:
            ret += f"{self.cls_name}Node::{self.cls_name}Node"
            ret += self.gen_arg_list()
            ret += " {\n"
            for field_name, field_type in self.cls_args:
                # TODO: assign expr
                assign_expr = f"std::move({field_name})" if field_type == "std::string" else field_name
                ret += f"  this->{field_name} = {assign_expr};\n"
            ret += "}\n"

        if len(self.cls_args) > 0:
            ret += "\n"
            ret += f"{self.cls_name}::{self.cls_name}"
            ret += self.gen_arg_list()
            ret += f"\n    : {self.cls_name}(make_object<{self.cls_name}Node>" \
                f"{self.gen_arg_list(with_type=False)}) {{}}\n"

        ret += "\n"
        ret += f"TVM_REGISTER_NODE_TYPE({self.cls_name}Node);\n"
        return ret

    def gen_api(self):
        # TODO: name
        ret = f"TVM_REGISTER_GLOBAL(\"_{self.cls_name}\")\n"
        ret += f".set_body_typed([]{self.gen_arg_list()} {{\n"
        ret += f"  return {self.cls_name}{self.gen_arg_list(with_type=False)};\n"
        ret += "});\n"
        return ret


def header_wrapper(objects):
    ret = "#ifndef TVM_TIR_EXPR_EXT_H_\n"
    ret += "#define TVM_TIR_EXPR_EXT_H_\n\n"

    ret += "#include <tvm/node/node.h>\n"
    ret += "#include <tvm/node/container.h>\n"
    ret += "#include <tvm/node/functor.h>\n"
    ret += "#include <tvm/runtime/c_runtime_api.h>\n"
    ret += "#include <tvm/runtime/data_type.h>\n"
    ret += "#include <tvm/ir/expr.h>\n"
    ret += "#include <string>\n"
    ret += "#include <algorithm>\n"
    ret += "#include <unordered_map>\n"
    ret += "#include <iostream>\n"
    ret += "#include <limits>\n"
    ret += "#include <utility>\n\n"

    ret += "namespace tvm {\n"
    ret += "namespace tir {\n\n"

    for obj in objects:
        ret += obj

    ret += "\n"
    ret += "}  // namespace tir\n"
    ret += "}  // namespace tvm\n"
    ret += "#endif  // TVM_TIR_EXPR_EXT_H_\n"

    return ret


def cc_wrapper(bodies):
    ret = "#include <tvm/tir/expr_ext.h>\n"

    ret += "\n"
    ret += "namespace tvm {\n"
    ret += "namespace tir {\n\n"

    for body in bodies:
        ret += body

    ret += "\n"
    ret += "}  // namespace tir\n"
    ret += "}  // namespace tvm\n"
    return ret


def api_wrapper(apis):
    ret = "#include <tvm/tir/expr.h>\n"
    # TODO: include file
    ret += "#include <tvm/tir/expr_ext.h>\n"
    ret += "#include <tvm/runtime/registry.h>\n"
    ret += "#include <tvm/tir/op.h>\n\n"

    ret += "namespace tvm {\n"
    ret += "namespace tir {\n\n"

    for api in apis:
        ret += api

    ret += "\n"
    ret += "}  // namespace tir\n"
    ret += "}  // namespace tvm\n"
    return ret


def parse_py_class(src):  # , args, symbols, closure_vars):
    """The helper function of calling the AST visitor

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    symbols : list of str
        The symbol list of the global context of the function.

    closure_vars: dict
        A dict of external name reference captured by this function.

    Returns
    -------
    root : Stmt
        The result Halide IR and the parser class instance.
    """
    root = ast.parse(src) if isinstance(src, str) else src
    root_str = ast.dump(root)
    # print(root_str, "\n")
    parser = PyObjectParser()
    parser.visit(root)

    with open("include/tvm/tir/expr_ext.h", "w") as fheader:
        fheader.write(header_wrapper([parser.gen_header()]))
        fheader.write("\n")

    with open("src/tir/ir/expr_ext.cc", "w") as fcc:
        fcc.write(cc_wrapper([parser.gen_cc()]))
        fcc.write("\n")

    with open("src/api/api_ir_ext.cc", "w") as fapi:
        fapi.write(api_wrapper([parser.gen_api()]))
        fapi.write("\n")

    # TODO: this is a hack
    bashCommand = "cd build && cmake .. && make -j18"
    import subprocess
    process = subprocess.call(bashCommand, shell=True)
    if process != 0:
        msg = "Pompilation error:\n"
        raise RuntimeError(msg)

    # mutator = PyObjectMutator("_XVar", ["name"])
    # new_cls = mutator.visit(root)
    #
    # new_cls.body = [
    #     ast.Import(names=[ast.alias(name='tvm', asname=None)]),
    #     ast.ImportFrom(module='tvm', names=[ast.alias(name='_api_internal', asname=None)], level=0),
    #     ast.ImportFrom(module='tvm.expr', names=[ast.alias(name='PrimExpr', asname=None)], level=0)
    # ] + new_cls.body
    # ast.fix_missing_locations(new_cls)
    #
    # import astunparse
    # print(astunparse.dump(new_cls))
    # print(astunparse.unparse(new_cls))
    #
    # new_mod = compile(new_cls, filename="<ast>", mode="exec")
    # eval(new_mod)
    # print(eval("XVar"))
    # return eval("XVar")

    # _internal_assert(root, ast.AST)
    # var_usage = determine_variable_usage(root, args, symbols, closure_vars)
    # parser = HybridParser(args, var_usage, symbols, closure_vars)
    # parser.parsed_body = parser.visit(root)
    # _internal_assert(parser.returned, 'No valid return found in the function body!')
    # return parser