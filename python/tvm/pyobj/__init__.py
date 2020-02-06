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
"""Compile Python objects to TVM objects
"""

from __future__ import absolute_import as _abs

from .util import _pruned_source
from .parser import parse_py_class
from .._ffi.object import Object, register_object


def py_tvm_object(type_key=None):
    """Decorate a Python class as TVM Object

    The PyTVM Object compiles to internal TVM object

    Returns
    -------
    pyTVM object : class
        A decorated pyTVM object
    """
    object_name = type_key if isinstance(type_key, str) else type_key.__name__

    def register(cls):
        """internal register class"""
        src = _pruned_source(cls)
        # closure_vars = inspect.getclosurevars(func).nonlocals
        # closure_vars.update(inspect.getclosurevars(func).globals)
        # TODO: only do this when compile tvm
        parse_py_class(src)  # , args, func.__globals__, closure_vars)

        return register_object(cls)

    if isinstance(type_key, str):
        return register

    return register(type_key)
