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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The layout nodes of the Relay language."""

from ..api import layout as _create_tvm_layout
from .base import RelayNode, register_relay_node
from . import _make

class RelayLayout(RelayNode):
    """The base type for all Relay layouts."""

    # def __eq__(self, other):
    #     """Compare two Relay types for structural equivalence using
    #        alpha equivalence.
    #     """
    #     return bool(_make._alpha_equal(self, other))
    #
    # def __ne__(self, other):
    #     return not self.__eq__(other)

    # def same_as(self, other):
    #     """Compares two Relay types by referential equality."""
    #     return super().__eq__(other)


@register_relay_node
class TensorLayout(RelayLayout):
    """A concrete TensorLayout in Relay.

    This is the layout assigned to a tensor

    Parameters
    ----------
    layout : tvm.tensor.Layout
        The layout of the Tensor

    Returns
    -------
    tensor_layout : tvm.relay.TensorLayout
        The tensor layout.
    """
    def __init__(self, layout):
        self.__init_handle_by_constructor__(
            _make.TensorLayout, layout)


@register_relay_node
class TupleLayout(RelayLayout):
    """A tuple layout in Relay.

    Lists the layout of each field in the tuple.
    """

    def __init__(self, fields):
        """Constructs a tuple layout

        Parameters
        ----------
        fields : List[tvm.tensor.Layout]
            The fields in the tuple

        Returns
        -------
        tuple_layout : tvm.relay.TupleLayout
            the tuple layout
        """
        layouts = [_create_tvm_layout(l) if isinstance(l, str) else l for l in fields]
        self.__init_handle_by_constructor__(_make.TupleLayout, layouts)