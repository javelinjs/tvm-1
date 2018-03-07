# pylint: disable=invalid-name, unused-variable, too-many-locals, unused-argument
"""Conv2D (kernel prepack) operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify

@tvm.target.generic_func
def conv2d_prepack(data, kernel, kernel_size, stride, padding, layout='NCHW', out_dtype='float32'):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    raise ValueError("missing register for topi.nn.conv2d_prepack")

@tvm.target.generic_func
def conv2d_nopack(data, kernel, kernel_size, stride, padding, layout='NCHWc', out_dtype='float32'):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    raise ValueError("missing register for topi.nn.conv2d_prepack")
