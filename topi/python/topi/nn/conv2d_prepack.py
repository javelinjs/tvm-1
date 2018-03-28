# pylint: disable=invalid-name, unused-variable, too-many-locals, unused-argument
"""Conv2D (kernel prepack) operators"""
from __future__ import absolute_import as _abs
import tvm

@tvm.target.generic_func
def conv2d_nChwc(data, kernel, num_filter, kernel_size, stride, padding, out_dtype='float32'):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    num_filter : int
        number of filters, i.e., output channel size

    kernel_size : tuple of two ints
        [kernel_height, kernel_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    # search platform specific declaration first
    # default declaration
    raise ValueError("missing register for topi.nn.conv2d_nChwc")


@tvm.target.generic_func
def _contrib_conv2d_nchwc_kernel_packed(data, kernel, num_filter, kernel_size, stride, padding, out_dtype='float32'):
    """Conv2D operator for NCHW layout with kernel pack.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    num_filter : int
        number of filters, i.e., output channel size

    kernel_size : tuple of two ints
        [kernel_height, kernel_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    raise ValueError("missing register for topi.nn._contrib_conv2d_nchwc_kernel_packed")
