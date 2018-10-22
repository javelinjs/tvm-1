# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Conv2D schedule on x86"""
from collections import namedtuple
import tvm
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride, get_const_int, get_pad_tuple
from ..util import get_const_tuple
from ..nn.depthwise_conv2d import depthwise_conv2d_nchw, depthwise_conv2d_NCHWc
from ..nn.pad import pad

from .check_targets import check_skylake

AVXDepthwiseConv = namedtuple('AVXDepthwiseConv', ['ic_bn', 'oc_bn', 'reg_n'])

def _get_default_schedule(out_width, in_channel, out_channel, simd_width=16):
    ic_bn = 1
    for bn in range(simd_width, 0, -1):
        if in_channel % bn == 0:
            ic_bn = bn
            break

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if out_channel % bn == 0:
            oc_bn = bn
            break

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    return AVXDepthwiseConv(ic_bn, oc_bn, reg_n)


def _schedule_depthwise_conv(s, sch, data, kernel, conv_out):
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        # batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        batch, ic, ih, iw = s[A].op.axis
        s[A].parallel(ic)

    # C, O = conv_out, last
    C = conv_out
    CC = s.cache_write(C, 'global')

    _, oc, oh, ow = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.reg_n)
    oc_chunk, oc_block = s[C].split(oc, factor=sch.oc_bn)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)
    # if C == O:
    s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc, oh, ow = s[CC].op.axis
    kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=sch.reg_n)
    oc_chunk, oc_block = s[CC].split(oc, factor=sch.oc_bn)

    # if sch.unroll_kw:
    #     s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
    #     s[CC].unroll(kw)
    # else:
    s[CC].reorder(oc_chunk, oh, ow_chunk, kh, kw, ow_block, oc_block)

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    # if C != O:
    #     batch, oc, oh, ow = s[O].op.axis
    #     ow_chunk, ow_block = s[O].split(ow, factor=sch.reg_n)
    #     oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
    #     s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    #     parallel_axis = s[O].fuse(oc_chunk, oh)
    #     s[C].compute_at(s[O], parallel_axis)
    #     s[O].vectorize(oc_block)
    #     s[O].parallel(parallel_axis)

    return s

@generic.schedule_depthwise_conv2d_nchw.register(["cpu"])
def schedule_depthwise_conv2d(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'depthwise_conv2d_nchw' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            # data = data_vec.op.input_tensors[0]
            # data_pad = None
            # if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
            #     data_pad = data
            #     data = data_pad.op.input_tensors[0]

            in_channel = kernel.shape[0].value
            n, out_channel, out_height, out_width = [x.value for x in conv_out.shape]

            sch = _get_default_schedule(out_width, in_channel, out_channel, simd_width=16)
            _schedule_depthwise_conv(s, sch, data_vec, kernel, conv_out)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

def _schedule_depthwise_conv_NCHWc(s, sch, data, kernel, conv_out, output):
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        p = s[A].fuse(ic_chunk, ih)
        s[A].parallel(p)

    C, O = conv_out, output
    CC = s.cache_write(C, 'global')

    _, ic_chunk, oh, ow, ic_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.reg_n)
    s[C].reorder(ic_chunk, oh, ow_chunk, ow_block, ic_block)
    s[C].vectorize(ic_block)
    parallel_axis = s[C].fuse(ic_chunk, oh)
    s[C].parallel(parallel_axis)
    s[C].unroll(ow_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, ic_chunk, oh, ow, ic_block = s[CC].op.axis
    kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=sch.reg_n)

    s[CC].reorder(ic_chunk, oh, kh, kw, ow_block, ic_block)

    s[CC].vectorize(ic_block)
    s[CC].unroll(ow_block)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=sch.reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s

@generic.schedule_depthwise_conv2d_NCHWc.register(["cpu"])
def schedule_depthwise_conv2d_NCHWc(outs):
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'depthwise_conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data = conv_out.op.input_tensors[0]
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_raw = data.op.input_tensors[0]
                _, in_channel_chunk, _, _, in_channel_block = [x.value for x in data_raw.shape]
            else:
                _, in_channel_chunk, _, _, in_channel_block = [x.value for x in data.shape]

            n, out_channel_chunk, out_height, out_width, out_channel_block = [x.value for x in conv_out.shape]

            sch = _get_default_schedule(out_width, in_channel_chunk*in_channel_block,
                                        out_channel_chunk*out_channel_block, simd_width=16)
            _schedule_depthwise_conv_NCHWc(s, sch, data, kernel, conv_out, outs[0])

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


@depthwise_conv2d_NCHWc.register("cpu")
def decl_depthwise_conv2d_NCHWc(Input, Filter, stride, padding, out_dtype=None):
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    batch, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(Input.shape)
    out_channel_chunk, filter_height, filter_width, out_channel_block = get_const_tuple(Filter.shape)

    from topi.util import get_const_int
    # assert(get_const_int(in_channel_chunk) == get_const_int(filter_channel_chunk))
    # assert(get_const_int(in_channel_block) == get_const_int(filter_channel_block))
    # assert(get_const_int(channel_multiplier) == 1)

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (filter_height, filter_width))
    # in_channel = in_channel_chunk * in_channel_block
    # out_channel = out_channel_chunk * out_channel_block
    out_height = (in_height - filter_height + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - filter_width + pad_left + pad_right) // stride_w + 1
    # sch = _get_default_schedule(get_const_int(out_width),
    #                             get_const_int(in_channel),
    #                             get_const_int(out_channel),
    #                             simd_width=16)
    #
    # out_channel_chunk = out_channel // sch.oc_bn
    # out_channel_block = sch.oc_bn

    # padding stage
    if pad_top > 0 or pad_left > 0 or pad_down > 0 or pad_right > 0:
        pad_before = [0, 0, pad_top, pad_left, 0]
        pad_after = [0, 0, pad_down, pad_right, 0]
        PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    else:
        PaddedInput = Input
    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_channel_chunk, out_height, out_width, out_channel_block),
        lambda b, oc_chunk, i, j, oc_block: tvm.sum(
            (PaddedInput[b,
                         (oc_chunk * out_channel_block + oc_block)//1//in_channel_block,
                         i*stride_h+di, j*stride_w+dj,
                         ((oc_chunk * out_channel_block + oc_block)//1) % in_channel_block].astype(out_dtype) *
             Filter[oc_chunk, di, dj, oc_block].astype(out_dtype)), axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_NCHWc")
    return Output

