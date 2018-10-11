# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Conv2D schedule on x86"""
from collections import namedtuple
import tvm
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride
from ..nn.depthwise_conv2d import depthwise_conv2d_nchw

from .check_targets import check_skylake

AVXDepthwiseConv = namedtuple('AVXDepthwiseConv', ['oc_bn', 'reg_n'])

def _get_default_schedule(out_width, out_channel, simd_width=16):
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

    print(AVXDepthwiseConv(oc_bn, reg_n))
    return AVXDepthwiseConv(oc_bn, reg_n)


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

            n, out_channel, out_height, out_width = [x.value for x in conv_out.shape]

            sch = _get_default_schedule(out_width, out_channel, simd_width=16)
            _schedule_depthwise_conv(s, sch, data_vec, kernel, conv_out)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s