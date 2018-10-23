import tvm
from tvm import autotvm
from tvm.autotvm.task.dispatcher import ApplyGraphBest
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.autotvm.task import register, get_config, ConfigEntity
from .. import generic, tag
from ..nn.pad import pad

from ..util import get_const_tuple
from ..nn.util import infer_pad, infer_stride, get_const_int, get_pad_tuple
from ..nn.depthwise_conv2d import depthwise_conv2d_NCHWc

def _fallback_schedule(wkl, simd_width):
    batch_size, in_channel, height, width, _ = wkl[1]
    _, channel_multiplier, hkernel, wkernel, _ = wkl[2]
    HPAD, WPAD = wkl[4]
    HSTR, WSTR = wkl[3]
    out_width = (width + 2 * WPAD - wkernel) // WSTR + 1
    out_channel = in_channel * channel_multiplier

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

    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [in_channel // ic_bn, ic_bn]],
                      ["tile_oc", "sp", [out_channel // oc_bn, oc_bn]],
                      ["tile_ow", "sp", [out_width // reg_n, reg_n]]],
                "t": ""}
    return ConfigEntity.from_json_dict(cfg_dict)


@depthwise_conv2d_NCHWc.register("cpu")
def depthwise_conv2d_NCHWc_cpu(data, kernel, channels, kernel_size,
                               strides, padding, dilation,
                               groups, layout, out_layout,
                               kernel_layout, out_dtype=None):
    """x86 conv2d_NCHWc declaration."""
    dispatch_ctx = autotvm.task.DispatchContext.current
    if not isinstance(dispatch_ctx, ApplyGraphBest):
        layout = out_layout = "NCHW"
    from .conv2d import _query_dispatcher, conv_NCHWc_arg_to_workload
    workload = conv_NCHWc_arg_to_workload(data, kernel, strides,
                                          padding, layout, out_layout, out_dtype, is_depthwise=True)
    cfg = _query_dispatcher(workload)
    return _declaration_depthwise_conv_NCHWc(cfg, data, kernel, channels, kernel_size,
                                             strides, padding, dilation,
                                             groups, layout, out_layout,
                                             kernel_layout, out_dtype)


@generic.schedule_depthwise_conv2d_NCHWc.register("cpu")
def schedule_depthwise_conv2d_NCHWc(strides, padding, layout, out_layout, outs):
    return _schedule_depthwise_conv2d_NCHWc(None, strides, padding, layout, out_layout, outs)


def _declaration_depthwise_conv_NCHWc(cfg, Input, Filter, channels, kernel_size,
                                      strides, padding, dilation,
                                      groups, layout, out_layout,
                                      kernel_layout, out_dtype):
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    out_dtype = Input.dtype if out_dtype is None else out_dtype
    batch, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(Input.shape)
    out_channel_chunk, filter_height, filter_width, out_channel_block = get_const_tuple(Filter.shape)
    assert(ic_bn == in_channel_block)
    assert(oc_bn == out_channel_block)
    assert(channels == out_channel_chunk * out_channel_block)

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (filter_height, filter_width))
    in_channel = in_channel_chunk * in_channel_block
    assert(channels % in_channel == 0)
    channel_multiplier = channels // in_channel
    out_height = (in_height - filter_height + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - filter_width + pad_left + pad_right) // stride_w + 1
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
                         (oc_chunk * oc_bn + oc_block)//channel_multiplier // ic_bn,
                         i*stride_h+di, j*stride_w+dj,
                         ((oc_chunk * oc_bn + oc_block)//channel_multiplier) % ic_bn].astype(out_dtype) *
             Filter[oc_chunk, di, dj, oc_block].astype(out_dtype)),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_NCHWc")
    return Output


def _schedule_depthwise_conv2d_NCHWc(cfg, strides, padding, layout, out_layout, outs):
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
            data = conv_out.op.input_tensors[0]
            input = data
            kernel = conv_out.op.input_tensors[1]
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data = data.op.input_tensors[0]
            _, in_channel_chunk, _, _, in_channel_block = [x.value for x in data.shape]

            current_cfg = cfg
            if current_cfg is None:
                from .conv2d import _query_dispatcher, conv_NCHWc_arg_to_workload
                workload = conv_NCHWc_arg_to_workload(data, kernel, strides,
                                                      padding, layout, out_layout,
                                                      conv_out.dtype, is_depthwise=True)
                current_cfg = _query_dispatcher(workload)
            _schedule_depthwise_conv2d_NCHWc_impl(s, current_cfg, input, kernel, conv_out, outs[0])
        scheduled_ops.append(op)
    traverse(outs[0].op)
    return s


def _schedule_depthwise_conv2d_NCHWc_impl(s, cfg, data, kernel, conv_out, output):
    ic_bn, oc_bn, reg_n = (cfg["tile_ic"].size[-1],
                           cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        p = s[A].fuse(ic_chunk, ih)
        s[A].parallel(p)
    C, O = conv_out, output
    CC = s.cache_write(C, 'global')
    _, ic_chunk, oh, ow, ic_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(ic_chunk, oh, ow_chunk, ow_block, ic_block)
    s[C].vectorize(ic_block)
    parallel_axis = s[C].fuse(ic_chunk, oh)
    s[C].parallel(parallel_axis)
    s[C].unroll(ow_block)
    s[CC].compute_at(s[C], ow_chunk)
    _, ic_chunk, oh, ow, ic_block = s[CC].op.axis
    kh, kw = s[CC].op.reduce_axis
    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    s[CC].reorder(ic_chunk, oh, kh, kw, ow_block, ic_block)
    s[CC].vectorize(ic_block)
    s[CC].unroll(ow_block)
    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)
    return s


def _create_schedule_template(cfg, data, kernel, strides, padding):
    """Create schedule configuration from input arguments"""
    n, ic, h, w = get_const_tuple(data.shape)
    _, channel_multiply, kh, kw = get_const_tuple(kernel.shape)
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1
    oc = ic * channel_multiply
    # Create schedule config
    cfg.define_split("tile_ic", ic, num_outputs=2)
    cfg.define_split("tile_oc", oc, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)


@register("topi_x86_depthwise_conv2d_NCHWc")
def _topi_nn_depthwise_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel, channels, kernel_size, strides, \
        padding, dilation, groups, layout, out_layout, kernel_layout, dtype = args
    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_schedule_template(cfg, data, kernel, strides, padding)

    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    new_kernel_shape = (raw_kernel_shape[0] * raw_kernel_shape[1] // oc_bn,
                        raw_kernel_shape[2], raw_kernel_shape[3], oc_bn)
    new_data = tvm.placeholder(new_data_shape, data.dtype)
    new_kernel = tvm.placeholder(new_kernel_shape, kernel.dtype)
    C = _declaration_depthwise_conv_NCHWc(cfg, new_data, new_kernel, channels, kernel_size, strides,
                                          padding, dilation, groups, layout, out_layout,
                                          kernel_layout, dtype)
    s = _schedule_depthwise_conv2d_NCHWc(cfg, strides, padding, layout, out_layout, [C])
    return s, [new_data, new_kernel, C]

