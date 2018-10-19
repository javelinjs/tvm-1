from tvm import autotvm
from tvm.autotvm.task.dispatcher import ApplyGraphBest
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.autotvm.task import register, get_config
from .. import generic, tag

from ..util import get_const_tuple

def _declaration_depthwise_conv_NCHWc_impl(cfg, data, kernel, kernel_size, strides, padding, layout,
                                           out_layout, out_dtype):
    HPAD, WPAD = padding
    HSTR, WSTR = strides

    n, ic_chunk, ih, iw, ic_block = get_const_tuple(data.shape)
    ic = ic_chunk * ic_block
    kh, kw = kernel_size
    oc_chunk, _, _, _, _, oc_block = get_const_tuple(kernel.shape)
    oc = oc_chunk * oc_block
    oh = (ih + 2 * HPAD - kh) // HSTR + 1
    ow = (iw + 2 * WPAD - kw) // WSTR + 1

    # DOPAD
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    if ic_bn != ic_block:
        raise RuntimeError("ic_bn in config is not equal to actual data ic_block: %d vs %d."
                           % (ic_bn, ic_block))
    if oc_bn != oc_block:
        raise RuntimeError("oc_bn in config is not equal to actual kernel oc_block: %d vs %d."
                           % (oc_bn, oc_block))

    # convolution
    oshape = (n, oc//oc_bn, oh, ow, oc_bn)

    ic = tvm.reduce_axis((0, ic), name='ic')
    kh = tvm.reduce_axis((0, kernel_size[0]), name='kh')
    kw = tvm.reduce_axis((0, kernel_size[1]), name='kw')

    workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size,
                                          strides, padding, layout,
                                          out_layout, out_dtype),
    attrs = {'workload': workload}
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
    tvm.sum(data_pad[n, ic//ic_bn, oh*HSTR+kh, ow*WSTR+kw,
                     ic%ic_bn].astype(out_dtype) *
            kernel[oc_chunk, ic//ic_bn, kh, kw, ic%ic_bn, oc_block],
            axis=[ic, kh, kw]),
                       name='conv2d_NCHWc', tag="conv2d_NCHWc", attrs=attrs)
    return conv

def _declaration_depthwise_conv_NCHWc(cfg, data, kernel, num_filter, kernel_size, strides,
                                      padding, layout, out_layout, out_dtype):
    n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
    ic = ic_chunk * ic_block
    kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else \
        (kernel_size, kernel_size)
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    args = [cfg, data, kernel, (kh, kw), (sh, sw), (ph, pw), layout, out_layout, out_dtype]
    return _declaration_depthwise_conv_NCHWc_impl(*args)


def _schedule_depthwise_conv2d_NCHWc(cfg, num_filter, kernel_size, strides, padding,
                                     layout, out_layout, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    dispatch_ctx = autotvm.task.DispatchContext.current
    if not isinstance(dispatch_ctx, ApplyGraphBest):
        layout = out_layout = "NCHW"

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else \
                (kernel_size, kernel_size)
            is_kernel_1x1 = kh == 1 and kw == 1
            n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
            ic = ic_chunk * ic_block
            original_data = tvm.placeholder((n, ic, h, w), dtype=data.dtype)

            kh, kw = kernel_size
            original_kernel = tvm.placeholder((num_filter, ic, kh, kw),
                                              dtype=kernel.dtype)
            current_cfg = cfg
            if current_cfg is None:
                workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                                                      padding, layout, out_layout,
                                                      conv_out.dtype)
                current_cfg = _query_dispatcher(workload)
            args = [s, current_cfg, data_vec, conv_out, outs[0]]
            if is_kernel_1x1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

def _create_schedule_template(cfg, data, kernel, strides, padding, layout):
    """Create schedule configuration from input arguments"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    if layout == 'NCHW':
        n, ic, h, w = dshape
        oc, _, kh, kw = kshape
    else:
        raise ValueError("Not support this layout {} with "
                         "schedule template.".format(layout))
    is_kernel_1x1 = kh == 1 and kw == 1
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1

    # Create schedule config
    cfg.define_split("tile_ic", ic, num_outputs=2)
    cfg.define_split("tile_oc", oc, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])


@register("topi_x86_depthwise_conv2d_NCHWc")
def _topi_nn_depthwise_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    strides = args[4]
    padding = args[5]
    layout = args[6]
    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_schedule_template(cfg, data, kernel, strides, padding, layout)

    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    data_layout = "NCHW%dc" % ic_bn
    out_layout = "NCHW%dc" % oc_bn
    new_kernel_shape = (raw_kernel_shape[0] // oc_bn, raw_kernel_shape[1] // ic_bn,
                        raw_kernel_shape[2], raw_kernel_shape[3], ic_bn, oc_bn)
    args[0] = tvm.placeholder(new_data_shape, data.dtype)
    args[1] = tvm.placeholder(new_kernel_shape, kernel.dtype)
    args[6] = data_layout
    args[7] = out_layout

    C = _declaration_depthwise_conv_NCHWc(cfg, *args, **kwargs)
    s = _schedule_depthwise_conv2d_NCHWc(cfg, args[2], args[3], args[4], args[5],
                                         args[6], args[7], [C])
    return s, [args[0], args[1], C]