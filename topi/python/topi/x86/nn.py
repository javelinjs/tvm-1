# pylint: disable=invalid-name,too-many-locals
"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from .. import tag

def _default_schedule(outs, auto_inline):
    """Default schedule for x86."""
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
        return s
    if len(s[x].op.axis) == 4:
        n, c, _, _ = s[x].op.axis
        fused = s[x].fuse(n, c) # for nhwc layout, fuse n and h
        s[x].parallel(fused)
    elif len(s[x].op.axis) == 5:
        n, C, h, _, _ = s[x].op.axis
        fused = s[x].fuse(n, C, h)
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s


@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) == 4:
        n, c, _, _ = s[x].op.axis
        fused = s[x].fuse(n, c) # for nhwc layout, fuse n and h
        s[x].parallel(fused)
    elif len(s[x].op.axis) == 5:
        n, C, h, w, c = s[x].op.axis
        fused = s[x].fuse(n, C, h)
        s[x].parallel(fused)
        s[x].vectorize(c)
    elif len(s[x].op.axis) == 3:
        n, c, _ = s[x].op.axis
        fused = s[x].fuse(n, c)
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s

@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _parallel_sch(sch):
        if len(sch.op.axis) == 4:
            n, c, _, _ = sch.op.axis
            fused = sch.fuse(n, c) # for nhwc layout, fuse n and h
            sch.parallel(fused)
        elif len(sch.op.axis) == 5:
            n, C, h, w, c = sch.op.axis
            fused = sch.fuse(n, C, h)
            sch.parallel(fused)
        else:
            sch.parallel(sch.op.axis[0])

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, tvm.tensor.ComputeOp):
            s[PaddedInput].compute_inline()
        _parallel_sch(s[Pool])

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('pool'):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)
    traverse(outs[0].op)
    return s


@generic.schedule_global_pool.register(["cpu"])
def schedule_global_pool(outs):
    """Schedule for global pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) == 4:
        n, c, _, _ = s[x].op.axis
        fused = s[x].fuse(n, c) # for nhwc layout, fuse n and h
        s[x].parallel(fused)
    elif len(s[x].op.axis) == 5:
        n, C, h, w, c = s[x].op.axis
        fused = s[x].fuse(n, C, h)
        s[x].parallel(fused)
        s[x].vectorize(c)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s

@generic.schedule_dense.register(["cpu"])
def schedule_dense(outs):
    """Schedule for dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'dense' in op.tag:
            C = op.output(0)
            x, y = C.op.axis

            # Write cache for blocks
            CC = s.cache_write(C, 'global')

            # Tile
            bnx = 1
            bny = 4
            _, yo, _, yi = s[C].tile(x, y, bnx, bny)
            s[CC].compute_at(s[C], yo)
            xc, yc = s[CC].op.axis
            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=4)
            s[CC].reorder(ko, xc, ki, yc)
            s[CC].unroll(ki)
            s[CC].vectorize(yc)

            # Vectorization
            s[C].vectorize(yi)

            # Parallelization
            s[C].parallel(yo)

    traverse(outs[0].op)
    return s
