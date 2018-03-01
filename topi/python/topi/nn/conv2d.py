# pylint: disable=invalid-name, unused-variable, too-many-locals, unused-argument
"""Conv2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify

# workload description of conv2d
Workload = namedtuple('Workload',
                      ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

# schedule description of spatial
SpatialPack = namedtuple('SpatialPack',
                         ['vh', 'vw', 'vc', 'ba', 'bc', 'unroll'])

# schedule description of im2col
Im2ColPack = namedtuple('Im2ColPack',
                        ['vp', 'vq', 'ba', 'bc', 'unroll'])

_WORKLOADS = [
    # workloads of resnet18_v1 on imagenet 12 0-11
    Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    # workloads of resnet34_v1 on imagenet, no extra workload required
    # workloads of resnet50_v1 on imagenet 14 12-25
    Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
    # workloads of resnet101_v1 on imagenet, no extra workload required
    # workloads of resnet152_v1 on imagenet, no extra workload required
    # workloads of resnet18_v2 on imagenet, no extra workload required
    # workloads of resnet34_v2 on imagenet, no extra workload required
    # workloads of resnet50_v2 on imagenet 3 26-28
    Workload('float32', 'float32', 56, 56, 128, 128, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 28, 28, 256, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 14, 14, 512, 512, 3, 3, 1, 1, 2, 2),
    # workloads of resnet101_v2 on imagenet, no extra workload required
    # workloads of resnet152_v2 on imagenet, no extra workload required
    # workloads of mobilenet 1.0 on imagenet 10 29-38
    Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),
    # workloads of mobilenet 0.75 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 24, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 24, 48, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 48, 96, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 96, 96, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 96, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 192, 384, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 384, 384, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 384, 768, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 768, 768, 1, 1, 0, 0, 1, 1),
    # workloads of mobilenet 0.5 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 16, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 16, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 256, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 512, 512, 1, 1, 0, 0, 1, 1),
    # workloads of mobilenet 0.25 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 8, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 8, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 16, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 32, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 256, 256, 1, 1, 0, 0, 1, 1),
    # workloads of vgg11 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 112, 112, 64, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 128, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 256, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 512, 512, 3, 3, 1, 1, 1, 1),
    # workloads of vgg13 on imagenet
    Workload('float32', 'float32', 224, 224, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 112, 112, 128, 128, 3, 3, 1, 1, 1, 1),
    # workloads of vgg16 on imagenet, no extra workload required
    # workloads of vgg19 on imagenet, no extra workload required
    # workloads of vgg11_bn on imagenet, no extra workload required
    # workloads of vgg13_bn on imagenet, no extra workload required
    # workloads of vgg16_bn on imagenet, no extra workload required
    # workloads of vgg19_bn on imagenet, no extra workload required
    # workloads of densenet 121 on imagenet
    Workload('float32', 'float32', 56, 56, 128, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 96, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 160, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 192, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 224, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 160, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 192, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 224, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 288, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 320, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 352, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 384, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 416, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 448, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 480, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 128, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 288, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 320, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 352, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 384, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 416, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 448, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 480, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 544, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 576, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 608, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 640, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 672, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 704, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 736, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 768, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 800, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 832, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 864, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 896, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 928, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 960, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 992, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 128, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 7, 7, 544, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 576, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 608, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 640, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 672, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 704, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 736, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 768, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 800, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 832, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 864, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 896, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 928, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 960, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 992, 128, 1, 1, 0, 0, 1, 1),
    # workloads of densenet 161 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 96, 7, 7, 3, 3, 2, 2),
    Workload('float32', 'float32', 56, 56, 96, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 192, 48, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 144, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 192, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 240, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 288, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 336, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 384, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 192, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 192, 48, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 240, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 288, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 336, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 384, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 432, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 480, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 528, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 576, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 624, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 672, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 720, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 768, 384, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 384, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 192, 48, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 432, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 480, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 528, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 576, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 624, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 672, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 720, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 768, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 816, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 864, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 912, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 960, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1008, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1056, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1104, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1152, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1200, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1248, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1296, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1344, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1392, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1440, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1488, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1536, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1584, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1632, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1680, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1728, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1776, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1824, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1872, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1920, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1968, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 2016, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 2064, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 2112, 1056, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1056, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 192, 48, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 7, 7, 1104, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1152, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1200, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1248, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1296, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1344, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1392, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1440, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1488, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1536, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1584, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1632, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1680, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1728, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1776, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1824, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1872, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1920, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1968, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 2016, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 2064, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 2112, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 2160, 192, 1, 1, 0, 0, 1, 1),
    # workloads of densenet 169 on imagenet
    Workload('float32', 'float32', 14, 14, 1024, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1056, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1088, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1120, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1152, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1184, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1216, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1248, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1280, 640, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1024, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1056, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1088, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1120, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1152, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1184, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1216, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1248, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1280, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1312, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1344, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1376, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1408, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1440, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1472, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1504, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1536, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1568, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1600, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1632, 128, 1, 1, 0, 0, 1, 1),
    # workloads of densenet 201 on imagenet
    Workload('float32', 'float32', 14, 14, 1024, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1056, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1088, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1120, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1152, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1184, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1216, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1248, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1280, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1312, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1344, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1376, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1408, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1440, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1472, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1504, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1536, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1568, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1600, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1632, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1664, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1696, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1728, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1760, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1792, 896, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1024, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1056, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1088, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1120, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1152, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1184, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1216, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1248, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1280, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1312, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1344, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1376, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1408, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1440, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1472, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1504, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1536, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1568, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1600, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1632, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1664, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1696, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1728, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1760, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1792, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1824, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1856, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1888, 128, 1, 1, 0, 0, 1, 1),
    # workloads of alexnet 201 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 64, 11, 11, 2, 2, 4, 4),
    Workload('float32', 'float32', 27, 27, 64, 192, 5, 5, 2, 2, 1, 1),
    Workload('float32', 'float32', 13, 13, 192, 384, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 13, 13, 384, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 13, 13, 256, 256, 3, 3, 1, 1, 1, 1),
    # workloads of squeezenet1.0 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 96, 7, 7, 0, 0, 2, 2),
    Workload('float32', 'float32', 54, 54, 96, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 54, 54, 16, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 54, 54, 16, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 54, 54, 128, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 54, 54, 128, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 54, 54, 32, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 54, 54, 32, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 27, 27, 256, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 32, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 32, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 27, 27, 256, 48, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 48, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 48, 192, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 27, 27, 384, 48, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 384, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 64, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 64, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 13, 13, 512, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 13, 13, 64, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 13, 13, 64, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 13, 13, 512, 1000, 1, 1, 0, 0, 1, 1),
    # workloads of squeezenet1.1 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 64, 3, 3, 0, 0, 2, 2),
    Workload('float32', 'float32', 55, 55, 64, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 55, 55, 16, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 55, 55, 16, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 55, 55, 128, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 27, 27, 128, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 13, 13, 256, 48, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 13, 13, 48, 192, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 13, 13, 48, 192, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 13, 13, 384, 48, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 13, 13, 384, 64, 1, 1, 0, 0, 1, 1),
]


# platform specific schedule
_CONV_SCHEDULE = {}

@tvm.target.generic_func
def conv2d(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
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
    if layout == 'NCHW':
        return conv2d_nchw(data, kernel, stride, padding, out_dtype)
    elif layout == 'HWCN':
        return conv2d_hwcn(data, kernel, stride, padding, out_dtype)
    elif layout == 'NHWC':
        return conv2d_nhwc(data, kernel, stride, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


def _get_workload(data, kernel, stride, padding, out_dtype):
    """ Get the workload structure. """
    _, CI, IH, IW = [x.value for x in data.shape]
    CO, _, KH, KW = [x.value for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert data.dtype == kernel.dtype, "Do not support inputs with different data types now."
    return Workload(data.dtype, out_dtype, IH, IW, CI, CO, KH, KW, HPAD, WPAD, HSTR, WSTR)


@tvm.target.generic_func
def _get_schedule(wkl):
    # pylint: disable=unreachable
    """ Get the platform specific schedule. """
    target = tvm.target.current_target()
    raise RuntimeError(
        "No schedule for current target:{}".format(target))
    # This return has no use, merely to supress pylint warning
    return wkl

def _spatial_pack(data, kernel, stride, padding, out_dtype):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH-1, KW-1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dshape = (1, CI, H, W)
    dpshape = (1, CI, TH, TW)
    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)

    kshape = (CO, CI, KH, KW)
    kvshape = (CO/VC, CI, KH, KW, VC)

    ovshape = (1, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw: \
        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
        tvm.sum(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw].astype(out_dtype) *
                kernel_vec[co, ci, dh, dw, vc].astype(out_dtype),
                axis=[ci, dh, dw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h/VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatial_conv_output')

    return output


def _im2col_pack(data, kernel, stride, padding, out_dtype):
    """ Compute convolution with im2col pack layout. """
    assert data.shape[0].value == 1, "im2col pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    N = 1
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.hpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    P = sch.vp
    Q = sch.vq
    UNROLL = sch.unroll

    dshape = (N, CI, H, W)
    dpshape = (N, CI, H+2*HPAD, W+2*WPAD)
    dcshape = (N, OH, OW, CI, KH, KW)
    dvshape = (N, OH * OW // P, CI, KH, KW, P)

    kshape = (CO, CI, KH, KW)
    kvshape = (CO // Q, CI, KH, KW, Q)

    ovshape = (N, CO // Q, OH * OW // P, P, Q)
    oshape = (N, CO, OH, OW)

    ############### declaration

    DO_PAD = (wkl.hpad != 0 and wkl.wpad != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_col = tvm.compute(dcshape, lambda n, oh, ow, ci, hk, wk: \
        data_pad[n][ci][oh*HSTR+hk][ow*WSTR+wk], name='data_col')

    data_vec = tvm.compute(dvshape, lambda n, im, ci, hk, wk, vim: \
        data_col[n][(im*P+vim)//OW][(im*P+vim)%OW][ci][hk][wk], name='data_vec')


    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*Q+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    hk = tvm.reduce_axis((0, KH), name='hk')
    wk = tvm.reduce_axis((0, KW), name='wk')

    conv = tvm.compute(ovshape, lambda n, co, im, vim, vco: \
        tvm.sum(data_vec[n][im][ci][hk][wk][vim].astype(out_dtype) *
                kernel_vec[co][ci][hk][wk][vco].astype(out_dtype),
                axis=[ci, hk, wk]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w: \
                         conv[n][co//Q][(h*OW+w)//P][(h*OW+w)%P][co%Q],
                         name='output_vec', tag='im2col_conv_output')

    return output


def conv2d_nchw(Input, Filter, stride, padding, out_dtype='float32'):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 2
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: tvm.sum(
            temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw")


def conv2d_hwcn(Input, Filter, stride, padding, out_dtype='float32'):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [in_height, in_width, in_channel, batch]

    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    assert isinstance(stride, int) or len(stride) == 2
    in_height, in_width, in_channel, batch = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [pad_top, pad_left, 0, 0]
    pad_after = [pad_down, pad_right, 0, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (out_height, out_width, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            PaddedInput[yy * stride_h + ry, xx * stride_w + rx, rc, nn].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_hwcn")
    return Output


def conv2d_nhwc(Input, Filter, stride, padding, out_dtype='float32'):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height,  out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: tvm.sum(
            PaddedInput[nn, yy * stride_h + ry, xx * stride_w + rx, rc].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_nhwc")
    return Output

# map from schedule type to declaration function
_SCH_TO_DECL_FUNC = {
    SpatialPack: _spatial_pack,
    Im2ColPack: _im2col_pack,
}
