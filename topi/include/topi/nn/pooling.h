/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Pooling op constructions
 * \file nn/pooling.h
 */
#ifndef TOPI_NN_POOLING_H_
#define TOPI_NN_POOLING_H_

#include <string>

#include "tvm/tvm.h"
#include "tvm/ir_pass.h"
#include "topi/tags.h"
#include "topi/detail/pad_utils.h"
#include "topi/nn.h"

namespace topi {
namespace nn {
using namespace tvm;

/*! \brief Pooling type */
enum PoolType : int {
  kAvgPool,
  kMaxPool,
};

/*!
* \brief Perform pooling on data
*
* \param x The input tensor
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param height_idx index of the height dimension
* \param width_idx index of the width dimension
*
* \return The output tensor in same layout order
*/

inline Tensor pool_impl(const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        PoolType pool_type,
                        bool ceil_mode,
                        const size_t height_idx,
                        const size_t width_idx) {
  CHECK(x->shape.size() == 4 || x->shape.size() == 5) << "Pooling input must be 4-D or 5-D";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 2) << "Pooling padding_size must have 2 elements";

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto stride_height = stride_size[0];
  auto stride_width = stride_size[1];
  auto padding_height = padding_size[0];
  auto padding_width = padding_size[1];

  auto height = x->shape[height_idx];
  auto width = x->shape[width_idx];

  auto pad_tuple = detail::GetPadTuple(padding_height, padding_width);
  auto pad_top = pad_tuple[0];
  auto pad_left = pad_tuple[1];
  auto pad_down = pad_tuple[2];
  auto pad_right = pad_tuple[3];

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_down += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before{ 0, 0, 0, 0, 0 };
  pad_before.Set(height_idx, pad_top);
  pad_before.Set(width_idx, pad_left);

  Array<Expr> pad_after{ 0, 0, 0, 0, 0 };
  pad_after.Set(height_idx, pad_down);
  pad_after.Set(width_idx, pad_right);

  auto out_height = tvm::ir::Simplify(
  (height - kernel_height + pad_top + pad_down) / stride_height + 1);
  auto out_width = tvm::ir::Simplify(
  (width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_idx, out_height);
  out_shape.Set(width_idx, out_width);

  if (pool_type == kMaxPool) {
    auto temp = pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp");
    return tvm::compute(out_shape,
    [&](const Array<Var>& output) {
      Array<Expr> indices{ output[0], output[1], output[2], output[3] };
      if (output.size() == 5) {
        indices.push_back(output[4]);
      }
      indices.Set(height_idx, output[height_idx] * stride_height + dheight);
      indices.Set(width_idx, output[width_idx] * stride_width + dwidth);
      return tvm::max(temp(indices), { dheight, dwidth });
    }, "tensor", "pool_max");
  } else if (pool_type == kAvgPool) {
    auto temp = pad(x, pad_before, pad_after, 0, "pad_temp");

    auto tsum = tvm::compute(out_shape,
    [&](const Array<Var>& output) {
      Array<Expr> indices{ output[0], output[1], output[2], output[3] };
      if (output.size() == 5) {
        indices.push_back(output[4]);
      }
      indices.Set(height_idx, output[height_idx] * stride_height + dheight);
      indices.Set(width_idx, output[width_idx] * stride_width + dwidth);
      return tvm::sum(temp(indices), { dheight, dwidth });
    }, "tensor", "pool_avg");

    return tvm::compute(out_shape,
    [&](const Array<Var>& output) {
      return tsum(output) / (kernel_height * kernel_width);
    }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

/*!
* \brief Perform pooling on data
*
* \param x The input tensor in NCHW or NHWC order
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param layout The input layout
*
* \return The output tensor in NCHW order
*/

inline Tensor pool(const Tensor& x,
                   const Array<Expr>& kernel_size,
                   const Array<Expr>& stride_size,
                   const Array<Expr>& padding_size,
                   PoolType pool_type,
                   bool ceil_mode,
                   const std::string& layout = "NCHW") {
  if (layout.rfind("NCHW") == 0) {
    return pool_impl(x, kernel_size, stride_size, padding_size, pool_type, ceil_mode, 2, 3);
  } else if (layout == "NHWC") {
    return pool_impl(x, kernel_size, stride_size, padding_size, pool_type, ceil_mode, 1, 2);
  } else {
    LOG(ERROR) << "Unsupported layout " << layout;
  }
}

/*!
* \brief Perform global pooling on data in NCHW order
*
* \param x The input tensor in NCHW order
* \param pool_type The type of pooling operator
*
* \return The output tensor with shape [batch, channel, 1, 1]
*/
inline Tensor global_pool(const Tensor& x,
                          PoolType pool_type) {
  CHECK_EQ(x->shape.size(), 4) << "Pooling input must be 4-D";

  auto batch = x->shape[0];
  auto channel = x->shape[1];
  auto height = x->shape[2];
  auto width = x->shape[3];

  auto dheight = tvm::reduce_axis(Range(0, height));
  auto dwidth = tvm::reduce_axis(Range(0, width));

  if (pool_type == kMaxPool) {
    return tvm::compute(
      { batch, channel, 1, 1 },
      [&](Var n, Var c, Var h, Var w) {
        return tvm::max(x(n, c, dheight, dwidth), { dheight, dwidth });  // NOLINT(*)
      }, "tensor", "global_pool_max");
  } else if (pool_type == kAvgPool) {
    auto tsum = tvm::compute(
      { batch, channel, 1, 1 },
      [&](Var n, Var c, Var h, Var w) {
        return tvm::sum(x(n, c, dheight, dwidth), { dheight, dwidth });
      }, "tensor", "global_pool_sum");

    return tvm::compute(
      { batch, channel, 1, 1 },
      [&](Var n, Var c, Var h, Var w) {
        return tsum(n, c, h, w) / tvm::cast(x->dtype, height * width);
      }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_POOLING_H_
