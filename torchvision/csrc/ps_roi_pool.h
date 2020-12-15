#pragma once

#include "cpu/ps_roi_pool_kernel.h"

#ifdef WITH_CUDA
#include "cuda/ps_roi_pool_kernel.h"
#endif

namespace vision {
namespace ops {

// C++ Forward
std::tuple<at::Tensor, at::Tensor> ps_roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

// Autocast Forward
#if defined(WITH_CUDA)
std::tuple<at::Tensor, at::Tensor> ps_roi_pool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);
#endif

// C++ Backward
at::Tensor _ps_roi_pool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

// Autograd Forward and Backward
std::tuple<at::Tensor, at::Tensor> ps_roi_pool_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

at::Tensor ps_roi_pool_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

} // namespace ops
} // namespace vision
