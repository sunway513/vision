#pragma once

#include "cpu/nms_kernel.h"

#ifdef WITH_CUDA
#include "cuda/nms_kernel.h"
#endif

namespace vision {
namespace ops {

// C++ Forward
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  if (dets.device().is_cuda()) {
#if defined(WITH_CUDA)
    if (dets.numel() == 0) {
      at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_cuda(dets, scores, iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif

} // namespace ops
} // namespace vision
