/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// -----------------------------------------------------------------------------
// CPU-less-CUDA fallback definitions for the non-templated tensor_ops kernel
// launchers. On a normal build these are CUDA kernels compiled in tensor_ops.cu /
// tensor_*_ops.cu, which are excluded from CUDA-less (e.g. macOS) builds. The
// templated launchers get real CPU bodies in tensor_ops.hpp; these remaining
// non-templated entry points are currently no-op / default-returning stubs so the
// library links and the Python module imports (Phase 0 GUI bring-up). They do NOT
// perform real computation yet — that is the Phase 1 CPU backend. A scene/training
// path that reaches one will produce empty output, not a crash.
// This file is only compiled when LFS_ENABLE_CUDA is OFF.
// -----------------------------------------------------------------------------

// std headers the shared functor/op headers rely on transitively (other TUs pull
// these in first; this minimal TU must include them before tensor_ops.hpp).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "internal/tensor_ops.hpp"

namespace lfs::core::tensor_ops {

    float direct_sum_scalar(const float*, size_t, cudaStream_t) { return 0.0f; }
    float direct_mean_scalar(const float*, size_t, cudaStream_t) { return 0.0f; }
    float direct_max_scalar(const float*, size_t, cudaStream_t) { return 0.0f; }
    float direct_min_scalar(const float*, size_t, cudaStream_t) { return 0.0f; }
    bool has_nan_or_inf_gpu(const float*, size_t, cudaStream_t) { return false; }

    void launch_adaptive_avg_pool2d(const float*, float*, int, int, int, int, int, int, cudaStream_t) {}
    void launch_bernoulli(float*, size_t, float, unsigned long long, cudaStream_t) {}
    void launch_bias_add(const float*, const float*, float*, int, int, int, cudaStream_t) {}
    void launch_bias_relu(const float*, const float*, float*, int, int, int, cudaStream_t) {}
    void launch_broadcast(const float*, float*, const size_t*, const size_t*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_broadcast_bool(const unsigned char*, unsigned char*, const size_t*, const size_t*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_broadcast_strided(const float*, float*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_broadcast_strided_bool(const unsigned char*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_cdist(const float*, const float*, float*, size_t, size_t, size_t, float, cudaStream_t) {}
    void launch_clamp_fused(const float*, float*, float, float, size_t, cudaStream_t) {}
    void launch_clamp_scalar(float*, float, float, size_t, cudaStream_t) {}
    void launch_clamp_scalar_int(int*, int, int, size_t, cudaStream_t) {}
    void launch_column_reduce(const float*, float*, size_t, size_t, ReduceOp, cudaStream_t) {}
    void launch_count_nonzero_bool(const unsigned char*, size_t*, size_t, cudaStream_t) {}
    void launch_count_nonzero_float(const float*, size_t*, size_t, cudaStream_t) {}
    void launch_cumsum(void*, const size_t*, size_t, int, DataType, cudaStream_t) {}
    void launch_diag(const float*, float*, size_t, cudaStream_t) {}
    void launch_dot_product(const float*, const float*, float*, size_t, cudaStream_t) {}
    void launch_eye(float*, size_t, size_t, cudaStream_t) {}
    void launch_fused_affine_transform(const float*, float*, size_t, float, float, cudaStream_t) {}
    void launch_fused_pointwise_chain(const float*, float*, size_t, const FusedPointwiseOpChain&, cudaStream_t) {}
    void launch_fused_segmented_transform_reduce(const float*, float*, size_t, size_t, const FusedPointwiseOpChain&, ReduceOp, cudaStream_t) {}
    void launch_fused_transform_reduce(const float*, float*, size_t, const FusedPointwiseOpChain&, ReduceOp, cudaStream_t) {}
    void launch_gather(const float*, const int*, float*, const size_t*, const size_t*, size_t, int, size_t, int, cudaStream_t) {}
    void launch_gather(const int64_t*, const int*, int64_t*, const size_t*, const size_t*, size_t, int, size_t, int, cudaStream_t) {}
    void launch_index_select(const float*, const int*, float*, const size_t*, size_t, int, size_t, int, cudaStream_t) {}
    void launch_index_select(const int*, const int*, int*, const size_t*, size_t, int, size_t, int, cudaStream_t) {}
    void launch_index_select(const int64_t*, const int*, int64_t*, const size_t*, size_t, int, size_t, int, cudaStream_t) {}
    void launch_index_select(const unsigned char*, const int*, unsigned char*, const size_t*, size_t, int, size_t, int, cudaStream_t) {}
    void launch_load_op(void*, const size_t*, size_t, LoadOp, const void*, DataType, cudaStream_t) {}
    void launch_masked_fill(_Float16*, const unsigned char*, _Float16, size_t, cudaStream_t) {}
    void launch_masked_fill(float*, const unsigned char*, float, size_t, cudaStream_t) {}
    void launch_masked_fill(int*, const unsigned char*, int, size_t, cudaStream_t) {}
    void launch_masked_fill(int64_t*, const unsigned char*, int64_t, size_t, cudaStream_t) {}
    void launch_masked_fill(unsigned char*, const unsigned char*, unsigned char, size_t, cudaStream_t) {}
    void launch_masked_scatter(float*, const unsigned char*, const float*, size_t, size_t, cudaStream_t) {}
    void launch_masked_select(const float*, const unsigned char*, float*, size_t, size_t, cudaStream_t) {}
    void launch_max_pool2d(const float*, float*, int, int, int, int, int, int, int, int, int, cudaStream_t) {}
    void launch_multinomial(const float*, int64_t*, size_t, size_t, bool, unsigned long long, cudaStream_t) {}
    size_t launch_nonzero(const float*, int64_t*, size_t, size_t, cudaStream_t) { return 0; }
    size_t launch_nonzero_bool(const unsigned char*, int64_t*, size_t, size_t, cudaStream_t) { return 0; }
    void launch_pad(const float*, float*, const size_t*, const size_t*, const size_t*, const size_t*, size_t, size_t, cudaStream_t) {}
    void launch_randint(int*, size_t, int, int, unsigned long long, cudaStream_t) {}
    void launch_reduce_op(const void*, void*, const size_t*, size_t, const int*, size_t, bool, ReduceOp, DataType, cudaStream_t) {}
    void launch_relu(const float*, float*, int, cudaStream_t) {}
    void launch_sgemm(const float*, const float*, float*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_sgemm_batched(const float*, const float*, float*, size_t, size_t, size_t, size_t, cudaStream_t) {}
    void launch_sgemm_bias_relu(const float*, const float*, const float*, float*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_sgemm_tn(const float*, const float*, float*, size_t, size_t, size_t, cudaStream_t) {}
    void launch_sort_1d(float*, int64_t*, size_t, bool, cudaStream_t) {}
    void launch_sort_2d(float*, int64_t*, size_t, size_t, size_t, int, bool, cudaStream_t) {}
    void launch_strided_copy(const void*, void*, const size_t*, const size_t*, size_t, size_t, DataType, cudaStream_t) {}
    void launch_strided_scatter(const void*, void*, const size_t*, const size_t*, size_t, size_t, DataType, cudaStream_t) {}
    void launch_strided_scatter_int32_to_float32(const void*, void*, const size_t*, const size_t*, size_t, size_t, cudaStream_t) {}
    void launch_strided_upload(const void*, void*, const size_t*, const size_t*, size_t, size_t, DataType, cudaStream_t) {}
    void launch_take(const float*, const int*, float*, size_t, size_t, cudaStream_t) {}
    void launch_uniform(float*, size_t, float, float, unsigned long long, cudaStream_t) {}
    void launch_where(const unsigned char*, const float*, const float*, float*, const size_t*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, size_t, cudaStream_t) {}

} // namespace lfs::core::tensor_ops
