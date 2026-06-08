/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU/no-op fallbacks for PPISPControllerPool, whose CUDA implementation
// (ppisp_controller_pool.cu) is excluded from CUDA-less builds. Lets lfs_training
// link and the Python module import on macOS (Phase 0). Per-pixel appearance
// prediction is a training feature; it returns the input unchanged here (Phase 1).
// Only compiled when LFS_ENABLE_CUDA is OFF.

#include "components/ppisp_controller_pool.hpp"

#include "core/tensor.hpp"

namespace lfs::training {

    PPISPControllerPool::PPISPControllerPool(int, int, Config) {}

    void PPISPControllerPool::allocate_buffers(size_t, size_t) {}

    lfs::core::Tensor PPISPControllerPool::predict(int, const lfs::core::Tensor& rendered_rgb, float) {
        return rendered_rgb;
    }

} // namespace lfs::training
