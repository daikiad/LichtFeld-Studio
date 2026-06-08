// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub <cuda_runtime_api.h> for CUDA-less builds (Phase 0 macOS port). The real
// header is the C API subset of cuda_runtime.h; here we just forward to the stub
// runtime. Only on the include path when LFS_ENABLE_CUDA is OFF.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/cuda_runtime_api.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include "cuda_runtime.h"
