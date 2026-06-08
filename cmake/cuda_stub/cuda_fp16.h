// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub <cuda_fp16.h> for CUDA-less builds (Phase 0 macOS port). Only on the
// include path when LFS_ENABLE_CUDA is OFF. Backs __half with clang's native
// _Float16 (available on Apple Silicon) so host code that stores/converts halves
// compiles and runs. See cuda_stub/cuda_runtime.h for the rationale.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/cuda_fp16.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include "cuda_runtime.h"

// 16-bit float, 2 bytes, trivially copyable — matches CUDA __half's storage so
// sizeof()/ptr arithmetic stay correct.
typedef _Float16 __half;
typedef _Float16 half;

struct __half2 {
    __half x;
    __half y;
};

inline float __half2float(__half h) { return static_cast<float>(h); }
inline __half __float2half(float f) { return static_cast<__half>(f); }
inline __half __float2half_rn(float f) { return static_cast<__half>(f); }
inline __half __float2half_rz(float f) { return static_cast<__half>(f); }
inline double __half2double(__half h) { return static_cast<double>(static_cast<float>(h)); }
inline __half __double2half(double d) { return static_cast<__half>(static_cast<float>(d)); }
inline int __half2int_rn(__half h) { return static_cast<int>(static_cast<float>(h)); }

inline __half2 __halves2half2(__half a, __half b) { return __half2{a, b}; }
inline __half2 make_half2(__half a, __half b) { return __half2{a, b}; }
inline __half __low2half(__half2 v) { return v.x; }
inline __half __high2half(__half2 v) { return v.y; }
inline float __low2float(__half2 v) { return static_cast<float>(v.x); }
inline float __high2float(__half2 v) { return static_cast<float>(v.y); }
