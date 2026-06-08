// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub <vector_functions.h> for CUDA-less builds (Phase 0 macOS port). Provides
// the canonical per-component make_<type>N() constructors that CUDA's real header
// supplies (helper_math.h and friends build their overloads on top of these).
// Only on the include path when LFS_ENABLE_CUDA is OFF. See cuda_stub/cuda_runtime.h.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/vector_functions.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include "vector_types.h"

#define LFS_CUDA_MAKE_VEC(name, T)                                                   \
    inline name##1 make_##name##1(T x) { name##1 r; r.x = x; return r; }             \
    inline name##2 make_##name##2(T x, T y) { name##2 r; r.x = x; r.y = y; return r; } \
    inline name##3 make_##name##3(T x, T y, T z) { name##3 r; r.x = x; r.y = y; r.z = z; return r; } \
    inline name##4 make_##name##4(T x, T y, T z, T w) { name##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

LFS_CUDA_MAKE_VEC(char, signed char)
LFS_CUDA_MAKE_VEC(uchar, unsigned char)
LFS_CUDA_MAKE_VEC(short, short)
LFS_CUDA_MAKE_VEC(ushort, unsigned short)
LFS_CUDA_MAKE_VEC(int, int)
LFS_CUDA_MAKE_VEC(uint, unsigned int)
LFS_CUDA_MAKE_VEC(long, long)
LFS_CUDA_MAKE_VEC(ulong, unsigned long)
LFS_CUDA_MAKE_VEC(longlong, long long)
LFS_CUDA_MAKE_VEC(ulonglong, unsigned long long)
LFS_CUDA_MAKE_VEC(float, float)
LFS_CUDA_MAKE_VEC(double, double)

#undef LFS_CUDA_MAKE_VEC
