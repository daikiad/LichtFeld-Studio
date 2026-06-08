// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub <vector_types.h> for CUDA-less builds (Phase 0 macOS port). Only on the
// include path when LFS_ENABLE_CUDA is OFF. Provides CUDA's builtin vector types
// (float2/3/4, int4, uchar4, dim3, ...) so host headers like helper_math.h
// compile. See cuda_stub/cuda_runtime.h.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/vector_types.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#define LFS_CUDA_VECTOR_TYPE(name, T) \
    struct name##1 { T x; };          \
    struct name##2 { T x, y; };       \
    struct name##3 { T x, y, z; };    \
    struct name##4 { T x, y, z, w; };

LFS_CUDA_VECTOR_TYPE(char, signed char)
LFS_CUDA_VECTOR_TYPE(uchar, unsigned char)
LFS_CUDA_VECTOR_TYPE(short, short)
LFS_CUDA_VECTOR_TYPE(ushort, unsigned short)
LFS_CUDA_VECTOR_TYPE(int, int)
LFS_CUDA_VECTOR_TYPE(uint, unsigned int)
LFS_CUDA_VECTOR_TYPE(long, long)
LFS_CUDA_VECTOR_TYPE(ulong, unsigned long)
LFS_CUDA_VECTOR_TYPE(longlong, long long)
LFS_CUDA_VECTOR_TYPE(ulonglong, unsigned long long)
LFS_CUDA_VECTOR_TYPE(float, float)
LFS_CUDA_VECTOR_TYPE(double, double)

#undef LFS_CUDA_VECTOR_TYPE

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};
