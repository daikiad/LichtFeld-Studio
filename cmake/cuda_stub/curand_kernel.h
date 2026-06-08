// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub device cuRAND header for CUDA-less builds (Phase 0 macOS port). Only on
// the include path when LFS_ENABLE_CUDA is OFF. Device RNG state/functions exist
// solely so host TUs that include this header compile; the device paths live in
// .cu files excluded from CUDA-less builds. See cuda_stub/cuda_runtime.h.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/curand_kernel.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include "cuda_runtime.h"

struct curandStateXORWOW {
    unsigned int d[6];
};
typedef struct curandStateXORWOW curandStateXORWOW_t;
typedef struct curandStateXORWOW curandState;
typedef struct curandStateXORWOW curandState_t;

struct curandStatePhilox4_32_10 {
    unsigned int d[16];
};
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

template <class State> __device__ inline void curand_init(unsigned long long, unsigned long long, unsigned long long, State*) {}
template <class State> __device__ inline float curand_uniform(State*) { return 0.0f; }
template <class State> __device__ inline float curand_normal(State*) { return 0.0f; }
template <class State> __device__ inline double curand_uniform_double(State*) { return 0.0; }
template <class State> __device__ inline double curand_normal_double(State*) { return 0.0; }
