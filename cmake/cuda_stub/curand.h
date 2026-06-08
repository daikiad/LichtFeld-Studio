// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub host cuRAND header for CUDA-less builds (Phase 0 macOS port). Only on the
// include path when LFS_ENABLE_CUDA is OFF. The generators fall back to a host
// std::mt19937 so randn/uniform tensors are still populated with plausible data.
// See cuda_stub/cuda_runtime.h for the rationale.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/curand.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include "cuda_runtime.h"

#include <cstddef>
#include <random>

typedef enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_INTERNAL_ERROR = 999,
} curandStatus_t;

typedef enum curandRngType {
    CURAND_RNG_PSEUDO_DEFAULT = 100,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
    CURAND_RNG_PSEUDO_XORWOW = 101,
} curandRngType_t;

typedef struct curandGenerator_st* curandGenerator_t;

namespace lfs_cuda_stub {
    inline std::mt19937_64& host_rng() {
        static thread_local std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
        return rng;
    }
}

inline curandStatus_t curandCreateGenerator(curandGenerator_t* g, curandRngType_t) {
    *g = nullptr;
    return CURAND_STATUS_SUCCESS;
}
inline curandStatus_t curandDestroyGenerator(curandGenerator_t) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t, unsigned long long seed) {
    lfs_cuda_stub::host_rng().seed(seed);
    return CURAND_STATUS_SUCCESS;
}
inline curandStatus_t curandSetGeneratorOffset(curandGenerator_t, unsigned long long) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandSetStream(curandGenerator_t, cudaStream_t) { return CURAND_STATUS_SUCCESS; }

inline curandStatus_t curandGenerateNormal(curandGenerator_t, float* out, size_t n, float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) out[i] = dist(lfs_cuda_stub::host_rng());
    return CURAND_STATUS_SUCCESS;
}
inline curandStatus_t curandGenerateUniform(curandGenerator_t, float* out, size_t n) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) out[i] = dist(lfs_cuda_stub::host_rng());
    return CURAND_STATUS_SUCCESS;
}
