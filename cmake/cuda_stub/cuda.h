// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub CUDA Driver API header for CUDA-less builds (Phase 0 macOS port). Only on
// the include path when LFS_ENABLE_CUDA is OFF. Provides the handful of driver
// types referenced by headers that are compiled on the host (e.g. the
// RasterizerMemoryArena VMM interface). The VMM implementation itself lives in a
// .cu that is excluded from CUDA-less builds. See cuda_stub/cuda_runtime.h.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/cuda.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include "cuda_runtime.h"

typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;

enum { CUDA_SUCCESS = 0 };

inline CUresult cuCtxGetCurrent(CUcontext* pctx) { if (pctx) *pctx = nullptr; return CUDA_SUCCESS; }
inline CUresult cuCtxSetCurrent(CUcontext) { return CUDA_SUCCESS; }
inline CUresult cuInit(unsigned int) { return CUDA_SUCCESS; }
