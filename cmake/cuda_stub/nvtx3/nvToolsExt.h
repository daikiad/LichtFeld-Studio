// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub NVTX (NVIDIA Tools Extension) header for CUDA-less builds (Phase 0 macOS
// port). NVTX is profiling instrumentation shipped with the CUDA toolkit; on
// macOS the ranges/marks become no-ops. Only on the include path when
// LFS_ENABLE_CUDA is OFF. See cuda_stub/cuda_runtime.h.
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/nvtx3/nvToolsExt.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include <cstdint>

#define NVTX_VERSION 2
#define NVTX_COLOR_UNKNOWN 0
#define NVTX_COLOR_ARGB 1
#define NVTX_MESSAGE_UNKNOWN 0
#define NVTX_MESSAGE_TYPE_ASCII 1
#define NVTX_MESSAGE_TYPE_UNICODE 2

typedef struct nvtxEventAttributes_t {
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
    } payload;
    int32_t messageType;
    union {
        const char* ascii;
        const wchar_t* unicode;
    } message;
} nvtxEventAttributes_t;

#define NVTX_EVENT_ATTRIB_STRUCT_SIZE ((uint16_t)sizeof(nvtxEventAttributes_t))

inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePush(const char*) { return 0; }
inline int nvtxRangePushEx(const nvtxEventAttributes_t*) { return 0; }
inline int nvtxRangePop(void) { return 0; }
inline void nvtxMarkA(const char*) {}
inline void nvtxMarkEx(const nvtxEventAttributes_t*) {}
