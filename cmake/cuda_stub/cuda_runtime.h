// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// -----------------------------------------------------------------------------
// Stub CUDA Runtime header for CUDA-less builds (Phase 0 macOS port).
//
// This file is ONLY on the include path when LFS_ENABLE_CUDA is OFF. It provides
// just enough of the CUDA Runtime API *surface* for LichtFeld Studio's host code
// to compile and link without a CUDA toolchain. Device allocations fall back to
// host malloc and "device" work is a no-op; real GPU kernels are excluded at the
// CMake level. This is throwaway scaffolding for getting the GUI up on macOS and
// will be replaced by a real compute backend (Metal / Vulkan compute) in later
// phases. Do not add real algorithms here.
// -----------------------------------------------------------------------------
#pragma once

#ifdef LFS_ENABLE_CUDA
#error "cuda_stub/cuda_runtime.h is only for non-CUDA builds (LFS_ENABLE_CUDA must be undefined)"
#endif

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "vector_types.h"
#include "vector_functions.h"

// ---- Device/host function-space qualifiers become no-ops under host clang ----
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __noinline__
#define __noinline__
#endif
#ifndef __restrict__
#define __restrict__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

// ---- Opaque handle types (match the real CUDA definitions exactly) -----------
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

// Texture / array / external-interop handles. Used by the CUDA-Vulkan interop
// layer, which cannot function under MoltenVK; these exist only so that code
// compiles. The interop API calls themselves resolve via dynamic_lookup and are
// never reached on macOS (interop is reported unavailable at runtime).
typedef struct cudaArray* cudaArray_t;
typedef struct cudaArray* cudaArray_const_t;
typedef struct cudaMipmappedArray* cudaMipmappedArray_t;
typedef struct cudaMipmappedArray* cudaMipmappedArray_const_t;
typedef struct CUexternalMemory_st* cudaExternalMemory_t;
typedef struct CUexternalSemaphore_st* cudaExternalSemaphore_t;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorNotReady = 600,
    cudaErrorNotSupported = 801,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidValue = 1,
    cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind;

enum {
    cudaEventDefault = 0x00,
    cudaEventBlockingSync = 0x01,
    cudaEventDisableTiming = 0x02,
    cudaStreamDefault = 0x00,
    cudaStreamNonBlocking = 0x01,
    cudaHostAllocDefault = 0x00,
    cudaHostAllocPortable = 0x01,
    cudaHostAllocMapped = 0x02,
    cudaMemAttachGlobal = 0x01,
    cudaMemAttachHost = 0x02,
};

struct cudaUUID_t {
    char bytes[16];
};

struct cudaDeviceProp {
    char name[256];
    cudaUUID_t uuid;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int major;
    int minor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int warpSize;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
};

typedef enum cudaLimit {
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02,
    cudaLimitDevRuntimeSyncDepth = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04,
} cudaLimit;

struct cudaPointerAttributes {
    int type;
    int device;
    void* devicePointer;
    void* hostPointer;
};

// ---- Runtime API stubs -------------------------------------------------------
inline cudaError_t cudaMalloc(void** p, size_t size) {
    *p = std::malloc(size);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaMallocAsync(void** p, size_t size, cudaStream_t = 0) {
    *p = std::malloc(size);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* p) {
    std::free(p);
    return cudaSuccess;
}
inline cudaError_t cudaFreeAsync(void* p, cudaStream_t = 0) {
    std::free(p);
    return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void** p, size_t size) {
    *p = std::malloc(size);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaHostAlloc(void** p, size_t size, unsigned int) {
    *p = std::malloc(size);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaMallocManaged(void** p, size_t size, unsigned int = cudaMemAttachGlobal) {
    *p = std::malloc(size);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFreeHost(void* p) {
    std::free(p);
    return cudaSuccess;
}

// CUDA's real header provides templated overloads so callers pass T** directly
// (e.g. cudaMalloc(&float_ptr, n)). Mirror that so existing call sites compile.
// Wrapped in extern "C++" because some third-party headers (e.g. ffmpeg's
// libavutil/hwcontext_cuda.h) include <cuda.h> inside an extern "C" block, and
// templates may not have C linkage.
extern "C++" {
template <class T> inline cudaError_t cudaMalloc(T** p, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(p), size);
}
template <class T> inline cudaError_t cudaMallocAsync(T** p, size_t size, cudaStream_t s = 0) {
    return cudaMallocAsync(reinterpret_cast<void**>(p), size, s);
}
template <class T> inline cudaError_t cudaMallocHost(T** p, size_t size) {
    return cudaMallocHost(reinterpret_cast<void**>(p), size);
}
template <class T> inline cudaError_t cudaMallocManaged(T** p, size_t size, unsigned int f = cudaMemAttachGlobal) {
    return cudaMallocManaged(reinterpret_cast<void**>(p), size, f);
}
template <class T> inline cudaError_t cudaHostAlloc(T** p, size_t size, unsigned int f) {
    return cudaHostAlloc(reinterpret_cast<void**>(p), size, f);
}
} // extern "C++"
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (n) std::memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t = 0) {
    if (n) std::memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    if (n) std::memset(p, v, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = 0) {
    if (n) std::memset(p, v, n);
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned int, int) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int = 0) { return cudaSuccess; }
inline cudaError_t cudaStreamQuery(cudaStream_t) { return cudaSuccess; }

inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { if (ms) *ms = 0.0f; return cudaSuccess; }

inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA disabled (macOS stub)"; }
inline const char* cudaGetErrorName(cudaError_t) { return "cudaErrorStub"; }

inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { if (d) *d = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* c) { if (c) *c = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    if (p) { std::memset(p, 0, sizeof(*p)); std::strcpy(p->name, "CUDA disabled (macOS stub)"); }
    return cudaSuccess;
}
inline cudaError_t cudaDeviceGetAttribute(int* v, int, int) { if (v) *v = 0; return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* freeMem, size_t* totalMem) {
    if (freeMem) *freeMem = 0;
    if (totalMem) *totalMem = 0;
    return cudaSuccess;
}
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a, const void*) {
    if (a) { a->type = 0; a->device = 0; a->devicePointer = nullptr; a->hostPointer = nullptr; }
    return cudaSuccess;
}
inline cudaError_t cudaRuntimeGetVersion(int* v) { if (v) *v = 0; return cudaSuccess; }
inline cudaError_t cudaDriverGetVersion(int* v) { if (v) *v = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceSetLimit(cudaLimit, size_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceGetLimit(size_t* v, cudaLimit) { if (v) *v = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceGetPCIBusId(char* buf, int len, int) {
    if (buf && len > 0) buf[0] = '\0';
    return cudaSuccess;
}

// Host-callback enqueue. With no real stream we run it synchronously so any
// dependent host logic still progresses.
typedef void (*cudaHostFn_t)(void*);
inline cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t fn, void* userData) {
    if (fn) fn(userData);
    return cudaSuccess;
}

// ---- External-memory / semaphore interop (CUDA<->Vulkan) --------------------
// These exist so the CUDA-Vulkan interop layer (cuda_vulkan_interop.cpp) and its
// consumers (e.g. VulkanUiTexture) compile and link. The interop path is never
// engaged on macOS (externalMemoryInteropEnabled() is false), so these no-ops are
// not reached for real work; they just keep object construction/teardown valid.
typedef enum cudaExternalMemoryHandleType {
    cudaExternalMemoryHandleTypeOpaqueFd = 1,
    cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalMemoryHandleTypeD3D12Heap = 4,
    cudaExternalMemoryHandleTypeD3D12Resource = 5,
} cudaExternalMemoryHandleType;

typedef enum cudaExternalSemaphoreHandleType {
    cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 7,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 8,
} cudaExternalSemaphoreHandleType;

typedef enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
} cudaChannelFormatKind;

typedef enum cudaResourceType {
    cudaResourceTypeArray = 0,
    cudaResourceTypeMipmappedArray = 1,
    cudaResourceTypeLinear = 2,
    cudaResourceTypePitch2D = 3,
} cudaResourceType;

enum {
    cudaExternalMemoryDedicated = 0x1,
    cudaArraySurfaceLoadStore = 0x2,
    cudaArrayDefault = 0x0,
};

// cudaDeviceGetAttribute attribute selector (only the one the interop layer queries).
enum { cudaDevAttrTimelineSemaphoreInteropSupported = 114 };

struct cudaExtent {
    size_t width;
    size_t height;
    size_t depth;
};

inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
    return cudaExtent{w, h, d};
}

struct cudaChannelFormatDesc {
    int x, y, z, w;
    cudaChannelFormatKind f;
};

struct cudaExternalMemoryHandleDesc {
    cudaExternalMemoryHandleType type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
    } handle;
    unsigned long long size;
    unsigned int flags;
};

struct cudaExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
};

struct cudaExternalMemoryMipmappedArrayDesc {
    unsigned long long offset;
    cudaChannelFormatDesc formatDesc;
    cudaExtent extent;
    unsigned int numLevels;
    unsigned int flags;
};

struct cudaExternalSemaphoreHandleDesc {
    cudaExternalSemaphoreHandleType type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
    } handle;
    unsigned int flags;
};

struct cudaExternalSemaphoreSignalParams {
    struct {
        struct {
            unsigned long long value;
        } fence;
        unsigned long long reserved;
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

struct cudaExternalSemaphoreWaitParams {
    struct {
        struct {
            unsigned long long value;
        } fence;
        unsigned long long reserved;
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

struct cudaResourceDesc {
    cudaResourceType resType;
    union {
        struct {
            cudaArray_t array;
        } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
    } res;
};

inline cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* mem, const cudaExternalMemoryHandleDesc*) {
    if (mem) *mem = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaExternalMemoryGetMappedBuffer(void** ptr, cudaExternalMemory_t, const cudaExternalMemoryBufferDesc*) {
    if (ptr) *ptr = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* arr, cudaExternalMemory_t, const cudaExternalMemoryMipmappedArrayDesc*) {
    if (arr) *arr = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t) { return cudaSuccess; }
inline cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* sem, const cudaExternalSemaphoreHandleDesc*) {
    if (sem) *sem = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t*, const cudaExternalSemaphoreSignalParams*, unsigned int, cudaStream_t = 0) {
    return cudaSuccess;
}
inline cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t*, const cudaExternalSemaphoreWaitParams*, unsigned int, cudaStream_t = 0) {
    return cudaSuccess;
}
inline cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t) { return cudaSuccess; }

inline cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) {
    cudaChannelFormatDesc d;
    d.x = x;
    d.y = y;
    d.z = z;
    d.w = w;
    d.f = f;
    return d;
}
inline cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* level, cudaMipmappedArray_const_t, unsigned int) {
    if (level) *level = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t) { return cudaSuccess; }
inline cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* obj, const cudaResourceDesc*) {
    if (obj) *obj = 0;
    return cudaSuccess;
}
inline cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t) { return cudaSuccess; }
