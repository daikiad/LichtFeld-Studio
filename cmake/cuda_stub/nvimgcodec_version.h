// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stub nvImageCodec version header for CUDA-less builds (Phase 0 macOS port).
// nvImageCodec (NVIDIA, GPU JPEG decode) is not built on macOS; its generated
// version header is therefore absent. This stub lets the vendored nvimgcodec.h
// API declarations parse. The nvImageCodec runtime path is unavailable on macOS
// (image decode falls back to OpenImageIO). Only on the include path when
// LFS_ENABLE_CUDA is OFF.
#pragma once

#define NVIMGCODEC_VER_MAJOR 0
#define NVIMGCODEC_VER_MINOR 0
#define NVIMGCODEC_VER_PATCH 0
#define NVIMGCODEC_VER_BUILD 0
#define NVIMGCODEC_VER (NVIMGCODEC_VER_MAJOR * 10000 + NVIMGCODEC_VER_MINOR * 100 + NVIMGCODEC_VER_PATCH)
