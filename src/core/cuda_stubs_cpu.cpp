/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU/no-op fallbacks for the core CUDA entry points whose .cu definitions are
// excluded from CUDA-less builds (memory arena / exportable device blocks /
// selection ops / SH swizzle). They let lfs_core link and the Python module import
// on macOS (Phase 0). They do not perform real GPU work yet (Phase 1 backend).
// Only compiled when LFS_ENABLE_CUDA is OFF.

#include "core/cuda/memory_arena.hpp"

#include "core/cuda/selection_ops.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/exportable_storage.hpp"
#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

// lfs_core is built with hidden default visibility and these CUDA entry points are
// not tagged LFS_CORE_API, so the (gated) .cu would normally provide them. The
// Python module resolves them from liblfs_core.dylib via flat-namespace lookup, which
// only sees exported symbols — force these stubs to be exported.
#pragma GCC visibility push(default)

namespace lfs::core {

    // ---- RasterizerMemoryArena (CUDA VMM arena; inert without CUDA) -----------
    // The arena exists only to back the CUDA<->Vulkan zero-copy rasterizer scratch.
    // Without CUDA it never holds storage, so every method is a no-op / empty default.
    // These must all be defined: with -undefined dynamic_lookup an *undefined* method
    // binds to 0x0 and crashes when called (e.g. Scene::clear -> full_reset()).
    RasterizerMemoryArena::RasterizerMemoryArena() {}
    RasterizerMemoryArena::RasterizerMemoryArena(const Config&) {}
    RasterizerMemoryArena::~RasterizerMemoryArena() {}
    RasterizerMemoryArena::RasterizerMemoryArena(RasterizerMemoryArena&&) noexcept {}
    RasterizerMemoryArena& RasterizerMemoryArena::operator=(RasterizerMemoryArena&&) noexcept { return *this; }

    uint64_t RasterizerMemoryArena::begin_frame(bool) { return 0; }
    std::optional<uint64_t> RasterizerMemoryArena::try_begin_frame(bool) { return std::nullopt; }
    void RasterizerMemoryArena::end_frame(uint64_t, bool) {}
    std::function<char*(size_t)> RasterizerMemoryArena::get_allocator(uint64_t) {
        return [](size_t) -> char* { return nullptr; };
    }
    std::vector<RasterizerMemoryArena::BufferHandle> RasterizerMemoryArena::get_frame_buffers(uint64_t) const { return {}; }
    void RasterizerMemoryArena::reset_frame(uint64_t) {}
    void RasterizerMemoryArena::cleanup_frames(int) {}
    void RasterizerMemoryArena::full_reset() {}
    bool RasterizerMemoryArena::install_external_backing(ExternalBacking) { return false; }
    bool RasterizerMemoryArena::try_install_external_backing(ExternalBacking) { return false; }
    bool RasterizerMemoryArena::grow_external_backing(const void*, size_t, const std::function<bool(size_t)>&) { return false; }
    void RasterizerMemoryArena::clear_external_backing(const void*) {}
    bool RasterizerMemoryArena::using_external_backing() const { return false; }
    RasterizerMemoryArena::Statistics RasterizerMemoryArena::get_statistics() const { return {}; }
    RasterizerMemoryArena::MemoryInfo RasterizerMemoryArena::get_memory_info() const { return {}; }
    void RasterizerMemoryArena::dump_statistics() const {}
    void RasterizerMemoryArena::log_memory_status(uint64_t, bool) {}
    bool RasterizerMemoryArena::is_under_memory_pressure() const { return false; }
    float RasterizerMemoryArena::get_memory_pressure() const { return 0.0f; }
    bool RasterizerMemoryArena::is_rendering_active() const { return false; }
    void RasterizerMemoryArena::set_rendering_active(bool) {}

    // ---- GlobalArenaManager --------------------------------------------------
    GlobalArenaManager& GlobalArenaManager::instance() {
        static GlobalArenaManager s;
        return s;
    }
    RasterizerMemoryArena& GlobalArenaManager::get_arena() {
        if (!arena_)
            arena_ = std::make_unique<RasterizerMemoryArena>();
        return *arena_;
    }
    RasterizerMemoryArena* GlobalArenaManager::try_get_arena() { return arena_.get(); }
    bool GlobalArenaManager::install_external_backing(RasterizerMemoryArena::ExternalBacking) { return false; }
    bool GlobalArenaManager::try_install_external_backing(RasterizerMemoryArena::ExternalBacking) { return false; }
    bool GlobalArenaManager::grow_external_backing(const void*, size_t, const std::function<bool(size_t)>&) { return false; }
    void GlobalArenaManager::clear_external_backing(const void*) {}
    void GlobalArenaManager::reset() { arena_.reset(); }

    // ---- Exportable device blocks (CUDA VMM) ---------------------------------
    std::expected<std::shared_ptr<ExportableBlock>, std::string>
    allocateExportableDeviceBlock(std::size_t, int, bool, std::size_t) {
        return std::unexpected(std::string("Exportable device blocks require CUDA (unavailable on macOS)"));
    }
    std::expected<bool, std::string>
    growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>&, std::size_t) {
        return false;
    }

    // ---- SH swizzle helpers (CPU equivalents of the CUDA kernels) ------------
    // The swizzled shN layout math lives in sh_layout.cuh (sh_swizzled_index). These
    // mirror the proven CPU paths extract_sh_coefficients_to_swizzled_host
    // (io/formats/ply.cpp) and SplatData::shN_canonical_cpu (core/splat_data.cpp), so
    // the buffer they produce/consume is byte-identical to what the PLY loader writes
    // (which already renders correctly on macOS). Without these, the SOG / canonical
    // load path (SplatData ctor -> reorder) and the deswizzle accessors (shN_canonical,
    // get_shs) call undefined symbols and crash. CUDA memory under the stub is host RAM,
    // so these operate directly on the tensor pointers.

    // canonical [n, src_coeffs_rest, 3] (row-major) -> swizzled (layout_coeffs_rest slots).
    void reorder_sh_to_swizzled(const float* src_canonical, float* dst_swizzled,
                                std::size_t n_primitives, std::uint32_t src_coeffs_rest,
                                std::uint32_t layout_coeffs_rest, cudaStream_t) {
        if (!src_canonical || !dst_swizzled || n_primitives == 0 ||
            src_coeffs_rest == 0 || layout_coeffs_rest == 0)
            return;
        const std::size_t row = static_cast<std::size_t>(src_coeffs_rest) * kShChannels;
        for (std::size_t p = 0; p < n_primitives; ++p) {
            const float* const src_row = src_canonical + p * row;
            for (std::size_t off = 0; off < row; ++off) {
                const auto slot = static_cast<std::uint32_t>(off / 4u);
                const auto comp = static_cast<std::uint32_t>(off % 4u);
                const std::size_t dst = static_cast<std::size_t>(sh_swizzled_index(
                                            static_cast<std::uint32_t>(p), slot, layout_coeffs_rest)) *
                                            4u +
                                        comp;
                dst_swizzled[dst] = src_row[off];
            }
        }
    }
    void reorder_sh_to_swizzled(const float* src, float* dst, std::size_t n,
                                std::uint32_t active_coeffs_rest, cudaStream_t s) {
        reorder_sh_to_swizzled(src, dst, n, active_coeffs_rest, active_coeffs_rest, s);
    }

    // swizzled (layout_coeffs_rest slots) -> canonical [n, dst_coeffs_rest, 3].
    void undo_reorder_sh_from_swizzled(const float* src_swizzled, float* dst_canonical,
                                       std::size_t n_primitives, std::uint32_t dst_coeffs_rest,
                                       std::uint32_t layout_coeffs_rest, cudaStream_t) {
        if (!src_swizzled || !dst_canonical || n_primitives == 0 ||
            dst_coeffs_rest == 0 || layout_coeffs_rest == 0)
            return;
        const std::size_t row = static_cast<std::size_t>(dst_coeffs_rest) * kShChannels;
        for (std::size_t p = 0; p < n_primitives; ++p) {
            float* const dst_row = dst_canonical + p * row;
            for (std::size_t off = 0; off < row; ++off) {
                const auto slot = static_cast<std::uint32_t>(off / 4u);
                const auto comp = static_cast<std::uint32_t>(off % 4u);
                const std::size_t src = static_cast<std::size_t>(sh_swizzled_index(
                                            static_cast<std::uint32_t>(p), slot, layout_coeffs_rest)) *
                                            4u +
                                        comp;
                dst_row[off] = src_swizzled[src];
            }
        }
    }
    void undo_reorder_sh_from_swizzled(const float* src, float* dst, std::size_t n,
                                       std::uint32_t active_coeffs_rest, cudaStream_t s) {
        undo_reorder_sh_from_swizzled(src, dst, n, active_coeffs_rest, active_coeffs_rest, s);
    }

    // Gather selected primitives from swizzled storage into contiguous canonical rows
    // [n_src, dst_coeffs_rest, 3]. (Densification read path; previously a no-op stub
    // that silently returned zeros.)
    template <typename IndexT>
    static void sh_gather_to_linear_impl(const float* src_swizzled, const IndexT* src_indices,
                                         float* dst_linear, std::size_t n_src,
                                         std::uint32_t dst_coeffs_rest, std::uint32_t layout_coeffs_rest) {
        if (!src_swizzled || !src_indices || !dst_linear || n_src == 0 ||
            dst_coeffs_rest == 0 || layout_coeffs_rest == 0)
            return;
        const std::size_t row = static_cast<std::size_t>(dst_coeffs_rest) * kShChannels;
        for (std::size_t i = 0; i < n_src; ++i) {
            const auto p = static_cast<std::uint32_t>(src_indices[i]);
            float* const dst_row = dst_linear + i * row;
            for (std::size_t off = 0; off < row; ++off) {
                const auto slot = static_cast<std::uint32_t>(off / 4u);
                const auto comp = static_cast<std::uint32_t>(off % 4u);
                const std::size_t src = static_cast<std::size_t>(sh_swizzled_index(p, slot, layout_coeffs_rest)) *
                                            4u +
                                        comp;
                dst_row[off] = src_swizzled[src];
            }
        }
    }
    void shN_swizzled_gather_to_linear(const float* src, const int* idx, float* dst,
                                       std::size_t n, std::uint32_t dst_rest,
                                       std::uint32_t layout_rest, cudaStream_t) {
        sh_gather_to_linear_impl(src, idx, dst, n, dst_rest, layout_rest);
    }
    void shN_swizzled_gather_to_linear(const float* src, const int* idx, float* dst,
                                       std::size_t n, std::uint32_t active_rest, cudaStream_t s) {
        shN_swizzled_gather_to_linear(src, idx, dst, n, active_rest, active_rest, s);
    }
    void shN_swizzled_gather_to_linear_i64(const float* src, const std::int64_t* idx, float* dst,
                                           std::size_t n, std::uint32_t dst_rest,
                                           std::uint32_t layout_rest, cudaStream_t) {
        sh_gather_to_linear_impl(src, idx, dst, n, dst_rest, layout_rest);
    }
    void shN_swizzled_gather_to_linear_i64(const float* src, const std::int64_t* idx, float* dst,
                                           std::size_t n, std::uint32_t active_rest, cudaStream_t s) {
        shN_swizzled_gather_to_linear_i64(src, idx, dst, n, active_rest, active_rest, s);
    }

    // Legacy VkSplat packed (16-coeff) buffer — the live renderer reads the split
    // buffers and sets the packed input to {}, so a no-op is sufficient here.
    void sh_swizzled_pack_full_from_split(const float*, const float*, float*, std::size_t,
                                          std::uint32_t, cudaStream_t) {}

} // namespace lfs::core

namespace lfs::core::cuda {

    // ---- Selection ops (return empty masks) ----------------------------------
    Tensor selection_grow(const Tensor&, const Tensor&, float, uint8_t) { return Tensor{}; }
    Tensor selection_shrink(const Tensor&, const Tensor&, float) { return Tensor{}; }
    Tensor select_by_opacity(const Tensor&, float, float, uint8_t) { return Tensor{}; }
    Tensor select_by_scale(const Tensor&, float, uint8_t) { return Tensor{}; }
    Tensor select_by_color(const Tensor&, float, float, float, float, uint8_t) { return Tensor{}; }

} // namespace lfs::core::cuda

// ---- UltraHDR (libuhdr) stubs --------------------------------------------------
// The vcpkg OpenImageIO arm64-osx build references the UltraHDR C API but no
// libuhdr is provided in the dependency set, leaving these symbols undefined.
// In an executable they bind lazily and are never reached (no UHDR images), but a
// dlopen'd .so (the Python module) fails to load on the missing flat symbol. Provide
// name-matching exported stubs so the module loads; they report "not a UHDR image" /
// failure and are never exercised on the no-UHDR path. C linkage => name-only match.
extern "C" {
    int is_uhdr_image(void*, int) { return 0; }
    void* uhdr_create_decoder(void) { return nullptr; }
    int uhdr_dec_set_image(void*, void*) { return -1; }
    int uhdr_decode(void*) { return -1; }
    void* uhdr_get_decoded_image(void*) { return nullptr; }
    void uhdr_release_decoder(void*) {}
}

#pragma GCC visibility pop
