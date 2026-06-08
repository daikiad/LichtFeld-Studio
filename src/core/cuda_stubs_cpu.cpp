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

    // ---- SH swizzle helpers --------------------------------------------------
    void shN_swizzled_gather_to_linear(const float*, const int*, float*, std::size_t,
                                       std::uint32_t, std::uint32_t, cudaStream_t) {}
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
