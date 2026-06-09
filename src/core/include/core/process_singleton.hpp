#pragma once

#include "core/export.hpp"

namespace lfs::core {

    // Process-wide single-instance slot keyed by a string. Returns the pointer registered for
    // `key`: the FIRST caller's `candidate` wins, and every later caller in this process — even
    // from a separately dynamically-loaded module (e.g. the lichtfeld Python extension) — gets
    // that same pointer back.
    //
    // This lives in the SHARED lfs_core library on purpose. A Meyers singleton in a library that
    // is STATICALLY linked into more than one binary (on macOS lfs_visualizer is static, so it is
    // baked into BOTH the executable and lichtfeld.so) yields one instance per binary, which
    // silently breaks cross-binary registries (e.g. operators registered from Python landed in the
    // extension's copy while the GUI queried the executable's copy). Routing the singleton through
    // this shared slot collapses those copies back to one. Thread-safe.
    LFS_CORE_API void* process_singleton_get_or_set(const char* key, void* candidate);

} // namespace lfs::core
