#include "core/process_singleton.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace lfs::core {

    void* process_singleton_get_or_set(const char* key, void* candidate) {
        // Defined once in the shared lfs_core library, so this map is a single process-wide
        // instance regardless of how many binaries statically link their callers.
        static std::mutex mutex;
        static std::unordered_map<std::string, void*> slots;
        std::lock_guard<std::mutex> lock(mutex);
        auto [it, inserted] = slots.try_emplace(key ? key : "", candidate);
        (void)inserted;
        return it->second;
    }

} // namespace lfs::core
