/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <concepts>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifdef LFS_EVENT_BRIDGE_EXPORTS
#define LFS_BRIDGE_API __declspec(dllexport)
#else
#define LFS_BRIDGE_API __declspec(dllimport)
#endif
#else
#define LFS_BRIDGE_API __attribute__((visibility("default")))
#endif

namespace lfs::event {

    using HandlerId = size_t;

    template <typename T>
    concept Event = requires {
        typename T::event_id;
    } && std::is_aggregate_v<T>;

    class LFS_BRIDGE_API EventBridge {
    public:
        static EventBridge& instance();

        using Handler = std::function<void(const void*)>;

        // Keyed by the type's mangled name (typeid(E).name()) rather than std::type_index.
        // On macOS the executable and the lichtfeld Python extension each get their own
        // hidden-visibility std::type_info for a header-defined event type, so type_index
        // (pointer-compared by libc++) mismatches across the boundary and a .so-side emit
        // never reaches an exe-side subscriber. The mangled name string is identical in both,
        // so cross-binary dispatch works regardless of RTTI/visibility.
        HandlerId subscribe(const std::string& type, Handler handler);
        void unsubscribe(const std::string& type, HandlerId id);
        void emit(const std::string& type, const void* data);
        size_t handler_count(const std::string& type) const;
        void clear_all();

    private:
        EventBridge() = default;
        EventBridge(const EventBridge&) = delete;
        EventBridge& operator=(const EventBridge&) = delete;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::vector<std::pair<HandlerId, Handler>>> handlers_;
        std::atomic<HandlerId> next_id_{1};
    };

    template <typename E>
    HandlerId when(std::function<void(const E&)> handler) {
        return EventBridge::instance().subscribe(
            typeid(E).name(), [h = std::move(handler)](const void* data) { h(*static_cast<const E*>(data)); });
    }

    template <typename E>
    void emit(const E& event) {
        EventBridge::instance().emit(typeid(E).name(), &event);
    }

    template <typename E>
    size_t subscriber_count() {
        return EventBridge::instance().handler_count(typeid(E).name());
    }

} // namespace lfs::event
