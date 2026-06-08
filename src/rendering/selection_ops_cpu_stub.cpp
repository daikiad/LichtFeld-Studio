/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU/no-op fallbacks for the tensor-based selection ops whose CUDA definitions
// (selection_ops.cu) are excluded from CUDA-less builds. They let lfs_rendering_tensor
// link and the Python module import on macOS (Phase 0). Selection does nothing yet —
// a real CPU/Metal implementation is Phase 1. Only compiled when LFS_ENABLE_CUDA is OFF.

#include "selection_ops.hpp"

#include "core/tensor.hpp"

#include <array>
#include <vector>

namespace lfs::rendering {

    void set_selection_element(bool* selection, int index, bool value) {
        if (selection)
            selection[index] = value;
    }

    Tensor project_screen_positions_tensor(const Tensor&, int, int,
                                           const std::array<float, 9>&, const std::array<float, 3>&,
                                           float, float, bool, float,
                                           const Tensor*, const Tensor*, const std::vector<bool>&) {
        return Tensor{};
    }

    int pick_projected_gaussian_tensor(const Tensor&, float, float, float) { return -1; }

    void brush_select_tensor(const Tensor&, float, float, float, Tensor&) {}
    void rect_select_tensor(const Tensor&, float, float, float, float, Tensor&) {}
    void polygon_select_tensor(const Tensor&, const Tensor&, Tensor&) {}

    void apply_selection_group_tensor_mask(const Tensor&, const Tensor&, Tensor&, uint8_t,
                                           const uint32_t*, bool, const Tensor*,
                                           const std::vector<bool>&, bool, Tensor*) {}

    void apply_selection_group_indexed_tensor_mask(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                                   uint8_t, const uint32_t*, bool, const Tensor*,
                                                   const std::vector<bool>&, bool) {}

    std::array<size_t, 256> count_selection_groups(const Tensor&, Tensor&) { return {}; }

    SelectionGroupDeltaResult read_selection_group_delta_result(const Tensor&) { return {}; }

    void merge_selection_mask_or(Tensor&, const Tensor&) {}

    void filter_selection_by_node_mask(Tensor&, const Tensor&, const std::vector<bool>&) {}

    void filter_selection_by_crop(Tensor&, const Tensor&, const Tensor*, const Tensor*, const Tensor*,
                                  bool, const Tensor*, const Tensor*, bool, const Tensor*, const Tensor*) {}

    namespace config {
        void setSelectionGroupColor(int, float3) {}
        void setSelectionPreviewColor(float3) {}
    } // namespace config

} // namespace lfs::rendering
