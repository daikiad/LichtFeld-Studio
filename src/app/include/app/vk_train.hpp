/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include <memory>

namespace lfs::app {

    // Headless, no-CUDA Vulkan training entry (macOS/MoltenVK). Bypasses the
    // CUDA-coupled Trainer: loads a COLMAP/NeRF dataset, builds the initial SplatData
    // from its point cloud, and drives the Vulkan backward+Adam path. Selected by the
    // --vk-train flag. Returns a process exit code.
    int runVkTrain(std::unique_ptr<lfs::core::param::TrainingParameters> params);

} // namespace lfs::app
