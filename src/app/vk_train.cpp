/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/vk_train.hpp"

#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"

#include <filesystem>
#include <variant>

namespace lfs::app {

    int runVkTrain(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        const auto& ds = params->dataset;
        if (ds.data_path.empty()) {
            LOG_ERROR("--vk-train requires --data-path");
            return 1;
        }
        LOG_INFO("vk-train: loading dataset {}", ds.data_path.string());

        // --- Load the dataset (CPU/IO, works on macOS no-CUDA) ---
        lfs::io::LoadOptions opts;
        opts.resize_factor = ds.resize_factor;
        opts.max_width = ds.max_width;
        opts.images_folder = ds.images;

        auto loader = lfs::io::Loader::create();
        auto load_result = loader->load(ds.data_path, opts);
        if (!load_result) {
            LOG_ERROR("vk-train: load failed: {}", load_result.error().format());
            return 1;
        }
        auto* scene = std::get_if<lfs::io::LoadedScene>(&load_result->data);
        if (!scene) {
            LOG_ERROR("vk-train: '{}' is not a COLMAP/NeRF dataset (need cameras + images)",
                      ds.data_path.string());
            return 1;
        }
        const std::size_t n_cams = scene->cameras.size();
        const std::int64_t n_pts = scene->point_cloud ? scene->point_cloud->size() : 0;
        LOG_INFO("vk-train: loaded {} cameras, {} points", n_cams, n_pts);
        if (n_cams == 0) {
            LOG_ERROR("vk-train: dataset has no cameras");
            return 1;
        }
        if (!scene->point_cloud || n_pts == 0) {
            LOG_ERROR("vk-train: dataset has no point cloud (random init not yet supported)");
            return 1;
        }

        // --- Build the initial SplatData from the point cloud (CPU swizzle on macOS) ---
        auto sd = lfs::core::init_model_from_pointcloud(
            *params, load_result->scene_center, *scene->point_cloud,
            params->optimization.max_cap, /*tensor_allocator=*/{});
        if (!sd) {
            LOG_ERROR("vk-train: init_model_from_pointcloud failed: {}", sd.error());
            return 1;
        }
        LOG_INFO("vk-train: SplatData initialized: {} splats", sd->size());

        // For now, save the freshly-initialised model so we can confirm the headless
        // no-CUDA data path end-to-end (load -> point cloud -> SplatData -> PLY) without
        // the CUDA Trainer. The Vulkan training loop is wired next.
        std::filesystem::path out_dir = ds.output_path;
        if (out_dir.empty()) {
            out_dir = std::filesystem::current_path() / "vk_train_output";
        }
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);

        lfs::io::PlySaveOptions po;
        po.output_path = out_dir / "vk_train_init.ply";
        auto saved = lfs::io::save_ply(*sd, po);
        if (!saved) {
            LOG_ERROR("vk-train: save_ply failed: {}", saved.error().format());
            return 1;
        }
        LOG_INFO("vk-train: saved init model to {}", po.output_path.string());
        return 0;
    }

} // namespace lfs::app
