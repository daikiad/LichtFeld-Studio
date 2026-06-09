/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/vk_train.hpp"

#include "core/camera.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "rendering/split_view_service.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "window/window_manager.hpp"

#include <cstdint>
#include <filesystem>
#include <variant>
#include <vector>

namespace lfs::app {

    int runVkTrain(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        const auto& ds = params->dataset;
        if (ds.data_path.empty()) {
            LOG_ERROR("--vk-train requires --data-path");
            return 1;
        }

        // The image loader backs Camera::load_and_get_image; it must be installed before
        // any camera image is requested (mirrors Application::run).
        lfs::io::CacheLoader::getInstance(
            ds.loading_params.use_cpu_memory, ds.loading_params.use_fs_cache);
        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8});
        });

        LOG_INFO("vk-train: loading dataset {}", ds.data_path.string());
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

        auto sd = lfs::core::init_model_from_pointcloud(
            *params, load_result->scene_center, *scene->point_cloud,
            params->optimization.max_cap, /*tensor_allocator=*/{});
        if (!sd) {
            LOG_ERROR("vk-train: init_model_from_pointcloud failed: {}", sd.error());
            return 1;
        }
        LOG_INFO("vk-train: SplatData initialized: {} splats, max SH degree {}",
                 sd->size(), sd->get_max_sh_degree());

        // Vulkan context (hidden window; no swapchain presentation needed).
        lfs::vis::WindowManager wm("vk-train", 1280, 720);
        if (!wm.init()) {
            LOG_ERROR("vk-train: failed to initialize Vulkan window/context");
            return 1;
        }
        lfs::vis::VulkanContext* ctx = wm.getVulkanContext();
        if (!ctx) {
            LOG_ERROR("vk-train: no Vulkan context");
            return 1;
        }

        // Build per-camera ground-truth (HWC float4, normalized) + render requests.
        std::vector<lfs::rendering::ViewportRenderRequest> requests;
        std::vector<std::vector<float>> gts;
        requests.reserve(n_cams);
        gts.reserve(n_cams);
        const int sh_deg = sd->get_max_sh_degree();
        for (const auto& cam : scene->cameras) {
            // load_and_get_image updates the camera's image dimensions (resize/cap).
            auto img = cam->load_and_get_image(ds.resize_factor, ds.max_width,
                                               /*output_uint8=*/true, /*update_dimensions=*/true)
                           .cpu();
            const int W = cam->image_width();
            const int H = cam->image_height();
            if (W <= 0 || H <= 0 || !img.is_valid()) {
                LOG_WARN("vk-train: skipping camera '{}' (no image)", cam->image_name());
                continue;
            }
            const std::uint8_t* g = img.ptr<std::uint8_t>(); // CHW: [c*H*W + y*W + x]
            std::vector<float> gt4(static_cast<std::size_t>(4) * H * W, 0.0f);
            const std::size_t plane = static_cast<std::size_t>(H) * W;
            for (std::size_t p = 0; p < plane; ++p) {
                gt4[4 * p + 0] = g[0 * plane + p] / 255.0f;
                gt4[4 * p + 1] = g[1 * plane + p] / 255.0f;
                gt4[4 * p + 2] = g[2 * plane + p] / 255.0f;
            }

            auto rc = lfs::vis::detail::buildGTRenderCamera(*cam, glm::ivec2{W, H}, glm::mat4(1.0f));
            if (!rc) {
                LOG_WARN("vk-train: skipping camera '{}' (no render camera)", cam->image_name());
                continue;
            }

            // Render in DATA (COLMAP) world space so the raw SplatData means line up with
            // the dataset cameras. buildGTRenderCamera bakes a data->visualizer axis flip
            // into the pose for the GUI (which applies the same flip to the splats via the
            // scene-node transform); our backward optimizes the RAW means with an identity
            // model transform, so we must NOT flip. Reconstruct the pose so getViewMatrix()
            // reproduces the dataset's world->camera matrix directly.
            // cam.R() is the camera->world rotation (loader stores transpose(w2c)); cam.T()
            // is the world->camera translation. frame_view.rotation/translation are the
            // camera->world pose makeViewMatrix() inverts, so: rotation = R_c2w,
            // translation = camera position = -R_c2w * T.
            auto R_cpu = cam->R().cpu().contiguous();
            auto T_cpu = cam->T().cpu().contiguous();
            const glm::mat3 R_c2w = lfs::rendering::mat3FromRowMajor3x3(R_cpu.ptr<float>());
            const glm::vec3 Tw(T_cpu.ptr<float>()[0], T_cpu.ptr<float>()[1], T_cpu.ptr<float>()[2]);
            const glm::vec3 cam_pos = -(R_c2w * Tw);

            lfs::rendering::ViewportRenderRequest req;
            req.frame_view.size = glm::ivec2{W, H};
            req.frame_view.rotation = R_c2w;
            req.frame_view.translation = cam_pos;
            req.frame_view.intrinsics_override = rc->intrinsics;
            req.frame_view.background_color = glm::vec3(0.0f);
            req.transparent_background = false;
            req.equirectangular = rc->equirectangular;
            req.sh_degree = sh_deg;
            requests.push_back(std::move(req));
            gts.push_back(std::move(gt4));
        }
        if (requests.empty()) {
            LOG_ERROR("vk-train: no usable cameras with images");
            return 1;
        }

        // Diagnostics: cam0 pose/intrinsics + model spatial extent (to sanity-check that
        // splats and the camera live in the same space).
        {
            const auto& fv0 = requests[0].frame_view;
            const auto in0 = fv0.intrinsics_override.value_or(lfs::rendering::CameraIntrinsics{});
            LOG_INFO("vk-train: cam0 size={}x{} fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f} t=({:.2f},{:.2f},{:.2f})",
                     fv0.size.x, fv0.size.y, in0.focal_x, in0.focal_y, in0.center_x, in0.center_y,
                     fv0.translation.x, fv0.translation.y, fv0.translation.z);
            auto m = sd->means().cpu().contiguous();
            const float* mp = m.ptr<float>();
            const std::int64_t nm = m.numel() / 3;
            float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
            for (std::int64_t i = 0; i < nm; ++i)
                for (int c = 0; c < 3; ++c) {
                    const float v = mp[3 * i + c];
                    lo[c] = std::min(lo[c], v);
                    hi[c] = std::max(hi[c], v);
                }
            LOG_INFO("vk-train: means bbox x[{:.2f},{:.2f}] y[{:.2f},{:.2f}] z[{:.2f},{:.2f}]",
                     lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        }

        const int iters = static_cast<int>(params->optimization.iterations);
        LOG_INFO("vk-train: training {} cameras for {} iterations", requests.size(), iters);

        lfs::vis::VksplatViewportRenderer vk;
        auto trained = vk.runMultiCameraTraining(*ctx, *sd, requests, gts, iters);
        if (!trained) {
            LOG_ERROR("vk-train: training failed: {}", trained.error());
            return 1;
        }

        // Save the trained model (params were read back into *sd).
        std::filesystem::path out_dir = ds.output_path;
        if (out_dir.empty()) {
            out_dir = std::filesystem::current_path() / "vk_train_output";
        }
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        lfs::io::PlySaveOptions po;
        po.output_path = out_dir / "vk_train.ply";
        auto saved = lfs::io::save_ply(*sd, po);
        if (!saved) {
            LOG_ERROR("vk-train: save_ply failed: {}", saved.error().format());
            return 1;
        }
        LOG_INFO("vk-train: saved trained model to {}", po.output_path.string());
        return 0;
    }

} // namespace lfs::app
