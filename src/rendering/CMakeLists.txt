# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(rasterizer)

# Local to this target: mesh2splat.cpp compiles GLSL → SPIR-V at runtime via glslang.
find_package(glslang CONFIG REQUIRED)

set(RENDERING_TENSOR_SOURCES
        cuda_vulkan_interop.cpp
        raster_rendering_engine.cpp
        mesh2splat.cpp
        screen_overlay_renderer.cpp
)

# cuda_vulkan_interop.cpp stays compiled even without CUDA: the CudaVulkanInterop
# class is held by value inside other types (e.g. VulkanUiTexture), so its
# ctor/dtor are called unconditionally — they must be real symbols, not deferred.
# The actual external-memory API calls are no-ops on macOS (see cuda_stub) and are
# only reached when interop is enabled, which it never is under MoltenVK.
# The .cu kernels (raster/selection/interaction) ARE CUDA-only and gated out.
if(LFS_ENABLE_CUDA)
    list(APPEND RENDERING_TENSOR_SOURCES
        cuda_vulkan_interop.cu
        selection_ops.cu
        cuda_interaction_kernels.cu
        point_cloud_raster.cu
    )
else()
    # CPU/no-op fallbacks for the gated selection CUDA kernels.
    list(APPEND RENDERING_TENSOR_SOURCES selection_ops_cpu_stub.cpp)
endif()

add_library(lfs_rendering_tensor STATIC ${RENDERING_TENSOR_SOURCES})

target_link_libraries(lfs_rendering_tensor
        PUBLIC
        lfs_core
        PRIVATE
        $<$<BOOL:${LFS_ENABLE_CUDA}>:CUDA::cudart>
        glm::glm
        Vulkan::Vulkan
        glslang::glslang
        glslang::SPIRV
        glslang::glslang-default-resource-limits
        OpenImageIO::OpenImageIO
        spdlog::spdlog
        TBB::tbb
)

target_include_directories(lfs_rendering_tensor
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(lfs_rendering_tensor PRIVATE
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

add_library(lfs_rendering INTERFACE)
target_link_libraries(lfs_rendering INTERFACE lfs_rendering_tensor)

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/rendering.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/frame_contract.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/mesh2splat.hpp
            DESTINATION include/rendering)
endif()
