// Split-raw variant of fused_projection_backward_optimizer for the no-CUDA Vulkan
// training path (macOS/MoltenVK). It reads/writes SplatData's split RAW buffers
// directly — means_raw, sh0, shN (swizzled), rotation_raw, scaling_raw (log),
// opacity_raw (logit) — so there is NO split<->packed pack and NO activated<->raw
// round-trip (the packed sibling shader round-trips activated values, which do not
// match the split-raw buffers the viewer renders). Adam runs in raw space and stays
// checkpoint-compatible with the CUDA trainer (same convention: raw storage,
// activation backward through project_gaussian_to_camera, Adam on raw).
//
// Minimal milestone: optimizes means, rotation, scale, opacity, and the SH DC term
// (sh0). The shN rest coefficients are read (for the forward recompute) but not yet
// optimized — see the TODO at the end. Not yet wired to a host driver.
#define NO_MAIN
#include "vertex_shader.slang"
#undef NO_MAIN

import strategy_utils;

// --- Split raw parameter buffers (updated in place by Adam) ---
layout(binding = 0) RWStructuredBuffer<float> xyz_ws;      // means_raw  [N,3]
layout(binding = 1) RWStructuredBuffer<float> sh0;         // SH DC      [N,1,3]
layout(binding = 2) RWStructuredBuffer<float4> shN;        // SH rest, swizzled (read + Adam update)
layout(binding = 3) RWStructuredBuffer<float4> rotations;  // raw quat   [N,4]
layout(binding = 4) RWStructuredBuffer<float> scaling_raw; // log        [N,3]
layout(binding = 5) RWStructuredBuffer<float> opacity_raw; // logit      [N,1]

// --- Screen-space gradients from the blending backward (rasterize_backward) ---
layout(binding = 6) StructuredBuffer<int> tiles_touched;
layout(binding = 7) StructuredBuffer<float2> v_output_xy_vs;
layout(binding = 8) StructuredBuffer<float4> v_output_inv_cov_vs;
layout(binding = 9) StructuredBuffer<float> v_output_rgb;

// --- Adam moment buffers (g1=m, g2=v). float buffers use stride-2 float3 packing
//     (m at offset 0, v at offset 1); float4 buffers use 2 slots per primitive. ---
layout(binding = 10) RWStructuredBuffer<float> g_xyz_ws;     // 2*3
layout(binding = 11) RWStructuredBuffer<float> g_sh0;        // 2*3
layout(binding = 12) RWStructuredBuffer<float4> g_rotations; // 2 (m,v)
layout(binding = 13) RWStructuredBuffer<float> g_scaling;    // 2*3
layout(binding = 14) RWStructuredBuffer<float> g_opacity;    // 2
// shN rest-coefficient Adam moments, in the SAME swizzled float4 layout as shN
// (shN_slots_per_primitive slots per primitive). g1 then g2, each a full shN-sized run.
layout(binding = 15) RWStructuredBuffer<float4> g_shN_1;
layout(binding = 16) RWStructuredBuffer<float4> g_shN_2;

// Densification stat: per-gaussian ||dL/d means_ws|| (3D world-space means gradient norm),
// the exact quantity MRNF's ADC thresholds on (growth_grad_threshold). Written every step.
layout(binding = 17) RWStructuredBuffer<float> grad_means_norm;

typedef vector<float, 1> float1;

// "standard" Adam with bias correction (identical to the packed sibling shader).
__generic<let dim : int> void optimizer_update(
    inout vector<float, dim> x, in vector<float, dim> v,
    inout vector<float, dim> g1, inout vector<float, dim> g2,
    float step, vector<float, dim> lr) {
    static const float beta_1 = 0.9;
    static const float beta_2 = 0.999;
    static const float eps = 1e-15;
    const float g1_sc = 1.0 / (1.0 - pow(beta_1, step));
    const float g2_sc = 1.0 / (1.0 - pow(beta_2, step));
    g1 = lerp(g1, v, 1.0 - beta_1);
    g2 = lerp(g2, v * v, 1.0 - beta_2);
    vector<float, dim> g1_t = g1 * g1_sc;
    vector<float, dim> g2_t = g2 * g2_sc;
    x = x - lr * g1_t / (sqrt(g2_t) + eps);
}

struct ShaderUniforms {
    uint32_t step;
    uint32_t active_sh; // low byte = sh degree, high byte = camera model
    uint32_t num_splats;
    uint32_t image_size; // hi16 = height, lo16 = width
    uint32_t shN_slots_per_primitive;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float lr_means;
    float lr_quats;
    float lr_scales;
    float lr_opacities;
    float lr_sh_dc;
    float reg_scale;
    float reg_opacity;
    float lr_sh_rest;
    float4x4 world_view_transform;
};

#define GROUP_SIZE SUBGROUP_SIZE

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)] void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform ShaderUniforms uniforms) {
    const uint active_sh = uniforms.active_sh & 0xff;
    const uint camera_model = uniforms.active_sh >> 8;
    const uint image_height = uniforms.image_size >> 16;
    const uint image_width = uniforms.image_size & ((1u << 16) - 1);
    const float4x4 world_view_transform = (uniforms.world_view_transform);

    uint g_idx = dispatchThreadID.x;
    if (g_idx >= uniforms.num_splats)
        return;

    const bool is_zero_grad = (tiles_touched[g_idx] == 0);

    Camera cam = load_camera(world_view_transform, uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
                             camera_model, uniforms.dist_coeffs, image_height, image_width);

    // Forward gaussian, activating the raw params exactly like load_gaussian_split.
    float3 raw_scaling = read_mean(g_idx, scaling_raw);
    raw_scaling = min(raw_scaling, float3(MAX_RAW_SCALE));
    const float opac_logit = opacity_raw[g_idx];
    Gaussian_3D gauss = {
        read_mean(g_idx, xyz_ws),
        read_spherical_harmonics_coeffs_split(g_idx, sh0, shN, active_sh, uniforms.shN_slots_per_primitive),
        rotations[g_idx],
        exp(raw_scaling),
        sigmoid_activation(opac_logit)
    };

    // Backward: scatter the screen-space gradients back through the projection.
    const float2 v_xy = is_zero_grad ? float2(0) : v_output_xy_vs[g_idx];
    const float4 v_inv_cov_opac = is_zero_grad ? float4(0) : v_output_inv_cov_vs[g_idx];
    const float3 v_rgb = is_zero_grad ? float3(0) : float3(read_t3_float3(g_idx, v_output_rgb));
    Splat_2D_Vertex v_splat = {
        float3(v_xy, 0.0), // xyz_vs
        v_rgb,             // rgb
        float3(0.0),       // cov_vs
        v_inv_cov_opac,    // inv_cov_vs + opacity
        0.0                // det
    };
    DifferentialPair<Gaussian_3D> v_gauss = diffPair(gauss);
    DifferentialPair<Camera> v_cam = diffPair(cam);
    bwd_diff(project_gaussian_to_camera)(v_gauss, v_cam, active_sh, v_splat);

    // v_gauss.d can be NaN when is_zero_grad; force-zero those gradients.
    const float3 v_means = is_zero_grad ? float3(0) : v_gauss.d.xyz_ws;
    grad_means_norm[g_idx] = length(v_means); // densification stat (== MRNF grad_means norm)
    const float4 v_rot = is_zero_grad ? float4(0) : v_gauss.d.rotations;
    const float3 v_scale_act = is_zero_grad ? float3(0) : v_gauss.d.scales;     // dL/d(activated scale)
    const float v_opac_act = is_zero_grad ? 0.0 : v_gauss.d.opacities;          // dL/d(activated opacity)
    const float3 v_dc = is_zero_grad ? float3(0) : v_gauss.d.sh_coeffs.coeff0;  // dL/d(SH DC)

    const uint32_t g1 = 2 * g_idx;
    const uint32_t g2 = 2 * g_idx + 1;

    // means (already raw)
    {
        float3 x = read_mean(g_idx, xyz_ws);
        float3 m = read_mean(g_idx, g_xyz_ws, 2, 0);
        float3 v = read_mean(g_idx, g_xyz_ws, 2, 1);
        optimizer_update(x, v_means, m, v, uniforms.step, float3(uniforms.lr_means));
        write_mean(x, g_idx, xyz_ws);
        write_mean(m, g_idx, g_xyz_ws, 2, 0);
        write_mean(v, g_idx, g_xyz_ws, 2, 1);
    }

    // rotation (raw quat; normalize Jacobian is inside project_gaussian_to_camera)
    {
        float4 x = rotations[g_idx];
        float4 m = g_rotations[g1];
        float4 v = g_rotations[g2];
        optimizer_update(x, v_rot, m, v, uniforms.step, float4(uniforms.lr_quats));
        rotations[g_idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;
        g_rotations[g1] = m;
        g_rotations[g2] = v;
    }

    // scale: optimize directly in raw log space (u = scaling_raw; dL/du = dL/dscale * exp(u))
    {
        float3 u = read_mean(g_idx, scaling_raw);
        float3 dLdu = (v_scale_act + uniforms.reg_scale) * exp(u);
        float3 m = read_mean(g_idx, g_scaling, 2, 0);
        float3 v = read_mean(g_idx, g_scaling, 2, 1);
        optimizer_update(u, dLdu, m, v, uniforms.step, float3(uniforms.lr_scales));
        write_mean(u, g_idx, scaling_raw);
        write_mean(m, g_idx, g_scaling, 2, 0);
        write_mean(v, g_idx, g_scaling, 2, 1);
    }

    // opacity: optimize in raw logit space (u = opacity_raw; dL/du = dL/dopac * sig*(1-sig))
    {
        const float u0 = opacity_raw[g_idx];
        const float sig = sigmoid_activation(u0);
        float1 u = float1(u0);
        float1 dLdu = float1((v_opac_act + uniforms.reg_opacity) * sig * (1.0 - sig));
        float1 m = float1(g_opacity[g1]);
        float1 v = float1(g_opacity[g2]);
        optimizer_update(u, dLdu, m, v, uniforms.step, float1(uniforms.lr_opacities));
        opacity_raw[g_idx] = u[0];
        g_opacity[g1] = m[0];
        g_opacity[g2] = v[0];
    }

    // SH DC term (raw)
    {
        float3 x = read_mean(g_idx, sh0);
        float3 m = read_mean(g_idx, g_sh0, 2, 0);
        float3 v = read_mean(g_idx, g_sh0, 2, 1);
        optimizer_update(x, v_dc, m, v, uniforms.step, float3(uniforms.lr_sh_dc));
        write_mean(x, g_idx, sh0);
        write_mean(m, g_idx, g_sh0, 2, 0);
        write_mean(v, g_idx, g_sh0, 2, 1);
    }

    // SH rest coefficients (raw), updated in place in the swizzled shN layout.
    // The 15 rest coeffs (45 floats) are densely packed into float4 slots exactly as
    // read_spherical_harmonics_coeffs_split unpacks them; mirror that mapping here.
    if (uniforms.shN_slots_per_primitive > 0u) {
        float3 c[15];
        c[0] = v_gauss.d.sh_coeffs.coeff1;
        c[1] = v_gauss.d.sh_coeffs.coeff2;
        c[2] = v_gauss.d.sh_coeffs.coeff3;
        c[3] = v_gauss.d.sh_coeffs.coeff4;
        c[4] = v_gauss.d.sh_coeffs.coeff5;
        c[5] = v_gauss.d.sh_coeffs.coeff6;
        c[6] = v_gauss.d.sh_coeffs.coeff7;
        c[7] = v_gauss.d.sh_coeffs.coeff8;
        c[8] = v_gauss.d.sh_coeffs.coeff9;
        c[9] = v_gauss.d.sh_coeffs.coeff10;
        c[10] = v_gauss.d.sh_coeffs.coeff11;
        c[11] = v_gauss.d.sh_coeffs.coeff12;
        c[12] = v_gauss.d.sh_coeffs.coeff13;
        c[13] = v_gauss.d.sh_coeffs.coeff14;
        c[14] = v_gauss.d.sh_coeffs.coeff15;

#if SH_REORDER_SIZE > 1
        const uint sh_base = (g_idx / SH_REORDER_SIZE) * (uniforms.shN_slots_per_primitive * SH_REORDER_SIZE);
        const uint sh_off = g_idx % SH_REORDER_SIZE;
#else
        const uint sh_base = uniforms.shN_slots_per_primitive * g_idx;
        const uint sh_off = 0u;
#endif
        for (uint s = 0u; s < uniforms.shN_slots_per_primitive; ++s) {
            const uint f = 4u * s; // first rest-float index of this float4 slot
            float4 grad = is_zero_grad ? float4(0.0) : float4(
                (f + 0u) < 45u ? c[(f + 0u) / 3u][(f + 0u) % 3u] : 0.0,
                (f + 1u) < 45u ? c[(f + 1u) / 3u][(f + 1u) % 3u] : 0.0,
                (f + 2u) < 45u ? c[(f + 2u) / 3u][(f + 2u) % 3u] : 0.0,
                (f + 3u) < 45u ? c[(f + 3u) / 3u][(f + 3u) % 3u] : 0.0);

            const uint idx = sh_base + SH_REORDER_SIZE * s + sh_off;
            float4 x = shN[idx];
            float4 m = g_shN_1[idx];
            float4 v = g_shN_2[idx];
            optimizer_update(x, grad, m, v, uniforms.step, float4(uniforms.lr_sh_rest));
            shN[idx] = x;
            g_shN_1[idx] = m;
            g_shN_2[idx] = v;
        }
    }
}
