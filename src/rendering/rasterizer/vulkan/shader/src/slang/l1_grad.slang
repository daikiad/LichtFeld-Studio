// GPU image-space L1 loss gradient for the no-CUDA Vulkan training path.
// Reads the raster accumulator (pixel_state) + a ground-truth image and writes the
// upstream gradient dL/d(final_pixel_state) the blending backward consumes, with no
// per-iteration CPU readback.
//
// The composed image (over a black background) is clamp(pixel_state.rgb, 0, 1); the
// transmittance-channel gradient is 0 for a black background, so v.a = 0. The clamp
// derivative zeroes the gradient for saturated pixels. Mean-L1 over the 3*num_pixels
// colour samples => inv_scale = 1 / (3 * num_pixels).
#include "config.slang"

struct Uniforms {
    uint32_t num_pixels;
    float inv_scale;
    uint32_t pad0;
    uint32_t pad1;
};

layout(binding = 0) StructuredBuffer<float4> pixel_state;        // rendered accumulator (rgb, T)
layout(binding = 1) StructuredBuffer<float4> gt;                 // target image (rgb in .xyz)
layout(binding = 2) RWStructuredBuffer<float4> v_current_pixel_state; // out: dL/d(pixel_state)

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint i = dispatchThreadID.x;
    if (i >= uniforms.num_pixels)
        return;

    const float4 ps = pixel_state[i];
    const float3 composed = clamp(ps.xyz, float3(0.0), float3(1.0));
    const float3 d = composed - gt[i].xyz;
    // clamp derivative: gradient only flows where the accumulator is inside [0,1].
    const float3 clamp_deriv = float3(
        (ps.x >= 0.0 && ps.x <= 1.0) ? 1.0 : 0.0,
        (ps.y >= 0.0 && ps.y <= 1.0) ? 1.0 : 0.0,
        (ps.z >= 0.0 && ps.z <= 1.0) ? 1.0 : 0.0);
    const float3 g = sign(d) * clamp_deriv * uniforms.inv_scale;
    v_current_pixel_state[i] = float4(g, 0.0);
}
