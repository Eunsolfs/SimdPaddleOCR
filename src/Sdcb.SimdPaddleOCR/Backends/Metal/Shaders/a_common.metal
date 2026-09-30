// Shared helpers for the OCR kernel set — concatenated first.
#include <metal_stdlib>
using namespace metal;

// Abramowitz-Stegun 7.1.26 erf (same polynomial the GLSL kernels use —
// identical rounding keeps CPU↔GPU diffs minimal).
static float erf1(float v)
{
    float s = v < 0.0 ? -1.0 : 1.0;
    v = fabs(v);
    float t = 1.0 / (1.0 + 0.47047 * v);
    float q = t * (0.3480242 + t * (-0.0958798 + t * 0.7478556));
    return s * (1.0 - q * exp(-v * v));
}

// fp16 alpha lo16 / fp16 beta hi16 packed pair (hardsigmoid/hardswish aux)
static inline float2 auxHalf2(uint aux)
{
    ushort2 u = as_type<ushort2>(aux);
    return float2(as_type<half>(u.x), as_type<half>(u.y));
}

// Conv epilogue activation (flags bits4-6; hardswish alpha in bits16-31):
//   0=none 1=relu 2=gelu 3=hardswish(a in hi16) 4=sigmoid
static inline float actf(float v, uint act, uint flg)
{
    if (act == 1u) return max(v, 0.0f);
    if (act == 2u) return 0.5f * v * (1.0f + erf1(v * 0.70710678118654752f));
    if (act == 3u) {
        float a = float(as_type<half>((ushort)(flg >> 16)));
        return v * clamp(v * a + 0.5f, 0.0f, 1.0f);
    }
    if (act == 4u) return 1.0f / (1.0f + exp(-v));
    if (act == 5u) return v / (1.0f + exp(-v));   // swish x*sigmoid(x)
    return v;
}

static inline float4 actf4(float4 v, uint act, uint flg)
{
    return float4(actf(v.x, act, flg), actf(v.y, act, flg),
                  actf(v.z, act, flg), actf(v.w, act, flg));
}
