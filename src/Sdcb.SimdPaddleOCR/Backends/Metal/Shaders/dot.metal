// Port of {conv1x1_dot, mmdot_sk, convk_dot, convk_f32n}.comp — direct-dot
// m4×n4 GEMM-ish kernels on fp16 data with k-major vec4 weights.
// flags: bit0 bias, bit1 residual, bit2 prescale(SE), bit3 dual-write
// (o2=gelu(o); bit10 1024 → o2=raw pre-act), bit8 apsorb, bits4-6 act.
#include <metal_stdlib>
using namespace metal;

struct PcDot { uint M, N, K, kv4, flags, scMod, mImg; };

// ---------------- conv1x1_dot ----------------
// O[m0:m0+4, n0:n0+4] = act(bias + res + A[m,k]·W[k,n4]) — A rows vec4 strided.
kernel void conv1x1_dot(device const half4* x [[buffer(0)]],
                        device const half4* w [[buffer(1)]],
                        device const half4* bias [[buffer(2)]],
                        device const half4* res [[buffer(3)]],
                        device half4* o [[buffer(4)]],
                        device const half4* s [[buffer(5)]],
                        device half4* o2 [[buffer(6)]],
                        device const half4* ra [[buffer(7)]],
                        constant PcDot& p [[buffer(8)]],
                        uint gid [[thread_position_in_grid]])
{
    uint nv = p.N >> 2;
    uint mv = (p.M + 3u) >> 2;
    if (gid >= mv * nv) return;
    uint n4 = gid % nv;
    uint m0 = (gid / nv) * 4u;

    float4 acc0 = float4(0.0), acc1 = float4(0.0), acc2 = float4(0.0), acc3 = float4(0.0);
    bool preScale = (p.flags & 4u) != 0u;
    bool apsorb = (p.flags & 512u) != 0u;
    uint sb0 = 0u, sb1 = 0u, sb2 = 0u, sb3 = 0u;
    if (p.mImg != 0u && (preScale || apsorb)) {
        sb0 = (m0 / p.mImg) * p.scMod;
        sb1 = ((m0 + 1u) / p.mImg) * p.scMod;
        sb2 = ((m0 + 2u) / p.mImg) * p.scMod;
        sb3 = ((m0 + 3u) / p.mImg) * p.scMod;
    }
    uint kv4t = (p.K + 3u) >> 2;
    for (uint k4 = 0u; k4 < kv4t; k4++) {
        float4 a0 = m0 < p.M ? float4(x[m0 * p.kv4 + k4]) : float4(0.0);
        float4 a1 = m0 + 1u < p.M ? float4(x[(m0 + 1u) * p.kv4 + k4]) : float4(0.0);
        float4 a2 = m0 + 2u < p.M ? float4(x[(m0 + 2u) * p.kv4 + k4]) : float4(0.0);
        float4 a3 = m0 + 3u < p.M ? float4(x[(m0 + 3u) * p.kv4 + k4]) : float4(0.0);
        if (apsorb) {
            if (m0 < p.M)       a0 = a0 * float4(s[sb0 + k4]) + float4(ra[m0 * p.kv4 + k4]);
            if (m0 + 1u < p.M)  a1 = a1 * float4(s[sb1 + k4]) + float4(ra[(m0 + 1u) * p.kv4 + k4]);
            if (m0 + 2u < p.M)  a2 = a2 * float4(s[sb2 + k4]) + float4(ra[(m0 + 2u) * p.kv4 + k4]);
            if (m0 + 3u < p.M)  a3 = a3 * float4(s[sb3 + k4]) + float4(ra[(m0 + 3u) * p.kv4 + k4]);
        }
        if (preScale) {
            a0 *= float4(s[sb0 + k4 % p.scMod]);
            a1 *= float4(s[sb1 + k4 % p.scMod]);
            a2 *= float4(s[sb2 + k4 % p.scMod]);
            a3 *= float4(s[sb3 + k4 % p.scMod]);
        }
        uint wRow = k4 * 4u * nv + n4;
        for (uint i = 0u; i < 4u; i++) {
            uint k = k4 * 4u + i;
            if (k >= p.K) break;
            float4 wv = float4(w[wRow + i * nv]);
            acc0 += a0[i] * wv; acc1 += a1[i] * wv;
            acc2 += a2[i] * wv; acc3 += a3[i] * wv;
        }
    }
    if ((p.flags & 1u) != 0u) {
        float4 bv = float4(bias[n4]);
        acc0 += bv; acc1 += bv; acc2 += bv; acc3 += bv;
    }
    uint act = (p.flags >> 4) & 7u;
    if ((p.flags & 2u) != 0u) {
        if (m0 < p.M)      acc0 += float4(res[(m0 * p.N >> 2) + n4]);
        if (m0 + 1u < p.M) acc1 += float4(res[((m0 + 1u) * p.N >> 2) + n4]);
        if (m0 + 2u < p.M) acc2 += float4(res[((m0 + 2u) * p.N >> 2) + n4]);
        if (m0 + 3u < p.M) acc3 += float4(res[((m0 + 3u) * p.N >> 2) + n4]);
    }
    float4 av0 = actf4(acc0, act, p.flags), av1 = actf4(acc1, act, p.flags);
    float4 av2 = actf4(acc2, act, p.flags), av3 = actf4(acc3, act, p.flags);
    if (m0 < p.M)      o[m0 * nv + n4]         = half4(av0);
    if (m0 + 1u < p.M) o[(m0 + 1u) * nv + n4]  = half4(av1);
    if (m0 + 2u < p.M) o[(m0 + 2u) * nv + n4]  = half4(av2);
    if (m0 + 3u < p.M) o[(m0 + 3u) * nv + n4]  = half4(av3);
    if ((p.flags & 8u) != 0u) {
        // dual-write: o2 = gelu(o); bit10 (1024) → o2 = raw pre-act acc
        float4 g0, g1, g2, g3;
        if ((p.flags & 1024u) != 0u) { g0 = acc0; g1 = acc1; g2 = acc2; g3 = acc3; }
        else {
            g0 = 0.5f * av0 * (float4(1.0) + float4(erf1(av0.x * 0.70710678f), erf1(av0.y * 0.70710678f), erf1(av0.z * 0.70710678f), erf1(av0.w * 0.70710678f)));
            g1 = 0.5f * av1 * (float4(1.0) + float4(erf1(av1.x * 0.70710678f), erf1(av1.y * 0.70710678f), erf1(av1.z * 0.70710678f), erf1(av1.w * 0.70710678f)));
            g2 = 0.5f * av2 * (float4(1.0) + float4(erf1(av2.x * 0.70710678f), erf1(av2.y * 0.70710678f), erf1(av2.z * 0.70710678f), erf1(av2.w * 0.70710678f)));
            g3 = 0.5f * av3 * (float4(1.0) + float4(erf1(av3.x * 0.70710678f), erf1(av3.y * 0.70710678f), erf1(av3.z * 0.70710678f), erf1(av3.w * 0.70710678f)));
        }
        if (m0 < p.M)      o2[m0 * nv + n4]        = half4(g0);
        if (m0 + 1u < p.M) o2[(m0 + 1u) * nv + n4] = half4(g1);
        if (m0 + 2u < p.M) o2[(m0 + 2u) * nv + n4] = half4(g2);
        if (m0 + 3u < p.M) o2[(m0 + 3u) * nv + n4] = half4(g3);
    }
}

// ---------------- mmdot_sk ----------------
// Split-K dot: 256 threads = 16 lanes (k-partitions) × 16 tiles. Adjacent
// local ids = adjacent tiles at the same k → weight reads coalesce.
// Partials go through threadgroup memory [lane][tile][16] (padded 17).
kernel void mmdot_sk(device const half4* x [[buffer(0)]],
                     device const half4* w [[buffer(1)]],
                     device const half4* bias [[buffer(2)]],
                     device const half4* res [[buffer(3)]],
                     device half4* o [[buffer(4)]],
                     device const half4* s [[buffer(5)]],
                     device half4* o2 [[buffer(6)]],
                     device const half4* ra [[buffer(7)]],
                     constant PcDot& p [[buffer(8)]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint wg [[threadgroup_position_in_grid]])
{
    // [lane][tile][16] partials, row padded to 17 floats vs bank conflicts
    threadgroup float sh[256 * 17];
    uint tl = lid & 15u;
    uint lane = lid >> 4;
    uint nv = p.N >> 2;
    uint mv = (p.M + 3u) >> 2;
    uint tile = wg * 16u + tl;
    bool live = tile < mv * nv;
    uint n4 = live ? tile % nv : 0u;
    uint m0 = live ? (tile / nv) * 4u : 0u;

    float4 acc0 = float4(0.0), acc1 = float4(0.0), acc2 = float4(0.0), acc3 = float4(0.0);
    bool preScale = (p.flags & 4u) != 0u;
    uint sb0 = 0u, sb1 = 0u, sb2 = 0u, sb3 = 0u;
    if (p.mImg != 0u && preScale) {
        sb0 = (m0 / p.mImg) * p.scMod;
        sb1 = ((m0 + 1u) / p.mImg) * p.scMod;
        sb2 = ((m0 + 2u) / p.mImg) * p.scMod;
        sb3 = ((m0 + 3u) / p.mImg) * p.scMod;
    }
    if (live) {
        for (uint k4 = lane; k4 < p.kv4; k4 += 16u) {
            float4 a0 = m0 < p.M ? float4(x[m0 * p.kv4 + k4]) : float4(0.0);
            float4 a1 = m0 + 1u < p.M ? float4(x[(m0 + 1u) * p.kv4 + k4]) : float4(0.0);
            float4 a2 = m0 + 2u < p.M ? float4(x[(m0 + 2u) * p.kv4 + k4]) : float4(0.0);
            float4 a3 = m0 + 3u < p.M ? float4(x[(m0 + 3u) * p.kv4 + k4]) : float4(0.0);
            if (preScale) {
                a0 *= float4(s[sb0 + k4 % p.scMod]);
                a1 *= float4(s[sb1 + k4 % p.scMod]);
                a2 *= float4(s[sb2 + k4 % p.scMod]);
                a3 *= float4(s[sb3 + k4 % p.scMod]);
            }
            uint wRow = k4 * 4u * nv + n4;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv = float4(w[wRow + i * nv]);
                acc0 += a0[i] * wv; acc1 += a1[i] * wv;
                acc2 += a2[i] * wv; acc3 += a3[i] * wv;
            }
        }
    }
    uint bb = (lane * 16u + tl) * 17u;
    sh[bb + 0u] = acc0.x; sh[bb + 1u] = acc0.y; sh[bb + 2u] = acc0.z; sh[bb + 3u] = acc0.w;
    sh[bb + 4u] = acc1.x; sh[bb + 5u] = acc1.y; sh[bb + 6u] = acc1.z; sh[bb + 7u] = acc1.w;
    sh[bb + 8u] = acc2.x; sh[bb + 9u] = acc2.y; sh[bb + 10u] = acc2.z; sh[bb + 11u] = acc2.w;
    sh[bb + 12u] = acc3.x; sh[bb + 13u] = acc3.y; sh[bb + 14u] = acc3.z; sh[bb + 15u] = acc3.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // this invocation now reduces element e = lane of tile tl over all lanes
    float sum = 0.0f;
    for (uint l = 0u; l < 16u; l++) sum += sh[(l * 16u + tl) * 17u + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sh[tl * 17u + lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || lane >= 4u) return;
    uint m = m0 + lane;
    if (m >= p.M) return;
    uint r = tl * 17u + lane * 4u;
    float4 v = float4(sh[r], sh[r + 1u], sh[r + 2u], sh[r + 3u]);
    if ((p.flags & 1u) != 0u) v += float4(bias[n4]);
    if ((p.flags & 2u) != 0u) v += float4(res[m * nv + n4]);
    uint act = (p.flags >> 4) & 7u;
    o[m * nv + n4] = half4(actf4(v, act, p.flags));
}

struct PcK {
    uint M, N, outW, inW, inH, cin4v, taps, kW, sH, sW, padT, padL, flags;
    uint dstCv, dstCoff, cin, inHW;
};

// ---------------- convk_dot ----------------
// Dense kxk conv as direct dot WITHOUT im2col: k = tap*cin4v + ci read
// straight from NHWC input. gy = batch.
kernel void convk_dot(device const half4* x [[buffer(0)]],
                      device const half4* w [[buffer(1)]],
                      device const half4* bias [[buffer(2)]],
                      device const half4* res [[buffer(3)]],
                      device half4* o [[buffer(4)]],
                      constant PcK& p [[buffer(5)]],
                      uint2 gp [[thread_position_in_grid]],
                      uint2 wg [[threadgroup_position_in_grid]])
{
    uint gid = gp.x;
    uint b = wg.y;
    uint nv = p.N >> 2;
    uint mv = (p.M + 3u) >> 2;
    if (gid >= mv * nv) return;
    uint n4 = gid % nv;
    uint m0 = (gid / nv) * 4u;
    uint xb = b * p.inH * p.inW * p.cin4v;
    uint rb = b * (p.M * p.N >> 2);
    uint dcv0 = p.dstCv != 0u ? p.dstCv : nv;
    uint ob = b * p.M * dcv0;

    float4 acc0 = float4(0.0), acc1 = float4(0.0), acc2 = float4(0.0), acc3 = float4(0.0);
    uint oh[4], ow[4];
    bool ok[4];
    for (uint i = 0u; i < 4u; i++) {
        uint m = m0 + i;
        ok[i] = m < p.M;
        oh[i] = m / p.outW;
        ow[i] = m - oh[i] * p.outW;
    }

    for (uint t = 0u; t < p.taps; t++) {
        uint ky = t / p.kW, kx = t - ky * p.kW;
        for (uint ci4 = 0u; ci4 < p.cin4v; ci4++) {
            float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
            for (uint i = 0u; i < 4u; i++) {
                if (!ok[i]) continue;
                int ih = int(oh[i] * p.sH + ky) - int(p.padT);
                int iw = int(ow[i] * p.sW + kx) - int(p.padL);
                if (ih < 0 || uint(ih) >= p.inH || iw < 0 || uint(iw) >= p.inW)
                    continue;
                uint px = uint(ih) * p.inW + uint(iw);
                float4 v = float4(x[xb + px * p.cin4v + ci4]);
                if (i == 0u) a0 = v; else if (i == 1u) a1 = v;
                else if (i == 2u) a2 = v; else a3 = v;
            }
            uint wRow = (t * p.cin4v + ci4) * 4u * nv + n4;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv = float4(w[wRow + i * nv]);
                acc0 += a0[i] * wv; acc1 += a1[i] * wv;
                acc2 += a2[i] * wv; acc3 += a3[i] * wv;
            }
        }
    }
    if ((p.flags & 1u) != 0u) {
        float4 bv = float4(bias[n4]);
        acc0 += bv; acc1 += bv; acc2 += bv; acc3 += bv;
    }
    uint act = (p.flags >> 4) & 7u;
    if ((p.flags & 2u) != 0u) {
        if (m0 < p.M)      acc0 += float4(res[rb + (m0 * p.N >> 2) + n4]);
        if (m0 + 1u < p.M) acc1 += float4(res[rb + ((m0 + 1u) * p.N >> 2) + n4]);
        if (m0 + 2u < p.M) acc2 += float4(res[rb + ((m0 + 2u) * p.N >> 2) + n4]);
        if (m0 + 3u < p.M) acc3 += float4(res[rb + ((m0 + 3u) * p.N >> 2) + n4]);
    }
    uint dcv = dcv0;
    uint obase = p.dstCoff + n4;
    if (m0 < p.M)      o[ob + m0 * dcv + obase]        = half4(actf4(acc0, act, p.flags));
    if (m0 + 1u < p.M) o[ob + (m0 + 1u) * dcv + obase] = half4(actf4(acc1, act, p.flags));
    if (m0 + 2u < p.M) o[ob + (m0 + 2u) * dcv + obase] = half4(actf4(acc2, act, p.flags));
    if (m0 + 3u < p.M) o[ob + (m0 + 3u) * dcv + obase] = half4(actf4(acc3, act, p.flags));
}

// ---------------- convk_f32n ----------------
// convk_dot whose A side reads the fp32 NCHW graph input directly (stem conv,
// cin <= 4 → single vec4 column per tap). gy = batch.
kernel void convk_f32n(device const float* x [[buffer(0)]],
                       device const half4* w [[buffer(1)]],
                       device const half4* bias [[buffer(2)]],
                       device const half4* res [[buffer(3)]],
                       device half4* o [[buffer(4)]],
                       constant PcK& p [[buffer(5)]],
                       uint2 gp [[thread_position_in_grid]],
                       uint2 wg [[threadgroup_position_in_grid]])
{
    uint gid = gp.x;
    uint b = wg.y;
    uint nv = p.N >> 2;
    uint mv = (p.M + 3u) >> 2;
    if (gid >= mv * nv) return;
    uint n4 = gid % nv;
    uint m0 = (gid / nv) * 4u;
    uint xb = b * p.cin * p.inHW;
    uint rb = b * (p.M * p.N >> 2);
    uint dcv0 = p.dstCv != 0u ? p.dstCv : nv;
    uint ob = b * p.M * dcv0;

    float4 acc0 = float4(0.0), acc1 = float4(0.0), acc2 = float4(0.0), acc3 = float4(0.0);
    uint oh[4], ow[4];
    bool ok[4];
    for (uint i = 0u; i < 4u; i++) {
        uint m = m0 + i;
        ok[i] = m < p.M;
        oh[i] = m / p.outW;
        ow[i] = m - oh[i] * p.outW;
    }

    for (uint t = 0u; t < p.taps; t++) {
        uint ky = t / p.kW, kx = t - ky * p.kW;
        float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
        for (uint i = 0u; i < 4u; i++) {
            if (!ok[i]) continue;
            int ih = int(oh[i] * p.sH + ky) - int(p.padT);
            int iw = int(ow[i] * p.sW + kx) - int(p.padL);
            if (ih < 0 || uint(ih) >= p.inH || iw < 0 || uint(iw) >= p.inW)
                continue;
            uint px = uint(ih) * p.inW + uint(iw);
            float4 v = float4(0.0);
            for (uint ci = 0u; ci < 4u; ci++)
                if (ci < p.cin) v[ci] = x[xb + ci * p.inHW + px];
            if (i == 0u) a0 = v; else if (i == 1u) a1 = v;
            else if (i == 2u) a2 = v; else a3 = v;
        }
        uint wRow = (t * p.cin4v) * 4u * nv + n4;
        for (uint i = 0u; i < 4u; i++) {
            float4 wv = float4(w[wRow + i * nv]);
            acc0 += a0[i] * wv; acc1 += a1[i] * wv;
            acc2 += a2[i] * wv; acc3 += a3[i] * wv;
        }
    }
    if ((p.flags & 1u) != 0u) {
        float4 bv = float4(bias[n4]);
        acc0 += bv; acc1 += bv; acc2 += bv; acc3 += bv;
    }
    uint act = (p.flags >> 4) & 7u;
    if ((p.flags & 2u) != 0u) {
        if (m0 < p.M)      acc0 += float4(res[rb + (m0 * p.N >> 2) + n4]);
        if (m0 + 1u < p.M) acc1 += float4(res[rb + ((m0 + 1u) * p.N >> 2) + n4]);
        if (m0 + 2u < p.M) acc2 += float4(res[rb + ((m0 + 2u) * p.N >> 2) + n4]);
        if (m0 + 3u < p.M) acc3 += float4(res[rb + ((m0 + 3u) * p.N >> 2) + n4]);
    }
    uint dcv = dcv0;
    uint obase = p.dstCoff + n4;
    if (m0 < p.M)      o[ob + m0 * dcv + obase]        = half4(actf4(acc0, act, p.flags));
    if (m0 + 1u < p.M) o[ob + (m0 + 1u) * dcv + obase] = half4(actf4(acc1, act, p.flags));
    if (m0 + 2u < p.M) o[ob + (m0 + 2u) * dcv + obase] = half4(actf4(acc2, act, p.flags));
    if (m0 + 3u < p.M) o[ob + (m0 + 3u) * dcv + obase] = half4(actf4(acc3, act, p.flags));
}
