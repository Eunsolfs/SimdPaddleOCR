// Metal GEMM for the conv1x1 / im2col / MatMul "cm" paths.
// O[M,N] = act(A[M,K] · W[N,K]^T + bias + res)   (+ dual-write o2 = gelu/raw)
//   A: fp16 [M, K] row-major (vec4-strided NHWC slabs; M folds the batch dim)
//   W: fp16 [Npad, K] n-major (ConstF16 / ConstGemmW repack)
// flags: bit0 bias, bit1 res, bits4-6 act, hi16 hsA, bit3 dual-write
//        (o2 = gelu(o); bit10 → o2 = raw pre-act).
// mm_sg: simdgroup_matrix 8x8x8 fp16 MMA, group tile 64x64, K-step 16 staged
//   fp16 — requires simdgroup matrix support (probed by MetalBackend).
// mm_ps_sg: same + per-batch SE prescale on A (bit2): a *= s[b*scMod + k].
#include <metal_stdlib>
using namespace metal;

struct PcGemm { uint M, N, K, kv4, flags, mImg, scMod; };

// Cooperative fp16 staging for the 64x64x16 k-tile:
//   A stage [64][17] halfs (row = m), B stage [64][17] halfs (row = n).
// 256 threads: A stage thread t covers m=t>>2, k4=(t&3)*4;
//              B stage covers n=t>>2, k4=(t&3)*4.
static inline void gemm_stage(
    threadgroup half* tA, threadgroup half* tB,
    device const half4* x, device const half* w,
    uint m0, uint n0, uint kb, uint tid,
    uint M, uint N, uint K, uint kv4)
{
    uint am = tid >> 2, ak = (tid & 3u) * 4u;
    uint gm = m0 + am, gk = kb + ak;
    half4 av = half4(0.0);
    if (gm < M && gk < K)
        av = x[(ulong)gm * kv4 + (gk >> 2)];   // K%4==0 guaranteed by emit
    *(threadgroup half4*)(tA + am * 17 + ak) = av;

    uint bn = tid >> 2, bk = (tid & 3u) * 4u;
    uint gn = n0 + bn;
    half4 wv = half4(0.0);
    if (gk < K)
        wv = *(device const half4*)(w + (ulong)gn * K + gk);  // n-major [N,K]
    *(threadgroup half4*)(tB + bn * 17 + bk) = wv;
}

kernel void mm_sg(device const half4* x [[buffer(0)]],
                  device const half* w [[buffer(1)]],
                  device const half* bias [[buffer(2)]],
                  device const half* res [[buffer(3)]],
                  device half4* o [[buffer(4)]],
                  device half4* o2 [[buffer(5)]],
                  device const half4* se [[buffer(6)]],   // prescale (unused here)
                  constant PcGemm& p [[buffer(7)]],
                  uint2 wg [[threadgroup_position_in_grid]],
                  uint2 tp2 [[thread_position_in_threadgroup]])
{
    uint tid = tp2.x;
    threadgroup half tA[64 * 17];
    threadgroup half tB[64 * 17];
    threadgroup float outst[8 * 64];   // per-sg 8x8 epilogue stage

    uint sgid = tid >> 5;
    uint sm = (sgid & 3u) * 16;        // 4 sgs along M (16 rows each)
    uint sn = (sgid >> 2) * 32;        // 2 sgs along N (32 cols each)
    uint m0 = wg.y * 64u + sm;
    uint n0 = wg.x * 64u + sn;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) for (uint j = 0u; j < 4u; j++)
        acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb = 0u; kb < p.K; kb += 16u) {
        gemm_stage(tA, tB, x, w, wg.y * 64u, wg.x * 64u, kb, tid,
                   p.M, p.N, p.K, p.kv4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 16u; kk += 8u) {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, tA + sm * 17 + kk, 17);
            simdgroup_load(a1, tA + (sm + 8u) * 17 + kk, 17);
            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, tB + (sn + 0u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b1, tB + (sn + 8u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b2, tB + (sn + 16u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b3, tB + (sn + 24u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: stage each 8x8 acc to threadgroup, scalars write fp16.
    // outst layout [sg][64] fp32; row r col c at stage[r*8+c].
    uint lane = tid & 31u;
    uint nv = p.N >> 2;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], outst + sgid * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 32 lanes × 2 elems: lane l covers (r=l>>2, c=(l&3)*2 + k)
            uint r = lane >> 2, c2 = (lane & 3u) * 2u;
            for (uint e = 0u; e < 2u; e++) {
                uint mm = m0 + i * 8u + r;
                uint nn = n0 + j * 8u + c2 + e;
                if (mm >= p.M || nn >= p.N) continue;
                float v = outst[sgid * 64 + r * 8 + c2 + e];
                if ((p.flags & 1u) != 0u) v += float(bias[nn]);
                if ((p.flags & 2u) != 0u) v += float(res[mm * p.N + nn]);
                float av = actf(v, (p.flags >> 4) & 7u, p.flags);
                // o/o2 are half4-bound: scalar write via half* view
                ((device half*)o)[mm * p.N + nn] = half(av);
                if ((p.flags & 8u) != 0u)
                    ((device half*)o2)[mm * p.N + nn] =
                        (p.flags & 1024u) != 0u ? half(v) : half(0.5f * av * (1.0f + erf1(av * 0.70710678f)));
            }
        }
}

// mm_sg_dd: same 64x64 tile / 8 simdgroup fan-out as mm_sg, but fragment loads
// read device memory directly — no threadgroup staging round-trip (this GPU's
// "threadgroup" memory is just a carve-out of the same DRAM, so staging only
// adds barriers). Emit-gated to K%16==0 only: edge tiles are partial — fragment
// loads do read past the logical M/N end, which is safe because A overread stays
// inside the arena's tail slack, W rows are padded, and the epilogue masks
// mm<p.M / nn<p.N before storing.
kernel void mm_sg_dd(device const half* x [[buffer(0)]],
                     device const half* w [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device const half* res [[buffer(3)]],
                     device half4* o [[buffer(4)]],
                     device half4* o2 [[buffer(5)]],
                     device const half4* se [[buffer(6)]],
                     constant PcGemm& p [[buffer(7)]],
                     uint2 wg [[threadgroup_position_in_grid]],
                     uint2 tp2 [[thread_position_in_threadgroup]])
{
    uint tid = tp2.x;
    threadgroup float outst[8 * 64];

    uint sgid = tid >> 5;
    uint sm = (sgid & 3u) * 16;
    uint sn = (sgid >> 2) * 32;
    uint m0 = wg.y * 64u + sm;
    uint n0 = wg.x * 64u + sn;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) for (uint j = 0u; j < 4u; j++)
        acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb = 0u; kb < p.K; kb += 16u) {
        simdgroup_half8x8 a0, a1, a00, a01;
        simdgroup_load(a0, x + (ulong)(m0 * p.K) + kb, p.K);
        simdgroup_load(a1, x + (ulong)((m0 + 8u) * p.K) + kb, p.K);
        simdgroup_load(a00, x + (ulong)(m0 * p.K) + kb + 8u, p.K);
        simdgroup_load(a01, x + (ulong)((m0 + 8u) * p.K) + kb + 8u, p.K);
        simdgroup_half8x8 b0, b1, b2, b3, b00, b01, b02, b03;
        simdgroup_load(b0, w + (ulong)((n0 + 0u) * p.K) + kb, p.K, ulong2(0, 0), true);
        simdgroup_load(b1, w + (ulong)((n0 + 8u) * p.K) + kb, p.K, ulong2(0, 0), true);
        simdgroup_load(b2, w + (ulong)((n0 + 16u) * p.K) + kb, p.K, ulong2(0, 0), true);
        simdgroup_load(b3, w + (ulong)((n0 + 24u) * p.K) + kb, p.K, ulong2(0, 0), true);
        simdgroup_load(b00, w + (ulong)((n0 + 0u) * p.K) + kb + 8u, p.K, ulong2(0, 0), true);
        simdgroup_load(b01, w + (ulong)((n0 + 8u) * p.K) + kb + 8u, p.K, ulong2(0, 0), true);
        simdgroup_load(b02, w + (ulong)((n0 + 16u) * p.K) + kb + 8u, p.K, ulong2(0, 0), true);
        simdgroup_load(b03, w + (ulong)((n0 + 24u) * p.K) + kb + 8u, p.K, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        simdgroup_multiply_accumulate(acc[0][0], a00, b00, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a00, b01, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a00, b02, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a00, b03, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a01, b00, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a01, b01, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a01, b02, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a01, b03, acc[1][3]);
    }

    uint lane = tid & 31u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], outst + sgid * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint r = lane >> 2, c2 = (lane & 3u) * 2u;
            for (uint e = 0u; e < 2u; e++) {
                uint mm = m0 + i * 8u + r;
                uint nn = n0 + j * 8u + c2 + e;
                if (mm >= p.M || nn >= p.N) continue;
                float v = outst[sgid * 64 + r * 8 + c2 + e];
                if ((p.flags & 1u) != 0u) v += float(bias[nn]);
                if ((p.flags & 2u) != 0u) v += float(res[mm * p.N + nn]);
                float av = actf(v, (p.flags >> 4) & 7u, p.flags);
                ((device half*)o)[mm * p.N + nn] = half(av);
                if ((p.flags & 8u) != 0u)
                    ((device half*)o2)[mm * p.N + nn] =
                        (p.flags & 1024u) != 0u ? half(v) : half(0.5f * av * (1.0f + erf1(av * 0.70710678f)));
            }
        }
}

// mm_ps_sg: prescale (bit2) variant — a[m,k] *= s[b*scMod + k]; mImg = output
// pixels per image (batch split of the flat M range).
kernel void mm_ps_sg(device const half4* x [[buffer(0)]],
                     device const half* w [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device const half* res [[buffer(3)]],
                     device half4* o [[buffer(4)]],
                     device half4* o2 [[buffer(5)]],
                     device const half4* se [[buffer(6)]],
                     constant PcGemm& p [[buffer(7)]],
                     uint2 wg [[threadgroup_position_in_grid]],
                     uint2 tp2 [[thread_position_in_threadgroup]])
{
    uint tid = tp2.x;
    threadgroup half tA[64 * 17];
    threadgroup half tB[64 * 17];
    threadgroup float outst[8 * 64];

    uint sgid = tid >> 5;
    uint sm = (sgid & 3u) * 16;
    uint sn = (sgid >> 2) * 32;
    uint m0 = wg.y * 64u + sm;
    uint n0 = wg.x * 64u + sn;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) for (uint j = 0u; j < 4u; j++)
        acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb = 0u; kb < p.K; kb += 16u) {
        // stage A with prescale applied
        {
            uint am = tid >> 2, ak = (tid & 3u) * 4u;
            uint gm = wg.y * 64u + am, gk = kb + ak;
            half4 av = half4(0.0);
            if (gm < p.M && gk < p.K) {
                av = x[(ulong)gm * p.kv4 + (gk >> 2)];
                // se[b*scMod + k4%scMod] — half4 group per k-block
                av = half4(float4(av) *
                    float4(se[(gm / p.mImg) * p.scMod + (gk >> 2) % p.scMod]));
            }
            *(threadgroup half4*)(tA + am * 17 + ak) = av;
        }
        {
            uint bn = tid >> 2, bk = (tid & 3u) * 4u;
            uint gn = wg.x * 64u + bn;
            half4 wv = half4(0.0);
            if (kb + bk < p.K)
                wv = *(device const half4*)(w + (ulong)gn * p.K + kb + bk);
            *(threadgroup half4*)(tB + bn * 17 + bk) = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 16u; kk += 8u) {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, tA + sm * 17 + kk, 17);
            simdgroup_load(a1, tA + (sm + 8u) * 17 + kk, 17);
            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, tB + (sn + 0u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b1, tB + (sn + 8u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b2, tB + (sn + 16u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b3, tB + (sn + 24u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint lane = tid & 31u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], outst + sgid * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint r = lane >> 2, c2 = (lane & 3u) * 2u;
            for (uint e = 0u; e < 2u; e++) {
                uint mm = m0 + i * 8u + r;
                uint nn = n0 + j * 8u + c2 + e;
                if (mm >= p.M || nn >= p.N) continue;
                float v = outst[sgid * 64 + r * 8 + c2 + e];
                if ((p.flags & 1u) != 0u) v += float(bias[nn]);
                if ((p.flags & 2u) != 0u) v += float(res[mm * p.N + nn]);
                float av = actf(v, (p.flags >> 4) & 7u, p.flags);
                ((device half*)o)[mm * p.N + nn] = half(av);
                if ((p.flags & 8u) != 0u)
                    ((device half*)o2)[mm * p.N + nn] =
                        (p.flags & 1024u) != 0u ? half(v) : half(0.5f * av * (1.0f + erf1(av * 0.70710678f)));
            }
        }
}

// ---------------- mm_dot (portable fallback) ----------------
// Same contract as mm_sg but no simdgroup builtins: thread per (m4,n4) tile,
// W still n-major [N,K] — each (m,ncol) acc accumulates dot(a_m, w_n4..).
// Used when the simdgroup-matrix probe fails (kernel still correct at any
// M/N/K; conv1x1-dot-style gating keeps it off the biggest shapes anyway).
kernel void mm_dot(device const half4* x [[buffer(0)]],
                   device const half4* w [[buffer(1)]],
                   device const half4* bias [[buffer(2)]],
                   device const half4* res [[buffer(3)]],
                   device half4* o [[buffer(4)]],
                   device half4* o2 [[buffer(5)]],
                   device const half4* se [[buffer(6)]],
                   constant PcGemm& p [[buffer(7)]],
                   uint gid [[thread_position_in_grid]])
{
    uint nv = p.N >> 2;
    uint mv = (p.M + 3u) >> 2;
    if (gid >= mv * nv) return;
    uint n4 = gid % nv;
    uint m0 = (gid / nv) * 4u;
    bool preScale = (p.flags & 4u) != 0u;
    uint sb0 = 0u, sb1 = 0u, sb2 = 0u, sb3 = 0u;
    if (preScale && p.mImg != 0u) {
        sb0 = (m0 / p.mImg) * p.scMod;
        sb1 = ((m0 + 1u) / p.mImg) * p.scMod;
        sb2 = ((m0 + 2u) / p.mImg) * p.scMod;
        sb3 = ((m0 + 3u) / p.mImg) * p.scMod;
    }
    float4 acc0 = float4(0.0), acc1 = float4(0.0), acc2 = float4(0.0), acc3 = float4(0.0);
    for (uint k4 = 0u; k4 < p.kv4; k4++) {
        float4 a0 = m0 < p.M ? float4(x[m0 * p.kv4 + k4]) : float4(0.0);
        float4 a1 = m0 + 1u < p.M ? float4(x[(m0 + 1u) * p.kv4 + k4]) : float4(0.0);
        float4 a2 = m0 + 2u < p.M ? float4(x[(m0 + 2u) * p.kv4 + k4]) : float4(0.0);
        float4 a3 = m0 + 3u < p.M ? float4(x[(m0 + 3u) * p.kv4 + k4]) : float4(0.0);
        if (preScale) {
            a0 *= float4(se[sb0 + k4 % p.scMod]);
            a1 *= float4(se[sb1 + k4 % p.scMod]);
            a2 *= float4(se[sb2 + k4 % p.scMod]);
            a3 *= float4(se[sb3 + k4 % p.scMod]);
        }
        // W n-major: w[n*K + k] — acc_i[nj] += dot(a_i, w4(nj)) per k4
        float4 wv0 = float4(w[(n4 * 4u + 0u) * p.kv4 + k4]);
        float4 wv1 = float4(w[(n4 * 4u + 1u) * p.kv4 + k4]);
        float4 wv2 = float4(w[(n4 * 4u + 2u) * p.kv4 + k4]);
        float4 wv3 = float4(w[(n4 * 4u + 3u) * p.kv4 + k4]);
        acc0.x += dot(a0, wv0); acc0.y += dot(a0, wv1);
        acc0.z += dot(a0, wv2); acc0.w += dot(a0, wv3);
        acc1.x += dot(a1, wv0); acc1.y += dot(a1, wv1);
        acc1.z += dot(a1, wv2); acc1.w += dot(a1, wv3);
        acc2.x += dot(a2, wv0); acc2.y += dot(a2, wv1);
        acc2.z += dot(a2, wv2); acc2.w += dot(a2, wv3);
        acc3.x += dot(a3, wv0); acc3.y += dot(a3, wv1);
        acc3.z += dot(a3, wv2); acc3.w += dot(a3, wv3);
    }
    if ((p.flags & 1u) != 0u) {
        float4 bv = float4(bias[n4]);
        acc0 += bv; acc1 += bv; acc2 += bv; acc3 += bv;
    }
    if ((p.flags & 2u) != 0u) {
        if (m0 < p.M)      acc0 += float4(res[m0 * nv + n4]);
        if (m0 + 1u < p.M) acc1 += float4(res[(m0 + 1u) * nv + n4]);
        if (m0 + 2u < p.M) acc2 += float4(res[(m0 + 2u) * nv + n4]);
        if (m0 + 3u < p.M) acc3 += float4(res[(m0 + 3u) * nv + n4]);
    }
    uint act = (p.flags >> 4) & 7u;
    float4 av0 = actf4(acc0, act, p.flags), av1 = actf4(acc1, act, p.flags);
    float4 av2 = actf4(acc2, act, p.flags), av3 = actf4(acc3, act, p.flags);
    if (m0 < p.M)      o[m0 * nv + n4]        = half4(av0);
    if (m0 + 1u < p.M) o[(m0 + 1u) * nv + n4] = half4(av1);
    if (m0 + 2u < p.M) o[(m0 + 2u) * nv + n4] = half4(av2);
    if (m0 + 3u < p.M) o[(m0 + 3u) * nv + n4] = half4(av3);
    if ((p.flags & 8u) != 0u) {
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

// mm_ic_sg: implicit-GEMM kxk conv — mm_sg's staged 64x64x16 pipeline, but the
// A stage gathers input pixels through tap-major im2col addressing instead of
// reading a materialized [M,K] matrix. One staged half4 never crosses a tap
// (cinE%4==0 gate on the emit side). p.K is the padded reduction extent
// (multiple of 16): pad k-slices gather REAL pixels (tap>=kH*kW still lands
// in-bounds) but multiply against W rows that ConstF16 zero-filled, so they
// contribute 0 — correct for finite activations; an inf activation would make
// NaN here (0*inf), which no real graph produces. Batch-free like the im2col
// path it replaces.
struct PcGemmIc {
    uint M, N, K, kv4, flags, mImg, scMod;      // K = Kp (padded, W stride)
    uint outW, inW, inH, sH, sW, pT, pL, kW, cinE;
};

kernel void mm_ic_sg(device const half4* x [[buffer(0)]],
                     device const half* w [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device const half* res [[buffer(3)]],
                     device half4* o [[buffer(4)]],
                     device half4* o2 [[buffer(5)]],
                     device const half4* se [[buffer(6)]],
                     constant PcGemmIc& p [[buffer(7)]],
                     uint2 wg [[threadgroup_position_in_grid]],
                     uint2 tp2 [[thread_position_in_threadgroup]])
{
    uint tid = tp2.x;
    threadgroup half tA[64 * 17];
    threadgroup half tB[64 * 17];
    threadgroup float outst[8 * 64];

    uint sgid = tid >> 5;
    uint sm = (sgid & 3u) * 16;
    uint sn = (sgid >> 2) * 32;
    uint m0 = wg.y * 64u + sm;
    uint n0 = wg.x * 64u + sn;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) for (uint j = 0u; j < 4u; j++)
        acc[i][j] = simdgroup_float8x8(0.0f);

    uint am = tid >> 2, ak = (tid & 3u) * 4u;
    uint gm0 = wg.y * 64u + am;
    uint bn = tid >> 2, bk = (tid & 3u) * 4u;
    uint gn0 = wg.x * 64u + bn;

    for (uint kb = 0u; kb < p.K; kb += 16u) {
        // A stage with inline im2col gather: element k = tap*cinE + ci,
        // tap = dy*kW + dx.
        uint gk = kb + ak;
        half4 av = half4(0.0);
        if (gm0 < p.M && gk < p.K) {
            uint tap = gk / p.cinE, ci = gk - tap * p.cinE;
            uint dy = tap / p.kW, dx = tap - dy * p.kW;
            uint oy = gm0 / p.outW, ox = gm0 - oy * p.outW;
            int iy = int(oy * p.sH + dy) - int(p.pT);
            int ix = int(ox * p.sW + dx) - int(p.pL);
            if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW))
                av = x[(uint(iy) * p.inW + uint(ix)) * (p.cinE >> 2) + (ci >> 2)];
        }
        *(threadgroup half4*)(tA + am * 17 + ak) = av;

        uint gkB = kb + bk;
        half4 wv = half4(0.0);
        if (gkB < p.K)
            wv = *(device const half4*)(w + (ulong)gn0 * p.K + gkB);
        *(threadgroup half4*)(tB + bn * 17 + bk) = wv;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 16u; kk += 8u) {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, tA + sm * 17 + kk, 17);
            simdgroup_load(a1, tA + (sm + 8u) * 17 + kk, 17);
            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, tB + (sn + 0u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b1, tB + (sn + 8u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b2, tB + (sn + 16u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_load(b3, tB + (sn + 24u) * 17 + kk, 17, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint lane = tid & 31u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], outst + sgid * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint r = lane >> 2, c2 = (lane & 3u) * 2u;
            for (uint e = 0u; e < 2u; e++) {
                uint mm = m0 + i * 8u + r;
                uint nn = n0 + j * 8u + c2 + e;
                if (mm >= p.M || nn >= p.N) continue;
                float v = outst[sgid * 64 + r * 8 + c2 + e];
                if ((p.flags & 1u) != 0u) v += float(bias[nn]);
                if ((p.flags & 2u) != 0u) v += float(res[mm * p.N + nn]);
                float av = actf(v, (p.flags >> 4) & 7u, p.flags);
                ((device half*)o)[mm * p.N + nn] = half(av);
                if ((p.flags & 8u) != 0u)
                    ((device half*)o2)[mm * p.N + nn] =
                        (p.flags & 1024u) != 0u ? half(v) : half(0.5f * av * (1.0f + erf1(av * 0.70710678f)));
            }
        }
}

// mm_ic_sg32: implicit-GEMM kxk conv — mm_sg's staged 64x64x16 pipeline, but the
// A stage gathers input pixels through tap-major im2col addressing instead of
// reading a materialized [M,K] matrix. One staged half4 never crosses a tap
// (cinE%4==0 gate on the emit side). p.K is the padded reduction extent
// (multiple of 16): pad k-slices gather REAL pixels (tap>=kH*kW still lands
// in-bounds) but multiply against W rows that ConstF16 zero-filled, so they
// contribute 0 — correct for finite activations; an inf activation would make
// NaN here (0*inf), which no real graph produces. Batch-free like the im2col
// path it replaces.
struct PcGemmIc32 {
    uint M, N, K, kv4, flags, mImg, scMod;      // K = Kp (padded, W stride)
    uint outW, inW, inH, sH, sW, pT, pL, kW, cinE;
};

kernel void mm_ic_sg32(device const half4* x [[buffer(0)]],
                     device const half* w [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device const half* res [[buffer(3)]],
                     device half4* o [[buffer(4)]],
                     device half4* o2 [[buffer(5)]],
                     device const half4* se [[buffer(6)]],
                     constant PcGemmIc32& p [[buffer(7)]],
                     uint2 wg [[threadgroup_position_in_grid]],
                     uint2 tp2 [[thread_position_in_threadgroup]])
{
    uint tid = tp2.x;
    threadgroup half tA[64 * 34];
    threadgroup half tB[64 * 34];
    threadgroup float outst[8 * 64];

    uint sgid = tid >> 5;
    uint sm = (sgid & 3u) * 16;
    uint sn = (sgid >> 2) * 32;
    uint m0 = wg.y * 64u + sm;
    uint n0 = wg.x * 64u + sn;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) for (uint j = 0u; j < 4u; j++)
        acc[i][j] = simdgroup_float8x8(0.0f);

    uint am = tid >> 2, ak = (tid & 3u) * 4u;
    uint gm0 = wg.y * 64u + am;
    uint bn = tid >> 2, bk = (tid & 3u) * 4u;
    uint gn0 = wg.x * 64u + bn;

    for (uint kb = 0u; kb < p.K; kb += 32u) {
        // A stage with inline im2col gather: element k = tap*cinE + ci,
        // tap = dy*kW + dx.
        uint oy = gm0 / p.outW, ox = gm0 - oy * p.outW;
        for (uint h = 0u; h < 2u; h++) {
            uint gk = kb + ak + h * 16u;
            half4 av = half4(0.0);
            if (gm0 < p.M && gk < p.K) {
                uint tap = gk / p.cinE, ci = gk - tap * p.cinE;
                uint dy = tap / p.kW, dx = tap - dy * p.kW;
                int iy = int(oy * p.sH + dy) - int(p.pT);
                int ix = int(ox * p.sW + dx) - int(p.pL);
                if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW))
                    av = x[(uint(iy) * p.inW + uint(ix)) * (p.cinE >> 2) + (ci >> 2)];
            }
            *(threadgroup half4*)(tA + am * 34 + ak + h * 16u) = av;

            uint gkB = kb + bk + h * 16u;
            half4 wv = half4(0.0);
            if (gkB < p.K)
                wv = *(device const half4*)(w + (ulong)gn0 * p.K + gkB);
            *(threadgroup half4*)(tB + bn * 34 + bk + h * 16u) = wv;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, tA + sm * 34 + kk, 34);
            simdgroup_load(a1, tA + (sm + 8u) * 34 + kk, 34);
            simdgroup_half8x8 b0, b1, b2, b3;
            simdgroup_load(b0, tB + (sn + 0u) * 34 + kk, 34, ulong2(0, 0), true);
            simdgroup_load(b1, tB + (sn + 8u) * 34 + kk, 34, ulong2(0, 0), true);
            simdgroup_load(b2, tB + (sn + 16u) * 34 + kk, 34, ulong2(0, 0), true);
            simdgroup_load(b3, tB + (sn + 24u) * 34 + kk, 34, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint lane = tid & 31u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], outst + sgid * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint r = lane >> 2, c2 = (lane & 3u) * 2u;
            for (uint e = 0u; e < 2u; e++) {
                uint mm = m0 + i * 8u + r;
                uint nn = n0 + j * 8u + c2 + e;
                if (mm >= p.M || nn >= p.N) continue;
                float v = outst[sgid * 64 + r * 8 + c2 + e];
                if ((p.flags & 1u) != 0u) v += float(bias[nn]);
                if ((p.flags & 2u) != 0u) v += float(res[mm * p.N + nn]);
                float av = actf(v, (p.flags >> 4) & 7u, p.flags);
                ((device half*)o)[mm * p.N + nn] = half(av);
                if ((p.flags & 8u) != 0u)
                    ((device half*)o2)[mm * p.N + nn] =
                        (p.flags & 1024u) != 0u ? half(v) : half(0.5f * av * (1.0f + erf1(av * 0.70710678f)));
            }
        }
}
