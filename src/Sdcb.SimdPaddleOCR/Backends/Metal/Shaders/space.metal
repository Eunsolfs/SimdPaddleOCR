// Port of {concat_c, concat4, resize_nn, resize4, resize4add, catresize,
// maxpool2e, maxpool4, avgpool4, reduce_hw, reduce_hw4, reduce_hw4b}.comp
// NHWC fp16; gid.y (thread_position_in_grid.y) = batch where applicable.
#include <metal_stdlib>
using namespace metal;

// ---------------- concat_c ----------------
// NHWC channel-concat slice copy: o[px, cOff + c] = x[px, c], c < Cin.
struct PcConcat { uint hw, Cin, Cout, cOff; };
kernel void concat_c(device const half* x [[buffer(0)]],
                     device half* o [[buffer(1)]],
                     constant PcConcat& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint total = p.hw * p.Cin;
    if (gid >= total) return;
    uint c = gid % p.Cin;
    uint px = gid / p.Cin;
    o[px * p.Cout + p.cOff + c] = x[gid];
}

// ---------------- concat4 ----------------
// vec4 concat slice (Cin%4==0, cOff%4==0); flags bit8: x is unfused addps
// → x[i]*se[c4] + ra[i].
struct PcConcat4 { uint hw, Cin, Cout, cOff, flags; };
kernel void concat4(device const half4* x [[buffer(0)]],
                    device half4* o [[buffer(1)]],
                    device const half4* ra [[buffer(2)]],
                    device const half4* se [[buffer(3)]],
                    constant PcConcat4& p [[buffer(4)]],
                    uint gid [[thread_position_in_grid]])
{
    uint cv = p.Cin >> 2;
    if (gid >= p.hw * cv) return;
    uint c4 = gid % cv;
    uint px = gid / cv;
    float4 v = float4(x[gid]);
    if ((p.flags & 256u) != 0u) v = v * float4(se[c4]) + float4(ra[gid]);
    o[px * (p.Cout >> 2) + (p.cOff >> 2) + c4] = half4(v);
}

// ---------------- resize_nn ----------------
// Nearest-neighbour integer-factor upscale on NHWC fp16.
struct PcRs { uint outW, outH, C, fh, fw; };
kernel void resize_nn(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcRs& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    uint total = p.outW * p.outH * p.C;
    if (gid >= total) return;
    uint c = gid % p.C;
    uint px = gid / p.C;
    uint oy = px / p.outW, ox = px % p.outW;
    uint inW = p.outW / p.fw;
    o[gid] = x[((oy / p.fh) * inW + (ox / p.fw)) * p.C + c];
}

// ---------------- resize4 ----------------
// vec4 variant; dstCv != 0 → write into a wider-C tensor (concat slice).
// flags bit8: input is unfused addps → x[i]*se[c4] + ra[i].
struct PcRs4 { uint outW, outH, C, fh, fw, dstCv, dstCoff, flags; };
kernel void resize4(device const half4* x [[buffer(0)]],
                    device half4* o [[buffer(1)]],
                    device const half4* ra [[buffer(2)]],
                    device const half4* se [[buffer(3)]],
                    constant PcRs4& p [[buffer(4)]],
                    uint gid [[thread_position_in_grid]])
{
    uint cv = p.C >> 2;
    if (gid >= p.outW * p.outH * cv) return;
    uint c4 = gid % cv;
    uint px = gid / cv;
    uint oy = px / p.outW, ox = px % p.outW;
    uint inW = p.outW / p.fw;
    uint dcv = p.dstCv != 0u ? p.dstCv : cv;
    uint idx = ((oy / p.fh) * inW + (ox / p.fw)) * cv + c4;
    float4 v = float4(x[idx]);
    if ((p.flags & 256u) != 0u) v = v * float4(se[c4]) + float4(ra[idx]);
    o[px * dcv + p.dstCoff + c4] = half4(v);
}

// ---------------- resize4add ----------------
// Fused upscale + elementwise add (FPN merge): o = a + resize(x).
// flags bit8: x unfused addps; bit9: a unfused addps.
struct PcRs4A { uint outW, outH, C, fh, fw, flags; };
kernel void resize4add(device const half4* x [[buffer(0)]],
                       device const half4* a [[buffer(1)]],
                       device half4* o [[buffer(2)]],
                       device const half4* ra [[buffer(3)]],
                       device const half4* se [[buffer(4)]],
                       constant PcRs4A& p [[buffer(5)]],
                       uint gid [[thread_position_in_grid]])
{
    uint cv = p.C >> 2;
    if (gid >= p.outW * p.outH * cv) return;
    uint c4 = gid % cv;
    uint px = gid / cv;
    uint oy = px / p.outW, ox = px % p.outW;
    uint inW = p.outW / p.fw;
    uint idx = ((oy / p.fh) * inW + (ox / p.fw)) * cv + c4;
    float4 xv = float4(x[idx]), av = float4(a[gid]);
    if ((p.flags & 256u) != 0u) xv = xv * float4(se[c4]) + float4(ra[idx]);
    if ((p.flags & 512u) != 0u) av = av * float4(se[c4]) + float4(ra[gid]);
    o[gid] = half4(av + xv);
}

// ---------------- catresize ----------------
// Fused channel-concat: each input slice is a (nearest) resize of a possibly
// addps-absorbed source. Up to 4 slices; slice s occupies output channel-quads
// [off4_s, off4_s + cq4_s). Per-slice flags bit0 = addps absorb
// (v = x[i]*se[q] + ra[i]).
struct PcCatRes {
    uint outW, outH, Cout;
    uint off4a, inWa, fha, fla;
    uint off4b, inWb, fhb, flb;
    uint off4c, inWc, fhc, flc;
    uint off4d, inWd, fhd, fld;
};
struct CatSlice { device const half4 *x, *r, *s; };
static inline float4 cat_src(device const half4* x, device const half4* r,
                             device const half4* s, uint idx, uint q, uint fl)
{
    float4 v = float4(x[idx]);
    if ((fl & 1u) != 0u) v = v * float4(s[q]) + float4(r[idx]);
    return v;
}
kernel void catresize(device const half4* x0 [[buffer(0)]],
                      device const half4* r0 [[buffer(1)]],
                      device const half4* s0 [[buffer(2)]],
                      device const half4* x1 [[buffer(3)]],
                      device const half4* r1 [[buffer(4)]],
                      device const half4* s1 [[buffer(5)]],
                      device const half4* x2 [[buffer(6)]],
                      device const half4* r2 [[buffer(7)]],
                      device const half4* s2 [[buffer(8)]],
                      device const half4* x3 [[buffer(9)]],
                      device const half4* r3 [[buffer(10)]],
                      device const half4* s3 [[buffer(11)]],
                      device half4* o [[buffer(12)]],
                      constant PcCatRes& p [[buffer(13)]],
                      uint gid [[thread_position_in_grid]])
{
    uint cv4 = p.Cout >> 2;
    if (gid >= p.outW * p.outH * cv4) return;
    uint c4 = gid % cv4;
    uint px = gid / cv4;
    uint oy = px / p.outW, ox = px % p.outW;

    for (uint s = 0u; s < 4u; s++) {
        uint offc, inW, fhfw, fl;
        switch (s) {
        case 0u: offc = p.off4a; inW = p.inWa; fhfw = p.fha; fl = p.fla; break;
        case 1u: offc = p.off4b; inW = p.inWb; fhfw = p.fhb; fl = p.flb; break;
        case 2u: offc = p.off4c; inW = p.inWc; fhfw = p.fhc; fl = p.flc; break;
        default: offc = p.off4d; inW = p.inWd; fhfw = p.fhd; fl = p.fld; break;
        }
        uint off4 = offc & 0xffffu, cq4 = offc >> 16;
        if (cq4 == 0u || c4 < off4 || c4 >= off4 + cq4) continue;
        uint q = c4 - off4;
        uint sy = oy / (fhfw & 0xffffu), sx = ox / (fhfw >> 16);
        uint idx = (sy * inW + sx) * cq4 + q;
        float4 v;
        switch (s) {
        case 0u: v = cat_src(x0, r0, s0, idx, q, fl); break;
        case 1u: v = cat_src(x1, r1, s1, idx, q, fl); break;
        case 2u: v = cat_src(x2, r2, s2, idx, q, fl); break;
        default: v = cat_src(x3, r3, s3, idx, q, fl); break;
        }
        o[gid] = half4(v);
        return;
    }
}

// ---------------- maxpool2e ----------------
// MaxPool 2x2 stride 1, pad-end, NHWC fp16 — same HxW out.
struct PcPool { uint W, H, C; };
kernel void maxpool2e(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcPool& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    uint total = p.W * p.H * p.C;
    if (gid >= total) return;
    uint c = gid % p.C;
    uint px = gid / p.C;
    uint y = px / p.W, xx = px % p.W;
    float m = -1.0e30f;
    for (uint dy = 0u; dy < 2u; dy++) {
        uint iy = y + dy;
        if (iy >= p.H) continue;
        for (uint dx = 0u; dx < 2u; dx++) {
            uint ix = xx + dx;
            if (ix >= p.W) continue;
            m = max(m, float(x[(iy * p.W + ix) * p.C + c]));
        }
    }
    o[gid] = half(m);
}

// ---------------- maxpool4 ----------------
// vec4 variant (C%4==0); dstCv != 0 → concat-slice write. gid.y = batch.
struct PcPool4 { uint W, H, C, dstCv, dstCoff; };
kernel void maxpool4(device const half4* x [[buffer(0)]],
                     device half4* o [[buffer(1)]],
                     constant PcPool4& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint cv = p.C >> 2;
    if (gid.x >= p.W * p.H * cv) return;
    uint c4 = gid.x % cv;
    uint px = gid.x / cv;
    uint y = px / p.W, xx = px % p.W;
    uint b = gid.y;
    uint xb = b * p.W * p.H * cv;

    half4 m = half4(half(-1.0e4));
    for (uint dy = 0u; dy < 2u; dy++) {
        uint iy = y + dy;
        if (iy >= p.H) continue;
        for (uint dx = 0u; dx < 2u; dx++) {
            uint ix = xx + dx;
            if (ix >= p.W) continue;
            half4 v = x[xb + (iy * p.W + ix) * cv + c4];
            m = max(m, v);
        }
    }
    uint dcv = p.dstCv != 0u ? p.dstCv : cv;
    o[(b * p.W * p.H + px) * dcv + p.dstCoff + c4] = m;
}

// ---------------- avgpool4 ----------------
// AveragePool NHWC fp16 vec4 (C%4==0); OOB excluded (count_include_pad=0).
// gid.y = batch. PC: outW, outH, C, kH, kW, sh, sw, pt, pl, inW, inH.
struct PcAvg {
    uint outW, outH, C, kH, kW, sh, sw, pt, pl, inW, inH;
};
kernel void avgpool4(device const half4* x [[buffer(0)]],
                     device half4* o [[buffer(1)]],
                     constant PcAvg& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.y;
    uint cv = p.C >> 2;
    uint total = p.outW * p.outH * cv;
    if (gid.x >= total) return;
    uint c4 = gid.x % cv;
    uint px = gid.x / cv;
    uint oy = px / p.outW, ox = px % p.outW;
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);
    uint xb = b * p.inH * p.inW * cv;

    float4 acc = float4(0.0);
    uint cnt = 0u;
    for (uint dy = 0u; dy < p.kH; dy++) {
        int iy = iy0 + int(dy);
        if (iy < 0 || iy >= int(p.inH)) continue;
        uint rowBase = xb + uint(iy) * p.inW * cv;
        for (uint dx = 0u; dx < p.kW; dx++) {
            int ix = ix0 + int(dx);
            if (ix < 0 || ix >= int(p.inW)) continue;
            acc += float4(x[rowBase + uint(ix) * cv + c4]);
            cnt++;
        }
    }
    if (cnt == 0u) cnt = 1u;
    o[b * total + gid.x] = half4(acc / float(cnt));
}

// ---------------- reduce_hw ----------------
// Per-channel spatial mean: out[c] = mean_px x[px, c]. One threadgroup per
// channel, 256 threads strided-reduce into shared, fp32 acc. gid.y = batch.
struct PcRed { uint hw, C; };
kernel void reduce_hw(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcRed& p [[buffer(2)]],
                      uint2 wg [[threadgroup_position_in_grid]],
                      uint2 lp2 [[thread_position_in_threadgroup]])
{
    threadgroup float sm[256];
    uint lane = lp2.x;
    uint c = wg.x;
    uint b = wg.y;
    float acc = 0.0f;
    for (uint px = lane; px < p.hw; px += 256u)
        acc += float(x[b * p.hw * p.C + px * p.C + c]);
    sm[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (lane < s) sm[lane] += sm[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) o[b * p.C + c] = half(sm[0] / float(p.hw));
}

// ---------------- reduce_hw4 (phase 1) ----------------
// Partition partial sums: group s covers pixels [s*P, s*P+P); 256 lanes laid
// out (strip = lane/cv, cLane = lane%cv) → coalesced vec4 reads. Writes fp32
// partials. gid.y = batch.
struct PcRed4 { uint hw, C, S, P; };
kernel void reduce_hw4(device const half4* x [[buffer(0)]],
                       device float* part [[buffer(1)]],
                       constant PcRed4& p [[buffer(2)]],
                       uint2 wg [[threadgroup_position_in_grid]],
                       uint2 lp2 [[thread_position_in_threadgroup]])
{
    threadgroup float4 sm[256];
    uint lane = lp2.x;
    uint s = wg.x;
    uint b = wg.y;
    uint cv = p.C >> 2;
    uint pb = b * p.S * p.C;
    uint xb = b * p.hw * cv;
    uint strips = 256u / cv;
    uint strip = lane / cv;
    uint cLane = lane - strip * cv;
    float4 acc = float4(0.0);
    uint beg = s * p.P;
    uint end = min(beg + p.P, p.hw);
    for (uint px = beg + strip; px < end; px += strips)
        acc += float4(x[xb + px * cv + cLane]);
    sm[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = strips >> 1; st > 0u; st >>= 1) {
        if (strip < st) sm[lane] += sm[lane + st * cv];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (strip == 0u) {
        uint ob = pb + s * p.C + cLane * 4u;
        part[ob] = sm[lane].x; part[ob + 1u] = sm[lane].y;
        part[ob + 2u] = sm[lane].z; part[ob + 3u] = sm[lane].w;
    }
}

// ---------------- reduce_hw4b (phase 2) ----------------
// out[c] = (sum_s part[s][c]) / hw. One thread per channel; gid.y = batch.
struct PcRed4b { uint S, C, hw; };
kernel void reduce_hw4b(device const float* part [[buffer(0)]],
                        device half* o [[buffer(1)]],
                        constant PcRed4b& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x;
    uint b = gid.y;
    if (c >= p.C) return;
    float acc = 0.0f;
    for (uint s = 0u; s < p.S; s++) acc += part[b * p.S * p.C + s * p.C + c];
    o[b * p.C + c] = half(acc / float(p.hw));
}
