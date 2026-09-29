// Port of {conv_dw, conv_dw4, conv_dw4t(+ABSORB), conv_dense, convt2s2,
// convt4, im2col}.comp — depthwise/dense/transpose convs on NHWC fp16.
// flags: bit0 bias, bit3 scalarBias(convt), bits4-6 act, bit8 absorb.
#include <metal_stdlib>
using namespace metal;

struct PcDw {
    uint outW, outH, C, kH, kW, sh, sw, pt, pl, inW, inH, flags;
};

// ---------------- conv_dw ----------------
// Depthwise conv, one thread per output element (scalar C).
kernel void conv_dw(device const half* x [[buffer(0)]],
                    device const half* w [[buffer(1)]],
                    device const half* bias [[buffer(2)]],
                    device half* o [[buffer(3)]],
                    constant PcDw& p [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]])
{
    uint total = p.outW * p.outH * p.C;
    uint g = gid.x;
    if (g >= total) return;
    uint c = g % p.C;
    uint px = g / p.C;
    uint oy = px / p.outW, ox = px % p.outW;
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);

    float acc = 0.0f;
    for (uint dy = 0u; dy < p.kH; dy++) {
        int iy = iy0 + int(dy);
        if (iy < 0 || iy >= int(p.inH)) continue;
        for (uint dx = 0u; dx < p.kW; dx++) {
            int ix = ix0 + int(dx);
            if (ix < 0 || ix >= int(p.inW)) continue;
            acc += float(w[c * p.kH * p.kW + dy * p.kW + dx]) *
                   float(x[(uint(iy) * p.inW + uint(ix)) * p.C + c]);
        }
    }
    if ((p.flags & 1u) != 0u) acc += float(bias[c]);
    o[g] = half(actf(acc, (p.flags >> 4) & 7u, p.flags));
}

// ---------------- conv_dw4 ----------------
// 4 channels per thread (C%4==0); weights tap-major [tap*C + c] (vec4).
// flags bit8: input is unfused addps → x*se[c4]+ra[i] reads (conv_dw4a).
kernel void conv_dw4(device const half4* x [[buffer(0)]],
                     device const half4* w [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device half4* o [[buffer(3)]],
                     constant PcDw& p [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint cv = p.C >> 2;
    uint total = p.outW * p.outH * cv;
    if (gid.x >= total) return;
    uint b = gid.y;
    uint xb = b * p.inH * p.inW * cv;
    uint c4 = gid.x % cv;
    uint px = gid.x / cv;
    uint oy = px / p.outW, ox = px % p.outW;
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);

    float4 acc = float4(0.0);
    for (uint dy = 0u; dy < p.kH; dy++) {
        int iy = iy0 + int(dy);
        if (iy < 0 || iy >= int(p.inH)) continue;
        uint rowBase = xb + uint(iy) * p.inW * cv;
        for (uint dx = 0u; dx < p.kW; dx++) {
            int ix = ix0 + int(dx);
            if (ix < 0 || ix >= int(p.inW)) continue;
            uint tap = dy * p.kW + dx;
            float4 wv = float4(w[tap * cv + c4]);
            acc += wv * float4(x[rowBase + uint(ix) * cv + c4]);
        }
    }
    if ((p.flags & 1u) != 0u) {
        uint cb = c4 * 4u;
        acc += float4(float(bias[cb]), float(bias[cb + 1u]),
                      float(bias[cb + 2u]), float(bias[cb + 3u]));
    }
    uint act = (p.flags >> 4) & 7u;
    o[b * total + gid.x] = half4(actf4(acc, act, p.flags));
}

// ---------------- conv_dw4a ----------------
// conv_dw4 + addps absorb (extra ra/se binds; bit8 always set on emit).
kernel void conv_dw4a(device const half4* x [[buffer(0)]],
                      device const half4* w [[buffer(1)]],
                      device const half* bias [[buffer(2)]],
                      device half4* o [[buffer(3)]],
                      device const half4* ra [[buffer(4)]],
                      device const half4* se [[buffer(5)]],
                      constant PcDw& p [[buffer(6)]],
                      uint2 gid [[thread_position_in_grid]])
{
    uint cv = p.C >> 2;
    uint total = p.outW * p.outH * cv;
    if (gid.x >= total) return;
    uint b = gid.y;
    uint xb = b * p.inH * p.inW * cv;
    uint c4 = gid.x % cv;
    uint px = gid.x / cv;
    uint oy = px / p.outW, ox = px % p.outW;
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);

    float4 acc = float4(0.0);
    for (uint dy = 0u; dy < p.kH; dy++) {
        int iy = iy0 + int(dy);
        if (iy < 0 || iy >= int(p.inH)) continue;
        uint rowBase = xb + uint(iy) * p.inW * cv;
        for (uint dx = 0u; dx < p.kW; dx++) {
            int ix = ix0 + int(dx);
            if (ix < 0 || ix >= int(p.inW)) continue;
            uint tap = dy * p.kW + dx;
            float4 wv = float4(w[tap * cv + c4]);
            uint gi = rowBase + uint(ix) * cv + c4;
            float4 xv = (p.flags & 256u) != 0u
                ? float4(x[gi]) * float4(se[b * cv + c4]) + float4(ra[gi])
                : float4(x[gi]);
            acc += wv * xv;
        }
    }
    if ((p.flags & 1u) != 0u) {
        uint cb = c4 * 4u;
        acc += float4(float(bias[cb]), float(bias[cb + 1u]),
                      float(bias[cb + 2u]), float(bias[cb + 3u]));
    }
    uint act = (p.flags >> 4) & 7u;
    o[b * total + gid.x] = half4(actf4(acc, act, p.flags));
}

// ---------------- conv_dw4t ----------------
// Shared-tile depthwise (C%4==0, k<=9, s<=2): one 256-thread group per
// 16x16 output tile × 4 channels; the (16*s+k-1)² input halo is staged in
// threadgroup memory once.
// grid x = tilesX*tilesY*(C/4) groups; gy = batch.
kernel void conv_dw4t(device const half4* x [[buffer(0)]],
                      device const half4* w [[buffer(1)]],
                      device const half* bias [[buffer(2)]],
                      device half4* o [[buffer(3)]],
                      device const half4* ra [[buffer(4)]],
                      device const half4* se [[buffer(5)]],
                      constant PcDw& p [[buffer(6)]],
                      uint2 wg [[threadgroup_position_in_grid]],
                      uint2 tp2 [[thread_position_in_threadgroup]])
{
    uint tid = tp2.x;
    // worst tile: s=2,k=9 → (16*2+8)² = 40² half4 = 12.8KB + 81 weights
    threadgroup half4 shTile[40 * 40];
    threadgroup half4 shW[81];
    uint cv = p.C >> 2;
    uint b = wg.y;
    uint xb = b * p.inH * p.inW * cv;
    uint seb = b * cv;
    uint tilesX = (p.outW + 15u) / 16u;
    uint g = wg.x;
    uint c4 = g % cv;
    uint tile = g / cv;
    uint tx = tile % tilesX, ty = tile / tilesX;

    int iyT = int(ty * 16u * p.sh) - int(p.pt);
    int ixT = int(tx * 16u * p.sw) - int(p.pl);
    uint stW = 16u * p.sw + p.kW - 1u;
    uint stH = 16u * p.sh + p.kH - 1u;
    uint tileElems = stW * stH;

    for (uint e = tid; e < tileElems; e += 256u) {
        uint iy = e / stW, ix = e - iy * stW;
        int gy = iyT + int(iy), gx2 = ixT + int(ix);
        half4 v = half4(0.0);
        if (gy >= 0 && uint(gy) < p.inH && gx2 >= 0 && uint(gx2) < p.inW) {
            uint gi = xb + (uint(gy) * p.inW + uint(gx2)) * cv + c4;
            v = x[gi];
            if ((p.flags & 256u) != 0u)
                v = half4(float4(v) * float4(se[seb + c4]) + float4(ra[gi]));
        }
        shTile[e] = v;
    }
    uint taps = p.kH * p.kW;
    for (uint e = tid; e < taps; e += 256u)
        shW[e] = w[e * cv + c4];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ox = tx * 16u + (tid & 15u);
    uint oy = ty * 16u + (tid >> 4);
    if (ox >= p.outW || oy >= p.outH) return;

    uint lx0 = (tid & 15u) * p.sw;
    uint ly0 = (tid >> 4) * p.sh;
    float4 acc = float4(0.0);
    for (uint dy = 0u; dy < p.kH; dy++)
        for (uint dx = 0u; dx < p.kW; dx++)
            acc += float4(shW[dy * p.kW + dx]) * float4(shTile[(ly0 + dy) * stW + lx0 + dx]);
    if ((p.flags & 1u) != 0u) {
        uint cb = c4 * 4u;
        acc += float4(float(bias[cb]), float(bias[cb + 1u]),
                      float(bias[cb + 2u]), float(bias[cb + 3u]));
    }
    uint act = (p.flags >> 4) & 7u;
    o[b * p.outH * p.outW * cv + (oy * p.outW + ox) * cv + c4] = half4(actf4(acc, act, p.flags));
}

// ---------------- conv_dense ----------------
// Small dense conv (any k/s/pad), one thread per output element.
// W OIHW flattened [Cout, Cin*kH*kW].
struct PcDense {
    uint outW, outH, Cout, Cin, kH, kW, sh, sw, pt, pl, inW, inH, flags;
};
kernel void conv_dense(device const half* x [[buffer(0)]],
                       device const half* w [[buffer(1)]],
                       device const half* bias [[buffer(2)]],
                       device half* o [[buffer(3)]],
                       constant PcDense& p [[buffer(4)]],
                       uint gid [[thread_position_in_grid]])
{
    uint total = p.outW * p.outH * p.Cout;
    if (gid >= total) return;
    uint co = gid % p.Cout;
    uint px = gid / p.Cout;
    uint oy = px / p.outW, ox = px % p.outW;
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);
    uint kSpan = p.kH * p.kW;

    float acc = 0.0f;
    for (uint dy = 0u; dy < p.kH; dy++) {
        int iy = iy0 + int(dy);
        if (iy < 0 || iy >= int(p.inH)) continue;
        for (uint dx = 0u; dx < p.kW; dx++) {
            int ix = ix0 + int(dx);
            if (ix < 0 || ix >= int(p.inW)) continue;
            uint xbase = (uint(iy) * p.inW + uint(ix)) * p.Cin;
            uint wbase = co * (p.Cin * kSpan) + dy * p.kW + dx;
            for (uint ci = 0u; ci < p.Cin; ci++)
                acc += float(w[wbase + ci * kSpan]) * float(x[xbase + ci]);
        }
    }
    if ((p.flags & 1u) != 0u) acc += float(bias[co]);
    o[gid] = half(actf(acc, (p.flags >> 4) & 7u, p.flags));
}

// ---------------- convt2s2 ----------------
// ConvTranspose 2x2 s2 (PP-OCR FPN head): out px depends on exactly one input
// px (oy/2, ox/2) and the (oy%2, ox%2) tap. W [Cin, Cout, 2, 2] flattened.
// flags bit7: write fp32 graph output to `of` instead of fp16 `o`.
struct PcCt { uint outW, outH, Cout, Cin, inW, flags; };
kernel void convt2s2(device const half* x [[buffer(0)]],
                     device const half* w [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device half* o [[buffer(3)]],
                     device float* of [[buffer(4)]],
                     constant PcCt& p [[buffer(5)]],
                     uint gid [[thread_position_in_grid]])
{
    uint total = p.outW * p.outH * p.Cout;
    if (gid >= total) return;
    uint co = gid % p.Cout;
    uint px = gid / p.Cout;
    uint oy = px / p.outW, ox = px % p.outW;
    uint iy = oy >> 1, ix = ox >> 1;
    uint dy = oy & 1u, dx = ox & 1u;

    float acc = 0.0f;
    uint xbase = (iy * p.inW + ix) * p.Cin;
    uint tap = dy * 2u + dx;
    for (uint ci = 0u; ci < p.Cin; ci++)
        acc += float(w[(ci * p.Cout + co) * 4u + tap]) * float(x[xbase + ci]);
    if ((p.flags & 1u) != 0u) acc += float(bias[(p.flags & 8u) != 0u ? 0u : co]);
    float v = actf(acc, (p.flags >> 4) & 7u, p.flags);
    if ((p.flags & 128u) != 0u) of[gid] = v;
    else o[gid] = half(v);
}

// ---------------- convt4 ----------------
// vec4 convtranspose: weights repacked [tap][ci][co] fp16.
kernel void convt4(device const half4* x [[buffer(0)]],
                   device const half4* w [[buffer(1)]],
                   device const half* bias [[buffer(2)]],
                   device half4* o [[buffer(3)]],
                   constant PcCt& p [[buffer(4)]],
                   uint gid [[thread_position_in_grid]])
{
    uint nv = p.Cout >> 2;
    if (gid >= p.outW * p.outH * nv) return;
    uint co4 = gid % nv;
    uint px = gid / nv;
    uint oy = px / p.outW, ox = px % p.outW;
    uint iy = oy >> 1, ix = ox >> 1;
    uint dy = oy & 1u, dx = ox & 1u;

    float4 acc = float4(0.0);
    uint xb = (iy * p.inW + ix) * (p.Cin >> 2);
    uint tap = dy * 2u + dx;
    uint civ = p.Cin >> 2;
    for (uint ci4 = 0u; ci4 < civ; ci4++) {
        float4 xv = float4(x[xb + ci4]);
        uint wbase = (tap * p.Cin + ci4 * 4u) * nv + co4;
        for (uint i = 0u; i < 4u; i++)
            acc += xv[i] * float4(w[wbase + i * nv]);
    }
    if ((p.flags & 1u) != 0u) {
        if ((p.flags & 8u) != 0u) acc += float4(float(bias[0]));
        else acc += float4(float(bias[co4 * 4u]), float(bias[co4 * 4u + 1u]),
                           float(bias[co4 * 4u + 2u]), float(bias[co4 * 4u + 3u]));
    }
    uint act = (p.flags >> 4) & 7u;
    o[gid] = half4(actf4(acc, act, p.flags));
}

// ---------------- im2col ----------------
// im2col gather, tap-major: k = (dy*kW+dx)*Cin + ci over [0,K); pad → 0.
// Each thread emits 4 consecutive k (Kp%4==0) → one half4 store.
// flags bit0: fully-scalar fallback when Cin%4 != 0.
struct PcIm2col {
    uint outW, outH, Cin, kH, kW, sH, sW, pT, pL, inW, inH, K, Kp, flags;
};
kernel void im2col(device const half4* x [[buffer(0)]],
                   device half4* a [[buffer(1)]],
                   constant PcIm2col& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
    uint kv = p.Kp >> 2;
    uint total = p.outW * p.outH * kv;
    if (gid >= total) return;
    uint k4 = (gid % kv) * 4u;
    uint m = gid / kv;
    uint oy = m / p.outW, ox = m % p.outW;
    half4 v = half4(0.0);
    if ((p.flags & 1u) == 0u) {
        if (k4 < p.K) {
            uint tap = k4 / p.Cin;
            uint ci = k4 - tap * p.Cin;
            uint dy = tap / p.kW, dx = tap % p.kW;
            int iy = int(oy * p.sH + dy) - int(p.pT);
            int ix = int(ox * p.sW + dx) - int(p.pL);
            if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW))
                v = x[(uint(iy) * p.inW + uint(ix)) * (p.Cin >> 2) + (ci >> 2)];
        }
    } else {
        for (uint i = 0u; i < 4u; i++) {
            uint k = k4 + i;
            if (k < p.K) {
                uint tap = k / p.Cin, ci = k - tap * p.Cin;
                uint dy = tap / p.kW, dx = tap % p.kW;
                int iy = int(oy * p.sH + dy) - int(p.pT);
                int ix = int(ox * p.sW + dx) - int(p.pL);
                if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW)) {
                    uint xb = (uint(iy) * p.inW + uint(ix)) * p.Cin + ci;
                    v[i] = x[xb >> 2][xb & 3u];
                }
            }
        }
    }
    a[gid] = v;
}
