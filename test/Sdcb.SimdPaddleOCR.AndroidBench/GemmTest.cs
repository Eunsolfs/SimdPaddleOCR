using System.Text.RegularExpressions;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>
/// GEMM checks with the graph's conv1x1 binding layout (x, w[N,K], bias,
/// res, o, o2; pc = M, N, K, flags): one dispatch vs an fp64 reference,
/// a variant sweep over hot shapes, and an FMA peak probe.
/// </summary>
static unsafe class GemmTest
{
    // --gemm <shader|file.spv> <M> <N> <K> [flags] [tileM] [tileN] [reqSg] [reps]
    internal static int Run(string[] args)
    {
        string shader = args[1];
        int M = int.Parse(args[2]), N = int.Parse(args[3]), K = int.Parse(args[4]);
        uint flags = args.Length > 5 ? uint.Parse(args[5]) : 0;
        uint tm = args.Length > 6 ? uint.Parse(args[6]) : 64, tn = args.Length > 7 ? uint.Parse(args[7]) : 64;
        uint reqSg = args.Length > 8 ? uint.Parse(args[8]) : 0;
        int reps = args.Length > 9 ? int.Parse(args[9]) : 10;
        using var dev = VkDevice.Create();
        using var h = new Harness(dev, (long)M * K, (long)N * K, (long)M * N, N);
        var pipe = dev.NewPipeline(dev.NewShaderModule(Spv(shader)), 6, 16, reqSg);
        double best = h.Time(pipe, M, N, K, flags, tm, tn, reps);
        var (maxErr, bad) = h.Verify(M, N, K, flags);
        Console.WriteLine($"gemm {shader} M={M} N={N} K={K} flags={flags} tile={tm}x{tn} reqSg={reqSg}: best={best:F4} ms ({2.0 * M * N * K / (best * 1e-3) / 1e12:F3} TFLOPS) maxErr={maxErr:G4} bad={bad}/{(long)M * N}");
        return bad == 0 ? 0 : 3;
    }

    static readonly (int M, int N, int K)[] s_hot =
    [
        (5760, 1024, 512), (5760, 512, 1024), (2880, 768, 1536), (2880, 1536, 768),
        (57600, 256, 128), (57600, 128, 256), (14400, 512, 256), (3600, 1024, 512),
        (900, 896, 1792), (57600, 64, 256), (11520, 256, 512),
    ];

    // --gemmsweep <spv[,spv...]> [rounds] [flags]: every variant is checked on
    // an odd shape against fp64, then timed on the hot shapes, variants
    // interleaved per round; min per shape over rounds
    internal static int Sweep(string[] args)
    {
        string[] shaders = args[1].Split(',');
        int rounds = args.Length > 2 ? int.Parse(args[2]) : 3;
        uint flags = args.Length > 3 ? uint.Parse(args[3]) : 0;
        using var dev = VkDevice.Create();
        using var h = new Harness(dev, s_hot.Max(s => (long)s.M * s.K), s_hot.Max(s => (long)s.N * s.K),
            s_hot.Max(s => (long)s.M * s.N), s_hot.Max(s => s.N));
        var pipes = new VkPipeline[shaders.Length];
        var tiles = new (uint, uint)[shaders.Length];
        for (int v = 0; v < shaders.Length; v++)
        {
            pipes[v] = dev.NewPipeline(dev.NewShaderModule(Spv(shaders[v])), 6, 16, 0);
            tiles[v] = Tile(shaders[v]);
            h.Time(pipes[v], 1000, 136, 264, flags | 3u, tiles[v].Item1, tiles[v].Item2, 1);
            var (err, bad) = h.Verify(1000, 136, 264, flags | 3u);
            Console.WriteLine($"variant {v} {shaders[v]} tile={tiles[v].Item1}x{tiles[v].Item2}: verify maxErr={err:G3} bad={bad}");
        }
        var best = new double[shaders.Length, s_hot.Length];
        for (int v = 0; v < shaders.Length; v++) for (int s = 0; s < s_hot.Length; s++) best[v, s] = double.MaxValue;
        for (int r = 0; r < rounds; r++)
            for (int s = 0; s < s_hot.Length; s++)
                for (int v = 0; v < shaders.Length; v++)
                {
                    var (M, N, K) = s_hot[s];
                    best[v, s] = Math.Min(best[v, s], h.Time(pipes[v], M, N, K, flags, tiles[v].Item1, tiles[v].Item2, 3));
                }
        Console.WriteLine("shape".PadRight(18) + string.Concat(Enumerable.Range(0, shaders.Length).Select(v => $"{"v" + v,10}")));
        var sum = new double[shaders.Length];
        for (int s = 0; s < s_hot.Length; s++)
        {
            var (M, N, K) = s_hot[s];
            var line = new System.Text.StringBuilder($"{M}x{N}x{K}".PadRight(18));
            for (int v = 0; v < shaders.Length; v++) { line.Append($"{best[v, s],10:F3}"); sum[v] += best[v, s]; }
            Console.WriteLine(line);
        }
        Console.WriteLine("sum ms".PadRight(18) + string.Concat(sum.Select(x => $"{x,10:F2}")));
        double flop = s_hot.Sum(s => 2.0 * s.M * s.N * s.K);
        Console.WriteLine("TFLOPS".PadRight(18) + string.Concat(sum.Select(x => $"{flop / (x * 1e-3) / 1e12,10:F3}")));
        Console.WriteLine($"thermal={E2eThermal()}");
        return 0;
    }

    // --peak: fp32 / fp16 FMA throughput of 16 independent vec4 chains per thread
    internal static int Peak(string[] args)
    {
        using var dev = VkDevice.Create();
        using var h = new Harness(dev, 4096, 4096, 4096, 64);
        foreach (string name in new[] { "peak.spv", "peak16.spv" })
        {
            using Stream s = typeof(GemmTest).Assembly.GetManifestResourceStream(name)!;
            byte[] b = new byte[s.Length];
            s.ReadExactly(b);
            var pipe = dev.NewPipeline(dev.NewShaderModule(b), 6, 16, 0);
            const uint groups = 4096;
            double best = h.Time(pipe, 0, 0, 0, 0, 0, 0, 10, groups);
            double flop = groups * 256.0 * 32768;
            Console.WriteLine($"peak {name}: {best:F3} ms -> {flop / (best * 1e-3) / 1e12:F3} TFLOPS");
        }
        return 0;
    }

    static int E2eThermal()
    {
#if ANDROID
        return E2e.Thermal();
#else
        return -1;
#endif
    }

    static (uint, uint) Tile(string shader)
    {
        var bm = Regex.Match(shader, @"bm(\d+)");
        var bn = Regex.Match(shader, @"bn(\d+)");
        return (bm.Success ? uint.Parse(bm.Groups[1].Value) : 64u, bn.Success ? uint.Parse(bn.Groups[1].Value) : 64u);
    }

    static byte[] Spv(string shader)
    {
        if (shader.EndsWith(".spv")) return File.ReadAllBytes(shader);
        using Stream s = typeof(VkDevice).Assembly.GetManifestResourceStream(
            $"Sdcb.SimdPaddleOCR.Backends.Vulkan.Shaders.{shader}.spv") ?? throw new FileNotFoundException(shader);
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }

    /// <summary>Random fp16 operands sized for the largest shape, bound once.</summary>
    sealed class Harness : IDisposable
    {
        readonly VkDevice _dev;
        readonly Half[] _x, _w, _bias, _res;
        readonly VkBuffer[] _binds;
        readonly IntPtr _cmd, _fence, _qp;

        public Harness(VkDevice dev, long xElems, long wElems, long mn, int maxN)
        {
            _dev = dev;
            var rr = new Random(3);
            mn = Math.Max(mn, 1000 * 136);
            _x = new Half[Math.Max(xElems, 1000 * 264) + 128 * 512];
            _w = new Half[Math.Max(wElems, 136 * 264) + 128 * 2048];
            _bias = new Half[Math.Max(maxN, 136) + 8];
            _res = new Half[mn];
            for (long i = 0; i < _x.Length; i++) _x[i] = (Half)(rr.NextDouble() * 2 - 1);
            for (long i = 0; i < _w.Length; i++) _w[i] = (Half)((rr.NextDouble() * 2 - 1) * 0.1);
            for (int i = 0; i < _bias.Length; i++) _bias[i] = (Half)(rr.NextDouble() - 0.5);
            for (long i = 0; i < _res.Length; i++) _res[i] = (Half)(rr.NextDouble() - 0.5);
            ulong outBytes = (ulong)mn * 2 + 4096;
            _binds = [Up(_x), Up(_w), Up(_bias), Up(_res),
                dev.NewStorageBuffer(outBytes, hostVisible: true, preferHost: true),
                dev.NewStorageBuffer(outBytes, hostVisible: true, preferHost: true)];
            _cmd = dev.NewCommandBuffer();
            _fence = dev.NewFence();
            var qci = new Vk.VkQueryPoolCreateInfo { SType = 11, QueryType = 2, QueryCount = 2 };
            Vk.Check(Vk.vkCreateQueryPool(dev.Device, &qci, null, out _qp), "qp");
        }

        VkBuffer Up(Half[] h)
        {
            var b = _dev.NewStorageBuffer((ulong)Math.Max(h.Length, 64) * 2, hostVisible: false);
            fixed (Half* p = h) _dev.Upload(b, p, (ulong)h.Length * 2);
            return b;
        }

        /// <summary>Min GPU time (ms) of <paramref name="reps"/> dispatches. The
        /// operands are laid out for the call's own (M, N, K).</summary>
        public double Time(VkPipeline pipe, int M, int N, int K, uint flags, uint tm, uint tn, int reps, uint groups = 0)
        {
            IntPtr set = _dev.NewDescriptorSet(pipe.SetLayout);
            for (uint i = 0; i < 6; i++) _dev.BindBuffer(set, i, _binds[i]);
            uint gx = groups != 0 ? groups : (uint)((M + tm - 1) / tm), gy = groups != 0 ? 1 : (uint)((N + tn - 1) / tn);
            uint[] pc = [(uint)M, (uint)N, (uint)K, flags];
            double best = double.MaxValue;
            ulong* ts = stackalloc ulong[2];
            for (int r = 0; r < reps; r++)
            {
                var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
                Vk.Check(Vk.vkResetCommandBuffer(_cmd, 0), "reset");
                Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "begin");
                Vk.vkCmdResetQueryPool(_cmd, _qp, 0, 2);
                Vk.vkCmdBindPipeline(_cmd, VkConst.BindPointCompute, pipe.Pipeline);
                Vk.vkCmdBindDescriptorSets(_cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
                fixed (uint* pp = pc) Vk.vkCmdPushConstants(_cmd, pipe.Layout, VkConst.StageComputeShader, 0, 16, pp);
                Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qp, 0);
                Vk.vkCmdDispatch(_cmd, gx, gy, 1);
                Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qp, 1);
                Vk.Check(Vk.vkEndCommandBuffer(_cmd), "end");
                _dev.Submit(_cmd, _fence); _dev.WaitFence(_fence);
                Vk.vkGetQueryPoolResults(_dev.Device, _qp, 0, 2, 16, ts, 8, 1 | 2);
                best = Math.Min(best, (ts[1] - ts[0]) * _dev.TimestampPeriodNs / 1e6);
            }
            _dev.FreeDescriptorSet(set);
            return best;
        }

        /// <summary>Checks the last dispatch's output (the operands as laid out
        /// for this M, N, K) against fp64.</summary>
        public (double MaxErr, long Bad) Verify(int M, int N, int K, uint flags)
        {
            Half* o = (Half*)_binds[4].Map();
            bool hasBias = (flags & 1) != 0, hasRes = (flags & 2) != 0;
            uint act = (flags >> 4) & 7;
            double maxErr = 0; long bad = 0;
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    double acc = 0;
                    for (int k = 0; k < K; k++) acc += (double)_x[(long)m * K + k] * (double)_w[(long)n * K + k];
                    if (hasBias) acc += (double)_bias[n];
                    if (hasRes) acc += (double)_res[(long)m * N + n];
                    acc = act switch
                    {
                        1 => Math.Max(acc, 0),
                        2 => 0.5 * acc * (1 + Erf(acc / Math.Sqrt(2))),
                        3 => acc * Math.Clamp(acc * (double)BitConverter.UInt16BitsToHalf((ushort)(flags >> 16)) + 0.5, 0, 1),
                        4 => 1 / (1 + Math.Exp(-acc)),
                        _ => acc,
                    };
                    double g = (double)o[(long)m * N + n];
                    double e = double.IsNaN(g) ? double.PositiveInfinity : Math.Abs(g - acc);
                    maxErr = Math.Max(maxErr, e);
                    if (e > 0.02 + 0.01 * Math.Abs(acc)) bad++;
                }
            _binds[4].Unmap();
            return (maxErr, bad);
        }

        public void Dispose()
        {
            Vk.vkDestroyQueryPool(_dev.Device, _qp, null);
            _dev.DestroyFence(_fence);
            foreach (var b in _binds) b.Free();
        }
    }

    // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    static double Erf(double v)
    {
        double s = Math.Sign(v); v = Math.Abs(v);
        double t = 1 / (1 + 0.3275911 * v);
        double y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-v * v);
        return s * y;
    }
}
