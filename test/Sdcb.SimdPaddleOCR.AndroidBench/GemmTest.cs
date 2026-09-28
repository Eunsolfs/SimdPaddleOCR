using System.Diagnostics;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>
/// Isolated GEMM check with the graph's conv1x1 binding layout
/// (x, w[N,K], bias, res, o, o2; pc = M, N, K, flags): GPU result vs a
/// double-precision reference on the same fp16 inputs, plus GPU time.
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
        byte[] spv = shader.EndsWith(".spv") ? File.ReadAllBytes(shader) : LibSpv(shader);

        using var dev = VkDevice.Create();
        var rr = new Random(3);
        int nPad = (N + 127) / 128 * 128;
        Half[] x = new Half[(long)M * K + 128 * 512], w = new Half[(long)nPad * K], bias = new Half[N], res = new Half[(long)M * N];
        for (long i = 0; i < (long)M * K; i++) x[i] = (Half)(rr.NextDouble() * 2 - 1);
        for (long i = 0; i < (long)N * K; i++) w[i] = (Half)((rr.NextDouble() * 2 - 1) * 0.1);
        for (int i = 0; i < N; i++) bias[i] = (Half)(rr.NextDouble() - 0.5);
        for (long i = 0; i < res.Length; i++) res[i] = (Half)(rr.NextDouble() - 0.5);

        VkBuffer Up(Half[] h)
        {
            var b = dev.NewStorageBuffer((ulong)Math.Max(h.Length, 64) * 2, hostVisible: false);
            fixed (Half* p = h) dev.Upload(b, p, (ulong)h.Length * 2);
            return b;
        }
        VkBuffer bx = Up(x), bw = Up(w), bb = Up(bias), br = Up(res);
        ulong outBytes = (ulong)M * (ulong)N * 2 + 4096;
        VkBuffer bo = dev.NewStorageBuffer(outBytes, hostVisible: true, preferHost: true);
        VkBuffer bo2 = dev.NewStorageBuffer(outBytes, hostVisible: true, preferHost: true);
        var pipe = dev.NewPipeline(dev.NewShaderModule(spv), 6, 16, reqSg);
        IntPtr set = dev.NewDescriptorSet(pipe.SetLayout);
        VkBuffer[] binds = [bx, bw, bb, br, bo, bo2];
        for (uint i = 0; i < 6; i++) dev.BindBuffer(set, i, binds[i]);
        uint gx = (uint)((M + tm - 1) / tm), gy = (uint)((N + tn - 1) / tn);
        uint[] pc = [(uint)M, (uint)N, (uint)K, flags];

        IntPtr cmd = dev.NewCommandBuffer();
        IntPtr fence = dev.NewFence();
        var qci = new Vk.VkQueryPoolCreateInfo { SType = 11, QueryType = 2, QueryCount = 2 };
        Vk.Check(Vk.vkCreateQueryPool(dev.Device, &qci, null, out IntPtr qp), "qp");
        double best = double.MaxValue;
        ulong* ts = stackalloc ulong[2];
        for (int r = 0; r < reps; r++)
        {
            var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
            Vk.Check(Vk.vkResetCommandBuffer(cmd, 0), "reset");
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
            Vk.vkCmdResetQueryPool(cmd, qp, 0, 2);
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
            Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
            fixed (uint* pp = pc) Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, 16, pp);
            Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, qp, 0);
            Vk.vkCmdDispatch(cmd, gx, gy, 1);
            Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, qp, 1);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
            dev.Submit(cmd, fence); dev.WaitFence(fence);
            Vk.vkGetQueryPoolResults(dev.Device, qp, 0, 2, 16, ts, 8, 1 | 2);
            best = Math.Min(best, (ts[1] - ts[0]) * dev.TimestampPeriodNs / 1e6);
        }

        Half* o = (Half*)bo.Map();
        bool hasBias = (flags & 1) != 0, hasRes = (flags & 2) != 0;
        uint act = (flags >> 4) & 7;
        double maxErr = 0; long bad = 0;
        var badRows = new SortedDictionary<int, int>();
        var badCols = new SortedDictionary<int, int>();
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double acc = 0;
                for (int k = 0; k < K; k++) acc += (double)x[(long)m * K + k] * (double)w[(long)n * K + k];
                if (hasBias) acc += (double)bias[n];
                if (hasRes) acc += (double)res[(long)m * N + n];
                acc = act switch
                {
                    1 => Math.Max(acc, 0),
                    2 => 0.5 * acc * (1 + Erf(acc / Math.Sqrt(2))),
                    3 => acc * Math.Clamp(acc * (double)BitConverter.UInt16BitsToHalf((ushort)(flags >> 16)) + 0.5, 0, 1),
                    4 => 1 / (1 + Math.Exp(-acc)),
                    _ => acc,
                };
                double g = (double)o[(long)m * N + n];
                double e = Math.Abs(g - acc);
                if (double.IsNaN(g)) e = double.PositiveInfinity;
                maxErr = Math.Max(maxErr, e);
                if (e > 0.02 + 0.01 * Math.Abs(acc))
                {
                    bad++;
                    badRows[m % 64] = badRows.GetValueOrDefault(m % 64) + 1;
                    badCols[n % 64] = badCols.GetValueOrDefault(n % 64) + 1;
                }
            }
        bo.Unmap();
        double tflops = 2.0 * M * N * K / (best * 1e-3) / 1e12;
        Console.WriteLine($"gemm {shader} M={M} N={N} K={K} flags={flags} grid={gx}x{gy} reqSg={reqSg}: best={best:F4} ms ({tflops:F3} TFLOPS) maxErr={maxErr:G4} bad={bad}/{(long)M * N}");
        if (bad > 0)
        {
            Console.WriteLine("  bad by row%64: " + string.Join(" ", badRows.Select(kv => $"{kv.Key}:{kv.Value}")));
            Console.WriteLine("  bad by col%64: " + string.Join(" ", badCols.Select(kv => $"{kv.Key}:{kv.Value}")));
        }
        Vk.vkDestroyQueryPool(dev.Device, qp, null);
        foreach (var b in binds) b.Free();
        return bad == 0 ? 0 : 3;
    }

    // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    static double Erf(double v)
    {
        double s = Math.Sign(v); v = Math.Abs(v);
        double t = 1 / (1 + 0.3275911 * v);
        double y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-v * v);
        return s * y;
    }

    static byte[] LibSpv(string name)
    {
        using Stream s = typeof(VkDevice).Assembly.GetManifestResourceStream(
            $"Sdcb.SimdPaddleOCR.Backends.Vulkan.Shaders.{name}.spv") ?? throw new FileNotFoundException(name);
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }
}
