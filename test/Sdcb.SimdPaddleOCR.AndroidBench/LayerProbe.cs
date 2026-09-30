using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>
/// Bisection aid: truncates the GPU graph after node k and reads that node's
/// output back through the graph's own fp32 writeback, printing statistics.
/// Plain .NET, so the desktop GpuBench links it too for reference numbers.
/// </summary>
static class LayerProbe
{
    // --layers <onnx> <N> <H> <W> [firstNode] [lastNode]
    internal static int Run(string onnx, int n, int h, int w, int first = 0, int last = int.MaxValue)
    {
        var mdl = Model.Load(File.ReadAllBytes(onnx));
        var compiled = new CompiledModel(mdl, intraOpThreads: 4);
        int[] shape = [n, 3, h, w];
        var inp = new float[n * 3 * h * w];
        var rr = new Random(7);
        for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);
        using var dev = VkDevice.Create();
        using var gg = new GpuDetGraph(dev, compiled);
        Console.WriteLine($"device={dev.DeviceName} nodes={mdl.Nodes.Length}");
        var nodes = mdl.Nodes;
        for (int k = first; k <= Math.Min(last, nodes.Length - 1); k++)
        {
            int t = checked((int)nodes[k].Outputs[0]);
            GpuSchedule s;
            try { s = gg.Prepare(shape, k + 1, t); }
            catch (Exception ex) { Console.WriteLine($"n{k,-4} {nodes[k].Operator,-14} plan failed: {ex.GetBaseException().Message}"); continue; }
            int p = t;
            while (s.Alias[p] != p) p = s.Alias[p];
            if (s.Off[p] < 0 || s.Recs.Length < 2)
            {
                if (Environment.GetEnvironmentVariable("LAYERS_VERBOSE") == "1")
                    Console.WriteLine($"n{k,-4} {nodes[k].Operator,-14} skipped: t{t} phys{p} off={s.Off[p]} recs={s.Recs.Length}");
                continue;
            }
            float[] v = gg.Run(shape, inp, k + 1, t).ToArray();
            double sum = 0, sumAbs = 0; float mn = float.MaxValue, mx = float.MinValue; int nz = 0, nan = 0;
            foreach (float x in v)
            {
                if (float.IsNaN(x) || float.IsInfinity(x)) { nan++; continue; }
                mn = MathF.Min(mn, x); mx = MathF.Max(mx, x); sum += x; sumAbs += Math.Abs(x);
                if (x != 0) nz++;
            }
            string last2 = string.Join(" | ", s.Recs.Take(s.Recs.Length - 1).TakeLast(2).Select(r => r.Tag));
            Console.WriteLine($"n{k,-4} {nodes[k].Operator,-14} numel={v.Length,-9} mean={sum / v.Length,10:G5} meanAbs={sumAbs / v.Length,10:G5} min={mn,10:G5} max={mx,10:G5} nz={nz}/{v.Length} bad={nan} recs={s.Recs.Length} [{last2}]");
        }
        return 0;
    }
}
