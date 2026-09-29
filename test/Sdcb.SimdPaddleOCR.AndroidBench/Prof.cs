using System.Diagnostics;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Pure-GPU profiles (GpuBench --detprof / --recprof) and the
/// concurrent-caller check (GpuBench --conc), on the phone.</summary>
static class Prof
{
    internal static int Run(string[] args)
    {
        bool det = args[0] == "--detprof";
        string model = args[1];
        int a = int.Parse(args[2]), b = int.Parse(args[3]);
        int reps = args.Length >= 5 ? int.Parse(args[4]) : 20;
        var mdl = Model.Load(File.ReadAllBytes(Paths.ModelPath(model, det ? "det" : "rec")));
        var compiled = new CompiledModel(mdl, intraOpThreads: 8);
        int[] shape = det ? [1, 3, a, b] : [a, 3, 48, b];
        var inp = new float[shape[0] * shape[1] * shape[2] * shape[3]];
        var rr = new Random(1);
        for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);
        int nodeLimit = int.MaxValue, outTensor = -1;
        if (!det)
        {
            var cpu = new InferenceSession(compiled);
            cpu.Reshape(shape);
            cpu.TryRunUntilCtcProjection(inp, out CtcProjectionOperands ops);
            nodeLimit = ops.MatMulNodeIndex;
            outTensor = checked((int)mdl.Nodes[nodeLimit].Inputs[0]);
            Console.WriteLine($"rec: batch={a} T={ops.Rows} C={ops.Inner} cols={ops.Columns}");
        }
        using var dev = VkDevice.Create();
        using var gg = new GpuDetGraph(dev, compiled);
        double best = double.MaxValue;
        var walls = new List<double>();
        for (int r = 0; r < reps; r++)
        {
            long t0 = Stopwatch.GetTimestamp();
            gg.Run(shape, inp, nodeLimit, outTensor);
            double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            walls.Add(ms);
            best = Math.Min(best, ms);
        }
        walls.Sort();
        Console.WriteLine($"best wall {best:F2} ms (median {walls[walls.Count / 2]:F2}), dispatches={gg.DispatchCount(shape, nodeLimit, outTensor)} thermal={E2e.Thermal()}");
        gg.DumpProfile(shape, nodeLimit, outTensor);
        return 0;
    }

    // --conc <model> [threads] [backend]: serial reference texts vs concurrent callers
    internal static int Conc(string[] args)
    {
        string model = args[1];
        int threads = args.Length > 2 ? int.Parse(args[2]) : 4;
        OcrBackend backend = args.Length > 3 ? Paths.ParseBackend(args[3]) : OcrBackend.Vulkan;
        using var ocr = PaddleOcrAll.Load(Paths.ModelPath(model, "det"), Paths.ModelPath(model, "cls"),
            Paths.ModelPath(model, "rec"), Paths.DictPath(model), new PaddleOcrOptions
            {
                Detector = new PaddleOcrDetectorOptions { Backend = backend },
                Recognizer = new PaddleOcrRecognizerOptions { Backend = backend },
                Classifier = new PaddleOcrClassifierOptions { Backend = backend },
            });
        var imgs = Paths.DatasetFiles().Take(24).Select(Paths.LoadBgr).ToArray();
        string Texts(int i) => string.Join("|", ocr.Run(imgs[i].Bgr, imgs[i].W, imgs[i].H).Lines.Select(l => l.Text));
        string[] reference = Enumerable.Range(0, imgs.Length).Select(Texts).ToArray();
        int bad = 0;
        double wsPeak = E2e.WorkingSetMb();
        var tc = Stopwatch.StartNew();
        Parallel.For(0, imgs.Length * 3, new ParallelOptions { MaxDegreeOfParallelism = threads }, j =>
        {
            int i = j % imgs.Length;
            if (Texts(i) != reference[i]) { Interlocked.Increment(ref bad); Console.WriteLine($"MISMATCH img{i}"); }
            double ws = E2e.WorkingSetMb();
            lock (reference) wsPeak = Math.Max(wsPeak, ws);
        });
        Console.WriteLine($"conc model={model} backend={backend} usesGpu={GpuBackend.UsesGpu(backend)} threads={threads} runs={imgs.Length * 3} mismatches={bad} wall={tc.ElapsedMilliseconds}ms ws-peak={wsPeak:F0}MB");
        return bad == 0 ? 0 : 5;
    }
}
