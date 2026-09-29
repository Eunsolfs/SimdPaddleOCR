using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Same input through the CPU and Vulkan paths on the phone.</summary>
static class Compare
{
    internal static int Run(string[] args) => args[0] == "--detcmp" ? Det(args) : Rec(args);

    // --detcmp <model> <img.jpg|all|N> [maxSide]
    static int Det(string[] args)
    {
        string model = args[1];
        int maxSide = args.Length > 3 ? int.Parse(args[3]) : 960;
        string[] files = args[2] == "all" ? Paths.DatasetFiles()
            : int.TryParse(args[2], out int n) ? Paths.DatasetFiles().Take(n).ToArray()
            : [Paths.Path_(args[2])];
        var mdl = Model.Load(File.ReadAllBytes(Paths.ModelPath(model, "det")));
        var compiled = new CompiledModel(mdl, intraOpThreads: 4);
        var opts = new PaddleOcrDetectorOptions();
        var scratch = new DbPostprocess.Workspace();
        using var sCpu = OcrSessionFactory.Create(compiled, OcrBackend.Cpu);
        using var sGpu = OcrSessionFactory.Create(compiled, OcrBackend.Vulkan);
        Console.WriteLine($"cpu session={sCpu.GetType().Name} gpu session={sGpu.GetType().Name}");
        int sameBoxes = 0;
        foreach (string f in files)
        {
            var (bgr, w, h) = Paths.LoadBgr(f);
            var size = PPOCRPreprocess.ComputeDetSize(w, h, maxSide);
            int W = size.Width, H = size.Height;
            float[][] maps = new float[2][];
            int[] nbox = new int[2];
            string[] boxStr = new string[2];
            IOcrSession[] ss = [sCpu, sGpu];
            for (int k = 0; k < 2; k++)
            {
                IOcrSession s = ss[k];
                s.Reshape([1, 3, H, W]);
                PPOCRPreprocess.Det(bgr, w, h, w * 3, W, H, s.InputData, s.ResizeWorkspace, 1, s.InputIsNhwc,
                    ImagePixelFormat.Bgr24);
                maps[k] = s.RunInternal(s.InputData).ToArray();
                var boxes = DbPostprocess.Run(maps[k], W, H, opts, w, h, size.WidthRatio, size.HeightRatio, scratch);
                nbox[k] = boxes.Length;
                boxStr[k] = string.Join(";", boxes.Select(b => $"{b.X1:F0},{b.Y1:F0},{b.X3:F0},{b.Y3:F0}"));
            }
            float[] pc = maps[0], pg = maps[1];
            double md = 0, sum = 0; int up = 0, dn = 0, nz = 0;
            for (int i = 0; i < pc.Length; i++)
            {
                double d = Math.Abs(pc[i] - pg[i]);
                md = Math.Max(md, d); sum += d;
                if (pg[i] != 0) nz++;
                if (pc[i] < 0.3f && pg[i] >= 0.3f) up++;
                if (pc[i] >= 0.3f && pg[i] < 0.3f) dn++;
            }
            bool same = boxStr[0] == boxStr[1];
            if (same) sameBoxes++;
            Console.WriteLine($"{Path.GetFileName(f)} {W}x{H}: maxAbs={md:F4} meanAbs={sum / pc.Length:E2} gpuNonZero={nz}/{pg.Length} flips(0.3) up={up} down={dn} boxes cpu={nbox[0]} gpu={nbox[1]} {(same ? "same" : "DIFF")} gpuAlive={((sGpu as GpuSession)?.GpuAlive.ToString() ?? "n/a")}");
        }
        Console.WriteLine($"detcmp {model}: identical box lists {sameBoxes}/{files.Length}");
        return 0;
    }

    // --reccmp <model> <batch> <W>: random input, activations before the CTC projection
    static int Rec(string[] args)
    {
        string model = args[1];
        int nb = int.Parse(args[2]), W = int.Parse(args[3]);
        var mdl = Model.Load(File.ReadAllBytes(Paths.ModelPath(model, "rec")));
        var compiled = new CompiledModel(mdl, intraOpThreads: 4);
        int[] shape = [nb, 3, 48, W];
        var inp = new float[nb * 3 * 48 * W];
        var rr = new Random(1234);
        for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);
        var cpu = new InferenceSession(compiled);
        cpu.Reshape(shape);
        if (!cpu.TryRunUntilCtcProjection(inp, out CtcProjectionOperands ops))
        { Console.WriteLine("cpu: no CTC projection tail"); return 2; }
        float[] a = ops.Activations.ToArray();
        int mm = ops.MatMulNodeIndex;
        using var dev = VkDevice.Create();
        using var gg = new GpuDetGraph(dev, compiled);
        float[] g = gg.Run(shape, inp, mm, checked((int)mdl.Nodes[mm].Inputs[0])).ToArray();
        double md = 0; int nz = 0;
        for (int i = 0; i < a.Length; i++) { md = Math.Max(md, Math.Abs(a[i] - g[i])); if (g[i] != 0) nz++; }
        int rows = ops.RowCount;
        int[] ic = new int[rows], ig = new int[rows];
        float[] sc = new float[rows], sg = new float[rows];
        Kernels.MatMul.TryArgMax(a, ops.Weights, ops.Bias, ic, sc, ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights, 4);
        Kernels.MatMul.TryArgMax(g, ops.Weights, ops.Bias, ig, sg, ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights, 4);
        int bad = 0, badText = 0;
        for (int r = 0; r < rows; r++)
            if (ic[r] != ig[r]) { bad++; if (ic[r] != 0) badText++; }
        Console.WriteLine($"reccmp {model} {nb}x{W}: act n={a.Length} maxAbs={md:F4} gpuNonZero={nz}/{g.Length} argmaxDiff={bad}/{rows} (non-blank {badText})");
        return 0;
    }
}
