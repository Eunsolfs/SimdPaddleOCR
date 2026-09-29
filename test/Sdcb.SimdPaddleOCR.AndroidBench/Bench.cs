namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Mode dispatch. Relative paths resolve against the app's external files dir.</summary>
static class Bench
{
    internal static string OutDir = "";

    internal static int Run(string[] args, string root, string outDir)
    {
        Paths.Root = root;
        OutDir = outDir;
        if (args.Length == 0) return Usage();
        return args[0] switch
        {
            "--init" => 0,
            "--caps" => Caps.Run(args),
            "--e2e" => E2e.Run(args),
            "--detprof" or "--recprof" => Prof.Run(args),
            "--detcmp" or "--reccmp" => Compare.Run(args),
            "--conc" => Prof.Conc(args),
            "--rawbench" => Caps.RawBench(args),
            "--gemm" => GemmTest.Run(args),
            "--gemmsweep" => GemmTest.Sweep(args),
            "--peak" => GemmTest.Peak(args),
            // --layers <model> <det|rec> <N> <H> <W> [first] [last]
            "--layers" => LayerProbe.Run(Paths.ModelPath(args[1], args[2]), int.Parse(args[3]), int.Parse(args[4]),
                int.Parse(args[5]), args.Length > 6 ? int.Parse(args[6]) : 0,
                args.Length > 7 ? int.Parse(args[7]) : int.MaxValue),
            _ => Usage(),
        };
    }

    static int Usage()
    {
        Console.WriteLine("""
            --caps
            --e2e --model tiny|small|medium --backend cpu|vulkan|auto [--workers 4] [--count N] [--warmup N] [--out name.json]
            --detprof <model> <H> <W> [reps]          SIMD_OCR_GPU_PROF=1 for per-kernel times
            --recprof <model> <batch> <W> [reps]
            --detcmp <model> <img.jpg|all|N> [maxSide]   CPU vs Vulkan prob map + boxes on the same input
            --reccmp <model> <batch> <W>               CPU vs Vulkan REC activations + CTC argmax
            --conc <model> [threads] [backend]
            --gemm <shader|file.spv> <M> <N> <K> [flags] [tileM] [tileN] [reqSg] [reps]
            --layers <model> <det|rec> <N> <H> <W> [first] [last]
            --rawbench <spv> <bindings> <gx> <gy> <reps> <bufMB> <reqSg> [pc uints...]
            """);
        return 2;
    }
}
