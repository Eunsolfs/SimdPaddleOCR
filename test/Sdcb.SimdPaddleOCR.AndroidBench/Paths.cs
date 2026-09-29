namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Model / dataset locations and image decoding shared by the phone
/// host and the desktop GpuBench (which links this file).</summary>
static class Paths
{
    internal static string Root = "";
    internal static string ModelsDir = "models";
    internal static string DatasetDir = "dataset";

    internal static string Path_(string p) => Path.IsPathRooted(p) ? p : Path.Combine(Root, p);
    internal static string ModelPath(string model, string kind) =>
        kind == "cls" ? Path.Combine(Path_(ModelsDir), "cls.onnx") : Path.Combine(Path_(ModelsDir), $"{model}-{kind}.onnx");
    internal static string DictPath(string model) => Path.Combine(Path_(ModelsDir), $"{model}-dict.txt");
    internal static string Dataset => Path_(DatasetDir);

    internal static OcrBackend ParseBackend(string s) => s.ToLowerInvariant() switch
    {
        "cpu" or "sharp" => OcrBackend.Cpu,
        "vulkan" => OcrBackend.Vulkan,
        "auto" => OcrBackend.Auto,
        _ => throw new ArgumentException("backend must be cpu, vulkan or auto"),
    };

    /// <summary>Decodes like the desktop Tests harness (ImageSharp, BGR24).</summary>
    internal static (byte[] Bgr, int W, int H) LoadBgr(string path)
    {
        using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(path);
        byte[] px = new byte[checked(image.Width * image.Height * 3)];
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                int o = (y * image.Width + x) * 3;
                px[o] = p.B; px[o + 1] = p.G; px[o + 2] = p.R;
            }
        return (px, image.Width, image.Height);
    }

    internal static string[] DatasetFiles() => Directory.GetFiles(Dataset, "img-*.jpg")
        .OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();
}
