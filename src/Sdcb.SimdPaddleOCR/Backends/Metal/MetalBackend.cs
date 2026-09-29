using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Metal;

/// <summary>
/// Process-wide Metal device for OCR graphs plus the backend-resolution rules
/// (option → SIMD_OCR_BACKEND env → probe). On Apple platforms Metal wins the
/// Auto slot ahead of Vulkan; on other platforms Metal is never selected.
/// </summary>
internal static class MetalBackend
{
    private static readonly object s_probeLock = new();
    private static MtlDevice? s_device;
    private static bool s_probed;
    private static int s_announced;

    /// <summary>The shared device, or null when Metal is unavailable (not
    /// macOS, no GPU, or the ObjC bridge can't be reached).</summary>
    internal static MtlDevice? TryGetDevice()
    {
        if (s_probed) return s_device;
        lock (s_probeLock)
        {
            if (s_probed) return s_device;
            try { s_device = OperatingSystem.IsMacOS() ? MtlDevice.Probe() : null; }
            catch { s_device = null; }
            s_probed = true;
            return s_device;
        }
    }

    /// <summary>Whether the given option resolves to the Metal path.</summary>
    internal static bool IsMetalSelected(OcrBackend backend) => backend switch
    {
        OcrBackend.Metal => true,
        OcrBackend.Cpu or OcrBackend.Vulkan => false,
        _ => Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND") switch
        {
            { } s when s.Equals("metal", StringComparison.OrdinalIgnoreCase) => true,
            { } s when s.Equals("cpu", StringComparison.OrdinalIgnoreCase)
                    || s.Equals("vulkan", StringComparison.OrdinalIgnoreCase) => false,
            // Auto: Apple Silicon only — the GEMM kernels are built around
            // 32-lane simdgroups and were never validated on Intel/AMD GPUs.
            _ => OperatingSystem.IsMacOS()
                && RuntimeInformation.ProcessArchitecture == Architecture.Arm64,
        },
    };

    /// <summary>Metal asked for by name: the option, or Auto with
    /// SIMD_OCR_BACKEND=metal.</summary>
    private static bool IsMetalExplicit(OcrBackend backend) =>
        backend == OcrBackend.Metal
        || backend == OcrBackend.Auto
            && string.Equals(Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND"),
                "metal", StringComparison.OrdinalIgnoreCase);

    /// <summary>Whether a session for this option runs on Metal. Without an
    /// Apple GPU family only an explicit choice gets Metal: the simdgroup
    /// path is unproven off Apple Silicon, so Auto stays on CPU there
    /// (mirrors <see cref="Vulkan.GpuBackend.UsesGpu"/>).</summary>
    internal static bool UsesGpu(OcrBackend backend) =>
        IsMetalSelected(backend) && TryGetDevice() is { } dev
        && (dev.SupportsApple7 || IsMetalExplicit(backend));

    /// <summary>Creates a session on Metal; CPU on any failure.</summary>
    internal static IOcrSession CreateSession(CompiledModel compiled, OcrBackend backend)
    {
        if (UsesGpu(backend) && TryGetDevice() is { } dev)
        {
            try
            {
                // one stderr breadcrumb per process so CI/users can tell the
                // Metal path actually engaged (a failed probe is silent).
                if (System.Threading.Interlocked.Exchange(ref s_announced, 1) == 0)
                    Console.Error.WriteLine($"[metal] device: {dev.Name}");
                return new MetalSession(dev, compiled);
            }
            catch { /* fall through to CPU */ }
        }
        return compiled.CreateRequest();
    }
}
