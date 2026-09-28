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
            // Auto: Metal is the GPU of choice on macOS only
            _ => OperatingSystem.IsMacOS(),
        },
    };

    /// <summary>Creates a session on Metal; CPU on any failure.</summary>
    internal static IOcrSession CreateSession(CompiledModel compiled, OcrBackend backend)
    {
        if (IsMetalSelected(backend) && TryGetDevice() is { } dev)
        {
            try { return new MetalSession(dev, compiled); }
            catch { /* fall through to CPU */ }
        }
        return compiled.CreateRequest();
    }
}
