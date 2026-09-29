using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>
/// Process-wide Vulkan device for OCR graphs plus the backend-resolution
/// rules (option → SIMD_OCR_BACKEND env → probe). Sessions serialize their
/// GPU submissions on <see cref="VkDevice.Sync"/>.
/// </summary>
internal static class GpuBackend
{
    private static readonly object s_probeLock = new();
    private static VkDevice? s_device;
    private static bool s_probed;

    /// <summary>The shared device, or null when no usable Vulkan GPU exists.</summary>
    internal static VkDevice? TryGetDevice()
    {
        if (s_probed) return s_device;
        lock (s_probeLock)
        {
            if (s_probed) return s_device;
            try { s_device = VkDevice.Create(); }
            catch { s_device = null; }
            s_probed = true;
            return s_device;
        }
    }

    /// <summary>Whether the given option resolves to the Vulkan path.</summary>
    internal static bool IsVulkanSelected(OcrBackend backend) => backend switch
    {
        OcrBackend.Vulkan => true,
        OcrBackend.Cpu => false,
        _ => Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND") switch
        {
            { } s when s.Equals("cpu", StringComparison.OrdinalIgnoreCase) => false,
            { } s when s.Equals("vulkan", StringComparison.OrdinalIgnoreCase) => true,
            _ => true, // Auto: prefer GPU when the probe succeeds
        },
    };

    /// <summary>Vulkan asked for by name: the option, or Auto with
    /// SIMD_OCR_BACKEND=vulkan.</summary>
    private static bool IsVulkanExplicit(OcrBackend backend) =>
        backend == OcrBackend.Vulkan
        || backend == OcrBackend.Auto
            && string.Equals(Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND"),
                "vulkan", StringComparison.OrdinalIgnoreCase);

    /// <summary>Whether a session for this option runs on the GPU. Without an
    /// fp16 cooperative matrix of the cm shaders' shape (<see cref="VkDevice.CoopGemm"/>)
    /// Auto takes the no-coopmat GEMM tier only where compute subgroups cannot
    /// go below 64 lanes: there (Adreno 750) it beats the CPU on every model
    /// measured (docs/vulkan-8gen3.md), while on UHD 770, which pins 8 lanes,
    /// it trails the CPU (docs/vulkan-uhd770.md). Other no-coopmat devices
    /// only get the GPU for an explicit Vulkan choice.</summary>
    internal static bool UsesGpu(OcrBackend backend) =>
        IsVulkanSelected(backend) && TryGetDevice() is { } dev
        && (dev.CoopGemm || dev.SubgroupMin >= 64 || IsVulkanExplicit(backend));

    /// <summary>Creates a session on the resolved backend; CPU on any GPU failure.</summary>
    internal static IOcrSession CreateSession(CompiledModel compiled, OcrBackend backend)
    {
        if (UsesGpu(backend) && TryGetDevice() is { } dev)
        {
            try { return new GpuSession(dev, compiled); }
            catch { /* fall through to CPU */ }
        }
        return compiled.CreateRequest();
    }
}
