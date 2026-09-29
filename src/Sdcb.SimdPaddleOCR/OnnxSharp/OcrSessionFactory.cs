namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Backend-neutral session factory. On net10.0+ it can hand out a GPU-backed
/// session — Metal on macOS (unless overridden), Vulkan elsewhere; on
/// netstandard2.0 (or when no GPU device is usable) it always yields the CPU
/// interpreter.
/// </summary>
internal static class OcrSessionFactory
{
    internal static IOcrSession Create(CompiledModel compiled, OcrBackend backend)
    {
#if NET10_0_OR_GREATER
        if (Backends.Metal.MetalBackend.IsMetalSelected(backend))
            return Backends.Metal.MetalBackend.CreateSession(compiled, backend);
        return Backends.Vulkan.GpuBackend.CreateSession(compiled, backend);
#else
        _ = backend;
        return compiled.CreateRequest();
#endif
    }

    /// <summary>True when the option resolves to a GPU session on the probed device.</summary>
    internal static bool IsGpuBackend(OcrBackend backend)
    {
#if NET10_0_OR_GREATER
        if (Backends.Metal.MetalBackend.IsMetalSelected(backend))
            return Backends.Metal.MetalBackend.UsesGpu(backend);
        return Backends.Vulkan.GpuBackend.UsesGpu(backend);
#else
        _ = backend;
        return false;
#endif
    }
}
