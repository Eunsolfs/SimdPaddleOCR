using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>
/// Vulkan-backed <see cref="IOcrSession"/>: the whole ONNX graph runs via
/// <see cref="GpuDetGraph"/> under the shared <see cref="GpuSessionBase"/>
/// contract (staging input, CPU CTC tail, permanent CPU fallback on any
/// plan/run failure). One VkDevice is shared process-wide; each session owns
/// its buffers and command buffer, so sessions only serialize on the queue
/// submit itself.
/// </summary>
internal sealed class GpuSession : GpuSessionBase
{
    internal GpuSession(VkDevice dev, CompiledModel compiled)
        : base(new GpuDetGraph(dev, compiled), compiled, "gpu")
    {
    }
}
