using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Metal;

/// <summary>
/// Metal-backed <see cref="IOcrSession"/>: the whole ONNX graph runs via
/// <see cref="MetalDetGraph"/> under the shared <see cref="GpuSessionBase"/>
/// contract (staging input, CPU CTC tail, permanent CPU fallback on any
/// plan/run failure). One MtlDevice is shared process-wide; each session owns
/// its buffers, so sessions only serialize at the command-buffer commits.
/// </summary>
internal sealed class MetalSession : GpuSessionBase
{
    internal MetalSession(MtlDevice dev, CompiledModel compiled)
        : base(new MetalDetGraph(dev, compiled), compiled, "metal")
    {
    }
}
