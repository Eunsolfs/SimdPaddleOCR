using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends;

/// <summary>
/// Per-session GPU executor behind <see cref="GpuSessionBase"/>: one runner
/// instance owns the session's device buffers and turns the shared,
/// model-level schedules into submissions. Implemented by the Vulkan
/// <c>GpuDetGraph</c> and the Metal <c>MetalDetGraph</c>.
/// </summary>
internal interface IOcrGraphRunner : IDisposable
{
    /// <summary>
    /// Run the graph: input fp32 NCHW [n,C,H,W] → fp32 [graph output].
    /// <paramref name="nodeLimit"/> truncates the emit loop (nodes beyond it
    /// are not dispatched) and <paramref name="outTensor"/> overrides the
    /// readback tensor — used to stop the REC graph before the vocab
    /// projection (activations readback) instead of the graph output.
    /// </summary>
    ReadOnlySpan<float> Run(int[] inputShape, ReadOnlySpan<float> input,
        int nodeLimit = int.MaxValue, int outTensor = -1);

    /// <summary>
    /// Runs several shapes of the same graph (inputs back to back in
    /// <paramref name="input"/>, unit i = numel(shapes[i]) floats) and hands
    /// the activations to <paramref name="onReady"/> in unit order as they
    /// land. Returns false when <paramref name="onReady"/> declined a batch;
    /// every submission has completed by the time this returns or throws.
    /// </summary>
    bool RunMany(IReadOnlyList<int[]> shapes, ReadOnlySpan<float> input,
        int nodeLimit, int outTensor, CtcUnitsReady onReady);
}
