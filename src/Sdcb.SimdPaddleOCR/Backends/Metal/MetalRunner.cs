using System.Diagnostics;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Metal;

/// <summary>One storage-buffer binding: a fixed (shared, immutable) buffer or
/// a <see cref="MetalGraphModel"/> role sentinel resolved per session.</summary>
internal readonly record struct MetalBind(MtlBuffer Buf, ulong ByteOffset);

internal sealed class MetalRec
{
    public required MtlPipeline Pipe;
    public required MetalBind[] Binds;
    public required byte[] Pc;
    public uint Gx, Gy;
    public uint Tx = 256;   // threads per threadgroup (attn/layernorm: 64)
    public string Tag = "";
}

/// <summary>
/// Compiled dispatch list for one (N,H,W,nodeLimit,outTensor): managed data
/// only. Arena/IO requirements are sizes, not handles, so a session growing
/// its arena never invalidates anything.
/// </summary>
internal sealed class MetalSchedule
{
    public readonly record struct Key(int N, int H, int W, int NodeLimit, int OutTensor);

    public required MetalRec[] Recs;
    public int OutElems;
    public long InElems, ArenaElems;
    public long[] Off = [];
    public int[] Alias = [];
    public long[] Numel = [];
    public int[][] Shapes = [];
    public long Im2colOff;
    public long LastTick;
}

/// <summary>
/// Per-session Metal runtime over a shared <see cref="MetalGraphModel"/>:
/// owns the fp16 arena, fp32 input/output buffers and the partials scratch —
/// all Shared-mode (unified memory: the CPU writes/reads the same mapping,
/// no staging copies). Each run re-records one command buffer per wave from
/// the pure-managed schedule; single-unit runs go through a serial compute
/// encoder (implicit ordering, no barriers needed), while interleaved
/// multi-unit runs use a concurrent encoder with one buffer barrier per
/// level — the Vulkan runner's per-rec pipeline barrier equivalent.
/// </summary>
internal sealed unsafe class MetalDetGraph : IOcrGraphRunner
{
    private readonly MtlDevice _dev;
    private readonly MetalGraphModel _model;
    private MtlBuffer? _arena, _in, _out, _part;
    private long _arenaElems, _inElems, _outElems;
    private float* _inMap, _outMap;
    private MetalSchedule? _last;
    private bool _disposed;

    private static readonly bool s_prof = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_PROF") == "1";
    private static readonly bool s_noBar = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_NOBAR") == "1";
    private static readonly bool s_dbgTime = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TIME") == "1";
    private static readonly bool s_onlyHead = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_ONLYHEAD") == "1";
    private static readonly int s_only =
        int.TryParse(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_ONLY"), out int o) ? o : -1;
    private static readonly int s_trunc =
        int.TryParse(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TRUNCATE"), out int t) ? t : -1;

    public MetalDetGraph(MtlDevice dev, CompiledModel compiled)
    {
        _dev = dev;
        _model = MetalGraphModel.Acquire(dev, compiled);
    }

    /// <summary>Compile (or fetch) the schedule for a shape without running it —
    /// lets callers probe GPU support before committing to the GPU path.</summary>
    public MetalSchedule Prepare(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
        => _model.GetSchedule(inputShape, nodeLimit, outTensor);

    /// <summary>Run the graph: input fp32 NCHW [n,C,H,W] → fp32 [graph output].
    /// nodeLimit truncates the emit loop (nodes beyond it are not dispatched) and
    /// outTensor overrides the readback tensor — used to stop the REC graph before
    /// the vocab projection (activations readback) instead of the graph output.
    /// </summary>
    public ReadOnlySpan<float> Run(int[] inputShape, ReadOnlySpan<float> input,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        MetalSchedule s = _model.GetSchedule(inputShape, nodeLimit, outTensor);
        return RunCore([s], input, out _).AsSpan(0, s.OutElems);
    }

    /// <summary>
    /// Runs several shapes of the same graph (inputs back to back in
    /// <paramref name="input"/>, unit i = numel(shapes[i]) floats) and hands
    /// the activations to <paramref name="onReady"/> in unit order as they
    /// land. The units go out in a few submissions: the GPU starts on the
    /// first while the rest are still being recorded, and the caller's CPU
    /// tail (CTC ArgMax) for early units overlaps the GPU work of later ones.
    /// Returns false when <paramref name="onReady"/> declined a batch; every
    /// submission has completed by the time this returns or throws.
    /// </summary>
    public bool RunMany(IReadOnlyList<int[]> shapes, ReadOnlySpan<float> input,
        int nodeLimit, int outTensor, CtcUnitsReady onReady)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var sched = new MetalSchedule[shapes.Count];
        for (int i = 0; i < sched.Length; i++)
            sched[i] = _model.GetSchedule(shapes[i], nodeLimit, outTensor);
        bool accepted = true;
        RunCore(sched, input, out _, (result, offsets, first, count) =>
            accepted = accepted && onReady(result, offsets, first, count));
        return accepted;
    }

    // in/out regions of consecutive units start on 256-byte boundaries
    private static long Align64(long floats) => (floats + 63) & ~63L;

    private float[] RunCore(MetalSchedule[] sched, ReadOnlySpan<float> input,
        out int[] outOffsets, CtcUnitsReady? sink = null)
    {
        long t0 = Stopwatch.GetTimestamp();
        int n = sched.Length;
        var inBase = new long[n];
        outOffsets = new int[n];
        var outBase = new long[n];
        long inTotal = 0, outTotal = 0, arena = 0;
        int outElems = 0;
        for (int i = 0; i < n; i++)
        {
            inBase[i] = inTotal;
            inTotal = Align64(inTotal + sched[i].InElems);
            outBase[i] = outTotal;
            outTotal = Align64(outTotal + sched[i].OutElems);
            outOffsets[i] = outElems;
            outElems += sched[i].OutElems;
            arena = Math.Max(arena, sched[i].ArenaElems);
        }
        if (n == 1) inTotal = Math.Max(inTotal, input.Length);
        // Level-interleaved multi-unit recording (one barrier per level) plus
        // wave cuts for streaming — same shape as the Vulkan runner.
        bool interleave = n > 1 && !s_prof && !s_onlyHead && s_only < 0 && s_trunc < 0 && !s_noBar;
        var arenaBase = new long[n];
        var waveStart = new List<int> { 0 };
        int maxWave = 1;
        if (interleave)
        {
            int[] cuts = sink is null ? [] : StreamCuts(sched);
            long a = 0, peak = 0;
            for (int i = 0, ci = 0; i < n; i++)
            {
                long need = (sched[i].ArenaElems + 127) & ~127L;
                bool cut = ci < cuts.Length && cuts[ci] == i;
                if (cut) ci++;
                if (i > 0 && (cut || a + need > InterleaveArenaBudget)) { waveStart.Add(i); a = 0; }
                arenaBase[i] = a;
                a += need;
                peak = Math.Max(peak, a);
                maxWave = Math.Max(maxWave, i - waveStart[^1] + 1);
            }
            arena = peak;
        }
        waveStart.Add(n);
        int waves = waveStart.Count - 1;
        EnsureBuffers(arena, inTotal, outTotal);
        long consumed = 0;
        fixed (float* src = input)
            for (int i = 0; i < n; i++)
            {
                long len = n == 1 ? input.Length : sched[i].InElems;
                Buffer.MemoryCopy(src + consumed, _inMap + inBase[i], len * 4, len * 4);
                consumed += len;
            }
        long t1 = Stopwatch.GetTimestamp();

        EnsurePart(interleave ? maxWave : 1);
        if (_result.Length < outElems) _result = new float[RoundUpPow2(outElems)];
        float[] result = _result;
        long t2 = 0, t3;
        var done = new List<IntPtr>(waves);
        try
        {
            for (int w = 0; w < waves; w++)
            {
                int s0 = waveStart[w], s1 = waveStart[w + 1];
                IntPtr cmd;
                using (var pool = AutoReleasePool.Create())
                {
                    // Interleave needs a concurrent encoder — under a serial
                    // encoder dispatches already run in order, so a per-rec
                    // barrier would be a no-op and the level overlap the whole
                    // point of interleaving is unreachable. Concurrent +
                    // one buffer barrier per level restores both.
                    MtlEncoder enc = _dev.BeginCommands(concurrent: interleave);
                    cmd = enc.Cmd;
                    // +1 on the command buffer so it survives the pool drain
                    // while we wait below.
                    ObjC.Retain(cmd);
                    MtlPipeline? bound = null;
                    if (interleave)
                    {
                        int levels = 0;
                        for (int i = s0; i < s1; i++) levels = Math.Max(levels, sched[i].Recs.Length);
                        for (int k = 0; k < levels; k++)
                        {
                            for (int i = s0; i < s1; i++)
                                if (k < sched[i].Recs.Length)
                                    RecordOne(enc, sched[i].Recs[k], ref bound,
                                        (ulong)inBase[i] * 4, (ulong)outBase[i] * 4,
                                        (ulong)arenaBase[i] * 2,
                                        (ulong)(i - s0) * MetalGraphModel.PartBytes);
                            if (!s_noBar) enc.BarrierBuffers();
                        }
                    }
                    else
                        for (int i = s0; i < s1; i++)
                            RecordRecs(enc, Selected(sched[i]), ref bound,
                                (ulong)inBase[i] * 4, (ulong)outBase[i] * 4);
                    enc.End();
                    enc.Commit();
                }
                done.Add(cmd);
            }
            t2 = Stopwatch.GetTimestamp();
            bool more = true;
            for (int w = 0; w < waves; w++)
            {
                IntPtr cmd = done[w];
                ObjC.Send0(cmd, ObjC.S.WaitUntilCompleted);
                long status = (long)ObjC.SendLong(cmd, ObjC.S.Status);
                if (status != 4)
                {
                    IntPtr err = ObjC.Send0(cmd, ObjC.S.Error);
                    throw new InvalidOperationException(
                        $"Metal command buffer failed (status={status}): {MtlDevice.DescribeError(err)}");
                }
                ObjC.Release(cmd);
                done[w] = IntPtr.Zero;
                int s0 = waveStart[w], s1 = waveStart[w + 1];
                for (int i = s0; i < s1; i++)
                    new ReadOnlySpan<float>(_outMap + outBase[i], sched[i].OutElems)
                        .CopyTo(result.AsSpan(outOffsets[i]));
                long tsk = s_dbgTime ? Stopwatch.GetTimestamp() : 0;
                if (sink is not null && more) more = sink(result, outOffsets, s0, s1 - s0);
                if (s_dbgTime) _dbgSink += Stopwatch.GetTimestamp() - tsk;
            }
        }
        finally
        {
            // never leave a submission in flight over buffers the next run reuses
            foreach (IntPtr cmd in done)
                if (cmd != IntPtr.Zero)
                {
                    ObjC.Send0(cmd, ObjC.S.WaitUntilCompleted);
                    ObjC.Release(cmd);
                }
        }
        t3 = Stopwatch.GetTimestamp();
        _last = sched[n - 1];
        if (s_dbgTime)
        {
            double f = Stopwatch.Frequency / 1e3;
            Console.WriteLine($"[t] units={n}{(interleave ? " il" : "")} waves={waves} write={(t1 - t0) / f:F2} record+submit={(t2 - t1) / f:F2} " +
                $"wait+read+sink={(t3 - t2) / f:F2} sink={_dbgSink / f:F2} ms recs={sched.Sum(x => x.Recs.Length)}");
            _dbgSink = 0;
        }
        return result;
    }

    // Unit indices where a streamed run opens a new submission: ~55% of the
    // input volume in the first wave (GPU starts while the rest records; the
    // caller's tail for it overlaps the remainder), a third wave for large
    // images so the last, un-overlapped tail stays short.
    private static int[] StreamCuts(MetalSchedule[] sched)
    {
        int n = sched.Length;
        if (n < 4) return [];
        double[] fractions = n >= 9 ? [0.45, 0.78] : [0.55];
        long total = 0;
        foreach (MetalSchedule s in sched) total += s.InElems;
        var cuts = new List<int>();
        long acc = 0;
        int f = 0;
        for (int i = 0; i < n - 1 && f < fractions.Length; i++)
        {
            acc += sched[i].InElems;
            if (acc >= fractions[f] * total) { cuts.Add(i + 1); f++; }
        }
        return [.. cuts];
    }

    // fp16 elements of concurrently live unit arenas in one interleaved wave (512 MB)
    private const long InterleaveArenaBudget = 256L << 20;

    private float[] _result = [];
    private long _dbgSink;   // SIMD_OCR_GPU_TIME: CPU time spent in the streamed sink

    private static long RoundUpPow2(long v)
    {
        long p = 1024;
        while (p < v) p <<= 1;
        return p;
    }

    private int _partUnits;

    private void EnsurePart(int units)
    {
        if (_part is not null && units <= _partUnits) return;
        int alloc = Math.Max(units, _partUnits * 3 / 2);
        _part?.Dispose();
        _part = null;
        // se partial-sum scratch; contents are fully rewritten per run.
        _part = _dev.NewBuffer((nuint)alloc * MetalGraphModel.PartBytes);
        _partUnits = alloc;
    }

    private void EnsureBuffers(long arenaElems, long needIn, long needOut)
    {
        if (_arena is null || arenaElems > _arenaElems)
        {
            long alloc = Math.Max(arenaElems, _arenaElems * 3 / 2);
            _arena?.Dispose();
            _arena = null;
            _arena = _dev.NewBuffer((nuint)alloc * 2);
            _arenaElems = alloc;
        }
        // grow in power-of-two steps so a stream of varying DET sizes settles
        if (_in is null || needIn > _inElems)
        {
            long alloc = RoundUpPow2(needIn);
            _in?.Dispose();
            _in = _dev.NewBuffer((nuint)alloc * 4);
            _inMap = (float*)_in.Contents;
            _inElems = alloc;
        }
        if (_out is null || needOut > _outElems)
        {
            long alloc = RoundUpPow2(needOut);
            _out?.Dispose();
            _out = _dev.NewBuffer((nuint)alloc * 4);
            _outMap = (float*)_out.Contents;
            _outElems = alloc;
        }
    }

    private MtlBuffer Resolve(MtlBuffer b) =>
        ReferenceEquals(b, MetalGraphModel.RoleArena) ? _arena!
        : ReferenceEquals(b, MetalGraphModel.RoleIn) ? _in!
        : ReferenceEquals(b, MetalGraphModel.RoleOut) ? _out!
        : ReferenceEquals(b, MetalGraphModel.RolePart) ? _part!
        : b;

    private IReadOnlyList<MetalRec> Selected(MetalSchedule s)
    {
        MetalRec[] r = s.Recs;
        if (s_onlyHead) return [r[0]];
        if (s_only >= 0 && s_only < r.Length) return Enumerable.Repeat(r[s_only], 20).ToArray();
        if (s_trunc >= 0 && s_trunc < r.Length) return r.Take(s_trunc).ToArray();
        return r;
    }

    private void RecordRecs(MtlEncoder enc, IReadOnlyList<MetalRec> recs,
        ref MtlPipeline? bound, ulong inBase, ulong outBase)
    {
        // serial encoder: dispatch order is the dependency order.
        foreach (MetalRec r in recs)
            RecordOne(enc, r, ref bound, inBase, outBase, 0, 0);
    }

    private void RecordOne(MtlEncoder enc, MetalRec r, ref MtlPipeline? bound,
        ulong inBase, ulong outBase, ulong arenaBase, ulong partBase)
    {
        if (!ReferenceEquals(bound, r.Pipe))
        {
            enc.SetPipeline(r.Pipe);
            bound = r.Pipe;
        }
        for (int b = 0; b < r.Binds.Length; b++)
        {
            MtlBuffer role = r.Binds[b].Buf;
            MtlBuffer buf = Resolve(role);
            ulong off = r.Binds[b].ByteOffset
                + (ReferenceEquals(role, MetalGraphModel.RoleIn) ? inBase
                   : ReferenceEquals(role, MetalGraphModel.RoleOut) ? outBase
                   : ReferenceEquals(role, MetalGraphModel.RoleArena) ? arenaBase
                   : ReferenceEquals(role, MetalGraphModel.RolePart) ? partBase : 0);
            enc.SetBuffer(buf, (nuint)off, b);
        }
        fixed (byte* pp = r.Pc)
            enc.SetPc(pp, (nuint)r.Pc.Length, r.Binds.Length);
        enc.Dispatch(r.Gx, r.Gy, 1, r.Tx, 1, 1);
    }

    // ---------------- debug readers ----------------

    private static int PhysOf(MetalSchedule s, int tensorIndex)
    {
        int t = tensorIndex;
        while (s.Alias[t] != t) t = s.Alias[t];
        return t;
    }

    /// <summary>SIMD_OCR_GPU_DUMP: read a tensor's fp16 arena content as fp32.</summary>
    public float[]? ReadTensor(int tensorIndex)
    {
        MetalSchedule? s = _last;
        if (s is null || _arena is null) return null;
        int t = PhysOf(s, tensorIndex);
        if (t < 0 || t >= s.Numel.Length) return null;
        long n = s.Numel[tensorIndex];
        long e = s.Off[t];
        if (e < 0) return new float[n];
        var v = new float[n];
        Half* p = (Half*)_arena.Contents + e;
        for (long i = 0; i < n; i++) v[i] = (float)p[i];
        return v;
    }

    private MetalSchedule Sched(int[] inputShape, int nodeLimit, int outTensor)
        => _model.GetSchedule(inputShape, nodeLimit, outTensor);

    /// <summary>Debug: raw values of a tensor's arena slot (NHWC fp16 → fp32),
    /// valid after a Run of the matching schedule.</summary>
    public float[] DebugValues(int tensorIndex, int[] inputShape, int count = 16,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        MetalSchedule s = Sched(inputShape, nodeLimit, outTensor);
        int t = PhysOf(s, tensorIndex);
        int n = Math.Min(count, (int)s.Numel[tensorIndex]);
        if (s.Off[t] < 0 || _arena is null) return new float[n];
        var v = new float[n];
        Half* p = (Half*)_arena.Contents + s.Off[t];
        for (int i = 0; i < n; i++) v[i] = (float)p[i];
        return v;
    }

    public float[] DebugValuesRaw(int[] inputShape, long elemOff, int count,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        Sched(inputShape, nodeLimit, outTensor);
        var v = new float[count];
        if (_arena is not null)
        {
            Half* p = (Half*)_arena.Contents + elemOff;
            for (int i = 0; i < count; i++) v[i] = (float)p[i];
        }
        return v;
    }

    /// <summary>Debug: min/max/mean of a tensor's arena slot after a Run.</summary>
    public (float Min, float Max, double Mean) DebugStats(int tensorIndex, int[] inputShape,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        MetalSchedule s = Sched(inputShape, nodeLimit, outTensor);
        int t = PhysOf(s, tensorIndex);
        int n = (int)s.Numel[tensorIndex];
        var v = new float[n];
        if (s.Off[t] >= 0 && _arena is not null)
        {
            Half* p = (Half*)_arena.Contents + s.Off[t];
            for (int i = 0; i < n; i++) v[i] = (float)p[i];
        }
        float mn = float.MaxValue, mx = float.MinValue; double sum = 0;
        foreach (float x in v) { mn = MathF.Min(mn, x); mx = MathF.Max(mx, x); sum += x; }
        return (mn, mx, sum / n);
    }

    internal float[]? DbgConstKm(int tensorIndex, int n) => _model.DbgConstKm(tensorIndex, n);
    internal int PhysProbe(int tensorIndex)
    {
        var s = _last; if (s is null) return -2;
        return PhysOf(s, tensorIndex);
    }
    internal long SlotProbe(int tensorIndex)
    {
        var s = _last; if (s is null) return -2;
        int t = PhysOf(s, tensorIndex);
        return t < 0 ? -3 : s.Off[t];
    }
    internal unsafe float[]? DbgPart(long elemOff, int n)   // fp32 partials/scale region
    {
        if (_part is null) return null;
        var sp = new ReadOnlySpan<float>((float*)_part.Contents + elemOff, n);
        return sp.ToArray();
    }
    internal float[]? DbgIn(int n)
    {
        if (_inMap == null) return null;
        return new ReadOnlySpan<float>(_inMap, n).ToArray();
    }

    public long Im2colOffset(int[] inputShape)
        => Sched(inputShape, int.MaxValue, -1).Im2colOff;

    public int DispatchCount(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
        => Sched(inputShape, nodeLimit, outTensor).Recs.Length;

    /// <summary>Per-dispatch wall timing: serializes+times each rec when
    /// SIMD_OCR_GPU_PROF=1; otherwise just lists the dispatch roster.</summary>
    public void DumpProfile(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        MetalSchedule s = Sched(inputShape, nodeLimit, outTensor);
        IReadOnlyList<MetalRec> recs = Selected(s);
        if (s_prof)
        {
            // Serialized per-rec commits — crude but correct wall clock.
            var agg = new Dictionary<string, (double ms, int n)>();
            float[] inp = new float[s.InElems];
            EnsureBuffers(s.ArenaElems, Align64(s.InElems), Align64(s.OutElems));
            EnsurePart(1);
            for (int i = 0; i < recs.Count; i++)
            {
                long t0 = Stopwatch.GetTimestamp();
                using (var pool = AutoReleasePool.Create())
                {
                    MtlEncoder enc = _dev.BeginCommands();
                    MtlPipeline? bound = null;
                    RecordOne(enc, recs[i], ref bound, 0, 0, 0, 0);
                    enc.End(); enc.CommitAndWait();
                }
                double ms = (Stopwatch.GetTimestamp() - t0) / (Stopwatch.Frequency / 1e3);
                string tag = recs[i].Pipe.Name;
                var a = agg.TryGetValue(tag, out var v) ? v : (ms: 0.0, n: 0);
                agg[tag] = (a.ms + ms, a.n + 1);
            }
            Console.WriteLine($"--- Metal profile ({recs.Count} dispatches, serialized) ---");
            foreach (var kv in agg.OrderByDescending(k => k.Value.ms))
                Console.WriteLine($"  {kv.Key,-16} {kv.Value.ms,9:F3} ms  x{kv.Value.n}");
            return;
        }
        Console.WriteLine($"--- Metal dispatches ({recs.Count}) ---");
        foreach (MetalRec r in recs)
            Console.WriteLine($"  {r.Tag,-40} {r.Pipe.Name} gx={r.Gx} gy={r.Gy} tx={r.Tx}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _arena?.Dispose(); _in?.Dispose(); _out?.Dispose(); _part?.Dispose();
        _arena = _in = _out = _part = null;
        _inMap = _outMap = null;
        _model.Release();
    }
}
