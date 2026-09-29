using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Device capability probe (M0) and single-dispatch timing.</summary>
static unsafe class Caps
{
    static string CompType(uint t) => t switch
    {
        0 => "f16", 1 => "f32", 2 => "f64", 3 => "s8", 4 => "s16", 5 => "s32", 6 => "s64",
        7 => "u8", 8 => "u16", 9 => "u32", 10 => "u64", 1000141000 => "bf16",
        _ => t.ToString(),
    };

    static string Scope(uint s) => s switch { 1 => "device", 2 => "workgroup", 3 => "subgroup", 5 => "queuefamily", _ => s.ToString() };

    internal static int Run(string[] args)
    {
        Console.WriteLine($"phone: {Android.OS.Build.Manufacturer} {Android.OS.Build.Model} ({Android.OS.Build.Device}) " +
            $"Android {Android.OS.Build.VERSION.Release} API {(int)Android.OS.Build.VERSION.SdkInt} soc={(OperatingSystem.IsAndroidVersionAtLeast(31) ? Android.OS.Build.SocModel : "?")}");
        Console.WriteLine($"runtime: {RuntimeInformation.FrameworkDescription} rid={RuntimeInformation.RuntimeIdentifier} " +
            $"cpus={Environment.ProcessorCount} advsimd={System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported} " +
            $"dotprod={System.Runtime.Intrinsics.Arm.Dp.IsSupported} vector={System.Numerics.Vector<float>.Count}x{System.Numerics.Vector.IsHardwareAccelerated} " +
            $"lib={typeof(PaddleOcrAll).Assembly.GetCustomAttributes(typeof(System.Runtime.Versioning.TargetFrameworkAttribute), false).OfType<System.Runtime.Versioning.TargetFrameworkAttribute>().FirstOrDefault()?.FrameworkName}");

        using var dev = VkDevice.Create();
        Console.WriteLine($"vulkan loader: {Vk.LoadedLibrary}");
        Vk.VkPhysicalDeviceProperties props;
        Vk.vkGetPhysicalDeviceProperties(dev.PhysDevice, &props);
        uint api = props.ApiVersion;
        Console.WriteLine($"device: {dev.DeviceName} vendor=0x{props.VendorID:x} deviceId=0x{props.DeviceID:x} type={props.DeviceType} " +
            $"api={api >> 22 & 0x7f}.{api >> 12 & 0x3ff}.{api & 0xfff} driver=0x{props.DriverVersion:x8}");

        byte* lim = props.LimitsAndSparse + 4;   // native VkPhysicalDeviceLimits is 8-aligned
        Console.WriteLine($"limits: maxComputeSharedMemorySize={*(uint*)(lim + 216)} maxWgInvocations={*(uint*)(lim + 232)} " +
            $"maxWgSize={*(uint*)(lim + 236)}x{*(uint*)(lim + 240)}x{*(uint*)(lim + 244)} " +
            $"maxWgCount={*(uint*)(lim + 220)}x{*(uint*)(lim + 224)}x{*(uint*)(lim + 228)}");
        Console.WriteLine($"limits: maxStorageBufferRange={*(uint*)(lim + 28)} minStorageBufferOffsetAlignment={*(ulong*)(lim + 328)} " +
            $"maxPushConstantsSize={*(uint*)(lim + 32)} maxMemoryAllocationCount={*(uint*)(lim + 36)} timestampPeriod={*(float*)(lim + 424)}ns");

        uint nqf = 0;
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(dev.PhysDevice, &nqf, null);
        var qf = stackalloc Vk.VkQueueFamilyProperties[(int)nqf];
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(dev.PhysDevice, &nqf, qf);
        for (uint i = 0; i < nqf; i++)
            Console.WriteLine($"queue family {i}: flags=0x{qf[i].QueueFlags:x} count={qf[i].QueueCount} timestampValidBits={qf[i].TimestampValidBits}{(i == dev.QueueFamily ? "  <- selected" : "")}");
        Console.WriteLine($"queue globalPriority={dev.QueuePriority}");

        Console.WriteLine($"subgroup: default={dev.SubgroupSize} stages=0x{dev.SubgroupStages:x} ops=0x{dev.SubgroupOps:x} " +
            $"sizeControl={dev.SubgroupSizeControl} sgRange={dev.SubgroupMin}-{dev.SubgroupMax} requiredStagesCompute={dev.ComputeSubgroupSize}");
        Console.WriteLine($"features: storage16={dev.Storage16Bit} shaderFloat16={dev.ShaderFloat16} coopMatrix={dev.CoopMatrix} " +
            $"push={dev.PushDescriptors} coop16x16x16={dev.Coop16x16x16} coop8x16x16={dev.Coop8x16x16} best={dev.CoopM}x{dev.CoopN}x{dev.CoopK}");
        Console.WriteLine($"routing: Sg32Subgroup={dev.Sg32Subgroup} CoopGemm={dev.CoopGemm}");

        for (int i = 0; i < (int)dev.MemProps.MemoryHeapCount; i++)
            Console.WriteLine($"heap {i}: {dev.MemProps.HeapAt(i).Size >> 20} MB flags=0x{dev.MemProps.HeapAt(i).Flags:x}");
        for (int i = 0; i < (int)dev.MemProps.MemoryTypeCount; i++)
            Console.WriteLine($"memtype {i}: flags=0x{dev.MemProps.TypeAt(i).PropertyFlags:x} heap={dev.MemProps.TypeAt(i).HeapIndex}");

        uint next = 0;
        Vk.vkEnumerateDeviceExtensionProperties(dev.PhysDevice, null, &next, null);
        var exts = new Vk.VkExtensionProperties[next];
        fixed (Vk.VkExtensionProperties* pe = exts)
            Vk.vkEnumerateDeviceExtensionProperties(dev.PhysDevice, null, &next, pe);
        var names = new List<string>();
        for (int i = 0; i < next; i++)
            fixed (byte* n = exts[i].ExtensionName) names.Add(new string((sbyte*)n));
        Console.WriteLine($"extensions ({names.Count}): {string.Join(' ', names.OrderBy(x => x))}");

        CoopList(dev);
        SubgroupProbe(dev);
        TimestampProbe(dev);
        return 0;
    }

    static void CoopList(VkDevice dev)
    {
        byte[] fn = System.Text.Encoding.ASCII.GetBytes("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0");
        IntPtr p;
        fixed (byte* f = fn) p = Vk.vkGetInstanceProcAddr(dev.Instance, f);
        if (p == IntPtr.Zero) { Console.WriteLine("coopmat: vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not exported"); return; }
        var q = (delegate* unmanaged[Cdecl]<IntPtr, uint*, Vk.VkCooperativeMatrixPropertiesKHR*, VkResult>)p;
        uint np = 0;
        q(dev.PhysDevice, &np, null);
        var cm = new Vk.VkCooperativeMatrixPropertiesKHR[np];
        for (int i = 0; i < np; i++) cm[i].SType = VkConst.StCooperativeMatrixPropertiesKHR;
        fixed (Vk.VkCooperativeMatrixPropertiesKHR* pc = cm) q(dev.PhysDevice, &np, pc);
        Console.WriteLine($"coopmat: {np} configurations");
        foreach (var c in cm)
            Console.WriteLine($"  coopmat {c.MSize}x{c.NSize}x{c.KSize} A={CompType(c.AType)} B={CompType(c.BType)} C={CompType(c.CType)} R={CompType(c.ResultType)} sat={c.SaturatingAccumulation} scope={Scope(c.Scope)}");
    }

    static byte[] Resource(string name)
    {
        using Stream s = typeof(Caps).Assembly.GetManifestResourceStream(name)!;
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }

    /// <summary>gl_SubgroupSize as seen by a pipeline built through the same
    /// VkDevice.NewPipeline path the graph uses, per requiredSubgroupSize.</summary>
    static void SubgroupProbe(VkDevice dev)
    {
        byte[] spv = Resource("sgsize.spv");
        foreach (uint req in new uint[] { 0, 16, 32, 64, 128 })
        {
            if (req != 0 && !(dev.SubgroupSizeControl && dev.ComputeSubgroupSize && req >= dev.SubgroupMin && req <= dev.SubgroupMax))
            {
                Console.WriteLine($"requiredSubgroupSize={req}: not requested (outside sgRange {dev.SubgroupMin}-{dev.SubgroupMax} or no compute size control; invalid usage)");
                continue;
            }
            try
            {
                var sp = dev.NewPipeline(dev.NewShaderModule(spv), 1, 4, req);
                var set = dev.NewDescriptorSet(sp.SetLayout);
                const int groups = 64;
                var ob = dev.NewStorageBuffer(groups * 16, hostVisible: true, preferHost: true);
                dev.BindBuffer(set, 0, ob);
                IntPtr cmd = dev.NewCommandBuffer();
                IntPtr f = dev.NewFence();
                var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
                Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
                Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, sp.Pipeline);
                Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, sp.Layout, 0, 1, &set, 0, null);
                Vk.vkCmdDispatch(cmd, groups, 1, 1);
                Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
                dev.Submit(cmd, f); dev.WaitFence(f);
                uint* o = (uint*)ob.Map();
                var sizes = new HashSet<string>();
                for (int g = 0; g < groups; g++) sizes.Add($"gl_SubgroupSize={o[g * 4]} numSubgroups={o[g * 4 + 1]} lastSubgroupId={o[g * 4 + 2]} lastLane={o[g * 4 + 3]}");
                ob.Unmap();
                Console.WriteLine($"requiredSubgroupSize={req}: {string.Join(" | ", sizes)}");
                dev.DestroyFence(f);
                ob.Free();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"requiredSubgroupSize={req}: FAILED {ex.Message}");
            }
        }
    }

    static void TimestampProbe(VkDevice dev)
    {
        byte[] spv = Resource("sgsize.spv");
        var sp = dev.NewPipeline(dev.NewShaderModule(spv), 1, 4, 0);
        var set = dev.NewDescriptorSet(sp.SetLayout);
        var ob = dev.NewStorageBuffer(1 << 20, hostVisible: false);
        dev.BindBuffer(set, 0, ob);
        IntPtr cmd = dev.NewCommandBuffer();
        IntPtr f = dev.NewFence();
        var qci = new Vk.VkQueryPoolCreateInfo { SType = 11, QueryType = 2, QueryCount = 2 };
        Vk.Check(Vk.vkCreateQueryPool(dev.Device, &qci, null, out IntPtr qp), "qp");
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        Vk.vkCmdResetQueryPool(cmd, qp, 0, 2);
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, sp.Pipeline);
        Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, sp.Layout, 0, 1, &set, 0, null);
        Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, qp, 0);
        Vk.vkCmdDispatch(cmd, 16384, 1, 1);
        Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, qp, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        dev.Submit(cmd, f); dev.WaitFence(f);
        double wall = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        ulong* ts = stackalloc ulong[2];
        var r = Vk.vkGetQueryPoolResults(dev.Device, qp, 0, 2, 16, ts, 8, 1 | 2);
        Console.WriteLine($"timestamp probe: result={r} t0={ts[0]} t1={ts[1]} delta={(ts[1] - ts[0]) * dev.TimestampPeriodNs / 1e6:F4} ms (wall {wall:F3} ms, 16384 WGs x 256)");
        Vk.vkDestroyQueryPool(dev.Device, qp, null);
        dev.DestroyFence(f);
        ob.Free();
    }

    // --rawbench <spv> <bindings> <gx> <gy> <reps> <bufMB> <reqSg> [pc uints...]
    internal static int RawBench(string[] args)
    {
        using var dev = VkDevice.Create();
        byte[] spv = File.ReadAllBytes(Paths.Path_(args[1]));
        int nbind = int.Parse(args[2]);
        uint gx = uint.Parse(args[3]), gy = uint.Parse(args[4]);
        int reps = int.Parse(args[5]);
        ulong bytes = ulong.Parse(args[6]) << 20;
        uint reqSg = uint.Parse(args[7]);
        uint[] pcs = args.Skip(8).Select(uint.Parse).ToArray();
        int pcBytes = Math.Max(16, pcs.Length * 4);
        var rp = dev.NewPipeline(dev.NewShaderModule(spv), nbind, pcBytes, reqSg);
        var set = dev.NewDescriptorSet(rp.SetLayout);
        var bufs = new List<VkBuffer>();
        for (int b = 0; b < nbind; b++)
        {
            var buf = dev.NewStorageBuffer(bytes, hostVisible: false);
            dev.Zero(buf);
            dev.BindBuffer(set, (uint)b, buf);
            bufs.Add(buf);
        }
        IntPtr cmd = dev.NewCommandBuffer();
        IntPtr f = dev.NewFence();
        var qci = new Vk.VkQueryPoolCreateInfo { SType = 11, QueryType = 2, QueryCount = 2 };
        Vk.Check(Vk.vkCreateQueryPool(dev.Device, &qci, null, out IntPtr qp), "qp");
        uint[] pcv = new uint[pcBytes / 4];
        pcs.CopyTo(pcv, 0);
        double best = double.MaxValue;
        ulong* ts = stackalloc ulong[2];
        for (int r = 0; r < reps; r++)
        {
            var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
            Vk.Check(Vk.vkResetCommandBuffer(cmd, 0), "reset");
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
            Vk.vkCmdResetQueryPool(cmd, qp, 0, 2);
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, rp.Pipeline);
            Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, rp.Layout, 0, 1, &set, 0, null);
            fixed (uint* pp = pcv) Vk.vkCmdPushConstants(cmd, rp.Layout, VkConst.StageComputeShader, 0, (uint)pcBytes, pp);
            Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, qp, 0);
            Vk.vkCmdDispatch(cmd, gx, gy, 1);
            Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, qp, 1);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
            dev.Submit(cmd, f); dev.WaitFence(f);
            Vk.vkGetQueryPoolResults(dev.Device, qp, 0, 2, 16, ts, 8, 1 | 2);
            best = Math.Min(best, (ts[1] - ts[0]) * dev.TimestampPeriodNs / 1e6);
        }
        Console.WriteLine($"rawbench best={best:F4} ms");
        Vk.vkDestroyQueryPool(dev.Device, qp, null);
        foreach (var b in bufs) b.Free();
        return 0;
    }
}
