# SimdPaddleOCR — Intel UHD Graphics 770（无协作矩阵）Vulkan

实测机：Intel UHD Graphics 770（Xe-LP，32 EU，核显最大动态频率 1.55 GHz），与 CPU 共享内存。驱动 101.7079（Vulkan API 1.4.323）。Windows，电源未单独锁频。.NET SDK 11.0.100-rc.1 编译 `net10.0`。`test/Sdcb.SimdPaddleOCR.Tests`，`--workers 4 --benchmark-kind simd --warmup 1`，同一 `dataset/` 100 张，墙钟 n=99。

**结论：没有 16×16×16 fp16 协作矩阵。无矩阵 GEMM 能跑通，和同机 CPU 逐图 hash / 框数 / 文本 100/100，但三档 median 都更慢（约 1.8× / 2.6× / 2.0×）。默认仍在建会话时退回 CPU，不编译协作矩阵 shader（Intel 编译器会把进程直接打掉）。sg16 / sg32 / sg32l 的 shader、spv 和路由没改。**

## 设备能力

| 项 | 值 |
|---|---|
| 设备 | Intel(R) UHD Graphics 770，`vendor=0x8086`，`INTEGRATED_GPU`。枚举里没有独显 |
| 队列 | family 0：`flags=0xf` count=1（图形+计算，被选中）；family 1：`flags=0x20` count=2（视频解码） |
| subgroup | 默认 32，`sgRange` 8–32，compute 可指定宽度。探针 `requiredSubgroupSize` 0 和 32 都是 `gl_SubgroupSize=32` |
| cooperative matrix | 无 `VK_KHR_cooperative_matrix` |
| fp16 | `shaderFloat16` 和 16-bit storage 都有。加速的整数点积只有 int8 / DP4A，16-bit 点积没有 |
| 共享内存 | `maxComputeSharedMemorySize` = **32 KB**。`maxComputeWorkGroupInvocations` = 1024 |
| 内存 | 一个 heap，65343 MB，`DEVICE_LOCAL`。三种类型全带 device-local：`0x1`、`0x7`（再加 host-visible + coherent）、`0xf`（再加 host-cached）。没有「非 device-local 的主机堆」 |
| push descriptor | 有。`globalPriority`=512（HIGH） |
| 峰值 | fp32 约 0.8 TFLOPS，fp16 约 1.6 TFLOPS（32 EU × 8 × 2 × 1.55 GHz；fp16 是普通 ALU 的打包，不是矩阵单元） |

## 走了哪一档

`SubgroupMin` 是 8，进不了 sg32。也没有 `Coop16x16x16`，所以不建任何 `conv1x1_cm*` / `convk_cm*` 管线。大 GEMM（1×1 里 K 大于 dot 门槛的、im2col 之后的 GEMM、窄 M 接不住的 MatMul）走 `gemm_nc`：fp16 加载、fp32 按 K 正序累加、fp16 写回，绑定和 16 字节 push constant 与 `conv1x1_cm` 相同，不用 subgroup 内置量。

保留的 tile：工作组 128 线程，输出块 32×64，每线程 4×4，K 方向 16，共享内存放 f16vec4。`PackSlabs` 只在这条档上额外打开，`_sg32` 的条件没动。kxk 隐式 GEMM 没有新开，仍是 `convk_dot` / im2col。

`GpuGraphModel.RouteNocm` 为 false。没有这种协作矩阵时，构造函数在创建任何管线之前抛 `NotSupportedException`，`GpuBackend.CreateSession` 接住后用 CPU。指定 `OcrBackend.Vulkan` 在这台机器上因此仍是 CPU，和改之前的路由意图一样，只是不再把驱动打崩。

## 端到端（4 workers，median ms/图）

同一时段先 Vulkan 再 sharp，各 100 张。medium 的 CPU 数是在整段 Vulkan 之后测的，包温度更高；差距仍然是约一倍，不是热噪声能解释的。

| 模型 | CPU median | Vulkan median | Vulkan / CPU | CER（CPU / Vulkan） |
|---|---:|---:|---:|---|
| tiny | 55.2 | 99.5 | 1.80× | 2.37% / 2.37% |
| small | 134.2 | 351.6 | 2.62× | 0.41% / 0.40% |
| medium | 823.0 | 1628.8 | 1.98× | 0.14% / 0.14% |

`cmp.ps1` 的 `hash` / `detected` / `texts`：三档都是 **100/100**。tiny 对标注的 exact_lines 是 CPU 742、Vulkan 740（CER 相同）；medium 1004 / 1005，落在允许的 1003–1006。`--conc` 4 线程和 8 线程（tiny，24 张 × 3）`mismatches=0`，stderr 没有 `fallback`。

分阶段 mean ms/图（n=99，4 workers；CPU 的 `rec_graph` 是各线程累加，`lines_wall` 才是墙钟）：

| 模型 | 阶段 | CPU | Vulkan |
|---|---|---:|---:|
| tiny | det_graph | 25.7 | 33.0 |
| tiny | lines_wall | 23.2 | 60.3 |
| small | det_graph | 55.3 | 92.4 |
| small | lines_wall | 71.4 | 244.7 |
| medium | det_graph | 414.3 | 522.1 |
| medium | lines_wall | 475.0 | 1068.4 |

识别在 CPU 上按 4 个 worker 重叠，在这块核显上只有一条计算队列，`lines_wall` 接近 `rec_graph`。这是端到端输的主要原因之一，不是某一档 GEMM单独能补上的。

## 纯 GPU

`SIMD_OCR_GPU_PROF=1`，最终 tile。medium DET 960×960：最好墙钟 **769 ms** / 128 次 dispatch，其中 `gemm_nc` 434、`conv1x1_dot` 102、`convk_dot` 85、`conv_dw4` 44、`im2col` 35。medium REC batch 8、宽 480：最好墙钟 **557 ms**，`gemm_nc` 476（约占 85%）。

相对峰值：fp16 理论峰值约 1.6 TFLOPS。这条 GEMM 是共享内存上的标量 FMA，聚合时间对应的有效吞吐在几十 GFLOPS 这一档，离 EU 峰值还差一个数量级以上。其余 shader 不是热点。

试过但没留下的 tile：

| 变体 | medium DET `gemm_nc` | medium REC `gemm_nc` |
|---|---:|---:|
| 32×32，标量进共享内存，64 线程 | 550 ms | 637 ms |
| 32×64，不要共享内存、直接全局读 | 1154 ms | 1339 ms |
| **32×64，f16vec4 进共享内存，128 线程** | **434 ms** | **476 ms** |
| 64×64，同样的向量共享内存，128 线程 | 611 ms | （更慢，未留） |

64×64 复用更好，但 32 EU 上占不满，墙钟变差。直接全局读少了复用，更差。没有再动 depthwise / SE：它们不在前几名。

## 默认路径

三档都没有快过同机 sharp，`RouteNocm` 保持 false。内核和 spv 留在树里。统一内存上的 `preferHost` 修正是一直生效的：先仍选非 device-local 的主机缓存类型（独显走这里），只有选不中时才接受 device-local + host-cached。独显的选择顺序不变。

没有改 sg16 / sg32 / sg32l 的 shader、spv 或选择条件。B580、3080 Ti、880M 上不需要为这次改动复测那三档。
