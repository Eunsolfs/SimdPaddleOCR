# SimdPaddleOCR — 骁龙 8 Gen 3（Adreno 750）Vulkan

实测机：真我 GT5 Pro（RMX3888），骁龙 8 Gen 3（SM8650），Adreno 750，16 GB LPDDR5X 与 GPU 共享，Android 16（API 36）。驱动报 Vulkan 1.3.128，`driverVersion`=0x802fa029（GLES 同版本串 V@0762.41）。进程是安卓应用 `test/Sdcb.SimdPaddleOCR.AndroidBench`（`net10.0-android`，CoreCLR，.NET 10.0.11），引用库的 `net10.0` 构建：CPU 走 AdvSimd 内核，Vulkan 直连系统 `libvulkan.so`。数据集用仓库 `dataset/` 100 张，`--workers 4 --warmup 1`，选项与桌面 `Tests` 相同，n=99。

**结论：Adreno 750 没有协作矩阵，compute subgroup 只能是 64 或 128，走无矩阵档（5.3 第 3 条）。原样上机时三档都算错（DET 概率图几乎全零，会话却报告 GPU 正常），修掉三处只在这颗 GPU 上暴露的问题后三档精度与本机 CPU 一致；再按 Adreno 调 GEMM 和两处路由，纯 GPU 快 14–33%。三档端到端都快于本机 CPU（tiny 约 1.3×，small 约 3.3×，medium 约 2.2×），所以 `Auto` 在“无协作矩阵且 subgroup 最小 64”的设备上改走 GPU。桌面 sg16 / sg32 / sg32l 的路由和 spv 没有改。**

## 设备能力（`--caps` 探针）

| 项 | 值 |
|---|---|
| 设备 | `Adreno (TM) 750`，vendor 0x5143，deviceId 0x43051401，`INTEGRATED_GPU`；加载的库是 `libvulkan.so` |
| subgroup | 默认 64；`VK_EXT_subgroup_size_control`，`sgRange` **64–128**，`requiredSubgroupSizeStages` 含 compute |
| 实际 lane 数 | 同一 `VkDevice.NewPipeline` 路径的探针 shader：请求 0 → `gl_SubgroupSize`=64，请求 64 → 64，请求 128 → 128；16 / 32 在范围外，按规范不能请求，没有请求 |
| cooperative matrix | 没有 `VK_KHR_cooperative_matrix`（`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` 不导出），协作矩阵列表为空 |
| fp16 | `shaderFloat16`、16-bit storage 都有 |
| 共享内存 | `maxComputeSharedMemorySize` = **32 KB**；`maxComputeWorkGroupInvocations` 1024 |
| 绑定限制 | `maxStorageBufferRange` = **128 MB**；`minStorageBufferOffsetAlignment` = 64 B |
| 内存 | heap0 15203 MB、heap1 4095 MB，都 `DEVICE_LOCAL`；有 DL\|HV\|HC 和 DL\|HV\|HC\|CACHED 类型（输入缓冲落前者，回读落后者） |
| 队列 | family 0 `0x19f`×3（图形+计算），family 1 `0x2`×1（仅计算，被选中），family 2 `0x8`；三族 `timestampValidBits`=48，周期 52.08 ns，时间戳可用（探针 0.26 ms 一次 dispatch 读回正常）；`globalPriority` 未授予（0） |
| push descriptor | 有 |
| 峰值 | `--peak`：纯寄存器 fp32 FMA **2.09 TFLOPS**；f16vec4 FMA 只有 0.91 TFLOPS（没有翻倍）。GEMM 必须 fp32 累加，按 2.09 算 |

`CoopGemm` 为假、`SubgroupMin` > 16，所以 `GpuGraphModel` 进 `_nocm`，不创建任何 `conv1x1_cm*` / `convk_cm*` 管线。锁不住 8 lane，所以是新加的 `_ncWide` 子档。

## 走了哪一档

5.3 第 3 条：无协作矩阵。大 GEMM（1×1 里没被 dot 接住的、MatMul）走 `gemm_nc` 的 `-DDIRECT` 构建（`gemm_nc_d` / `gemm_nc_ds`），不请求 subgroup 大小（驱动给 64），内核不用 subgroup 内置量；kxk 稠密卷积一律走直接卷积 `convk_dot`；`conv1x1_dot` / `mmdot_sk` / `conv_dw*` / 其余内核照旧。能锁 8 lane 的 UHD 770 仍走原来的 `gemm_nc` / `gemm_nc_s`（`requiredSubgroupSize=8`），spv 逐字节不变。

## 上机后修掉的正确性问题

逐层对比用 `--layers`（把 GPU 图截断到节点 k，读回该节点输出的统计量），同一份代码链接进桌面 GpuBench，在 3080 Ti 上取参照；再用 `--gemm` 把单个 GEMM 和 fp64 参考比。

1. **`gemm_nc` 的寄存器预取在 Adreno 编译器下算错。** 第一处分叉就是第一个走 `gemm_nc_s` 的卷积；单测 6400×32×64 有 188871/204800 个输出错。fma 换乘加、if 换 select 都没用；去掉“下一块 A/B 先读进 `f16vec4` 数组”的预取、或把块缩到 32×32 / 4×4，就完全正确。改成 `-DDIRECT`：每块直接从全局读进共享内存。
2. **标量常量上传只有 2 字节，Adreno 按绑定范围做越界检查。** `elem4` 按 `uint` 读 `b[0]`，桌面驱动不检查，Adreno 越界读回 0：CLS 头里乘常量那一步整个变 0，tiny CLS 只对 306/1022，转错方向的行又把 REC 拖到 CER 57%。`ConstF16` / `VecF16` 补齐到 16 字节的整数倍（只追加 0）。
3. **im2col 矩阵超过 `maxStorageBufferRange`。** medium DET 960×960 的 7×7 卷积（M 57600，K 1568）im2col 有 180 MB，128 MB 之后全读成 0，medium 在大图上丢框（只检出 818 行，CER 13.2%）。放不进一个绑定时改走 `convk_dot`；仍会超范围的计划在运行前抛异常回退 CPU，不再返回半张零图。

试过但确认不需要的：把 arena slab 起点按 `minStorageBufferOffsetAlignment`（64 B）对齐、SE 计数器按 64 B 步进、把描述符 range 截到 128 MB。在这个驱动上都不改变任何输出（逐图、逐像素相同），所以没有留下；按规范它们仍属于未对齐 / 超范围的用法，换驱动时值得再看。

## 端到端（4 workers，median ms/图）

同一时段 CPU / Vulkan 交替 3 轮（`ab.ps1`，轮间降温 60 s；tiny、small 第 2、1 轮的 CPU 因启动脚本竞态丢了，修好脚本后补测一轮，记在最后）。热状态（`PowerManager.CurrentThermalStatus`）tiny/small 为 2–3，medium 全程 3，机身明显发烫。

| 模型 | 本机 CPU | Vulkan | 对 CPU |
|---|---|---|---:|
| tiny | 218.8 / 202.6 / 276.4 | 156.1 / 170.4 / 154.9 / 151.7 | 约 1.3× |
| small | 867.1 / 785.0 / 811.7 | 251.3 / 247.3 / 246.2 / 255.1 | 约 3.3× |
| medium | 3073.8 / 3294.1 / 3368.9 | 1466.4 / 1458.1 / 1467.0 | 约 2.2× |

- 发热是最大的误差来源：同一版本 medium Vulkan 在热状态 2 时 median 833 ms（det_graph 213、rec_graph 558），热状态 3 时 1460 ms（385 / 984）。纯 GPU 的最小值基本不受影响，受影响的是持续负载下的频率。表里 CPU / Vulkan 都是同一热状态下交替测的。
- 工作集峰值：medium CPU 约 1.60 GB，Vulkan 约 1.09 GB；small 约 1.01 / 0.99 GB；tiny 约 0.96 / 0.95 GB。

分阶段 mean ms/图（第 3 轮）：

| 模型 | 阶段 | CPU | Vulkan |
|---|---|---:|---:|
| tiny | det_graph | 118.9 | 20.1 |
| tiny | rec_graph（CPU 为各线程累加） | 270.0 | 64.0 |
| tiny | lines_wall | 83.2 | 84.5 |
| small | det_graph | 384.4 | 40.2 |
| small | lines_wall | 382.5 | 166.2 |
| medium | det_graph | 1365.1 | 384.8 |
| medium | lines_wall | 2036.3 | 1010.8 |

tiny 的收益全在检测上，识别墙钟与 CPU 4 线程持平。Vulkan 运行时 `crop` 阶段（CPU 做透视裁剪）从 9–12 ms 涨到 16–27 ms，推测是 GPU 忙时 CPU 频率被压低，没有再查。

### 精度

| 模型 | 本机 CPU exact_lines / CER | Vulkan exact_lines / CER | 逐图 hash/detected/texts 一致 | 框数一致 |
|---|---|---|---:|---:|
| tiny | 742 / 2.37% | 743 / 2.40% | 78/100 | 99/100 |
| small | 950 / 0.41% | 949 / 0.42% | 90/100 | 100/100 |
| medium | 1004 / 0.14% | 1003 / 0.15% | 97/100 | 100/100 |

本机 CPU 三档与 Windows 基线（742 / 950 / 1004，CER 2.37% / 0.41% / 0.14%）完全一致；CLS 三档都是满分。DET 对拍（同一输入进 CPU / Vulkan 会话）：tiny 10/10 图框数相同、概率图最大偏差 0.05–0.15（3080 Ti 对它自己的 CPU 是 0.03–0.18）；medium 6/6 图框数相同、0.03–0.07。

并发（`--conc`，24 张 × 3）：tiny 4 / 8 线程、small 4 线程、medium 4 线程、`Auto` tiny 4 线程，`mismatches=0`，日志里没有 fallback；medium 4 线程工作集峰值 980 MB，没有被系统杀掉。

## 纯 GPU

`gprof.ps1`（逐 dispatch 时间戳，次数取最小，ms）。“调优前”是三处正确性修复之后、调优之前的版本；两列之间的每一步都做过同一时段交替 A/B，见各提交说明。

| 用例 | 调优前 | 调优后 |
|---|---:|---:|
| medium DET 960×960 | 380.5 | 298.6 |
| medium DET 640×960 | 270.1 | 205.2 |
| medium REC 8×480 | 358.9 | 242.3 |
| medium REC 1×320 | 40.0 | 34.4 |
| small DET 960×960 | 64.5 | 52.7 |
| small REC 8×480 | 72.6 | 54.5 |
| tiny DET 960×960 | 27.9 | 23.7 |
| tiny REC 8×480 | 15.9 | 12.3 |

调优后热点：medium DET 是 `gemm_nc_ds` 92.6、`convk_dot` 88.9、`gemm_nc_d` 53.3、`conv1x1_dot` 19.4、`conv_dw4` 15.8、`conv_dw4t` 11.1；medium REC 是 `gemm_nc_ds` 110.9 + `gemm_nc_d` 95.7（合计 85%），其余是 `conv_dw4t` 20.6、`convk_dot` 6.3。

相对峰值：热点 GEMM 形状合计 0.64 TFLOPS，是 fp32 FMA 峰值 2.09 的 31%；`convk_dot` 在 57600×2304×64 上约 0.56 TFLOPS。

## 调优（每项都在手机上有收益才保留）

1. **GEMM：fp16 共享内存 + BK 8。** `--gemmsweep` 在 11 个最热形状上交替比较（总 ms）：原 64×64 / BK16 / fp32 暂存 109.5；BK 8 83.1；128×128、每线程 8×4 84.1；BK 8 + fp16 暂存 **76.2**（保留）；同样配置换 128×64 / 64×128 块 74.6 / 75.0（噪声内，64×128 在 N=64 上浪费半块）。暂存的值本来就来自 fp16，逐位不变，e2e 与上一版 100/100。
2. **两阶段空间均值在任意尺寸都用。** `reduce_hw` 一个工作组只算一个通道，在 Adreno 上按 C×2 字节跨步读；medium REC 8×480 269.7 → 243.3，medium DET 324.4 → 317.0。
3. **kxk 稠密卷积一律直接卷积。** `CONVD_KMAX` 256 / 512 / 1024 / 2304：medium DET 315.2 / 312.3 / 304.0 / 299.3，small 64.6 / 60.7 / 52.8 / 52.8，tiny 32.8 / 29.9 / 23.7 / 23.7；640×960 216.9 → 206.2。直接卷积也不需要 im2col 暂存区。

重扫但保持原值的：

- K≤128 的 dot → GEMM 门槛（M×cout > 2¹² / 2¹⁴ / 2¹⁶ / 2¹⁸ / 2²⁰）：2¹²–2¹⁶ 相差 0.3 ms 以内，2¹⁸ 起 tiny DET 23.7 → 26.5，保持 2¹⁶。
- `StreamCuts`（small 端到端，两轮）：现有切分 median 234 / 235；不切分 254 / 290；0.3/0.6/0.85 236 / 258；0.7 252 / 262。CPU ArgMax 和后面的 GPU 波重叠在手机上仍然赚，保持原样。

## 试过但没留下的

| 尝试 | 结果 |
|---|---|
| 9×9 depthwise 走共享内存分块的 `conv_dw4t`（sg32 档允许到 9） | medium DET 303.7 → 317.0，small 52.7 → 55.3，更慢 |
| GEMM 不用共享内存，每线程直接按 f16vec4 读自己的 A/B 行 | 慢 2–4 倍；全展开的 8×8 版本又被编译器算错 |
| GEMM BK 4 / BK 32 / 32×32 4×4 块 | 83.2 / 200.6 / 148.1（对 76.2） |
| 共享内存按 f16vec4 向量读 | 76.8，与标量读相同 |
| fp16 乘法 fp32 累加 | 没做：f16vec4 FMA 峰值只有 fp32 的 0.43 倍，且会改变数值 |
| slab / 计数器按 64 B 对齐、描述符 range 截到 128 MB | 输出不变，没有收益 |

## 留下的优化

- GEMM 只到峰值的 31%。不用共享内存、全靠缓存的版本慢 2–4 倍，推测 SSBO 读在这里命中不了足够近的缓存；下一步可以试通过纹理路径读（需要给管线加 texel buffer 绑定），没有验证。
- medium DET 的 `convk_dot` 已经第二（88.9 ms），都是 7×7 / 大 K 卷积，M 方向 4 行一组，可以试更大的寄存器块。
- REC 的 `conv_dw4t` 20 ms、DET 的 `conv_dw4` 16 ms。
- 发热：持续跑 medium 时热状态到 3，端到端从 833 ms 涨到 1460 ms。更少的 dispatch / 更短的峰值比单纯缩短纯 GPU 时间更值。
- 规范层面：未对齐的绑定偏移（64 B）和超出 128 MB 的描述符 range 在这个驱动上无害，但属于无效用法。

## `Auto`

三档在所有交替轮次里 Vulkan 都快于本机 CPU，所以 `GpuBackend.UsesGpu` 对 `Auto` 在“没有协作矩阵、`SubgroupMin` ≥ 64”的设备上返回真（按能力判定，没有写死骁龙 / Adreno）。实测 `--backend auto`：usesGpu=True，与 `--backend vulkan` 逐图 100/100；`Auto` + `SIMD_OCR_BACKEND=cpu`：与 CPU 100/100。能锁 8 lane 的 UHD 770 和其它无矩阵设备（最小 subgroup < 64，例如不带协作矩阵驱动的 NVIDIA、Mali）没有在这条规则里，`Auto` 仍走 CPU，显式 `Vulkan` 才走 GPU。

## 桌面路径

sg16 / sg32 / sg32l 的 shader、spv 和选择条件都没动；全量重编后 `gemm_nc.spv`、`gemm_nc_s.spv` 和其余 spv 逐字节不变（`se_fused_dbg.spv` 删掉，不提交）。共享的 `GpuGraph` 改了两处会在桌面执行的地方：fp16 常量补齐到 16 字节、按 `maxStorageBufferRange` 的门槛和保护（桌面范围是 GB 级，不触发）。3080 Ti（驱动 581.80）同一时段对 `feature/2.0` 交替 2 轮，`--engine vulkan` median ms：tiny 18.0 / 19.9 → 18.5 / 20.7，small 26.7 / 30.3 → 29.2 / 28.5，medium 36.5 / 37.6 → 37.3 / 37.3（噪声内）；精度 741 / 950 / 1006 不变；逐图 100/100（三档），`--engine sharp` tiny / small 100/100。B580、880M、UHD 770 没法测：它们都不满足 `SubgroupMin` ≥ 64，走的路由没变，只有常量缓冲补齐这一处共享改动。

## 复现

```text
cd test/Sdcb.SimdPaddleOCR.AndroidBench
.\run.ps1 -Build -Install -Push -Bench "--caps"            # 推 bench-out/models 和 dataset/，打印能力
.\run.ps1 -Bench "--e2e --model medium --backend vulkan"   # JSON 与 logcat 拉回 bench-out/android
.\ab.ps1 -tag x -models tiny,small,medium -rounds 3         # CPU / Vulkan 交替
.\gprof.ps1 -tag x                                          # 纯 GPU
```

需要 .NET SDK ≥ 10.0.302 和 `android` workload、Android SDK platform 36、JDK 17/21。模型从手机存储读（medium 的 DET+REC 约 130 MB，不打进 APK）。
