# SimdPaddleOCR — Ryzen 9 5950X / Intel Arc B580（2.0 发布复测）

实测机：主机名 3800X，Windows 10.0.26100，电源方案“高性能”，Ryzen 9 5950X（16C/32T，AVX2，无 AVX-512），128 GB，Intel Arc B580（驱动 32.0.101.8331）。运行时 .NET 10.0.12，SDK 10.0.401。`test/Sdcb.SimdPaddleOCR.Tests`，`--workers 4 --benchmark-kind simd --warmup 1`，n=99，同一 `dataset/` 100 张变尺寸图（对 GPU 最不利的逐图新 shape）。先用当前树 tiny 25 张烤机，再按 tiny → small → medium、每档 **1.4.2 CPU → 当前 CPU → Vulkan** 交替跑 3 轮。本表 `f8b1197`。

两条 CPU 用同一套选项（`LineWorkerCount=4`、`AdaptiveWidth`、`TargetWidth=320`、session cache 32、REC 批大小保持默认 1）。1.4.2 是 NuGet `Sdcb.SimdPaddleOCR` **1.4.2** 加模型包 **1.0.0**（harness 临时改成 `UseNuget142=true` 另编一份）。当前树是同一 harness 的源码引用，`--engine sharp` 钉死 CPU，`--engine vulkan` 走 GPU；GPU 在批大小未显式设置时用默认 16。仓库里的 onnx 自 `v1.4.2` 起没有改过，CPU 差值是代码，不是权重。

**结论：Vulkan 三档都快于两份 CPU。相对当前 CPU，端到端中位是 tiny 3.0× / small 6.4× / medium 9.6×；相对已发布的 1.4.2 是 3.3× / 7.3× / 10.4×。当前 CPU 比 1.4.2 快一截（1.09× / 1.15× / 1.08×），行精确和 CER 与 1.4.2 相同。Vulkan 正确率与当前 CPU 持平（tiny 少 2 行，medium 少 1 行，CER 持平或更好）。small / medium 的工作集峰值低于同轮 CPU。**

## 端到端（4 workers，median ms/图，越低越好）

三轮 median 都列出来，加粗的是三轮的中位数。加速是较慢一列的中位毫秒除以较快一列（大于 1 表示更快）。

| 模型 | 1.4.2 CPU | 当前 CPU | Vulkan | 当前 CPU / 1.4.2 | Vulkan / 当前 CPU | Vulkan / 1.4.2 |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 72.6 / 71.5 / **72.2** | 66.4 / **66.0** / 63.8 | 21.9 / **22.2** / 25.3 | 1.09× | **3.0×** | 3.3× |
| small | **237.3** / 244.5 / 234.9 | 204.3 / 215.3 / **205.9** | 32.1 / 33.9 / **32.3** | 1.15× | **6.4×** | 7.3× |
| medium | 603.0 / **591.5** / 586.3 | 545.2 / **546.5** / 552.9 | 57.9 / **57.1** / 56.7 | 1.08× | **9.6×** | 10.4× |

三轮 mean 的均值，以及由此得到的 img/s（1000 / mean）。p95 是三轮 p95 的中位数。工作集峰值是三轮里的最大值。

| 模型 | 列 | mean | p95 | img/s | WS peak |
|---|---|---:|---:|---:|---:|
| tiny | 1.4.2 CPU | 75.4 | 113.4 | 13.3 | 520 MB |
| tiny | 当前 CPU | 70.0 | 104.2 | 14.3 | 516 MB |
| tiny | Vulkan | 26.5 | 36.8 | **37.7** | 579 MB |
| small | 1.4.2 CPU | 237.9 | 301.9 | 4.2 | 676 MB |
| small | 当前 CPU | 210.5 | 273.2 | 4.8 | 652 MB |
| small | Vulkan | 35.8 | 58.3 | **27.9** | 662 MB |
| medium | 1.4.2 CPU | 589.0 | 739.7 | 1.7 | 1219 MB |
| medium | 当前 CPU | 544.3 | 694.0 | 1.8 | 1173 MB |
| medium | Vulkan | 60.0 | 83.2 | **16.7** | 936 MB |

同机更早的 `26ad4c3` 单轮（tiny CPU 51.8 / Vulkan 19.8，small 199 / 28.3，medium 539 / 55.7）绝对毫秒更低。这次三轮彼此差在几毫秒以内，发布用这一把尺子。比值的结论没变：tiny 大约 3×，small 大约 6–7×，medium 大约 10×。

## 分阶段耗时（三轮 stage mean 再平均，ms/图）

4 worker 下各阶段重叠，加总可以大于墙钟。倍数是 Vulkan 相对当前 CPU。

| 模型 | 阶段 | 1.4.2 CPU | 当前 CPU | Vulkan | 相对当前 CPU |
|---|---|---:|---:|---:|---:|
| tiny | det_graph | 29.8 | 30.1 | 6.2 | 4.9× |
| tiny | cls_graph | 18.5 | 19.7 | 1.3 | 15× |
| tiny | rec_graph | 98.4 | 84.3 | 7.0 | 12× |
| small | det_graph | 95.8 | 98.4 | 7.9 | 12× |
| small | cls_graph | 24.7 | 24.2 | 1.3 | 19× |
| small | rec_graph | 419.1 | 332.3 | 16.6 | 20× |
| medium | det_graph | 225.2 | 229.8 | 22.3 | 10× |
| medium | cls_graph | 20.0 | 22.2 | 1.4 | 16× |
| medium | rec_graph | 1284.2 | 1103.2 | 27.7 | 40× |

当前 CPU 相对 1.4.2 的墙钟差几乎全在 `rec_graph`：tiny 98.4 → 84.3（0.86×），small 419 → 332（0.79×），medium 1284 → 1103（0.86×）。`det_graph` 三档都在 1.4.2 的 ±3% 里。GPU 三段都更快。`lines_wall`（1.4.2 → 当前 CPU → Vulkan）：tiny 35.2 → 31.9 → **11.6**，small 124 → 102 → **20.6**，medium 355 → 309 → **31.5**。

## 正确率（100 张满勤，1036 行；三轮逐轮相同）

| 模型 | 1.4.2 exact_lines / CER / exact_img | 当前 CPU | Vulkan |
|---|---:|---:|---:|
| tiny | 742 / 2.37% / 5 | 742 / 2.37% / 5 | 740 / 2.36% / 5 |
| small | 950 / 0.41% / 44 | 950 / 0.41% / 44 | 950 / 0.40% / 44 |
| medium | 1004 / 0.14% / 71 | 1004 / 0.14% / 71 | 1003 / 0.14% / 70 |

cls 在各自检出的行上全对：1.4.2 与当前 CPU 都是 1022/1022、1034/1034、1035/1035；Vulkan 是 1023/1023、1034/1034、1035/1035（tiny 分母多 1，因为检出的行集合不同）。small 与两份 CPU 逐行持平。medium 少的那 1 行、tiny 少的 2 行仍是 fp16 边界，见下面已知缺口。

## 内存（MB）

loaded / peak 是三轮范围，Δ WS 是三轮（last − loaded）的均值。

| 模型 | 1.4.2 loaded | 1.4.2 peak | 1.4.2 Δ | 当前 CPU loaded | 当前 CPU peak | 当前 CPU Δ | Vulkan loaded | Vulkan peak | Vulkan Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | 400–401 | 517–520 | 119 | 401 | 514–516 | 112 | 442–443 | 577–579 | 136 |
| small | 451–452 | 674–676 | 219 | 452 | 649–652 | 194 | 493 | 644–662 | 153 |
| medium | 698 | 1214–1219 | 519 | 698–699 | 1171–1173 | 474 | 739–740 | **933–936** | 168 |

当前 CPU 的峰值低于 1.4.2，差在跑图过程的 Δ WS，loaded 几乎一样。Vulkan 的 loaded 更高（图和 arena 在加载时就占上），small 峰值与 CPU 同级，medium 峰值低于两份 CPU，三轮之后不再爬。

JSON：`bench-out/b580-v20/{v142,cpu,vk}-{tiny,small,medium}-r{1,2,3}.json`。

## 内存是怎么收到有界的

`26ad4c3` 那次单轮的工作集峰值是 CPU 523 / 680 / 1225 MB，Vulkan 585 / 652 / 942 MB。上面三轮的量级一样：small / medium 的 GPU 峰值低于同轮 CPU。修复前 tiny / small / medium 曾到 3897 / 5781 / **24133 MB（无界爬升）**；`512bcd2` 收到 964 / 1036 / 1457 MB；共享 graph 之后落到现在这张表。

修复内容(commit 512bcd2):arena/inF32/outF32 改为 graph 级共享 grow-only buffer(扩容即失效全部 plan 重绑);plan 缓存 LRU,evict 时真正释放 cmd buffer/descriptor set/query pool(VkDevice 补 vkFreeCommandBuffers/vkFreeDescriptorSets,desc pool 开 FREE_DESCRIPTOR_SET_BIT);VkBuffer.Free() 落地。随后 `2be784a` 共享 GPU graph model,权重不再按 session 复制。

**~32-shape 崩坏验证(应 3080Ti reviewer 要求):** 临时 `MaxPlans`→64 跑全 100 图(每张都是不同 det shape,plan 数必然越过旧崩坏点):零 mass-corruption,与 CPU 行数差只剩 img-014 一例(见下)。→ root cause 确证是资源耗尽/陈旧绑定,已被真释放+invalidate 杀死,不是被 LRU 上限掩盖。随后把上限定为 64:shape 剧烈变化的负载下重建更少。

## 已修的正确性/覆盖问题

- **det 残差 Add 吸收顺序**:relu(conv+res) vs relu(conv)+res —— `act==0` 门控修复,det 全 shape IoU 0.9996+。
- **回落路径输入布局**:模型图输入被 TensorNhwc 标记而 GpuSession 报 InputIsNhwc=false → 回落 CPU 时 NCHW 被当 NHWC 读 → 确定性乱码。`CpuInput()` 转置修复,--reciso 对拍 diff=0/11。
- **回落后的批形状**:`GpuAlive` 反馈让 recognizer 在首个 rec plan 失败后切回 CPU 风格分组,避免单批串行+最大宽 padding 双重损失。
- **SVTR rec GPU emit**(`2be784a`):MaxPool batch、rank-3 ReduceMean、5-D Transpose、Slice、Concat(axis=0) 等已覆盖;small/medium rec 不再回落 CPU。
- **REC/CTC overlap**(`26ad4c3`):GPU rec 与 CPU CTC 尾重叠,砍掉 GPU 路径上的 CPU glue。

## 已知缺口(按收益排序)

1. tiny 的 GPU rec 精度:fp16 特征噪声在低置信行翻转 argmax —— 本轮已从 678/1036 收到 740/1036(CER 反超 CPU)。要么 fp32 段,要么接受(det 同款抖动已证实无害)。
2. **det fp16 边界丢框(img-014,已定位):** CPU 16 框 / GPU 15 框 —— GPU 把右侧两个竖排条带合并。根因:单个桥接像素 (918,71) 概率 cpu=0.1892 vs gpu=0.2020 跨过 0.2 bitmap 阈值;全图 400K 像素仅 15 个阈值翻转且全部落在 [0.189,0.205] 窄带,输出 map maxAbs=0.029 即 ~50 节点 fp16 存储的累计噪声(最大的是 FPN concat t399 elemMax=0.41,张量量程 ±160 时 fp16 ulp≈0.125)。**非 kernel bug**:同一输入用 GDI+ 解码跑纯 CPU 也合并(15 行),本来就是临界输入;唯一实质性修法是 fp32 arena,代价是带宽翻倍。暂不修,记为已知限制。
3. per-dispatch ~55µs 固定开销 × 78 dispatch ≈ 4.3ms/Run 下限 —— 继续融合或 push-descriptor 直录可压。
4. `_dev.Sync` 全局锁串行 det/rec —— 双流 overlap 未做(REC 与 CTC 尾已重叠,det/rec 之间还没有)。

## 目标框架:GPU 仅 net10.0,netstandard2.0 不走 Vulkan

`Backends/Vulkan/**` 在 ns2.0 下整目录 `Compile Remove`,`OcrSessionFactory` 在该 TFM 上恒返回 CPU session。**这不是能力限制而是刻意的维护决策**——实测过一遍 API 差距,全部可移植但没有一处是免费的:

| ns2.0 缺的 API | 用量 | 移植代价 |
|---|---|---|
| `LibraryImport`(net7+ source-gen P/Invoke) | 59 个入口 | 退回 `DllImport` 或自写委托加载——失去 source-gen marshal 的可维护性收益,这是有意保留的现代写法 |
| `NativeLibrary.SetDllImportResolver`(net5+) | 1 处 | 用于定位 vulkan-1.dll/libvulkan.so.1;ns2.0 只能 kernel32 `LoadLibrary`/libdl `dlopen` 手写 loader,又要一套平台分叉 |
| `System.Half`(net5+) | ~40 处 | fp16 权重转换;可换成 ushort+位运算 helper,但多一条自编码路径要维护 |
| `BitConverter.SingleToUInt32Bits`(ns2.1+) | 1 处 | 一行 unsafe 转换,小事 |
| `MathF`/`Span`/`MemoryMarshal`/`ArrayPool`/`stackalloc`/`delegate*` | 多处 | 已有 BCL 包(`System.Memory`/`Unsafe`/`Microsoft.Bcl.Numerics`)或语言特性,均可直接用 |

结论:移植是纯体力活(~半天),但会在 P/Invoke 层和 loader 层各长出第二套实现。ns2.0 的定位本来就是 best-effort 兼容旧消费方,而 GPU 场景的用户天然在 modern .NET 上;为不让两份 Vulkan 互操作代码同步腐烂,决定 **ns2.0 只留 CPU**。若未来真有需求(如 .NET Framework 应用要吃 GPU),再按上表做一次性移植即可。
