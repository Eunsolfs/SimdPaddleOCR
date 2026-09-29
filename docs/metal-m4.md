# SimdPaddleOCR — Apple Silicon Metal 后端实测（Devin VM / M4 paravirt）

实测机：macOS VM（Apple M4 paravirt GPU，8 vCPU），SDK .NET 10，`test/Sdcb.SimdPaddleOCR.Tests` bench（`--workers 4 --benchmark-kind simd --engine sharp|metal`，warmup=1，n=99，dataset/ 100 张固定图、张张不同尺寸）。commit：`metal-backend` 分支（PR #21），下表为 review 修订（concurrent encoder、device 级 MSL lib/PSO cache）之后在同一轮重测的数据。

**结论：三档 Metal 全部净胜 CPU（tiny 1.75× / small 4.00× / medium 6.47×），检测行数与 CPU 完全一致，文本差异为 fp16 级噪声（vs CPU 逐图差：tiny 11 图、small 4 图、medium 3 图，多数为单字符空格增减；medium 的 exact_lines/CER 反而略优于 CPU）。受 paravirt 虚拟化所限绝对数偏保守——实测 MMA≈fp32≈3TFLOPS、copy ~84GB/s、dispatch ~45µs，均低于真机 M4，真机数字预期更好。**

## 端到端（4 workers，median ms/图，越低越好）

| 模型 | CPU(sharp) | Metal | 比值 | CPU img/s | GPU img/s |
|---|---:|---:|---:|---:|---:|
| tiny   | 68.8  | **39.3**  | 1.75× | 14.5 | 25.4 |
| small  | 218.4 | **54.6**  | 4.00× | 4.6  | 18.3 |
| medium | 693.0 | **107.0** | 6.47× | 1.44 | 9.35 |

注：与 Vulkan/B580 表口径相同（dataset 变 shape 是最不利 GPU 的场景）。Metal 走 `OcrBackend.Metal`（Auto 在 macOS+Metal 可用时自动选）。

## 分阶段耗时（mean ms/图）

| 模型 | 阶段 | CPU | Metal | 倍率 |
|---|---|---:|---:|---:|
| tiny   | det_graph | 44.0 | 13.0 | 3.4× |
| tiny   | cls_graph | 12.2 | 4.3 | 2.8× |
| tiny   | rec_graph | 101.7 | 18.6 | 5.5× |
| small  | det_graph | 124.6 | 13.6 | 9.2× |
| small  | cls_graph | 12.0 | 3.2 | 3.8× |
| small  | rec_graph | 495.4 | 31.4 | 15.8× |
| medium | det_graph | 317.4 | 33.1 | 9.6× |
| medium | cls_graph | 12.4 | 3.2 | 3.9× |
| medium | rec_graph | 1877.8 | 68.9 | 27.3× |

精度（与性能表同一轮 JSON 跑出的三个口径）：

- **检出**：metal/sharp 逐图 `detected` 完全一致（1016/1009/1013，共 300 runs 0 差）。
- **对真值**（dataset 标注）：exact_lines — tiny 744/1024 = 744/1024、small 811/1024 = 811/1024、medium **897 vs 894（Metal 更高）**；CER — tiny 1.640% vs 1.614%、small 1.392% vs 1.396%、medium 0.665% vs 0.677%。
- **对 CPU 逐图文本差**（同一轮的两个 JSON 直比）：tiny 11 图 / 49 字符（10 张为单字符空格增删，最差 img-078 差 38 字符）、small 4 图 / 4 字符、medium 3 图 / 3 字符；逐图 hash 差与文本差同集；CLS 旋转角 1 图翻转（tiny，含在上述集合内）。属 fp16 阈值临界噪声，与 Vulkan 同型已知限制。

并发正确性：`--conc`（4 线程 × 72 runs/模型）0 mismatch；det medium 228 节点逐节点对拍（本地 probe 工具，未提交）× 8 跑 bitwise 一致。

## 内存与分配

| 模型 | CPU peak WS | Metal peak WS |
|---|---:|---:|
| tiny   | 688 MB  | 578 MB |
| small  | 955 MB  | 649 MB |
| medium | 1709 MB | 942 MB |

GPU 侧 arena 是共享 grow-only device buffer + 托管 schedule（MetalSchedule 纯数据 LRU=512，逐 shape 录制代价为零），与 Vulkan 的 plan/arena 形态同构。托管堆分配 ~1.4–2.5MB/图。

## GPU 能力实测（paravirt VM microbench）

| 项 | 实测 |
|---|---|
| fp32 FMA 峰值 | ~3.19 TFLOPS |
| fp16 MMA（half8x8×float-acc） | ~2.5–3.0 TFLOPS（≈fp32，疑似模拟实现） |
| copy 带宽 | ~84 GB/s |
| dispatch 摊销（单 CB 内） | ~40–48µs |
| commit+wait 往返 | ~250–300µs |
| threadgroup mem | 32KB；execWidth=32；maxTg=1024 |

**含义：paravirt 上 fp16 MMA 无 tensor-core 加速**——kernel 收益主要来自减少 DRAM 流量和 dispatch/barrier 次数，而不是算力切换。

## M4 实测调优记录（保留的改动）

| 优化 | 机制 | 实测 |
|---|---|---|
| `mm_sg_dd` | 64×64 MMA tile 直接从 device 内存 simdgroup_load，省掉 threadgroup staging 往返；emit gate `K%16==0`（边界 tile 的越界读被 arena ≥1MB 尾 slack + W /128 pad + epilogue 掩码兜住，数学不变） | 5760×512×1024 GEMM 2.63→2.31ms；det tiny 8.6→6.8ms、small 20.8→17.7ms |
| `mm_ic_sg`（隐式 GEMM 卷积） | kxk conv 不再物化 `[M,Kp]` im2col 矩阵：A staging 内联 tap 寻址 gather；同时接管 nb==1 的 convk_dot 分支 | det medium 960² 86.2→**80.0ms**（-7%）；消掉 ~1.28GB 无谓读写 |
| SE 两阶段定稿 | `se_part`（S WG 写 partial，零原子）+ buffer barrier + `se_join`（1 WG/batch）| 修掉跨 WG ticket 竞态（MSL 原子仅 relaxed 序，跨 WG 握手不可能）；det 逐节点对拍全绿 |
| conv_dw4a flat | 平铺 depthwise 省 threadgroup staging | **未采纳**：实测比 conv_dw4t 慢（staging 省 halo 重读仍有收益）→ 留 `SIMD_OCR_DWA` 开关 |
| `mm_ic_sg32`（BK=32） | staging barrier 减半 | **未采纳**：medium −1ms 但 tiny/small 更差 → 留 `SIMD_OCR_IC32` 开关 |
| NODOT（conv1x1 全 MMA） | 关 dot kernel | 与现状持平 → 阈值保持 B580 标定值 |
| rec bias-free SE 融合 | rec 的 fc1/fc2 conv 无 bias（Inputs[2]==MaxValue），IsPwConv 拒绝 | **跳过**（Vulkan 同缺），记为遗留机会 |

## 已知限制 / 未验证项

- DET bs>1 resize 不支持（同 Vulkan）。
- MSL 通用限制：device `memory_order` 只有 relaxed（非 paravirt 独有）→ 跨 WG 定序走两阶段 split + `memoryBarrierWithScope:Buffers`；paravirt 另限 `char16/uchar16` 不可用。
- 序列化逐 dispatch profile（`SIMD_OCR_GPU_PROF`）有 ~0.4–0.5ms/dispatch 地板，绝对值偏大；真实比例看 `SIMD_OCR_GPU_TIME` 的墙钟分解。
- CI：osx-arm64 smoke（macos-26 runner，paravirt GPU）以 `--engine metal` 跑 tiny 20 张，断言 stderr 含 `[metal] device:`、无 `fallback`、逐图 `detected` 与 CPU 一致；不比较文本，文本漂移靠本文的逐图对比数据把关（已接受的限制：CI 发现不了「框数对、文本错」类回归）；其他 OS 的测试不会走到 Metal 路径。
