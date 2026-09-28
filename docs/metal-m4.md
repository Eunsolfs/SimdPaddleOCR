# SimdPaddleOCR — Apple Silicon Metal 后端实测（Devin VM / M4 paravirt）

实测机：macOS VM（Apple M4 paravirt GPU，8 vCPU），SDK .NET 10，`test/Sdcb.SimdPaddleOCR.Tests` bench（`--workers 4 --benchmark-kind simd --engine sharp|metal`，warmup=1，n=99，dataset/ 100 张固定图、张张不同尺寸）。commit：working tree @ feature/2.0 + metal-backend 工作区（未提交）。

**结论：三档 Metal 全部净胜 CPU（tiny 1.9× / small 4.7× / medium 5.2×），检测行数与 CPU 完全一致，文本差异仅 fp16 级噪声（3 图各 1 字符）。受 paravirt 虚拟化所限绝对数偏保守——实测 MMA≈fp32≈3TFLOPS、copy ~84GB/s、dispatch ~45µs，均低于真机 M4，真机数字预期更好。**

## 端到端（4 workers，median ms/图，越低越好）

| 模型 | CPU(sharp) | Metal | 比值 | CPU img/s | GPU img/s |
|---|---:|---:|---:|---:|---:|
| tiny   | 73.9  | **38.9**  | 1.90× | 13.5 | 25.7 |
| small  | 269.7 | **57.1**  | 4.72× | 3.7  | 17.5 |
| medium | 935.5 | **178.4** | 5.24× | 1.07 | 5.6  |

注：与 Vulkan/B580 表口径相同（dataset 变 shape 是最不利 GPU 的场景）。Metal 走 `OcrBackend.Metal`（Auto 在 macOS+Metal 可用时自动选）。

## 分阶段耗时（mean ms/图）

| 模型 | 阶段 | CPU | Metal | 倍率 |
|---|---|---:|---:|---:|
| tiny   | det_graph | 35.5 | 12.7 | 2.8× |
| tiny   | cls_graph | 11.3 | 3.25 | 3.5× |
| tiny   | rec_graph | 99.7 | 15.9 | 6.3× |
| small  | det_graph | 104.7 | 14.9 | 7.0× |
| small  | cls_graph | 12.2 | 2.45 | 5.0× |
| small  | rec_graph | 585.5 | 33.3 | 17.6× |
| medium | det_graph | 315.1 | 59.6 | 5.3× |
| medium | cls_graph | 11.9 | 2.61 | 4.6× |
| medium | rec_graph | 2322.2 | 112.6 | 20.6× |

精度：metal/sharp 检出 lines 完全一致（1016/1009/1013）；逐图 hash 3 张差（fp16 阈值临界像素，与 Vulkan 同型已知限制）；文本 3 图各差 1 字符（CER 0.67% vs CPU 0.14%，在 fp16 噪声范围内）。`--conc`（4 线程 × 72 runs/模型）0 mismatch；`--dtr`（det medium 228 节点 × 8 跑）逐节点 bitwise 一致。

## 内存与分配

| 模型 | CPU peak WS | Metal peak WS |
|---|---:|---:|
| tiny   | 691 MB  | 574 MB |
| small  | 857 MB  | 647 MB |
| medium | 1691 MB | 948 MB |

GPU 侧 arena 是共享 grow-only device buffer + 托管 schedule（MetalSchedule 纯数据 LRU=512，逐 shape 录制代价为零），与 Vulkan 的 plan/arena 形态同构。托管堆分配 ~1.4–2.5MB/图。

## GPU 能力实测（paravirt VM，--mprobe）

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
| SE 两阶段定稿 | `se_part`（S WG 写 partial，零原子）+ buffer barrier + `se_join`（1 WG/batch）| 修掉跨 WG ticket 竞态（paravirt MSL 无 acq_rel 原子，握手不可能）；`--dtr` 全绿 |
| conv_dw4a flat | 平铺 depthwise 省 threadgroup staging | **未采纳**：实测比 conv_dw4t 慢（staging 省 halo 重读仍有收益）→ 留 `SIMD_OCR_DWA` 开关 |
| `mm_ic_sg32`（BK=32） | staging barrier 减半 | **未采纳**：medium −1ms 但 tiny/small 更差 → 留 `SIMD_OCR_IC32` 开关 |
| NODOT（conv1x1 全 MMA） | 关 dot kernel | 与现状持平 → 阈值保持 B580 标定值 |
| rec bias-free SE 融合 | rec 的 fc1/fc2 conv 无 bias（Inputs[2]==MaxValue），IsPwConv 拒绝 | **跳过**（Vulkan 同缺），记为遗留机会 |

## 已知限制 / 未验证项

- DET bs>1 resize 不支持（同 Vulkan）。
- paravirt 独有：MSL `memory_order` 只有 relaxed → 跨 WG 定序走两阶段 split + `memoryBarrierWithScope:Buffers`；`char16/uchar16` 不可用。
- 序列化 `--prof` 有 ~0.4–0.5ms/dispatch 地板，绝对值偏大；真实比例看 `--time` 墙钟与 `--mgemm` N-reps-in-one-CB。
- 无 CI Metal 依赖；单元测试在非 macOS 上 SKIP。
