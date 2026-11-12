> Working material (idea-forming, not a decision). Copied from the owner's untracked `workload-modeling/pattern-table-meeting.md`; RQs do not cite it as a starting point. See [README](README.md).

# VLA Workload Patterns：按计算模式拆解 VLA（meeting 讨论稿）

> 2026-11-06 · 工作稿，未入 wiki · 详细版见 `workload-modeling/pattern-table.md`
>
> 证据标签：**measured (3rd-party)** = 独立 benchmark（Jung et al. ASPLOS'27、VLA-HWBench）；**measured (authors)** = 模型论文自报；**inference** = 我们自己的推断。
> † = 读过全文、未入 wiki 的论文；‡ = 只读过 alphaXiv 自动生成的报告，使用前需对照原文核对。

---

## 0. 背景速览 (Primer)：VLA 系统长什么样

> 给不熟悉 VLA 的同学。图里用英文是为了等宽对齐，解释在图下方。

### 0.1 三层结构：S2 / S1 / S0

```text
        observation: camera images + language instruction + robot state
                                    |
   +--------------------------------v--------------------------------+
   | SYSTEM 2  "big brain"                                           |  slow
   |   VLM trunk (PaliGemma / Qwen-VL / Eagle ...)                   |  ~1-10 Hz
   |   understands the scene and the instruction                     |
   +--------------------------------+--------------------------------+
                                    |  latent / KV cache / features
   +--------------------------------v--------------------------------+
   | SYSTEM 1  "small brain"                                         |  fast
   |   action expert / policy (DiT, flow expert, ...)                |  10-200Hz
   |   outputs an action chunk = the next H actions                  |
   +--------------------------------+--------------------------------+
                                    |  one "action" = EE pose, joint angles,
                                    |  sparse body poses, or a motion token
   +--------------------------------v--------------------------------+
   | SYSTEM 0  low-level control                                     |  fastest
   |   arm:      analytic IK / OSC, no learning                      |
   |   humanoid: learned whole-body tracker, also keeps balance      |
   |   policy rate: 20 Hz (LIBERO), 50 Hz (SONIC), 1 kHz (Helix 02)  |
   +--------------------------------+--------------------------------+
                                    |  joint targets (latest value wins)
   +--------------------------------v--------------------------------+
   | command streaming: non-neural, re-sends the latest targets      |  ~500 Hz
   +--------------------------------+--------------------------------+
   ==================== hard real-time boundary ======================
   +--------------------------------v--------------------------------+
   | joint PD / FOC on per-joint driver MCUs (via EtherCAT)          |  ~16 kHz
   |   outside SoftHier's modeling scope                             |
   +--------------------------------+--------------------------------+
                                    |  currents / torques
                                    v
                                 motors
```

- **S2 + S1 常常是同一个模型**：π0.5、SmolVLA 里 VLM trunk 和 action expert 每次调用一起跑，频率相同。Helix 1（S2 7–9 Hz / S1 200 Hz）是真正分开跑的 dual-system；GR00T N1 自称 dual-system（10 Hz / 120 Hz），但论文没有描述实际调度。OpenVLA、OFT、Holo-M 没有单独的 S1 网络。
- 图中的频率是数量级示意。**S0 policy 本身是 20 Hz–1 kHz**：LIBERO 机械臂跟随 action 频率，是 20 Hz；SONIC 的 tracker 是 50 Hz（measured, authors）；Figure Helix 02 的 learned S0 只有 10M 参数，跑在 1 kHz（first-party）。
- **SONIC 的 500 Hz 不是神经网络**：它是一个单独的 command writer 进程，只负责通过 Unitree 底层 API 把最新的电机目标值不断推送出去。
- **hard real-time boundary 以下是关节驱动层**：每个关节驱动板上的 MCU 跑 PD / FOC。例如 TI TIDUFG6 人形参考设计里，一颗 C2000 F28P65 MCU 跑 6 路 FOC，控制周期 16 kHz，PWM 64 kHz，位置环指令经 EtherCAT 从 motion controller 下发（vendor 文档）。SONIC 的 policy 把关节目标交给 G1 内置的关节 PD，但这个 PD 物理上跑在哪里，论文没有说明。**这一层不在 SoftHier 的建模范围内。**
- **S0 在机械臂上是解析控制器**（LIBERO 用 OSC，VLABench 用 IK），不需要训练，一般也不单独称作一个 "system"；**在人形上是一个学出来的网络**，要负责平衡，所以单独训练、单独讨论。
- "S0" 这个名字来自 TANGO 论文（triple-system），wiki 沿用了它（见 glossary `System 0`）。

### 0.2 一次 VLA 调用里有哪些组件（对应 P 编号）

```text
 images --> [P1 ViT] ------> visual tokens --+
 text   --> tokenizer -----> text tokens ----+--> [P2 LM prefill] -> KV cache
 state  --> (projector) ---> state tokens ---+    VLM trunk (S2)     / features
                                                                        |
                                                                        v
 ACTION HEAD (S1 part, if the model has one), one of:
   (a) P3  decode 1 token per pass, x N tokens       OpenVLA, pi0-FAST
   (b) P4  one pass + small MLP head                 OpenVLA-OFT
   (c) P5  denoise a noisy chunk, x S steps (4-10)   pi0.5, GR00T, SmolVLA
   (d) P6  de-mask token groups, groups x steps      Holo-M
                                  |
                                  v
                        action chunk (H actions)

 VIDEO-MODEL ROUTE (no separate VLM, no action head):
 images --> [P7 VAE] --> latent --> [P7 video DiT, x S steps]
        --> actions read from latent slots                      Cosmos-Policy
```

- 前半段（P1 → P2）结构相似，但规模差别不小：用哪个 VLM、保留多少层、prefix 有多少 token（GR00T N1.7 156 个，π0.5 574–712 个）。token 数的差别足以改变 P2 的瓶颈（见 §2）。
- 后半段 **action head 的形式** 是 workload 差别最大的地方：同样是 3–4B 的模型，(a) VLA-0 要串行生成 208 个 token，(b) OFT 只要 1 次 pass，(c) π0.5 要 10 步，但每步只处理很少的 token。
- Cosmos-Policy 的 VAE 主要是卷积（GPU 时间的 64%），不是 transformer。

### 0.3 怎么组合：一次调用的时间线

```text
 one call, time -->   (block widths are NOT to scale)

 OpenVLA     |P1|--P2--|P3|P3|P3|P3|P3|P3|              6 decode passes
 OpenVLA-OFT |P1|----P2 + P4----|                       1 pass, long sequence
 pi0.5       |P1|--P2--|P5|P5|P5|P5|P5|P5|P5|P5|P5|P5|  10 steps, short seq
 GR00T N1.6  |P1|-P2-|P5|P5|P5|P5|                      4 steps
 HAF         |P1|--P2--|P5 x S|P5 x S|P5 x S|           3 stages in series:
                        stage1 stage2 stage3            move+head, +waist, +arms
 Holo-M      |P1|--P2--|P6xk|P6xk|P6xk|P6xk|            4 groups in series,
                        EEF  body hand kin              k = 2-8 passes each
 Cosmos-Pol. |--P7 VAE--|--P7 DiT--|--P7 DiT--| ... x5  long, compute-bound
```

- **理想情况下前缀（P1、P2）只算一次**，后面的 P3/P5/P6 反复读它的 KV cache 或 features。具体是否复用取决于实现：Isaac-GR00T 每步都重算 context 的 KV，FlashRT 每次调用只算一次。GR00T 通过 cross-attention 读 VLM features，不读 LM 的 KV cache。Cosmos-Policy 没有前缀。
- 在 §4 的 7 个人形系统里，有 2 个（HAF、Holo-M）**把 action head 拆成多段串行**，但顺序不同：HAF 先生成下半身（移动 + 头部 → 腰 → 双臂）；Holo-M 先生成末端位姿，最后才生成底盘运动（EEF → body → hand → kinematics）。

### 0.4 人形 stack：多个循环以不同频率同时跑

```text
1 char=20ms  0 ms                     500 ms                   1000 ms
             |                        |                        |
VLA S2+S1    V------------------------V------------------------V  2 Hz   TANGO
planner      P----P----P----P----P----P----P----P----P----P----P  10 Hz  SONIC
S0 tracker   TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT  50 Hz  SONIC
cmd writer   ###################################################  500 Hz non-NN
- - - - - - - - - - - hard real-time boundary - - - - - - - - - - - - - -
joint PD/FOC MCU, ~16 kHz (320 cycles per char), outside SoftHier scope
```

- 这张图是 **示意性的拼合**：VLA 的频率取自 TANGO，其余各层取自 SONIC（SONIC 的输入循环是 100 Hz，图中省略）。TANGO 本身没有 planner 这一层。
- TANGO 原文说 "the low-level control loop runs at approximately 200 Hz"，但没说这 200 Hz 指的是 policy 推理还是指令推送；SONIC 自己的 policy 是 50 Hz。这个 200 Hz 含义不明，**不作为 P9 的频率使用**。
- 如果把 S0 换成 Helix 02 的 1 kHz learned S0，tracker 那一行要密 20 倍（每个字符 20 次推理）。
- 各层之间按 "latest data wins" 交换数据：慢层还没算完，快层就继续用上一次的结果。
- 对 workload 建模的含义：**每个循环有自己的 deadline**（例如 tracker 每 20 ms 必须出一次结果），不能只看一个端到端延迟（→ 讨论问题 3）。
- 对应 pattern：VLA = P1 + P2 + P5/P6，planner = P8，tracker = P9。

---

## 1. 核心观点 (Key message)

1. **拆到模块层面，VLA 主要由少数几种模块拼成**：ViT、LM prefill、decode、denoiser、video DiT 等，几乎都是 transformer（例外：Cosmos-Policy 的 VAE 主要是卷积）。
2. **但 "基于 PaliGemma / Qwen / Wan" 这种按 backbone 的分类，不是 workload 建模的主轴。** backbone 的 config 仍决定每个 pass 的成本（FLOPs、bytes），但决定 workload 形态的是：
   - 每个 stage 的 **tensor 形状**（每次 pass 处理几个 token），它决定这个 stage 是 compute-bound 还是 memory-bound；
   - **组合方式**：跑几遍、串行还是并行、KV cache 怎么复用、各层以什么频率运行。
3. **Benchmark 支持这个观点**：VLA-HWBench 测了 18 个模型，用参数量预测延迟几乎预测不准（R² ≤ 0.15）。参数量差不多的两个模型（GR00T N1.7 3.1B、VLA-0 3.8B），延迟能差 **53 倍**。主要原因是 **生成动作要迭代几次**。（measured, 3rd-party）
   注意：迭代次数决定数量级；同一数量级内部，每次迭代的成本（Jetson 上每步 4–107 ms，和模型大小有关）决定排序。所以参数量是次要因素，不是没有影响。
4. **建模策略**：每种 compute pattern 建一个**参数化**模型（token 数、权重大小、精度，再加一个软件效率系数），然后按组合方式拼起来，新模型就可以复用这些模型。目前覆盖不到的部分：P6、P8 还没有测量数据；ω-0 的 predictor 还没有归到任何 pattern。

---

### 1.1 两类参数：backbone config 和组合参数

| | **Backbone config** | **组合参数 (composition)** |
|---|---|---|
| 是什么 | 模型的结构超参数，即 HF checkpoint 里 `config.json` 记录的值 | 推理时怎么调用这些模块 |
| 包括 | 层数、`hidden_size`、FFN 维度、attention head 数 / KV head 数（GQA、MQA）、`head_dim`、视觉编码器的分辨率和 patch 大小（决定每张图几个 token）、精度 | 去噪步数 S、chunk 长度 H、串行几个 stage / token group、各层频率、KV 是否复用 |
| 决定 | **单次 pass 有多贵**（FLOPs、bytes） | **要跑几次 pass、按什么顺序** |
| 从哪里来 | `config.json` | 模型论文和推理代码 |

例子（π0.5，**凭记忆写的，需对照 config 核实**）：Gemma 2B trunk 是 18 层、hidden 2048、FFN 16384、8 个 query head / 1 个 KV head（MQA）、head_dim 256；300M action expert 深度相同，但宽度更窄（hidden 1024、FFN 4096）。SigLIP 输入 224²、patch 14，每张图 256 token。

**粗略公式**（一个处理 T 个 token 的 pass）：

```text
weight bytes  ~ params x bytes_per_param    (read once per pass, any T)
GEMM FLOPs    ~ 2 x params x T
attention     ~ 4 x layers x T x context x (heads x head_dim)       FLOPs
KV-cache read ~ layers x context x 2 x kv_heads x head_dim x bytes  bytes

intensity = FLOPs / weight bytes  ~  T   (grows with tokens per pass)
  P5 denoise step: T = 10-41   -> low intensity  -> memory-bound
  P2 prefill:      T = 100s    -> near / above the ridge point
```

- FLOPs 和权重 bytes 都和参数量成正比，所以两者的比值主要由 T 决定。这就是"tensor 形状决定瓶颈"在公式上的来源。
- 所以 **backbone 名字不是主轴**：workload 的形态由组合参数决定；但**每个 pass 的成本仍要从 backbone config 算出来**。
- 这个公式只是 roofline 下限。实测耗时比下限高 1.6–17 倍，差距来自软件效率（见 §5）。

---

## 2. Compute patterns（P0–P9）

| ID | Pattern | 形状特征 (shape) | Batch=1 时的瓶颈 | 例子 |
|---|---|---|---|---|
| P0 | CPU preprocessing | resize、tokenize、host-to-device 拷贝 | 软件问题（串行 Python、线程唤醒）；优化后占 GR00T N1.7 整次调用的 39% | 所有模型 |
| P1 | Vision encode (ViT) | 一次 pass；SigLIP 每张图 256 token，用 pixel shuffle 后每帧 64 token | **随平台变化**：4090 上 compute-bound；Jetson FP8 下偏 memory；AMD FP32 下用不上矩阵单元 | SigLIP、Qwen-VL ViT、V-JEPA |
| P2 | LM prefill | 一次 pass，几百个 token | **由 prefix 长度 × 平台决定**：π0.5（574–712 token）接近 ridge；GR00T N1.7（156 token）在 Jetson 上 memory-bound | Gemma、Qwen、Llama 2 |
| P3 | Cached AR decode | 每次 pass 生成 1 个 token，KV 不断变长 | **memory-bound**；Jetson 上每 token 20–52 ms；对 token 数随场景变化的配置，延迟在 24 帧之间变化 12–122% | OpenVLA、π0-FAST、reasoning VLA |
| P4 | Parallel decode + MLP head | 一次 pass 加 empty action query | compute-bound，backbone 占 99% | OpenVLA-OFT |
| P5 | Iterative denoise（短动作序列） | S 步 × 10–41 个 token，读缓存的 prefix | **所有测过的平台上都是 memory-bound**；eager 模式下是最大的 stage（π0.5 占 75%，GR00T N1.6 占 65%） | π0/π0.5 expert、GR00T DiT、SmolVLA |
| P6 | Masked discrete de-mask | 每组 14–100 个 action token，每组 2–8 次 de-mask | 未测量；inference：应和 P5 一样是 memory-bound | Holo-M † |
| P7 | Video DiT + VAE | 长 latent 序列（Cosmos-Policy 有 7056 个位置） | 4090 上 compute-bound（DiT 占端到端 71%）；torch.compile 反而更慢（0.97×） | Cosmos-Policy |
| P8 | Motion generator | 在短 motion window 上做 diffusion，20 DDIM 步（SP-MGM；SONIC planner 架构未核实） | 没有 roofline 数据；SP-MGM 7.8 Hz；SONIC planner 在 Jetson Orin 上约 12 ms，在 laptop 上 <5 ms（measured, authors） | SP-MGM †、SONIC planner |
| P9 | System 0 tracker | 很小的网络，每个 tick 跑一次 | 主要受控制频率（延迟）约束（inference）。实测锚点：SONIC policy 每次 forward 1–2 ms，周期 20 ms（Jetson Orin、TensorRT + CUDA Graph，measured, authors） | SONIC（50 Hz）、Helix 02 S0（10M 参数，1 kHz，first-party）、SP-GMT † |

**一句话总结**（bound 默认指 RTX 4090 上的结果，除非另注）：
- 少 token、多次 pass（P3、P5）→ memory-bound，**这是目前唯一在所有测过的平台上都成立的规律**。P6 推测也属于这一类，但还没有测量。
- 多 token、单次 pass（P2 长 prefix、P4、P7）→ 在 4090 上 compute-bound；在 Jetson Thor 上所有 stage 都更受带宽影响（Jung 的 DVFS 结果：OFT backbone R = 0.27，Cosmos DiT R = 0.59）。
- P1 和 P2 的瓶颈看 token 数和平台而定。
- ⚠️ **对 SoftHier 最重要的一点**：P5 在 GPU 上 memory-bound，是因为 action head 的权重**每一步都要从 DRAM 重新读一遍**。Jung §6 把"跨去噪步骤把权重留在片上"列为硬件可以发力的地方。所以 P5 在 SoftHier 上还是不是 memory-bound，取决于 head 的权重能不能常驻片上（→ 讨论问题 4）。

---

## 3. Composition axes（C1–C8）

| ID | 维度 | 观察到的取值 |
|---|---|---|
| C1 | 动作怎么解码 | AR token (P3) · parallel decode (P4) · flow/diffusion expert (P5) · masked discrete (P6) · 动作放在 video latent 里 (P7) · **先推理再出动作**（P3 推理 token → P3/P5） |
| C2 | action 部分怎么读 trunk | 共享 self-attention + 缓存 prefix KV (π0) · 对中间层做 cross-attention (GR00T) · cross/self 交替 (SmolVLA) · 和语言共用同一个 LM 和词表 (Holo-M) |
| C3 | 每次调用跑几遍、串行还是并行 | 1 遍 (OFT) · S 步 (π0.5 10 步、GR00T 4 步) · stage × step 串行 (HAF 3 stage) · 4 组串行 × 2–8 次 de-mask (Holo-M) |
| C4 | KV cache 的生命周期 | prefix 跨步复用 · 跨 stage cache (HAF) · ⚠️ **由软件实现决定**：Isaac-GR00T 每步重算，FlashRT 每次调用只算一次 |
| C5 | 运行频率 / 循环 | 单一频率 · S2 慢 S1 快（GR00T N1 10/120 Hz、Helix 1 7–9/200 Hz）· 四个循环（SONIC：输入 100 / planner 10 / policy 50 / 非神经网络的 command writer 500 Hz）· Helix 02：S1 200 Hz、learned S0 1 kHz，S2 频率未公开 |
| C6 | VLA 下面接什么 | 解析控制器 IK/OSC（机械臂）· learned tracker（人形）· VLA → motion generator → tracker（SP-HLM） |
| C7 | Chunking / 异步执行 | chunk 长度 H、实际执行的前 H_e 步、RTC、异步 overlap |
| C8 | **Placement：每个循环跑在哪里** | 云端 / server、机载 GPU、MCU，以及每个循环 deadline 里要算进的网络延迟。Gemini Robotics：backbone 在云端（query-to-response <160 ms），decoder 在机载电脑上，端到端约 250 ms，靠 chunk 做到等效 50 Hz（measured, authors）· TANGO：S2+S1 在 RTX PRO 6000 server，SONIC S0 在机载 Jetson Orin NX，观测上传约 20 ms，VLA 每 0.5 s 一次 · π0：机载 73 ms，off-board 86 ms，网络 13 ms · SONIC：全部推理都在机载 Jetson Orin 上 |

---

## 4. 用 pattern 表示模型（例子）

> 不同行的 ms 数字来自不同硬件、精度和 chunk 设置，**不能横向比较**。

### 固定基座机械臂 manipulation

| 模型 | Pattern 组合 | 延迟（证据） |
|---|---|---|
| OpenVLA | P1 → P2 → P3 ×6 | 209.56 ms，其中 decode 140.9（4090 eager，3rd-party） |
| OpenVLA-OFT | P1 → P4 | 117 ms（4090，3rd-party） |
| π0.5 | P1 → P2 → **P5 ×10** | 294 ms，其中 expert 221（4090，3rd-party） |
| GR00T N1.6 | P1 → P2 → **P5 ×4** | 120 ms，其中 DiT 78（4090，3rd-party） |
| SmolVLA | P1 → P2（只用前一半层）→ P5 ×10 | 191 ms（Jetson Thor，3rd-party） |
| Cosmos-Policy | P7 (VAE) → P7 (DiT ×5) | 396 ms（4090，3rd-party） |

### 人形 whole-body

| 系统 | VLA 部分 | 下层 | 延迟 / 频率（证据） |
|---|---|---|---|
| SONIC + GR00T N1.5 | P1 → P2 → P5 (GR00T N1.5) | P8 planner 10 Hz → P9 50 Hz → 500 Hz command writer（非神经网络） | S0 policy 1–2 ms、planner 约 12 ms（Jetson Orin，authors）；VLA 延迟未报 |
| TANGO | P1 → P2 → P5（远程 server） | P9 SONIC（Jetson Orin NX；原文 "~200 Hz" 含义不明） | VLA 每 0.5 s 一次；观测上传约 20 ms（authors）；VLA 延迟未报 |
| Helix 02 (first-party) | S2（频率未公开）→ S1 200 Hz transformer，输出全身关节目标 | learned S0：10M 参数、1 kHz | 每层跑在什么硬件上未公开 |
| OpenHLM ‡ | P1 → P2 → P5 ×10 (π0.5) | P9 50 Hz | 未报；改成 1 步生成会掉约 20 pp |
| HAF ‡ | P1 → P2 → **[P5 ×S] ×3 stage 串行** | 人形控制器 | 约 0.12 s（RTX 5090，authors） |
| Holo-M † | P1 → P2 → **P6：4 组串行 × 2–8 次 de-mask** | decoupled WBC | 176 / 276 / 477 ms（RTX 5090，authors） |
| SP-HLM † | P1 → P2 → P5 ×4 (GR00T N1.7) | **P8 → P9** | VLA 25 Hz、MGM 10 Hz（TensorRT，authors） |
| ω-0 ‡ | Qwen-VL + V-JEPA + predictor → P5 | P9 SONIC | > 7 Hz（authors） |

**观察**：
- VLM trunk 几乎都还是 π0.5、GR00T、Qwen3-VL 这几个。人形工作改的是 **C3（怎么生成）** 和 **C6（输出到哪一层）**。
- ω-0 的组件里有 Wan，但 Wan encoder 只在训练时用来提取未来帧的 latent 作为监督目标，**推理时不跑 Wan**（‡，来自 AI 生成的报告，需核对原文）。只看组件名字会误导。

---

## 5. 和 benchmark 对照后的修正

| 原来的直觉 | Benchmark 的结论 |
|---|---|
| "Prefill 是 compute-bound" | ❌ 不总是：取决于 prefix 长度和平台（VLA-HWBench） |
| "P5 去噪永远是最慢的阶段" | ⚠️ 只在 eager 模式下成立。GR00T N1.7 完全优化后（Jetson，46 ms），**CPU 预处理（39%）是最大的 stage**，比生成动作（约 4 × 3.3 ms）还大。π0.5 优化后 prefill 占比从 14% 涨到 34%，但按每步 1.92 ms × 10 步估算，去噪约 19 ms，仍比 prefill（15.2 ms）大（inference） |
| "KV 复用是模型本身的属性" | ⚠️ 由软件实现决定（Isaac-GR00T 和 FlashRT 做法不同） |
| "Roofline bound ≈ 实际耗时" | ❌ 优化后的 stage 仍是 roofline 下限的 **1.6–16.9 倍**；π0.5 的 head 里 44% 是 elementwise 操作 |
| 我们最初的 C1 漏了一种形式 | ➕ VLA-HWBench 的 4 种形式里就有 "先推理再出动作"，我们最初的 C1 漏了：推理 token 占整次调用约 91%。已补进 C1 |

---

## 6. 覆盖面检查：和 benchmark 论文比

### 6.1 和两篇 benchmark 的覆盖面对照

| 维度 | Jung et al. (ASPLOS'27) | VLA-HWBench | 我们的拆解 |
|---|---|---|---|
| 模型数 | 5 | 18（19 个配置） | 机械臂 9 个 + 人形 8 个（人形大多只有作者自报数据） |
| 动作生成形式 | AR、iterative denoise、parallel decode、video generation | flow matching、parallel decode、AR token、reasoning then act | 两者的全部形式，**再加** masked discrete（P6）、多段串行的 head（HAF） |
| 单次调用拆成哪些 stage | backbone / head（OpenVLA 再拆 prefill / decode） | 7 个：S0 CPU 预处理 … S6 residual | P0–P7 对应 S0–S4；**没有单列** S5 postprocessing 和 S6 residual（Python dispatch） |
| 平台 | RTX 4090、Jetson Thor、AGX Orin | Jetson Thor、AMD Ryzen AI Max+ | 不绑定平台，引用这两篇的测量结果 |
| 软件优化 | torch.compile | compile、FlashRT / TensorRT、FP8 / NVFP4 | 只体现为精度参数和软件效率系数 |
| 闭环 / 实时性 | LIBERO 闭环 43,200 episodes，扫 overlap | 无闭环；用 T/H < 50 ms 判断能否跟上 20 Hz | C7 只有概念，**没有成功率** |
| 能耗 / DVFS | 有 | 无 | **无** |
| 机器人形态 | 固定基座机械臂（LIBERO） | 同左 | **另加人形 stack**：S0、planner、多频率循环（C5）、placement（C8） |

**结论**：
- **单次 VLA 调用 + 固定基座机械臂**这个场景，我们的 pattern 拆解覆盖了两篇 benchmark 的全部 stage 和生成形式，粒度相当，个别地方更细（P6、多段串行 head）。
- **人形 whole-body stack 两篇 benchmark 都没有覆盖**，这是我们多出来的部分。但这部分的数据几乎都是作者自报，不是在同一套环境下的受控测量。
- 两篇有、我们没有的：**闭环成功率**和**能耗 / DVFS**。我们只有它们的结论，没有对应的维度。

### 6.2 主流场景里还没覆盖的（我们和两篇 benchmark 基本都没有）

是否算"主流"是我们的判断（inference）。

| 场景 | 为什么重要 | 落在哪个 pattern / 维度 | 现状 |
|---|---|---|---|
| **π0.5 的分层推理**：先用 AR 解码出一段子任务文本，再跑 10 步去噪 | π0.5 在家庭场景部署时就是这么用的；两篇 benchmark 跑的是 LIBERO checkpoint，没测这一步 | P3（子任务文本）→ P5；属于 C1 里的"先推理再出动作" | pattern 已有，**成本没测** |
| **多相机 / 多帧历史输入** | π0 用 3 个相机，HWBench 每次调用 1–3 个 view；导航类任务还要保留历史帧 | P1、P2 的 token 数 = view 数 × 帧数 × 每张图的 token 数 | 需要把这个乘积写成显式参数 |
| **推理时采样多个候选 chunk + critic 选择**（Real-Time EXPO-FT）；旁边并行跑小 actor / critic（RL Token，50 Hz） | **打破 batch = 1 的假设**：采样 N 个候选，P5 的计算强度会上升；另外多出一个小网络循环 | P5 的 batch = N；小网络近似 P9 | 两篇 benchmark 都假设 batch = 1 |
| **云端 / 车队服务**：一台 server 服务多个机器人 | 放在云端（C8，例如 Gemini Robotics）以后，多个请求可以合成真正的 batch；Jung 明确说他们的场景"没有请求并发" | C8 + batch 维度 | 未覆盖 |
| **同一颗芯片上同时跑多个负载**：VLA、S0，再加上感知（深度、SLAM 等） | 机载芯片上各个循环会争抢带宽和算力；我们按循环算 deadline 时，默认每个循环独占硬件 | C5 + C8 + 资源争用 | 两篇 benchmark 和我们都没有 |
| **能耗 / 功耗上限** | 人形是电池供电，机载芯片有 power cap | 需要新加一个 energy 维度 | 只有 Jung 测过（固定基座、单个模型） |
| **1 步蒸馏的 head** | 两篇 benchmark 都把它排除在外；OpenHLM 发现改成 1 步会掉约 20 pp | P5 取 S = 1 | 参数化后可以覆盖，没有实测 |

---

## 7. 讨论问题 (Discussion questions)

1. **建模粒度**：按 pattern 建模，能不能覆盖 SoftHier 关心的模型？有没有必须单独建模的例子，比如 P7 video DiT？
2. **软件效率系数**：roofline 下限和实测之间差了 1.6–17 倍。我们的模型要不要显式加一个 per-pattern 的效率系数？还是假设 SoftHier 上能接近 roofline？
3. **多频率循环 (C5) 和 placement (C8)**：人形 stack 有 10 / 50 Hz 的神经网络循环，Helix 02 的 S0 甚至到 1 kHz，下面还有 500 Hz 的指令推送。workload 模型要不要从"单个端到端延迟"改成"每个循环一个 deadline"？如果要，每个 deadline 里还要算上这个循环路径上的传输时间（网络上传、host-device 拷贝）。
   - *inference，仅供讨论*：SoftHier 对应的是 GPU / NPU 那一层（P1–P9），kHz 级的 FOC 留在 MCU 上。如果 S0 是 1 kHz 的 learned policy，它就是**最紧的单循环 deadline**（1 ms），这一点把问题 3 和问题 4 连了起来：这么小的网络，权重能不能常驻片上、每 1 ms 跑一次？
4. **更多串行的小 pass (C3) 和片上权重常驻**：
   - HAF 是在 π0.5 上增加工作量：3 个 stage × S 步，对比 π0.5 的 10 步。论文自己也把额外的去噪计算列为局限。
   - Holo-M 相比逐 token 生成，串行深度从 208 降到 32。
   - 这两者共同的特点是：同一组权重被反复读取，每次只处理很少的 token。如果 SoftHier 能把 action head 的权重放在片上，P5、P6 可能就不再是 memory-bound。片上存储能放下多少？这对 tile-based dataflow 硬件是好事还是坏事？
5. **动态形状**：Qwen ViT 的动态分辨率、P3 随场景变化的 token 数，这些和静态编译怎么兼容？
6. **P0 CPU 预处理**：要不要放进 SoftHier 的建模范围？
7. **建模范围要不要超出"batch = 1、单个负载"**（见 §6.2）：要不要考虑推理时采样多个候选（batch = N）、云端多机器人合成的 batch、同一颗芯片上 VLA + S0 + 感知的资源争用，以及功耗上限？哪些对 SoftHier 的目标场景最重要？

---

## 8. 下一步 (Next steps)

- [ ] 从 config 算出 **π0.5 和 GR00T N1.7** 的每 pass token 数、FLOPs、bytes（权重 + activation + KV）
- [ ] 用 **VLA-HWBench Table 15**（按 stage 给的 F/P、W/B 和实测时间）校验；第一步看单步去噪耗时是否落在带宽下限的 1.6–1.8 倍
- [ ] 把 HAF、OpenHLM ingest 进 wiki，让 ‡ 行有可靠来源
- [ ] 补 P6（Holo-M）、P8（motion generator）的 roofline 估算
- [ ] 把 P1 / P2 的 token 数写成参数：view 数 × 帧数 × 每张图的 token 数
- [ ] 估算 π0.5 分层推理里子任务文本解码（P3）的成本

**Sources**：`wiki://knowledge/vla-benchmark-workloads.md`、`wiki://resources/serving/vla-hwbench-edge-latency.md`、`wiki://resources/serving/vla-workload-characterization-kaist.md`、`wiki://knowledge/vla-architecture-overview.md`、`wiki://knowledge/dual-system-vla.md`；arXiv 2608.16837 (HAF)、2609.35709 (Holo-M)、2606.22174 (OpenHLM)、2610.30044 (SP-HLM)、2608.06375 (ω-0)、2511.07820 (SONIC)、2609.09158 (TANGO)、2503.20020 (Gemini Robotics)；Figure Helix 02 (https://www.figure.ai/news/helix-02, first-party)；TI TIDUFG6 (https://www.ti.com/lit/pdf/tidufg6, vendor)。
