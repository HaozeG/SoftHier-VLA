# SoftHier-VLA stage 1: specification

*   **Author:** Haoze (with Claude)
*   **Status:** Draft, for owner review
*   **Date:** 2026-11-19
*   **Companion:** [`stage1-problem-definition.md`](stage1-problem-definition.md) (problem statement, RQs).
    This document fixes the hardware, the workloads and the hand-written kernel requirements that stage 1
    is measured on. Where the two disagree, this document wins for those three topics.

Labels follow the wiki's: `measured`, `estimate`, `inference`. Constants keys refer to
[`infra/workload-modeling/constants.yaml`](../../infra/workload-modeling/constants.yaml).

--------------------------------------------------------------------------------

## 1. Scope

Stage 1 has two workloads and one hardware configuration:

| ID | Role | Section |
| --- | --- | --- |
| **W0** `pr1-1cam` | Infrastructure workload: sets up and debugs the SoftHier toolchain against softhier-mlir's end-to-end run (PR1). Kept as is; not a stage-1 measurement target. | §3.1 |
| **W1** `vlabench-3cam` | Stage-1 target workload. All stage-1 hand-kernel work and RQ numbers use it. | §3.2 |
| **H1** stage-1 arch: 8 x 8 clusters, RedMulE 32 x 32 | The configuration every stage-1 kernel time is measured on. | §2.1 |

PR1, its arch and its results (`pr1.*`, `arch.pr1.*`, `docs/analysis/crosscheck.*`) are not changed by
this specification. The `libero-2cam` composed estimate from the kernel-repo arch (79.2 ms) was not a
stage-1 reference and has been removed from the repo: its operator-graph page and its `hand_kernels.*`
times in `constants.yaml` (last present at `dcd221d`). The `libero-2cam` workload entry stays, only as
the reference for PR1's cross-check.

--------------------------------------------------------------------------------

## 2. Hardware configuration

### 2.1 H1: stage-1 arch (owner spec, 2026-11-19)

The **owner sets** the mesh, matrix unit, TCDM, Spatz and HBM rows. The owner confirmed the rows
**carried over** from the kernel-repo arch (§2.3). Peaks and totals are computed from the
rows (inference).

| Part | Setting | Set by |
| --- | --- | --- |
| Mesh | **8 x 8 = 64 clusters** | owner |
| Matrix unit | **RedMulE 32 x 32 PE array** per cluster: 2,048 FLOP per cycle, 2.048 TFLOP/s per cluster at 1 GHz, **131.1 TFLOP/s** in total (fp16) | owner; peak inference |
| Cores | **4 Spatz vector units, each attached to one Snitch core**, plus one Snitch data-mover core (iDMA): 5 cores per cluster | owner (Spatz); data mover carried over, confirmed |
| Spatz | 4 FPUs, VLEN 16384 bit per unit; 128 FLOP per cycle per cluster, 8.2 TFLOP/s in total, **not counted in peak** | carried over; throughput inference (same formula as [`hw-config.md`](../analysis/hw-config.md)) |
| L1 (TCDM) | **384 KiB per cluster** for now, 24 MiB in total. **Design range: 128 to 512 KiB per cluster** (8 to 32 MiB in total), to be explored. | owner |
| Clock | 1 GHz | carried over, confirmed |
| Instruction memory | 64 KiB per cluster | carried over, confirmed |
| NoC | 1024-bit links, 128 GB/s per direction | carried over, confirmed |
| HBM | **16 channels on the west and south edges (8 each; `hbm_chan_placement = [8,0,0,8]`, order west, north, east, south)**, DRAMSys JEDEC HBM2E-4000, 64 GB/s per channel, **1,024 GB/s** in total | owner (channels, edges); per-channel bandwidth carried over (inference, as in [`hw-config.md`](../analysis/hw-config.md)) |
| Balance | 131.1 TFLOP/s / 1,024 GB/s = 128 FLOP per byte | inference |
| Data type | fp16 for activations, weights and outputs | |

**Compared with the kernel-repo arch** (inference; both at 64 GB/s per channel, from DRAMSys's report):

| | Kernel-repo arch | H1 | Change |
| --- | --- | --- | --- |
| Peak fp16 | 16.4 TFLOP/s | 131.1 TFLOP/s | x8 |
| HBM bandwidth | 512 GB/s (8 channels) | 1,024 GB/s (16 channels) | x2 |
| HBM bandwidth per cluster | 32 GB/s | 16 GB/s | x0.5 |
| Balance | 32 FLOP/B | 128 FLOP/B | x4 |

W1's FLOP per weight byte (§3.2, stage-1 lowering) is vision 1,244, prefix 253 and expert 57. On H1 the
expert falls below the balance point, so it is limited by weight bandwidth: about 1,968 MB of weights
per call is about 1.9 ms at 1,024 GB/s. On the kernel-repo arch it was above. These ratios count each
weight once and ignore activation traffic and tiling re-reads (inference).

**Memory model.** H1 is simulated with **DRAMSys** HBM, not with the ideal HBM PR1 used. Every
stage-1 time states this.

**Arch file.** Drafted as `implementation/config/arch/arch_h1.py` on the kernel repo branch `gelu-layernorm-kernels`, uncommitted. It differs from `arch.py` in mesh, RedMulE width, HBM placement and HBM controller grouping, and passes the simulator's HBM and mesh assertions (`flex_cluster.py:86-99`), checked statically. HBM grouping (owner, 2026-11-19): one controller per edge with 8 channels (`num_node_per_ctrl = 8`, `hbm_node_aliase = 8`, XOR scrambling on), so a tensor placed on an edge is spread over all 8 channels of that edge and each cluster row reaches HBM through its own edge node. With the earlier 2 x 4 grouping a contiguous tensor reached only 4 channels.

**Build.** H1 has not been built yet. The next task is a build check: the
8 x 8 mesh, RedMulE 32 x 32, TCDM banking with the Spatz ports at 384 KiB, and the 64 KiB instruction
memory. Then one kernel per family is run.

**Consequence for existing times.** All existing hand-kernel times were measured on the kernel-repo
arch (§2.3). It differs from H1 in mesh (4 x 4) and RedMulE (32 x 16). Those times are **not** H1
times, and the §4.1 reuse rule does not apply to them across arches. They stay as reference values
labelled with their arch.

### 2.2 Thor alignment: not in stage 1

H1 has 131.1 TFLOP/s, about half of Thor's 258 TFLOP/s dense FP16 (`thor.fp16_dense`). It is **not
aligned with Thor for now**: stage 1 builds the baseline first. The earlier 16 x 16 iso-peak proposal
([`hw-config.md`](../analysis/hw-config.md)) is not stage-1 hardware.

### 2.3 Other arches (not stage-1 hardware)

| Arch | Where it appears | Setting |
| --- | --- | --- |
| Kernel-repo arch | `arch.stage1` in `constants.yaml` (its `hand_kernels.*` times were removed, last at `dcd221d`) | 4 x 4 clusters, RedMulE 32 x 16, 384 KiB TCDM, Spatz on 4 of 5 cores, 8 HBM channels (4 west, 4 south), DRAMSys; 16.4 TFLOP/s (`~/softhier@52859c3 implementation/config/arch/arch.py`) |
| PR1 arch | W0 only, as the setting of PR1's run | 4 x 4 clusters, RedMulE 128 x 32, 3 cores, Spatz off, 1 MiB TCDM, ideal HBM; 131.1 TFLOP/s |

Numbers from these arches are never mixed with H1 numbers without saying so.

--------------------------------------------------------------------------------

## 3. Workload configurations

Both checkpoints have the same model dimensions: `lerobot/smolvla_base@5e8d12a6` and
`lerobot/smolvla_libero@a24f1f38` give identical `Dims` in `smolvla_ops.load_dims`. The workloads differ
only in their inputs.

### 3.1 W0: `pr1-1cam` (infrastructure and debugging)

W0 is the configuration PR1 ran end to end with softhier-mlir. It is kept to bring up and debug the
SoftHier setup. Stage 1 does not require matching it on H1.

| Input | Value |
| --- | --- |
| Checkpoint | `lerobot/smolvla_base@5e8d12a6` |
| Cameras | 1, at 256 x 256 |
| Tokens | 256 vision, 16 image, 48 language, 1 state; 65 prefix (padded to 128 in PR1's program) |
| Chunk, flow steps | 50, 10 |
| PR1 result | `pr1.e2e_ms` 120.7 ms, PR1 arch, ideal HBM, measured |

### 3.2 W1: `vlabench-3cam` (stage-1 target)

The workload is the observation format of **VLABench** (OpenMOSS; wiki
`wiki://resources/benchmarks/vlabench.md`).

| Input | Value | Source |
| --- | --- | --- |
| Cameras | **3**: `image`, `second_image`, `wrist_image` | dataset `VLABench/vlabench_primitive_ft_lerobot` (and `..._lerobot_video`), `meta/info.json`, read 2026-11-19 |
| Camera resolution | 480 x 480 RGB | same |
| Robot, rate | Franka Panda, 10 fps; state 7, action 7 | same |
| Model input size | each image resized with padding to **512 x 512** | SmolVLA config `resize_imgs_with_padding: [512, 512]` |
| Checkpoint | `lerobot/smolvla_base@5e8d12a6`: three camera inputs (`camera1-3`). No SmolVLA checkpoint fine-tuned on VLABench is assumed; only shapes matter here. | fixture `config.json` |
| State, action width | padded to 32 (`max_state_dim`, `max_action_dim`), so shapes do not change for 7-dof | fixture `config.json` |
| Language | 48 tokens (`tokenizer_max_length` 48, padded to max length) | fixture `config.json` |
| Chunk, flow steps | 50, 10 | fixture `config.json` |

**Derived token counts** (inference, from `smolvla_ops`):

| | Value |
| --- | --- |
| Vision tokens | 1024 per camera, encoder run once per camera |
| Image tokens after pixel shuffle | 3 x 64 = 192 |
| Prefix tokens | 192 + 48 + 1 = **241** |
| Expert self-attention keys | 241 + 50 = 291 |
| Expert cross-attention keys | 241 |

**Per-stage size** (inference, standalone model, stage-1 lowering of §3.3):

| Stage | GFLOP | Weight MB read per call |
| --- | ---: | ---: |
| vision (3 encoder calls) | 645.4 | 519 |
| connector | 4.5 | 24 |
| prefix | 79.6 | 315 |
| expert (10 steps) | 107.5 | 1,968 |

`workload-modeling` already had a workload with exactly these shapes: `release-3cam` (smolvla_base,
3 cameras at 512 x 512, 48 language tokens). `vlabench-3cam` is now a workload ID of its own and the
primary one (2026-11-20).

**Variant W1-b: cameras batched** (a §3.3 rewrite). The vision encoder runs once over all 3 x 1024 tokens, so every
per-token product has M = 3072 and the encoder weights are read once per call, not three times.
Attention still runs per camera. Vision weight bytes fall from 519 MB to 173 MB; FLOP are unchanged.
It is model option `Lowering(batch_cameras=True)`. W1-b is reported next to W1, never in place of it.

### 3.3 Lowering: shapes the operators are counted and measured at

A *lowering* fixes the matrix and attention shapes of each operator. It is option `Lowering` in
`smolvla_ops`. The **reference lowering** is the computation of lerobot's PyTorch SmolVLA:

*   true token counts;
*   cross-attention K/V projected again in every flow step;
*   the time MLP as one 50 x 1440 x 720 product;
*   one vision-encoder call per camera.

**The stage-1 lowering** is the reference lowering with one rewrite: cross-attention K/V projected
**once per call**. The prefix does not change during the 10 flow steps, so this rewrite is exact.
Model option: `Lowering(kv_proj_per_step=False)`.

**Rewrites.** Every shape-changing rewrite is recorded here with its effect on W1. A stage-1 number
states which rewrites it includes. Effects are inference, from the standalone model, each applied on
top of the stage-1 lowering.

| Rewrite | What changes | Effect on W1 | Exact? | Stage 1 | Origin |
| --- | --- | --- | --- | --- | --- |
| Cross K/V once per call | expert `cross_k/v_proj` 241 x 320 x 320: count 80 -> 8 | expert 114.6 -> 107.5 GFLOP; weights read 1,997 -> 1,968 MB | yes | **in the stage-1 lowering** | PR1 (softhier-mlir) |
| Cameras batched (W1-b) | vision per-token products and row operators at M = 3072, one call; attention stays per camera | vision weights read 519 -> 173 MB; FLOP unchanged | yes | reported variant (§3.2) | owner, 2026-11-19 |
| Time-MLP split | `concat(action, time) @ W = action @ W_a + time @ W_t`. The action half is 50 x 720 x 720; the time half is one row, the same for all 50 tokens, folded into a per-step bias on the host. | expert 107.5 -> 107.0 GFLOP; one device product fewer per step | yes | recorded, not applied | PR1 (`softhier-mlir@b11cd5f frontend/smolvla_expert.py`) |
| Pad to tile multiples | prefix rows 241 -> 256; expert keys aligned to 32 (self 291 -> 320, cross 241 -> 256) (PR1's padding; a kernel's own padding depends on its tile or block) | prefix 79.6 -> 84.7 GFLOP; expert 107.5 -> 108.2 GFLOP | no: adds work | **required by today's SummaGEMM and FlatAttention** for correct output (§4.2); the model rows stay unpadded, and each measured time includes its kernel's padding, with the ratio recorded (§4.3) | PR1 (`runtime/sh_expert.inc.c:83`); kernel code |

New rewrites are added to this table before a number that uses them is reported.

--------------------------------------------------------------------------------

## 4. Hand-written kernel requirements

### 4.1 Kernels and the operators they must cover

Every device operator of W1 maps to one of six kernels in the kernel repo's `implementation/sw/`. Host
operators (`lang_embed_scale`, `state_proj`, `bias_state`) are out of scope.

| Kernel | Operators of W1 | Unique shapes | Same shape measured on the kernel-repo arch |
| --- | --- | ---: | ---: |
| SummaGEMM | every matrix product | 18 | 12 |
| FlatAttention | QK^T, softmax and P.V in one call; K transposed on chip | 4 | 1 |
| Eltwise | bias, residual add, RoPE, scale, Euler axpy, pixel shuffle | 10 | 6 |
| Activation | GELU (tanh form), SiLU, SiLU x gate | 4 | 3 |
| LayerNorm | SigLIP ln1, ln2, ln_post | 1 | 1 |
| RMSNorm | VLM and expert rmsnorm | 2 | 1 |

**All 39 shapes, plus the W1-b shapes, need runs on H1.** The last column counts shapes identical to
a `libero-2cam` entry in `constants.yaml` (`hand_kernels.*`, removed since; see `dcd221d`): the per-camera vision shapes and the
expert shapes with M = 50. Those were taken on the kernel-repo arch (4 x 4, RedMulE 32 x 16, §2.3),
not on H1, so they are reference values only. On H1 the tile and block sweeps are redone: an 8 x 8
mesh changes the SUMMA and FlatAttention group scale, and RedMulE 32 x 32 changes the best tiles.
Some of those runs, the SummaGEMM, FlatAttention and RMSNorm ones, were timing only, with placeholder
data.

**Reuse rule (within H1).** A time measured on H1 at the same kernel, parameters and shape is valid
for any stage-1 workload once that kernel's numerical correctness check exists and passes (§4.3,
item 6):

1.  If the time was taken before the check existed, re-run the recorded setting (tile or block, mode,
    padded shape) with real inputs and the golden.
2.  The check passes and the time is within the 3 % noise floor (rule 7) of the recorded value: reuse
    the recorded time.
3.  Otherwise: re-measure, re-sweeping if the kernel changed.

A time is never reused across arches. Shapes H1 needs, by group:

*   vision (1024 rows per camera): all products, attention 12 x 1024 x 1024 x 64, bias, residual,
    LayerNorm, GELU;
*   connector: the 192 x 960 x 12288 projection, pixel shuffle of 3 cameras, embed scale;
*   prefix (241 rows): q/o, k/v, gate/up, down projections; RoPE, residual, RMSNorm; SiLU x gate at
    241 x 2560;
*   expert (50 rows): all products including cross K/V 241 x 320 x 320, row operators;
*   attention (unmasked): prefix 15 x 241 x 241 x 64, expert self 15 x 50 x 291 x 64, expert cross
    15 x 50 x 241 x 64;
*   variant W1-b: vision products at M = 3072 (3072 x 768 x 768, 3072 x 3072 x 768, 3072 x 768 x 3072)
    and the vision bias, residual, LayerNorm and GELU rows at 3072 rows.

### 4.2 Per-kernel requirements

| Kernel | Coverage required | Functional gaps to close | Shape and padding limits |
| --- | --- | --- | --- |
| **SummaGEMM** | Every matrix product: M of 50, 192, 241 and 1024 rows | **Output check: added 2026-11-19, bit-exact.** Real fp16 inputs on the padded shape (padding zero-filled), compared on the unpadded region after the timer stops. The golden replays the RedMulE model's arithmetic (`light_redmule.cpp`, `matmul_fp16`, with GVSoC patches 0007 and 0008): within each K tile, products are added one k at a time in fp32 on top of the fp16 partial sum, which is rounded to fp16 (round to nearest even) once per tile, in ascending K order. Tolerance 0 ULP; on H1, 512 x 512 x 512 (tiles 128, 4 x 4 groups) gives 0 mismatches. A golden rounded once at the end left 312 mismatches (up to 4,485 ULP near zero), and one summed in BLAS order left 12, so the per-tile replay is required. Before patches 0007 and 0008, the unpatched tree gave about 1,600 corrupt elements (up to 44,528 ULP). **Edge-tile handling: not in stage 1**; shapes are padded instead. | **Padding is mandatory.** The kernel has no edge handling: it runs ceil(M / (scale x `m_tile`)) x ceil(N / (scale x `n_tile`)) x ceil(K / `k_tile`) full tiles, and every DMA and RedMulE call uses full tile sizes (`SummaGEMM.h:142-144, 180-182, 443`). An unpadded K or N tail reads or writes across row boundaries and corrupts valid outputs; an M tail writes past the end of Z. So tensors live in HBM at padded sizes, with K padding zero-filled in X and W. The padded work is at most 2.5x the nominal work. Small-M shapes (50, 192, 241) are where padding hurts most; record the ratio for each. |
| **FlatAttention** | QK^T, softmax (on Spatz), P.V; GQA with 15 query heads over 5 KV heads; d = 64 | **Masking: not in stage 1.** Stage-1 times are unmasked, as an approximation: same FLOP, but padded and masked keys are not excluded. The reference mask (§4.2.1) is recorded for later. **Output check (required before reuse)**: real inputs and a golden of the same padded, unmasked computation, so it checks the kernel, not the model; it equals the model's result only where no padding is needed (vision, 1024 x 1024). | **Padding is mandatory.** The block counts are floor divisions (`Tr = Sq / Br`, `Tc = Skv / Bc`, `FlatAttentionUtil.h:234-235`), so a tail that is not a block multiple is silently dropped. Sequences are padded up to a block multiple; 241 is prime, so no block size divides it. Padded keys still enter the softmax unless masked, so unmasked runs on padded lengths are valid for timing only. Record the padding ratio. Odd head counts per group: use collective mode, or pad the heads and say so. |
| **Eltwise** | bias, add, RoPE (head_dim 64), scale, axpy, pixel shuffle (3 cameras, 4 x 4 shuffle) | none known; bit-exact against the replay golden today | exact shape; one row per cluster at a time |
| **Activation** | GELU (tanh), SiLU, SiLU x gate | none known; bit-exact today | exact shape |
| **LayerNorm** | SigLIP LayerNorm, eps 1e-6 | The ISS leaves some vector instructions (`vfwadd.wv`, `vfwsub.vf`, `vmv.*`) untimed, so the time is optimistic. Record this with the value. | exact shape |
| **RMSNorm** | VLM and expert RMSNorm | **Output check (required before reuse)**: upstream kernel, placeholder data today. Uses one core per cluster, so a Spatz version is wanted for RQ1. | exact shape |

#### 4.2.1 Attention mask in the reference code (for later; not a stage-1 requirement)

Read from a local copy of lerobot's SmolVLA, `~/.cache/smolvla/modeling_smolvla.py`; the lerobot
revision was not recorded.

*   `embed_prefix` gives image and language tokens `att_mask` 0 and the state token 1.
*   `embed_suffix` gives each of the 50 action tokens 1.
*   `make_att_2d_masks` combines these with `pad_masks`.
*   The suffix mask is `concat(prefix pad mask, make_att_2d_masks(suffix))`.

From this, assuming the usual cumulative-sum rule of `make_att_2d_masks`, which was not read (inference):

| Attention | Query to key rule |
| --- | --- |
| Prefix (241 x 241) | Image and language tokens see each other but not the state token; the state token sees every prefix token. Padded language keys are masked. |
| Expert self (50 x 291) | Action tokens see every valid prefix key, and earlier and current action tokens (causal over the chunk). |
| Expert cross (50 x 241) | Action tokens see every valid prefix key (key padding only). |

The FlatAttention mask therefore needs two parts:

*   a key-padding vector, which also covers keys padded to a block multiple;
*   one block boundary: the state token in prefix, and the causal action block in expert self-attention.

### 4.3 Measurement protocol

A kernel time counts as a stage-1 `measured` value only if all of the following hold.

1.  **Hardware and simulator.** H1 (§2.1): one committed H1 arch file
    (`infra/kernel-runs/arch/arch_h1.py`), the same for every run, with DRAMSys HBM
    (`SOFTHIER_IDEAL_HBM` unset). The TCDM size is recorded with each time. The simulator is the
    project's patched GVSoC: upstream `soft_hier_release` plus the patch series in
    `infra/softhier/gvsoc-patches/` (expected tree in `gvsoc-patches/expected`), built for that arch
    with `infra/softhier/build_arch.sh`. Every run goes through `infra/softhier/require_sim.sh`, which
    refuses unless `stamp.py check <arch>` passes, and copies the stamp into the run folder. The tree
    comes only from `SOFTHIER_HOME` (`infra/env.sh`): the worktree's own `third_party/gvsoc` if
    initialised, otherwise one shared tree at `$SHVLA_DATA/gvsoc` used by all worktrees, under a lock
    so one build or run uses it at a time. The kernels come from the kernel repo (its
    `implementation/` folder); only the simulator and its SDK runtime come from the patched tree.
    All stage-1 kernel times before 2026-11-19, including the 79.2 ms composed estimate, were taken on
    the kernel repo's own tree (`softhier_deeploy` 52859c3), which lacks the iDMA fixes (patches 0003,
    0005: hangs and lost transfers) and the RedMulE fp16 fixes (0007, 0008: sparse corrupt elements).
    Patch 0011 (2026-11-19) zero-initialises the RedMulE model's buffers: each cluster's first RedMulE
    job used to multiply uninitialised heap, which showed as sparse wrong blocks on 8x8 SUMMA groups
    and passed elsewhere only because fresh pages are zero (found with valgrind memcheck). Runs before
    0011 are not reused. `require_sim.sh` exports `MALLOC_PERTURB_=85` so a model that reads
    uninitialised heap fails the output check the same way every run. The measurement run calls gvsoc
    directly with no trace flag (`make run` adds `--trace=/chip/cluster_0/redmule`).
2.  **Timer region.** Core 0 of cluster 0 calls `flex_timer_start` / `flex_timer_end` around the kernel
    call only, between global barriers: a global barrier sits directly before `flex_timer_end`, so the
    time ends when the slowest cluster is done. Before 2026-11-19, SummaGEMM, Eltwise, Activation,
    LayerNorm and RMSNorm stopped the timer when cluster 0 finished (no barrier before
    `flex_timer_end`); their earlier times, including PR1's, are cluster 0's time and read low. On H1,
    Eltwise add 1024 x 768 timed 15.55 us that way, while the slowest cluster finished at 21.5 us in
    the trace. FlatAttention already had the barrier. The value is the simulator's `[Performance Counter]` line in ns.
    Data starts and ends in HBM; preload and boot are excluded.
3.  **HBM placement.** Every kernel preload uses one shared layout (`scripts/utils/hbm_layout.py` in the
    kernel repo) and is written as a 64-bit ELF. Rule (owner, 2026-11-19): only the edge matters, since
    each edge's controller scrambles a tensor over its 8 channels.

    | Kernel | West | South |
    | --- | --- | --- |
    | SummaGEMM | input X, output Z | weights W, golden |
    | FlatAttention | Q, output O | K, V, golden |
    | Eltwise | input, golden | second operand Z, output |
    | Activation | input, golden | gate, bias, output |
    | LayerNorm | input, golden | gamma, beta, output |
    | RMSNorm | input, golden | output |

    RedMulE kernels keep the original layout (activations west, weights south, output with the input).
    Spatz kernels put the output on south too, so reads and writes use both edges. The golden sits on
    whichever edge the output does not use; it is read only after the timer stops. Why the shared
    layout: the original preloads were 32-bit ELFs, so arrays at or above 4 GiB (every south address)
    were truncated onto west addresses. The Spatz kernels therefore kept all arrays on west, and the
    SummaGEMM and FlatAttention weights were preloaded onto X or Q (their old runs used placeholder data).
4.  **One operator instance per run.** The composed time multiplies each time by the operator's
    `count`.
5.  **Sweeps** (SummaGEMM tiles, FlatAttention blocks and modes):
    *   the sweep driver and its raw results (every setting tried, with its time) are committed in the
        kernel repo or here, so any pick can be re-run;
    *   the candidate rule is written down: tile >= 32, padded work <= 2.5x nominal;
    *   the pick is the fastest valid setting. The constants entry records the setting, the number of
        settings tried, their min-max range and the padding ratio.
6.  **Checks.**
    *   Output (**required**): real inputs and a golden, compared on the unpadded region. The check is
        **structural** (owner, 2026-11-20): its job is to prove the run did its whole problem in the right
        place (no dropped heads or rows, no zeroed, NaN or misplaced tiles), because those errors change
        the time; GVSoC timing does not depend on the data values. A float64 reference of the kernel's own
        formula with a stated fp16 tolerance is enough. Bit-exact replays already in place stay
        (Eltwise, Activation, LayerNorm, SummaGEMM), but new checks do not replay the kernel's arithmetic
        (RMSNorm: 32 ULP; FlatAttention: absolute 0.02). Purely numerical kernel defects that leave the
        work unchanged (RMSNorm without weight and eps) are recorded, not fixed. A timing-only run, with
        placeholder data, is not a stage-1 measured value.
    *   Traffic: DRAMSys bytes moved are at least the padded problem's minimum. This proves the run did
        its whole problem. Requests per HBM channel are counted once per kernel (GVSoC trace on the
        HBM controllers and channels) to confirm the run reaches the channels the placement rule
        intends.
    *   Monotonic: a contained GEMM shape is not slower than the larger one.
7.  **Provenance.** The constants entry (section `h1_kernels`, written by
    `infra/kernel-runs/to_constants.py`) carries the label `measured`, `workload: vlabench-3cam`,
    `arch: h1` (constants section `arch.h1`), the kernel-repo commit and build hash, the simulator stamp (`sim_stamp.json`: GVSoC tree, patch
    series, arch SHA-256), and the exact kernel parameters in `source`.
8.  **Noise.** Code placement alone moves per-call times by about 3 %. A difference smaller than that is
    not reported as a result.
9.  **Trace for inspection.** For each shape, the recorded run, i.e. the picked setting after a sweep, is
    also run with GVSoC tracing on RedMulE, iDMA, Spatz and the cluster registers (`make runv`) and
    converted to a Perfetto trace (`make pfto`). The trace JSON and the raw log are kept per shape
    under `~/.cache/softhier-vla/stage1-h1/<kernel>/<shape>/`, outside git because of their size.
    The constants entry names that path. Sweep candidates keep their time only, no trace. Tracing
    must not change the timer value: checked on the first H1 run. Open simulator issue (2026-11-19):
    with any trace enabled (RedMulE, `hbm_ctrl`), a run with an 8x8 SUMMA group livelocks or advances
    only very slowly, so shapes whose picked setting uses an 8x8 group have no trace until it is fixed.
    A 4x4 SUMMA group (241x2560x960, split-N scale 4) also failed: 81 us simulated in 15 min for a
    27.6 us kernel. For the 26 row kernels and nine GEMM shapes that were traced, the traced timer
    equals the untraced one.
10. **NoC reductions (patch 0012, 2026-11-20).** GVSoC's collective reduction (FlooNoC,
    `collective_generate`) folded each kid request into a parent buffer that never started at the
    identity, so every router where a reduction forks added uninitialised heap once. Found with
    FlatAttention: with `sx > 1` every output element was off by the same constant ((sx - 1) x `0xAAAA`
    under `MALLOC_PERTURB_`); `sx = 1` cases passed. Patch 0012 fills the parent with the identity and
    fixes the fp16 subnormal conversion in the same file; data only (the two row-wise debug cases keep
    their timer values to the ns). FlatAttention runs before 0012 are not reused; SummaGEMM (Nosplit,
    split-N) and the row kernels use no reductions. Still open after 0012: FlatAttention async mode with
    `sx = 4, sy = 8` (12x1024x1024x64, blocks 1024x512) gives 11 % wrong elements; `sx = 8, sy = 4` async
    and all collective-mode candidates pass. Such candidates are dropped by the output check.
11. **One build for recorded values.** Code outside the timed region (the output check) moves the
    timer, about 0.75 % on one GEMM shape. Sweep candidates may span builds; the recorded run of each
    picked setting is re-run on the final kernel build and simulator stamp (`final_runs.py`; the kernel
    build is the worktree diff against its commit, kept as `kernel_build.diff` with its hash in each row).
    Exception: the nine M = 50 expert GEMM picks of `gemm_smallm_sweep.py` are recorded on kernel commit
    `de53537`, the frozen build plus an optional absolute-tolerance GEMM check (`GEMM_CHECK_ABS_TOL`, for
    splitK). Without that option the binary is the same (a frozen-build pick re-runs at 32,353 ns,
    unchanged); the splitK picks use it, so their times may carry the ~1-2 % placement effect.

### 4.4 Composed estimate

The stage-1 composed estimate for W1 on H1 is the sum over operators of `count` x the measured time
per call, as `render.kernel_sums` computes it. It is labelled **estimate** and states its assumptions:
no overlap between kernels, no layout change between kernels, every kernel starting and ending in HBM.

**Result (2026-11-20, estimate).** W1: **29.0 ms** per policy call (GEMM 58 %, attention 17 %, row
operators 24 %); W1-b: 27.9 ms. The first composition gave 35.2 ms and 34.1 ms; the small-M GEMM sweep
(§5) cut the expert GEMMs from 13.6 to 7.4 ms. Per operator and stage:
[`smolvla-op-graph.md`](../analysis/smolvla-op-graph.md) (W1) and
[`smolvla-op-graph-w1b.md`](../analysis/smolvla-op-graph-w1b.md) (W1-b), generated from
`constants.yaml` (`h1_kernels`) by `workload_modeling.build`. Every operator outside the host prepare
step has a measured kernel time; none is missing.

--------------------------------------------------------------------------------

## 5. Decisions and open questions

Decided by the owner (2026-11-19):

*   **W0** is kept for infrastructure and debugging only; no H1 match is required.
*   **Masking:** stage-1 attention times are unmasked.
*   **Camera batching:** W1-b is a reported variant.
*   **Padding:** SummaGEMM and FlatAttention shapes are padded to tile or block multiples. The
    hand-written kernels are not changed for edge tiles in stage 1.
*   **Reuse:** an H1 time is reused after the numerical correctness check is added and passes (§4.1
    reuse rule). Times from the kernel-repo arch are not reused on H1.
*   **Hardware (H1):** 8 x 8 clusters, RedMulE 32 x 32, 4 Spatz units each on one Snitch core, TCDM
    384 KiB within a 128-512 KiB design range (§2.1).
*   **Lowering:** the reference lowering plus cross K/V once per call (§3.3). Every other rewrite is
    recorded in §3.3 and stated whenever a number uses it.

*   **HBM:** 16 channels on the same edges as the kernel-repo arch: 8 west, 8 south; one controller
    per edge spreading a tensor over its 8 channels (§2.1).
*   **HBM placement:** the shared rule in §4.3 item 3: inputs west, second operands south, Spatz
    kernel outputs south, RedMulE kernel outputs west.
*   **Thor alignment:** none for now; the baseline comes first.
*   **Carried-over rows:** clock, instruction memory, NoC and the data-mover core are confirmed.

Open questions from the H1 baseline (2026-11-20):

*   **Expert GEMMs: group shape and splitK (resolved 2026-11-20).** The first picks for the nine M = 50
    expert shapes used 8 or 16 of 64 clusters (`splitN_s2g2`, `splitN_s2g4`), a limit of the search:
    `gemm_sweep.py` allowed only square SUMMA groups and no splitK, so scale 4 or 8 padded 50 rows to at
    least 128. Non-square groups are safe when M fits one group step (`M_iter = 1`): `SummaGEMM.h:165`
    steps X along M by `summa_group_x` instead of `summa_group_y`, which matters only for `M_iter > 1`.
    Upstream's `opt()` (soft_hier_je `gemm_auto.py`, used by its DeepSeek decode flow with M = 1) relies on
    this: one-cluster-tall groups up to 8 wide, the rest of the mesh filled by splitK or splitN groups.
    `gemm_smallm_sweep.py` measured such groups (one tile per group structure; full-mesh pass, 82
    candidates): every expert shape got faster, 1.1x to 2.4x, with groups one or two clusters tall and 4
    or 8 wide (splitK, or splitN over 16 to 32 groups); expert GEMMs 13.6 -> 7.4 ms. These picks are the
    recorded values (§4.3 item 11). Measured limit of SummaGEMM splitK: the output is correct only when
    the groups fill the mesh and each group takes at most 2 N steps (`N_pad / (scale_x n_tile) <= 2`);
    33 of 33 such runs pass, the other 42 splitK runs fail the check. Probable cause, not verified in
    the code: only group 0 stores and clears its Z buffers, so the other groups' two Z buffers carry
    stale partial sums from a third N step on, and with idle clusters the mesh-wide NoC reduction mask
    takes in clusters outside the groups.
*   **Tile search is not exhaustive.** Per GEMM shape about 10 settings were run (9 to 12): the top of
    a rough cost model, K-tile variants of its best, the best of each scale and upstream `opt()`. The
    pick is the fastest measured setting, not proven best. The model does not separate K tiles, so
    candidates outside its top pick were run with whatever K tile it ranked first (the 64-cluster
    `splitN_s2g16` of 50 x 4096 x 720 only at k_tile 32, where k_tile 256 was 1.8x faster on `s2g4`).
    For M = 50 the small-M sweep used one fixed tile per group structure (largest k_tile within 6 % of
    the least padded work); tiles inside a structure are not swept. Next step decided by the owner
    (2026-11-20): a cost model for the analysis instead of wider sweeps.
*   **FlatAttention async with `sx = 4, sy = 8`** gives wrong output (§4.3 item 10); cause not found.
