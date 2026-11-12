# Stage-1 conventions

How the stage-1 documents name workloads, label numbers, count bytes and compare with Jetson Thor.
Numbers cite a key in [`infra/workload-modeling/constants.yaml`](../../infra/workload-modeling/constants.yaml)
or state their workload, arch and label.

## Workloads

| ID | Checkpoint | Cameras | Image | Tokens (vision per camera / prefix) | Role |
| --- | --- | --- | --- | --- | --- |
| `libero-2cam` | `lerobot/smolvla_libero@a24f1f38` | 2 | 512x512 | 1024 / 177 | primary |
| `pr1-1cam` | `lerobot/smolvla_base@5e8d12a6` | 1 | 256x256 | 256 / 65 | PR1's measured configuration |
| `release-3cam` | `lerobot/smolvla_base@5e8d12a6` | 3 | 512x512 | 1024 / 241 | released config; no RQ uses it |

Prefix tokens = cameras x vision tokens / 16 (pixel shuffle, scale factor 4) + 48 language + 1 state.

## Arches

| Name | What | Peak FP16 |
| --- | --- | --- |
| `stage1` (16-cluster) | kernel repo `~/softhier@52859c3`, `implementation/config/arch/arch.py`, as configured | 16.384 TFLOP/s |
| `stage1-256` (256-cluster) | the same cluster, 16 x 16 mesh, sized to Thor | 262.144 TFLOP/s |
| `pr1` | PR1's arch file (RedMulE 128x32, Spatz disabled) | 131.072 TFLOP/s |

See [hw-config.md](hw-config.md).

## Evidence labels

`measured`, `estimate`, `inference`: the wiki's set (`softhier_vla_wiki/tools/wiki.py:88`), so rows copy
into wiki status entries unchanged. Measured: a reported run. Estimate: computed from specs, configs or
partial runs. Inference: a conclusion drawn from other numbers, including model output. A community
figure that was not independently checked keeps its label and says "self-reported" in `note`.

**Inputs read as given** (an arch-file parameter, a checkpoint config value, a vendor spec-sheet
figure such as Thor's 273 GB/s) are labelled `measured`, with the file and line or the spec as source.
They are read, not computed, and the wiki has no separate label for them. This extends the glossary's
"a reported run", which covers results, to inputs (owner decision, stage 1).
A spec figure a paper derives (Thor's dense FP16 258) is `inference`.

## Operator rows and byte columns

The standalone model (`infra/workload-modeling/workload_modeling/smolvla_ops.py`) writes one row per
operator group. All bytes assume fp16 (2 B per element); stored checkpoint dtypes (BF16, some FP32)
are ignored.

| Column | Meaning |
| --- | --- |
| `shape` | one instance: (M, N, K) for matrix products; (heads, Sq, Skv, head dim) for attention; (rows, cols) otherwise |
| `count` | instances per call: layers x flow steps x cameras that apply |
| `flop` | all instances |
| `bytes_weight` | weight bytes of one instance x `count` |
| `bytes_unique` | (inputs + outputs + weights) of one instance x `count` |
| FLOP per weight byte | `flop / bytes_weight`; empty for operators without weights |

**How bytes scale with `count`:**

* *Layers:* each layer has its own weights, so weight bytes scale with layers.
* *Heads:* they share one weight matrix and are folded into the shape, so weight bytes do not
  scale with them.
* *Cameras:* the reference code runs the vision encoder once per camera (lerobot `embed_prefix`
  loops over cameras; PR1 does the same), so vision weight bytes scale with cameras. Batching all
  cameras into one encoder call (`Lowering(batch_cameras=True)`) reads them once and doubles vision's
  FLOP per weight byte for two cameras; it is an optimization, not the default.
* *Flow steps:* the expert's weights (about 200 MB per step) do not fit on chip, so they are read again
  in every step and weight bytes scale with steps.

**Formulas per class** (`h` heads, `hkv` KV heads, `d` head dim, `Sq`/`Skv` query/key length):

| Class | FLOP | Weight bytes | Unique bytes |
| --- | --- | --- | --- |
| gemm (M x K) . (K x N) | 2 M N K | 2 K N | 2 (M K + M N) + weight |
| attn: QK^T | 2 h Sq Skv d | 0 | 2 (h Sq d + hkv Skv d) |
| attn: P.V | 2 h Sq Skv d | 0 | 2 (hkv Skv d + h Sq d) |
| softmax (rows x cols) | 5 per element | 0 | 2 x 2 x elements (the score matrix, in and out) |
| LayerNorm (rows x n) | 8 per element | 2 x 2 n | 2 x 2 x elements + weight |
| RMSNorm (rows x n) | 4 per element | 2 n | 2 x 2 x elements + weight |
| GELU (tanh form) | 8 per element | 0 | 2 x 2 x elements |
| SiLU | 4 per element | 0 | 2 x 2 x elements |
| SiLU x up | 5 per element | 0 | 2 x 3 x elements |
| RoPE | 3 per element | 0 | 2 x 2 x elements |
| residual add | 1 per element | 0 | 2 x 3 x elements |
| bias add | 1 per element | 2 N | 2 x 2 x elements + weight |
| scale | 1 per element | 0 | 2 x 2 x elements |
| Euler update (axpy) | 2 per element | 0 | 2 x 3 x elements |
| data (transpose, pixel shuffle) | 0 | 0 | 2 x 2 x elements |

Bias adds are separate rows, as in PR1's programs, so matrix-product FLOP compare directly with
RedMulE work. The per-element counts for non-matrix classes are conventions (inference); they are
small next to the matrix products. On SoftHier the cost of these operators is set by core issue rate,
not FLOP (PR1 PROFILE), which RQ1 measures.

**Worked example (expert MLP gate/up, `pr1-1cam`):** M = 50 action tokens, K = 720, N = 2 x 2048.
One instance: 2 x 50 x 720 x 4096 = 294.9 MFLOP and 2 x 720 x 4096 = 5.90 MB of weights. Count =
16 layers x 10 steps = 160. Totals: 47.19 GFLOP, 943.7 MB of weights, 50 FLOP per weight byte (= M,
as for any M-row product). If weights were not re-read per step the figure would read 500. The
tests check these numbers.

**Not counted:** re-reads caused by tiling (left to the measurement work), masks, and the
embedding table beyond the 48 rows looked up.

## Reference computation and PR1's program

The model follows the reference code (lerobot SmolVLA) at true token counts. PR1's compiled program
differs in four ways, which `smolvla_ops.pr1_lowering` reproduces for the cross-check:
prefix rows padded (65 to 128, 177 to 256); expert attention keys padded to a multiple of 32;
cross-attention K/V projected once per call instead of every step; the time half of the time-MLP input
projection folded into a bias. With these the model matches PR1's RedMulE FLOP within 0.011 % per
stage ([crosscheck.md](crosscheck.md)).

## Comparing with Jetson Thor

* **Alignment:** iso-peak FP16 by cluster count. The SoftHier cluster count is the smallest square mesh
  whose RedMulE peak reaches `thor.fp16_dense` (258 TFLOP/s, inference): 16 x 16 = 256 clusters,
  262.1 TFLOP/s. Spatz is not counted in peak.
* **Second reference:** `thor.fp16_measured` (153 TFLOP/s, measured).
* **Bandwidth:** an open parameter. Provisional value for the expert RQ: Thor's `thor.dram_bw`
  (273 GB/s), labelled inference, then swept.
* **Thor's balance:** 258 TFLOP/s / 273 GB/s = 945 FLOP per byte (inference).

## Rule

Every number in a stage-1 document cites a constants key, or states its workload, arch and label.
