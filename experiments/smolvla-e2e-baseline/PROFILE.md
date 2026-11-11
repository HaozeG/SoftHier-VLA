# PROFILE: where the simulated time goes (SmolVLA on SoftHier)

Step PROF.1 (LOG.md). Per-section, per-op profile of the validated SmolVLA runs, plus what the hardware does inside each op. Everything here is derived from simulator logs that already exist or from short traced runs; no simulator, arch file or softhier-mlir patch was changed (GVSoC f99d5a2, stamp OK). Time is simulated ns at 1 GHz, ideal HBM.

**Labels.** *measured* = read straight from a full simulated run (`softhier_vla_wiki/tools/wiki.py glossary "measured"`: a reported run). *estimate* = scaled or computed (`glossary "estimate"`). Each table below says which one it is.

## Structure

1. [Where the data comes from](#where-the-data-comes-from): which runs, what instrumentation exists, how ops are delimited.
2. [Phase table](#phase-table): vision / connector / prefix / expert, and the cross-check against the gate (measured).
3. [Phase to layer to op](#phase-to-layer-to-op): every phase's ops with repetition counts, the top ops overall, the expert step anatomy (measured).
4. [Op classes](#op-classes): GEMM / attention / norm / elementwise / data movement / sync, with %; compute vs data-movement vs core bound (measured counts, heuristic labels).
5. [Hardware activity](#hardware-activity): RedMulE, iDMA, barrier wait, HBM/NoC bytes per class; one representative layer per phase; RedMulE utilisation of the GEMMs; what the cores do (cluster 0, sampled).
6. [What dominates and why](#what-dominates-and-why).
7. [LIBERO configuration (estimate)](#libero-configuration-estimate).
8. [Files, parser, how to redo](#files-parser-how-to-redo).
9. [Caveats](#caveats).

## Where the data comes from

| run (under `$SHVLA_DATA/runs/`) | what | used for | label |
| --- | --- | --- | --- |
| `published/fixi-rung6` | the validated full chain, published config (1 camera, 256 px, `smolvla_base`), 4 programs, `--e2e-trace` | all tables of sections 2 to 5 (every op instance of every phase) | measured |
| `published/prof-vision-L1` | `--vision-only 256 --layers 1` (the chain's vision program, 1 layer) with cluster-0 instruction traces aggregated in the pipe | core activity of vision ops | measured, 1 layer, cluster 0 only |
| `published/prof-prefix-L1` | the chain with the vision phase reused from `fixi-rung6`; `prefix_a` simulated until layer 1 ended | core activity of prefix ops | measured, 1 layer, cluster 0 only |
| `published/prof-expert-S1` | `--expert-only ref:e2e_c1_t256.npz --steps 1`, stopped at the end of step 0 | core activity of expert ops | measured, 1 step, cluster 0 only |
| `libero/G4a.{1,2,3}` | G4a partial runs (vision 1 layer at 1024 tokens, prefix 8 layers at 177 tokens, expert 2 steps) | section 7 | estimate (scaled by the G4a recipe) |

The three `prof-*` runs reproduce the full run exactly (same program, tracing only prints): vision layer 1 attention 1.2255 ms and MLP 1.0405 ms against 1.2254 / 1.0405 in `fixi-rung6`; prefix layer 1 0.9614 ms against 0.9618; expert KV projection + step 0 7.902 ms against 7.901.

**Existing instrumentation (inventory).**
- `softhier.mark {tag}` (`dialects/softhier.py` `MarkOp`, `backend/emit_c.py`): cluster 0 core 0 prints `[mark] <tag><idx> <mcycle>`; mcycle is the global clock, so differences are ns; the host unwraps the 32-bit counter. The e2e emitters place marks at segment level only: vision `start, vemb, attn<L>, layer<L>, vis, conn, end`; prefix `start, emb, attn<L>, layer<L>, end`; expert `start, kvproj, step<s>, xdump<s>, end`. `emit_flow(profile=True)` has per-op expert marks (`qkv, rope, attn, oproj, gateup, silu, down`) but `tests/gvsoc/run.py` does not expose it, and it was not needed.
- `--e2e-trace` runs gvsoc with `--trace=redmule --trace=idma --trace=cluster_registers` and keeps three line types: RedMulE `Finished : t0 ns ---> t1 ns | ... | uti = .. | GEMM id = .. | M-N-K = M-N-K`, iDMA `Finished : t0 ---> t1 | Txn i = {src, dst, size ..}` and `Cluster Sync: arrive ns -> release ns`. `softhier_mlir.sim.trace.segment_stats` turns them into the per-mark-segment busy times and HBM bytes in `summary.json`.
- The cycle counters the runtime prints are `sh_cycles()` (mcycle) in the marks and the ROI (`roi_ns`, which includes dump printing: 80 ms for vision against 27.9 ms of marked time, so it is not used).

**Op boundaries without new marks.** Every softhier library call ends with `sh_barrier_global()`, and gvsoc prints one `Cluster Sync` line per cluster and barrier. The release times on cluster 0 therefore delimit the ops: op k = (release k-1, release k]. The expanded op list of the phase's `.mlir` (loops unrolled) is aligned to these barriers segment by segment; the program's op count equals the barrier count in every mark segment of all four phases (a SigLIP layer: 45 barriers in the attention half = LN + 3 x (GEMM, bias) + transpose + 12 heads x 3 + group barrier, and 10 in the MLP half), except the expert's `xdump` segments (a debug dump, not timed, excluded). So the profile needed no patch 0007 and no new simulation for the time breakdown. Per-op RedMulE busy, iDMA busy, barrier wait, FLOPs (from the RedMulE M-N-K) and HBM/NoC bytes are the overlap of the trace events with each op window. `parse_profile.py ops` does this.

## Phase table

Measured. The sum of op windows reconstructs the gate's phase totals (marks) to within 1 % (the connector differs by 3 us: the mark printf, about 2 us, falls into the first op after a mark).

| phase | sum of ops | gate (marks) | diff % | % of total |
| --- | ---: | ---: | ---: | ---: |
| vision | 27.636 | 27.640 | -0.01 | 22.9 |
| connector | 0.283 | 0.280 | +1.01 | 0.2 |
| prefix | 15.032 | 15.039 | -0.04 | 12.5 |
| expert | 77.754 | 77.756 | -0.00 | 64.4 |
| total | 120.705 | 120.715 | -0.01 | 100 |

Definitions follow `run.py e2e_segments`: vision = start to `vis` (patch embed, 12 layers, post-LN); connector = pixel shuffle + connector GEMM (end of the vision program) + the embedding scale (start of `prefix_a`); prefix = the 16 decoder layers; expert = KV projection + 10 flow steps. Not in the total: the final RMSNorm after layer 16 (about 0.107 ms, `prefix_tail`), the per-step X dumps and the program ends.

## Phase to layer to op

Measured. Each row is an op label aggregated over all its instances in the run: `reps` = layers x steps x heads. Per-op, per-instance rows (phase, layer, step, op, start/end, duration, activity) are in `ops.csv`.

Repetition structure: vision = 12 layers x 55 ops (36 of them per-head attention ops: 12 heads x QK^T, softmax, P.V) + patch embed (GEMM, bias, positional add) + post-LN; prefix = 16 layers x 16 ops; expert = KV projection (8 prefix-layer pairs x 2 GEMMs) + 10 steps x (7 time/action-embedding ops + 16 layers (8 self-attention layers x 12 ops, 8 cross-attention layers x 11 ops) + 4 output ops).

### vision: 27.636 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attn_softmax | attention | 144 | 50.3 | 7.243 | 26.2 | 6.0 |
| gelu | elementwise | 12 | 485.3 | 5.823 | 21.1 | 4.8 |
| ln1 | norm | 12 | 149.4 | 1.792 | 6.5 | 1.5 |
| ln2 | norm | 12 | 148.2 | 1.779 | 6.4 | 1.5 |
| attn_pv | attention | 144 | 10.8 | 1.562 | 5.7 | 1.3 |
| attn_qk | attention | 144 | 9.0 | 1.300 | 4.7 | 1.1 |
| mlp_down | GEMM | 12 | 107.3 | 1.288 | 4.7 | 1.1 |
| bias_up | elementwise | 12 | 103.7 | 1.244 | 4.5 | 1.0 |
| mlp_up | GEMM | 12 | 75.9 | 0.911 | 3.3 | 0.8 |
| transpose_k | data movement | 12 | 58.3 | 0.699 | 2.5 | 0.6 |
| gemm_v | GEMM | 12 | 33.6 | 0.404 | 1.5 | 0.3 |
| gemm_q | GEMM | 12 | 33.6 | 0.404 | 1.5 | 0.3 |

### connector: 0.280 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| connector_gemm | GEMM | 1 | 263.2 | 0.263 | 94.0 | 0.2 |
| pixel_shuffle | data movement | 1 | 9.7 | 0.010 | 3.5 | 0.0 |
| embed_scale | elementwise | 1 | 7.0 | 0.007 | 2.5 | 0.0 |

### prefix: 15.032 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| silu_mul | elementwise | 16 | 198.7 | 3.179 | 21.1 | 2.6 |
| attention_fused | attention | 16 | 113.1 | 1.810 | 12.0 | 1.5 |
| rmsnorm_attn | norm | 16 | 107.0 | 1.712 | 11.4 | 1.4 |
| rmsnorm_mlp | norm | 16 | 106.7 | 1.708 | 11.4 | 1.4 |
| mlp_down | GEMM | 16 | 89.7 | 1.435 | 9.5 | 1.2 |
| mlp_up | GEMM | 16 | 64.0 | 1.023 | 6.8 | 0.8 |
| mlp_gate | GEMM | 16 | 63.0 | 1.008 | 6.7 | 0.8 |
| gemm_q | GEMM | 16 | 38.1 | 0.610 | 4.1 | 0.5 |
| gemm_o | GEMM | 16 | 38.1 | 0.609 | 4.1 | 0.5 |
| rope_q | elementwise | 16 | 31.1 | 0.498 | 3.3 | 0.4 |
| gemm_k | GEMM | 16 | 22.1 | 0.353 | 2.3 | 0.3 |
| gemm_v | GEMM | 16 | 22.0 | 0.352 | 2.3 | 0.3 |

### expert: 77.754 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| silu_mul | elementwise | 160 | 77.9 | 12.467 | 16.0 | 10.3 |
| mlp_gateup | GEMM | 160 | 67.1 | 10.743 | 13.8 | 8.9 |
| xattn_self_and_prefix | attention | 80 | 130.6 | 10.445 | 13.4 | 8.7 |
| mlp_down | GEMM | 160 | 61.4 | 9.819 | 12.6 | 8.1 |
| xattn_prefix | attention | 80 | 97.8 | 7.826 | 10.1 | 6.5 |
| gemm_o | GEMM | 160 | 36.4 | 5.831 | 7.5 | 4.8 |
| rmsnorm_attn | norm | 160 | 26.0 | 4.154 | 5.3 | 3.4 |
| rmsnorm_mlp | norm | 160 | 25.4 | 4.062 | 5.2 | 3.4 |
| gemm_qkv | GEMM | 80 | 42.2 | 3.376 | 4.3 | 2.8 |
| rope_q | elementwise | 160 | 16.3 | 2.609 | 3.4 | 2.2 |
| gemm_q | GEMM | 80 | 27.4 | 2.193 | 2.8 | 1.8 |
| residual_attn | elementwise | 160 | 5.5 | 0.877 | 1.1 | 0.7 |

### Top ops overall

| phase | op | class | reps | total ms | mean us | % phase | % total | RedMulE busy % | iDMA busy % | barrier wait % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expert | silu_mul | elementwise | 160 | 12.467 | 77.9 | 16.0 | 10.3 | 0.0 | 4.3 | 23.2 |
| expert | mlp_gateup | GEMM | 160 | 10.743 | 67.1 | 13.8 | 8.9 | 10.4 | 69.4 | 23.4 |
| expert | xattn_self_and_prefix | attention | 80 | 10.445 | 130.6 | 13.4 | 8.7 | 1.0 | 17.3 | 6.6 |
| expert | mlp_down | GEMM | 160 | 9.819 | 61.4 | 12.6 | 8.1 | 14.2 | 66.2 | 25.9 |
| expert | xattn_prefix | attention | 80 | 7.826 | 97.8 | 10.1 | 6.5 | 1.1 | 12.9 | 6.0 |
| vision | attn_softmax | attention | 144 | 7.243 | 50.3 | 26.2 | 6.0 | 0.0 | 2.5 | 1.3 |
| expert | gemm_o | GEMM | 160 | 5.831 | 36.4 | 7.5 | 4.8 | 10.6 | 58.6 | 23.9 |
| vision | gelu | elementwise | 12 | 5.823 | 485.3 | 21.1 | 4.8 | 0.0 | 5.3 | 1.8 |
| expert | rmsnorm_attn | norm | 160 | 4.154 | 26.0 | 5.3 | 3.4 | 0.0 | 2.8 | 22.0 |
| expert | rmsnorm_mlp | norm | 160 | 4.062 | 25.4 | 5.2 | 3.4 | 0.0 | 2.9 | 22.5 |
| expert | gemm_qkv | GEMM | 80 | 3.376 | 42.2 | 4.3 | 2.8 | 11.9 | 56.7 | 23.7 |
| prefix | silu_mul | elementwise | 16 | 3.179 | 198.7 | 21.1 | 2.6 | 0.0 | 8.5 | 2.6 |
| expert | rope_q | elementwise | 160 | 2.609 | 16.3 | 3.4 | 2.2 | 0.0 | 5.5 | 23.3 |
| expert | gemm_q | GEMM | 80 | 2.193 | 27.4 | 2.8 | 1.8 | 11.0 | 56.5 | 23.7 |
| prefix | attention_fused | attention | 16 | 1.810 | 113.1 | 12.0 | 1.5 | 1.5 | 25.2 | 7.7 |
| vision | ln1 | norm | 12 | 1.792 | 149.4 | 6.5 | 1.5 | 0.0 | 3.5 | 1.5 |
| vision | ln2 | norm | 12 | 1.779 | 148.2 | 6.4 | 1.5 | 0.0 | 3.5 | 1.7 |
| prefix | rmsnorm_attn | norm | 16 | 1.712 | 107.0 | 11.4 | 1.4 | 0.0 | 2.6 | 23.0 |
| prefix | rmsnorm_mlp | norm | 16 | 1.708 | 106.7 | 11.4 | 1.4 | 0.0 | 2.8 | 23.3 |
| vision | attn_pv | attention | 144 | 1.562 | 10.8 | 5.7 | 1.3 | 1.5 | 2.6 | 84.0 |
| prefix | mlp_down | GEMM | 16 | 1.435 | 89.7 | 9.5 | 1.2 | 9.0 | 23.7 | 73.4 |
| vision | attn_qk | attention | 144 | 1.300 | 9.0 | 4.7 | 1.1 | 1.4 | 3.2 | 81.1 |

### Expert step anatomy (step 5, measured)

One flow step is 7.761 ms: time/action embedding 0.142 ms (1.8 %: action in-projection, bias, the 3-op time MLP with SiLU, bias), 8 self layers 4.038 ms (52.0 %, 505 us each), 8 cross layers 3.543 ms (45.6 %, 443 us each), output (final RMSNorm, action out-projection, bias, Euler update) 0.039 ms (0.5 %). The KV projection of the prefix (cross-attention K and V of 8 layer pairs) is 0.137 ms once per inference. Prefix layers take 0.90 to 0.97 ms each (layers 3 to 5 and 11 to 13 are the fast ones, 0.903 ms; mean 0.940 ms).

Cross-attention to the prefix KV (`xattn_prefix`, 98 us) and self + prefix attention (`xattn_self_and_prefix`, 131 us) are one library call each; their parts are derived in the next section. Representative layers (step 5): self layer 6 and cross layer 7.

#### expert layer 6 step 5: 488.6 us in 12 ops

| # | op | class | us | % | RedMulE busy % | iDMA busy % | barrier wait % | active cl. | TFLOP/s | HBM rd MB | bound |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | rmsnorm_attn | norm | 26.5 | 5.4 | 0.0 | 2.6 | 21.5 | 0 | 0.00 | 0.09 | core-bound |
| 2 | gemm_qkv | GEMM | 35.8 | 7.3 | 14.1 | 58.9 | 17.9 | 16 | 3.22 | 4.10 | data-movement-bound |
| 3 | rope_q | elementwise | 16.3 | 3.3 | 0.0 | 5.6 | 23.3 | 0 | 0.00 | 0.10 | core-bound |
| 4 | rope_k | elementwise | 6.6 | 1.4 | 0.0 | 5.0 | 20.5 | 0 | 0.00 | 0.04 | core-bound |
| 5 | xattn_self_and_prefix | attention | 130.6 | 26.7 | 1.0 | 17.3 | 6.6 | 15 | 0.19 | 0.54 | core-bound |
| 6 | gemm_o | GEMM | 36.3 | 7.4 | 10.6 | 54.6 | 27.7 | 15 | 1.91 | 2.82 | data-movement-bound |
| 7 | residual_attn | elementwise | 5.5 | 1.1 | 0.0 | 15.2 | 27.2 | 0 | 0.00 | 0.14 | core-bound |
| 8 | rmsnorm_mlp | norm | 25.4 | 5.2 | 0.0 | 2.8 | 22.4 | 0 | 0.00 | 0.09 | core-bound |
| 9 | mlp_gateup | GEMM | 63.0 | 12.9 | 11.1 | 62.0 | 30.3 | 16 | 4.68 | 7.05 | data-movement-bound |
| 10 | silu_mul | elementwise | 77.9 | 16.0 | 0.0 | 4.3 | 23.2 | 0 | 0.00 | 0.41 | core-bound |
| 11 | mlp_down | GEMM | 59.3 | 12.1 | 14.7 | 58.2 | 33.6 | 15 | 2.49 | 6.02 | data-movement-bound |
| 12 | residual_mlp | elementwise | 5.5 | 1.1 | 0.0 | 15.1 | 27.2 | 0 | 0.00 | 0.14 | core-bound |

#### expert layer 7 step 5: 440.1 us in 11 ops

| # | op | class | us | % | RedMulE busy % | iDMA busy % | barrier wait % | active cl. | TFLOP/s | HBM rd MB | bound |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | rmsnorm_attn | norm | 25.3 | 5.8 | 0.0 | 2.8 | 22.4 | 0 | 0.00 | 0.09 | core-bound |
| 2 | gemm_q | GEMM | 27.9 | 6.3 | 10.8 | 53.0 | 27.5 | 15 | 2.47 | 2.46 | data-movement-bound |
| 3 | rope_q | elementwise | 16.3 | 3.7 | 0.0 | 5.6 | 23.3 | 0 | 0.00 | 0.10 | core-bound |
| 4 | xattn_prefix | attention | 97.8 | 22.2 | 1.1 | 12.9 | 6.0 | 15 | 0.19 | 0.35 | core-bound |
| 5 | gemm_o | GEMM | 36.2 | 8.2 | 10.7 | 54.7 | 27.8 | 15 | 1.91 | 2.82 | data-movement-bound |
| 6 | residual_attn | elementwise | 5.5 | 1.2 | 0.0 | 15.1 | 27.1 | 0 | 0.00 | 0.14 | core-bound |
| 7 | rmsnorm_mlp | norm | 25.3 | 5.8 | 0.0 | 2.8 | 22.5 | 0 | 0.00 | 0.09 | core-bound |
| 8 | mlp_gateup | GEMM | 63.0 | 14.3 | 11.1 | 62.1 | 30.4 | 16 | 4.68 | 7.05 | data-movement-bound |
| 9 | silu_mul | elementwise | 77.9 | 17.7 | 0.0 | 4.3 | 23.2 | 0 | 0.00 | 0.41 | core-bound |
| 10 | mlp_down | GEMM | 59.3 | 13.5 | 14.7 | 58.3 | 33.6 | 15 | 2.49 | 6.02 | data-movement-bound |
| 11 | residual_mlp | elementwise | 5.5 | 1.2 | 0.0 | 15.1 | 27.1 | 0 | 0.00 | 0.14 | core-bound |

## Op classes

Measured. Class of an op: GEMM = `softhier.gemm` outside attention; attention = per-head QK^T / softmax / P.V (vision) and the fused `attention` / `cross_attention` calls (prefix, expert); norm = LayerNorm / RMSNorm; elementwise = bias add, residual add, GELU, SiLU-multiply, RoPE, scale, Euler update; data movement = K transpose, pixel shuffle; sync = the group barrier.

| class | ms | % total | % vision | % connector | % prefix | % expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMM | 42.431 | 35.2 | 13.9 | 93.1 | 35.9 | 42.4 |
| elementwise | 31.761 | 26.3 | 33.5 | 2.5 | 29.3 | 23.3 |
| attention | 30.186 | 25.0 | 36.6 | 0.0 | 12.0 | 23.5 |
| norm | 15.611 | 12.9 | 13.5 | 0.0 | 22.7 | 10.9 |
| data movement | 0.709 | 0.6 | 2.5 | 3.4 | 0.0 | 0.0 |
| sync | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Bound per op class.** Judged on the critical-path cluster of each op (the cluster that waits least at the closing barrier). GEMM: RedMulE busy >= 50 % of the window = compute-bound; otherwise iDMA busy > RedMulE busy and >= 25 % = data-movement-bound; otherwise latency-bound. Attention: RedMulE >= 50 % compute, iDMA >= 50 % data movement, otherwise core-bound (the softmax and the head loop run on the cores). Elementwise and norm: core-bound unless iDMA >= 50 %. `imbalance` flags an op whose clusters wait on average more than 50 % of the window at the barrier (most clusters had no work). The counts are measured; the labels are this heuristic.

| class | compute-bound | data-movement-bound | latency-bound | core-bound | sync | of which imbalance | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMM | 3.901 (9 %) | 38.384 (90 %) | 0.146 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 8.317 | 42.431 |
| elementwise | 0.000 (0 %) | 0.552 (2 %) | 0.000 (0 %) | 31.209 (98 %) | 0.000 (0 %) | 0.012 | 31.761 |
| attention | 0.000 (0 %) | 1.300 (4 %) | 0.000 (0 %) | 28.886 (96 %) | 0.000 (0 %) | 2.863 | 30.186 |
| norm | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 15.611 (100 %) | 0.000 (0 %) | 0.000 | 15.611 |
| data movement | 0.000 (0 %) | 0.709 (100 %) | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 0.000 | 0.709 |
| sync | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 0.004 (100 %) | 0.000 | 0.004 |

Whole run: cluster-mean RedMulE busy 4.11 %, iDMA busy 25.20 %, barrier wait 21.91 % of the time; 189.2 GFLOP in 120.705 ms = 1.57 TFLOP/s = 1.20 % of the 131.1 TFLOP/s peak.

**Fused attention, derived split.** `attention` (prefix) and `cross_attention` (expert) are one call and one barrier, so they cannot be cut by barriers. Each cluster runs one head with exactly two RedMulE operations (QK^T then P.V); the parts below are taken from their event times on the clusters that ran a head. Derived, not delimited: load / setup = window start to QK^T start (Q, K, V staging by iDMA), softmax gap = QK^T end to P.V start (scale, mask, max, exp, sum, normalise on the cores), tail = P.V end to the barrier (store).

| phase | part | reps | mean us | % of attention op |
| --- | --- | ---: | ---: | ---: |
| prefix | attn_load | 16 | 31.9 | 28.2 |
| prefix | attn_qk | 16 | 0.8 | 0.7 |
| prefix | attn_softmax_gap | 16 | 63.6 | 56.2 |
| prefix | attn_pv | 16 | 1.0 | 0.9 |
| prefix | attn_tail | 16 | 15.8 | 13.9 |
| expert | attn_load | 160 | 30.4 | 26.7 |
| expert | attn_qk | 160 | 0.5 | 0.5 |
| expert | attn_softmax_gap | 160 | 74.8 | 65.5 |
| expert | attn_pv | 160 | 0.7 | 0.6 |
| expert | attn_tail | 160 | 7.7 | 6.7 |

## Hardware activity

Measured (trace events of the full run). RedMulE busy = sum of RedMulE `Finished` intervals, iDMA busy = sum of iDMA front-end `Finished` intervals, barrier wait = sum of `Cluster Sync` intervals; all are averaged over the 16 clusters and expressed as % of the op window. Peak RedMulE throughput from the arch (`arch/softhier_mlir_arch.py`, `Arch.macs_per_cycle`): 128 x 32 CEs = 4096 MAC/cycle = 8192 FLOP/cycle per cluster; x 16 clusters x 1 GHz = **131.07 TFLOP/s** (spec figure). The RedMulE `uti` field is MAC/(cycle x 4096) of one cluster while it is busy.

### Per phase and class

| phase | class | ms | RedMulE busy % | iDMA busy % | barrier wait % | chip FLOP/s, % of peak | HBM read MB | HBM write MB | c2c MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vision | attention | 10.105 | 0.4 | 2.6 | 24.4 | 0.18 | 51.9 | 42.5 | 0.0 |
| vision | elementwise | 9.265 | 0.0 | 12.4 | 4.8 | 0.00 | 84.1 | 71.6 | 0.0 |
| vision | GEMM | 3.843 | 10.0 | 17.5 | 78.1 | 8.69 | 284.9 | 42.9 | 0.0 |
| vision | norm | 3.720 | 0.0 | 3.5 | 1.6 | 0.00 | 11.1 | 9.8 | 0.0 |
| vision | data movement | 0.699 | 0.0 | 70.8 | 27.2 | 0.00 | 4.7 | 4.7 | 0.0 |
| connector | GEMM | 0.263 | 11.1 | 15.1 | 83.2 | 1.09 | 24.8 | 0.0 | 0.0 |
| prefix | GEMM | 5.391 | 8.2 | 28.8 | 65.6 | 5.70 | 492.8 | 35.4 | 0.0 |
| prefix | elementwise | 4.411 | 0.0 | 11.7 | 4.6 | 0.00 | 44.4 | 25.5 | 0.0 |
| prefix | norm | 3.420 | 0.0 | 2.7 | 23.2 | 0.00 | 8.6 | 7.9 | 0.0 |
| prefix | attention | 1.810 | 1.5 | 25.2 | 7.7 | 0.42 | 11.9 | 3.9 | 0.0 |
| expert | GEMM | 32.934 | 11.7 | 63.7 | 24.6 | 2.27 | 3119.4 | 111.9 | 0.0 |
| expert | attention | 18.270 | 1.0 | 15.4 | 6.3 | 0.14 | 71.0 | 15.4 | 0.0 |
| expert | elementwise | 18.078 | 0.0 | 5.5 | 23.4 | 0.00 | 134.6 | 76.7 | 0.0 |
| expert | norm | 8.472 | 0.0 | 2.9 | 22.2 | 0.00 | 29.9 | 23.8 | 0.0 |

| phase | ms | GFLOP (RedMulE) | % of chip peak | HBM read GB | HBM write GB | mean HBM GB/s | c2c GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vision | 27.636 | 46.2 | 1.28 | 0.437 | 0.171 | 22.0 | 0.000 |
| connector | 0.283 | 0.4 | 1.02 | 0.025 | 0.001 | 91.4 | 0.000 |
| prefix | 15.032 | 41.3 | 2.09 | 0.558 | 0.073 | 41.9 | 0.000 |
| expert | 77.754 | 101.3 | 0.99 | 3.355 | 0.228 | 46.1 | 0.000 |

HBM bytes are the iDMA transactions whose source or destination is in HBM (>= 0xC0000000). `c2c` (cluster-to-cluster over the NoC) is 0 everywhere: the generated SmolVLA programs stream everything from HBM and never use cluster-to-cluster multicast. HBM is ideal in these runs (`SOFTHIER_IDEAL_HBM=1`), so the GB/s shown are what the clusters pulled, not what a real HBM could deliver.

### One representative layer per phase

Vision layer 6 (the 12 heads are summed; layers differ by up to 1.7 %: 2.264 to 2.276 ms for layers 1 to 7, 2.303 ms for layers 8 to 12):

#### vision layer 6: 2273.9 us, 55 ops

| op | class | reps | us total | % | RedMulE busy % | iDMA busy % | barrier wait % | active cl. | TFLOP/s | bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ln1 | norm | 1 | 149.1 | 6.6 | 0.0 | 3.3 | 1.5 | 0.0 | 0.00 | core-bound |
| gemm_q | GEMM | 1 | 34.5 | 1.5 | 7.9 | 10.4 | 84.5 | 2.0 | 8.76 | compute-bound+imbalance |
| bias_q | elementwise | 1 | 26.9 | 1.2 | 0.0 | 19.1 | 9.3 | 0.0 | 0.00 | core-bound |
| gemm_k | GEMM | 1 | 34.5 | 1.5 | 7.9 | 10.4 | 84.6 | 2.0 | 8.76 | compute-bound+imbalance |
| bias_k | elementwise | 1 | 26.9 | 1.2 | 0.0 | 19.1 | 9.3 | 0.0 | 0.00 | core-bound |
| gemm_v | GEMM | 1 | 34.5 | 1.5 | 7.9 | 10.4 | 84.6 | 2.0 | 8.76 | compute-bound+imbalance |
| bias_v | elementwise | 1 | 26.8 | 1.2 | 0.0 | 19.1 | 9.3 | 0.0 | 0.00 | core-bound |
| transpose_k | data movement | 1 | 58.3 | 2.6 | 0.0 | 70.8 | 27.2 | 0.0 | 0.00 | data-movement-bound |
| attn_qk | attention | 12 | 108.4 | 4.8 | 1.4 | 3.2 | 81.1 | 1.0 | 0.93 | data-movement-bound+imbalance |
| attn_softmax | attention | 12 | 603.6 | 26.5 | 0.0 | 2.5 | 1.3 | 0.0 | 0.00 | core-bound |
| attn_pv | attention | 12 | 130.2 | 5.7 | 1.5 | 2.6 | 84.0 | 1.0 | 0.77 | core-bound+imbalance |
| group_barrier | sync | 1 | 0.3 | 0.0 | 0.0 | 0.0 | 27.8 | 0.0 | 0.00 | sync |
| gemm_o | GEMM | 1 | 34.0 | 1.5 | 8.0 | 10.3 | 84.8 | 2.0 | 8.88 | compute-bound+imbalance |
| bias_o | elementwise | 1 | 26.8 | 1.2 | 0.0 | 19.1 | 9.3 | 0.0 | 0.00 | core-bound |
| residual_attn | elementwise | 1 | 22.1 | 1.0 | 0.0 | 38.9 | 16.0 | 0.0 | 0.00 | data-movement-bound |
| ln2 | norm | 1 | 148.0 | 6.5 | 0.0 | 3.3 | 1.7 | 0.0 | 0.00 | core-bound |
| mlp_up | GEMM | 1 | 61.9 | 2.7 | 17.5 | 38.7 | 55.0 | 8.0 | 19.50 | data-movement-bound+imbalance |
| bias_up | elementwise | 1 | 102.9 | 4.5 | 0.0 | 23.6 | 8.5 | 0.0 | 0.00 | core-bound |
| gelu | elementwise | 1 | 485.3 | 21.3 | 0.0 | 5.3 | 1.8 | 0.0 | 0.00 | core-bound |
| mlp_down | GEMM | 1 | 110.3 | 4.8 | 9.1 | 11.4 | 86.6 | 2.0 | 10.96 | compute-bound+imbalance |
| bias_down | elementwise | 1 | 26.8 | 1.2 | 0.0 | 19.2 | 9.3 | 0.0 | 0.00 | core-bound |
| residual_mlp | elementwise | 1 | 22.1 | 1.0 | 0.0 | 38.9 | 16.0 | 0.0 | 0.00 | data-movement-bound |

Prefix layer 6:

#### prefix layer 6: 951.8 us in 16 ops

| # | op | class | us | % | RedMulE busy % | iDMA busy % | barrier wait % | active cl. | TFLOP/s | HBM rd MB | bound |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | rmsnorm_attn | norm | 107.4 | 11.3 | 0.0 | 2.8 | 22.9 | 0 | 0.00 | 0.28 | core-bound |
| 2 | gemm_q | GEMM | 37.4 | 3.9 | 7.9 | 20.7 | 72.5 | 5 | 6.31 | 3.07 | data-movement-bound+imbalance |
| 3 | gemm_k | GEMM | 22.1 | 2.3 | 3.8 | 4.4 | 88.6 | 1 | 3.56 | 0.86 | compute-bound+imbalance |
| 4 | gemm_v | GEMM | 22.0 | 2.3 | 3.9 | 4.4 | 88.9 | 1 | 3.57 | 0.86 | compute-bound+imbalance |
| 5 | rope_q | elementwise | 31.1 | 3.3 | 0.0 | 11.5 | 5.0 | 0 | 0.00 | 0.26 | core-bound |
| 6 | rope_k | elementwise | 11.6 | 1.2 | 0.0 | 9.9 | 4.0 | 0 | 0.00 | 0.10 | core-bound |
| 7 | attention_fused | attention | 113.1 | 11.9 | 1.5 | 25.1 | 7.7 | 15 | 0.56 | 0.74 | core-bound |
| 8 | pad_rows_zero | elementwise | 6.0 | 0.6 | 0.0 | 16.1 | 13.0 | 0 | 0.00 | 0.12 | core-bound |
| 9 | gemm_o | GEMM | 37.3 | 3.9 | 7.9 | 20.6 | 72.6 | 5 | 6.33 | 3.07 | data-movement-bound+imbalance |
| 10 | residual_attn | elementwise | 14.2 | 1.5 | 0.0 | 34.4 | 16.8 | 0 | 0.00 | 0.49 | core-bound |
| 11 | rmsnorm_mlp | norm | 106.7 | 11.2 | 0.0 | 2.8 | 23.3 | 0 | 0.00 | 0.28 | core-bound |
| 12 | mlp_gate | GEMM | 63.2 | 6.6 | 9.4 | 45.3 | 48.4 | 10 | 9.96 | 7.37 | data-movement-bound |
| 13 | mlp_up | GEMM | 70.7 | 7.4 | 8.4 | 45.9 | 48.5 | 10 | 8.90 | 7.37 | data-movement-bound |
| 14 | silu_mul | elementwise | 198.7 | 20.9 | 0.0 | 8.5 | 2.6 | 0 | 0.00 | 1.31 | core-bound |
| 15 | mlp_down | GEMM | 96.2 | 10.1 | 8.3 | 25.9 | 71.3 | 5 | 6.54 | 8.19 | data-movement-bound+imbalance |
| 16 | residual_mlp | elementwise | 14.2 | 1.5 | 0.0 | 34.4 | 16.9 | 0 | 0.00 | 0.49 | core-bound |

Expert: see the self layer 6 / cross layer 7 tables above.

### RedMulE utilisation of the GEMMs

FLOPs are counted from the executed RedMulE operations (2 M N K, padded shapes: the prefix runs 128 token rows for 65 valid tokens), time is the barrier-to-barrier window. TFLOP/s and % of peak are chip level (16 clusters). `active clusters` = clusters that ran at least one RedMulE operation in the window.

| phase | op | reps | mean us | GFLOP per op | TFLOP/s | % of chip peak | active clusters | sim-reported array uti | crit-cluster RedMulE busy % | crit-cluster iDMA busy % | bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| expert | mlp_gateup | 160 | 67.1 | 0.295 | 4.39 | 3.4 | 16.0 | 0.32 | 10.4 | 92.8 | data-movement-bound |
| expert | xattn_self_and_prefix | 80 | 130.6 | 0.025 | 0.19 | 0.1 | 15.0 | 0.15 | 1.0 | 19.2 | core-bound |
| expert | mlp_down | 160 | 61.4 | 0.147 | 2.40 | 1.8 | 15.0 | 0.13 | 15.2 | 91.8 | data-movement-bound |
| expert | xattn_prefix | 80 | 97.8 | 0.018 | 0.19 | 0.1 | 15.0 | 0.12 | 1.2 | 13.9 | core-bound |
| expert | gemm_o | 160 | 36.4 | 0.069 | 1.90 | 1.4 | 15.0 | 0.14 | 11.3 | 81.9 | data-movement-bound |
| expert | gemm_qkv | 80 | 42.2 | 0.115 | 2.73 | 2.1 | 16.0 | 0.17 | 15.3 | 76.4 | data-movement-bound |
| expert | gemm_q | 80 | 27.4 | 0.069 | 2.52 | 1.9 | 15.0 | 0.17 | 11.8 | 79.5 | data-movement-bound |
| prefix | attention_fused | 16 | 113.1 | 0.063 | 0.56 | 0.4 | 15.0 | 0.29 | 1.6 | 28.7 | core-bound |
| vision | attn_pv | 144 | 10.8 | 0.008 | 0.77 | 0.6 | 1.0 | 0.40 | 23.6 | 40.8 | core-bound+imbalance |
| prefix | mlp_down | 16 | 89.7 | 0.629 | 7.01 | 5.4 | 5.0 | 0.60 | 28.6 | 94.0 | data-movement-bound+imbalance |
| vision | attn_qk | 144 | 9.0 | 0.008 | 0.93 | 0.7 | 1.0 | 0.52 | 22.0 | 51.1 | data-movement-bound+imbalance |
| vision | mlp_down | 12 | 107.3 | 1.208 | 11.26 | 8.6 | 2.0 | 0.92 | 74.6 | 88.6 | compute-bound+imbalance |
| prefix | mlp_up | 16 | 64.0 | 0.629 | 9.84 | 7.5 | 10.0 | 0.81 | 14.8 | 91.6 | data-movement-bound |
| prefix | mlp_gate | 16 | 63.0 | 0.629 | 9.98 | 7.6 | 10.0 | 0.81 | 15.0 | 91.5 | data-movement-bound |
| vision | mlp_up | 12 | 75.9 | 1.208 | 15.92 | 12.1 | 8.0 | 0.85 | 28.5 | 92.1 | data-movement-bound+imbalance |
| prefix | gemm_q | 16 | 38.1 | 0.236 | 6.19 | 4.7 | 5.0 | 0.61 | 24.8 | 86.5 | data-movement-bound+imbalance |
| prefix | gemm_o | 16 | 38.1 | 0.236 | 6.20 | 4.7 | 5.0 | 0.61 | 24.8 | 86.6 | data-movement-bound+imbalance |
| vision | gemm_v | 12 | 33.6 | 0.302 | 8.98 | 6.8 | 2.0 | 0.85 | 64.8 | 79.2 | compute-bound+imbalance |
| vision | gemm_q | 12 | 33.6 | 0.302 | 8.98 | 6.9 | 2.0 | 0.85 | 64.8 | 78.6 | compute-bound+imbalance |
| vision | gemm_o | 12 | 33.6 | 0.302 | 8.98 | 6.9 | 2.0 | 0.85 | 64.7 | 79.4 | compute-bound+imbalance |
| vision | gemm_k | 12 | 33.6 | 0.302 | 8.99 | 6.9 | 2.0 | 0.85 | 64.8 | 78.7 | compute-bound+imbalance |
| prefix | gemm_k | 16 | 22.1 | 0.079 | 3.56 | 2.7 | 1.0 | 0.71 | 61.4 | 70.7 | compute-bound+imbalance |

### What the cores do (cluster 0, sampled)

Measured on cluster 0 only, for one layer (vision, prefix) or one step (expert), by aggregating `pe0-2` and `fp_ss0-2` instruction traces in the pipe (1 us bins; see the parser's `insn`). A core issues at most one instruction per ns. `pe2` is the DMA core (the one that issues iDMA commands and polls `dmstati`); `pe0` / `pe1` drive RedMulE and the compute loops. In GEMM windows cluster 0 is one of the active clusters in most ops (see `active clusters`), so its numbers are representative for those, not for every cluster.

| phase | class | window ms | pe0/pe1 issue per ns | pe2 (DMA core) issue per ns | pe2 `dmstati` polls per ns | pe0/pe1 not issuing % | FP ops per ns per FP sub-system |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vision | attention | 0.842 | 0.107 | 0.286 | 0.029 | 89.3 | 0.126 |
| vision | elementwise | 0.820 | 0.136 | 0.163 | 0.006 | 86.4 | 0.191 |
| vision | norm | 0.447 | 0.115 | 0.137 | 0.001 | 88.5 | 0.161 |
| vision | GEMM | 0.326 | 0.013 | 0.556 | 0.235 | 98.7 | 0.000 |
| connector | GEMM | 0.263 | 0.005 | 0.500 | 0.244 | 99.5 | 0.000 |
| vision | data movement | 0.058 | 0.002 | 0.180 | 0.057 | 99.8 | 0.000 |
| prefix | GEMM | 0.358 | 0.011 | 0.311 | 0.133 | 98.9 | 0.000 |
| prefix | elementwise | 0.276 | 0.116 | 0.134 | 0.005 | 88.4 | 0.177 |
| prefix | norm | 0.215 | 0.056 | 0.067 | 0.001 | 94.4 | 0.069 |
| prefix | attention | 0.113 | 0.131 | 0.159 | 0.007 | 86.9 | 0.112 |
| expert | GEMM | 3.416 | 0.013 | 0.237 | 0.101 | 98.7 | 0.000 |
| expert | attention | 1.827 | 0.123 | 0.187 | 0.021 | 87.7 | 0.137 |
| expert | elementwise | 1.813 | 0.119 | 0.148 | 0.005 | 88.1 | 0.175 |
| expert | norm | 0.846 | 0.080 | 0.151 | 0.002 | 92.0 | 0.109 |

Reading: during GEMMs `pe0`/`pe1` issue less than 0.015 instructions per ns (>= 98 % of the cycles waiting for RedMulE or the barrier), while `pe2` retires 0.24 to 0.56 per ns with 40 to 47 % of its instructions being `dmstati` polls plus the loop branch (10 to 24 % of all its cycles are polls). During elementwise, norm and attention windows all three cores issue only 0.06 to 0.14 instructions per ns and the FP sub-systems 0.07 to 0.19 FP operations per ns, so the cores are stalled 86 to 94 % of the cycles. The raw sample (`insn_sample_pe0.txt.gz`, pe0 in a fused-attention window) shows runs of single-cycle integer instructions separated by 5 to 100 ns gaps after `andi`, `add`, `bne`, `beq` and `jal`; the cause of the stalls (FP sub-system hand-off, TCDM contention, or the model's instruction-fetch path) was not diagnosed here.

## What dominates and why

1. **The expert is 64 % of the time, and it is not a RedMulE story.** 77.8 ms of 120.7 ms. Per step the 16 layers cost 7.58 ms. Its GEMMs have only 50 rows (M = 50), so each GEMM is a weight stream: the critical cluster's iDMA is busy 76 to 93 % of the window against 10 to 15 % for RedMulE, and the array runs at `uti` 0.13 to 0.32. Together with the full set of layer weights re-read on each of the 10 steps (about 193 MB of weights per step, an estimate computed from the op shapes; 335 MB of HBM reads measured per step), GEMMs reach 1.9 to 4.4 TFLOP/s (1.4 to 3.4 % of peak). The expert's GEMM class is 42 % of the phase; **silu_mul alone is 16 %** (77.9 us for 50 x 2048 elements, 12.5 ms in total), the two attention calls 23.5 %, the RMSNorms 10.9 %.
2. **Less than a third of the whole run is RedMulE-friendly work.** GEMM is 35.2 % of the total; elementwise 26.3 %, attention 25.0 % and norm 12.9 % (64.5 % together) run on the three Snitch cores of each cluster. Cluster-mean RedMulE busy is 4.1 % of the time; the whole run achieves 1.57 TFLOP/s = 1.2 % of the 131 TFLOP/s peak (189 GFLOP in 120.7 ms).
3. **Attention is a softmax problem, not a matmul problem.** RedMulE is busy 0.4 to 1.5 % of an attention window. Vision `attn_softmax` is 144 calls x 50.3 us = 7.24 ms = 26 % of the vision phase and 6 % of the run; QK^T plus P.V together are 2.9 ms. In the fused attention of the prefix and expert, 56 to 66 % of the window is the softmax gap between the two RedMulE operations, 27 to 28 % is staging, 7 to 14 % the tail. 96 % of the attention time is classified core-bound (92 % in LIBERO).
4. **Core-executed element-wise code is slow per element.** `gelu` (vision) takes 485 us per layer for 256 x 3072 elements (21 % of the vision phase), `silu_mul` 199 us per prefix layer (21 % of the prefix phase), LayerNorm / RMSNorm 107 to 149 us per prefix / vision op for 128 or 256 x 960 / 768 elements. The cores issue only 0.1 to 0.14 instructions per ns in these ops (section above).
5. **Vision and prefix GEMMs leave most clusters idle.** The tile policy (`vision.mlir`: `tile_m = 256, tile_n = 384` for the 256 x 768 x 768 GEMMs) gives only 768 / 384 = 2 output tiles, so 2 of 16 clusters work (9.0 TFLOP/s = 6.9 % of peak; `barrier wait` is 78 % of the vision GEMM class); the MLP up / gate GEMMs use 8 to 10 clusters (10 to 16 TFLOP/s, 7.5 to 12 % of peak) and the down-projections (K = 3072 / 2560) only 2 to 5. Per-head attention GEMMs (256 x 256 x 64) use 1 cluster. Those clusters are RedMulE-compute-bound while running (RedMulE busy 65 to 75 % on the critical cluster, `uti` 0.85 to 0.92), so the lever is parallelism over clusters, not the array.
6. **Per-op fixed cost.** Every op ends in a global barrier and costs at least 2 to 5 us; there are 2 890 timed ops in the run (664 vision, 3 connector, 256 prefix, 1 967 expert). Ops with a mean under 20 us (residual adds 5.5 us, RoPE-K 6.6 us, RoPE-Q 16 us, the small biases, KV projection GEMMs, Euler update) are 7 % of the expert phase; the fixed cost matters less than the per-element cost. The mark printf (about 2 us) is inside the window of the op after each mark and is included.
7. **Barrier wait is 21.9 % of the cluster-time** (idle clusters waiting for the slowest one), mostly in GEMMs (vision 78 %, prefix 66 %, expert 25 %) and in the expert's small elementwise / norm ops (22 to 27 %, 50 rows spread over 16 clusters).

## LIBERO configuration (estimate)

**Estimate.** Scaled from the G4a partial runs (`libero/G4a.{1,2,3}`), with the G4a recipe: vision = 2 cameras x (patch embed + 12 x one measured layer + post-LN) from a 1-layer run at 1024 tokens (x 24 for layer ops, x 2 for embed and post-LN), connector = the start-to-emb segment of the 2-camera prefix run, prefix = 2 x the measured 8 layers, expert = KV projection + 10 x the mean of the 2 measured steps (x 5). The scaled total is 479.099 ms against the G4a estimate of 479.117 ms (the difference is the op windows' mark gaps). Layer-to-layer variation inside a phase (up to about 7 % in the published run) is not modelled. The cold first layer is part of the 1-layer vision sample. Not measured at full length (G4b was not run).

| phase | sum of ops | % of total |
| --- | ---: | ---: |
| vision | 351.819 | 73.4 |
| connector | 0.534 | 0.1 |
| prefix | 32.947 | 6.9 |
| expert | 93.799 | 19.6 |
| total | 479.099 | 100 |

### vision: 351.819 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attn_softmax | attention | 288 | 729.1 | 209.973 | 59.7 | 43.8 |
| gelu | elementwise | 24 | 2227.9 | 53.470 | 15.2 | 11.2 |
| ln1 | norm | 24 | 549.4 | 13.186 | 3.7 | 2.8 |
| ln2 | norm | 24 | 548.5 | 13.163 | 3.7 | 2.7 |
| attn_qk | attention | 288 | 39.4 | 11.358 | 3.2 | 2.4 |
| attn_pv | attention | 288 | 39.3 | 11.328 | 3.2 | 2.4 |

### connector: 0.534 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| connector_gemm | GEMM | 1 | 448.8 | 0.449 | 84.1 | 0.1 |
| pixel_shuffle | data movement | 2 | 35.2 | 0.070 | 13.2 | 0.0 |
| embed_scale | elementwise | 1 | 14.6 | 0.015 | 2.7 | 0.0 |

### prefix: 32.947 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attention_fused | attention | 16 | 578.9 | 9.262 | 28.1 | 1.9 |
| silu_mul | elementwise | 16 | 396.2 | 6.339 | 19.2 | 1.3 |
| rmsnorm_attn | norm | 16 | 217.4 | 3.478 | 10.6 | 0.7 |
| rmsnorm_mlp | norm | 16 | 217.1 | 3.474 | 10.5 | 0.7 |
| mlp_down | GEMM | 16 | 119.1 | 1.905 | 5.8 | 0.4 |
| mlp_gate | GEMM | 16 | 115.7 | 1.851 | 5.6 | 0.4 |

### expert: 93.799 ms

| op | class | reps | mean us | total ms | % of phase | % of total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| xattn_self_and_prefix | attention | 80 | 243.7 | 19.498 | 20.8 | 4.1 |
| xattn_prefix | attention | 80 | 186.4 | 14.909 | 15.9 | 3.1 |
| silu_mul | elementwise | 160 | 77.9 | 12.468 | 13.3 | 2.6 |
| mlp_gateup | GEMM | 160 | 67.2 | 10.750 | 11.5 | 2.2 |
| mlp_down | GEMM | 160 | 60.6 | 9.695 | 10.3 | 2.0 |
| gemm_o | GEMM | 160 | 36.0 | 5.761 | 6.1 | 1.2 |

| class | ms | % total | % vision | % connector | % prefix | % expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention | 276.328 | 57.7 | 66.1 | 0.0 | 28.1 | 36.7 |
| elementwise | 99.026 | 20.7 | 20.6 | 2.7 | 26.0 | 19.3 |
| GEMM | 56.470 | 11.8 | 4.3 | 84.1 | 24.8 | 35.0 |
| norm | 42.863 | 8.9 | 7.8 | 0.0 | 21.1 | 9.0 |
| data movement | 4.405 | 0.9 | 1.2 | 13.2 | 0.0 | 0.0 |
| sync | 0.007 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| class | compute-bound | data-movement-bound | latency-bound | core-bound | sync | of which imbalance | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention | 0.000 (0 %) | 22.686 (8 %) | 0.000 (0 %) | 253.642 (92 %) | 0.000 (0 %) | 11.328 | 276.328 |
| elementwise | 0.000 (0 %) | 4.701 (5 %) | 0.000 (0 %) | 94.325 (95 %) | 0.000 (0 %) | 0.025 | 99.026 |
| GEMM | 0.179 (0 %) | 56.176 (99 %) | 0.115 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 13.487 | 56.470 |
| norm | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 42.863 (100 %) | 0.000 (0 %) | 0.000 | 42.863 |
| data movement | 0.000 (0 %) | 4.405 (100 %) | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 0.000 | 4.405 |
| sync | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 0.000 (0 %) | 0.007 (100 %) | 0.000 | 0.007 |

Whole run: cluster-mean RedMulE busy 1.99 %, iDMA busy 16.00 %, barrier wait 11.44 % of the time; 620.3 GFLOP in 479.099 ms = 1.29 TFLOP/s = 0.99 % of the 131.1 TFLOP/s peak.

| phase | ms | GFLOP (RedMulE) | % of chip peak | HBM read GB | HBM write GB | mean HBM GB/s | c2c GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vision | 351.819 | 427.6 | 0.93 | 4.415 | 2.278 | 19.0 | 0.000 |
| connector | 0.534 | 3.0 | 4.32 | 0.060 | 0.004 | 118.8 | 0.000 |
| prefix | 32.947 | 84.6 | 1.96 | 1.111 | 0.142 | 38.0 | 0.000 |
| expert | 93.799 | 105.1 | 0.86 | 3.430 | 0.229 | 39.0 | 0.000 |

What changes against the published config: the vision phase becomes 73 % of the run (351.8 ms, 2 cameras x 1024 tokens), and **one op, the per-head softmax (288 calls x 729 us = 210 ms), is 43.8 % of the whole inference**; with GELU (53.5 ms) and the two LayerNorms (26.3 ms) these three core-executed ops are about 60 % of the run. Attention is 57.7 % of the LIBERO run (against 25.0 %) and GEMM only 11.8 % (35.2 %). The expert grows from 77.8 to 93.8 ms (longer prefix: 177 tokens, S_pad 256) and its attention calls roughly double (131 -> 244 us, 98 -> 186 us), while its GEMMs are unchanged (the weights, not the prefix, set their cost).

## Files, parser, how to redo

- **Kept traces and results**: `~/.cache/softhier-vla/profile/` (not in git). `README.md` there has the exact commands, filter regexes, line formats and the file list. Sizes: 24 MB in total, every file well under 50 MB (largest: `published-fixi-rung6/expert.log.zst`, 10.8 MB).
  - `published-fixi-rung6/`: the four phase logs (`*.log.zst`: filtered to program lines, `Finished` and `Cluster Sync` lines, zstd), the `.mlir` programs, `ops.csv` (3 788 rows), `summary.md`.
  - `prof-vision-L1/`, `prof-prefix-L1/`, `prof-expert-S1/`: filtered logs, `ops.csv`, `insn_bins.csv.gz` (aggregated instruction counts), `insn_sample_pe0.txt.gz` (raw sample), `insn_ops.csv`, `insn_summary.md`.
  - `libero-G4a/`: the G4a logs, per-run `ops.csv`, `libero_ops_scaled.csv` (estimate), `summary.md`.
- **Parser** (stdlib only, no re-simulation): `experiments/smolvla-e2e-baseline/profile/parse_profile.py`
  - `ops <run-dir>` writes `ops.csv` (columns: phase, program, seg, layer, step, op, class, level, start_ns, end_ns, dur_ns, cluster, count, source, redmule_busy_ns, idma_busy_ns, sync_wait_ns, sync_wait_max_ns, redmule_flops, hbm_rd, hbm_wr, c2c, local, active_clusters, crit_redmule_ns, crit_idma_ns, redmule_uti, bound; `level` = op | sub (derived attention parts, not additive) | seg (a segment that could not be split)); `source` = `measured:barrier`, `derived:redmule-events`, `mark`, or `estimate:scaled xN`.
  - `summary`, `phases`, `layer` print the tables of this file; `insn` joins the instruction bins with an `ops.csv`; `libero` scales the G4a runs; `compact` filters and compresses a log. Logs may be `.zst`.
- **Wrapper** for new traced runs: `profile/prof_wrap.py` (+ `run_prof.sh`, `collect.sh`): gvsoc trace lines are filtered or aggregated in the pipe, kept trace lines are capped at 50 MB (program output is not counted; no run came near it), instruction lines are never written except a capped raw sample of one core (`insn_sample_pe0.txt.gz`, 103 to 5 122 lines). The filtered logs were written plain into the run directory (the run script reads them back as text) and compressed afterwards by `collect.sh`, not in the pipe. Same idea as `diag8/trace_wrap.py`; no repo patch.
- Redo the tables: `profile/collect.sh` (about 2 minutes, no simulator).

## Caveats

- **Ideal HBM.** All runs use `SOFTHIER_IDEAL_HBM=1`. HBM latency and bandwidth never limit; real HBM would add time, most for the expert (3.4 GB of HBM reads, 335 MB per step) and for the vision GEMMs.
- **Transpose residual.** The iDMA K transpose is still 2.3x slower than the published number (G3.3-gap). `transpose_k` is 0.70 ms of the vision phase (0.6 % of the run; 4.3 ms, 0.9 % in LIBERO) and probably an over-estimate by that factor. The attention segment is also +1.1 % against published.
- **Timing basis.** All numbers come from runs after the `instr_mem` fix (G3.3-fix-instr: `fixi-rung6`, G4a, the `prof-*` runs). Rung 3 to 5 timings (and `fixrm-*`) predate the fix and are not used here; if you reuse them, the times are about 25 % too low (the `fixrm-rung6` total is 91.1 ms against 120.7 ms).
- **Sampling vs full.** The time breakdown (sections 2 to 5, except the next two points) comes from the full rung-6 run: every op instance. Core-level instruction activity is a sample: cluster 0 only, one layer / one step, 1 us bins (windows shorter than about 10 us carry up to about 10 % quantisation error). The LIBERO tables are scaled from 1-layer / 8-layer / 2-step runs.
- **Derived values.** The fused-attention split is inferred from RedMulE event times, not delimited by barriers; the `bound` labels are a threshold heuristic on measured busy fractions; FLOPs are those of the padded shapes the hardware executes; the peak (131.07 TFLOP/s) is a spec figure from the arch.
- **Mark overhead.** The mark printf (about 2 us) lands in the first op after each mark; with marks every half layer (about 100 marks in the run) this is about 0.2 % of the run.
- **Excluded.** The final RMSNorm after prefix layer 16 (0.107 ms), the per-step X dumps, preload and program-end segments are not in the totals (the gate does not count them either).
- **Stall cause not diagnosed.** The 86 to 94 % non-issuing cycles of the cores in elementwise / norm / attention ops are measured, their microarchitectural cause is not.
