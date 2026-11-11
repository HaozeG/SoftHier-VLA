# Results: SmolVLA end-to-end baseline on SoftHier

Design: [docs/designs/smolvla-softhier-baseline.md](../../docs/designs/smolvla-softhier-baseline.md) (outcomes: "Outcomes and deviations from plan (G5)"). Step-by-step trace: [LOG.md](LOG.md). Reproduction: [SETUP.md](SETUP.md). Input configurations: [INPUT_CONFIG.md](INPUT_CONFIG.md).

## Evidence labels

The wiki glossary (`wiki.py glossary "measured"`, decision 0004) defines **measured** as a reported run and **estimate** as a figure computed from specs or configs. Here:

*   **measured**: a simulator run or an evaluation run that finished and whose output is quoted. Simulated times are measured on the GVSoC simulator (1 GHz, ideal HBM), not on hardware.
*   **estimate**: a number composed or projected from measured pieces (the G4a latency, the G4b wall-time projection).

## Headline numbers

| result | value | label | source |
| --- | --- | --- | --- |
| G3 gate (published configuration, `lerobot/smolvla_base`, 1 camera, 256 tokens) | **PASS**: 120.716 ms vs 119.1 ms published (+1.4 %, window 113.14..125.05); action max-abs 0.0068 vs LeRobot (limit 0.10, published 0.032) | measured (simulated) | LOG G3.7-gate; run `$SHVLA_DATA/runs/published/fixi-rung6` |
| LIBERO latency (G4a, `lerobot/smolvla_libero`, 2 cameras, 1024 tokens each) | **479.1 ms** per action chunk (vision 351.8 + connector 0.53 + prefix 33.0 + expert 93.8); about 3.97x the published configuration; 19 % of the 2.5 s a 50-action chunk covers at 20 Hz (synchronous, no overlap) | **estimate** (composed from partial simulated runs; G4b, the full chain, was not run) | LOG G4a and its addendum; `$SHVLA_DATA/runs/libero/g4a_estimate.json` |
| LIBERO success rate (G6b, `libero_spatial`, 10 tasks x 10 episodes, GPU fp32, batch 2, seed 1000, synchronous) | **59/100 = 59.0 %**, Wilson 95 % interval [49.2, 68.1]. No published figure for this checkpoint on this suite (model card has none; the paper's LIBERO average of 87.3 is for another model and the four-suite mean), so the gap is unexplained and not a verdict | measured | LOG G6b.1; `$SHVLA_DATA/eval/g6b_spatial_10x10_b2/summary.json` |
| G2 golden crosscheck | 48 shared numeric keys (16 non-KV + 32 KV) relative error 0.000e+00 against softhier-mlir's reference, tolerance 1e-4 | measured | LOG G2.3 |
| Wall time and memory | see the last section | measured | LOG G3.7-gate, G4a, G6b.1 |

The published and LIBERO latencies are different checkpoints and configurations. The LIBERO figure carries no published comparison (softhier-mlir has none for it).

## G3: published-configuration latency (measured)

Run `fixi-rung6`: `smolvla-e2e --npz e2e_c1_t256.npz --e2e-trace`, four simulator processes, exit 0, `gate.py g3` PASS. Times in ms at 1 GHz with ideal HBM.

| phase | published | before the `instr_mem` fix (`fixrm-rung6`) | `fixi-rung6` | vs published |
| --- | --- | --- | --- | --- |
| vision | 27.32 | 19.877 | 27.640 | +1.2 % |
| connector | 0.28 | 0.277 | 0.280 | 0 |
| prefix (16 layers) | 14.80 | 11.188 | 15.039 | +1.6 % |
| expert (KV projection 0.137 + 10 steps x 7.762) | 76.70 | 59.914 | 77.756 | +1.4 % |
| **total** | **119.1** | 91.139 (-23.5 %, gate MISS) | **120.716** | **+1.4 %** |

Ladder state at the end (all rungs exit 0):

| rung | what | result | note |
| --- | --- | --- | --- |
| 1 | `gemm`, 7 shapes | 7/7 PASS; 1024x3072x768 520479 ns vs published 520.5 us | re-run after the `instr_mem` fix (`fixi-rung1`) |
| 2 | `mlir examples/*.mlir`, 9 files | 8 self-checks PASS, `siglip_encoder_layer` ran to the end (no self-check; 18.00 ms vs published 17.85 ms, +0.85 %) | re-run after the fix |
| c | `siglip-mlir --seq 256 --cluster all` | 11 tags PASS; marks 0.324 / 0.840 / 1.040 ms, layer 2.204 ms vs 2.195 ms published (+0.4 %) | re-run after the fix |
| 3 | `smolvla --layers 1` | PASS: EMB 0.0006, layer 1 0.0072, OUT 0.0285 | last run before the fix: **timing predates it** (1.625 ms/layer then; published 2.29) |
| 4 | `smolvla-vlm --layers 1` | PASS: EMB 0.62 (|ref| max 2213), L1 0.48, K1 0.0095, V1 0.0005 | timing predates the fix |
| 5 | `expert.py flow --steps 1` | PASS: X0 0.0016 (fp16 floor 0.0020) | timing predates the fix |
| 6 | `smolvla-e2e` | PASS, table above | re-run after the fix |

Rowops (`rowops --rows 256 --cols 768`, all 16 clusters), published vs ours with the `instr_mem` fix in the shim run (G3.3-gap, run C; the same number as the fixed tree): LN 149 / 148.7 us, softmax 145 / 145.1, GELU 123 / 122.2, add 22 / 22.1, bias 27 / 27.1, scale 17 / 16.8, **iDMA transpose 25 / 58.2 (open residual)**.

## G4a: LIBERO latency (estimate)

Three partial simulated runs, all exit 0, composed by `gate.py g4a` (stamp OK, GVSoC base `d806a84` + patch series; dev commit `f99d5a2`, same tree). Times in ms at 1 GHz, ideal HBM.

| run | what | simulated (timed segments) | wall | peak memory (cgroup) |
| --- | --- | --- | --- | --- |
| G4a.1 | 1 SigLIP layer at 1024 tokens: embed 0.230, attention 10.844, MLP 3.751, post-LN 0.550 (connector, 1 camera, 0.395: cross-check only) | 15.8 | 600 s | 1196 MB |
| G4a.2 | connector 0.534 + prefix layers 1-8 at 177 tokens, 16.477 (2.060 ms/layer) | 17.0 | 616 s | 1563 MB |
| G4a.3 | expert: KV projection 0.245 + 9.356 ms/step, 2 steps, KV from G2 | 19.0 | 586 s | 1819 MB |

| phase | ms (**estimate**) | formula |
| --- | --- | --- |
| vision | 351.827 | 2 x (0.230 + 12 x 14.594 + 0.550) |
| connector | 0.534 | G4a.2 start to embedding |
| prefix | 32.954 | 2 x 16.477 |
| expert | 93.801 | 0.245 + 10 x 9.356 |
| **total** | **479.117** | |

Limits of the estimate: one vision layer stands for 12 layers x 2 cameras (layer-to-layer variation up to 7 % in the published runs, the first layer is cold); the prefix and expert partial runs are 8 of 16 layers and 2 of 10 steps scaled; no published reference exists for this configuration.

Accuracy of the partial runs: vision output vs the fp16 floor max-abs 0.1633 (|ref| max 44.10, threshold 0.03 x max|ref|), PASS; expert actions vs the fp16 floor 0.0029 and vs G2 `xt[2]` 0.0032 (limit 0.10), PASS. **The prefix partial run (G4a.2) has no accuracy check**, and the expert is compared at `xt[2]` only (see residuals).

G4b projection (estimate): vision 13386 s = 3.72 h, over the 7200 s limit (R1); prefix 616 s and 597 s; expert 2900 s. Under rule R1 the session stopped. **Owner decision: G4b is not run, G4a is the LIBERO latency result.**

## Accuracy figures

| check | value | label |
| --- | --- | --- |
| `golden_ref.py` vs softhier-mlir reference (G2.3, 48 keys) | relative error 0 on every key. Both programs run the same LeRobot modules on the same CPU, so this validates capture points and layout (KV reshape, cache read, `n_valid`, flow quantities), not two independent numerical implementations | measured |
| `golden_ref.py` libero mode vs `predict_action_chunk` (G2.4 test d) | max-abs 0.000e+00 (limit 1e-5); `infra/lerobot` tests 12 passed in 96.6 s | measured |
| rung 6 actions vs LeRobot | max-abs 0.0068 (mean 0.00067, |a| max 1.576); fp16 floor 0.0015, floor vs LeRobot 0.0070; x_t per step 0.0018 to 0.0068 | measured |
| rung 6 hand-overs | VIS 0.0488 (|ref| max 23.6), IMG 0.3986 (|ref| max 57.7), prefix K max-abs at most 0.097, V at most 0.056 | measured |
| `chain_outputs.npz`, `fixrm-rung6` vs `fixi-rung6` | 36 arrays bit-identical (the timing fix does not touch values) | measured |
| G4a vision / expert | see above | measured |
| G6b | success rate 59.0 % (above) | measured |

## Wall time and memory

| item | value | source |
| --- | --- | --- |
| `setup.sh` (`make softhier_preparation`) | 1857 s; GVSoC tree 4.8 GB, 5.4 GB after the first build | G1.3, G1.4 |
| first `make hw` / incremental rebuild | 98 s / 4-10 s | G1.4, G3 |
| rung 6 chain (`fixi-rung6`) | 4914 s = 82 min: vision 1290 s, prefix_a 567 s, prefix_b 582 s, expert 2475 s (one simulator process each, serial); peak 2614 MB (cgroup incl. page cache), max child RSS 1486 MB | G3.7-gate |
| G4a runs | about 600 s each; 30-38 s of wall per simulated ms (rung 6 vision: 47 s/ms) | G4a |
| G6b evaluation | 3781 s (63 min), 37.6 s/episode, measured under CPU contention from a concurrent simulator job (MuJoCo renders on the CPU); 3 processes of about 3 GB each; largest process 4145 MiB; a batch-5 attempt was OOM-killed | G6b.1 |
| `golden_ref.py` | about 3.8 GiB RSS, 1.8 GiB GPU | `infra/lerobot/README.md` |

## Simulator fixes found

The GVSoC fixes are a patch series, [`infra/softhier/gvsoc-patches/`](../../infra/softhier/gvsoc-patches/README.md), on upstream `soft_hier_release` [`d806a84`](https://github.com/gvsoc/gvsoc/tree/d806a84c5785237547f504c2e34a4d9f230cb437); the submodule is pinned at that base and `setup.sh` applies the series. The hashes below (`4558184` .. `f99d5a2`) are the commit ids on the original local dev branch `shvla/soft_hier_release`, which was never pushed; the links point at the corresponding patch file. The tree of base + series is identical to the tree of `f99d5a2` that produced every measured number (`def18fe6856947bab3009fa551cc43282c5ddbf1`). F numbers are the design's findings table; "SN #" is softhier-mlir's SIMULATOR_NOTES number.

| F | GVSoC commit | symptom | root cause | evidence | upstream? |
| --- | --- | --- | --- | --- | --- |
| F13 | [`0e4d719`](../../infra/softhier/gvsoc-patches/0003-shvla-iDMA-backend-re-arm-FSM-in-IDmaBe-update-fixes.patch) iDMA fsm kick | `gemm` with K tile 256 never terminates; memory grows about 300 MB/s until the cap | release `IDmaBe::update()` enqueues `transfer_regulation_event` instead of `fsm_event`; the backend FSM is not re-armed | G3.1-diag3 (deeploy A/B: swapping only the iDMA `.so` flips pass/hang both ways) | **yes** |
| F14 | [`2910e62`](../../infra/softhier/gvsoc-patches/0005-shvla-iDMA-backend-can_accept_transfer-also-requires.patch) iDMA `can_accept_transfer` race | `gemm1024_summa`, `siglip_encoder_layer` hang in `hbm_fill`; DRAMSys runs grow to the cap | `can_accept_transfer()` checks only `current_transfer_size == 0`; the regulation queue forwards a cycle later, so the next row overwrites the first (283 of 1024 rows lost) | G3.1-diag4 (A/B matrix; the 283 lost rows are exactly those queued at regulation depth 1) | **yes** |
| F5 (SN #5) | [`cdb56a2`](../../infra/softhier/gvsoc-patches/0004-soft_hier-SOFTHIER_IDEAL_HBM-1-replaces-DRAMSys-HBM-.patch) ideal-HBM switch | DRAMSys always instantiated; 16-cluster GEMMs +46-48 % vs published | `SOFTHIER_IDEAL_HBM` read by no GVSoC code | G3.1-ihbm (1024x3072x768 520114 ns vs 520.5 us) | feature the stock tree lacks; optional |
| F8 (SN #10) | [`070936b`](../../infra/softhier/gvsoc-patches/0006-shvla-ISS-ISA-tables-nseq-tag-on-scalar-FP-instructi.patch) `nseq` tag | softmax, LayerNorm, GELU depend on binary layout | scalar FP-to-int instructions lack the tag | G3.2-fix10: timing +1 % only, accuracy failure unchanged | no effect here; follows the owner's notes |
| F15 | [`cd200ad`](../../infra/softhier/gvsoc-patches/0007-shvla-RedMulE-model-fp16-subnormal-conversion-fix-fl.patch) RedMulE fp16 subnormal | rung c, 3, 4, 5 fail: sparse corrupt GEMM elements (Q[23,273] = -2.0 instead of 4.6e-5), inf and NaN downstream | `float_to_fp16` shifts subnormals wrongly (`1 - exponent` for `14 - exponent`, no mask); `fp16_to_float` mis-decodes them | G3.2-diag7, diag8, fix-redmule (576021-case offline unit test; ladder PASS after the fix) | **yes**; the published accuracy may have been produced with the bug |
| F3 (SN #3) | [`7c9cd2f`](../../infra/softhier/gvsoc-patches/0008-shvla-RedMulE-model-float_to_fp16-IEEE-round-to-near.patch) RedMulE RNE | truncating `float_to_fp16` | rounding mode | G3.2-fix-redmule | with F15 |
| F1 (SN #1) | [`fd1124a`](../../infra/softhier/gvsoc-patches/0009-shvla-snitch-fp_ss-Iss-handle_event-waits-while-exec.patch) fp_ss pending | stale `flw`/`fsw` to HBM or remote TCDM | FP subsystem completes an offloaded access without waiting | G3.3-fix1: no ROI changed by one ns (the workload's FP accesses are local TCDM, synchronous). Kept as a correctness fix, no timing effect | harmless; owner's notes |
| F16 | [`f99d5a2`](../../infra/softhier/gvsoc-patches/0010-G3.3-gvsoc-instr_mem-width_log2-default-fetch-bandwi.patch) `instr_mem` `width_log2` | whole-chain -23.5 % vs published; rowops LN/SM/GELU 30-50 % fast | `cluster_unit.py:209` passes `width_log2=-1` (`size >> -1` undefined, no fetch bandwidth limit); deeploy uses the default 2 | G3.3-gap (shim reproduces published rowops within 1 %), G3.3-fix-instr (rung c +0.4 %, rung 6 +1.4 %) | **yes** |
| F17 | [`4558184`](../../infra/softhier/gvsoc-patches/0001-G3.1-sdk-add-flex_alloc_api.h-from-bowwang-dev-softh.patch) `flex_alloc_api.h` | header missing from `soft_hier_release` | exists only on `bowwang-dev/softhier-deeploy` | G3.1 | **yes** |
| F18 | [`d42c382`](../../infra/softhier/gvsoc-patches/0002-G3.1-power-capture-only-with-GVSOC_POWER_CAPTURE-1-o.patch) power-capture gate (and `pwr_interval_ps` in `build_arch.sh`, infra `18ad903`) | simulations over 100 us ran `make ice_power` | capture at the compiled interval | G3.1 follow-up: removing it **did not** stop the memory growth (that was F13/F14); kept because harmless | no |

Also patch 0006 on softhier-mlir (opt-in SigLIP dumps) and patches 0001-0005 (paths, timeout flag, expert KV source, LeRobot 0.7 port; patch 0003 also found that the older `vision_position_ids` formula puts rows one bucket low under transformers 5). `patches/README.md` marks the upstream candidates.

## Open residuals

| item | state |
| --- | --- |
| iDMA transpose (rowops `T`) is 2.3x slower than published: 58.2 us vs 25 us | not investigated (F19). Candidates: the iDMA 2-D path (the kernel uses the 2-D gather with `SH_TRANSPOSE_CORES=0`) or the extra `TransposeEngine` unit. Marks stay within 1.1 % of published, so the effect on the chain is small; not verified |
| G4a.2 has no prefix accuracy check | `run_prefix_cost` writes no accuracy dump; the PASS only means the run finished. Also the expert is compared at `xt[2]` only, not `xt[1..2]`. Fixing either exceeds patches 0001-0005 (rule R3) |
| G4b not run | owner chose to report G4a only (R1: vision projected 3.72 h > 2 h). No full-chain LIBERO measurement and no G2 comparison at every hand-over for the LIBERO configuration |
| Rungs 3-5 were not re-run after the `instr_mem` fix | their accuracy is unaffected by it (timing model only; chain outputs were bit-identical in rung 6), but their timing numbers predate it (rung 3 1.625 ms/layer vs 2.265 in the shim run) |
| Kernel tests (G1.5) | no output check, run on DRAMSys before F5 and F16; timings not comparable |
| G6b success rate 59.0 % has no published comparison | hypotheses not tried: training `fps` 30 vs eval 20 Hz, 50-step open-loop chunks, 360 vs 256 camera size, `max_steps` 280. The wall time was measured under contention |
| GVSoC fixes: no fork | resolved (G5.2): the fixes are the patch series in the repository; record-block permalinks now name the upstream base and the series hash |
| **Lesson: 64-sample checks hide sparse errors** | softhier-mlir's end-of-program dumps compare 64 sampled elements. F15 corrupted a handful of elements (4 distinct bad ones in the raw Q, K, V outputs, among about 67 000 sampled each), so a 64-point pass said nothing. Only the 150 000-point dump (patch 0006, `SHVLA_SIGLIP_H_FULL`) found it. Earlier "passing" SmolVLA stages from the 64-point checks were therefore not evidence of correctness |

## Provenance

*   Simulator: GVSoC upstream base [`d806a84`](https://github.com/gvsoc/gvsoc/tree/d806a84c5785237547f504c2e34a4d9f230cb437) plus the patch series [`infra/softhier/gvsoc-patches/`](../../infra/softhier/gvsoc-patches/README.md) (ten patches, series SHA-256 `6116080fc3842bf8...`, tree `def18fe6856947bab3009fa551cc43282c5ddbf1`; the measured runs used the dev commit `f99d5a2`, same tree), `core` [bda268a](https://github.com/gvsoc/gvsoc-core/tree/bda268a4808708a48eab21981c7d6bbced1577d7), `pulp` [3ca2016](https://github.com/gvsoc/gvsoc-pulp/tree/3ca2016d056dda72317e27c4319fc6174b5f8c4d). Stamp: arch `arch/softhier_mlir_arch.py` SHA-256 `bea6c2c4728e1dcdd3a966d9e6fa4068f8ca292cd5f2dbd8552bab8a9fbb134a`, `pwr_interval_ps` 9e15, `models_fast` and `models_fix` absent, `SOFTHIER_MODEL_DIR` and `SOFTHIER_STOCK_MODELS` unset.
*   Compiler: softhier-mlir [b11cd5f](https://github.com/runwangdl/softhier-mlir/tree/b11cd5fa4f3e717374d1e9ba14442ff302d1f970) plus patches 0001-0006 (`run.sh` record block: 0001 `145e81bb`, 0002 `1b9c65fe`, 0003 `0664b49e`, 0004 `2cd6b752`, 0005 `ec77f4bb`, 0006 `4575085c`).
*   Checkpoints: `lerobot/smolvla_base` `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`; `lerobot/smolvla_libero` `a24f1f380461b665158d52acd140152c79e8b36d`. LeRobot 0.7.0, transformers 5.22.2.
*   Arch: 4x4 clusters, 3 cores each, RedMulE 128x32, NoC link 512, instruction memory 64 KB, HBM channel placement [4, 0, 0, 4], 1 GHz, ideal HBM.
*   Timeouts: `fixi-rung6` was started with `SHVLA_SIM_TIMEOUT=14400` (no owner approval for a raised limit is recorded; no phase exceeded 2475 s); G4a used the default 7200 s.
