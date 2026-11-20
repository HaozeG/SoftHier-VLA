# VLM / action-expert overlap and DP-Cache on W1 and H1 (estimate)

What V-AEFusion (overlap the VLM with the first flow steps) and DP-Cache (reuse flow-step results)
would gain on SmolVLA W1 (`vlabench-3cam`) on H1, computed from the stage-1 kernel sums after
`dcd221d` (expert GEMMs on the full mesh, 28.99 ms per call). **Label: estimate** throughout
([conventions](conventions.md)). Every number comes from
[`async_overlap_estimate.py`](async_overlap_estimate.py) (`--json` for the page data). Figures:
[`async-overlap-timeline.html`](async-overlap-timeline.html), published as a private artifact.

## Source of the methods

Zhou et al. (SJTU), arXiv 2604.24447; wiki note `wiki://resources/serving/vla-xpu-characterization-sjtu.md`.

- **V-AEFusion (§5.3, Fig. 9).** Within one inference, while the VLM processes the current
  observation, the action expert runs its first k denoising steps ("stale steps") on the previous
  observation's KV cache, then switches to the fresh cache. The paper picks k = 5 of 10 on π0; its
  real-robot ablation (Table 7) collapses past 6 stale steps. Speedup: 1.32x RTX 4090, 1.14x Orin,
  1.04x Thor, 1.0x Ascend 310P. Distinct from overlapping inference with robot execution (RTC,
  SmolVLA async inference).
- **DP-Cache (§5.2.2, Fig. 7).** Inside an offline-fixed "stable segment" of the diffusion, a
  computed result is reused for the next S - 1 steps. Segment given only for a 100-step diffusion
  (steps 20-80); π0 102 -> 79 ms (1.29x).

## Inputs

| quantity | value | source |
| --- | ---: | --- |
| VLM part (vision + connector + prefix) | 17.86 ms | [smolvla-op-graph.md](smolvla-op-graph.md): 14.69 + 0.17 + 3.01 |
| cross K/V projection | 0.09 ms | once per call; needs the fresh prefix |
| expert flow step, full mesh | 1.104 ms | (11.13 - 0.09) / 10 |
| expert flow step on 8 clusters | 2.93-4.14 ms | one row of 8 at the origin: measured GEMMs that fit, attention heads in rounds (below); row ops x1 to x8 |
| expert flow step on 16 clusters | 1.86-2.38 ms | two rows of 8 at the origin, as above; row ops x1 to x4 |
| baseline per call | 28.99 ms | stage-1 composed estimate |

Sub-mesh expert kernels. SummaGEMM and FlatAttention tile the mesh from cluster (0, 0) with
power-of-two groups (`grid_sync_group_init`: group id = (8 / gx) (y / gy) + x / gx; ids below the
group count are active), so a sub-mesh time counts only if its active groups lie inside one fixed
footprint: one row of 8 for c = 8, two rows of 8 for c = 16.

- GEMMs: the fastest checked setting in `gemm_sweep_h1.csv` / `gemm_smallm_h1.csv` whose groups fit
  (measured): 2.18 ms per step on the row of 8 (1 x 1 or 1 x 2 groups; the 2 x 2 settings need a
  4 x 2 block), 1.35 ms on two rows, 0.73 ms on the full mesh.
- Attention: the picked `coll_s8x1` spreads the 5 KV heads (GQA folded) over 5 groups of 8 x 1, i.e.
  40 clusters, so no sub-mesh time was measured. Inside the footprint fewer groups fit and the heads
  run in ceil(5 / groups) rounds at the measured per-head time (estimate): 0.57 ms per step on the
  row of 8, 0.34 ms on two rows, 0.20 ms on the full mesh.

Model: T = max(T_VLM(64 - c), n_stale x e_stale(c)) + cross K/V + n_fresh x e, with k <= 5.

## Results

| scenario | overlap phase | stale / fresh / reused | per call | vs 28.99 ms | one-tick rate |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline | none | 0 / 10 / 0 | 28.99 ms | 1.00x | 34.5 Hz |
| DP-Cache | none | 0 / 6 / 4 | 24.58 ms | 1.18x | 40.7 Hz |
| Overlap, ideal | both at full speed (bound) | 5 / 5 / 0 | 23.47 ms | 1.24x | 42.6 Hz |
| Overlap, 8-cluster split | VLM on 56 (20.42 ms), expert on 8 | 5 / 5 / 0 | 26.30 ms | 1.10x | 38.0 Hz |
| Overlap, 16-cluster split | VLM on 48 (23.82 ms), expert on 16 | 5 / 5 / 0 | 29.43 ms | 0.985x | 34.0 Hz |
| Overlap, ideal + DP-Cache | bound | 3 / 3 / 4 | 21.27 ms | 1.36x | 47.0 Hz |
| Overlap, 8-cluster split + DP-Cache | VLM on 56, expert on 8 | 3 / 3 / 4 | 23.82 ms | 1.22x | 42.0 Hz |

One-tick rate: the highest control rate at which a call still fits in one control period (1 / T).

- **The earlier "tuned GEMMs" variant is now the baseline.** The earlier estimate assumed full-mesh expert GEMMs
  at 29.09 ms; `dcd221d` measured 28.99 ms. The overlap variants therefore differ only in the overlap phase: the
  fresh-step phase is the full-mesh expert in every scenario.
- **Both sides limit a split.** The split gains while the VLM loses less than the hidden expert
  time, 5 x 1.10 = 5.52 ms, i.e. a VLM slowdown of at most 1.31x. With linear scaling, c = 16 costs
  5.96 ms (net loss). At c = 8 the VLM costs 2.55 ms, and the stale steps on the row of 8 take
  2.93-4.14 ms each; at the upper end five of them (20.7 ms) outlast the VLM (20.42 ms), giving 1.10x.
- **The linear VLM slowdown is likely optimistic.** The kernels start their groups at cluster (0, 0)
  with no origin offset, so the VLM cannot yet run on "the other 56" at all, and power-of-two groups
  tile only part of an 8 x 7 region (4 x 4 groups: 8 x 4 = 32 clusters; 2 x 2 groups: 8 x 6 = 48). At
  an effective 48 clusters (1.33x) the 8-cluster split also loses.
- **Overlap and DP-Cache do not multiply** (1.18 x 1.24 = 1.46, model 1.36). Under overlap the stale
  window is hidden behind the VLM, so reusing steps 4 and 5 saves nothing; only reuse in the fresh tail
  (7, 8) shortens the call. Put reused steps in the fresh tail.
- **The ideal overlap has no physical mapping on SoftHier**: a cluster runs one kernel at a time. It is a bound.

## The controller

The test environment runs at about 20 Hz (LIBERO 20 Hz; VLABench 10 Hz). At 10, 20 and 30 Hz every
scenario fits in one control tick; at 50 Hz every one needs two. So neither method changes the number
of stale actions at these rates.

- **Stop-and-go** (the paper's Fig. 9 view, robot stops during inference): the pause shrinks from 29.0
  to 21.3 ms in a cycle of 25 executed actions (1,250 ms at 20 Hz), 2.3 % -> 1.7 % of the time. In
  simulation the world pauses during inference, so this does not show in success rates.
- **Asynchronous execution**: latency is observation staleness (29.0 -> 21.3 ms), hidden under the
  action queue.
- **Age of the stale KV** is one replan interval (m ticks; 25 ticks = 1,250 ms at 20 Hz in the example), not one
  latency. The paper's KV similarity (97.99 % / 96.91 %) is measured over the same interval, a chunk's
  execution.
- SmolVLA puts the robot state in the VLM prefix and its self-attention layers also read the prefix
  KV (Skv = 291), so stale steps see an old state as well as old images. No source measures SmolVLA's
  accuracy under stale steps or under the combination with DP-Cache.

## Spatial mapping per phase

| scenario | phase 1: VLM (with stale steps) | phase 2: fresh steps |
| --- | --- | --- |
| Baseline / DP-Cache | VLM on 64 | expert on 64 (6 steps with DP-Cache) |
| Overlap, ideal | both on 64, no slowdown (bound) | expert on 64 |
| Overlap, 8-cluster split | VLM on rows 1-7, stale steps on row 0 (origin) | expert on 64 |
| Overlap, 16-cluster split | VLM on rows 2-7, stale steps on rows 0-1 (origin) | expert on 64 |

The stale-step group takes the origin rows because the kernels place groups from cluster (0, 0);
which HBM edge those rows touch is not checked here.
The reconfiguration from the c-cluster group to the full mesh at the KV switch is not modelled.

## Assumptions

- VLM on 64 - c clusters slows by 64 / (64 - c) (not measured).
- Expert row operators on c clusters: between unchanged and 64 / c slower (the table uses 64 / c).
- DP-Cache reuses 4 of 10 steps (4, 5 reuse 3; 7, 8 reuse 6), calibrated to the paper's π0 saving
  (about 38 % of expert time); not a SmolVLA measurement.
- NoC and HBM contention between the two groups is not modelled. HBM traffic is low at these rates
  (the expert streams about 197 MB per step).

## Open measurements

1. VLM kernels (vision and prefix) on 56 and 48 clusters. This decides whether the split pays.
2. Expert attention and row kernels confined to one and two rows (attention needs a head-to-group
   limit), to replace the head-rounds estimate and the row-op range.
3. The two cluster groups running concurrently in GVSoC, for NoC and HBM contention and the cost of
   the KV-switch reconfiguration.
4. SmolVLA accuracy with k stale steps and with step reuse (LIBERO or VLABench).
