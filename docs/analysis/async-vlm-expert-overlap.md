# VLM / action-expert overlap on W1 and H1 (estimate)

What a V-AEFusion-style overlap of the VLM and the action expert would gain on SmolVLA W1
(`vlabench-3cam`) on H1, computed from the stage-1 kernel sums. **Label: estimate** throughout
([conventions](conventions.md)). Numbers come from
[`async_overlap_estimate.py`](async_overlap_estimate.py); rerun it to reproduce every figure and table here.

## Source of the method

Zhou et al. (SJTU), arXiv 2604.24447, §5.3, wiki note
`wiki://resources/serving/vla-xpu-characterization-sjtu.md`. Within one inference, while the VLM
processes the current observation, the action expert runs its first k denoising steps ("stale steps")
on the previous observation's KV cache, then switches to the fresh KV cache for the remaining steps.
The paper picks k = 5 of 10 on π0. Its real-robot ablation (Table 7) keeps success near baseline up to
5 stale steps and collapses past 6 (7 steps: 50 / 42 %; 9 steps: 10 / 6 %). Speedup is 1.32x on an
RTX 4090, 1.14x on Orin, 1.04x on Thor and 1.0x on the Ascend 310P. This is distinct from overlapping
inference with robot execution (RTC, SmolVLA async inference).

## Inputs

From [smolvla-op-graph.md](smolvla-op-graph.md) (W1, H1, kernel sums per stage, no overlap):

| quantity | value | how |
| --- | ---: | --- |
| VLM part (vision + connector + prefix) | 17.86 ms | 14.69 + 0.17 + 3.01 |
| cross K/V projection | 0.09 ms | once per call; needs the fresh prefix, so it cannot run early |
| expert, per flow step | 1.728 ms | (17.37 - 0.09) / 10 |
| baseline per call | 35.24 ms | stage-1 composed estimate |

Model: T = max(T_VLM, k x e_stale) + cross K/V + (10 - k) x e_fresh.

## Scenarios

| | overlap phase (VLM with stale steps) | fresh-step phase (after the VLM) |
| --- | --- | --- |
| **A** ideal | VLM and expert both run at their stage-1 speed, no interference (an upper bound, not a mapping) | stage-1 mapping |
| **B** split | VLM on 64 - c clusters, time scaled by 64 / (64 - c) (pessimistic); the expert's GEMMs, attention and row ops all confined to c clusters (GEMMs at the picked time, which already uses 8-16 clusters; attention and row ops scaled by 64 / c) | stage-1 mapping |
| **C** tuned GEMMs, ideal | as A | expert GEMMs on the full mesh: the `gemm_smallm_sweep.py` first result (17.7 vs 32.4 us) applied to every expert GEMM (assumption); step 1.728 -> 1.113 ms |
| **C+B** | as B | as C |

A and B differ only in the overlap phase. C also changes the fresh-step mapping, and an expert that
wants the full mesh makes the overlap phase harder, so C+B is the version a real mapping can reach.

## Figure 1: one policy call (2 characters = 1 ms, k = 5, c = 16)

```text
             0         5         10        15        20        25        30        35        ms
             |----+----|----+----|----+----|----+----|----+----|----+----|----+----|----+----
Baseline     VVVVVVVVVVVVVVVVVVVVVVVVVVVVVPPPPPPP1112222333444455566667778888999000      -> 35.24 ms

A  VLM       VVVVVVVVVVVVVVVVVVVVVVVVVVVVVPPPPPPP
   expert    11122223334444555...................66677778889999000                       -> 26.60 ms

B  VLM(48)   VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
   expert(16)1111112222233333344444455555....................66677778889999000           -> 32.55 ms

C  VLM       VVVVVVVVVVVVVVVVVVVVVVVVVVVVVPPPPPPP
   expert    11223334455.........................66778889900                             -> 23.52 ms

C+B VLM(48)  VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
   expert(16)1111112222233333344444455555....................66778889900                 -> 29.48 ms
             '- steps 1-5: stale, KV(o_n-1) -'    '- steps 6-0: fresh, KV(o_n) -'

V = vision   P = connector + prefix   digit = flow step   . = waiting for the VLM
(cross K/V, 0.09 ms, is narrower than one character)
```

## Figure 2: the call against the controller (1 character = 1 control tick)

Illustration at 30 Hz (control period 33.3 ms, the rate SmolVLA's async inference assumes) with a
50-action chunk (chunk duration 1.67 s) and a new call every m = 20 ticks; m is an example.

```text
tick          0    5    10   15   20   25   30   35   40
              |----|----|----|----|----|----|----|----|
observation   *n-1                *n                  *n+1
policy call   #                   #                   #         # = one call, l = 0.8-1.06 tick
                                  |-- stale steps read KV(o_n-1): m x 33.3 ms = 667 ms old, not l
                                  |-- fresh steps read KV(o_n)
chunk ready    v(n-1)              v(n)                v(n+1)
controller    .aaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbb.ccc...   one action per tick from the queue
                                   ^ the first ceil(l / 33.3 ms) actions of a new chunk are stale
                                     baseline l = 35.2 ms -> 2; A, C, B (32.6 ms) -> 1
only the first m actions of each 50-action chunk run; the next chunk replaces the rest
```

- At 30 Hz the call latency is far below the chunk duration, so throughput is not the limit. The
  in-call gain shows up as fewer stale actions (2 -> 1) and a fresher observation behind each action.
- Replanning every tick (m = 1) needs l < 33.3 ms: the baseline (35.2 ms) misses it; A, B, C and C+B meet it.
- The age of the KV that the stale steps read is the call interval m x 33.3 ms, not l. The paper's
  97.99 % / 96.91 % KV similarity is between adjacent inferences at an unstated interval. A larger m
  means staler stale steps; a smaller m keeps the hardware busy and costs energy.
- SmolVLA puts the robot state in the VLM prefix, and its self-attention layers also read the prefix
  KV (Skv = 291). Stale steps therefore see an old state as well as old images, a wider exposure than
  in π0. No source measures SmolVLA's accuracy under stale steps.

## Figure 3: spatial mapping per phase (8 x 8 mesh)

```text
            (1) overlap: VLM with stale steps     (2) fresh steps (expert alternates within a layer)
                                                       GEMM moment      attn / row moment
A           XXXXXXXX  both at their stage-1           EEEE....         EEEEEEEE
(bound)     XXXXXXXX  mappings, assumed not to        EEEE....         EEEEEEEE
            XXXXXXXX  slow each other down            EEEE....         EEEEEEEE
            XXXXXXXX  (not physically reachable)      EEEE....         EEEEEEEE
            XXXXXXXX                                  ........         EEEEEEEE
            XXXXXXXX                                  ........         EEEEEEEE
            XXXXXXXX                                  ........         EEEEEEEE
            XXXXXXXX                                  ........         EEEEEEEE
                                                      picked expert GEMMs use 8-16 clusters

B           VVVVVVVV  VLM on 48 clusters              as A             as A
            VVVVVVVV  (time x 1.33, linear)
            VVVVVVVV
            VVVVVVVV
            VVVVVVVV
            VVVVVVVV
            EEEEEEEE  expert GEMM, attention and
            EEEEEEEE  row ops on 16 (1.73 -> 2.85 ms per step)

C           as A, but the expert GEMMs now also       EEEEEEEE         EEEEEEEE
(ideal)     want all 64 clusters                      (all 64)         (all 64)
                                                      1.73 -> 1.11 ms per step

C+B         as B                                      as C             as C

X = VLM and expert time-share the cluster   V = VLM   E = expert   . = idle
The placement of the 16 GEMM clusters is illustrative; stage 1 records the count, not the position.
```

## Results

| scenario | k | c | per call, ms | vs stage-1 baseline | vs its own no-overlap baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| A ideal | 3 | - | 30.05 | 1.17x | 1.17x |
| A ideal | 5 | - | 26.60 | 1.32x | 1.32x |
| B split | 3 | 8 | 32.61 | 1.08x | 1.08x |
| B split | 5 | 8 | 37.21 | 0.95x | 0.95x |
| B split | 3 | 16 | 36.01 | 0.98x | 0.98x |
| B split | 5 | 16 | 32.55 | 1.08x | 1.08x |
| C ideal | 3 | - | 25.75 | 1.37x | 1.13x |
| C+B split | 3 | 16 | 31.70 | 1.11x | 0.92x |
| C ideal | 5 | - | 23.52 | 1.50x | 1.24x |
| C+B split | 5 | 16 | 29.48 | 1.20x | 0.99x |

The no-overlap baseline for C and C+B is the tuned-GEMM call, 29.09 ms.

**Criterion.** With a spatial split, the overlap gains

    gain = k x e_fresh - [T_VLM(64 - c) - T_VLM(64)],  subject to  k x e_stale(c) <= T_VLM(64 - c)

| scenario (k = 5, c = 16) | VLM cost of ceding c clusters | saved fresh-step time | net |
| --- | ---: | ---: | ---: |
| A | 0 | 8.64 ms | 8.64 ms |
| B | 5.96 ms | 8.64 ms | 2.68 ms |
| C+B | 5.96 ms | 5.57 ms | -0.39 ms |

A faster expert leaves less to hide, so once the expert GEMMs use the full mesh the overlap pays only
if the VLM slows down by less than linear when it gives up clusters.

- HBM bandwidth is not the limit: the expert streams about 114 GB/s during its steps, vision about
  40 GB/s, against 1,024 GB/s. NoC contention is not modelled.
- At k = 5 the A result (1.32x) matches the paper's RTX 4090 figure, because the VLM-to-expert time
  ratio is close in both cases (17.9 : 17.4 ms here, about 1 : 2 for π0 on the RTX 4090).

## Open measurements

1. T_VLM on 48 and 56 clusters (vision and prefix kernels re-swept on a cluster subset), to replace the
   linear assumption in B and C+B. This decides whether the overlap is worth building.
2. Expert attention and row kernels on 8 and 16 clusters, to replace the 64 / c scaling of e_stale.
3. The two cluster groups running concurrently in GVSoC, to capture NoC and HBM contention.
4. SmolVLA accuracy with k stale steps (LIBERO or VLABench), since the paper's ablation is π0 only.
