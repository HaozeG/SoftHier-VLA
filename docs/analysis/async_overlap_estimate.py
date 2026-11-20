"""V-AEFusion-style VLM/expert overlap on W1 (vlabench-3cam) and H1: estimate and ASCII timelines.

Inputs are the stage-1 kernel sums per stage in smolvla-op-graph.md (label: estimate, composed from
measured kernel times). Everything this script prints is an estimate; see async-vlm-expert-overlap.md
for the assumptions behind scenarios A, B, C and C+B.

Run: python3 docs/analysis/async_overlap_estimate.py
"""

# Stage-1 kernel sums for W1 on H1, microseconds (smolvla-op-graph.md, "stage | GEMM | attention | row").
VISION_US = 14_688.6
CONNECTOR_US = 165.8
PREFIX_US = 3_010.4
EXPERT_US = 17_374.5
EXPERT_GEMM_US = 13_643.6
EXPERT_ATTN_US = 2_003.0
EXPERT_ROW_US = 1_727.8
# cross_k_proj + cross_v_proj: 8 layers x 2 x 5.67 us, once per call, needs the fresh prefix.
CROSS_KV_US = 2 * 8 * 5.67

FLOW_STEPS = 10
CLUSTERS = 64
# gemm_smallm_sweep.py first result (50 x 4096 x 720): 17.7 us on all 64 clusters vs 32.4 us picked.
SMALLM_GEMM_RATIO = 17.7 / 32.4

CHARS_PER_MS = 2


def ms(us: float) -> float:
    return us / 1000.0


VLM = ms(VISION_US + CONNECTOR_US + PREFIX_US)
KV = ms(CROSS_KV_US)
E_STEP = ms(EXPERT_US - CROSS_KV_US) / FLOW_STEPS
GEMM_STEP = ms(EXPERT_GEMM_US - CROSS_KV_US) / FLOW_STEPS
ATTN_ROW_STEP = ms(EXPERT_ATTN_US + EXPERT_ROW_US) / FLOW_STEPS
E_STEP_TUNED = E_STEP - GEMM_STEP * (1 - SMALLM_GEMM_RATIO)
BASELINE = ms(VISION_US + CONNECTOR_US + PREFIX_US + EXPERT_US)
BASELINE_TUNED = VLM + KV + FLOW_STEPS * E_STEP_TUNED


def overlap_time(t_vlm: float, e_stale: float, e_fresh: float, k: int) -> float:
    """k stale steps run beside the VLM; cross K/V and the remaining steps follow it."""
    return max(t_vlm, k * e_stale) + KV + (FLOW_STEPS - k) * e_fresh


def vlm_on(clusters: int) -> float:
    """Pessimistic: VLM time scales linearly with the clusters it keeps."""
    return VLM * CLUSTERS / clusters


def stale_step_on(clusters: int) -> float:
    """Expert step confined to `clusters`: GEMMs as picked (8-16 clusters), attention and row ops scaled."""
    return GEMM_STEP * max(1.0, 16 / clusters) + ATTN_ROW_STEP * CLUSTERS / clusters


def lane(segments: list[tuple[float, str]]) -> str:
    out: list[str] = []
    t = 0.0
    for dur, ch in segments:
        a, b = round(t * CHARS_PER_MS), round((t + dur) * CHARS_PER_MS)
        out.extend(ch * (b - len(out)) if b > len(out) else "")
        t += dur
    return "".join(out)


def steps(n: int, dur: float, first: int = 1) -> list[tuple[float, str]]:
    return [(dur, str((first + i) % 10)) for i in range(n)]


def timelines(k: int = 5, c: int = 16) -> str:
    vlm_b = vlm_on(CLUSTERS - c)
    stale_b = stale_step_on(c)
    rows = [
        ("Baseline", [(ms(VISION_US), "V"), (ms(CONNECTOR_US + PREFIX_US), "P"), (KV, "k")]
         + steps(FLOW_STEPS, E_STEP), BASELINE),
        None,
        ("A  VLM", [(ms(VISION_US), "V"), (ms(CONNECTOR_US + PREFIX_US), "P")], None),
        ("   expert", steps(k, E_STEP) + [(VLM - k * E_STEP, "."), (KV, "k")]
         + steps(FLOW_STEPS - k, E_STEP, k + 1), overlap_time(VLM, E_STEP, E_STEP, k)),
        None,
        (f"B  VLM({CLUSTERS - c})", [(vlm_b, "V")], None),
        (f"   expert({c})", steps(k, stale_b) + [(vlm_b - k * stale_b, "."), (KV, "k")]
         + steps(FLOW_STEPS - k, E_STEP, k + 1), overlap_time(vlm_b, stale_b, E_STEP, k)),
        None,
        ("C  VLM", [(ms(VISION_US), "V"), (ms(CONNECTOR_US + PREFIX_US), "P")], None),
        ("   expert", steps(k, E_STEP_TUNED) + [(VLM - k * E_STEP_TUNED, "."), (KV, "k")]
         + steps(FLOW_STEPS - k, E_STEP_TUNED, k + 1), overlap_time(VLM, E_STEP_TUNED, E_STEP_TUNED, k)),
        None,
        (f"C+B VLM({CLUSTERS - c})", [(vlm_b, "V")], None),
        (f"   expert({c})", steps(k, stale_b) + [(vlm_b - k * stale_b, "."), (KV, "k")]
         + steps(FLOW_STEPS - k, E_STEP_TUNED, k + 1), overlap_time(vlm_b, stale_b, E_STEP_TUNED, k)),
    ]
    width = 13
    header = " " * width + "".join(f"{t:<10}" for t in range(0, 40, 5)) + "ms"
    rule = " " * width + "|----+----" * 8
    lines = [header, rule]
    for r in rows:
        if r is None:
            lines.append("")
            continue
        name, segs, end = r
        text = f"{name:<{width}}{lane(segs)}"
        if end is not None:
            text = f"{text:<{width + 76}}-> {end:5.2f} ms"
        lines.append(text.rstrip())
    return "\n".join(lines)


def table() -> str:
    lines = ["| scenario | k | c | per call, ms | vs stage-1 baseline | vs its own no-overlap baseline |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k in (3, 5):
        t = overlap_time(VLM, E_STEP, E_STEP, k)
        lines.append(f"| A ideal | {k} | - | {t:.2f} | {BASELINE / t:.2f}x | {BASELINE / t:.2f}x |")
    for c in (8, 16):
        for k in (3, 5):
            t = overlap_time(vlm_on(CLUSTERS - c), stale_step_on(c), E_STEP, k)
            lines.append(f"| B split | {k} | {c} | {t:.2f} | {BASELINE / t:.2f}x | {BASELINE / t:.2f}x |")
    for k in (3, 5):
        t = overlap_time(VLM, E_STEP_TUNED, E_STEP_TUNED, k)
        lines.append(f"| C ideal | {k} | - | {t:.2f} | {BASELINE / t:.2f}x | {BASELINE_TUNED / t:.2f}x |")
        t = overlap_time(vlm_on(CLUSTERS - 16), stale_step_on(16), E_STEP_TUNED, k)
        lines.append(f"| C+B split | {k} | 16 | {t:.2f} | {BASELINE / t:.2f}x | {BASELINE_TUNED / t:.2f}x |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"VLM part {VLM:.2f} ms, expert step {E_STEP:.3f} ms (tuned {E_STEP_TUNED:.3f}), "
          f"cross K/V {KV:.3f} ms")
    print(f"baseline {BASELINE:.2f} ms, tuned-GEMM baseline {BASELINE_TUNED:.2f} ms")
    print(f"B: VLM on 48 clusters {vlm_on(48):.2f} ms, stale step on 16 clusters {stale_step_on(16):.3f} ms\n")
    print(timelines())
    print()
    print(table())
