"""V-AEFusion and DP-Cache on SmolVLA W1 / H1. Label: estimate.

Inputs
  * Stage-1 kernel sums per stage: docs/analysis/smolvla-op-graph.md (after dcd221d: expert GEMMs
    on the full mesh, W1 28.99 ms per call).
  * Sub-mesh expert kernel times: the fastest checked setting per shape that uses at most c clusters,
    from infra/kernel-runs/results/{gemm_sweep_h1,gemm_smallm_h1,attn_sweep_h1}.csv (measured).
Methods: Zhou et al. (SJTU), arXiv 2604.24447, sec. 5.2.2 (DP-Cache) and 5.3 (V-AEFusion);
wiki://resources/serving/vla-xpu-characterization-sjtu.md.

Assumptions (each one is stated on the page that uses these numbers)
  * VLM on 64 - c clusters slows down by 64 / (64 - c) (pessimistic; not measured).
  * Expert row operators (50 x 720 Snitch kernels) on c clusters: between unchanged and 64 / c slower.
  * DP-Cache reuses 4 of 10 flow steps (4, 5 reuse 3; 7, 8 reuse 6). The paper defines the stable
    segment only for a 100-step diffusion; 4 of 10 matches its pi0 saving (102 -> 79 ms, about 38 %
    of expert time).
  * Stale steps k <= 5 (SJTU's accuracy limit on pi0, Table 7).

Run: python3 docs/analysis/async_overlap_estimate.py [--json]
"""

import collections
import csv
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "infra/kernel-runs/results"

# Stage-1 kernel sums, us (smolvla-op-graph.md, W1, H1, after dcd221d)
VISION, CONNECTOR, PREFIX = 14688.6, 165.8, 3010.4
EXPERT = 11129.1
EXP_GEMM, EXP_ATTN, EXP_ROW = 7398.2, 2003.0, 1727.8
CROSS_KV = 2 * 8 * 5.671  # cross K/V projection: once per call, needs the fresh prefix
STEPS, MESH = 10, 64

VLM = VISION + CONNECTOR + PREFIX
BASE = VLM + EXPERT
E = (EXPERT - CROSS_KV) / STEPS
E_ROW = EXP_ROW / STEPS

# Per-flow-step expert matrix and attention kernels: shape -> calls per step
STEP_KERNELS = {
    "50x720x32": 1, "50x720x1440": 1, "50x720x720": 1, "50x32x720": 1,  # embed, time MLP, out
    "50x1600x720": 8, "50x960x720": 8, "50x720x960": 16,                # qkv, q, o
    "50x4096x720": 16, "50x720x2048": 16,                               # MLP
    "15x50x291x64": 8, "15x50x241x64": 8,                               # self, cross attention
}
COMPUTED = {1, 2, 3, 6, 9, 10}  # DP-Cache: steps that run; the others reuse


def submesh_best():
    best = collections.defaultdict(dict)

    def add(shape, clusters, t):
        for c in (8, 16, 32, 64):
            if clusters <= c:
                best[shape][c] = min(best[shape].get(c, math.inf), t)

    ok = lambda r: r["status"] == "ok" and r["bad"] == "0"
    for r in csv.DictReader(open(RES / "gemm_sweep_h1.csv")):
        if ok(r):
            add(r["shape"], int(r["scale"]) ** 2 * int(r["groups"]), int(r["time_ns"]) / 1e3)
    for r in csv.DictReader(open(RES / "gemm_smallm_h1.csv")):
        if ok(r):
            add(r["shape"], int(r["active"]), int(r["time_ns"]) / 1e3)
    for r in csv.DictReader(open(RES / "attn_sweep_h1.csv")):
        if ok(r):
            add(r["shape"], int(r["sx"]) * int(r["sy"]), int(r["time_ns"]) / 1e3)
    return best


BEST = submesh_best()


def stale_step(c, row):
    """One expert flow step confined to c clusters (us); row in {'same', 'lin'}."""
    k = sum(n * BEST[s][c] for s, n in STEP_KERNELS.items())
    return k + E_ROW * (1 if row == "same" else MESH / c)


def run(name, k, c=0, row="same", cache=False, vlm_scale=None):
    vlm = VLM * (vlm_scale if vlm_scale else (MESH / (MESH - c) if c else 1))
    es = stale_step(c, row) if c else E
    stale = [s for s in range(1, k + 1) if not cache or s in COMPUTED]
    fresh = [s for s in range(k + 1, STEPS + 1) if not cache or s in COMPUTED]
    reused = [s for s in range(1, STEPS + 1) if cache and s not in COMPUTED]
    overlap = max(vlm, len(stale) * es)
    total = overlap + CROSS_KV + len(fresh) * E
    return dict(name=name, k=k, c=c, row=row, cache=cache, vlm_ms=vlm / 1e3,
                stale_step_ms=es / 1e3, fresh_step_ms=E / 1e3, stale=stale, fresh=fresh,
                reused=reused, total_ms=total / 1e3, speedup=BASE / total,
                one_tick_hz=1e6 / total)


def best_k(name, **kw):
    return min((run(name, k, **kw) for k in range(0, 6)), key=lambda r: r["total_ms"])


ROWS = [
    run("Baseline", 0),
    run("DP-Cache", 0, cache=True),
    run("Fusion A: no contention", 5),
    best_k("Fusion B: split c=8", c=8, row="lin"),
    best_k("Fusion B: split c=16", c=16, row="lin"),
    run("Fusion A + DP-Cache", 5, cache=True),
    best_k("Fusion B c=8 + DP-Cache", c=8, row="lin", cache=True),
]

OUT = dict(
    inputs=dict(vision_ms=VISION / 1e3, connector_prefix_ms=(CONNECTOR + PREFIX) / 1e3,
                vlm_ms=VLM / 1e3, cross_kv_ms=CROSS_KV / 1e3, step_ms=E / 1e3,
                baseline_ms=BASE / 1e3),
    stale_step_ms={c: dict(same=stale_step(c, "same") / 1e3, lin=stale_step(c, "lin") / 1e3)
                   for c in (8, 16)},
    breakeven=dict(hidden_ms_k5=5 * E / 1e3, max_vlm_slowdown_k5=(VLM + 5 * E) / VLM,
                   max_stale_step_c8_ms=VLM * MESH / (MESH - 8) / 5 / 1e3),
    rows=ROWS,
)

if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps(OUT, indent=1))
    else:
        i = OUT["inputs"]
        print(f"VLM {i['vlm_ms']:.2f} ms, step {i['step_ms']:.3f} ms, baseline {i['baseline_ms']:.2f} ms")
        print("stale step on c clusters (row ops same / 64/c):",
              {c: f"{v['same']:.2f}/{v['lin']:.2f}" for c, v in OUT["stale_step_ms"].items()})
        print("break-even:", {k: round(v, 3) for k, v in OUT["breakeven"].items()})
        for r in ROWS:
            print(f"{r['name']:28s} k={r['k']} c={r['c']:2d} stale={r['stale']} fresh={r['fresh']}"
                  f" -> {r['total_ms']:.2f} ms  {r['speedup']:.3f}x  one-tick {r['one_tick_hz']:.1f} Hz")
