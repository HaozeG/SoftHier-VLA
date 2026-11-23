"""Render the SmolVLA operator graph (Markdown + CSV) from the standalone model."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import smolvla_ops as m

TEMPLATES = Path(__file__).resolve().parent / "templates"
CLASSES = ("gemm", "attn", "softmax", "norm", "acti", "elementwise", "data")
ROW_CLASSES = ("norm", "acti", "elementwise", "data")   # row operators outside the GEMM and attention kernels

STAGE_NOTES = {
    "vision": "SigLIP encoder, run once per camera (reference code); attention per camera and head.",
    "connector": "Pixel shuffle (scale factor 4: 16 patches into one token) and one linear projection to the VLM width.",
    "prefix": "Image, language and state tokens through the kept VLM layers, once per call; the per-layer keys "
              "and values are kept as the KV cache the expert reads.",
    "expert": "Flow-matching transformer run once per flow step. Even layers: self-attention over the prefix cache "
              "and the action tokens; odd layers: cross-attention to the projected prefix cache. The reference code "
              "repeats the cross-attention K/V projection in every step.",
}


def _g(x: float) -> str:
    return f"{x / 1e9:,.3f}"


def _mb(x: float) -> str:
    return f"{x / 1e6:,.2f}"


def _ai(x: float | None) -> str:
    return "" if x is None else f"{x:,.0f}"


def _shape(r: m.OpRow) -> str:
    if r.cls == "gemm":
        M, N, K = r.shape
        return f"M {M}, N {N}, K {K}"
    if r.cls == "attn":
        h, sq, skv, d = r.shape
        return f"h {h}, Sq {sq}, Skv {skv}, d {d}"
    return " x ".join(str(v) for v in r.shape)


def _us(ns: float | None) -> str:
    return "" if ns is None else f"{ns / 1e3:,.2f}"


GROUPS = (("GEMM", ("gemm",)), ("attention", ("attn", "softmax")), ("row operators", ROW_CLASSES))


def _grouped_included(included: list[m.OpRow]) -> str:
    """'`softmax`, `pv` (timed in `qk`)': rows whose work runs inside another row's kernel call, by host row."""
    by: dict[str, list[str]] = {}
    for r in included:
        by.setdefault(m.INCLUDED_IN[(r.stage, r.op)], []).append(f"`{r.op}`")
    return "; ".join(f"{', '.join(ops)} (timed in `{host}`)" for host, ops in by.items())


def kernel_sums(c, workload_id: str, rows: list[m.OpRow]) -> dict:
    """Composed per-stage time: sum over rows of count x measured ns per call of the row's kernel.

    Rows whose work runs inside another row's kernel call (m.INCLUDED_IN) add nothing; host rows and rows
    without a simulated kernel are listed as not covered.

    MFU (estimate): nominal FLOP of the covered rows (included rows count, their work runs in the call that
    contains them; host rows do not) over composed time x the arch's matrix-unit peak (c.peak_total)."""
    peak = c.peak_total(m.kernel_set(workload_id).arch)
    stages, tot = [], {g: 0.0 for g, _ in GROUPS}
    flop_tot, gflop = 0.0, {g: 0.0 for g, _ in GROUPS}
    mfu = lambda f, us: f"{100 * f / (us * 1e-6 * peak):.1f}" if us else "-"
    for st in m.STAGES:
        rs = [r for r in rows if r.stage == st]
        ns = {r.op: m.hand_kernel_ns(c, workload_id, r) for r in rs}
        covered = [r for r in rs if ns[r.op] is not None or (r.stage, r.op) in m.INCLUDED_IN]
        grp = {}
        for g, classes in GROUPS:
            grp[g] = sum(r.count * ns[r.op] for r in rs if r.cls in classes and ns[r.op] is not None) / 1e3
            tot[g] += grp[g]
            gflop[g] += sum(r.flop for r in covered if r.cls in classes)
        flop = sum(r.flop for r in covered)
        flop_tot += flop
        included = [r for r in rs if (r.stage, r.op) in m.INCLUDED_IN]
        uncovered = [r for r in rs if ns[r.op] is None and (r.stage, r.op) not in m.INCLUDED_IN]
        stages.append({
            "stage": st, **{g: f"{v:,.1f}" for g, v in grp.items()}, "total": f"{sum(grp.values()):,.1f}",
            "total_us": sum(grp.values()), "gflop": f"{flop / 1e9:,.1f}", "mfu": mfu(flop, sum(grp.values())),
            "included": _grouped_included(included) or "none",
            "uncovered": ", ".join(f"`{r.op}`" + (" (host)" if r.op in m.HOST_OPS else "") for r in uncovered) or "none",
            "ms": f"{sum(grp.values()) / 1e3:,.2f}",
        })
    grand = sum(tot.values())
    all_uncovered = [r for r in rows if m.hand_kernel_ns(c, workload_id, r) is None
                     and (r.stage, r.op) not in m.INCLUDED_IN]
    total_uncovered = ", ".join(f"`{r.op}` ({r.stage})" for r in all_uncovered)
    for k in stages:
        k["share"] = f"{100 * k['total_us'] / grand:.0f}" if grand else "0"
    return {"stages": stages, "total": {**{g: f"{v:,.1f}" for g, v in tot.items()}, "total": f"{grand:,.1f}",
                                        "gflop": f"{flop_tot / 1e9:,.1f}", "mfu": mfu(flop_tot, grand)},
            "total_ms": f"{grand / 1e3:,.2f}", "total_uncovered": total_uncovered,
            "share": {g: f"{100 * v / grand:.0f}" if grand else "0" for g, v in tot.items()},
            "mfu_group": {g: mfu(gflop[g], tot[g]) for g, _ in GROUPS},
            "peak_tflops": f"{peak / 1e12:.1f}"}


def hardware(c, arch: str) -> dict:
    """Arch parameters the hand-kernel times were measured on, plus totals computed from them."""
    raw = lambda k: c.get(f"arch.{arch}.{k}").value
    v = lambda k: (lambda x: int(x) if x == int(x) else x)(c.value(f"arch.{arch}.{k}"))
    peak = c.peak_total(arch)
    bw = v("hbm_channels") * v("hbm_bw_per_channel") * 1e9
    return {
        "mesh": f"{v('clusters_x')} x {v('clusters_y')}", "clusters": c.clusters(arch),
        "clock_ghz": f"{v('clock_hz') / 1e9:g}", "cores": v("cores_per_cluster"), "spatz_cores": v("spatz_cores"),
        "spatz_fu": v("spatz_function_units"), "vlen": v("spatz_vlen_bits"),
        "redmule": f"{v('redmule_h')} x {v('redmule_w')}", "tcdm_kib": v("tcdm_bytes") // 1024,
        "tcdm_total_mib": f"{c.clusters(arch) * v('tcdm_bytes') / 2**20:g}", "noc_bits": v("noc_link_bits"),
        "hbm_channels": v("hbm_channels"), "hbm_device": raw("hbm_device"), "hbm_bw_ch": v("hbm_bw_per_channel"),
        "hbm_bw_total": f"{bw / 1e9:g}", "peak_cluster": f"{c.peak_per_cluster(arch) / 1e12:.3f}",
        "peak_total": f"{peak / 1e12:.1f}", "balance": f"{peak / bw:.0f}",
    }


# Text of the composed-estimate section per kernel group (what was run, on which arch, how it was checked).
KERNEL_PAGES = {
    "h1_kernels": {
        "intro": "kernel repo `HaozeG/gvsoc@aa57690` (branch gelu-layernorm-kernels, the frozen stage-1 build), "
                 "H1 (64 clusters, RedMulE 32 x 32, Spatz on 4 cores per cluster), GVSoC with the SoftHier-VLA patch "
                 "series 0001-0012, DRAMSys HBM; protocol: docs/designs/stage1-spec.md §4.3",
        "arch_text": "H1, `infra/kernel-runs/arch/arch_h1.py` (`constants.yaml` section `arch.h1`)",
        "kernels": [
            ("SummaGEMM", "every matrix product", "yes, bit-exact against a replay golden",
             "M, N, K padded to tile multiples (stage-1 store fix); tile, scale and Nosplit or split-N: fastest checked "
             "setting of a sweep per shape, re-run on the frozen build"),
            ("FlatAttention", "QK^T, softmax, P.V", "yes, within 0.02 of a float64 golden",
             "no masking (prefix and expert need one: approximation); GQA folds query heads; group shape, blocks and "
             "collective or async mode: fastest checked setting of a sweep; sequence lengths padded to a block multiple"),
            ("LayerNorm, Activation, Eltwise", "norms, activations, elementwise, pixel shuffle",
             "yes, bit-exact against a replay golden", "one run per shape, exact shape"),
            ("RMSNorm", "RMSNorm", "yes, 32 ULP against float64 x / rms(x)",
             "one run per shape; the kernel has no weight and no eps (one multiply per element short)"),
        ],
        "checks": "Every time comes from a run whose output check passed on all 64 clusters, under "
                  "`MALLOC_PERTURB_` (uninitialised simulator memory would show). Simulator fixes found on the way "
                  "(RedMulE buffers, FlooNoC reductions) are patches 0011 and 0012.",
        "platform": "It is on H1 with DRAMSys HBM, not PR1's arch with ideal HBM.",
    },
}


def context(c, workload_id: str, lowering: m.Lowering | None = None, variant: str = "") -> dict:
    wl = c.workloads[workload_id]
    d = m.load_dims(wl)
    ks = m.kernel_set(workload_id)
    lowering = lowering or ks.lowering
    rows = m.annotate(m.build(c, workload_id, lowering))
    tot = m.stage_totals(rows)
    all_flop = sum(t.flop for t in tot.values())
    stages = {
        k: {"gflop": _g(t.flop), "mm_gflop": _g(t.matmul_flop), "share": f"{100 * t.flop / all_flop:.1f}",
            "traffic_mb": _mb(t.memory_traffic_bytes), "weights_mb": _mb(t.weights_read_bytes), "ai": _ai(t.flop_per_weight_byte)}
        for k, t in tot.items()
    }
    tw = sum(t.weights_read_bytes for t in tot.values())
    total = {"gflop": _g(all_flop), "mm_gflop": _g(sum(t.matmul_flop for t in tot.values())),
             "traffic_mb": _mb(sum(t.memory_traffic_bytes for t in tot.values())), "weights_mb": _mb(tw), "ai": _ai(all_flop / tw)}
    classes = []
    for cl in CLASSES:
        f = sum(r.flop for r in rows if r.cls == cl)
        classes.append({"name": cl, "gflop": _g(f), "share": f"{100 * f / all_flop:.2f}"})
    view = [{
        "stage": r.stage, "op": r.op, "layers": r.layers, "cls": r.cls, "shape": _shape(r), "count": r.count,
        "gflop": _g(r.flop), "traffic_mb": _mb(r.memory_traffic_bytes), "weights_mb": _mb(r.weights_read_bytes), "ai": _ai(r.flop_per_weight_byte),
        "pr1_unit": r.pr1_unit, "hand_kernel": r.hand_kernel, "kunits": r.hand_kernel_units, "kus": (f"timed in `{m.INCLUDED_IN[(r.stage, r.op)]}`" if (r.stage, r.op) in m.INCLUDED_IN
                else _us(m.hand_kernel_ns(c, workload_id, r))),
    } for r in rows]
    n_vis = m.vision_tokens(wl, d)
    tok = {"vision_out": n_vis, "image": wl.cameras * n_vis // d.shuffle ** 2, "prefix": m.prefix_tokens(wl, d)}
    return {"wl": wl, "d": d, "s": stages, "stages": list(m.STAGES), "total": total, "classes": classes,
            "rows": view, "notes": STAGE_NOTES, "tok": tok, "_rows": rows, "ksum": kernel_sums(c, workload_id, rows),
            "hw": hardware(c, ks.arch), "kp": KERNEL_PAGES[ks.group], "group": ks.group, "variant": variant,
            "lowering": lowering, "lowering_text": lowering_text(lowering)}


def lowering_text(low: m.Lowering) -> str:
    """The lowering's departures from the reference computation, in words (empty for the reference)."""
    parts = []
    if not low.kv_proj_per_step:
        parts.append("the expert's cross-attention K/V projection runs once per call, not in every flow step")
    if low.batch_cameras:
        parts.append("the cameras are batched into one vision-encoder call (M = cameras x 1024 for every per-token "
                     "product and row operator; attention stays per camera)")
    if low.prefix_rows or low.kv_align > 1 or low.time_in_split:
        parts.append("PR1's padding and time-MLP split (see crosscheck.md)")
    return "; ".join(parts)


def render_md(c, workload_id: str, lowering: m.Lowering | None = None, variant: str = "") -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, keep_trailing_newline=True,
                      autoescape=False)
    return env.get_template("op_graph.md.j2").render(**context(c, workload_id, lowering, variant))


def render_csv(c, workload_id: str, lowering: m.Lowering | None = None) -> str:
    rows = m.annotate(m.build(c, workload_id, lowering or m.kernel_set(workload_id).lowering))
    f = io.StringIO()
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["stage", "layers", "op", "class", "shape", "count", "flop", "memory_traffic_bytes", "weights_read_bytes",
                "flop_per_weight_byte", "pr1_unit", "hand_kernel", "hand_kernel_units", "kernel_ns_per_call", "note"])
    for r in rows:
        ai = r.flop_per_weight_byte
        w.writerow([r.stage, r.layers, r.op, r.cls, "x".join(map(str, r.shape)), r.count, f"{r.flop:.0f}",
                    f"{r.memory_traffic_bytes:.0f}", f"{r.weights_read_bytes:.0f}", "" if ai is None else f"{ai:.2f}",
                    r.pr1_unit, r.hand_kernel, r.hand_kernel_units, "" if (ns := m.hand_kernel_ns(c, workload_id, r)) is None else f"{ns:.0f}",
                    r.note])
    return f.getvalue()
