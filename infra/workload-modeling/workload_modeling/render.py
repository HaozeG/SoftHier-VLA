"""Render the SmolVLA operator graph (Markdown + CSV) from the standalone model."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import smolvla_ops as m

TEMPLATES = Path(__file__).resolve().parent / "templates"
CLASSES = ("gemm", "attn", "softmax", "norm", "acti", "elementwise", "data")

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


def context(c, workload_id: str) -> dict:
    wl = c.workloads[workload_id]
    d = m.load_dims(wl)
    rows = m.annotate(m.build(c, workload_id))
    tot = m.stage_totals(rows)
    all_flop = sum(t.flop for t in tot.values())
    stages = {
        k: {"gflop": _g(t.flop), "mm_gflop": _g(t.matmul_flop), "share": f"{100 * t.flop / all_flop:.1f}",
            "umb": _mb(t.bytes_unique), "wmb": _mb(t.bytes_weight), "ai": _ai(t.flop_per_wbyte)}
        for k, t in tot.items()
    }
    tw = sum(t.bytes_weight for t in tot.values())
    total = {"gflop": _g(all_flop), "mm_gflop": _g(sum(t.matmul_flop for t in tot.values())),
             "umb": _mb(sum(t.bytes_unique for t in tot.values())), "wmb": _mb(tw), "ai": _ai(all_flop / tw)}
    classes = []
    for cl in CLASSES:
        f = sum(r.flop for r in rows if r.cls == cl)
        classes.append({"name": cl, "gflop": _g(f), "share": f"{100 * f / all_flop:.2f}"})
    view = [{
        "stage": r.stage, "op": r.op, "layers": r.layers, "cls": r.cls, "shape": _shape(r), "count": r.count,
        "gflop": _g(r.flop), "umb": _mb(r.bytes_unique), "wmb": _mb(r.bytes_weight), "ai": _ai(r.flop_per_wbyte),
        "unit_today": r.unit_today, "hand_kernel": r.hand_kernel,
    } for r in rows]
    n_vis = m.vision_tokens(wl, d)
    tok = {"vision_out": n_vis, "image": wl.cameras * n_vis // d.shuffle ** 2, "prefix": m.prefix_tokens(wl, d)}
    return {"wl": wl, "d": d, "s": stages, "stages": list(m.STAGES), "total": total, "classes": classes,
            "rows": view, "notes": STAGE_NOTES, "tok": tok, "_rows": rows}


def render_md(c, workload_id: str) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, keep_trailing_newline=True,
                      autoescape=False)
    return env.get_template("op_graph.md.j2").render(**context(c, workload_id))


def render_csv(c, workload_id: str) -> str:
    rows = m.annotate(m.build(c, workload_id))
    f = io.StringIO()
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["stage", "layers", "op", "class", "shape", "count", "flop", "bytes_unique", "bytes_weight",
                "flop_per_wbyte", "unit_today", "hand_kernel", "note"])
    for r in rows:
        ai = r.flop_per_wbyte
        w.writerow([r.stage, r.layers, r.op, r.cls, "x".join(map(str, r.shape)), r.count, f"{r.flop:.0f}",
                    f"{r.bytes_unique:.0f}", f"{r.bytes_weight:.0f}", "" if ai is None else f"{ai:.2f}",
                    r.unit_today, r.hand_kernel, r.note])
    return f.getvalue()
