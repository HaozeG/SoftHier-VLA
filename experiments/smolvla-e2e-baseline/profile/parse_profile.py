#!/usr/bin/env python3
"""Per-op profile of a SoftHier run, from a gvsoc log (no re-simulation).  Standard library only.

A run of softhier-mlir's smolvla-e2e (`--e2e-trace`) writes, per phase, <phase>.log (gvsoc trace lines for RedMulE,
iDMA and the cluster barrier unit, plus the program's own `[mark] <tag> <mcycle>` lines) and <phase>.mlir (the program).
Every softhier op that is a library call (gemm, add, layernorm, ...) ends with one global barrier (`sh_barrier_global`),
and gvsoc prints one `Cluster Sync: <arrive> ns -> <release> ns` line per cluster and barrier. So the release times on
cluster 0 delimit the ops: op k = (release k-1, release k]. The expanded op list of the .mlir is aligned to the barriers
segment by segment (between two marks), and the count of ops must equal the count of barriers or the segment is
reported as one row `segment:<tag>` (source=mark) instead.

Subcommands
  ops      <run-dir> [--phases a,b,..] [--out ops.csv]   one CSV row per op instance (see COLUMNS)
  summary  <ops.csv> [--top N]                           markdown tables: phases, ops, classes, hardware activity
  insn     <ops.csv> <insn_bins.csv.gz> --out f.csv      per-op instruction mix of the traced cores (prof_wrap.py output)
  phases   <ops.csv> [--top N]                           every op of every phase: reps, mean, total, shares
  layer    <ops.csv> --phase P --layer L [--step S]      one layer's ops in order
  libero   --vision v.csv --prefix p.csv --expert e.csv --out o.csv   G4a partial runs scaled to the LIBERO composition (ESTIMATE)
  compact  <log> <out.zst>                               strip the simulator start-up lines and compress (zstd)
Logs may be plain or *.zst (decompressed through the `zstd` binary).

Time is simulated ns at 1 GHz. Peak RedMulE: macs_per_cycle = 128*32 = 4096 MAC/cycle/cluster = 8192 FLOP/cycle,
16 clusters -> 131.07 TFLOP/s (softhier_mlir.sim.gvsoc.Arch.macs_per_cycle; the RedMulE `uti` field is MAC/(cycle*4096)).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NCL = 16
PEAK_FLOP_PER_NS_CLUSTER = 128 * 32 * 2     # FLOP per ns per cluster (1 GHz)
COLUMNS = ["phase", "program", "seg", "layer", "step", "op", "class", "level", "start_ns", "end_ns", "dur_ns", "cluster", "count", "source",
           "redmule_busy_ns", "idma_busy_ns", "sync_wait_ns", "sync_wait_max_ns", "redmule_flops", "hbm_rd", "hbm_wr", "c2c", "local",
           "active_clusters", "crit_redmule_ns", "crit_idma_ns", "redmule_uti", "bound"]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RED = re.compile(r"/chip/cluster_(\d+)/redmule/trace\s*\] \[LightRedmule\] Finished : (\d+) ns ---> (\d+) ns .*?uti = ([0-9.]+).*?"
                  r"M-N-K = (\d+)-(\d+)-(\d+)")
_DMA = re.compile(r"/chip/cluster_(\d+)/idma/fe/trace\s*\] \[iDMA\] Finished : (\d+) ns ---> (\d+) ns")
_SYNC = re.compile(r"/chip/cluster_(\d+)/cluster_registers/trace\s*\] Cluster Sync: (\d+) ns -> (\d+) ns")
_TXN = re.compile(r"Txn \d+ = \{type: (1D|2D), src: 0x([0-9a-f]+)(?:, src_stride: 0x[0-9a-f]+)?, dst: 0x([0-9a-f]+)"
                  r"(?:, dst_stride: 0x[0-9a-f]+, repeats: 0x([0-9a-f]+))?, size: 0x([0-9a-f]+)")
_MARK = re.compile(r"\[mark\] (\w+) (\d+)")
_TRACE_HEAD = re.compile(r": (-?\d+): \[")
HBM, ZOMEM, REMOTE, REMOTE_END = 0xC0000000, 0x18000000, 0x30000000, 0x30000000 + 64 * 0x100000


# ---------------------------------------------------------------------------------------------- log reading
def open_log(path):
    path = str(path)
    if path.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True, errors="replace")
        return p.stdout
    return open(path, errors="replace")


def _trace_start(ln: str) -> int:
    """Index where a gvsoc trace line starts inside a physical line (the program prints char by char, so a trace line can
    land in the middle of a program line, `[mark] start 30` + `86454000: 86454: [...`); -1 if none."""
    for m in _TRACE_HEAD.finditer(ln):
        ns = int(m[1])
        ps = "0" if ns < 0 else str(ns * 1000)
        if ln[m.start() - len(ps):m.start()] == ps:
            return m.start() - len(ps)
    return -1


def kind(src: int, dst: int) -> str:
    if src >= HBM:
        return "hbm_rd"
    if dst >= HBM:
        return "hbm_wr"
    if REMOTE <= src < REMOTE_END or REMOTE <= dst < REMOTE_END:
        return "c2c"
    if ZOMEM <= src < ZOMEM + 0x1000000:
        return "zero"
    return "local"


def txn_bytes(info: str) -> dict:
    out = defaultdict(int)
    for m in _TXN.finditer(info):
        n = int(m[5], 16) * (int(m[4], 16) if m[4] else 1)
        out[kind(int(m[2], 16), int(m[3], 16))] += n
    return out


def read_log(path):
    """-> marks [(tag, ns)], red [(c, t0, t1, M, N, K)], dma [(c, t0, t1, bytes-by-kind)], sync [(c, t0, t1)]."""
    marks, red, dma, sync = [], [], [], []
    buf, prev_c, acc = "", None, 0
    f = open_log(path)

    def program_line(s):
        nonlocal prev_c, acc
        m = _MARK.match(s)
        if m:
            c = int(m[2])
            acc = c if prev_c is None else acc + ((c - prev_c) % (1 << 32))
            prev_c = c
            marks.append((m[1], acc))
    for ln in f:
        if "/chip/" in ln or "/sys/" in ln:
            k = _trace_start(ln)
            if k >= 0:
                buf += ln[:k]
                tl = _ANSI.sub("", ln[k:])
                if "Finished" in tl or "Cluster Sync" in tl:
                    m = _RED.search(tl)
                    if m:
                        red.append((int(m[1]), int(m[2]), int(m[3]), int(m[5]), int(m[6]), int(m[7]), float(m[4])))
                    else:
                        m = _DMA.search(tl)
                        if m:
                            dma.append((int(m[1]), int(m[2]), int(m[3]), txn_bytes(tl[m.end():])))
                        else:
                            m = _SYNC.search(tl)
                            if m:
                                sync.append((int(m[1]), int(m[2]), int(m[3])))
                continue
        buf += ln
        if buf.endswith("\n"):
            if buf.startswith("[mark] "):
                program_line(buf)
            buf = ""
    if buf.startswith("[mark] "):
        program_line(buf)
    return marks, red, dma, sync


# ---------------------------------------------------------------------------------------------- mlir expansion
_OPS_SKIP = {"hbm_buffer", "view", "preload_wait"}
_NAME = re.compile(r"%([A-Za-z_0-9]+)")


def _split_op(text: str):
    """softhier.<op> operands -> (op, srcs, dst, attrs-text)"""
    m = re.match(r"(?:%\w+ = )?softhier\.(\w+)\s*(.*)", text)
    op, rest = m[1], m[2]
    head = rest.split("{")[0].split(":")[0]
    if "->" in head:
        a, b = head.split("->", 1)
    elif " into " in head:
        a, b = head.split(" into ", 1)
    else:
        a, b = head, ""
    srcs = _NAME.findall(re.sub(r"\b(own|mask)\b", ",", a))
    dsts = _NAME.findall(b)
    return op, srcs, (dsts[0] if dsts else ""), text


def parse_mlir(path):
    """-> tree of ('op', op, srcs, dst, text) | ('mark', tag) | ('for', var, count, children); count from arith.constant."""
    consts, root, stack = {}, [], []
    stack.append(root)
    loops = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        m = re.match(r"%(\w+) = arith.constant (-?\d+) : index", s)
        if m:
            consts[m[1]] = int(m[2])
            continue
        m = re.match(r"scf.for %(\w+) = %(\w+) to %(\w+) step %(\w+) \{", s)
        if m:
            node = ["for", m[1], consts.get(m[3], 1) - consts.get(m[2], 0), []]
            stack[-1].append(node)
            stack.append(node[3])
            loops.append(node)
            continue
        if s == "}" and len(stack) > 1:
            stack.pop()
            continue
        m = re.match(r"softhier\.mark (?:%\w+ )?\{tag = \"(\w+)\"\}", s)
        if m:
            stack[-1].append(("mark", m[1]))
            continue
        if re.match(r"(?:%\w+ = )?softhier\.", s):
            op, srcs, dst, text = _split_op(s)
            if op not in _OPS_SKIP:
                stack[-1].append(("op", op, srcs, dst, text))
        m = re.match(r"%(\w+) = arith\.\w+ (.*)", s)
        if m:
            consts.setdefault("_const_" + m[1], 0)
    return root, consts


def flatten(tree, env=None, out=None):
    """Expand loops: list of ('op', op, srcs, dst, text, env) | ('mark', tag)."""
    env = env or {}
    out = [] if out is None else out
    for n in tree:
        if n[0] == "for":
            for i in range(n[2]):
                flatten(n[3], {**env, n[1]: i}, out)
        elif n[0] == "op":
            out.append(n + (dict(env),))
        else:
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------------- op labels / classes
def label(op, srcs, dst, text, phase, state):
    s1 = srcs[1] if len(srcs) > 1 else ""
    s0 = srcs[0] if srcs else ""
    base = re.sub(r"_[sc]$", "", s1)
    if op == "gemm":
        table = {"wq": "gemm_q", "wk": "gemm_k", "wv": "gemm_v", "wqkv": "gemm_qkv", "wo": "gemm_o", "w1": "mlp_up", "w2": "mlp_down",
                 "wg": "mlp_gate", "wu": "mlp_up", "wd": "mlp_down", "wgu": "mlp_gateup", "wkx_c": "kvproj_k", "wvx_c": "kvproj_v",
                 "wkx": "kvproj_k", "wvx": "kvproj_v", "wpe": "patch_embed", "wc": "connector_gemm", "wa": "action_in_proj",
                 "wti": "time_mlp_in", "wto": "time_mlp_out", "wout": "action_out_proj", "kTh": "attn_qk", "vh": "attn_pv"}
        return table.get(base, table.get(s1, f"gemm_{dst}"))
    if op == "add_bias":
        table = {"q": "bias_q", "k": "bias_k", "v": "bias_v", "ao": "bias_o", "f1": "bias_up", "f2": "bias_down", "x": "bias_patch",
                 "e": "bias_action_in", "e1": "bias_time", "h": "bias_time_out", "vt": "bias_action_out"}
        return table.get(dst, f"bias_{dst}")
    if op == "add":
        return {"ao": "residual_attn", "f2": "residual_mlp", "pos": "pos_embed_add", "zero": "init_x"}.get(s1, f"add_{dst}")
    if op == "layernorm":
        return {"g1": "ln1", "g2": "ln2", "gpost": "ln_post"}.get(base, f"ln_{base}")
    if op == "rmsnorm":
        return {"g1": "rmsnorm_attn", "g2": "rmsnorm_mlp", "gf": "rmsnorm_final", "gfin": "rmsnorm_final"}.get(base, f"rmsnorm_{base}")
    if op == "rope":
        return "rope_q" if s0 == "q" else "rope_k"
    if op == "attention":
        return "attention_fused"
    if op == "cross_attention":
        return "xattn_self_and_prefix" if " own " in text else "xattn_prefix"
    if op == "softmax":
        return "attn_softmax"
    if op == "scale":
        return "embed_scale" if state.get("pre_layers") else "pad_rows_zero"
    if op == "transpose":
        return "transpose_k"
    if op == "axpy":
        return "euler_update"
    if op == "silu_mul":
        return "time_silu" if s0 == "e1" else "silu_mul"
    return op


def op_class(lab: str, op: str) -> str:
    if lab in ("attn_qk", "attn_pv", "attn_softmax", "attention_fused", "xattn_self_and_prefix", "xattn_prefix"):
        return "attention"
    if op == "gemm":
        return "GEMM"
    if op in ("layernorm", "rmsnorm"):
        return "norm"
    if op in ("transpose", "pixel_shuffle"):
        return "data movement"
    if op in ("group_barrier", "dump_all", "dump_samples"):
        return "sync"
    return "elementwise"


# ---------------------------------------------------------------------------------------------- the profile
def window_stats(w0, w1, red_by_c, dma_by_c, sync_by_c, red_end, dma_end):
    """Hardware activity of the window (w0, w1] (events clipped to it; flops and bytes by event end time)."""
    dur = max(1, w1 - w0)
    rb, db, sb = [0] * NCL, [0] * NCL, [0] * NCL
    flops = 0
    for c in range(NCL):
        for arr, dst in ((red_by_c[c], rb), (dma_by_c[c], db), (sync_by_c[c], sb)):
            t0s, t1s = arr
            i = bisect.bisect_right(t1s, w0)
            while i < len(t1s) and t0s[i] < w1:
                dst[c] += max(0, min(t1s[i], w1) - max(t0s[i], w0))
                i += 1
    by = defaultdict(int)
    uw = ut = 0
    for (t1, c, M, N, K, t0, uti) in red_end.in_range(w0, w1):
        flops += 2 * M * N * K
        uw += uti * (t1 - t0)
        ut += t1 - t0
    uti_mean = uw / ut if ut else 0.0
    for (t1, c, kb) in dma_end.in_range(w0, w1):
        for k, v in kb.items():
            by[k] += v
    return dur, rb, db, sb, flops, by, uti_mean


class EndIndex:
    """events sorted by end time; in_range(w0, w1) -> those with w0 < end <= w1"""

    def __init__(self, items):
        self.items = sorted(items, key=lambda e: e[0])
        self.keys = [e[0] for e in self.items]

    def in_range(self, w0, w1):
        return self.items[bisect.bisect_right(self.keys, w0):bisect.bisect_right(self.keys, w1)]


def by_cluster(ev):
    d = [[] for _ in range(NCL)]
    for e in ev:
        d[e[0]].append((e[1], e[2]))
    out = []
    for c in range(NCL):
        d[c].sort(key=lambda x: x[1])
        out.append(([a for a, _ in d[c]], [b for _, b in d[c]]))
    return out


def bound_of(cls, dur, rb, db, sb, crit):
    """What limits the op, judged on the critical-path cluster (the one that waits least at the closing barrier):
    GEMM: RedMulE busy >= 50 % of the window -> compute-bound; else iDMA busy > RedMulE busy and >= 25 % ->
    data-movement-bound; else latency-bound (core issue, DMA set-up, barrier). Attention: RedMulE >= 50 % compute-bound,
    iDMA >= 50 % data-movement-bound, else core-bound (softmax). Elementwise / norm run on the cores:
    core-bound unless the critical iDMA is busy >= 50 %. `imbalance` is appended when the mean barrier wait of all clusters
    is > 50 % of the window (most clusters idle)."""
    r, d = rb[crit] / dur, db[crit] / dur
    if cls == "attention":      # RedMulE / iDMA both mostly idle -> the softmax and the head loop run on the cores
        b = "compute-bound" if r >= 0.5 else ("data-movement-bound" if d >= 0.5 else "core-bound")
    elif cls == "GEMM":
        b = "compute-bound" if r >= 0.5 else ("data-movement-bound" if d > r and d >= 0.25 else "latency-bound")
    elif cls == "data movement":
        b = "data-movement-bound"
    elif cls == "sync":
        b = "sync"
    else:
        b = "data-movement-bound" if d >= 0.5 else "core-bound"
    if sum(sb) / NCL / dur > 0.5:
        b += "+imbalance"
    return b


_GROUP = {
    "vision": {"vemb": "vision", "attn": "vision", "layer": "vision", "vis": "vision", "conn": "connector"},
    "prefix": {"emb": "connector", "attn": "prefix", "layer": "prefix", "end": "prefix_tail"},
    "expert": {"kvproj": "expert", "step": "expert"},
}


def group_of(program: str, seg_tag: str) -> str:
    """The rung-6 phase an op belongs to (run.py e2e_segments): vision = start -> vis (embed, layers, post-LN), connector =
    vis -> conn (pixel shuffle + GEMM) + prefix start -> emb (embedding scale), prefix = the decoder layers, expert = KV
    projection + the flow steps (the per-step X dump and the end are not timed). Everything else is `excluded`
    (`prefix_tail` = the final RMSNorm after the last layer, which the gate does not count)."""
    key = re.sub(r"\d+$", "", seg_tag)
    return _GROUP.get(re.sub(r"_[ab]$", "", program), {}).get(key, "excluded")


def phase_ops(run: Path, phase: str, log=None, tol=30):
    """Rows for one phase of a run dir. Returns (rows, notes)."""
    log = log or next(p for p in (run / f"{phase}.log", run / f"{phase}.log.zst") if p.exists())
    marks, red, dma, sync = read_log(log)
    prog = flatten(parse_mlir(run / f"{phase}.mlir")[0])
    notes = []
    consts = parse_mlir(run / f"{phase}.mlir")[1]
    layer0 = consts.get("cL0", 1)
    c0 = sorted(e[2] for e in sync if e[0] == 0)
    red_by_c, dma_by_c, sync_by_c = by_cluster(red), by_cluster(dma), by_cluster(sync)
    red_end = EndIndex([(e[2], e[0], e[3], e[4], e[5], e[1], e[6]) for e in red])
    dma_end = EndIndex([(e[2], e[0], e[3]) for e in dma])
    pm = [n for n in prog if n[0] == "mark"]
    got_tags = [re.sub(r"\d+$", "", t) for t, _ in marks]
    if got_tags != [n[1] for n in pm][:len(marks)]:      # a log cut short (PROF_STOP_NS) is a prefix of the program's marks
        notes.append(f"{phase}: mark sequence of the log ({len(marks)}) differs from the program ({len(pm)}); segment rows only")
        pm = None
    # split the expanded program into segments of ops between marks
    segs, cur = [], []
    for n in prog:
        if n[0] == "mark":
            segs.append(cur)
            cur = []
        else:
            cur.append(n)
    rows = []
    state = {"pre_layers": True}
    prev_rel = marks[0][1] if marks else 0
    for i, (tag, t_mark) in enumerate(marks):
        t_start = marks[i - 1][1] if i else t_mark
        rel = [t for t in c0 if t_start < t <= t_mark + tol] if i else []
        ops = segs[i] if (pm is not None and i < len(segs)) else []
        if i == 0:
            continue
        if len(ops) != len(rel):
            notes.append(f"{phase}: segment {tag} has {len(rel)} barriers, program {len(ops)} ops -> one row")
            ops = []
        if not ops:
            r = _row(phase, "", "", f"segment:{tag}", "segment", "seg", t_start, t_mark, "mark", t_start, t_mark,
                     red_by_c, dma_by_c, sync_by_c, red_end, dma_end)
            r["seg"] = tag
            rows.append(r)
            continue
        w0 = t_start
        for n, rel_t in zip(ops, rel):
            _, op, srcs, dst, text, env = n
            if op == "rmsnorm" or op == "layernorm":
                if re.sub(r"_[sc]$", "", srcs[1]) in ("g1",):
                    state["pre_layers"] = False
            lab = label(op, srcs, dst, text, phase, state)
            for nm in srcs:
                m_ = re.search(r"_([sc])$", nm)      # expert: weights / KV of a self (_s) or a cross (_c) layer; later ops inherit the kind
                if m_:
                    state["kind"] = m_[1]
            layer, step = _layer_step(phase, env, state.get("kind", "s"), layer0)
            r = _row(phase, layer, step, lab, op_class(lab, op), "op", w0, rel_t, "measured:barrier", w0, rel_t,
                     red_by_c, dma_by_c, sync_by_c, red_end, dma_end, op=op)
            r["seg"] = tag
            rows.append(r)
            w0 = rel_t
    subs = []
    redc = defaultdict(list)
    for e in red:
        redc[e[0]].append(e)
    for r in rows:
        r["program"] = phase
        r["phase"] = group_of(phase, r["seg"])
        if r["op"] in _ATTN_SPLIT:
            subs += attention_split(r, redc)
    rows += subs
    return rows, notes


def attention_split(r, redc):
    """Fused attention is ONE library call (one barrier), so its parts cannot be delimited by barriers. Each cluster runs
    one head per RedMulE pair: QK^T first, P.V second (RedMulE `Finished` events of the window). The parts are DERIVED from
    the event times, averaged over the clusters that have both events: load/setup = window start -> QK^T start, QK^T,
    softmax gap = QK^T end -> P.V start (cores: scale, mask, max, exp, sum, normalise; plus DMA), P.V, tail = P.V end -> barrier."""
    w0, w1 = r["start_ns"], r["end_ns"]
    parts = defaultdict(list)
    for c, evs in redc.items():
        ev = sorted(e for e in evs if w0 < e[2] <= w1)
        if len(ev) < 2:
            continue
        q, v = ev[0], ev[-1]
        for k, d in (("attn_load", q[1] - w0), ("attn_qk", q[2] - q[1]), ("attn_softmax_gap", v[1] - q[2]), ("attn_pv", v[2] - v[1]),
                     ("attn_tail", w1 - v[2])):
            parts[k].append(d)
    out = []
    for k, v in parts.items():
        d = round(sum(v) / len(v))
        out.append({**{c: "" for c in COLUMNS}, "phase": r["phase"], "program": r["program"], "seg": r["seg"], "layer": r["layer"],
                    "step": r["step"], "op": k, "class": "attention", "level": "sub", "start_ns": w0, "end_ns": w0 + d, "dur_ns": d,
                    "cluster": f"mean of {len(v)}", "count": 1, "source": "derived:redmule-events", "bound": ""})
    return out


def _layer_step(phase, env, kind, layer0):
    step = env.get("s", "")
    if phase == "vision":
        return (env["L"] + 1 if "L" in env else ""), ""
    if phase.startswith("prefix"):
        return (env["L"] + layer0 if "L" in env else ""), ""
    if phase == "expert":
        if "p" in env:
            if "s" not in env:
                return env["p"], ""                  # KV projection of prefix layer p
            return 2 * env["p"] + (1 if kind == "c" else 0), step
        return "", step
    return "", ""


_ATTN_SPLIT = {"attention_fused", "xattn_self_and_prefix", "xattn_prefix"}


def _row(phase, layer, step, lab, cls, level, w0, w1, source, a, b, red_by_c, dma_by_c, sync_by_c, red_end, dma_end, op=""):
    dur, rb, db, sb, flops, by, uti = window_stats(w0, w1, red_by_c, dma_by_c, sync_by_c, red_end, dma_end)
    crit = min(range(NCL), key=lambda c: (sb[c], -rb[c]))     # the cluster that waits least at the barrier is on the critical path
    return {"phase": phase, "program": phase, "seg": "", "layer": layer, "step": step, "op": lab, "class": cls, "level": level, "start_ns": w0, "end_ns": w1,
            "dur_ns": w1 - w0, "cluster": "all", "count": 1, "source": source,
            "redmule_busy_ns": round(sum(rb) / NCL), "idma_busy_ns": round(sum(db) / NCL), "sync_wait_ns": round(sum(sb) / NCL),
            "sync_wait_max_ns": max(sb), "redmule_flops": flops, "hbm_rd": by["hbm_rd"], "hbm_wr": by["hbm_wr"], "c2c": by["c2c"],
            "local": by["local"], "active_clusters": sum(1 for v in rb if v > 0), "crit_redmule_ns": rb[crit], "crit_idma_ns": db[crit],
            "redmule_uti": round(uti, 3), "bound": bound_of(cls, dur, rb, db, sb, crit) if cls != "segment" else ""}


def cmd_ops(a):
    run = Path(a.run)
    phases = a.phases.split(",") if a.phases else [p for p in ("vision", "prefix_a", "prefix_b", "expert") if (run / f"{p}.mlir").exists()]
    rows, notes = [], []
    for ph in phases:
        r, n = phase_ops(run, ph)
        rows += r
        notes += n
    out = Path(a.out or run / "ops.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    for n in notes:
        print("note:", n)


# ---------------------------------------------------------------------------------------------- summary tables
def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("dur_ns", "start_ns", "end_ns", "redmule_busy_ns", "idma_busy_ns", "sync_wait_ns", "sync_wait_max_ns", "redmule_flops",
                  "hbm_rd", "hbm_wr", "c2c", "local", "count"):
            r[k] = int(float(r[k] or 0))
    return rows


def excluded(r):
    return r["phase"] in ("excluded", "prefix_tail")


def _agg(rows, keyf):
    out = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = out[keyf(r)]
        a["dur"] += r["dur_ns"]
        a["n"] += r["count"]
        for k in ("redmule_busy_ns", "idma_busy_ns", "sync_wait_ns", "redmule_flops", "hbm_rd", "hbm_wr", "c2c", "local"):
            a[k] += r[k]
        a["crit_r"] += float(r["crit_redmule_ns"] or 0)
        a["crit_d"] += float(r["crit_idma_ns"] or 0)
    return out


REF = {"vision": 27.640, "connector": 0.280, "prefix": 15.039, "expert": 77.756}   # LOG G3.7-gate / run.log of fixi-rung6 (ms)
PEAK = NCL * PEAK_FLOP_PER_NS_CLUSTER                                               # FLOP per ns, whole chip (= GFLOP/s)
PH_ORDER = ["vision", "connector", "prefix", "expert"]


def cmd_summary(a):
    rows = [r for r in load(a.csv) if r["level"] in ("op", "seg") and not excluded(r)]
    tot = sum(r["dur_ns"] for r in rows)
    pt = {ph: sum(r["dur_ns"] for r in rows if r["phase"] == ph) for ph in PH_ORDER}
    ref = a.ref and {k: v for k, v in REF.items()}
    print("## Phases (sum of op windows" + (" vs the gate's marks" if ref else "") + ", ms)\n")
    print("| phase | sum of ops | gate (marks) | diff % | % of total |\n| --- | ---: | ---: | ---: | ---: |" if ref else
          "| phase | sum of ops | % of total |\n| --- | ---: | ---: |")
    for ph in PH_ORDER:
        if pt[ph]:
            print(f"| {ph} | {pt[ph] / 1e6:.3f} | " + (f"{REF[ph]:.3f} | {100 * (pt[ph] / 1e6 / REF[ph] - 1):+.2f} | " if ref else "") + f"{100 * pt[ph] / tot:.1f} |")
    print(f"| total | {tot / 1e6:.3f} | " + (f"{sum(REF.values()):.3f} | {100 * (tot / 1e6 / sum(REF.values()) - 1):+.2f} | " if ref else "") + "100 |\n")
    ag = _agg(rows, lambda r: (r["phase"], r["op"], r["class"]))
    print(f"## Top {a.top} ops by total time\n\n| phase | op | class | reps | total ms | mean us | % phase | % total | RedMulE busy % | iDMA busy % | barrier wait % |\n"
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (ph, op, cl), v in sorted(ag.items(), key=lambda kv: -kv[1]["dur"])[:a.top]:
        d = v["dur"]
        print(f"| {ph} | {op} | {cl} | {int(v['n'])} | {d / 1e6:.3f} | {d / v['n'] / 1e3:.1f} | {100 * d / pt[ph]:.1f} | {100 * d / tot:.1f} | "
              f"{100 * v['redmule_busy_ns'] / d:.1f} | {100 * v['idma_busy_ns'] / d:.1f} | {100 * v['sync_wait_ns'] / d:.1f} |")
    print("\n## Op classes (share of each phase and of the total)\n")
    cls = _agg(rows, lambda r: r["class"])
    print("| class | ms | % total | " + " | ".join(f"% {p}" for p in PH_ORDER) + " |\n| --- | ---: | ---: | " + " | ".join("---:" for _ in PH_ORDER) + " |")
    cp = _agg(rows, lambda r: (r["class"], r["phase"]))
    for k, v in sorted(cls.items(), key=lambda kv: -kv[1]["dur"]):
        if k == "segment":
            continue
        print(f"| {k} | {v['dur'] / 1e6:.3f} | {100 * v['dur'] / tot:.1f} | " + " | ".join(f"{100 * cp[(k, p)]['dur'] / pt[p]:.1f}" if pt[p] else "-" for p in PH_ORDER) + " |")
    print("\n## Hardware activity per phase and class (cluster-mean busy % of the op windows; 'barrier wait' = idle at the closing barrier)\n")
    print("| phase | class | ms | RedMulE busy % | iDMA busy % | barrier wait % | chip FLOP/s, % of peak | HBM read MB | HBM write MB | c2c MB |\n"
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    cph = _agg(rows, lambda r: (r["phase"], r["class"]))
    for (ph, cl), v in sorted(cph.items(), key=lambda kv: (PH_ORDER.index(kv[0][0]) if kv[0][0] in PH_ORDER else 9, -kv[1]["dur"])):
        if cl == "segment" or v["dur"] < 0.002 * tot:
            continue
        d = v["dur"]
        print(f"| {ph} | {cl} | {d / 1e6:.3f} | {100 * v['redmule_busy_ns'] / d:.1f} | {100 * v['idma_busy_ns'] / d:.1f} | "
              f"{100 * v['sync_wait_ns'] / d:.1f} | {100 * v['redmule_flops'] / (d * PEAK):.2f} | {v['hbm_rd'] / 1e6:.1f} | {v['hbm_wr'] / 1e6:.1f} | {v['c2c'] / 1e6:.1f} |")
    print("\n## Phase totals: bytes and FLOPs\n\n| phase | ms | GFLOP (RedMulE) | % of chip peak | HBM read GB | HBM write GB | mean HBM GB/s | c2c GB |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    pp = _agg(rows, lambda r: r["phase"])
    for ph in PH_ORDER:
        v = pp.get(ph)
        if v:
            d = v["dur"]
            print(f"| {ph} | {d / 1e6:.3f} | {v['redmule_flops'] / 1e9:.1f} | {100 * v['redmule_flops'] / (d * PEAK):.2f} | {v['hbm_rd'] / 1e9:.3f} | {v['hbm_wr'] / 1e9:.3f} | "
                  f"{(v['hbm_rd'] + v['hbm_wr']) / d:.1f} | {v['c2c'] / 1e9:.3f} |")
    print("\n## GEMM / attention-GEMM utilisation per op (FLOPs from the RedMulE M-N-K events of the window; peak 131.07 TFLOP/s = 16 x 8192 FLOP/cycle x 1 GHz)\n")
    print("| phase | op | reps | mean us | GFLOP per op | TFLOP/s | % of chip peak | active clusters | sim-reported array uti | crit-cluster RedMulE busy % | crit-cluster iDMA busy % | bound |\n"
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    g = defaultdict(list)
    for r in rows:
        if r["redmule_flops"] > 0 and r["class"] in ("GEMM", "attention"):
            g[(r["phase"], r["op"])].append(r)
    for (ph, op), rs in sorted(g.items(), key=lambda kv: -sum(x["dur_ns"] for x in kv[1]))[:a.top]:
        d = sum(x["dur_ns"] for x in rs)
        fl = sum(x["redmule_flops"] for x in rs)
        act = sum(float(x["active_clusters"] or 0) for x in rs) / len(rs)
        uti = sum(float(x["redmule_uti"] or 0) for x in rs) / len(rs)
        cr = sum(float(x["crit_redmule_ns"] or 0) for x in rs)
        cd = sum(float(x["crit_idma_ns"] or 0) for x in rs)
        bounds = defaultdict(int)
        for x in rs:
            bounds[x["bound"]] += 1
        bd = max(bounds, key=bounds.get)
        print(f"| {ph} | {op} | {len(rs)} | {d / len(rs) / 1e3:.1f} | {fl / len(rs) / 1e9:.3f} | {fl / d / 1e3:.2f} | {100 * fl / (d * PEAK):.1f} | {act:.1f} | {uti:.2f} | "
              f"{100 * cr / d:.1f} | {100 * cd / d:.1f} | {bd} |")
    print("\n## Bound classification (share of time)\n\n| bound | ms | % |\n| --- | ---: | ---: |")
    bb = _agg(rows, lambda r: r["bound"] or "n/a")
    for k, v in sorted(bb.items(), key=lambda kv: -kv[1]["dur"]):
        print(f"| {k} | {v['dur'] / 1e6:.3f} | {100 * v['dur'] / tot:.1f} |")
    print("\n## Class x bound (ms; `imbalance` = mean barrier wait of all clusters > 50 % of the window, counted inside its base class)\n")
    bases = ["compute-bound", "data-movement-bound", "latency-bound", "core-bound", "sync"]
    print("| class | " + " | ".join(bases) + " | of which imbalance | total ms |\n| --- | " + " | ".join("---:" for _ in bases) + " | ---: | ---: |")
    cb = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r["class"] == "segment":
            continue
        b = (r["bound"] or "n/a").replace("+imbalance", "")
        cb[r["class"]][b] += r["dur_ns"]
        cb[r["class"]]["_tot"] += r["dur_ns"]
        if "+imbalance" in (r["bound"] or ""):
            cb[r["class"]]["_imb"] += r["dur_ns"]
    for k, v in sorted(cb.items(), key=lambda kv: -kv[1]["_tot"]):
        print(f"| {k} | " + " | ".join(f"{v[b] / 1e6:.3f} ({100 * v[b] / v['_tot']:.0f} %)" for b in bases) + f" | {v['_imb'] / 1e6:.3f} | {v['_tot'] / 1e6:.3f} |")
    allr = sum(r["redmule_busy_ns"] for r in rows), sum(r["idma_busy_ns"] for r in rows), sum(r["sync_wait_ns"] for r in rows), sum(r["redmule_flops"] for r in rows)
    print(f"\nWhole run: cluster-mean RedMulE busy {100 * allr[0] / tot:.2f} %, iDMA busy {100 * allr[1] / tot:.2f} %, barrier wait {100 * allr[2] / tot:.2f} % of the time; "
          f"{allr[3] / 1e9:.1f} GFLOP in {tot / 1e6:.3f} ms = {allr[3] / tot / 1e3:.2f} TFLOP/s = {100 * allr[3] / (tot * PEAK):.2f} % of the {PEAK / 1e3:.1f} TFLOP/s peak.")
    print("\n## RedMulE-derived split of the fused attention ops (level=sub rows; mean over the clusters that ran a head)\n\n| phase | part | reps | mean us | % of attention op |\n| --- | --- | ---: | ---: | ---: |")
    subs = [r for r in load(a.csv) if r["level"] == "sub" and r["phase"] in PH_ORDER]
    sg = _agg(subs, lambda r: (r["phase"], r["op"]))
    ops_t = {ph: sum(v["dur"] for (p, o, c), v in ag.items() if p == ph and o in _ATTN_SPLIT) for ph in PH_ORDER}
    for (ph, part), v in sorted(sg.items(), key=lambda kv: (PH_ORDER.index(kv[0][0]), ["attn_load", "attn_qk", "attn_softmax_gap", "attn_pv", "attn_tail"].index(kv[0][1]))):
        print(f"| {ph} | {part} | {int(v['n'])} | {v['dur'] / v['n'] / 1e3:.1f} | {100 * v['dur'] / ops_t[ph]:.1f} |")



def cmd_phases(a):
    """Per phase: every op label with its repetition count (layers x steps x heads), mean time and share of the phase."""
    rows = [r for r in load(a.csv) if r["level"] == "op" and not excluded(r)]
    for ph in PH_ORDER:
        rs = [r for r in rows if r["phase"] == ph]
        if not rs:
            continue
        pt = sum(r["dur_ns"] for r in rs)
        ag = _agg(rs, lambda r: (r["op"], r["class"]))
        print(f"### {ph}: {pt / 1e6:.3f} ms\n\n| op | class | reps | mean us | total ms | % of phase | % of total |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        tot = sum(r["dur_ns"] for r in rows)
        for (op, cl), v in sorted(ag.items(), key=lambda kv: -kv[1]["dur"])[:a.top]:
            print(f"| {op} | {cl} | {int(v['n'])} | {v['dur'] / v['n'] / 1e3:.1f} | {v['dur'] / 1e6:.3f} | {100 * v['dur'] / pt:.1f} | {100 * v['dur'] / tot:.1f} |")
        print()


def cmd_layer(a):
    """The ops of one layer (and step) in order: the 'representative layer' table."""
    rows = [r for r in load(a.csv) if r["level"] == "op" and r["phase"] == a.phase and str(r["layer"]) == str(a.layer)
            and (a.step is None or str(r["step"]) == str(a.step))]
    tot = sum(r["dur_ns"] for r in rows)
    print(f"## {a.phase} layer {a.layer}" + (f" step {a.step}" if a.step is not None else "") + f": {tot / 1e3:.1f} us in {len(rows)} ops\n")
    print("| # | op | class | us | % | RedMulE busy % | iDMA busy % | barrier wait % | active cl. | TFLOP/s | HBM rd MB | bound |\n| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for i, r in enumerate(rows):
        d = r["dur_ns"]
        print(f"| {i + 1} | {r['op']} | {r['class']} | {d / 1e3:.1f} | {100 * d / tot:.1f} | {100 * r['redmule_busy_ns'] / d:.1f} | {100 * r['idma_busy_ns'] / d:.1f} | "
              f"{100 * r['sync_wait_ns'] / d:.1f} | {r['active_clusters']} | {r['redmule_flops'] / d / 1e3:.2f} | {r['hbm_rd'] / 1e6:.2f} | {r['bound']} |")


LIBERO_RULES = {
    # (input, seg tag regex) -> multiplier; the G4a recipe (LOG G4a): vision 2 x (embed + 12 x layer + post-LN) from the 1-layer run,
    # connector = the start -> emb segment of the 2-camera prefix run, prefix 2 x (8 layers), expert = KV projection + 10 x step
    # from the 2-step run. Anything else (the vision run's own connector, the prefix tail, dumps) is dropped.
    "vision": [(r"vemb", 2), (r"attn|layer", 24), (r"vis", 2)],
    "prefix": [(r"emb", 1), (r"attn|layer", 2)],
    "expert": [(r"kvproj", 1), (r"step", 5)],
}
_SCALED = ("dur_ns", "count", "redmule_busy_ns", "idma_busy_ns", "sync_wait_ns", "redmule_flops", "hbm_rd", "hbm_wr", "c2c", "local",
           "crit_redmule_ns", "crit_idma_ns")


def cmd_libero(a):
    """Scale the three G4a partial runs into the LIBERO (2 cameras, 1024 tokens/camera) phase composition. ESTIMATE: every row
    is multiplied by its repetition factor (rows then hold the SUM over the repetitions, count likewise; start/end are the
    measured instance's)."""
    out = []
    for name, path in (("vision", a.vision), ("prefix", a.prefix), ("expert", a.expert)):
        for r in csv.DictReader(open(path)):
            seg = re.sub(r"\d+$", "", r["seg"])
            mult = next((m for rx, m in LIBERO_RULES[name] if re.fullmatch(rx, seg)), 0)
            if r["level"] == "seg" or mult == 0 or r["phase"] in ("excluded", "prefix_tail"):
                continue
            for k in _SCALED:
                if r[k] != "":
                    r[k] = float(r[k]) * mult
            r["source"] = f"estimate:scaled x{mult} (from the {name} partial run)"
            out.append(r)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(out)
    tot = sum(float(r["dur_ns"]) for r in out if r["level"] == "op")
    print(f"wrote {a.out}: {len(out)} rows; scaled total {tot / 1e6:.3f} ms (G4a estimate 479.117)")


INSN_CLASSES = ["int_alu", "branch", "int_load", "int_store", "fp_alu", "fp_mem", "frep_ssr", "dma_issue", "dma_poll", "csr", "atomic", "wfi"]


def read_bins(path):
    """-> {(unit): {bin_start_ns: {class: n}}}, bin_ns, funcs {(unit, function): n}  (cluster 0 only is traced by default)."""
    import gzip
    bins, funcs, bin_ns = defaultdict(lambda: defaultdict(lambda: defaultdict(int))), defaultdict(int), 1000
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                m = re.search(r"bin = (\d+) ns", ln)
                bin_ns = int(m[1]) if m else bin_ns
                continue
            t = ln.rstrip("\n").split(",")
            if t[0] == "F":
                funcs[(t[2], t[3])] += int(t[5])
            else:
                bins[t[1]][int(t[2])][t[3]] += int(t[4])
    return bins, bin_ns, funcs


def cmd_insn(a):
    """Per-op instruction mix of the traced cluster-0 cores (pe0..2 = integer cores, fp_ss0..2 = their FP sub-systems):
    instructions issued per window cycle (1 GHz, so per ns) and the class shares. Bins (1 us) are apportioned to windows by
    overlap (uniform inside a bin): for ops shorter than ~10 us the numbers carry up to ~10 % quantisation error."""
    rows = [r for r in load(a.ops) if r["level"] == "op" and r["phase"] != "excluded"]
    bins, bin_ns, funcs = read_bins(a.bins)
    out = []
    for r in rows:
        w0, w1 = r["start_ns"], r["end_ns"]
        for u, ub in sorted(bins.items()):
            cnt = defaultdict(float)
            b = (w0 // bin_ns) * bin_ns
            while b < w1:
                ov = (min(b + bin_ns, w1) - max(b, w0)) / bin_ns
                if ov > 0 and b in ub:
                    for k, n in ub[b].items():
                        cnt[k] += n * ov
                b += bin_ns
            out.append({"phase": r["phase"], "layer": r["layer"], "step": r["step"], "op": r["op"], "class": r["class"], "unit": u,
                        "window_ns": r["dur_ns"], "insns": round(sum(cnt.values())), **{k: round(cnt.get(k, 0)) for k in INSN_CLASSES}})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["phase", "layer", "step", "op", "class", "unit", "window_ns", "insns"] + INSN_CLASSES)
        w.writeheader()
        w.writerows(out)
    print("## Core activity of cluster 0 per op class (instructions retired per ns of window; a core issues at most 1 per ns)\n")
    print("| phase | class | window ms | pe0/pe1 issue/ns | pe2 (DMA core) issue/ns | pe2 dmstati polls/ns | pe0/pe1 idle % (1 - issue) | FP-subsystem FP ops/ns (mean of 3) |\n"
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    agg = defaultdict(lambda: defaultdict(float))
    for o in out:
        k = (o["phase"], o["class"])
        u = o["unit"]
        g = agg[k]
        if u == "pe0":
            g["win"] += o["window_ns"]
        if u in ("pe0", "pe1"):
            g["c01"] += o["insns"]
        elif u == "pe2":
            g["c2"] += o["insns"]
            g["poll"] += o["dma_poll"]
        elif u.startswith("fp_ss"):
            g["fp"] += o["fp_alu"] + o["fp_mem"]
    for (ph, cl), g in sorted(agg.items(), key=lambda kv: -kv[1]["win"]):
        w = g["win"]
        if w < 1e4 or cl in ("segment", "sync"):
            continue
        print(f"| {ph} | {cl} | {w / 1e6:.3f} | {g['c01'] / 2 / w:.3f} | {g['c2'] / w:.3f} | {g['poll'] / w:.3f} | {100 * (1 - g['c01'] / 2 / w):.1f} | {g['fp'] / 3 / w:.3f} |")
    print()
    print(f"wrote {a.out}: {len(out)} rows; hottest functions (instructions, cluster 0):")
    for (u, fn), n in sorted(funcs.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {u:8} {fn:28} {n}")


def cmd_compact(a):
    keep = re.compile(r"Finished|Cluster Sync|\[mark\]|Preloading Done|Preheating Done|^PASS|^FAIL")
    p = subprocess.Popen(["zstd", "-q", "-19", "-o", a.out, "-f"], stdin=subprocess.PIPE, text=True)
    n = k = 0
    with open_log(a.log) as f:
        for ln in f:
            n += 1
            if _trace_start(ln) >= 0 or "[mark]" in ln:
                if keep.search(_ANSI.sub("", ln)) or "[mark]" in ln:
                    p.stdin.write(ln)
                    k += 1
            elif not (ln.startswith("0: -1:") or "Launching" in ln or "pkg_resources" in ln or "BFD:" in ln):
                p.stdin.write(ln)
    p.stdin.close()
    p.wait()
    print(f"{a.log}: {n} lines, kept {k} trace/mark lines -> {a.out} ({Path(a.out).stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("ops"); o.add_argument("run"); o.add_argument("--phases"); o.add_argument("--out"); o.set_defaults(f=cmd_ops)
    s = sp.add_parser("summary"); s.add_argument("csv"); s.add_argument("--top", type=int, default=25); s.add_argument("--ref", action="store_true", help="compare phases with the rung-6 gate numbers");  s.set_defaults(f=cmd_summary)
    i = sp.add_parser("insn"); i.add_argument("ops"); i.add_argument("bins"); i.add_argument("--out", required=True); i.set_defaults(f=cmd_insn)
    q = sp.add_parser("phases"); q.add_argument("csv"); q.add_argument("--top", type=int, default=40); q.set_defaults(f=cmd_phases)
    y = sp.add_parser("layer"); y.add_argument("csv"); y.add_argument("--phase", required=True); y.add_argument("--layer", required=True); y.add_argument("--step"); y.set_defaults(f=cmd_layer)
    l = sp.add_parser("libero"); l.add_argument("--vision", required=True); l.add_argument("--prefix", required=True); l.add_argument("--expert", required=True); l.add_argument("--out", required=True); l.set_defaults(f=cmd_libero)
    c = sp.add_parser("compact"); c.add_argument("log"); c.add_argument("out"); c.set_defaults(f=cmd_compact)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
