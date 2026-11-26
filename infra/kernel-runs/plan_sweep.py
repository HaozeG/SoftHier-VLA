#!/usr/bin/env python3
"""Plan-constrained sweep: the fastest kernels that still deploy as upstream's whole-graph plan says.

    plan_sweep.py plan | run | pick   [--page W1|W1-b] [--top N] [--kind gemm|attn|elt]...

Every run here keeps the deployment plan's constraints (workload_modeling.deploy_plan): one HBM edge per tensor
(producer and consumer agree, HBM_PLACE as the plan says), attention operands head-major through the GEMMs'
RESHA_Z_TO / X_FROM, head padding carried by every kernel that touches a head-major tensor. Only the free knobs are
searched:
  - attention settings, and with them the head padding. The padding couples kernels across stages: expert self and
    cross attention share Q (q_pad), and the prefix K/V feed the expert's cross K/V projections (kv_pad). So the
    selection unit is one attention setting per attention call for the whole graph ("combination"): Plan(attn=...)
    builds the run list of each combination and rejects invalid ones (its asserts). Padding options per attention
    call: the (at most 2) least padded ones within 1.3x of its least padded work (only the unpadded one when
    the call needs no padding) (attn_sweep's valid space: Br / sy a multiple of 64, Bc / sx at
    most 128, head coverage, L1), plus upstream's own (attn_opt), so upstream's combination is always scored;
  - GEMM tile, group shape and strategy: the earlier sweeps' generators on the plan's shape, fitted to the plan
    (plan_runs.fit: reshaped dimension tiles 64 or 32, padded by whole heads, head-major group gaps; non-square
    groups only with M_iter = 1), the --top best by kernel_model, plus upstream's opt() settings and the earlier
    sweep's selected setting for the operator, both fitted;
  - FlatAttention sx, sy, Br, Bc, async for each padding: the --top best (earlier attn_sweep time where that setting
    ran, then attn_sweep's order), plus upstream's setting;
  - Eltwise: the stage-1 schedule and the pipelined one (ELT_PIPELINE). Other row kernels have no knob.
The sweep is independent of the upstream-rule runs: it runs every candidate itself (upstream's opt() / attn_auto
settings are among its seeds) and picks from its own runs only, so a key can come out slower than upstream's rule
where the sweep's candidates miss upstream's setting or fix. --with-upstream-runs instead takes the runs of
results/plan_runs_h1.csv as candidates too.

A run is identified by its variant id `<stage.key>@<padded sizes>` (the key names the logical shape) and its
candidate. Results: results/plan_sweep_h1.csv (every run, also those that failed the check); pick writes
results/plan_sweep_pick_<page>_h1.csv (selected runs of the best combination) and
results/plan_sweep_combos_<page>_h1.csv (every combination's composed time). Run folders:
$SHVLA_DATA/stage1-h1/plan_sweep/<variant>/<cand>/.
"""
import argparse
import csv
import itertools
import re
import subprocess
import sys

from gemm_sweep import HERE, DATA, parse_run
import gemm_sweep as G
import gemm_smallm_sweep as S
from gemm_upstream import opt
import attn_sweep as A
import row_kernels as RK
import plan_runs as PR

sys.path.insert(0, str(HERE.parent / "workload-modeling"))
from workload_modeling.deploy_plan import Plan  # noqa: E402
from workload_modeling.upstream_rules import AttnSetting  # noqa: E402

CSV = HERE / "results/plan_sweep_h1.csv"
RUNS = DATA / "stage1-h1/plan_sweep"
ATTN_OPS = (("vision", "qk"), ("prefix", "qk"), ("expert", "self_qk"), ("expert", "cross_qk"))
PAD_SLACK = 1.3
PAD_KEEP = 2
FIELDS = ["vid", "key", "kernel", "ops", "count", "cand", "source", "strategy", "m_size", "n_size", "k_size",
          "m_tile", "n_tile", "k_tile", "scale_x", "scale_y", "groups", "gap_x", "gap_z", "q_pad", "kv_pad", "sx",
          "sy", "Br", "Bc", "async", "pipeline", "fix", "est_ns", "time_ns", "bad", "max_ulp", "status", "run_dir"]


def page_file(kind, page):
    return HERE / f"results/plan_sweep_{kind}_{page.lower().replace('-', '')}_h1.csv"


def vid(key, cfg):
    return key + "@" + "x".join(str(cfg[k]) for k in ("m_size", "n_size", "k_size", "q_pad", "kv_pad") if k in cfg)


# ------------------------------------------------------------------------------------------- attention
def attn_space(hq, hkv, q, kv):
    """attn_sweep.candidates' valid space for any shape (the same rules), as dicts with an AttnSetting."""
    r = hq // hkv
    out = []
    for sx in (1, 2, 4, 8):
        for sy in (1, 2, 4, 8):
            G_ = 64 // (sx * sy)
            if hkv > G_ and hkv % G_:
                continue
            per_group = max(hkv // G_, 1)
            for brs in (64, 128, 192, 256):
                for bcs in (32, 64, 96, 128):
                    Br, Bc = brs * sy, bcs * sx
                    qp = q
                    while (qp * r) % Br:
                        qp += 1
                    kvp = A.up(kv, Bc)
                    work = qp * kvp / (q * kv)
                    if work > A.WORK_CAP:
                        continue
                    if 2 * 2 * (2 * brs * A.D + 2 * bcs * A.D + brs * bcs) + 10 * brs * 2 > A.L1_BUDGET:
                        continue
                    for asy in ((1, 0) if per_group % 2 == 0 else (0,)):
                        out.append(dict(q_pad=qp, kv_pad=kvp, sx=sx, sy=sy, Br=Br, Bc=Bc, async_=asy,
                                        work_ratio=round(work, 3), active=min(G_, hkv) * sx * sy))
    out.sort(key=lambda c: (round(c["work_ratio"] / c["active"], 5), -(c["Br"] // c["sy"]) * (c["Bc"] // c["sx"])))
    for c in out:
        c["cand"] = attn_name(c)
    return out


def attn_name(c):
    return f"attn_s{c['sx']}x{c['sy']}_b{c['Br']}x{c['Bc']}" + ("_async" if c.get("async_") else "")


def setting(c, fix=""):
    return AttnSetting(c["sx"], c["sy"], c["Br"], c["Bc"], c["q_pad"], c["kv_pad"], c["async_"], fix)


def attn_heads(plan, stage, op):
    h, q, kv, _ = plan._row(stage, op).shape
    return h, (h if stage == "vision" else plan.dims.t_kv_heads), q, kv


def old_attn_times():
    """Earlier attn_sweep runs (same placement as the plan: Q, O west; K, V south): time per (shape, setting)."""
    t = {}
    for r in A.rows():
        if r["status"] == "ok":
            c = dict(sx=int(r["sx"]), sy=int(r["sy"]), Br=int(r["Br"]), Bc=int(r["Bc"]), async_=int(r["async"]))
            t[(r["shape"], int(r["q_pad"]), int(r["kv_pad"]), attn_name(c))] = int(r["time_ns"])
    return t


def combos(low):
    """Valid combinations: (stage, op) -> padding (q_pad, kv_pad), with a representative setting each."""
    p0 = Plan(lowering=low)
    opts = {}
    for so in ATTN_OPS:
        h, hkv, q, kv = attn_heads(p0, *so)
        space = attn_space(h, hkv, q, kv)
        w0 = min(c["work_ratio"] for c in space)
        up = p0.attn[so]
        rep = {(up.q_pad, up.kv_pad): up}             # upstream's padding with upstream's setting
        # paddings within PAD_SLACK of the least padded work, at most PAD_KEEP of them by work; none if the call
        # needs no padding (unpadded rows admit valid settings, padding only adds work)
        work = {}
        for c in space:
            pd = (c["q_pad"], c["kv_pad"])
            work[pd] = min(work.get(pd, 9e9), c["work_ratio"])
        keep = sorted((pd for pd, wk in work.items() if wk <= w0 * PAD_SLACK), key=lambda pd: (work[pd], pd))
        keep = keep[:1] if w0 == 1.0 else keep[:PAD_KEEP]
        for c in space:
            pd = (c["q_pad"], c["kv_pad"])
            if pd in keep:
                rep.setdefault(pd, setting(c))
        opts[so] = rep
    out = []
    for pads in itertools.product(*(sorted(opts[so]) for so in ATTN_OPS)):
        attn = {so: opts[so][pd] for so, pd in zip(ATTN_OPS, pads)}
        try:
            plan = Plan(lowering=low, attn=attn)
        except AssertionError:
            continue
        upstream = all(pd == (p0.attn[so].q_pad, p0.attn[so].kv_pad) for so, pd in zip(ATTN_OPS, pads))
        out.append(dict(pads=dict(zip(ATTN_OPS, pads)), plan=plan, upstream=upstream))
    return out


def attn_cands(plan, key, run, top, old):
    so = next(so for so in ATTN_OPS if f"{so[0]}." in key and so[1] in run["ops"])
    h, hkv, q, kv = attn_heads(plan, *so)
    cfg = run["cfg"]
    name = f"{h}x{q}x{kv}x64"
    space = [c for c in attn_space(h, hkv, q, kv) if (c["q_pad"], c["kv_pad"]) == (cfg["q_pad"], cfg["kv_pad"])]
    space.sort(key=lambda c: old.get((name, c["q_pad"], c["kv_pad"], c["cand"]), 10 ** 12))
    picked = space[:top]
    up = dict(sx=cfg["sx"], sy=cfg["sy"], Br=cfg["Br"], Bc=cfg["Bc"], async_=cfg["async_"], q_pad=cfg["q_pad"],
              kv_pad=cfg["kv_pad"])
    up["cand"] = attn_name(up)
    if up["cand"] not in {c["cand"] for c in picked}:
        picked.append(dict(up, source="plan setting"))
    return [(dict(c, source=c.get("source", "attention space"), fix=""), None) for c in picked]


# ------------------------------------------------------------------------------------------- GEMM
def _sq(c):
    """gemm_sweep candidate (square groups) in the smallm / plan_runs form."""
    return dict(c, scale_x=c["scale"], scale_y=c["scale"],
                cand=f"{c['strategy']}_x{c['scale']}y{c['scale']}g{c['groups']}_t{c['m_tile']}x{c['n_tile']}x{c['k_tile']}")


def square_all(M, N, K):
    out = []
    for mt in G.TILES:
        for nt in G.TILES:
            for kt in G.TILES:
                for s in G.SCALES:
                    c = G.make_cand(M, N, K, mt, nt, kt, s, 1, "Nosplit")
                    if c:
                        out.append(_sq(c))
                    g = 2
                    while g * s * s <= G.MESH * G.MESH:
                        c = G.make_cand(M, N, K, mt, nt, kt, s, g, "splitN")
                        if c and c["n_size"] // (g * s * nt) >= 1:
                            out.append(_sq(c))
                        g *= 2
    return out


def earlier_pick(logical):
    """The earlier sweep's selected setting for this operator (results/final_h1.csv), as a candidate dict."""
    name = "gemm_" + "x".join(map(str, logical))
    r = next((r for r in csv.DictReader(open(HERE / "results/final_h1.csv")) if r["name"] == name), None)
    if not r:
        return None
    m = re.match(r"(?:T:)?(\w+?)_(?:s(\d+)|x(\d+)y(\d+))g(\d+)_t(\d+)x(\d+)x(\d+)", r["setting"])
    if not m:
        return None
    st, s, sx, sy, g, mt, nt, kt = m.groups()
    sx, sy = (int(s), int(s)) if s else (int(sx), int(sy))
    M, N, K = logical
    c = S.make_cand(M, N, K, sy, sx, int(nt), int(kt), int(g), st) if st != "Nosplit" else None
    if st == "Nosplit" or c is None:
        mt_, nt_, kt_ = int(mt), int(nt), int(kt)
        c = dict(m_size=A.up(M, sy * mt_), n_size=A.up(N, int(g) * sx * nt_), k_size=A.up(K, kt_), m_tile=mt_,
                 n_tile=nt_, k_tile=kt_, scale_x=sx, scale_y=sy, groups=int(g), strategy=st,
                 est_ns=int(G.model_ns(A.up(M, sy * mt_), A.up(N, int(g) * sx * nt_), A.up(K, kt_), mt_, nt_, kt_,
                                       sx, sy, int(g), st)))
    c.update(m_tile=int(mt), cand=f"{st}_x{sx}y{sy}g{g}_t{mt}x{nt}x{kt}")
    return c


def gemm_cands(run, logical, top):
    cfg = run["cfg"]
    M, N, K = cfg["m_size"], cfg["n_size"], cfg["k_size"]
    pool = square_all(M, N, K) + S.enumerate_all(M, N, K)
    fitted = {}
    for c in pool:
        for f in PR.fit(c, cfg):
            f.setdefault("m_iter", f["m_size"] // (f["scale_y"] * f["m_tile"]))
            if f["scale_x"] != f["scale_y"] and f["m_iter"] > 1:
                continue
            if f["cand"] not in fitted or f["est_ns"] < fitted[f["cand"]]["est_ns"]:
                fitted[f["cand"]] = dict(f, source="sweep space")
    out = sorted(fitted.values(), key=lambda f: f["est_ns"])[:top]
    names = {f["cand"] for f in out}
    extra = [(c, "upstream opt()") for c in opt(M, N, K)]
    e = earlier_pick(logical)
    if e:
        extra.append((e, "earlier sweep pick"))
    for c, src in extra:
        for f in PR.fit(c, cfg):
            f.setdefault("m_iter", f["m_size"] // (f["scale_y"] * f["m_tile"]))
            if f["scale_x"] != f["scale_y"] and f["m_iter"] > 1:
                continue
            if f["cand"] not in names:
                names.add(f["cand"])
                out.append(dict(f, source=src))
    return [(f, None) for f in out]


# ------------------------------------------------------------------------------------------- rows and runs
def row_cands(run):
    out = [(dict(cand="default", source="kernel default", fix="", pipeline=0), None)]
    if run["kernel"] == "elt":
        out.append((dict(cand="pipeline", source="ELT_PIPELINE", fix="", pipeline=1), None))
    return out


def write_cfg(path, key, run, c, logical):
    if run["kernel"] == "gemm":
        PR.write_gemm_cfg(path, run, c, logical)
        return
    if run["kernel"] == "attn":
        name = key.split(".", 1)[1].split("_p")[0].split("_", 1)[1]
        A.write_cfg(path, name, c)
    else:
        RK.write_cfg(path, run["kernel"], run["cfg"])
        if c.get("pipeline"):
            s = path.read_text()
            assert "        self.dtype" in s, key
            path.write_text(s.replace("        self.dtype", "        self.pipeline                = 1\n        self.dtype", 1))
    PR._place(path, run)


def load():
    return list(csv.DictReader(CSV.open())) if CSV.exists() else []


def plan_rows_as_sweep():
    """results/plan_runs_h1.csv rows under the variant id of their run (upstream's padding): candidates too."""
    out = []
    for low in (PR.PAGES[p][0] for p in PR.PAGES):
        runs = Plan(lowering=low).runs()
        for r in PR.load_csv():
            if r["key"] in runs and r.get("time_ns"):
                out.append(dict(r, vid=vid(r["key"], runs[r["key"]]["cfg"]), source="plan_runs_h1.csv"))
    seen, uniq = set(), []
    for r in out:
        if (r["vid"], r["cand"]) not in seen:
            seen.add((r["vid"], r["cand"]))
            uniq.append(r)
    return uniq


def variants(page):
    """vid -> (key, run, plan) over every valid combination of the page."""
    low = PR.PAGES[page][0]
    cs = combos(low)
    out = {}
    for c in cs:
        for k, run in c["plan"].runs().items():
            out.setdefault(vid(k, run["cfg"]), (k, run, c["plan"]))
    return cs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--page", choices=sorted(PR.PAGES), action="append")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--attn-top", type=int, default=4)
    ap.add_argument("--kind", action="append", choices=["gemm", "attn", "elt"])
    ap.add_argument("--with-upstream-runs", action="store_true",
                    help="take results/plan_runs_h1.csv as candidates too (not independent of the upstream plan)")
    a = ap.parse_args()
    pages = a.page or ["W1", "W1-b"]
    rows = load()
    # independent by default: the sweep runs and picks from its own runs only (upstream's settings are among its
    # seeds, but run by the sweep itself); --with-upstream-runs reuses results/plan_runs_h1.csv as candidates
    prior = plan_rows_as_sweep() if a.with_upstream_runs else []
    if a.cmd == "pick":
        for page in pages:
            pick(page, rows + prior)
        return
    old = old_attn_times()
    done = {(r["vid"], r["cand"]) for r in rows + prior}
    todo, seen = [], set()
    for page in pages:
        cs, vs = variants(page)
        print(f"{page}: {len(cs)} valid attention-padding combinations "
              f"({sum(c['upstream'] for c in cs)} upstream), {len(vs)} run variants")
        for v, (k, run, plan) in vs.items():
            kind = run["kernel"] if run["kernel"] in ("gemm", "attn") else "elt" if run["kernel"] == "elt" else "row"
            if a.kind and kind not in a.kind:
                continue
            logical = PR.logical_shape(plan, k)
            if run["kernel"] == "gemm":
                cands = gemm_cands(run, logical, a.top)
            elif run["kernel"] == "attn":
                cands = attn_cands(plan, k, run, a.attn_top, old)
            else:
                cands = row_cands(run)
            for c, skip in cands:
                if (v, c["cand"]) in seen:
                    continue
                seen.add((v, c["cand"]))
                todo.append((v, k, run, plan, logical, c, (v, c["cand"]) in done))
    n_new = sum(not d for *_, d in todo)
    print(f"{len(todo)} runs, {len(todo) - n_new} already done, {n_new} to run")
    if a.cmd == "plan":
        for v, k, run, plan, logical, c, d in todo:
            print(f"{v:54} {c['cand']:<32} {c.get('source', ''):18} est {c.get('est_ns') or 0:>8}"
                  f"  {c.get('fix', '')}{'  done' if d else ''}")
        return
    for v, k, run, plan, logical, c, d in todo:
        if d:
            continue
        out = RUNS / v / c["cand"]
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / f"{run['kernel']}_cfg.py"
        write_cfg(cfg, k, run, c, logical)
        print(f"[{v}] {c['cand']} ...", flush=True)
        p = subprocess.run([str(HERE / "run_kernel.sh"), run["kernel"], str(cfg), str(out)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / "run_kernel.out").write_text(f"exit {p.returncode}\n{p.stdout}")
        t, bad, mu = parse_run(out)
        status = "ok" if (p.returncode == 0 and t is not None and bad == 0) else \
            ("unfit" if "build failed" in p.stdout else "timeout_or_fail" if t is None else "wrong")
        row = dict(c, vid=v, key=k, kernel=run["kernel"], ops=",".join(run["ops"]), count=run["count"],
                   time_ns=t, bad=bad, max_ulp=mu, status=status, run_dir=str(out), **{"async": c.get("async_", "")})
        print(f"    {t} ns, bad {bad}, {status}", flush=True)
        new = not CSV.exists()
        with CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def pick(page, rows):
    """Per combination: composed time = sum over its runs of count x the fastest checked candidate of the run's
    variant; selected = the fastest complete combination. Writes the pick and the combination table."""
    best = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("time_ns"):
            b = best.get(r["vid"])
            if b is None or int(r["time_ns"]) < int(b["time_ns"]):
                best[r["vid"]] = r
    cs, _ = variants(page)
    table = []
    for c in cs:
        runs = c["plan"].runs()
        miss = [k for k, run in runs.items() if vid(k, run["cfg"]) not in best]
        tot = sum(run["count"] * int(best[vid(k, run["cfg"])]["time_ns"]) for k, run in runs.items()
                  if vid(k, run["cfg"]) in best)
        desc = "; ".join(f"{s}.{o} q{p[0]} kv{p[1]}" for (s, o), p in c["pads"].items())
        table.append(dict(combo=desc, upstream=int(c["upstream"]), total_ns=tot, missing=len(miss),
                          missing_keys=" ".join(miss[:6]), c=c))
    with page_file("combos", page).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["combo", "upstream", "total_ns", "missing", "missing_keys"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(table, key=lambda t: (t["missing"], t["total_ns"])))
    full = [t for t in table if not t["missing"]]
    for t in sorted(table, key=lambda t: (t["missing"], t["total_ns"])):
        print(f"{page} {'U' if t['upstream'] else ' '} {t['total_ns'] / 1e6:9.3f} ms  missing {t['missing']}  {t['combo']}")
    if not full:
        print(f"{page}: no complete combination yet")
        return
    sel = min(full, key=lambda t: t["total_ns"])
    with page_file("pick", page).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "vid", "kernel", "count", "cand", "source", "time_ns", "total_ns", "ops", "fix", "run_dir"])
        for k, run in sel["c"]["plan"].runs().items():
            b = best[vid(k, run["cfg"])]
            w.writerow([k, vid(k, run["cfg"]), run["kernel"], run["count"], b["cand"], b.get("source", ""),
                        b["time_ns"], int(b["time_ns"]) * run["count"], ",".join(run["ops"]), b.get("fix", ""),
                        b["run_dir"]])
    print(f"{page}: selected {sel['total_ns'] / 1e6:.3f} ms ({sel['combo']})")


if __name__ == "__main__":
    main()
