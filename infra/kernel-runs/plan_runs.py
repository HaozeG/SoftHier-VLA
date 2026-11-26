#!/usr/bin/env python3
"""Stage-1 baseline runs under the deployment plan (stage1-spec §4.3 item 3): every distinct run of
workload_modeling.deploy_plan, each kernel set by upstream's selection rule, placed as the plan says.

    plan_runs.py plan | run | pick   [--kernel gemm|attn|elt|acti|lnorm|norm|rope]... [--key <stage.key>]...

Attention: the attn_auto opt() setting carried by the plan (upstream_rules.attn_opt, fixes recorded). Row kernels and
RoPE: template defaults with the plan's shape and operator. GEMM: upstream opt() (gemm_upstream.opt) on the plan's shape (M = the plan's padded rows), both of its settings
simulated and the faster kept. Where a setting cannot run under the plan, the smallest change is made and written in
the `fix` column:
  - reshaped operands (RESHA_Z_TO / X_FROM, head-major, chunk = head dimension 64): see fit(); M stays the rows
    per head, the reshaped dimension takes tile 64 or 32 and is padded by whole zero heads;
  - group gaps: SummaGEMM adds the group gap as a raw byte offset into the reshaped array, so splitK under X_FROM
    and splitN under Z_TO use gap = block * M * 2 bytes (the head-major offset of the group's block), which needs the
    block to be a whole number of chunks; otherwise the strategy is not run.
Results: results/plan_runs_h1.csv. Run folders: $SHVLA_DATA/stage1-h1/plan/<stage.key>/<cand>/.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from gemm_sweep import HERE, DATA, MESH, model_ns, parse_run
import gemm_smallm_sweep as S
from gemm_upstream import opt
import attn_sweep as A
import row_kernels as RK

sys.path.insert(0, str(HERE.parent / "workload-modeling"))
from workload_modeling.deploy_plan import Plan, D  # noqa: E402

CSV = HERE / "results/plan_runs_h1.csv"
PICK = HERE / "results/plan_pick_h1.csv"
RUNS = DATA / "stage1-h1/plan"
FIELDS = ["key", "kernel", "ops", "count", "cand", "strategy", "m_size", "n_size", "k_size", "m_tile", "n_tile",
          "k_tile", "scale_x", "scale_y", "groups", "gap_x", "gap_z", "fix", "est_ns", "time_ns", "bad", "max_ulp",
          "status", "run_dir"]


def _tile(n, per, cap):
    """Largest tile t <= cap with n % (per * t) == 0."""
    for t in range(min(cap, n), 0, -1):
        if n % (per * t) == 0:
            return t
    return None


def _up(x, a):
    return -(-x // a) * a


def fit(c, cfg):
    """Settings derived from opt() setting c that run under the plan's cfg; [] if none can.

    Without reshaping: c as opt() gives it. With reshaping (chunk D = 64): M stays the plan's rows per head (the M
    tile is the largest one not above opt()'s that divides M per scale); the reshaped dimension (N for Z_TO, K for
    X_FROM) takes a tile of 64 or 32 (whichever are not above opt()'s tile, both kept as candidates: upstream
    simulates its candidates and keeps the faster) and is padded to a multiple of scale (x groups) x tile, i.e. by
    whole zero heads after the real ones, so the real heads keep the layout the attention kernel reads;
    resha_*_m = M x padded / 64."""
    M, N, K = cfg["m_size"], cfg["n_size"], cfg["k_size"]
    zto, xfrom = cfg.get("resha_z_to_enable", 0), cfg.get("resha_x_from_enable", 0)
    if not (zto or xfrom):
        return [dict(c, gap_x=None, gap_z=None, fix="", resha={})]
    sx, sy, g, st = c["scale_x"], c["scale_y"], c["groups"], c["strategy"]
    mt = _tile(M, sy, c["m_tile"])
    nper, kper = sx * (g if st == "splitN" else 1), (g if st == "splitK" else 1)
    nts = [t for t in (64, 32) if t <= c["n_tile"]] if zto else [c["n_tile"]]
    kts = [t for t in (64, 32) if t <= c["k_tile"]] if xfrom else [c["k_tile"]]
    out = []
    for nt in nts:
        for kt in kts:
            Np = _up(N, nper * nt) if zto else c["n_size"]
            Kp = _up(K, kper * kt) if xfrom else c["k_size"]
            if zto and Np % D or xfrom and Kp % D:
                continue
            f = dict(c, m_tile=mt, n_tile=nt, k_tile=kt, m_size=M, n_size=Np, k_size=Kp, gap_x=None, gap_z=None)
            fix = [f"{n} {c[n]}->{f[n]}" for n in ("m_tile", "n_tile", "k_tile") if f[n] != c[n]]
            if (c["m_size"], c["n_size"], c["k_size"]) != (M, Np, Kp):
                fix.append(f"size {c['m_size']}x{c['n_size']}x{c['k_size']}->{M}x{Np}x{Kp}")
            resha = {}
            if zto:
                resha["resha_z_to_m"] = M * Np // D
            if xfrom:
                resha["resha_x_from_m"] = M * Kp // D
            if st == "splitK" and xfrom:
                blk = Kp // g
                if blk % D:
                    continue
                f["gap_x"] = blk * M * 2
                fix.append(f"group gap X = {blk}x{M}x2 B (head-major)")
            if st == "splitN" and zto:
                blk = Np // g
                if blk % D:
                    continue
                f["gap_z"] = blk * M * 2
                fix.append(f"group gap Z = {blk}x{M}x2 B (head-major)")
            f.update(fix="; ".join(fix), resha=resha, m_iter=M // (sy * mt),
                     est_ns=int(model_ns(M, Np, Kp, mt, nt, kt, sx, sy, g, st)),
                     cand=f"{st}_x{sx}y{sy}g{g}_t{mt}x{nt}x{kt}")
            out.append(f)
    return out


def gemm_cands(run, logical):
    """(setting, skip reason) for upstream opt() on the plan shape; logical = the operator's unpadded (M, N, K)."""
    cfg = run["cfg"]
    out = []
    for c in opt(cfg["m_size"], cfg["n_size"], cfg["k_size"]):
        fs = fit(c, cfg)
        if not fs:
            out.append((dict(c, fix="no runnable tile under the reshaped layout"), "unfit"))
        for f in fs:
            if f["scale_x"] != f["scale_y"] and f["m_iter"] > 1:
                out.append((f, "skipped_m_iter"))      # SummaGEMM.h:165, as gemm_upstream.py
            else:
                out.append((f, None))
    fb = fallback(logical, [c for c, skip in out if skip is None])
    if fb:
        out.append((fb, None))
    return out


EARLIER = ("gemm_upstream_h1.csv", "gemm_smallm_h1.csv")   # same simulator, default placement


def fallback(logical, cands):
    """Recorded fix: when every runnable opt() setting already failed the output check in the earlier sweeps
    (default placement), add the fastest setting that passed there, so the key has a correct run to select."""
    shape = "x".join(map(str, logical))
    rows = [r for f in EARLIER if (HERE / "results" / f).exists()
            for r in csv.DictReader(open(HERE / "results" / f)) if r["shape"] == shape]
    seen = {r["cand"]: r["status"] for r in rows}
    if not cands or any(seen.get(c["cand"]) != "wrong" for c in cands):
        return None
    ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: float(r["time_ns"]))
    if not ok:
        return None
    r = ok[0]
    c = {k: int(r[k]) for k in ("m_size", "n_size", "k_size", "m_tile", "n_tile", "k_tile",
                                "scale_x", "scale_y", "groups")}
    c.update(strategy=r["strategy"], cand=r["cand"], est_ns=int(float(r["est_ns"])),
             m_iter=c["m_size"] // (c["scale_y"] * c["m_tile"]), gap_x=None, gap_z=None, resha={},
             fix=f"upstream opt() setting(s) {', '.join(x['cand'] for x in cands)} fail the output check "
                 f"(also in {', '.join(EARLIER)}); fastest passing earlier setting added")
    return c


def write_gemm_cfg(path, run, c, logical):
    M, N, K = logical
    S.write_cfg(path, M, N, K, c)
    s = path.read_text()
    cfg = run["cfg"]
    sets = {k: v for k, v in cfg.items() if k.startswith("resha_")}
    sets.update(c["resha"])
    if c.get("gap_x") is not None:
        sets["summa_group_gap_x"] = c["gap_x"]
    if c.get("gap_z") is not None:
        sets["summa_group_gap_z"] = c["gap_z"]
    for k, v in sets.items():
        s, n = re.subn(rf"(self\.{k}\s*=\s*)\S+", lambda m: m.group(1) + str(v), s)
        assert n == 1, k
    path.write_text(s)
    _place(path, run)


def _place(path, run):
    path.write_text(path.read_text() + "\n# deployment plan (workload_modeling.deploy_plan): HBM edge of every array\n"
                    f"HBM_PLACE = {run['place']!r}\n")


def other_cands(run):
    """Attention: the attn_auto opt() setting the plan carries (upstream_rules.attn_opt, fixes recorded); row kernels
    and RoPE: the kernel at its template defaults with the plan's shape and operator (stage1-spec §4.3 item 3)."""
    if run["kernel"] == "attn":
        c = run["cfg"]
        name = f"attn_s{c['sx']}x{c['sy']}_b{c['Br']}x{c['Bc']}"
        return [(dict(cand=name, fix=run["fix"]), None)]
    return [(dict(cand="default", fix=""), None)]


def write_other_cfg(path, key, run):
    if run["kernel"] == "attn":
        name = key.split(".", 1)[1].split("_p")[0].split("_", 1)[1]      # flatattention_<hq>x<q>x<kv>x<d>
        A.write_cfg(path, name, run["cfg"])
    else:
        RK.write_cfg(path, run["kernel"], run["cfg"])
    _place(path, run)


def load_csv():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def logical_shape(plan, key):
    stage, base = key.split(".", 1)
    d = next(d for d in plan.deploys if d.stage == stage and f"{stage}.{d.key}" == key)
    r = next(r for r in plan.rows if r.stage == stage and r.op == d.op)
    return r.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--kernel", action="append", choices=["gemm", "attn", "elt", "acti", "lnorm", "norm", "rope"],
                    help="default: every kernel")
    ap.add_argument("--key", action="append")
    a = ap.parse_args()
    plan = Plan()
    runs = {k: v for k, v in plan.runs().items()
            if (not a.kernel or v["kernel"] in a.kernel) and (not a.key or k in a.key)}
    rows = load_csv()
    if a.cmd == "pick":
        # selected = fastest run that passed the output check; composed = sum of selected time x count (the
        # operator's multiplicity in the op graph), kernels back to back, per stage
        tot, missing = {}, []
        with PICK.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "kernel", "count", "cand", "time_ns", "total_ns", "ops", "fix", "run_dir"])
            for k, run in runs.items():
                ok = [r for r in rows if r["key"] == k and r["status"] == "ok"]
                b = min(ok, key=lambda r: int(r["time_ns"])) if ok else None
                if not b:
                    missing.append(k)
                    print(f"{k:44} no checked run")
                    continue
                t = int(b["time_ns"])
                st = k.split(".", 1)[0]
                tot[st] = tot.get(st, 0) + t * run["count"]
                w.writerow([k, run["kernel"], run["count"], b["cand"], t, t * run["count"], b["ops"], b["fix"],
                            b["run_dir"]])
                print(f"{k:44} x{run['count']:<4} {t:>9} ns  {b['cand']}")
        for st, v in tot.items():
            print(f"{st:10} {v / 1e6:9.3f} ms")
        print(f"{'total':10} {sum(tot.values()) / 1e6:9.3f} ms" + (f"  (missing: {missing})" if missing else ""))
        return
    done = {(r["key"], r["cand"]) for r in rows}
    for k, run in runs.items():
        logical = logical_shape(plan, k)
        gemm = run["kernel"] == "gemm"
        for c, skip in (gemm_cands(run, logical) if gemm else other_cands(run)):
            if a.cmd == "plan":
                print(f"{k:44} {c['cand']:<30} est {c.get('est_ns') or 0:>7}  {skip or ''}  {c.get('fix', '')}"
                      f"{'  done' if (k, c['cand']) in done else ''}")
                continue
            if (k, c["cand"]) in done:
                continue
            row = dict(c, key=k, kernel=run["kernel"], ops=",".join(run["ops"]), count=run["count"])
            if skip:
                row.update(status=skip)
            else:
                out = RUNS / k / c["cand"]
                out.mkdir(parents=True, exist_ok=True)
                cfg = out / f"{run['kernel']}_cfg.py"
                if gemm:
                    write_gemm_cfg(cfg, run, c, logical)
                else:
                    write_other_cfg(cfg, k, run)
                print(f"[{k}] {c['cand']} ...", flush=True)
                p = subprocess.run([str(HERE / "run_kernel.sh"), run["kernel"], str(cfg), str(out)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (out / "run_kernel.out").write_text(f"exit {p.returncode}\n{p.stdout}")
                rc = p.returncode
                t, bad, mu = parse_run(out)
                row.update(time_ns=t, bad=bad, max_ulp=mu, run_dir=str(out),
                           status="ok" if (rc == 0 and t is not None and bad == 0) else
                           ("timeout_or_fail" if t is None else "wrong"))
                print(f"    {t} ns, bad {bad}, {row['status']}", flush=True)
            new = not CSV.exists()
            with CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
