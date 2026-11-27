#!/usr/bin/env python3
"""Perfetto traces (run_kernel.sh --trace) for the picked settings, one at a time (stage-1 spec §4.3 item 8).
Before GVSoC patch 0013, 4x4 and 8x8 SUMMA groups were too slow to trace (stage-1 spec §4.3 item 9); failed rows
are retried, ok rows are skipped.
Also the selected run of every plan key (results/plan_pick*_h1.csv, plan_sweep_pick*_h1.csv), named
"plan:<run folder under stage1-h1>"; --plan traces only those.
Trace folders: $SHVLA_DATA/stage1-h1/traces/<kernel>/<name>/; status in results/traces_h1.csv."""
import csv, os, subprocess, sys
from pathlib import Path
import gemm_sweep as G, gemm_smallm_sweep as SM, row_kernels as R
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1/traces"
CSV = HERE / "results/traces_h1.csv"
jobs = []
# GEMM picks: fastest checked run per shape over both sweeps (gemm_sweep and the small-M gemm_smallm_sweep), the rule
# final_runs / constants and the kernel explorer use
rows = list(csv.DictReader(open(G.CSV))) + list(csv.DictReader(open(SM.CSV)))
for shape in sorted({r["shape"] for r in rows}):
    ok = [r for r in rows if r["shape"] == shape and r["status"] == "ok"]
    b = min(ok, key=lambda r: int(r["time_ns"]))
    jobs.append(("gemm", f"{shape}_{b['cand']}", Path(b["run_dir"]) / "kernel_cfg.py", int(b["time_ns"])))
# FlatAttention picks: fastest checked run per shape of attn_sweep (the rule the explorer and constants use)
arows = list(csv.DictReader(open(HERE / "results/attn_sweep_h1.csv")))
for shape in sorted({r["shape"] for r in arows}):
    ok = [r for r in arows if r["shape"] == shape and r["status"] == "ok"]
    b = min(ok, key=lambda r: int(r["time_ns"]))
    jobs.append(("attn", f"{shape}_{b['cand']}", Path(b["run_dir"]) / "kernel_cfg.py", int(b["time_ns"])))
for r in csv.DictReader(open(R.CSV)):
    if r["status"] == "ok":
        jobs.append((r["kernel"], r["name"], Path(r["run_dir"]) / "kernel_cfg.py", int(r["time_ns"])))
# plan picks: the selected run of every key of every plan (upstream rule and plan-constrained sweep, W1 and W1-b),
# one trace per run folder; name "plan:<run folder under stage1-h1>", which the kernel explorer matches by run folder
PLAN_PICKS = ("plan_pick_h1.csv", "plan_pick_w1b_h1.csv", "plan_sweep_pick_w1_h1.csv", "plan_sweep_pick_w1b_h1.csv")
ROOT = OUT.parent
plan_jobs = {}
for fn in PLAN_PICKS:
    f = HERE / "results" / fn
    if not f.exists():
        continue
    for r in csv.DictReader(open(f)):
        rd = Path(r["run_dir"])
        name = "plan:" + str(rd.relative_to(ROOT))
        plan_jobs.setdefault(name, (r["kernel"], name, rd / "kernel_cfg.py", int(r["time_ns"])))
only_plan = "--plan" in sys.argv[1:]
sys.argv = [a for a in sys.argv if a != "--plan"]
jobs = list(plan_jobs.values()) if only_plan else jobs + list(plan_jobs.values())
done = {r["name"] for r in csv.DictReader(open(CSV)) if r["status"] == "ok"} if CSV.exists() else set()
for kern, name, cfg, t_meas in jobs:
    if name in done or (sys.argv[1:] and name not in sys.argv[1:]):
        continue
    out = OUT / kern / name.replace("plan:", "plan/", 1)
    print(f"[{name}] ...", flush=True)
    # plan picks were measured by their own runs: the traced run alone (it times and checks the same binary)
    flag = "--trace-only" if name.startswith("plan:") else "--trace"
    rc = subprocess.run(["bash", str(HERE / "run_kernel.sh"), kern, str(cfg), str(out), flag],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    pf = out / "perfetto.json"
    t = G.parse_run(out)[0] if hasattr(G, "parse_run") else None
    status = "ok" if (rc == 0 and pf.exists() and pf.stat().st_size > 0) else "failed_or_timeout"
    new = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "kernel", "measured_ns", "status", "perfetto_mb", "dir"])
        w.writerow([name, kern, t_meas, status, round(pf.stat().st_size / 2**20, 1) if pf.exists() else "", out])
    print(f"    {status}", flush=True)
