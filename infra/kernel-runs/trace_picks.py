#!/usr/bin/env python3
"""Perfetto traces (run_kernel.sh --trace) for the picked settings, one at a time (stage-1 spec §4.3 item 8).
GEMM picks with scale 8 are skipped: with a trace flag an 8x8 SUMMA group livelocks (open simulator issue).
Trace folders: $SHVLA_DATA/stage1-h1/traces/<kernel>/<name>/; status in results/traces_h1.csv."""
import csv, os, subprocess, sys
from pathlib import Path
import gemm_sweep as G, row_kernels as R
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1/traces"
CSV = HERE / "results/traces_h1.csv"
jobs = []
rows = list(csv.DictReader(open(G.CSV)))
for shape in {r["shape"] for r in rows}:
    ok = [r for r in rows if r["shape"] == shape and r["status"] == "ok"]
    b = min(ok, key=lambda r: int(r["time_ns"]))
    if "_s8" in b["cand"]:
        continue
    jobs.append(("gemm", f"{shape}_{b['cand']}", Path(b["run_dir"]) / "kernel_cfg.py", int(b["time_ns"])))
for r in csv.DictReader(open(R.CSV)):
    if r["status"] == "ok":
        jobs.append((r["kernel"], r["name"], Path(r["run_dir"]) / "kernel_cfg.py", int(r["time_ns"])))
done = {r["name"] for r in csv.DictReader(open(CSV))} if CSV.exists() else set()
for kern, name, cfg, t_meas in jobs:
    if name in done or (sys.argv[1:] and name not in sys.argv[1:]):
        continue
    out = OUT / kern / name
    print(f"[{name}] ...", flush=True)
    rc = subprocess.run([str(HERE / "run_kernel.sh"), kern, str(cfg), str(out), "--trace"],
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
