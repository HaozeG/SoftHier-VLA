#!/usr/bin/env python3
"""Recorded runs (stage-1 spec §4.3 item 11): the picked setting of every GEMM shape and every row kernel, re-run
once on the frozen kernel build ($SHVLA_DATA/stage1-h1/kernel_build.diff, sha in kernel_build.sha) and the final
simulator stamp. Folders: $SHVLA_DATA/stage1-h1/final/<kernel>/<name>/; results: results/final_h1.csv."""
import csv, os, subprocess
from pathlib import Path
import gemm_sweep as G, row_kernels as R
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1"
CSV = HERE / "results/final_h1.csv"
BUILD = (DATA / "kernel_build.sha").read_text().strip()


def jobs():
    rows = list(csv.DictReader(open(G.CSV)))
    for shape in sorted({r["shape"] for r in rows}, key=lambda s: [int(x) for x in s.split("x")]):
        b = min((r for r in rows if r["shape"] == shape and r["status"] == "ok"), key=lambda r: int(r["time_ns"]))
        yield "gemm", f"gemm_{shape}", b["cand"], Path(b["run_dir"]) / "kernel_cfg.py", b["time_ns"]
    for r in csv.DictReader(open(R.CSV)):
        if r["status"] == "ok":
            yield r["kernel"], r["name"], r["config"], Path(r["run_dir"]) / "kernel_cfg.py", r["time_ns"]


if __name__ == "__main__":
    done = {r["name"] for r in csv.DictReader(open(CSV))} if CSV.exists() else set()
    for kern, name, setting, cfg, t_sweep in jobs():
        if name in done:
            continue
        out = DATA / "final" / kern / name
        print(f"[{name}] ...", flush=True)
        rc = subprocess.run([str(HERE / "run_kernel.sh"), kern, str(cfg), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        t, bad, mu = R.parse_run(out)
        status = "ok" if (rc == 0 and t is not None and bad == 0) else "failed"
        new = not CSV.exists()
        with CSV.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["name", "kernel", "setting", "time_ns", "sweep_ns", "bad", "max_ulp", "status", "build", "dir"])
            w.writerow([name, kern, setting, t, t_sweep, bad, mu, status, BUILD, out])
        print(f"    {t} ns (sweep {t_sweep}), bad {bad}, {status}", flush=True)
