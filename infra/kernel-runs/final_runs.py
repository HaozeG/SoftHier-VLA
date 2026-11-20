#!/usr/bin/env python3
"""Recorded runs (stage-1 spec §4.3 item 11): the picked setting of every GEMM shape and every row kernel, re-run
once on the frozen kernel build ($SHVLA_DATA/stage1-h1/kernel_build.diff, sha in kernel_build.sha) and the final
simulator stamp. Folders: $SHVLA_DATA/stage1-h1/final/<kernel>/<name>/; results: results/final_h1.csv.

GEMM picks: the fastest checked setting of a shape over gemm_sweep.py and gemm_smallm_sweep.py. A pick that changes
replaces its row; the old run folder is kept as <name>.superseded. Picks from gemm_smallm_sweep.py run on kernel
commit de53537 (aa57690 + the optional GEMM_CHECK_ABS_TOL check, BUILD2): same binary as the frozen build for
configs without check_abs_tol (re-run of 50x4096x720 splitN_s2g4: 32353 ns, unchanged); splitK configs set it."""
import csv, os, subprocess
from pathlib import Path
import gemm_sweep as G, gemm_smallm_sweep as S, row_kernels as R
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1"
CSV = HERE / "results/final_h1.csv"
BUILD = (DATA / "kernel_build.sha").read_text().strip()
BUILD2 = "de53537"


def jobs():
    rows = [dict(r, _build=BUILD) for r in csv.DictReader(open(G.CSV))]
    if S.CSV.exists():
        rows += [dict(r, _build=BUILD2) for r in csv.DictReader(open(S.CSV))]
    for shape in sorted({r["shape"] for r in rows}, key=lambda s: [int(x) for x in s.split("x")]):
        b = min((r for r in rows if r["shape"] == shape and r["status"] == "ok"), key=lambda r: int(r["time_ns"]))
        yield "gemm", f"gemm_{shape}", b["cand"], Path(b["run_dir"]) / "kernel_cfg.py", b["time_ns"], b["_build"]
    for r in csv.DictReader(open(R.CSV)):
        if r["status"] == "ok":
            yield r["kernel"], r["name"], r["config"], Path(r["run_dir"]) / "kernel_cfg.py", r["time_ns"], BUILD


HEADER = ["name", "kernel", "setting", "time_ns", "sweep_ns", "bad", "max_ulp", "status", "build", "dir"]


if __name__ == "__main__":
    have = {r["name"]: r for r in csv.DictReader(open(CSV))} if CSV.exists() else {}
    for kern, name, setting, cfg, t_sweep, build in jobs():
        if name in have and have[name]["setting"] == setting:
            continue
        out = DATA / "final" / kern / name
        if out.exists():
            out.rename(out.with_name(name + ".superseded"))
        print(f"[{name}] {setting} ...", flush=True)
        rc = subprocess.run([str(HERE / "run_kernel.sh"), kern, str(cfg), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        t, bad, mu = R.parse_run(out)
        status = "ok" if (rc == 0 and t is not None and bad == 0) else "failed"
        have[name] = dict(zip(HEADER, [name, kern, setting, t, t_sweep, bad, mu, status, build, out]))
        with CSV.open("w", newline="") as f:          # rewrite: a changed pick replaces its row in place
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(have.values())
        print(f"    {t} ns (sweep {t_sweep}), bad {bad}, {status}", flush=True)
