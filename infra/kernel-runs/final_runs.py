#!/usr/bin/env python3
"""Recorded runs: the picked setting of every GEMM shape and every row kernel, re-run once on the kernel source of the
run (stage-1 spec §4.3 item 11; first done on the frozen kernel build, $SHVLA_DATA/stage1-h1/kernel_build.diff).
Folders: $SHVLA_DATA/stage1-h1/final/<kernel>/<name>/; results: results/final_h1.csv, which to_constants.py turns
into the h1_kernels section of constants.yaml (the W1 preset: the fastest checked run found so far per kernel).

    final_runs.py            re-run every pick that changed
    final_runs.py --plan     print the picks only

Picks: the fastest run that passed its output check, over every result set:
  GEMM   gemm_sweep.py, gemm_smallm_sweep.py, and gemm_transpose.py (Z^T = W^T X^T run as N x M x K; setting
         "T:<candidate>", config = the transposed shape)
  rows   row_kernels.py (stage-1 schedule), and for Eltwise eltwise_pipeline.py (setting "<config> pipeline=1": the
         kernel's ELT_PIPELINE schedule; only runs whose config sets pipeline, i.e. kernel commits with the switch)
A pick that changes replaces its row; the old run folder is kept as <name>.superseded. The re-run uses the kernel
source run_kernel.sh builds (KREPO, default the submodule third_party/softhier-kernels); "build" is the kernel commit
it records (kernel_repo_commit.txt), except for the stage-1 rows kept from the frozen build (its build hash) and the
gemm_smallm_sweep.py picks of de53537 (aa57690 + the optional GEMM_CHECK_ABS_TOL check, BUILD2)."""
import csv, os, subprocess, sys
from pathlib import Path
import gemm_sweep as G, gemm_smallm_sweep as S, gemm_transpose as T, gemm_upstream as U, row_kernels as R
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1"
CSV = HERE / "results/final_h1.csv"
ELT_PIPE = HERE / "results/eltwise_pipeline_h1.csv"
BUILD = (DATA / "kernel_build.sha").read_text().strip()
BUILD2 = "de53537"


def _rows(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def jobs():
    """(kernel, name, setting, config, time in the sweep) of every pick."""
    rows = [dict(r, _set=r["cand"]) for r in _rows(G.CSV)]
    rows += [dict(r, _set=r["cand"]) for r in _rows(S.CSV)]
    rows += [dict(r, _set=r["cand"]) for r in _rows(U.CSV) if r["status"] != "skipped_m_iter"]
    rows += [dict(r, shape=r["orig_shape"], _set="T:" + r["cand"]) for r in _rows(T.CSV)]
    for shape in sorted({r["shape"] for r in rows}, key=lambda s: [int(x) for x in s.split("x")]):
        b = min((r for r in rows if r["shape"] == shape and r["status"] == "ok"), key=lambda r: int(r["time_ns"]))
        yield "gemm", f"gemm_{shape}", b["_set"], Path(b["run_dir"]) / "kernel_cfg.py", b["time_ns"]
    pipe = {}
    for r in _rows(ELT_PIPE):
        cfg = Path(r["run_dir"]) / "kernel_cfg.py"
        if r["status"] == "ok" and cfg.exists() and "self.pipeline" in cfg.read_text():
            if r["name"] not in pipe or int(r["time_ns"]) < int(pipe[r["name"]]["time_ns"]):
                pipe[r["name"]] = r
    for r in _rows(R.CSV):
        if r["status"] != "ok":
            continue
        p = pipe.get(r["name"])
        if p and int(p["time_ns"]) < int(r["time_ns"]):
            yield r["kernel"], r["name"], r["config"] + " pipeline=1", Path(p["run_dir"]) / "kernel_cfg.py", p["time_ns"]
        else:
            yield r["kernel"], r["name"], r["config"], Path(r["run_dir"]) / "kernel_cfg.py", r["time_ns"]


HEADER = ["name", "kernel", "setting", "time_ns", "sweep_ns", "bad", "max_ulp", "status", "build", "dir"]


if __name__ == "__main__":
    have = {r["name"]: r for r in csv.DictReader(open(CSV))} if CSV.exists() else {}
    for kern, name, setting, cfg, t_sweep in jobs():
        if name in have and have[name]["setting"] == setting:
            continue
        if "--plan" in sys.argv:
            print(f"{name}: {have.get(name, {}).get('setting', '-')} -> {setting} ({t_sweep} ns in the sweep)")
            continue
        out = DATA / "final" / kern / name
        if out.exists():
            sup = out.with_name(name + ".superseded")
            if sup.exists():
                sup = out.with_name(f"{name}.superseded.{len(list(out.parent.glob(name + '.superseded*')))}")
            out.rename(sup)
        print(f"[{name}] {setting} ...", flush=True)
        rc = subprocess.run(["bash", str(HERE / "run_kernel.sh"), kern, str(cfg), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        t, bad, mu = R.parse_run(out)
        status = "ok" if (rc == 0 and t is not None and bad == 0) else "failed"
        kc = out / "kernel_repo_commit.txt"
        build = kc.read_text().split()[0][:7] if kc.exists() and kc.read_text().strip() else "?"
        have[name] = dict(zip(HEADER, [name, kern, setting, t, t_sweep, bad, mu, status, build, out]))
        with CSV.open("w", newline="") as f:          # rewrite: a changed pick replaces its row in place
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(have.values())
        print(f"    {t} ns (sweep {t_sweep}), bad {bad}, {status}", flush=True)
