#!/usr/bin/env python3
"""Transposed-GEMM test on H1: run Z^T = W^T X^T (shape N x M x K) instead of Z = X W (M x N x K).

    gemm_transpose.py plan [--shape MxNxK] [--top N]   candidates for the transposed shape
    gemm_transpose.py run  [--shape MxNxK] [--top N]   run them one at a time (resumable)
    gemm_transpose.py pick [--shape MxNxK]             fastest checked run, original vs transposed

Why: GVSoC's RedMulE times each call in whole 32 x 64 output blocks (kernel explorer, f_pad block fill). With
N = 768 on an 8 x 8 SUMMA group each cluster gets n_tile = 96, which fills 75 % of the blocks. Transposed, the
output columns are M (1024 or 3072), so n_tile can be a multiple of 64. SummaGEMM times the same whatever the
data means, so the transposed kernel is simply the GEMM of shape N x M x K; W^T is free (weights are laid out
offline), but X^T and Z^T are activation layouts, so the producer and consumer ops would have to agree (not
modelled here).

Candidates, configs and parsing are gemm_sweep's (same tile rules, same output check against the golden).
Raw results: results/gemm_transpose_h1.csv, kept apart from the stage-1 sweep. Run folders:
$SHVLA_DATA/stage1-h1/gemm_transpose/<shape>/<cand>/.
"""
import argparse
import csv
import subprocess
from pathlib import Path

import gemm_sweep as G

HERE = Path(__file__).resolve().parent
CSV = HERE / "results/gemm_transpose_h1.csv"
RUNS = G.RUNS.parent / "gemm_transpose"
DEFAULT = ["3072x768x768"]                    # original shapes; the run is of the transposed shape N x M x K
FIELDS = ["orig_shape", "shape"] + [f for f in G.FIELDS if f != "shape"]


def transposed(sh):
    M, N, K = sh
    return (N, M, K)


def rows():
    return list(csv.DictReader(CSV.open())) if CSV.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--shape", action="append")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    shapes = [tuple(map(int, s.split("x"))) for s in (a.shape or DEFAULT)]

    if a.cmd == "pick":
        orig = list(csv.DictReader(G.CSV.open()))
        for sh in shapes:
            o = [r for r in orig if r["shape"] == G.shape_key(sh) and r["status"] == "ok"]
            t = [r for r in rows() if r["orig_shape"] == G.shape_key(sh) and r["status"] == "ok"]
            bo = min(o, key=lambda r: int(r["time_ns"])) if o else None
            bt = min(t, key=lambda r: int(r["time_ns"])) if t else None
            print(f"{G.shape_key(sh)}: original {bo['cand'] if bo else '-'} {bo['time_ns'] if bo else '-'} ns; "
                  f"transposed {G.shape_key(transposed(sh))} {bt['cand'] if bt else '-'} "
                  f"{bt['time_ns'] if bt else '-'} ns ({len(t)} ok)")
        return

    done = {(r["shape"], r["cand"]) for r in rows()}
    for sh in shapes:
        tsh = transposed(sh)
        for c in G.candidates(*tsh, a.top):
            key = (G.shape_key(tsh), c["cand"])
            if a.cmd == "plan":
                print(f"{G.shape_key(tsh):>14} {c['cand']:<32} pad {c['m_size']}x{c['n_size']}x{c['k_size']} "
                      f"work {c['work_ratio']:<5}{'  done' if key in done else ''}")
                continue
            if key in done:
                continue
            out = RUNS / G.shape_key(tsh) / c["cand"]
            out.mkdir(parents=True, exist_ok=True)
            cfg = out / "gemm_cfg.py"
            G.write_cfg(cfg, *tsh, c)
            print(f"[{G.shape_key(tsh)}] {c['cand']} ...", flush=True)
            rc = subprocess.run(["bash", str(HERE / "run_kernel.sh"), "gemm", str(cfg), str(out)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
            t, bad, mu = G.parse_run(out)
            status = "ok" if (rc == 0 and t is not None and bad == 0) else \
                     ("timeout_or_fail" if t is None else "wrong")
            new = not CSV.exists()
            with CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(dict(c, orig_shape=G.shape_key(sh), shape=G.shape_key(tsh), time_ns=t, bad=bad,
                                max_ulp=mu, status=status, run_dir=str(out)))
            print(f"    {t} ns, bad {bad}, {status}", flush=True)


if __name__ == "__main__":
    main()
