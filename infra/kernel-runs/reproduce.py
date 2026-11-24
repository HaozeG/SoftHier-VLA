#!/usr/bin/env python3
"""Reproduce the stage-1 kernel times on H1 (the `h1_kernels` section of infra/workload-modeling/constants.yaml).

    reproduce.py list                          recorded runs, fastest first
    reproduce.py run [--only NAME ...] [--kernel K ...] [--max-ns N]
                                               re-run them; exit 1 unless every time is identical and every check passes
    reproduce.py export                        (owner) copy the recorded kernel configs into configs/h1/ + manifest

One recorded run per distinct setting: GEMM and row kernels from results/final_h1.csv, FlatAttention from the fastest
checked row per shape of results/attn_sweep_h1.csv (the same rule as to_constants.py). The configs are committed under
configs/h1/, so `run` needs only the setup in README.md ("Reproduce the stage-1 numbers"): the patched simulator built
for arch/arch_h1.py and the kernel submodule third_party/softhier-kernels. Run folders go to
$SHVLA_DATA/stage1-h1/repro/<name>/ (never over the recorded final/ folders).

GVSoC simulation is deterministic, so the comparison is exact (ns). Recorded values were taken on kernel build aa57690
(de53537 for the M = 50 expert GEMM picks; de53537 = aa57690 + an opt-in GEMM check, same binary for configs that do
not set check_abs_tol) and GVSoC tree 058a39b (patches 0001-0012)."""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG_DIR = HERE / "configs/h1"
MANIFEST = CFG_DIR / "recorded.csv"
CONST = REPO / "infra/workload-modeling/constants.yaml"
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1"
# final_h1.csv "build" of the frozen kernel build (hash of its worktree diff); that diff plus the then-untracked
# implementation/scripts/utils/hbm_layout.py is commit aa57690 (checked 2026-11-24).
FROZEN_BUILD_HASH = "9482644ba4551a9b"
FIELDS = ["name", "kernel", "config", "recorded_ns", "build", "constants_keys"]


def rd(p):
    return list(csv.DictReader(open(HERE / "results" / p)))


def picks():
    """constants key -> (run name, run_kernel.sh kernel, recorded run folder, ns, build), as to_constants.py picks."""
    import yaml  # export only; list and run need the standard library alone
    h1 = yaml.safe_load(open(CONST))["h1_kernels"]
    final = {r["name"]: r for r in rd("final_h1.csv")}
    asweep = rd("attn_sweep_h1.csv")
    out = {}
    for stage, entries in h1.items():
        for key, e in entries.items():
            kern, shape = key.rsplit("_", 1)
            if kern == "flatattention":
                b = min((r for r in asweep if r["shape"] == shape and r["status"] == "ok"), key=lambda r: int(r["time_ns"]))
                name, k, d, t, build = f"attn_{shape}_{b['cand']}", "attn", b["run_dir"], int(b["time_ns"]), "de53537"
            else:
                f = final[f"gemm_{shape}" if kern == "gemm" else key]
                name, k, d, t, build = f["name"], f["kernel"], f["dir"], int(f["time_ns"]), f["build"]
                if build == FROZEN_BUILD_HASH:
                    build = "aa57690"
            # the manifest must hold exactly the numbers the analysis pages render
            assert t == e["value"], (stage, key, t, e["value"])
            out[f"{stage}.{key}"] = (name, k, d, t, build)
    return out


def export():
    p = picks()
    runs = {}
    for ckey, (name, k, d, t, build) in p.items():
        runs.setdefault(name, [k, d, t, build, []])[4].append(ckey)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (k, d, t, build, keys) in sorted(runs.items(), key=lambda kv: kv[1][2]):
        shutil.copy(Path(d) / "kernel_cfg.py", CFG_DIR / f"{name}.py")
        rows.append(dict(zip(FIELDS, [name, k, f"{name}.py", t, build, " ".join(sorted(keys))])))
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(p)} constants entries -> {len(rows)} distinct runs -> {MANIFEST.relative_to(REPO)}")


def parse(out):
    log = (out / "run.log").read_text(errors="replace") if (out / "run.log").exists() else ""
    import re
    t = re.search(r"Execution period is (\d+) ns", log)
    bad = [int(m.group(1)) for m in re.finditer(r"\[check\] cluster \d+: .*?bad (\d+)", log)]
    return (int(t.group(1)) if t else None), (sum(bad) if bad else None)


def tree(out):
    import json
    try:
        return json.load(open(out / "sim_stamp.json"))["gvsoc_series"]["tree"][:7]
    except (OSError, KeyError, ValueError):
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["list", "run", "export"])
    ap.add_argument("--only", action="append", help="run name(s) from `list`")
    ap.add_argument("--kernel", action="append", help="gemm | attn | elt | acti | lnorm | norm")
    ap.add_argument("--max-ns", type=int, help="only runs recorded at or below this time (quick spot check)")
    a = ap.parse_args()
    if a.cmd == "export":
        return export()
    rows = list(csv.DictReader(open(MANIFEST)))
    rows = [r for r in rows if (not a.only or r["name"] in a.only) and (not a.kernel or r["kernel"] in a.kernel)
            and (not a.max_ns or int(r["recorded_ns"]) <= a.max_ns)]
    if a.cmd == "list":
        for r in rows:
            print(f"{r['name']:<48} {r['kernel']:<6} {int(r['recorded_ns']):>9} ns  build {r['build']:<16} {r['constants_keys']}")
        return
    fails = 0
    print(f"{'name':<48} {'recorded':>9} {'got':>9} {'delta':>7}  bad  tree")
    for r in rows:
        out = DATA / "repro" / r["name"]
        if out.exists():
            shutil.rmtree(out)
        rc = subprocess.run([str(HERE / "run_kernel.sh"), r["kernel"], str(CFG_DIR / r["config"]), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        t, bad = parse(out)
        rec = int(r["recorded_ns"])
        ok = rc.returncode == 0 and t == rec and bad == 0
        fails += not ok
        delta = "" if t is None else f"{t - rec:+d}"
        print(f"{r['name']:<48} {rec:>9} {t if t is not None else 'FAIL':>9} {delta:>7}  {bad if bad is not None else '?':<4} "
              f"{tree(out)}{'' if ok else '  <-- ' + (rc.stderr.strip().splitlines() or ['differs'])[-1]}", flush=True)
    print(f"{len(rows) - fails}/{len(rows)} identical")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
