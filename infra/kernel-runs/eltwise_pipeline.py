#!/usr/bin/env python3
"""Re-run the 17 recorded Eltwise settings (configs/h1/eltwise_*.py) with the kernel source in KREPO (default the
submodule) and compare with the recorded stage-1 times (configs/h1/recorded.csv). Written for the pipelined Eltwise
(row j-1 stored and row j+1 loaded while Spatz computes row j); the stage-1 baseline files are not touched.

    eltwise_pipeline.py [NAME ...]     run (all, or the named configs), append to results/eltwise_pipeline_h1.csv

Run folders: $SHVLA_DATA/stage1-h1/eltwise_pipeline/<kernel commit>/<name>/."""
import csv, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KREPO = Path(os.environ.get("KREPO", HERE / "../../third_party/softhier-kernels")).resolve()
OUT = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla")) / "stage1-h1/eltwise_pipeline"
CSV = HERE / "results/eltwise_pipeline_h1.csv"
FIELDS = ["name", "kernel_commit", "recorded_ns", "time_ns", "speedup", "check", "status", "run_dir"]

commit = subprocess.run(["git", "-C", str(KREPO), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
recorded = {r["name"]: int(r["recorded_ns"]) for r in csv.DictReader(open(HERE / "configs/h1/recorded.csv"))}
names = sorted(p.stem for p in (HERE / "configs/h1").glob("eltwise_*.py"))


def parse(out):
    """(time_ns, clusters checked, bad, max_ulp) from run.log; the per-cluster [check] lines of fp16_check_rows."""
    log = (out / "run.log").read_text() if (out / "run.log").exists() else ""
    m = re.search(r"Execution period is (\d+) ns", log)
    chk = re.findall(r"\[check\] cluster \d+: rows \d+, bad (\d+) \(nan \d+\), max_ulp (\d+)", log)
    return (int(m.group(1)) if m else None, len(chk), sum(int(b) for b, _ in chk),
            max((int(u) for _, u in chk), default=None))


def pipelined_cfg(name, d):
    """configs/h1/<name>.py with pipeline = 1 (the ELT_PIPELINE schedule; the kernel builds the stage-1 schedule
    without it). Kernel commits before 2026-11-24's ELT_PIPELINE switch (e7829dc, 46589cd) pipelined always."""
    src = (HERE / f"configs/h1/{name}.py").read_text()
    anchor = "        self.dtype"
    assert anchor in src, name
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.py"
    out.write_text(src.replace(anchor, "        self.pipeline                = 1\n" + anchor, 1))
    return out


old = {(r["name"], r["kernel_commit"]): r for r in csv.DictReader(CSV.open())} if CSV.exists() else {}
for name in names:
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    out = OUT / commit / name
    t, ncl, bad, mu = parse(out)
    if t is None:                                   # not run yet with this kernel commit
        print(f"[{name}] ...", flush=True)
        cfg = pipelined_cfg(name, OUT / commit / "cfg")
        subprocess.run(["bash", str(HERE / "run_kernel.sh"), "elt", str(cfg), str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t, ncl, bad, mu = parse(out)
    ok = t is not None and ncl > 0 and bad == 0
    row = {"name": name, "kernel_commit": commit, "recorded_ns": recorded.get(name, ""), "time_ns": t or "",
           "speedup": round(recorded[name] / t, 3) if (t and name in recorded) else "",
           "check": f"{ncl} clusters, bad {bad}, max_ulp {mu}", "status": "ok" if ok else "failed", "run_dir": out}
    old.pop((name, commit), None)
    old[(name, commit)] = row
    print(f"    {name}: {row['status']} {t} ns (recorded {row['recorded_ns']}, x{row['speedup']}) {row['check']}",
          flush=True)
with CSV.open("w", newline="") as f:                # one row per (name, kernel commit), rewritten each time
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    w.writerows(old.values())                       # in run order: the last row of a name is the newest
