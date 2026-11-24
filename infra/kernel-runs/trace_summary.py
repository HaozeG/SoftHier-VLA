#!/usr/bin/env python3
"""Per-cluster unit activity in the kernel window of each Perfetto trace in results/traces_h1.csv.

    trace_summary.py            -> results/trace_summary_h1.json

The traces (run_kernel.sh --trace) span the whole simulation: preload, the kernel, and the output check after it.
The kernel window is taken as [t0, t0 + T]: t0 = the first redmule, idma or Spatz event on any cluster, T = the
"Execution period" of that trace run (run.log). Both are read from the run (measured); busy fractions computed
from them are inference. Per cluster and unit: busy ns inside the window (union of the unit's events; for Spatz the
busiest of the pe<n>/spatz tracks), and which clusters had any unit event at all (the active set the mapping
diagrams are checked against; only events inside the window count, since the output check after the
kernel moves data on other clusters). Traces live outside git ($SHVLA_DATA/stage1-h1/traces/); this summary is committed
so the report builds without them.
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES_CSV = HERE / "results/traces_h1.csv"
OUT = HERE / "results/trace_summary_h1.json"
UNITS = ("redmule", "idma", "spatz")
MESH = 64


def unit_of(tid):
    if tid in ("redmule", "idma"):
        return tid
    if tid.endswith("/spatz"):
        return "spatz"
    return None


def union_ns(iv, lo, hi):
    """Length in ns of the union of [s, e) intervals (µs) clipped to [lo, hi)."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot * 1000.0


def summarize(d: Path):
    log = (d / "run.log").read_text(errors="replace")
    m = re.search(r"Execution period is (\d+) ns", log)
    period = int(m.group(1))
    ev = json.load(open(d / "perfetto.json"))["traceEvents"]
    tracks = {}                       # (cluster, unit, track) -> intervals in µs
    for e in ev:
        if e.get("ph") != "X":
            continue
        pid, u = str(e.get("pid", "")), unit_of(str(e.get("tid", "")))
        if u is None or not pid.startswith("cluster_"):
            continue
        tracks.setdefault((int(pid[8:]), u, e["tid"]), []).append((e["ts"], e["ts"] + e["dur"]))
    t0 = min(s for iv in tracks.values() for s, _ in iv)
    t1 = t0 + period / 1000.0
    busy = {u: [0.0] * MESH for u in UNITS}
    for (c, u, _), iv in tracks.items():
        busy[u][c] = max(busy[u][c], union_ns(iv, t0, t1))
    active = {u: [c for c in range(MESH) if busy[u][c] > 0] for u in UNITS}
    return dict(period_ns=period, t0_us=round(t0, 3), active_clusters=sorted({c for u in UNITS for c in active[u]}),
                active_by_unit=active,
                busy_frac={u: [round(b / period, 4) for b in busy[u]] for u in UNITS})


def main():
    out = {}
    for r in csv.DictReader(open(TRACES_CSV)):
        d = Path(r["dir"])
        entry = dict(kernel=r["kernel"], status=r["status"], measured_ns=int(r["measured_ns"]),
                     dir=r["dir"], perfetto_mb=r["perfetto_mb"])
        if r["status"] == "ok" and (d / "perfetto.json").exists():
            print(f"[{r['name']}] ...", flush=True)
            entry.update(summarize(d))
        out[r["name"]] = entry
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(HERE)} ({len(out)} traces)")


if __name__ == "__main__":
    main()
