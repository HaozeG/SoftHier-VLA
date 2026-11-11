#!/usr/bin/env python3
"""prof_wrap: run tests/gvsoc/run.py (from the softhier-mlir root) with extra, narrow gvsoc traces whose lines are
FILTERED / AGGREGATED IN THE PIPE, so nothing unfiltered or uncapped reaches the disk. Profiling-only: it changes
neither the simulator, the arch, nor the emitted program (the same idea as diag8/trace_wrap.py, which it generalises).

It monkeypatches `softhier_mlir.sim.gvsoc.subprocess.run` for the gapy command only:
  * appends `--trace=<regex>` for every entry of PROF_TRACES (comma list; the e2e traces redmule / idma /
    cluster_registers that `--e2e-trace` already adds stay);
  * gvsoc's stdout is read line by line. Kept lines go to the log file run.py passes (<app>/<phase>.log):
      - the program's own lines ([mark], dumps ...), re-assembled around trace lines that landed in the middle of them;
      - RedMulE / iDMA `Finished` lines and `Cluster Sync` lines (the three e2e traces);
    everything else is dropped (start-up chatter, other trace lines). Trace bytes written are capped (PROF_MAXBYTES,
    default 50e6); past the cap trace lines are dropped and counted, program lines are still kept.
  * instruction lines (`/pe<k>/insn`, `/fp_ss<k>/insn`) are never written: they are AGGREGATED into 1 us bins per
    (cluster, unit, instruction class) and per (cluster, unit, function, class) totals -> PROF_INSN_OUT (CSV, gz);
    a capped raw sample (PROF_SAMPLE=lo_ns:hi_ns:maxlines of pe0 of the first cluster) -> PROF_SAMPLE_OUT.
  * PROF_STOP_NS: kill gapy when a line's time passes it (the run then ends as a failure by design; marks, events and
    bins up to that time are complete).
  * at exit the filtered log is compressed to <log>.zst next to it (zstd -19).

Env: PROF_TRACES, PROF_MAXBYTES, PROF_INSN_OUT, PROF_BIN_NS (1000), PROF_INSN_WINDOW=lo_ns:hi_ns, PROF_SAMPLE,
     PROF_SAMPLE_OUT, PROF_STOP_NS.
Usage (through run.sh, from experiments/smolvla-e2e-baseline):
     PROF_TRACES='/chip/cluster_0/pe[0-2]/insn,/chip/cluster_0/fp_ss[0-2]/insn' PROF_INSN_OUT=... \\
     ./run.sh published sim <run-name> -- "$PWD/profile/prof_wrap.py" smolvla-e2e --vision-only 256 --layers 1 --e2e-trace
"""
import gzip
import os
import re
import runpy
import subprocess as _real
import sys
import types
from collections import defaultdict
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
import softhier_mlir.sim.gvsoc as g  # noqa: E402

TRACES = [t for t in os.environ.get("PROF_TRACES", "").split(",") if t]
MAXB = int(float(os.environ.get("PROF_MAXBYTES", "50e6")))
BIN = int(os.environ.get("PROF_BIN_NS", "1000"))
INSN_OUT = os.environ.get("PROF_INSN_OUT", "")
WIN = [int(float(x)) for x in os.environ.get("PROF_INSN_WINDOW", "0:1e18").split(":")]
SAMPLE = os.environ.get("PROF_SAMPLE", "")
SAMPLE_OUT = os.environ.get("PROF_SAMPLE_OUT", "")
STOP_NS = int(float(os.environ.get("PROF_STOP_NS", "0")))

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_HEAD = re.compile(r": (-?\d+): \[")
_INSN = re.compile(r"/chip/cluster_(\d+)/(pe\d+|fp_ss\d+)/insn\s*\] (\S+)\s+\S+\s+[0-9a-f]{8}\s+(\S+)")
KEEP = ("Finished", "Cluster Sync", "Preloading Done", "Preheating Done")


def trace_start(ln):
    for m in _HEAD.finditer(ln):
        ns = int(m[1])
        ps = "0" if ns < 0 else str(ns * 1000)
        if ln[m.start() - len(ps):m.start()] == ps:
            return m.start() - len(ps), ns
    return -1, 0


def klass(mn):
    if mn.startswith("dmstat"):
        return "dma_poll"
    if mn.startswith("dm"):
        return "dma_issue"
    if mn.startswith(("frep", "scfg", "ssr")):
        return "frep_ssr"
    if mn in ("flw", "fld", "flh", "fsw", "fsd", "fsh"):
        return "fp_mem"
    if mn[0] == "f" or mn.startswith("vf"):
        return "fp_alu"
    if mn in ("lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"):
        return "int_load"
    if mn in ("sb", "sh", "sw", "sd"):
        return "int_store"
    if mn.startswith(("amo", "lr.", "sc.")):
        return "atomic"
    if mn.startswith("csr"):
        return "csr"
    if mn == "wfi":
        return "wfi"
    if mn[0] == "b" or mn in ("j", "jal", "jalr", "ret", "jr"):
        return "branch"
    return "int_alu"


class Agg:
    def __init__(self):
        self.bins = defaultdict(int)       # (cluster, unit, bin, class) -> n
        self.funcs = defaultdict(int)      # (cluster, unit, func, class) -> n
        self.lines = 0
        self.sample = []

    def add(self, ln, ns):
        m = _INSN.search(_ANSI.sub("", ln))
        if not m:
            return
        if not WIN[0] <= ns <= WIN[1]:
            return
        c, unit, fn, mn = m[1], m[2], m[3].split(":")[0], m[4]
        k = klass(mn)
        self.bins[(c, unit, ns // BIN, k)] += 1
        self.funcs[(c, unit, fn, k)] += 1
        self.lines += 1

    def dump(self, path):
        if not path:
            return
        with gzip.open(path, "wt") as f:
            f.write("# cluster,unit,bin_start_ns,class,count   (bin = %d ns)\n" % BIN)
            for (c, u, b, k), n in sorted(self.bins.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
                f.write(f"{c},{u},{b * BIN},{k},{n}\n")
            f.write("# FUNCS cluster,unit,function,class,count\n")
            for (c, u, fn, k), n in sorted(self.funcs.items()):
                f.write(f"F,{c},{u},{fn},{k},{n}\n")


def run(cmd, **kw):
    if not str(cmd[0]).endswith("gapy"):
        return _real.run(cmd, **kw)
    cmd = list(cmd) + [f"--trace={t}" for t in TRACES]
    print("[prof] gapy trace args:", " ".join(a for a in cmd if a.startswith("--trace=")), file=sys.stderr, flush=True)
    lf = kw.get("stdout")
    p = _real.Popen(cmd, stdout=_real.PIPE, stderr=_real.STDOUT, text=True, errors="replace", **{k: v for k, v in kw.items() if k in ("cwd", "env")})
    agg, wrote, dropped, seen, buf = Agg(), 0, 0, 0, []
    s_lo, s_hi, s_max = (SAMPLE.split(":") + ["0", "0", "0"])[:3] if SAMPLE else ("0", "-1", "0")
    s_lo, s_hi, s_max = int(float(s_lo)), int(float(s_hi)), int(float(s_max))
    samples = []
    out = lf if lf is not None else open(os.devnull, "w")
    for ln in p.stdout:
        k, ns = trace_start(ln)
        if k < 0:
            out.write(ln)
            buf.append(ln) if lf is None else None
            continue
        seen += 1
        if k > 0:
            out.write(ln[:k])                    # program characters that precede the trace line
        tl = ln[k:]
        if "/insn" in tl[:200]:
            agg.add(tl, ns)
            if SAMPLE and s_lo <= ns <= s_hi and len(samples) < s_max and "/pe0/insn" in tl[:200]:
                samples.append(_ANSI.sub("", tl))
        elif any(x in tl for x in KEEP):
            if wrote < MAXB:
                out.write(tl)
                wrote += len(tl)
            else:
                dropped += 1
        if STOP_NS and ns > STOP_NS:
            print(f"[prof] stop at {ns} ns", file=sys.stderr, flush=True)
            p.kill()
            break
    p.wait()
    if lf is not None:
        lf.flush()
    agg.dump(INSN_OUT)
    if SAMPLE_OUT and samples:
        with gzip.open(SAMPLE_OUT, "wt") as f:
            f.writelines(samples)
    print(f"[prof] trace lines seen {seen}, kept-trace bytes {wrote}, dropped (cap) {dropped}, insn lines aggregated {agg.lines}, "
          f"sample lines {len(samples)}", file=sys.stderr, flush=True)
    return _real.CompletedProcess(cmd, p.returncode, "".join(buf), "")


g.subprocess = types.SimpleNamespace(run=run, TimeoutExpired=_real.TimeoutExpired, Popen=_real.Popen, PIPE=_real.PIPE,
                                     CalledProcessError=_real.CalledProcessError, STDOUT=_real.STDOUT, DEVNULL=_real.DEVNULL)
script = str(root / "tests/gvsoc/run.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
