"""diag8: run tests/gvsoc/run.py with gvsoc --trace on narrow paths, filtered in-process (streaming, nothing unfiltered
reaches disk). Env: D8_TRACES=comma list of --trace regexes; D8_RULES=json [[path_substr, keep_regex, cap_lines],...];
D8_OUT=output file for kept trace lines; D8_LEVEL=optional --trace-level; D8_MAXBYTES (default 50e6)."""
import sys, runpy, os, re, json, subprocess, types
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root))
import softhier_mlir.sim.gvsoc as g
_real = subprocess
traces = [t for t in os.environ.get("D8_TRACES", "").split(",") if t]
rules = [(r[0], re.compile(r[1]), int(r[2]), int(r[3]) if len(r) > 3 else 0, int(r[4]) if len(r) > 4 else 1 << 62) for r in json.loads(os.environ.get("D8_RULES", "[]"))]
stop_ps = int(float(os.environ.get("D8_STOP_PS", "0")))
out_path = os.environ["D8_OUT"]
maxbytes = int(float(os.environ.get("D8_MAXBYTES", "50e6")))
level = os.environ.get("D8_LEVEL")
TSRE = re.compile(r"\d+: (\d+): \[")
TR = re.compile(r"^\s*\d+: \d+: \[")


def run(cmd, **kw):
    if not str(cmd[0]).endswith("gapy"):
        return _real.run(cmd, **kw)
    cmd = list(cmd) + [f"--trace={t}" for t in traces] + ([f"--trace-level={level}"] if level else [])
    print("[d8] gapy cmd tail:", " ".join(cmd[-8:]), file=sys.stderr)
    p = _real.Popen(cmd, stdout=_real.PIPE, stderr=_real.STDOUT, text=True, errors="replace",
                    **{k: v for k, v in kw.items() if k in ("cwd", "env")})
    counts = [0] * len(rules)
    prog = []
    wrote = 0
    seen = 0
    with open(out_path, "w") as fo:
        for ln in p.stdout:
            if TR.match(ln) or "/chip/" in ln[:120]:
                seen += 1
                m = TSRE.search(ln)
                ts = int(m.group(1)) * 1000 if m else 0
                if stop_ps and ts > stop_ps:
                    print('[d8] stop at line:', ln[:200], file=sys.stderr)
                    p.kill()
                    break
                done = False
                for i, (ps, kr, cap, lo, hi) in enumerate(rules):
                    if ps in ln and counts[i] < cap and wrote < maxbytes and lo <= ts <= hi and kr.search(ln):
                        counts[i] += 1
                        if not done:
                            wrote += len(ln)
                            fo.write(ln)
                            done = True
            else:
                prog.append(ln)
    p.wait()
    print(f"[d8] trace lines seen {seen}, kept per-rule {counts}, bytes {wrote}", file=sys.stderr)
    return _real.CompletedProcess(cmd, p.returncode, "".join(prog), "")


g.subprocess = types.SimpleNamespace(run=run, TimeoutExpired=_real.TimeoutExpired, Popen=_real.Popen, PIPE=_real.PIPE,
                                     CalledProcessError=_real.CalledProcessError, STDOUT=_real.STDOUT,
                                     DEVNULL=_real.DEVNULL)
script = str(root / "tests/gvsoc/run.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
