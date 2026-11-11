#!/usr/bin/env python3
"""Build stamp for the GVSoC simulator: records what the installed build was made from.

  stamp.py write <arch>   write third_party/gvsoc/.shvla_stamp.json for the current tree
  stamp.py check <arch>   exit 0 if the stamp matches the current tree, 1 (difference printed) if not
  stamp.py permalinks     GitHub tree links of GVSoC and every nested submodule

The stamp holds: arch path (relative to $SHVLA_ROOT) and SHA-256; commit of GVSoC and of every
nested submodule; per repository the SHA-256 of `git diff HEAD` plus
`git status --porcelain --untracked-files=no`; and the model folders softhier-mlir's run_sim
picks at run time (install/models_fast, install/models_fix: existence and SHA-256 over the .so
files; values of SOFTHIER_MODEL_DIR and SOFTHIER_STOCK_MODELS); and the build's effective
PWR_INTERVAL_PS (CMake cache). `check` also fails if that value differs from $SHVLA_PWR_INTERVAL_PS
(default 9000000000000000, as in infra/env.sh).
Untracked files are not covered; build products are untracked by design.
Standard library only.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SHVLA_ROOT") or HERE.parent.parent).resolve()
GVSOC = Path(os.environ.get("SOFTHIER_HOME") or ROOT / "third_party" / "gvsoc").resolve()
STAMP = GVSOC / ".shvla_stamp.json"
DEFAULT_PWR_INTERVAL_PS = "9000000000000000"


def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
    return r.stdout


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repos():
    """GVSoC plus every nested submodule: {relative path: absolute path}."""
    out = {".": GVSOC}
    for line in git(GVSOC, "submodule", "status", "--recursive").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            out[parts[1]] = GVSOC / parts[1]
    return out


def repo_entry(path):
    diff = git(path, "diff", "HEAD")
    status = git(path, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git(path, "rev-parse", "HEAD").strip(),
        "diff_sha256": sha256_bytes((diff + "\n--status--\n" + status).encode()),
    }


def models_entry(name):
    d = GVSOC / "install" / name
    if not d.is_dir():
        return {"exists": False, "so_sha256": None}
    h = hashlib.sha256()
    for so in sorted(d.rglob("*.so")):
        h.update(str(so.relative_to(d)).encode() + b"\0" + sha256_file(so).encode() + b"\0")
    return {"exists": True, "so_sha256": h.hexdigest()}


def pwr_interval_ps():
    """PWR_INTERVAL_PS the build was configured with (build/CMakeCache.txt), None if absent."""
    cache = GVSOC / "build" / "CMakeCache.txt"
    if cache.is_file():
        m = re.search(r"^PWR_INTERVAL_PS:[A-Z]+=(.*)$", cache.read_text(), re.M)
        if m:
            return m.group(1).strip()
    return None


def current(arch):
    arch_abs = Path(arch).resolve()
    try:
        rel = str(arch_abs.relative_to(ROOT))
    except ValueError:
        rel = str(arch_abs)
    return {
        "arch": {"path": rel, "sha256": sha256_file(arch_abs)},
        "repos": {k: repo_entry(v) for k, v in sorted(repos().items())},
        "pwr_interval_ps": pwr_interval_ps(),
        "models": {n: models_entry(n) for n in ("models_fast", "models_fix")},
        "env": {k: os.environ.get(k) for k in ("SOFTHIER_MODEL_DIR", "SOFTHIER_STOCK_MODELS")},
    }


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def cmd_write(arch):
    STAMP.write_text(json.dumps(current(arch), indent=2, sort_keys=True) + "\n")
    print(f"stamp written: {STAMP}")
    return 0


def cmd_check(arch):
    if not STAMP.is_file():
        print(f"stamp missing: {STAMP} (run infra/softhier/build_arch.sh {arch})")
        return 1
    want = dict(flatten(json.loads(STAMP.read_text())))
    have = dict(flatten(current(arch)))
    diffs = [f"  {k}: stamp={want.get(k)!r} now={have.get(k)!r}"
             for k in sorted(set(want) | set(have)) if want.get(k) != have.get(k)]
    expect = os.environ.get("SHVLA_PWR_INTERVAL_PS") or DEFAULT_PWR_INTERVAL_PS
    if have.get("pwr_interval_ps") != expect:
        diffs.append(f"  pwr_interval_ps: build={have.get('pwr_interval_ps')!r} expected "
                     f"SHVLA_PWR_INTERVAL_PS={expect!r} (rebuild with build_arch.sh)")
    if diffs:
        print("stamp MISMATCH:")
        print("\n".join(diffs))
        return 1
    print("stamp OK")
    return 0


def cmd_permalinks():
    for rel, path in repos().items():
        url = git(path, "config", "--get", "remote.origin.url").strip()
        url = re.sub(r"\.git$", "", url)
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):]
        print(f"{'GVSoC' if rel == '.' else rel}: {url}/tree/{git(path, 'rev-parse', 'HEAD').strip()}")
    return 0


def main(argv):
    if len(argv) == 3 and argv[1] in ("write", "check"):
        if not Path(argv[2]).is_file():
            sys.exit(f"arch file not found: {argv[2]}")
        return cmd_write(argv[2]) if argv[1] == "write" else cmd_check(argv[2])
    if len(argv) == 2 and argv[1] == "permalinks":
        return cmd_permalinks()
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
