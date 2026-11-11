#!/usr/bin/env python3
"""Build stamp for the GVSoC simulator: records what the installed build was made from.

  stamp.py write <arch>   write third_party/gvsoc/.shvla_stamp.json for the current tree
  stamp.py check <arch>   exit 0 if the stamp matches the current tree, 1 (difference printed) if not
  stamp.py permalinks     GitHub links: GVSoC upstream base (+ series hash) and every nested submodule
  stamp.py series-hash    SHA-256 of the GVSoC patch series (infra/softhier/gvsoc-patches/)

The stamp holds: arch path (relative to $SHVLA_ROOT) and SHA-256; commit of GVSoC and of every
nested submodule; per repository the SHA-256 of `git diff HEAD` plus
`git status --porcelain --untracked-files=no`; and the model folders softhier-mlir's run_sim
picks at run time (install/models_fast, install/models_fix: existence and SHA-256 over the .so
files; values of SOFTHIER_MODEL_DIR and SOFTHIER_STOCK_MODELS); and the build's effective
PWR_INTERVAL_PS (CMake cache); and the GVSoC patch series: its SHA-256 (over `series` and every listed
patch, each as name NUL bytes NUL), the base commit, the expected tree and the tree of HEAD. `check` fails
if the series files no longer match the SHA-256 in gvsoc-patches/expected, if HEAD^{tree} is not the
expected tree, or if the base commit is not an ancestor of HEAD. `check` also fails if that value differs from $SHVLA_PWR_INTERVAL_PS
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
PATCHES = HERE / "gvsoc-patches"
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


def series_files():
    """Names listed in gvsoc-patches/series (comments and blank lines skipped)."""
    return [l.strip() for l in (PATCHES / "series").read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def series_sha256():
    """SHA-256 over `series` and each listed patch in order: name, NUL, bytes, NUL."""
    h = hashlib.sha256()
    for name in ["series", *series_files()]:
        if not (PATCHES / name).is_file():
            sys.exit(f"gvsoc-patches/series lists {name}, which is missing")
        h.update(name.encode() + b"\0" + (PATCHES / name).read_bytes() + b"\0")
    return h.hexdigest()


def expected():
    """gvsoc-patches/expected: BASE, TREE, SERIES_SHA256 (key=value lines)."""
    out = {}
    for line in (PATCHES / "expected").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def series_state():
    exp = expected()
    r = subprocess.run(["git", "-C", str(GVSOC), "merge-base", "--is-ancestor", exp["BASE"], "HEAD"])
    return {
        "series_sha256": series_sha256(),
        "base": exp["BASE"],
        "expected_tree": exp["TREE"],
        "tree": git(GVSOC, "rev-parse", "HEAD^{tree}").strip(),
        "base_is_ancestor": r.returncode == 0,
    }


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
        "gvsoc_series": series_state(),
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
    ser, exp = series_state(), expected()
    if ser["series_sha256"] != exp.get("SERIES_SHA256"):
        diffs.append(f"  gvsoc series files: sha256 {ser['series_sha256']} but gvsoc-patches/expected "
                     f"says {exp.get('SERIES_SHA256')!r} (re-export: infra/softhier/README.md)")
    if ser["tree"] != ser["expected_tree"]:
        diffs.append(f"  gvsoc tree: HEAD^{{tree}}={ser['tree']} is not base+series {ser['expected_tree']} "
                     f"(run infra/softhier/apply_gvsoc_patches.sh)")
    if not ser["base_is_ancestor"]:
        diffs.append(f"  gvsoc base {ser['base']} is not an ancestor of HEAD")
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
        if rel == ".":
            ser = series_state()
            print(f"GVSoC: https://github.com/gvsoc/gvsoc/tree/{ser['base']} (upstream base) + patch series "
                  f"infra/softhier/gvsoc-patches sha256 {ser['series_sha256'][:16]}, tree {ser['tree']}"
                  + ("" if ser["tree"] == ser["expected_tree"] else " (NOT the expected tree)"))
            continue
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
    if len(argv) == 2 and argv[1] == "series-hash":
        print(series_sha256())
        return 0
    if len(argv) == 2 and argv[1] == "permalinks":
        return cmd_permalinks()
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
