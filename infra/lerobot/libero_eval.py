#!/usr/bin/env python3
"""Closed-loop LIBERO evaluation of a LeRobot policy: a thin wrapper around `lerobot-eval`.

Builds the `lerobot-eval` command, runs it under a memory cap, then reads
`eval_info.json`, prints the success rate per task and overall with Wilson 95 %
intervals, and writes `summary.json` next to it.

Run it from the LeRobot environment (`source infra/lerobot/env.sh`):

    python infra/lerobot/libero_eval.py --run-name smoke --suite libero_spatial \
        --task-ids 0 --n-episodes 1 --batch-size 1

Output goes to `$SHVLA_DATA/eval/<run-name>/` (the run name must be new).
`--n-episodes` is per task: LeRobot passes it to every task of the suite
(`lerobot_eval.py`, `eval_policy_all`), so 10 tasks x 10 episodes is `--n-episodes 10`.

Standard library and numpy only (numpy for nothing but the interval arithmetic).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_POLICY = "lerobot/smolvla_libero"
DEFAULT_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
# The checkpoint names its cameras camera1/camera2; the LIBERO env names them image/image2.
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
CAP_PREFIX = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "MemorySwapMax=0"]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, as fractions."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def stats(successes: list) -> dict:
    n = len(successes)
    k = int(sum(bool(s) for s in successes))
    lo, hi = wilson(k, n)
    return {"n_success": k, "n_episodes": n, "rate": (k / n if n else math.nan), "wilson95": [lo, hi]}


def build_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    exe = shutil.which("lerobot-eval") or str(Path(sys.executable).parent / "lerobot-eval")
    cmd = [
        exe,
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--policy.path={args.policy_path}",
        f"--policy.pretrained_revision={args.revision}",
        f"--policy.device={args.device}",
        "--rename_map=" + json.dumps(RENAME_MAP),
        f"--seed={args.seed}",
        f"--output_dir={out_dir}",
    ]
    if args.task_ids:
        cmd.append("--env.task_ids=" + json.dumps(args.task_ids))
    return cmd


def summarise(out_dir: Path, meta: dict) -> dict:
    info = json.loads((out_dir / "eval_info.json").read_text())
    per_task = []
    all_succ: list = []
    for t in sorted(info["per_task"], key=lambda t: (t["task_group"], t["task_id"])):
        succ = [bool(s) for s in (t["metrics"].get("successes") or [])]
        all_succ.extend(succ)
        per_task.append({"task_group": t["task_group"], "task_id": t["task_id"], **stats(succ)})
    summary = {**meta, "per_task": per_task, "overall": stats(all_succ)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_summary(summary: dict) -> None:
    def line(label: str, s: dict) -> str:
        lo, hi = s["wilson95"]
        return (f"{label:<24} {s['n_success']:>3}/{s['n_episodes']:<3} "
                f"{100 * s['rate']:5.1f} %  [{100 * lo:5.1f}, {100 * hi:5.1f}] (Wilson 95 %)")

    for t in summary["per_task"]:
        print(line(f"{t['task_group']}_{t['task_id']}", t))
    print(line("overall", summary["overall"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True, help="new folder name under $SHVLA_DATA/eval/")
    ap.add_argument("--suite", default="libero_spatial", help="LIBERO suite (--env.task)")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None, help="restrict to these task ids (default: all)")
    ap.add_argument("--n-episodes", type=int, default=10, help="episodes per task")
    ap.add_argument("--batch-size", type=int, default=1, help="parallel envs per task (--eval.batch_size)")
    ap.add_argument("--policy-path", default=DEFAULT_POLICY)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-cap", action="store_true", help="do not run under systemd-run's memory cap")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    data = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache" / "softhier-vla"))
    out_dir = data / "eval" / args.run_name
    if out_dir.exists():
        print(f"error: {out_dir} exists; use a new --run-name", file=sys.stderr)
        return 2

    cmd = build_command(args, out_dir)
    if not args.no_cap:
        cmd = CAP_PREFIX + cmd
    print("command:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    wall = time.time() - t0
    # Largest resident set of any single process in the (waited-for) process tree, in MiB.
    peak_rss_mib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    if rc != 0:
        print(f"error: lerobot-eval exited with {rc}", file=sys.stderr)
        return rc
    if not (out_dir / "eval_info.json").is_file():
        print(f"error: {out_dir / 'eval_info.json'} was not written", file=sys.stderr)
        return 1

    meta = {
        "run_name": args.run_name,
        "suite": args.suite,
        "task_ids": args.task_ids,
        "n_episodes_per_task": args.n_episodes,
        "batch_size": args.batch_size,
        "policy_path": args.policy_path,
        "revision": args.revision,
        "seed": args.seed,
        "command": cmd,
        "wall_s": round(wall, 1),
        "peak_rss_mib_largest_process": round(peak_rss_mib),
    }
    summary = summarise(out_dir, meta)
    print_summary(summary)
    print(f"wall {wall:.0f} s, peak RSS of the largest process {peak_rss_mib:.0f} MiB")
    print(f"summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
