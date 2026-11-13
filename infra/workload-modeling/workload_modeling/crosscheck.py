"""Cross-check the standalone model against the programs PR1 compiled and ran (run by hand, once).

    python -m workload_modeling.crosscheck \\
        --parse-profile <experiments/smolvla-e2e-baseline/profile/parse_profile.py> \\
        --pr1 ~/.cache/softhier-vla/profile/published-fixi-rung6 \\
        --libero ~/.cache/softhier-vla/profile/libero-G4a \\
        --experiment-commit 68cd4eb

For each workload it reads, with PR1's parse_profile.py (parse_mlir + flatten + label):
  - vision tokens per camera (rows of the patch-embedding product),
  - prefix rows as compiled (rows of the prefix query projection, padded),
  - true prefix tokens (key rows of the expert's cross-attention),
checks the first and last against the model, builds the model with PR1's lowering
(smolvla_ops.pr1_lowering), and compares per-stage matrix-product FLOP with the sum of
`redmule_flops` over `level == op` rows of ops.csv (pr1-1cam, measured) or of the scaled
libero_ops_scaled.csv (libero-2cam, estimate: G4a partial runs scaled with PR1's recipe).
Writes docs/analysis/crosscheck.md and crosscheck.json. Exits 1 if any stage differs by more than 1 %.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from . import constants
from . import smolvla_ops as m

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "docs" / "analysis"
TOLERANCE_PCT = 1.0

# workload -> (program dir relative to the given root, ops csv relative to the root, reference label)
LAYOUT = {
    "pr1-1cam": {
        "vision": "vision.mlir", "prefix": "prefix_a.mlir", "expert": "expert.mlir",
        "ops": "ops.csv", "label": "measured",
    },
    "libero-2cam": {
        "vision": "G4a.1/vision.mlir", "prefix": "G4a.2/prefix_a.mlir", "expert": "G4a.3/expert.mlir",
        "ops": "libero_ops_scaled.csv", "label": "estimate",
    },
}
_MEMREF = re.compile(r"memref<(\d+)x(\d+)x")


def load_parse_profile(path: Path):
    spec = importlib.util.spec_from_file_location("parse_profile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ops(pp, path: Path, phase: str):
    tree, _ = pp.parse_mlir(path)
    for n in pp.flatten(tree):
        if n[0] == "op":
            _, op, srcs, dst, text, _env = n
            yield pp.label(op, srcs, dst, text, phase, {}), text


def _first_rows(text: str, i: int = 0) -> int:
    return int(_MEMREF.findall(text)[i][0])


def program_shapes(pp, root: Path, lay: dict) -> dict[str, int]:
    vis = next(t for lab, t in _ops(pp, root / lay["vision"], "vision") if lab == "patch_embed")
    pre = next(t for lab, t in _ops(pp, root / lay["prefix"], "prefix") if lab == "gemm_q")
    xat = next(t for lab, t in _ops(pp, root / lay["expert"], "expert") if lab == "xattn_prefix")
    return {
        "vision_rows": _first_rows(vis),      # all cameras of the program
        "prefix_rows": _first_rows(pre),
        "prefix_tokens": _first_rows(xat, 1),  # key rows of the cross-attention
    }


def redmule_by_phase(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["level"] != "op":
                continue
            out[r["phase"]] = out.get(r["phase"], 0.0) + float(r["redmule_flops"] or 0)
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run(c, parse_profile: Path, roots: dict[str, Path], experiment_commit: str) -> tuple[dict, bool]:
    pp = load_parse_profile(parse_profile)
    result = {"tolerance_pct": TOLERANCE_PCT, "experiment_commit": experiment_commit,
              "parse_profile": str(parse_profile.name), "workloads": {}}
    ok = True
    for wid, root in roots.items():
        lay = LAYOUT[wid]
        wl = c.workloads[wid]
        d = m.load_dims(wl)
        shapes = program_shapes(pp, root, lay)
        n_vis = m.vision_tokens(wl, d)
        P = m.prefix_tokens(wl, d)
        checks = {
            "vision tokens per camera": (shapes["vision_rows"] in (n_vis, n_vis * wl.cameras), shapes["vision_rows"], n_vis),
            "prefix tokens": (shapes["prefix_tokens"] == P, shapes["prefix_tokens"], P),
        }
        padded = m.stage_totals(m.build(c, wid, m.pr1_lowering(shapes["prefix_rows"])))
        unpadded = m.stage_totals(m.build(c, wid, m.Lowering(kv_proj_per_step=False, time_in_split=True)))
        reference = m.stage_totals(m.build(c, wid))
        red = redmule_by_phase(root / lay["ops"])
        stages = {}
        for s in m.STAGES:
            ref = red.get(s, 0.0)
            diff = 100.0 * (padded[s].matmul_flop / ref - 1) if ref else float("nan")
            passed = abs(diff) <= TOLERANCE_PCT
            ok &= passed
            stages[s] = {
                "model_padded_gflop": round(padded[s].matmul_flop / 1e9, 4),
                "model_unpadded_gflop": round(unpadded[s].matmul_flop / 1e9, 4),
                "model_reference_gflop": round(reference[s].matmul_flop / 1e9, 4),
                "pr1_redmule_gflop": round(ref / 1e9, 4),
                "diff_pct": round(diff, 4),
                "padding_overhead_pct": round(100.0 * (padded[s].matmul_flop / unpadded[s].matmul_flop - 1), 2),
                "pass": passed,
            }
        ok &= all(good for good, _, _ in checks.values())
        result["workloads"][wid] = {
            "label": lay["label"],
            "programs": {k: lay[k] for k in ("vision", "prefix", "expert")},
            "ops_csv": {"path": lay["ops"], "sha256_16": _sha(root / lay["ops"])},
            "program_shapes": shapes,
            "shape_checks": {k: {"pass": g, "program": got, "model": want} for k, (g, got, want) in checks.items()},
            "stages": stages,
        }
    return result, ok


def render_md(res: dict) -> str:
    lines = [
        "# Cross-check: standalone model against PR1's compiled programs",
        "",
        "<!-- generated by `python -m workload_modeling.crosscheck`; do not edit by hand -->",
        "",
        "The standalone SmolVLA model (`infra/workload-modeling/workload_modeling/smolvla_ops.py`) is compared with the "
        "RedMulE work PR1's programs actually did. PR1's programs and per-op CSVs are kept outside git "
        "(`~/.cache/softhier-vla/profile/`); they were parsed with PR1's `parse_profile.py` "
        f"(experiment branch `worktree-smolvla-e2e-baseline`, commit `{res['experiment_commit']}`).",
        "",
        "Columns, per model stage, matrix-product FLOP (GEMM + attention QK^T and P.V) in GFLOP:",
        "",
        "* **model padded**: the model with PR1's lowering: prefix rows as compiled, expert attention keys padded to a "
        "multiple of 32 by the runtime, cross-attention K/V projected once per call, time-MLP input split "
        "(`smolvla_ops.pr1_lowering`).",
        "* **PR1 RedMulE**: sum of `redmule_flops` over `level == op` rows of the run's ops CSV.",
        f"* **diff**: model padded against PR1 RedMulE; the pass rule is |diff| <= {res['tolerance_pct']} %.",
        "* **model unpadded**: PR1's rewrites with true token counts; **padding** is the padded overhead over it.",
        "* **reference**: the reference computation (lerobot code: K/V projection in every step, full time-MLP "
        "input), true token counts. This is what `stage-totals.md` reports.",
        "",
    ]
    for wid, w in res["workloads"].items():
        sh = w["program_shapes"]
        lines += [
            f"## `{wid}` (reference label: {w['label']})",
            "",
            f"Programs: {', '.join('`' + p + '`' for p in w['programs'].values())}; ops CSV `{w['ops_csv']['path']}` "
            f"(sha256 prefix `{w['ops_csv']['sha256_16']}`).",
            f"Read from the programs: vision rows {sh['vision_rows']}, prefix rows {sh['prefix_rows']} (padded), "
            f"prefix tokens {sh['prefix_tokens']}.",
            "",
            "| shape check | program | model | pass |",
            "| --- | ---: | ---: | --- |",
        ]
        for k, v in w["shape_checks"].items():
            lines.append(f"| {k} | {v['program']} | {v['model']} | {'yes' if v['pass'] else 'NO'} |")
        lines += [
            "",
            "| stage | model padded | PR1 RedMulE | diff % | pass | model unpadded | padding % | reference |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for s, v in w["stages"].items():
            lines.append(
                f"| {s} | {v['model_padded_gflop']:.4f} | {v['pr1_redmule_gflop']:.4f} | {v['diff_pct']:+.4f} | "
                f"{'yes' if v['pass'] else 'NO'} | {v['model_unpadded_gflop']:.4f} | {v['padding_overhead_pct']:.2f} | "
                f"{v['model_reference_gflop']:.4f} |")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "* All PR1 runs used ideal HBM (`SOFTHIER_IDEAL_HBM=1`); FLOP counts do not depend on it.",
        "* The `libero-2cam` reference is an estimate: G4a partial runs (vision 1 layer at 1024 tokens, prefix 8 layers, "
        "expert 2 steps) scaled with PR1's recipe (PROFILE.md, LIBERO configuration). The model is not scaled.",
        "* Remaining differences: the model counts the host-side one-row time half of the time MLP "
        "(`time_mlp_in_time`), which PR1 folds into a per-step bias table on the host "
        "(softhier-mlir `b11cd5f` `frontend/smolvla_expert.py:27-28`) and does not run on RedMulE.",
        "* The expert attention key padding is inside the runtime (`runtime/sh_expert.inc.c:83`, "
        "`(Lp + So + 31u) & ~31u`), so the programs show 65 / 177 key rows while RedMulE works on 96 / 192.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m workload_modeling.crosscheck")
    p.add_argument("--parse-profile", type=Path, required=True)
    p.add_argument("--pr1", type=Path, required=True, help="published-fixi-rung6 directory")
    p.add_argument("--libero", type=Path, required=True, help="libero-G4a directory")
    p.add_argument("--experiment-commit", required=True, help="commit of the experiment branch parse_profile.py was read at")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    c = constants.load()
    res, ok = run(c, a.parse_profile.expanduser(), {"pr1-1cam": a.pr1.expanduser(), "libero-2cam": a.libero.expanduser()},
                  a.experiment_commit)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "crosscheck.json").write_text(json.dumps(res, indent=2) + "\n")
    (a.out / "crosscheck.md").write_text(render_md(res))
    for wid, w in res["workloads"].items():
        for s, v in w["stages"].items():
            print(f"{wid:12s} {s:10s} model {v['model_padded_gflop']:10.4f}  pr1 {v['pr1_redmule_gflop']:10.4f}  "
                  f"{v['diff_pct']:+.4f} %  {'ok' if v['pass'] else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
