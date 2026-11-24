#!/usr/bin/env python3
"""Row-operator kernels (Eltwise, Activation, LayerNorm, RMSNorm) on H1: one run per stage-1 shape.

    row_kernels.py plan | run | summary

These kernels have no tile parameter: one run per shape, exact shape (stage-1 spec §4.2). Each run goes through
run_kernel.sh (patched GVSoC, stamp check, MALLOC_PERTURB_, output check). The list is every non-GEMM,
non-attention constants entry the stage-1 model needs for W1 (vlabench-3cam, = release-3cam shapes) and W1-b
(cameras batched), from smolvla_ops.hand_kernel_key, with the rows at the W1 counts (prefix 241, not 177).
Raw results: results/row_kernels_h1.csv. Run folders: $SHVLA_DATA/stage1-h1/<kernel>/<name>/.
"""
import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
KREPO = Path(os.environ.get("KREPO", HERE.parents[1] / "third_party/softhier-kernels"))  # kernel source (submodule)
CFG_DIR = KREPO / "implementation/config/kernels"
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla"))
RUNS = DATA / "stage1-h1"
CSV = HERE / "results/row_kernels_h1.csv"

# (constants name, run_kernel kernel, config template, field values, workloads, operators)
RUNS_LIST = [
    ("eltwise_add_1024x768", "elt", "elt", dict(op="'add'", m_size=1024, n_size=768), "W1", "vision residual_attn/mlp"),
    ("eltwise_add_3072x768", "elt", "elt", dict(op="'add'", m_size=3072, n_size=768), "W1-b", "vision residual_attn/mlp"),
    ("eltwise_add_241x960", "elt", "elt", dict(op="'add'", m_size=241, n_size=960), "W1 W1-b", "prefix residual_attn/mlp"),
    ("eltwise_add_50x720", "elt", "elt", dict(op="'add'", m_size=50, n_size=720), "W1 W1-b", "expert residual_attn/mlp"),
    ("eltwise_axpy_50x32", "elt", "elt", dict(op="'axpy'", m_size=50, n_size=32), "W1 W1-b", "expert euler_update"),
    ("eltwise_bias_1024x768", "elt", "elt", dict(op="'bias'", m_size=1024, n_size=768), "W1", "vision bias_patch/q/k/v/o/down"),
    ("eltwise_bias_1024x3072", "elt", "elt", dict(op="'bias'", m_size=1024, n_size=3072), "W1", "vision bias_up"),
    ("eltwise_bias_3072x768", "elt", "elt", dict(op="'bias'", m_size=3072, n_size=768), "W1-b", "vision bias_patch/q/k/v/o/down"),
    ("eltwise_bias_3072x3072", "elt", "elt", dict(op="'bias'", m_size=3072, n_size=3072), "W1-b", "vision bias_up"),
    ("eltwise_bias_50x32", "elt", "elt", dict(op="'bias'", m_size=50, n_size=32), "W1 W1-b", "expert bias_action_out"),
    ("eltwise_bias_50x720", "elt", "elt", dict(op="'bias'", m_size=50, n_size=720), "W1 W1-b", "expert bias_action_in/time_in/time_out"),
    ("eltwise_pixel_shuffle_3072x768", "elt", "elt", dict(op="'pixel_shuffle'", m_size=192, n_size=12288), "W1 W1-b", "connector pixel_shuffle (3 cameras; config m=192 n=12288)"),
    ("eltwise_rope_241x320", "elt", "elt", dict(op="'rope'", m_size=241, n_size=320, head_dim=64), "W1 W1-b", "prefix rope_k"),
    ("eltwise_rope_241x960", "elt", "elt", dict(op="'rope'", m_size=241, n_size=960, head_dim=64), "W1 W1-b", "prefix rope_q"),
    ("eltwise_rope_50x320", "elt", "elt", dict(op="'rope'", m_size=50, n_size=320, head_dim=64), "W1 W1-b", "expert rope_k"),
    ("eltwise_rope_50x960", "elt", "elt", dict(op="'rope'", m_size=50, n_size=960, head_dim=64), "W1 W1-b", "expert rope_q"),
    ("eltwise_scale_192x960", "elt", "elt", dict(op="'scale'", m_size=192, n_size=960), "W1 W1-b", "connector embed_scale"),
    ("gelu_1024x3072", "acti", "acti", dict(algo="'gelu'", m_size=1024, n_size=3072, gate_enable=0, bias_enable=0), "W1", "vision gelu"),
    ("gelu_3072x3072", "acti", "acti", dict(algo="'gelu'", m_size=3072, n_size=3072, gate_enable=0, bias_enable=0), "W1-b", "vision gelu"),
    ("silu_50x720", "acti", "acti", dict(algo="'silu'", m_size=50, n_size=720, gate_enable=0, bias_enable=0), "W1 W1-b", "expert time_silu"),
    ("silu_gate_241x2560", "acti", "acti", dict(algo="'silu'", m_size=241, n_size=2560, gate_enable=1, bias_enable=0), "W1 W1-b", "prefix silu_mul"),
    ("silu_gate_50x2048", "acti", "acti", dict(algo="'silu'", m_size=50, n_size=2048, gate_enable=1, bias_enable=0), "W1 W1-b", "expert silu_mul"),
    ("layernorm_1024x768", "lnorm", "lnorm", dict(m_size=1024, n_size=768), "W1", "vision ln1/ln2/ln_post"),
    ("layernorm_3072x768", "lnorm", "lnorm", dict(m_size=3072, n_size=768), "W1-b", "vision ln1/ln2/ln_post"),
    ("rmsnorm_241x960", "norm", "norm", dict(m_size=241, n_size=960), "W1 W1-b", "prefix rmsnorm_attn/mlp/final"),
    ("rmsnorm_50x720", "norm", "norm", dict(m_size=50, n_size=720), "W1 W1-b", "expert rmsnorm_attn/mlp/final"),
]
KDIR = {"elt": "eltwise", "acti": "activation", "lnorm": "layernorm", "norm": "rmsnorm"}
FIELDS = ["name", "kernel", "workloads", "ops", "config", "time_ns", "bad", "max_ulp", "status", "run_dir"]


def write_cfg(path, tmpl, vals):
    s = (CFG_DIR / f"{tmpl}.py").read_text()
    for k, v in vals.items():
        s, n = re.subn(rf"(self\.{k}\s*=\s*)\S+", lambda m: m.group(1) + str(v), s)
        assert n == 1, (tmpl, k)
    path.write_text(s)


def parse_run(out):
    log = (out / "run.log").read_text(errors="replace") if (out / "run.log").exists() else ""
    t = re.search(r"Execution period is (\d+) ns", log)
    bad, mu, n = 0, 0, 0
    for m in re.finditer(r"\[check\] cluster \d+: rows \d+, bad (\d+) \(nan \d+\), max_ulp (\d+)", log):
        bad += int(m.group(1)); mu = max(mu, int(m.group(2))); n += 1
    return (int(t.group(1)) if t else None), (bad if n else None), (mu if n else None)


def rows():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "summary"])
    ap.add_argument("--only", action="append", help="constants name(s) to run")
    a = ap.parse_args()
    done = {r["name"] for r in rows() if r["status"] == "ok"}
    todo = [r for r in RUNS_LIST if not a.only or r[0] in a.only]
    if a.cmd == "summary":
        for r in rows():
            print(f"{r['name']:<32} {r['time_ns']:>9} ns  bad {r['bad']:<6} {r['status']:<8} {r['workloads']:<8} {r['ops']}")
        return
    for name, kern, tmpl, vals, wl, ops in todo:
        cfgtxt = " ".join(f"{k}={v}" for k, v in vals.items()).replace("'", "")
        if a.cmd == "plan":
            print(f"{name:<32} {kern:<6} {cfgtxt:<50} {wl:<8} {'done' if name in done else ''}")
            continue
        if name in done:
            continue
        out = RUNS / KDIR[kern] / name
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / f"{tmpl}_cfg.py"
        write_cfg(cfg, tmpl, vals)
        print(f"[{name}] ...", flush=True)
        rc = subprocess.run([str(HERE / "run_kernel.sh"), kern, str(cfg), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
        t, bad, mu = parse_run(out)
        status = "ok" if (rc == 0 and t is not None and bad == 0) else ("timeout_or_fail" if t is None else "wrong")
        new = not CSV.exists()
        CSV.parent.mkdir(parents=True, exist_ok=True)
        with CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(dict(name=name, kernel=kern, workloads=wl, ops=ops, config=cfgtxt, time_ns=t, bad=bad,
                            max_ulp=mu, status=status, run_dir=str(out)))
        print(f"    {t} ns, bad {bad}, {status}", flush=True)


if __name__ == "__main__":
    main()
