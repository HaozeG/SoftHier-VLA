#!/usr/bin/env python3
"""FlatAttention correctness bisection on H1 (debug only, not a measurement). Small configs that switch on one
kernel feature at a time: plain tile, online-softmax rescale (Tc > 1), row-wise collectives (sx > 1), column-wise
broadcast (sy > 1), stacked GQA query heads. Run folders: $SHVLA_DATA/stage1-h1/attn_debug/<name>/."""
import subprocess, sys
import attn_sweep as A

# name: (q heads, kv heads, q len, kv len, sx, sy, Br, Bc)
CASES = {
    "d1_plain":      (1, 1, 64, 128, 1, 1, 64, 128),
    "d2_tc2":        (1, 1, 64, 128, 1, 1, 64, 64),
    "d3_rowwise":    (1, 1, 64, 128, 2, 1, 64, 128),
    "d4_colwise":    (1, 1, 128, 128, 1, 2, 128, 128),
    "d5_gqa":        (3, 1, 64, 128, 1, 1, 192, 128),
    "d6_rowwise4":   (1, 1, 64, 256, 4, 1, 64, 256),
}
for name in (sys.argv[1:] or CASES):
    hq, hkv, q, kv, sx, sy, Br, Bc = CASES[name]
    A.SHAPES[name] = (hq, hkv, q, kv, "debug")
    c = dict(q_pad=q, kv_pad=kv, sx=sx, sy=sy, Br=Br, Bc=Bc, async_=0)
    out = A.DATA / "stage1-h1/attn_debug" / name
    out.mkdir(parents=True, exist_ok=True)
    A.write_cfg(out / "attn_cfg.py", name, c)
    rc = subprocess.run([str(A.HERE / "run_kernel.sh"), "attn", str(out / "attn_cfg.py"), str(out)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    t, bad, mu, cfgerr = A.parse_run(out)
    log = (out / "run.log").read_text(errors="replace")
    samples = [l for l in log.splitlines() if "[check] c0 row" in l][:4]
    print(f"{name:<12} rc {rc} t {t} bad {bad} max_ulp {mu} cfgerr {cfgerr}", flush=True)
    for s in samples:
        print("    " + s, flush=True)
