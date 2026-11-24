#!/usr/bin/env python3
"""SummaGEMM tile / group sweep on H1 for the stage-1 shapes (W1 vlabench-3cam and variant W1-b).

    gemm_sweep.py plan [--shape MxNxK] [--top N]   list the candidates that would run
    gemm_sweep.py run  [--shape MxNxK] [--top N]   run them one at a time (resumable)
    gemm_sweep.py pick                              fastest checked run per shape

Spec: docs/designs/stage1-spec.md §4.3 (items 1-6). Every run goes through run_kernel.sh (patched GVSoC, stamp check,
MALLOC_PERTURB_, output check against the bit-exact golden). A run counts only with 0 bad elements.

Candidates per shape:
  - tiles m/n/k any multiple of 32 up to 256 (RedMulE is 32 x 32); square SUMMA groups of 1, 2, 4 or 8;
  - strategy Nosplit (one group) or splitN (groups split N; exact, same golden). splitK is left out: the group
    reduction changes the summation order, so it needs its own golden first;
  - the shape is padded up to whole tiles (the kernel has no edge handling): M to scale * m_tile,
    N to groups * scale * n_tile, K to k_tile; padded work at most 2.5x nominal;
  - L1 double buffers 2 * (X + W + Z) tiles fit in 352 KiB of the 384 KiB TCDM;
  - ranked by a rough model, max(RedMulE time, HBM time) plus a per-call overhead (estimate, only to order the
    candidates), and the top N run; upstream's opt() choice (soft_hier_je gemm_auto.py, tiles 4 x RedMulE height,
    group = M/m_tile, N/n_tile rounded to a power of 2 and capped at the mesh) is always added if it is valid.
Raw results: results/gemm_sweep_h1.csv (one row per run). Run folders: $SHVLA_DATA/stage1-h1/gemm/<shape>/<cand>/.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KREPO = Path(os.environ.get("KREPO", HERE.parents[1] / "third_party/softhier-kernels"))  # kernel source (submodule)
TEMPLATE = KREPO / "implementation/config/kernels/gemm.py"
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla"))
RUNS = DATA / "stage1-h1/gemm"
CSV = HERE / "results/gemm_sweep_h1.csv"

MESH = 8
CE = 32                       # RedMulE 32 x 32
TILES = tuple(range(32, 257, 32))   # multiples of the RedMulE width
SCALES = (1, 2, 4, 8)
L1_BUDGET = 352 * 1024
WORK_CAP = 2.5

# (M, N, K): operators (smolvla_ops, release-3cam shapes = vlabench-3cam, stage-1 lowering; W1-b = cameras batched)
SHAPES = {
    (50, 32, 720): "W1 expert.action_out_proj",
    (50, 720, 32): "W1 expert.action_in_proj",
    (50, 720, 720): "W1 expert.time_mlp_out",
    (50, 720, 960): "W1 expert.o_proj",
    (50, 720, 1440): "W1 expert.time_mlp_in",
    (50, 720, 2048): "W1 expert.mlp_down",
    (50, 960, 720): "W1 expert.q_proj",
    (50, 1600, 720): "W1 expert.qkv_proj",
    (50, 4096, 720): "W1 expert.mlp_gateup",
    (192, 960, 12288): "W1 connector.connector_proj",
    (241, 320, 320): "W1 expert.cross_k/v_proj",
    (241, 320, 960): "W1 prefix.k/v_proj",
    (241, 960, 960): "W1 prefix.q/o_proj",
    (241, 960, 2560): "W1 prefix.mlp_down",
    (241, 2560, 960): "W1 prefix.mlp_gate/up",
    (1024, 768, 768): "W1 vision.patch_embed, q/k/v/o_proj",
    (1024, 768, 3072): "W1 vision.mlp_down",
    (1024, 3072, 768): "W1 vision.mlp_up",
    (3072, 768, 768): "W1-b vision.patch_embed, q/k/v/o_proj",
    (3072, 768, 3072): "W1-b vision.mlp_down",
    (3072, 3072, 768): "W1-b vision.mlp_up",
}

FIELDS = ["shape", "cand", "m_size", "n_size", "k_size", "m_tile", "n_tile", "k_tile", "scale", "groups",
          "strategy", "work_ratio", "est_ns", "upstream_opt", "time_ns", "bad", "max_ulp", "status", "run_dir"]


def up(x, a):
    return -(-x // a) * a


def pow2_up(n):
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def make_cand(M, N, K, mt, nt, kt, s, g, strategy):
    Mp, Np, Kp = up(M, s * mt), up(N, g * s * nt), up(K, kt)
    if 2 * 2 * (mt * kt + nt * kt + mt * nt) > L1_BUDGET:
        return None
    work = Mp * Np * Kp / (M * N * K)
    if work > WORK_CAP:
        return None
    # rough model (estimate, ranking only): RedMulE 1024 MAC/cycle at 1 GHz, per call overhead 150 ns,
    # HBM 2 edges x 8 channels, taken as 200 B/ns total; SUMMA traffic: per group and (m, n) step, X band + W band
    active = g * s * s
    calls = (Mp // (s * mt)) * (Np // (g * s * nt)) * (Kp // kt)
    t_comp = calls * (mt * nt * kt / 1024.0 + 150.0)
    hbm = g * (Mp // (s * mt)) * (Np // (g * s * nt)) * 2 * Kp * s * (mt + nt) + 2 * Mp * Np
    t_hbm = hbm / 200.0
    est = max(t_comp, t_hbm)
    return dict(m_size=Mp, n_size=Np, k_size=Kp, m_tile=mt, n_tile=nt, k_tile=kt, scale=s, groups=g,
                strategy=strategy, work_ratio=round(work, 3), est_ns=int(est), active=active)


def upstream_opt(M, N, K):
    """soft_hier_je opt() Nosplit / splitN choice for the padded shape (it asserts divisibility; we pad)."""
    t = 4 * CE
    mt, nt = min(t, M), min(t, N)
    sy = min(pow2_up(-(-M // mt)), MESH)
    sx = min(pow2_up(-(-N // nt)), MESH)
    if sx != sy:
        return None           # this kernel needs square groups (X_tile_M_iter_offset uses summa_group_x)
    return (mt, nt, min(t, K), sx)


def candidates(M, N, K, top):
    out = []
    for mt in TILES:
        for nt in TILES:
            for kt in TILES:
                for s in SCALES:
                    c = make_cand(M, N, K, mt, nt, kt, s, 1, "Nosplit")
                    if c:
                        out.append(c)
                    g = 2
                    while g * s * s <= MESH * MESH:
                        c = make_cand(M, N, K, mt, nt, kt, s, g, "splitN")
                        if c and c["n_size"] // (g * s * nt) >= 1:
                            out.append(c)
                        g *= 2
    out.sort(key=lambda c: (c["est_ns"], c["work_ratio"], -c["k_tile"]))
    # the model does not separate K tiles, so pick distinct (m_tile, n_tile, scale, groups) combinations first
    # (each with its best K tile), then add K-tile variants of the best combination
    picked, seen = [], set()
    for c in out:
        key = (c["m_tile"], c["n_tile"], c["scale"], c["groups"])
        if key not in seen:
            seen.add(key); picked.append(c)
        if len(picked) == top:
            break
    if picked:
        b = picked[0]
        for c in out:
            if (c["m_tile"], c["n_tile"], c["scale"], c["groups"]) == (b["m_tile"], b["n_tile"], b["scale"], b["groups"]) \
                    and c["k_tile"] in (64, 128, 256) and c not in picked:
                picked.append(c)
    # keep the best of each scale in the list, so the sweep does not rest on the model's scale preference alone
    for s in SCALES:
        best = next((c for c in out if c["scale"] == s), None)   # best of each scale, if not in yet
        if best and best not in picked:
            picked.append(best)
    u = upstream_opt(M, N, K)
    if u:
        mt, nt, kt, s = u
        c = make_cand(M, N, K, mt, nt, kt, s, 1, "Nosplit") or None
        if c:
            c["upstream_opt"] = 1
            if c not in picked:
                picked.append(c)
            else:
                picked[picked.index(c)]["upstream_opt"] = 1
    for c in picked:
        c.setdefault("upstream_opt", 0)
        c["cand"] = f"{c['strategy']}_s{c['scale']}g{c['groups']}_t{c['m_tile']}x{c['n_tile']}x{c['k_tile']}"
    return picked


def write_cfg(path, M, N, K, c):
    s = TEMPLATE.read_text()
    vals = dict(m_size=c["m_size"], n_size=c["n_size"], k_size=c["k_size"], m_tile=c["m_tile"],
                n_tile=c["n_tile"], k_tile=c["k_tile"], summa_scale_x=c["scale"], summa_scale_y=c["scale"],
                summa_group_number=c["groups"], summa_group_reduce=0, summa_group_splitk=0,
                summa_group_splitn=1 if c["strategy"] == "splitN" else 0,
                summa_group_gap_x=0,
                # splitN: each group owns a contiguous block of N_pad / groups columns. The kernel's N loop steps
                # by scale * n_tile inside a group (Z/W_tile_N_iter_offset), so the upstream spacing of
                # scale * n_tile between groups (soft_hier_je opt()) is right only with one N step per group.
                summa_group_gap_w=c["n_size"] // c["groups"] * 2 if c["strategy"] == "splitN" else 0,
                summa_group_gap_z=c["n_size"] // c["groups"] * 2 if c["strategy"] == "splitN" else 0)
    for k, v in vals.items():
        s, n = re.subn(rf"(self\.{k}\s*=\s*)\S+", lambda m: m.group(1) + str(v), s)
        assert n == 1, k
    s = s.replace("        self.dtype", f"        self.m_valid = {M}\n        self.n_valid = {N}\n"
                  f"        self.k_valid = {K}\n        self.dtype", 1)
    path.write_text(s)


def load_csv():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def append_csv(row):
    new = not CSV.exists()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def parse_run(out_dir):
    log = (out_dir / "run.log").read_text(errors="replace") if (out_dir / "run.log").exists() else ""
    t = re.search(r"Execution period is (\d+) ns", log)
    bad, mu, n = 0, 0, 0
    for m in re.finditer(r"\[check\] cluster \d+: rows \d+, bad (\d+) \(nan \d+\), max_ulp (\d+)", log):
        bad += int(m.group(1)); mu = max(mu, int(m.group(2))); n += 1
    return (int(t.group(1)) if t else None), (bad if n else None), (mu if n else None)


def shape_key(sh):
    return "x".join(map(str, sh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--shape", action="append")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    shapes = sorted(SHAPES, key=lambda s: s[0] * s[1] * s[2])
    if a.shape:
        shapes = [tuple(map(int, x.split("x"))) for x in a.shape]

    if a.cmd == "pick":
        rows = load_csv()
        for sh in shapes:
            ok = [r for r in rows if r["shape"] == shape_key(sh) and r["status"] == "ok"]
            if not ok:
                print(f"{shape_key(sh):>14}  no checked run"); continue
            b = min(ok, key=lambda r: int(r["time_ns"]))
            n_all = sum(1 for r in rows if r["shape"] == shape_key(sh))
            print(f"{shape_key(sh):>14}  {int(b['time_ns']):>9} ns  {b['cand']:<32} work {b['work_ratio']}x  "
                  f"({len(ok)} ok of {n_all})  {SHAPES.get(sh, '')}")
        return

    done = {(r["shape"], r["cand"]) for r in load_csv()}
    total = 0
    for sh in shapes:
        M, N, K = sh
        cands = candidates(M, N, K, a.top)
        for c in cands:
            total += 1
            key = (shape_key(sh), c["cand"])
            if a.cmd == "plan":
                print(f"{shape_key(sh):>14} {c['cand']:<32} pad {c['m_size']}x{c['n_size']}x{c['k_size']} "
                      f"work {c['work_ratio']:<5} est {c['est_ns']:>8} ns  opt {c['upstream_opt']}"
                      f"{'  done' if key in done else ''}")
                continue
            if key in done:
                continue
            out = RUNS / shape_key(sh) / c["cand"]
            out.mkdir(parents=True, exist_ok=True)
            cfg = out / "gemm_cfg.py"
            write_cfg(cfg, M, N, K, c)
            print(f"[{shape_key(sh)}] {c['cand']} ...", flush=True)
            rc = subprocess.run([str(HERE / "run_kernel.sh"), "gemm", str(cfg), str(out)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
            t, bad, mu = parse_run(out)
            status = "ok" if (rc == 0 and t is not None and bad == 0) else \
                     ("timeout_or_fail" if t is None else "wrong")
            row = dict(c, shape=shape_key(sh), time_ns=t, bad=bad, max_ulp=mu, status=status, run_dir=str(out))
            append_csv(row)
            print(f"    {t} ns, bad {bad}, {status}", flush=True)
    if a.cmd == "plan":
        print(f"{total} candidates")


if __name__ == "__main__":
    main()
