#!/usr/bin/env python3
"""SummaGEMM sweep for the M = 50 expert shapes with groups one or two clusters tall, splitN or splitK.

    gemm_smallm_sweep.py plan|run [--shape MxNxK] [--mode struct|tiles --struct splitK_x8y1g8] [--cand NAME]
    gemm_smallm_sweep.py pick

Why: gemm_sweep.py allowed only square SUMMA groups and no splitK, so the M = 50 shapes used 8 to 16 clusters.
Upstream's opt() (soft_hier_je gemm_auto.py, used by its DeepSeek decode flow with M = 1) makes the group one
row of m_tile = M tall and up to 8 wide, and fills the mesh with splitK or splitN groups. Non-square groups are
safe here because M_iter = 1 (M padded to scale_y * m_tile = 64): SummaGEMM.h:165 steps X along M by
summa_group_x, which only matters for M_iter > 1.

Candidates per shape: scale_y 1 (m_tile 64) or 2 (m_tile 32); scale_x 1, 2, 4, 8; n_tile, k_tile multiples of
32 up to 256; groups a power of 2 with groups * scale_x * scale_y <= 64;
  - splitN: groups own contiguous N blocks, N padded to groups * scale_x * n_tile (exact golden, 0 ULP check);
  - splitK: groups own contiguous K blocks (K padded to groups * k_tile), partial sums added by the FlooNoC
    group reduction (patch 0012) before group 0 stores; the summation order differs from the golden's replay,
    so the check is structural: |out - golden| <= ABS_TOL (GEMM_CHECK_ABS_TOL, stage-1 spec §4.3 item 6);
  - L1 double buffers within 352 KiB; padded work at most 2.5x nominal.
Two stages (each run about 20 s; all tiles of all structures would be about 4,000 runs): --mode struct runs one
tile per group structure (group shape, groups, strategy) with at least 16 active clusters; --mode tiles runs every
tile of chosen structures. The rough model is kept only as a column (est_ns), not for selection.
Raw results: results/gemm_smallm_h1.csv. Run folders: $SHVLA_DATA/stage1-h1/gemm/<shape>/<cand>/.
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

from gemm_sweep import HERE, TEMPLATE, RUNS, TILES, L1_BUDGET, WORK_CAP, up, parse_run, shape_key

CSV = HERE / "results/gemm_smallm_h1.csv"
ABS_TOL = 0.02
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
}
FIELDS = ["shape", "cand", "m_size", "n_size", "k_size", "m_tile", "n_tile", "k_tile", "scale_x", "scale_y",
          "groups", "strategy", "active", "work_ratio", "est_ns", "time_ns", "bad", "max_ulp", "status", "run_dir"]


def make_cand(M, N, K, sy, sx, nt, kt, g, strategy):
    mt = 64 // sy
    Mp = up(M, sy * mt)
    if strategy == "splitK":
        Np, Kp = up(N, sx * nt), up(K, g * kt)
        Kg = Kp // g
    else:
        Np, Kp = up(N, g * sx * nt), up(K, kt)
        Kg = Kp
    if 2 * 2 * (mt * kt + nt * kt + mt * nt) > L1_BUDGET:
        return None
    work = Mp * Np * Kp / (M * N * K)
    if work > WORK_CAP:
        return None
    Ni = Np // (sx * nt) // (g if strategy == "splitN" else 1)
    calls = Ni * (Kg // kt)
    t_comp = calls * (mt * nt * kt / 1024.0 + 150.0) + (Ni * 300.0 if strategy == "splitK" else 0.0)
    hbm = g * Ni * 2 * Kg * (sy * mt + sx * nt) + 2 * Mp * Np
    est = max(t_comp, hbm / 200.0)
    c = dict(m_size=Mp, n_size=Np, k_size=Kp, m_tile=mt, n_tile=nt, k_tile=kt, scale_x=sx, scale_y=sy, groups=g,
             strategy=strategy, active=g * sx * sy, work_ratio=round(work, 3), est_ns=int(est))
    c["cand"] = f"{strategy}_x{sx}y{sy}g{g}_t{mt}x{nt}x{kt}"
    return c


def enumerate_all(M, N, K):
    out = []
    for sy in (1, 2):
        for sx in (1, 2, 4, 8):
            for nt in TILES:
                for kt in TILES:
                    g = 2
                    while g * sx * sy <= 64:
                        for st in ("splitN", "splitK"):
                            # square groups with splitN were swept by gemm_sweep.py (1x1 splitN = its scale 1)
                            if sx == sy and st == "splitN":
                                continue
                            c = make_cand(M, N, K, sy, sx, nt, kt, g, st)
                            if c:
                                out.append(c)
                        g *= 2
    return out


def struct_key(c):
    return (c["scale_x"], c["scale_y"], c["groups"], c["strategy"])


def candidates(M, N, K, mode, min_active=16, structs=None):
    """mode 'struct': one tile per group structure (largest k_tile, then n_tile, within 6 % of the least padded work);
    mode 'tiles': every tile of the given structures."""
    out = [c for c in enumerate_all(M, N, K) if c["active"] >= min_active]
    if mode == "struct":
        groups = {}
        for c in out:
            groups.setdefault(struct_key(c), []).append(c)
        best = []
        for cs in groups.values():
            w0 = min(c["work_ratio"] for c in cs)
            best.append(max((c for c in cs if c["work_ratio"] <= w0 * 1.06),
                            key=lambda c: (c["k_tile"], c["n_tile"], -c["work_ratio"])))
        return sorted(best, key=lambda c: (-c["active"], struct_key(c)))
    return [c for c in out if struct_key(c) in structs]


def write_cfg(path, M, N, K, c):
    s = TEMPLATE.read_text()
    g, sk = c["groups"], c["strategy"] == "splitK"
    vals = dict(m_size=c["m_size"], n_size=c["n_size"], k_size=c["k_size"], m_tile=c["m_tile"],
                n_tile=c["n_tile"], k_tile=c["k_tile"], summa_scale_x=c["scale_x"], summa_scale_y=c["scale_y"],
                summa_group_number=g, summa_group_reduce=1 if sk else 0, summa_group_splitk=1 if sk else 0,
                summa_group_splitn=0 if sk else 1,
                # splitK (soft_hier_je opt()): group i reads K block i of X and W, Z offsets equal (reduced)
                summa_group_gap_x=c["k_size"] // g * 2 if sk else 0,
                summa_group_gap_w=(c["k_size"] // g * c["n_size"] * 2) if sk else c["n_size"] // g * 2,
                summa_group_gap_z=0 if sk else c["n_size"] // g * 2)
    for k, v in vals.items():
        s, n = re.subn(rf"(self\.{k}\s*=\s*)\S+", lambda m: m.group(1) + str(v), s)
        assert n == 1, k
    extra = f"        self.m_valid = {M}\n        self.n_valid = {N}\n        self.k_valid = {K}\n"
    if sk:
        extra += f"        self.check_abs_tol = {ABS_TOL}\n"
    s = s.replace("        self.dtype", extra + "        self.dtype", 1)
    path.write_text(s)


def load_csv():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--shape", action="append")
    ap.add_argument("--mode", choices=["struct", "tiles"], default="struct")
    ap.add_argument("--struct", action="append", help="tiles mode: strategy_xSXySYgG, e.g. splitK_x8y1g8")
    ap.add_argument("--cand", action="append", help="run only these candidate names")
    ap.add_argument("--min-active", type=int, default=16, help="struct/tiles mode: fewest active clusters")
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
            print(f"{shape_key(sh):>14}  {int(b['time_ns']):>9} ns  {b['cand']:<30} active {b['active']:>2}  "
                  f"({len(ok)} ok of {sum(1 for r in rows if r['shape'] == shape_key(sh))})  {SHAPES.get(sh, '')}")
        return
    done = {(r["shape"], r["cand"]) for r in load_csv()}
    total = 0
    for sh in shapes:
        M, N, K = sh
        structs = None
        if a.struct:
            structs = set()
            for t in a.struct:
                m = re.fullmatch(r"(split[NK])_x(\d+)y(\d+)g(\d+)", t)
                structs.add((int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(1)))
        cands = [c for c in enumerate_all(M, N, K) if c["cand"] in a.cand] if a.cand else \
            candidates(M, N, K, a.mode, min_active=a.min_active, structs=structs)
        for c in cands:
            total += 1
            key = (shape_key(sh), c["cand"])
            if a.cmd == "plan":
                print(f"{shape_key(sh):>14} {c['cand']:<30} pad {c['m_size']}x{c['n_size']}x{c['k_size']} "
                      f"active {c['active']:>2} work {c['work_ratio']:<5} est {c['est_ns']:>7}"
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
            new = not CSV.exists()
            with CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(dict(c, shape=shape_key(sh), time_ns=t, bad=bad, max_ulp=mu, status=status,
                                run_dir=str(out)))
            print(f"    {t} ns, bad {bad}, {status}", flush=True)
    if a.cmd == "plan":
        print(f"{total} candidates")


if __name__ == "__main__":
    main()
