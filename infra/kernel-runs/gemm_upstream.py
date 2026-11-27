#!/usr/bin/env python3
"""Upstream's own GEMM settings on H1: soft_hier_je opt() (implementation/scripts/kernels/auto_config/gemm_auto.py:46-166,
gvsoc/gvsoc commit 4610405), ported rule for rule, run on every stage-1 GEMM shape.

    gemm_upstream.py plan | run | pick   [--shape MxNxK]

opt() emits up to two settings per shape and upstream's engine (softhier_engine.py:129-245) simulates both and keeps
the faster; there is no cost model upstream. The rules:
  1. m_tile = n_tile = 4 * redmule_ce_height (128 on H1), capped at M and N;
  2. scale_y = M / m_tile, scale_x = N / n_tile; a scale that is not a power of 2 is rounded up to one and the tile
     shrunk to M / scale (N / scale); then each scale is capped at the mesh (8);
  3. splitK: groups = the largest power of 2 <= free groups (64 / (scale_x * scale_y)) that divides K;
     k_tile = K / groups if that is <= 128, else max(largest divisor of K / groups that is <= 128, 96) when
     K / groups < 4 * 128, else 128. With one group the setting is Nosplit (k_tile 128);
  4. splitN (only if free groups > 1 and N / (scale_x * n_tile) > 1): groups = min(free groups,
     N / (scale_x * n_tile)), k_tile = 128 (or K if smaller).
Where opt() asserts divisibility (M % m_tile, N % n_tile, after the rescale, K % k_tile), the shape is padded
instead, as gemm_sweep.py does. Group spacing is gemm_smallm_sweep.write_cfg's (splitN: groups own contiguous N
blocks; upstream's spacing of scale_x * n_tile is right only with one N step per group). Non-square groups with
more than one M step are not run: SummaGEMM.h:165 steps X along M by summa_group_x (status skipped_m_iter).
A setting identical to a gemm_sweep.py run reuses that run (same run_dir).
Raw results: results/gemm_upstream_h1.csv. Run folders: $SHVLA_DATA/stage1-h1/gemm/<shape>/<cand>/.
"""
import argparse
import csv
import subprocess
from pathlib import Path

from gemm_sweep import HERE, RUNS, SHAPES, MESH, CE, up, pow2_up, parse_run, shape_key, model_ns
import gemm_smallm_sweep as S
import gemm_sweep as G

CSV = HERE / "results/gemm_upstream_h1.csv"
FIELDS = S.FIELDS


def _is_pow2(x):
    return x > 0 and x & (x - 1) == 0


def _max_divisor(n, cap):
    return next(d for d in range(cap, 0, -1) if n % d == 0)


def opt(M, N, K):
    """soft_hier_je opt() on H1 (rules in the module docstring); returns its settings as candidate dicts."""
    t = 4 * CE
    mt, nt = min(t, M), min(t, N)
    Mp, Np = up(M, mt), up(N, nt)
    sy, sx = Mp // mt, Np // nt
    if not _is_pow2(sx):
        sx = pow2_up(sx); nt = -(-Np // sx); Np = sx * nt
    if not _is_pow2(sy):
        sy = pow2_up(sy); mt = -(-Mp // sy); Mp = sy * mt
    sy, sx = min(sy, MESH), min(sx, MESH)
    free = MESH * MESH // (sx * sy)
    out = []
    g = 1
    while g * 2 <= free and K % (g * 2) == 0:
        g *= 2
    if g > 1:
        ks = K // g
        kt = ks if ks <= t else (max(_max_divisor(ks, t), t * 3 // 4) if ks // t < 4 else t)
        out.append(_cand(M, N, K, Mp, Np, up(K, g * kt), mt, nt, kt, sx, sy, g, "splitK"))
    else:
        kt = min(t, K)
        out.append(_cand(M, N, K, Mp, Np, up(K, kt), mt, nt, kt, sx, sy, 1, "Nosplit"))
    if free > 1 and Np // (sx * nt) > 1:
        kt = min(t, K)
        g = min(free, Np // (sx * nt))
        out.append(_cand(M, N, K, Mp, up(Np, g * sx * nt), up(K, kt), mt, nt, kt, sx, sy, g, "splitN"))
    return out


def _cand(M, N, K, Mp, Np, Kp, mt, nt, kt, sx, sy, g, st):
    Mp = up(Mp, sy * mt)
    Np = up(Np, sx * nt * (g if st == "splitN" else 1))
    c = dict(m_size=Mp, n_size=Np, k_size=Kp, m_tile=mt, n_tile=nt, k_tile=kt, scale_x=sx, scale_y=sy, groups=g,
             strategy=st, active=g * sx * sy, work_ratio=round(Mp * Np * Kp / (M * N * K), 3),
             est_ns=int(model_ns(Mp, Np, Kp, mt, nt, kt, sx, sy, g, st)))
    c["cand"] = f"{st}_x{sx}y{sy}g{g}_t{mt}x{nt}x{kt}"
    c["m_iter"] = Mp // (sy * mt)
    return c


def _same_run(shape, c):
    """The gemm_sweep.py run of exactly this setting (square groups only), if any."""
    if c["scale_x"] != c["scale_y"] or c["strategy"] == "splitK" or not G.CSV.exists():
        return None
    keys = ("m_size", "n_size", "k_size", "m_tile", "n_tile", "k_tile", "groups", "strategy")
    with G.CSV.open() as f:
        for r in csv.DictReader(f):
            if r["shape"] == shape and int(r["scale"]) == c["scale_x"] and all(str(c[k]) == r[k] for k in keys):
                return r
    return None


def load_csv():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--shape", action="append")
    a = ap.parse_args()
    shapes = sorted(SHAPES, key=lambda s: s[0] * s[1] * s[2])
    if a.shape:
        shapes = [tuple(map(int, x.split("x"))) for x in a.shape]
    rows = load_csv()
    if a.cmd == "pick":
        for sh in shapes:
            ok = [r for r in rows if r["shape"] == shape_key(sh) and r["status"] == "ok"]
            b = min(ok, key=lambda r: int(r["time_ns"])) if ok else None
            print(f"{shape_key(sh):>14}  " + (f"{int(b['time_ns']):>9} ns  {b['cand']}" if b else "no checked run"))
        return
    done = {(r["shape"], r["cand"]) for r in rows}
    for sh in shapes:
        M, N, K = sh
        for c in opt(M, N, K):
            key = (shape_key(sh), c["cand"])
            skip = c["scale_x"] != c["scale_y"] and c["m_iter"] > 1
            if a.cmd == "plan":
                print(f"{shape_key(sh):>14} {c['cand']:<30} pad {c['m_size']}x{c['n_size']}x{c['k_size']} "
                      f"active {c['active']:>2} work {c['work_ratio']:<5} est {c['est_ns']:>7}"
                      f"{'  skipped (non-square groups, M_iter > 1)' if skip else ''}{'  done' if key in done else ''}")
                continue
            if key in done:
                continue
            row = dict(c, shape=shape_key(sh))
            same = _same_run(shape_key(sh), c)
            if skip:
                row.update(status="skipped_m_iter")
            elif same:              # the identical setting already ran in gemm_sweep.py: reuse that run
                row.update({k: same[k] for k in ("time_ns", "bad", "max_ulp", "status", "run_dir")})
                print(f"[{shape_key(sh)}] {c['cand']} = gemm_sweep {same['cand']}: {same['time_ns']} ns", flush=True)
            else:
                out = RUNS / shape_key(sh) / c["cand"]
                out.mkdir(parents=True, exist_ok=True)
                cfg = out / "gemm_cfg.py"
                S.write_cfg(cfg, M, N, K, c)
                print(f"[{shape_key(sh)}] {c['cand']} ...", flush=True)
                rc = subprocess.run([str(HERE / "run_kernel.sh"), "gemm", str(cfg), str(out)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
                t, bad, mu = parse_run(out)
                row.update(time_ns=t, bad=bad, max_ulp=mu, run_dir=str(out),
                           status="ok" if (rc == 0 and t is not None and bad == 0) else
                           ("timeout_or_fail" if t is None else "wrong"))
                print(f"    {t} ns, bad {bad}, {row['status']}", flush=True)
            new = not CSV.exists()
            with CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
