#!/usr/bin/env python3
"""VLM GEMMs confined to the first R rows of the mesh (sub-mesh footprint for VLM / expert overlap), on H1.

    gemm_submesh_sweep.py plan|run [--shape MxNxK] [--rows 7 6] [--top 2]
    gemm_submesh_sweep.py pick

Why: the overlap estimate (async-vlm-expert-overlap.md, open measurement 1) assumes the VLM slows by
64 / (64 - c) when c clusters run the expert's stale steps. SummaGEMM places groups from cluster (0, 0) in id
order (grid_sync_group_init: id = (8 / s) (y / s) + x / s; ids below the group count are active), so a setting
whose active groups lie in rows 0 .. R-1 is a real measurement of the VLM GEMM on that footprint, with row R..7
idle. gemm_sweep.py tried power-of-two group counts only, which reach 32 clusters (rows 0-3) at most; here the
group count is any integer, so square groups fill rows 0-5 (scale 2, 12 groups) or 0-6 (scale 1, up to 56).
Square groups only: these shapes have M_iter > 1 (SummaGEMM.h:165 steps X along M by summa_group_x).
Candidates per (shape, footprint, scale): every group count that fits, its tile ranked first by gemm_sweep's rough
model; per scale the --top group counts with the least padded work per active cluster run. Exact golden check as in gemm_sweep.py (splitN, 0 ULP).
Raw results: results/gemm_submesh_h1.csv. Run folders: $SHVLA_DATA/stage1-h1/gemm/<shape>/<cand>/.
"""
import argparse
import csv
import subprocess

from gemm_sweep import HERE, RUNS, TILES, MESH, make_cand, write_cfg, parse_run, shape_key

CSV = HERE / "results/gemm_submesh_h1.csv"
SHAPES = {
    (1024, 768, 768): "W1 vision.patch_embed, q/k/v/o_proj",
    (1024, 768, 3072): "W1 vision.mlp_down",
    (1024, 3072, 768): "W1 vision.mlp_up",
    (241, 960, 960): "W1 prefix.q/o_proj",
    (241, 960, 2560): "W1 prefix.mlp_down",
    (241, 2560, 960): "W1 prefix.mlp_gate/up",
    (241, 320, 960): "W1 prefix.k/v_proj",
    (192, 960, 12288): "W1 connector.connector_proj",
}
FIELDS = ["shape", "rows", "cand", "m_size", "n_size", "k_size", "m_tile", "n_tile", "k_tile", "scale", "groups",
          "strategy", "active", "work_ratio", "est_ns", "time_ns", "bad", "max_ulp", "status", "run_dir"]


def rows_used(s, g):
    return -(-g // (MESH // s)) * s


def candidates(M, N, K, R, top):
    out = []
    for s in (1, 2):   # scale 4 fits only 2 groups (rows 0-3): measured by gemm_sweep.py
        cs = []
        gmax = (MESH // s) * (R // s)
        for g in range(1, gmax + 1):
            for mt in TILES:
                for nt in TILES:
                    for kt in TILES:
                        c = make_cand(M, N, K, mt, nt, kt, s, g, "splitN" if g > 1 else "Nosplit")
                        if c and c["n_size"] // (g * s * nt) >= 1:
                            cs.append(c)
        # group count: the top --top by padded work per active cluster (the rough model's HBM term favours few
        # groups, which defeats the point); within a group count, the tile the rough model ranks first
        best = {}
        for c in sorted(cs, key=lambda c: (c["est_ns"], c["work_ratio"], -c["k_tile"])):
            best.setdefault(c["groups"], c)
        out += sorted(best.values(), key=lambda c: (c["work_ratio"] / c["active"], -c["active"]))[:top]
    for c in out:
        c["cand"] = f"{c['strategy']}_s{c['scale']}g{c['groups']}_t{c['m_tile']}x{c['n_tile']}x{c['k_tile']}"
        c["rows"] = rows_used(c["scale"], c["groups"])
    return out


def load_csv():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--shape", action="append")
    ap.add_argument("--rows", type=int, nargs="+", default=[7, 6])
    ap.add_argument("--top", type=int, default=2)
    a = ap.parse_args()
    shapes = [tuple(map(int, x.split("x"))) for x in a.shape] if a.shape else list(SHAPES)
    if a.cmd == "pick":
        rows = [r for r in load_csv() if r["status"] == "ok"]
        for sh in shapes:
            for R in a.rows:
                ok = [r for r in rows if r["shape"] == shape_key(sh) and int(r["rows"]) <= R]
                if ok:
                    b = min(ok, key=lambda r: int(r["time_ns"]))
                    print(f"{shape_key(sh):>14} rows<={R}  {int(b['time_ns']):>9} ns  {b['cand']:<30} "
                          f"active {b['active']}")
        return
    done = {(r["shape"], r["cand"]) for r in load_csv()}
    # settings gemm_sweep.py already ran (power-of-two group counts) are read from its CSV, not re-run
    with (HERE / "results/gemm_sweep_h1.csv").open() as f:
        done |= {(r["shape"], r["cand"]) for r in csv.DictReader(f)}
    cands, seen = [], set()
    for sh in shapes:
        for R in sorted(a.rows, reverse=True):
            for c in candidates(*sh, R, a.top):
                if (sh, c["cand"]) not in seen:
                    seen.add((sh, c["cand"])); cands.append((sh, c))
    for sh, c in cands:
        key = (shape_key(sh), c["cand"])
        if a.cmd == "plan":
            print(f"{shape_key(sh):>14} {c['cand']:<30} rows {c['rows']} active {c['active']:>2} "
                  f"work {c['work_ratio']:<5} est {c['est_ns']:>8}{'  done' if key in done else ''}")
            continue
        if key in done:
            continue
        out = RUNS / shape_key(sh) / c["cand"]
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / "gemm_cfg.py"
        write_cfg(cfg, *sh, c)
        print(f"[{shape_key(sh)}] {c['cand']} ...", flush=True)
        rc = subprocess.run([str(HERE / "run_kernel.sh"), "gemm", str(cfg), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
        t, bad, mu = parse_run(out)
        status = "ok" if (rc == 0 and t is not None and bad == 0) else ("timeout_or_fail" if t is None else "wrong")
        new = not CSV.exists()
        with CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(dict(c, shape=shape_key(sh), time_ns=t, bad=bad, max_ulp=mu, status=status, run_dir=str(out)))
        print(f"    {t} ns, bad {bad}, {status}", flush=True)
    if a.cmd == "plan":
        print(f"{len(cands)} candidates")


if __name__ == "__main__":
    main()
