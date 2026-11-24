#!/usr/bin/env python3
"""FlatAttention group / block sweep on H1 for the stage-1 attention shapes (unmasked, padded).

    attn_sweep.py plan | run | pick   [--shape NAME] [--top N]

Valid settings, read from the kernel (FlatAttentionUtil.h, flat_attention_analyze and the Spatz routines):
  - groups G = 64 / (sx * sy); a group takes floor(H / G) KV heads (H < G: one head each, extra groups idle), so
    H % G != 0 with H > G silently drops heads: such settings are skipped;
  - Br_s = Br / sy rows per cluster, split over the 4 Spatz units in steps of 16 rows: Br_s a multiple of 64;
  - Bc_s = Bc / sx columns per cluster, one vector op per row at e16/m8 (VLEN 256): Bc_s <= 128;
  - sequences padded to block multiples (Tr = Sq / Br, Tc = Skv / Bc are floor divisions): the query rows of one
    KV head (q * heads per KV head) to Br, the keys to Bc; padded score work at most 2.5x;
  - L1 double buffers (Q, K^T, V, O, A, row vectors) within 352 KiB;
  - async mode only with an even head count per group (upstream note), else collective mode.
The output check compares with a float64 golden of the same padded, unmasked computation (abs tolerance, see
FlatAttention main.c). Raw results: results/attn_sweep_h1.csv; run folders $SHVLA_DATA/stage1-h1/attn/<shape>/<cand>/.
"""
import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
KREPO = Path(os.environ.get("KREPO", HERE.parents[1] / "third_party/softhier-kernels"))  # kernel source (submodule)
TEMPLATE = KREPO / "implementation/config/kernels/attn.py"
DATA = Path(os.environ.get("SHVLA_DATA", Path.home() / ".cache/softhier-vla"))
RUNS = DATA / "stage1-h1/attn"
CSV = HERE / "results/attn_sweep_h1.csv"
D = 64
L1_BUDGET = 352 * 1024
WORK_CAP = 2.5

# name: (query heads, KV heads, q length, kv length, operators). W1 and W1-b share these (attention per camera).
SHAPES = {
    "12x1024x1024x64": (12, 12, 1024, 1024, "vision qk, softmax, pv (per camera)"),
    "15x241x241x64": (15, 5, 241, 241, "prefix qk, softmax, pv"),
    "15x50x291x64": (15, 5, 50, 291, "expert self qk, softmax, pv"),
    "15x50x241x64": (15, 5, 50, 241, "expert cross qk, softmax, pv"),
}
FIELDS = ["shape", "cand", "q_pad", "kv_pad", "sx", "sy", "Br", "Bc", "async", "work_ratio", "time_ns", "bad",
          "max_ulp", "status", "run_dir"]


def up(x, a):
    return -(-x // a) * a


def candidates(name):
    hq, hkv, q, kv, _ = SHAPES[name]
    r = hq // hkv
    out = []
    for sx in (1, 2, 4, 8):
        for sy in (1, 2, 4, 8):
            G = 64 // (sx * sy)
            if hkv > G and hkv % G:
                continue
            per_group = max(hkv // G, 1)
            for brs in (64, 128, 192, 256):
                for bcs in (32, 64, 96, 128):
                    Br, Bc = brs * sy, bcs * sx
                    qp = q
                    while (qp * r) % Br:
                        qp += 1
                    kvp = up(kv, Bc)
                    work = (qp * kvp) / (q * kv)
                    if work > WORK_CAP:
                        continue
                    l1 = 2 * 2 * (2 * brs * D + 2 * bcs * D + brs * bcs) + 10 * brs * 2
                    if l1 > L1_BUDGET:
                        continue
                    for asy in ((1, 0) if per_group % 2 == 0 else (0,)):
                        out.append(dict(q_pad=qp, kv_pad=kvp, sx=sx, sy=sy, Br=Br, Bc=Bc, async_=asy,
                                        work_ratio=round(work, 3), active=min(G, hkv) * sx * sy))
    # order: padded work per busy cluster (a rough time proxy), then larger per-cluster blocks
    out.sort(key=lambda c: (round(c["work_ratio"] / c["active"], 5), -(c["Br"] // c["sy"]) * (c["Bc"] // c["sx"])))
    for c in out:
        c["cand"] = f"{'async' if c['async_'] else 'coll'}_s{c['sx']}x{c['sy']}_b{c['Br']}x{c['Bc']}"
    return out


def write_cfg(path, name, c):
    hq, hkv, q, kv, _ = SHAPES[name]
    s = TEMPLATE.read_text()
    vals = dict(kv_sequence_length=c["kv_pad"], q_sequence_length=c["q_pad"], speculative_length=1,
                head_dimemsion=D, num_head=hq, num_head_group=hkv, batch_size=1, flatten_scale_x=c["sx"],
                flatten_scale_y=c["sy"], flatten_shape_x=c["Bc"], flatten_shape_y=c["Br"], flatten_async=c["async_"])
    for k, v in vals.items():
        s, n = re.subn(rf"(self\.{k}\s*=\s*)\S+", lambda m: m.group(1) + str(v), s)
        assert n == 1, k
    path.write_text(s)


def parse_run(out):
    log = (out / "run.log").read_text(errors="replace") if (out / "run.log").exists() else ""
    t = re.search(r"Execution period is (\d+) ns", log)
    bad, mu, n = 0, 0, 0
    for m in re.finditer(r"\[check\] cluster \d+: rows \d+, bad (\d+) \(nan \d+\), max_ulp (\d+)", log):
        bad += int(m.group(1)); mu = max(mu, int(m.group(2))); n += 1
    return (int(t.group(1)) if t else None), (bad if n else None), (mu if n else None), \
        ("Configuration Error" in log)


def rows():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "pick"])
    ap.add_argument("--shape", action="append")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    names = a.shape or list(SHAPES)
    if a.cmd == "pick":
        rs = rows()
        for n in names:
            ok = [r for r in rs if r["shape"] == n and r["status"] == "ok"]
            if not ok:
                print(f"{n:>16}  no checked run"); continue
            b = min(ok, key=lambda r: int(r["time_ns"]))
            print(f"{n:>16}  {int(b['time_ns']):>9} ns  {b['cand']:<24} pad q {b['q_pad']} kv {b['kv_pad']} "
                  f"work {b['work_ratio']}x  ({len(ok)} ok of {sum(1 for r in rs if r['shape'] == n)})")
        return
    if a.cmd == "run" and (HERE / "ATTN_HOLD").exists():
        print(f"held: {HERE / 'ATTN_HOLD'} exists (FlatAttention output wrong, debugging)"); return
    done = {(r["shape"], r["cand"]) for r in rows()}
    total = 0
    for n in names:
        for c in candidates(n)[:a.top]:
            total += 1
            if a.cmd == "plan":
                print(f"{n:>16} {c['cand']:<24} pad q {c['q_pad']:<5} kv {c['kv_pad']:<5} work {c['work_ratio']:<6} "
                      f"active {c['active']}{'  done' if (n, c['cand']) in done else ''}")
                continue
            if (n, c["cand"]) in done:
                continue
            out = RUNS / n / c["cand"]
            out.mkdir(parents=True, exist_ok=True)
            cfg = out / "attn_cfg.py"
            write_cfg(cfg, n, c)
            print(f"[{n}] {c['cand']} ...", flush=True)
            rc = subprocess.run([str(HERE / "run_kernel.sh"), "attn", str(cfg), str(out)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
            t, bad, mu, cfgerr = parse_run(out)
            status = "config_error" if cfgerr else ("ok" if (rc == 0 and t is not None and bad == 0) else
                                                   ("timeout_or_fail" if t is None else "wrong"))
            new = not CSV.exists()
            CSV.parent.mkdir(parents=True, exist_ok=True)
            with CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(dict(c, shape=n, async_=None, **{"async": c["async_"]}, time_ns=t, bad=bad,
                                max_ulp=mu, status=status, run_dir=str(out)))
            print(f"    {t} ns, bad {bad}, {status}", flush=True)
    if a.cmd == "plan":
        print(f"{total} candidates")


if __name__ == "__main__":
    main()
