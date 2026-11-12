#!/usr/bin/env python3
"""HBM traffic per edge port, from iDMA trace lines (step PROF.3, hardware overview).

Usage: python -I hbm_ports.py <run-or-profile dir> <ops.csv> [op ...]
For each program log in the directory and each listed op (default: a fixed set of large GEMMs), sums the
bytes (size x repeats) of the iDMA transactions that finished inside the op's windows, by HBM node.
HBM address map (flex_cluster.py:277-343, hbm_start_base 0xc0000000, hbm_node_addr_space 64 MiB,
hbm_node_aliase 1): nodes 0-3 west edge (one per mesh row), 4-7 north, 8-11 east, 12-15 south (one per
column). With hbm_chan_placement [4, 0, 0, 4] only the west and south nodes have an HBM channel.
A transaction is attributed to the node of its start address (a 2-D transfer that crosses a 64 MiB
boundary would be split wrongly; the per-node share is approximate in that case).
"""
import bisect
import collections
import csv
import io
import os
import re
import subprocess
import sys

BASE, NODE = 0xc0000000, 0x04000000
EDGE = ['W'] * 4 + ['N'] * 4 + ['E'] * 4 + ['S'] * 4
RX = re.compile(r'Finished : (\d+) ns ---> (\d+) ns')
# One trace line can list many transactions; same pattern as parse_profile.py _TXN.
TXN = re.compile(r"Txn \d+ = \{type: (1D|2D), src: 0x([0-9a-f]+)(?:, src_stride: 0x[0-9a-f]+)?, dst: 0x([0-9a-f]+)"
                 r"(?:, dst_stride: 0x[0-9a-f]+, repeats: 0x([0-9a-f]+))?, size: 0x([0-9a-f]+)")
DEFAULT = ['mlp_gate', 'mlp_up', 'mlp_down', 'mlp_gateup', 'gemm_qkv', 'gemm_o', 'gemm_q']


def lines(path):
    if path.endswith('.zst'):
        out = subprocess.run(['zstd', '-dc', path], capture_output=True, check=True).stdout
        return io.StringIO(out.decode(errors='replace'))
    return open(path, errors='replace')


def node_name(n):
    return f'{EDGE[n]}{n % 4}'


def main():
    args = sys.argv[1:]
    out_csv = None
    if '--csv' in args:  # --csv <path>: per-port totals over all listed ops and programs
        i = args.index('--csv')
        out_csv = args[i + 1]
        del args[i:i + 2]
    d, ops_csv = args[0], args[1]
    want = set(args[2:]) or set(DEFAULT)
    total = collections.Counter()  # node -> bytes, all ops and programs
    total_rd = collections.Counter()
    total_ns = 0
    rows = [r for r in csv.DictReader(open(ops_csv)) if r['level'] == 'op' and r['op'] in want]
    for prog in ('vision', 'prefix_a', 'prefix_b', 'expert'):
        log = next((os.path.join(d, f) for f in (f'{prog}.log.zst', f'{prog}.log') if os.path.exists(os.path.join(d, f))), None)
        pr = sorted((int(r['start_ns']), int(r['end_ns']), r['op']) for r in rows if r['program'] == prog)
        if not log or not pr:
            continue
        starts = [p[0] for p in pr]
        acc = collections.defaultdict(collections.Counter)  # op -> node -> bytes
        rdwr = collections.defaultdict(collections.Counter)  # op -> (node, rd|wr) -> bytes
        dur = collections.Counter()
        for p in pr:
            dur[p[2]] += p[1] - p[0]
        for ln in lines(log):
            if 'idma/fe/trace' not in ln:
                continue
            m = RX.search(ln)
            if not m:
                continue
            t1 = int(m.group(2))
            i = bisect.bisect_right(starts, t1 - 1) - 1
            if i < 0 or not (pr[i][0] < t1 <= pr[i][1]):
                continue
            for t in TXN.finditer(ln):
                src, dst = int(t.group(2), 16), int(t.group(3), 16)
                nbytes = int(t.group(5), 16) * int(t.group(4) or '1', 16)
                if src >= BASE:  # HBM read: data leaves the HBM port towards the mesh
                    acc[pr[i][2]][(a := (src - BASE) // NODE)] += nbytes
                    rdwr[pr[i][2]][(a, 'rd')] += nbytes
                if dst >= BASE:  # HBM write: data enters the HBM port from the mesh
                    acc[pr[i][2]][(a := (dst - BASE) // NODE)] += nbytes
                    rdwr[pr[i][2]][(a, 'wr')] += nbytes
        total_ns += sum(dur.values())
        for op in acc:
            total.update(acc[op])
            total_rd.update({n: b for (n, k), b in rdwr[op].items() if k == 'rd'})
        for op in sorted(acc, key=lambda o: -sum(acc[o].values())):
            c = acc[op]
            tot = sum(c.values())
            share = ' '.join(f'{node_name(n)}:{100 * b / tot:.0f}%' for n, b in sorted(c.items(), key=lambda x: -x[1]))
            print(f'{prog:9s} {op:12s} {tot / 2**20:9.1f} MiB  {tot / dur[op]:6.1f} GB/s  ports used {len(c):2d}  {share}')
            hot = max(c, key=c.get)
            print(f'{"":22s} hottest port {node_name(hot)}: read {rdwr[op][(hot, "rd")] / dur[op]:5.1f} GB/s, '
                  f'write {rdwr[op][(hot, "wr")] / dur[op]:5.1f} GB/s (mean over the op windows; link 64 GB/s per direction)')
    if out_csv:
        tot = sum(total.values())
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['port', 'edge', 'has_channel', 'share_pct', 'gib', 'mean_read_gbs'])
            for n in list(range(4)) + list(range(12, 16)) + list(range(4, 12)):
                has = n < 4 or n >= 12
                w.writerow([node_name(n), EDGE[n], int(has), round(100 * total[n] / tot, 1),
                            round(total[n] / 2**30, 3), round(total_rd[n] / total_ns, 1) if total_ns else 0])
        print(f'wrote {out_csv}: {tot / 2**30:.2f} GiB over {total_ns / 1e6:.2f} ms of GEMM windows')


if __name__ == '__main__':
    main()
