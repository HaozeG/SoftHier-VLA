#!/usr/bin/env python3
"""Perfetto traces of a SoftHier run with GVSoC's own trace_perfetto scripts (step PROF.3).

The official flow (gvsoc README, `make runv pfto`) runs gvsoc with
`--trace=redmule --trace=idma --trace=cluster_registers`, then
`soft_hier/flex_cluster_utilities/trace_perfetto/parse.py` (log -> roi.json) and
`visualize.py` (roi.json -> Chrome/Perfetto JSON). `tests/gvsoc/run.py --e2e-trace` uses exactly
these three trace flags, so its per-program logs are valid input. This script runs the two official
scripts unmodified and only changes the packaging:

  build   <log[.zst]> <outdir> --name N [--ops ops.csv --program P] [--window T0:T1 --suffix S]
          writes <N>.perfetto.json.gz      (official output, re-dumped compact, asserted equal)
          and    <N>.perfetto_ops.json.gz  (the same events plus an "ops" track from ops.csv,
                                            our addition: op names on the barrier-delimited windows)
          --window keeps only events overlapping [T0, T1] ns (a readable slice of a long program).
  check   <outdir>/<N>.perfetto.json.gz <ops.csv> --program P
          recomputes per-op RedMulE busy, iDMA busy and barrier wait (cluster mean) from the Perfetto
          events and compares them with parse_profile.py's ops.csv (PROF.1).

Time base: the official JSON has ts/dur in us (ns / 1000) of absolute simulated time, the same time
base as ops.csv (start_ns, end_ns). Each program is its own gvsoc run starting at t = 0.
"""
import argparse
import bisect
import collections
import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GVSOC = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'third_party', 'gvsoc'))
TP = os.path.join(GVSOC, 'soft_hier', 'flex_cluster_utilities', 'trace_perfetto')
NCLUSTERS = 16


def official(log, tmp):
    """Run parse.py and visualize.py unmodified; return the loaded official JSON."""
    plain = log
    if log.endswith('.zst'):
        plain = os.path.join(tmp, 'trace.log')
        with open(plain, 'wb') as f:
            subprocess.run(['zstd', '-dc', log], stdout=f, check=True)
    roi = os.path.join(tmp, 'roi.json')
    out = os.path.join(tmp, 'perfetto.json')
    subprocess.run([sys.executable, os.path.join(TP, 'parse.py'), plain, roi], check=True)
    subprocess.run([sys.executable, os.path.join(TP, 'visualize.py'), roi, '-o', out], check=True)
    with open(out) as f:
        return json.load(f)


def dump_gz(obj, path):
    data = json.dumps(obj, separators=(',', ':')).encode()
    with gzip.open(path, 'wb', compresslevel=9) as f:
        f.write(data)
    with gzip.open(path) as f:
        assert json.load(f) == obj, path
    return len(data), os.path.getsize(path)


def ops_rows(path, program):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r['program'] == program and r['level'] in ('op', 'seg')]


def pct(part, whole):
    return round(100.0 * float(part) / float(whole), 1) if float(whole) > 0 else 0.0


def ops_events(rows):
    ev = [{'name': 'process_name', 'ph': 'M', 'pid': 'ops', 'args': {'name': 'ops (SoftHier-VLA overlay)'}}]
    seg = None
    for r in rows:
        t0, t1 = int(r['start_ns']), int(r['end_ns'])
        d = t1 - t0
        args = {k: r[k] for k in ('phase', 'seg', 'layer', 'step', 'class', 'bound', 'source') if r[k]}
        args.update({'dur_ns': d, 'redmule_busy_pct': pct(r['redmule_busy_ns'], d),
                     'idma_busy_pct': pct(r['idma_busy_ns'], d), 'barrier_wait_pct': pct(r['sync_wait_ns'], d),
                     'redmule_gflop': round(float(r['redmule_flops'] or 0) / 1e9, 3),
                     'hbm_read_kib': round(float(r['hbm_rd'] or 0) / 1024, 1)})
        ev.append({'name': r['op'], 'cat': r['class'], 'ph': 'X', 'ts': t0 / 1000, 'dur': d / 1000,
                   'pid': 'ops', 'tid': 'op', 'args': args})
        key = (r['seg'], r['layer'], r['step'])
        if seg and seg['key'] == key:
            seg['t1'] = t1
        else:
            if seg:
                ev.append(seg_event(seg))
            seg = {'key': key, 't0': t0, 't1': t1}
    if seg:
        ev.append(seg_event(seg))
    return ev


def seg_event(s):
    tag, layer, step = s['key']
    name = tag + (f' L{layer}' if layer else '') + (f' step{step}' if step else '')
    return {'name': name, 'ph': 'X', 'ts': s['t0'] / 1000, 'dur': (s['t1'] - s['t0']) / 1000,
            'pid': 'ops', 'tid': 'segment', 'args': {'seg': tag, 'layer': layer, 'step': step}}


def window(events, t0, t1):
    keep = []
    for e in events:
        if e.get('ph') != 'X':
            keep.append(e)
            continue
        s = e['ts'] * 1000
        if s + e['dur'] * 1000 >= t0 and s <= t1:
            keep.append(e)
    return keep


def cmd_build(a):
    os.makedirs(a.outdir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
        off = official(a.log, tmp)
    events = off['traceEvents']
    counts = collections.Counter(e['name'] for e in events if e.get('ph') == 'X')
    if a.window:
        t0, t1 = (float(x) for x in a.window.split(':'))
        events = window(events, t0, t1)
    suffix = a.suffix or ''
    res = {'counts': dict(counts)}
    base = dict(off, traceEvents=events)
    res['official'] = dump_gz(base, os.path.join(a.outdir, f'{a.name}{suffix}.perfetto.json.gz'))
    if a.ops:
        rows = ops_rows(a.ops, a.program or a.name)
        if a.window:
            rows = [r for r in rows if int(r['end_ns']) >= t0 and int(r['start_ns']) <= t1]
        res['with_ops'] = dump_gz(dict(off, traceEvents=events + ops_events(rows)),
                                  os.path.join(a.outdir, f'{a.name}{suffix}.perfetto_ops.json.gz'))
        res['op_rows'] = len(rows)
    print(json.dumps({'name': a.name + suffix, **res}))


def cmd_check(a):
    with gzip.open(a.trace) as f:
        events = json.load(f)['traceEvents']
    iv = collections.defaultdict(list)  # (cluster, agent) -> sorted [start, end] ns
    for e in events:
        if e.get('ph') == 'X' and str(e.get('pid', '')).startswith('cluster_'):
            s = round(e['ts'] * 1000)
            iv[(e['pid'], e['name'])].append((s, s + round(e['dur'] * 1000)))
    for v in iv.values():
        v.sort()
    starts = {k: [x[0] for x in v] for k, v in iv.items()}
    maxdur = {k: max(x[1] - x[0] for x in v) for k, v in iv.items()}

    def busy(agent, t0, t1):
        tot = 0
        for c in range(NCLUSTERS):
            k = (f'cluster_{c}', agent)
            if k not in iv:
                continue
            v, st = iv[k], starts[k]
            i = bisect.bisect_left(st, t0 - maxdur[k])
            while i < len(v) and v[i][0] < t1:
                tot += max(0, min(v[i][1], t1) - max(v[i][0], t0))
                i += 1
        return tot / NCLUSTERS

    worst = collections.defaultdict(lambda: (0.0, None))
    n = 0
    with open(a.ops) as f:
        for r in csv.DictReader(f):
            if r['program'] != a.program or r['level'] != 'op':
                continue
            t0, t1 = int(r['start_ns']), int(r['end_ns'])
            d = t1 - t0
            if d <= 0:
                continue
            n += 1
            for agent, col in (('redmule', 'redmule_busy_ns'), ('idma', 'idma_busy_ns'), ('cluster sync', 'sync_wait_ns')):
                diff = abs(busy(agent, t0, t1) - float(r[col])) / d * 100
                if diff > worst[agent][0]:
                    worst[agent] = (diff, f"{r['op']} L{r['layer']} s{r['step']} @{t0}")
    print(json.dumps({'program': a.program, 'ops_checked': n,
                      'max_abs_diff_pct_of_window': {k: [round(v[0], 3), v[1]] for k, v in worst.items()}}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('log')
    b.add_argument('outdir')
    b.add_argument('--name', required=True)
    b.add_argument('--ops')
    b.add_argument('--program')
    b.add_argument('--window')
    b.add_argument('--suffix')
    b.add_argument('--tmp', default=None, help='scratch dir for the plain log and official JSON')
    c = sub.add_parser('check')
    c.add_argument('trace')
    c.add_argument('ops')
    c.add_argument('--program', required=True)
    a = p.parse_args()
    {'build': cmd_build, 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    main()
