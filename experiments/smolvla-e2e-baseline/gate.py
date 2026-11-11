#!/usr/bin/env python3
"""G3 validity gate (design Component 6, file 34).

    gate.py g3 <run folder>

Reads <run folder>/summary.json (written by `run.py smolvla-e2e`):
  * total simulated time `segments_ms.total` must be within +-5 % of 119.1 ms (softhier-mlir's published figure);
  * action max-abs error `acc.actions.max_abs` (computed by run.py against the --ref npz, i.e. G2) must be <= 0.10
    (published: 0.032).
Prints both values and PASS or MISS; exit 0 on PASS, 1 on MISS, 2 on usage / missing data.

`gate.py g4a ...` (the G4a composition) is not implemented yet.
"""
import json
import os
import sys

PUBLISHED_MS = 119.1
TOL = 0.05
ACTION_MAX_ABS = 0.10


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def g3(folder):
    path = os.path.join(folder, 'summary.json')
    if not os.path.isfile(path):
        print(f'gate.py g3: {path} not found', file=sys.stderr)
        return 2
    s = json.load(open(path))
    total = dig(s, 'segments_ms', 'total')
    err = dig(s, 'acc', 'actions', 'max_abs')
    if total is None or err is None:
        print(f'gate.py g3: segments_ms.total={total!r} acc.actions.max_abs={err!r}: missing in {path}', file=sys.stderr)
        return 2
    lo, hi = PUBLISHED_MS * (1 - TOL), PUBLISHED_MS * (1 + TOL)
    t_ok = lo <= total <= hi
    a_ok = err <= ACTION_MAX_ABS
    print(f'total simulated time : {total:.3f} ms  (published {PUBLISHED_MS} ms, window {lo:.2f}..{hi:.2f} ms, '
          f'{(total / PUBLISHED_MS - 1) * 100:+.1f} %)  {"PASS" if t_ok else "MISS"}')
    print(f'action max-abs error : {err:.4f}  (limit {ACTION_MAX_ABS}, published 0.032)  {"PASS" if a_ok else "MISS"}')
    ok = t_ok and a_ok
    print('G3 gate:', 'PASS' if ok else 'MISS')
    return 0 if ok else 1


def main(argv):
    if len(argv) == 3 and argv[1] == 'g3':
        return g3(argv[2])
    if len(argv) >= 2 and argv[1] == 'g4a':
        print('gate.py g4a: not implemented (G4a)', file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
