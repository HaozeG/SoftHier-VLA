#!/usr/bin/env python3
"""G3 validity gate (design Component 6, file 34).

    gate.py g3 <run folder>

Reads <run folder>/summary.json (written by `run.py smolvla-e2e`):
  * total simulated time `segments_ms.total` must be within +-5 % of 119.1 ms (softhier-mlir's published figure);
  * action max-abs error `acc.actions.max_abs` (computed by run.py against the --ref npz, i.e. G2) must be <= 0.10
    (published: 0.032).
Prints both values and PASS or MISS; exit 0 on PASS, 1 on MISS, 2 on usage / missing data.

    gate.py g4a <vision run> <prefix run> <expert run> [--out <g4a_estimate.json>]

The G4a composition (design Component 6, G4 staged plan). The three folders are the G4a.1 (`--vision-only 1024
--layers 1`), G4a.2 (`--prefix-only 2 --layers 8`) and G4a.3 (`--expert-only ref:... --steps 2`) runs; each holds the
summary.json of softhier-mlir's cost runs (per-segment simulated ms, plus `roi_ns` and `wall_s` of the simulator
process). Composed full-chain time, all labelled ESTIMATE:
  vision    = cams x (embed + 12 x (attention + mlp of the one measured layer) + post-LN)
  connector = G4a.2 start->emb (pixel shuffle + connector GEMM of both cameras + the sqrt(960) scaling); G4a.1's
              1-camera connector is kept as a cross-check only
  prefix    = 2 x (sum of the 8 measured layers)
  expert    = KV projection + 10 x mean of the 2 measured steps
  total     = vision + connector + prefix + expert
Also the G4b wall-time projection per phase (V, Pa, Pb, X): measured wall seconds per simulated ms of the source run
(wall_s / the run's timed segments, whole program incl. fixed start-up cost, so somewhat pessimistic) x the composed simulated ms of
that phase. R1: any projected phase above 7200 s means STOP and ask the owner before G4b. Exit 0 if the estimate was
written and the G4a accuracy checks pass, 1 if an accuracy check misses, 2 on usage / missing data.
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


CAMS = 2
VISION_LAYERS = 12
EXPERT_STEPS = 10
R1_LIMIT_S = 7200.0
EXPERT_VS_G2_MAX_ABS = 0.10      # same limit as the G3 gate, applied to G2 xt[2] of the 2-step expert run
SRC_FIELDS = ('wall_s', 'roi_ns', 'max_child_rss_mb', 'trace')


def load(folder):
    path = os.path.join(folder, 'summary.json')
    if not os.path.isfile(path):
        print(f'gate.py g4a: {path} not found', file=sys.stderr)
        return None
    return json.load(open(path))


def ratio(s, timed_ms):
    """wall seconds per simulated ms of the run's timed segments (the whole-program ROI also counts the sample-dump
    printing between marks, which the segment times exclude: 222 ms ROI for 15.8 ms of timed segments in G4a.1)"""
    return s['wall_s'] / timed_ms


def g4a(vision_dir, prefix_dir, expert_dir, out=None):
    v, p, x = load(vision_dir), load(prefix_dir), load(expert_dir)
    if v is None or p is None or x is None:
        return 2
    try:
        vs, ps, xs = v['segments_ms'], p['segments_ms'], x['segments_ms']
        n_layers = len(vs['attention_per_layer'])
        n_pre = len(ps['prefix_per_layer'])
        if n_pre != 8 or n_layers != 1:
            print(f'gate.py g4a: unexpected run shapes: {n_layers} vision layer(s) (want 1), {n_pre} prefix layer(s) (want 8)', file=sys.stderr)
            return 2
        layer = sum(vs['attention_per_layer']) / n_layers + sum(vs['mlp_per_layer']) / n_layers
        per_cam = vs['embed'] + VISION_LAYERS * layer + vs['post_ln']
        vision = CAMS * per_cam
        connector = ps['connector']
        pre_half = sum(ps['prefix_per_layer'])
        prefix = 2 * pre_half
        expert = xs['expert_kv_projection'] + EXPERT_STEPS * xs['expert_step_mean']
        total = vision + connector + prefix + expert
        # simulated ms of each G4b phase, and the wall seconds the source run's rate gives it
        phase_ms = {'V': vision, 'Pa': connector + pre_half, 'Pb': pre_half, 'X': expert}
        timed = {'V': vs['embed'] + layer + vs['post_ln'] + vs['connector'], 'Pa': connector + pre_half,
                 'X': xs['expert_kv_projection'] + sum(xs['expert_per_step'])}
        timed['Pb'] = timed['Pa']
        src = {'V': v, 'Pa': p, 'Pb': p, 'X': x}
        proj = {k: {'simulated_ms': phase_ms[k], 'source_timed_ms': timed[k], 'wall_s_per_sim_ms': ratio(src[k], timed[k]),
                    'projected_wall_s': ratio(src[k], timed[k]) * phase_ms[k]} for k in phase_ms}
    except (KeyError, TypeError, ZeroDivisionError) as e:
        print(f'gate.py g4a: missing or invalid field in the summary files: {e!r}', file=sys.stderr)
        return 2
    # accuracy checks of the partial runs: their own references, plus G2 xt[2] for the expert
    vis_err, vis_ref = dig(v, 'acc', 'VIS_vs_floor_max_abs'), dig(v, 'acc', 'ref_max')
    x_floor, x_g2 = dig(x, 'acc', 'actions_vs_floor'), dig(x, 'acc', 'actions_vs_lerobot')
    if None in (vis_err, vis_ref, x_floor, x_g2):
        print('gate.py g4a: accuracy fields missing (VIS_vs_floor_max_abs, ref_max, actions_vs_floor, actions_vs_lerobot)', file=sys.stderr)
        return 2
    vis_ok = vis_err <= 0.03 * vis_ref     # softhier-mlir's atol: 3 % of max|ref| (its additional 5 % relative term is per element, not applied)
    x_ok = x_g2 <= EXPERT_VS_G2_MAX_ABS
    est = {
        'label': 'estimate (composed from partial simulations on GVSoC, ideal HBM)',
        'composed_ms': {'vision': vision, 'connector': connector, 'prefix': prefix, 'expert': expert, 'total': total},
        'terms': {
            'vision_per_camera_ms': per_cam, 'vision_embed_ms': vs['embed'], 'vision_layer_ms': layer,
            'vision_attention_ms': sum(vs['attention_per_layer']) / n_layers, 'vision_mlp_ms': sum(vs['mlp_per_layer']) / n_layers,
            'vision_post_ln_ms': vs['post_ln'], 'cams': CAMS, 'vision_layers': VISION_LAYERS,
            'connector_ms (G4a.2 start->emb)': connector, 'connector_1cam_G4a1_crosscheck_ms': vs['connector'],
            'prefix_8_layers_ms': pre_half, 'prefix_per_layer_ms': ps['prefix_per_layer'],
            'expert_kv_projection_ms': xs['expert_kv_projection'], 'expert_step_mean_ms': xs['expert_step_mean'],
            'expert_per_step_ms': xs['expert_per_step'], 'expert_steps': EXPERT_STEPS,
        },
        'sources': {k: {'run': os.path.abspath(d), **{f: s.get(f) for f in SRC_FIELDS}} for k, d, s in
                    (('vision (G4a.1)', vision_dir, v), ('prefix (G4a.2)', prefix_dir, p), ('expert (G4a.3)', expert_dir, x))},
        'accuracy': {'vision_VIS_vs_fp16_floor_max_abs': vis_err, 'vision_ref_max': vis_ref, 'vision_ok': vis_ok,
                     'expert_actions_vs_fp16_floor': x_floor, 'expert_xt2_vs_G2_max_abs': x_g2,
                     'expert_limit': EXPERT_VS_G2_MAX_ABS, 'expert_ok': x_ok},
        'g4b_projection': proj,
        'r1_limit_s': R1_LIMIT_S,
        'r1_triggered': any(e['projected_wall_s'] > R1_LIMIT_S for e in proj.values()),
    }
    out = out or os.path.join(os.path.dirname(os.path.abspath(vision_dir)), 'g4a_estimate.json')
    json.dump(est, open(out, 'w'), indent=1)
    print('G4a composed time (ESTIMATE, simulated on GVSoC, ideal HBM, 1 GHz):')
    print(f'  vision    {vision:10.3f} ms  = {CAMS} x (embed {vs["embed"]:.3f} + {VISION_LAYERS} x layer {layer:.3f} + post-LN {vs["post_ln"]:.3f})')
    print(f'  connector {connector:10.3f} ms  (G4a.2 start->emb; G4a.1 1-camera connector {vs["connector"]:.3f} ms, cross-check only)')
    print(f'  prefix    {prefix:10.3f} ms  = 2 x {pre_half:.3f} (8 layers)')
    print(f'  expert    {expert:10.3f} ms  = KV projection {xs["expert_kv_projection"]:.3f} + {EXPERT_STEPS} x {xs["expert_step_mean"]:.3f}')
    print(f'  total     {total:10.3f} ms')
    print('accuracy of the partial runs:')
    print(f'  vision VIS vs fp16 floor max abs {vis_err:.4f} (|ref| max {vis_ref:.2f})  {"PASS" if vis_ok else "MISS"}')
    print(f'  expert actions vs fp16 floor {x_floor:.4f}; G2 xt[2] max abs {x_g2:.4f} (limit {EXPERT_VS_G2_MAX_ABS})  {"PASS" if x_ok else "MISS"}')
    print('G4b projection (measured wall s per simulated ms x composed simulated ms; limit 7200 s per process):')
    for k, e in proj.items():
        print(f'  {k:<2} {e["simulated_ms"]:10.3f} ms x {e["wall_s_per_sim_ms"]:7.2f} s/ms = {e["projected_wall_s"]:9.0f} s = '
              f'{e["projected_wall_s"] / 3600:5.2f} h  {"OVER R1 LIMIT" if e["projected_wall_s"] > R1_LIMIT_S else "ok"}')
    print('R1:', 'TRIGGERED: stop and ask the owner before G4b' if est['r1_triggered'] else 'not triggered')
    print(f'written {out}')
    return 0 if (vis_ok and x_ok) else 1


def main(argv):
    if len(argv) == 3 and argv[1] == 'g3':
        return g3(argv[2])
    if len(argv) in (5, 7) and argv[1] == 'g4a' and (len(argv) == 5 or argv[5] == '--out'):
        return g4a(argv[2], argv[3], argv[4], argv[6] if len(argv) == 7 else None)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
