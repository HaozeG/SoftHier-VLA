"""Stage-1 deployment plan (upstream's HBM plan and layout, stage1-spec §4.3 item 3): every device operator is
deployed, shared keys agree, and every kernel touching a head-major tensor uses that tensor's rows per head."""
from workload_modeling import deploy_plan as dp
from workload_modeling.upstream_rules import attn_opt


def test_every_device_operator_is_deployed_and_keys_agree():
    p = dp.Plan()
    runs = p.runs()                       # asserts operators sharing a key share cfg and placement
    mapped = {(d.stage, d.op) for d in p.deploys}
    from workload_modeling import smolvla_ops as m
    for r in p.rows:
        if r.op in m.HOST_OPS or (r.stage, r.op) in m.INCLUDED_IN:
            continue
        assert (r.stage, r.op) in mapped, (r.stage, r.op)
    assert all(set(v["place"]) == set(dp.DEFAULTS[v["kernel"]]) for v in runs.values())


def test_head_major_rows_are_shared():
    p = dp.Plan()
    for d in p.deploys:
        for arr, t in d.tensors.items():
            T = p.T[t]
            if not T.heads or T.rows is None:
                continue
            if d.kernel == "gemm":
                assert d.cfg["m_size"] == T.rows, (d.op, t)
                n = "resha_x_from_m" if arr == "X_addr" else "resha_z_to_m"
                assert d.cfg[n] == T.heads * T.rows, (d.op, t)
                # chunk = head dimension: (M * N) / resha_*_m with N (Z) or K (X) = heads * d
                width = d.cfg["k_size"] if arr == "X_addr" else d.cfg["n_size"]
                assert d.cfg["m_size"] * width // d.cfg[n] == dp.D, (d.op, t)
            elif d.kernel == "attn":
                rows = d.cfg["q_pad"] if arr in ("Q_addr", "O_eaddr") else d.cfg["kv_pad"]
                assert rows == T.rows, (d.op, t)
            elif d.kernel in ("rope", "acti"):
                assert d.cfg["m_size"] == T.rows, (d.op, t)


def test_space_a_is_west_and_b_south():
    p = dp.Plan()
    assert p.T["vision.res"].space == "south" and p.T["vision.act"].space == "west"
    assert p.T["prefix.k"].space == "south" and p.T["prefix.q"].space == "west"


def test_attn_rule_fixes_are_recorded():
    s = attn_opt(15, 5, 241, 241)          # upstream's setting drops heads here (5 KV heads, 4 groups)
    assert "head" in s.fix and 64 // (s.sx * s.sy) >= 5
    v = attn_opt(12, 12, 1024, 1024)
    assert v.fix == "" and (v.sx, v.sy, v.q_pad) == (8, 8, 1024)


def test_w1b_plan_batches_the_vision_encoder():
    # W1-b under the plan: same FLOP as W1; vision attention as one call per layer over cameras x heads, its q/k/v
    # head-major tensors (heads, cameras x rows, d); prefix and expert runs are W1's
    from workload_modeling import constants, smolvla_ops as m
    from workload_modeling.deploy_plan import Plan
    c = constants.load()
    w = c.primary_workload.id
    flop = lambda low: sum(r.flop for r in m.build(c, w, low))
    assert flop(m.PLAN_LOWERING_W1B) == flop(m.PLAN_LOWERING)
    a, b = Plan().runs(), Plan(lowering=m.PLAN_LOWERING_W1B).runs()
    assert {k: v for k, v in a.items() if not k.startswith("vision.")} == \
           {k: v for k, v in b.items() if not k.startswith("vision.")}
    at = b["vision.flatattention_36x1024x1024x64"]
    assert at["count"] == 12 and at["cfg"]["q_pad"] == 1024
    q = next(v for k, v in b.items() if k.startswith("vision.gemm_3072x768x768_zto") and "q_proj" in v["ops"])
    assert q["cfg"]["resha_z_to_m"] * 64 == 3072 * 768       # chunk = head dim: (12, 3072, 64)
