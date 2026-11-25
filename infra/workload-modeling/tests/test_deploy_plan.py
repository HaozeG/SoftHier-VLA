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
