import json
from pathlib import Path

import pytest

from workload_modeling import constants, fixtures
from workload_modeling import smolvla_ops as m

CROSSCHECK = Path(__file__).resolve().parents[3] / "docs" / "analysis" / "crosscheck.json"


@pytest.fixture(scope="module")
def c():
    return constants.load()


@pytest.fixture(scope="module")
def base_header():
    return fixtures.load("lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")[1]


def test_every_tensor_has_a_part(base_header):
    for name in base_header:
        m.part_of(name)  # raises for an unknown tensor


def test_param_counts_match_wiki_table(base_header):
    # wiki://resources/models/smolvla.md per-part table (millions, one decimal)
    counts = m.param_counts(base_header)
    assert sum(counts.values()) == 450_046_176
    want = {"vision": 86.4, "connector": 11.8, "llm": 157.3, "expert": 98.2, "token_embedding": 47.3, "lm_head": 47.3}
    for part, mil in want.items():
        assert round(counts[part] / 1e6, 1) == mil, part
    assert counts["io"] / 1e6 == pytest.approx(1.5, abs=0.15)


@pytest.mark.parametrize("wid", ["pr1-1cam", "libero-2cam", "release-3cam"])
def test_config_shapes_match_header(c, wid):
    d = m.load_dims(c.workloads[wid])  # check_header raises on any mismatch
    assert (d.v_layers, d.v_hidden, d.v_mlp, d.t_layers, d.e_hidden, d.e_mlp) == (12, 768, 3072, 16, 720, 2048)
    assert d.kv_width == 320


def test_token_counts(c):
    for wid, (vis, pre) in {"pr1-1cam": (256, 65), "libero-2cam": (1024, 177)}.items():
        wl = c.workloads[wid]
        d = m.load_dims(wl)
        assert m.vision_tokens(wl, d) == vis
        assert m.prefix_tokens(wl, d) == pre


def test_worked_example_expert_gateup(c):
    row = next(r for r in m.build(c, "pr1-1cam") if r.op == "mlp_gateup")
    assert row.shape == (50, 4096, 720)
    assert row.count == 160
    assert row.flop == 47_185_920_000
    assert row.bytes_weight == 2 * 720 * 4096 * 160
    assert row.flop_per_wbyte == 50


def test_weights_not_scaled_by_cameras(c):
    one = m.stage_totals(m.build(c, "pr1-1cam"))["vision"].bytes_weight
    vis = [r for r in m.build(c, "libero-2cam") if r.stage == "vision" and r.op != "pos_embed_add"]
    two = sum(r.bytes_weight for r in vis)
    one_np = sum(r.bytes_weight for r in m.build(c, "pr1-1cam") if r.stage == "vision" and r.op != "pos_embed_add")
    assert two == one_np and one > 0


def test_lowering_defaults_are_reference(c):
    ref = m.stage_totals(m.build(c, "pr1-1cam"))["expert"].matmul_flop
    hoisted = m.stage_totals(m.build(c, "pr1-1cam", m.Lowering(kv_proj_per_step=False)))["expert"].matmul_flop
    # K/V projection of 65 prefix tokens, 320 -> 320, k and v, 8 cross layers, 9 extra steps
    assert ref - hoisted == pytest.approx(2 * 65 * 320 * 320 * 2 * 8 * 9)


def test_padded_matches_crosscheck_json(c):
    res = json.loads(CROSSCHECK.read_text())
    for wid, w in res["workloads"].items():
        rows = m.build(c, wid, m.pr1_lowering(w["program_shapes"]["prefix_rows"]))
        tot = m.stage_totals(rows)
        for s, v in w["stages"].items():
            assert v["pass"], (wid, s)
            assert tot[s].matmul_flop / 1e9 == pytest.approx(v["model_padded_gflop"], rel=1e-4), (wid, s)
            assert tot[s].matmul_flop / 1e9 == pytest.approx(v["pr1_redmule_gflop"], rel=1e-2), (wid, s)


IO_OPS = {"state_proj", "bias_state", "action_in_proj", "bias_action_in", "time_mlp_in", "bias_time_in",
          "time_mlp_out", "bias_time_out", "action_out_proj", "bias_action_out"}


def test_weight_bytes_equal_header_parameters(c):
    """One pass (one flow step for the expert) reads every called tensor once: 2 B x parameters per part.

    Uses libero-2cam (1024 vision positions = the full position table). The token embedding is read
    only for the 48 looked-up rows; lm_head is never called.
    """
    wl = c.workloads["libero-2cam"]
    header = fixtures.load(wl.repo, wl.revision)[1]
    params = m.param_counts(header)
    steps = m.load_dims(wl).steps
    rows = m.build(c, "libero-2cam")

    def wb(stage, ops=None, exclude=()):
        return sum(r.bytes_weight for r in rows
                   if r.stage == stage and (ops is None or r.op in ops) and r.op not in exclude)

    assert wb("vision") == 2 * params["vision"]
    assert wb("connector") == 2 * params["connector"]
    assert wb("prefix", exclude=IO_OPS | {"lang_embed_scale"}) == 2 * params["llm"]
    assert wb("expert", exclude=IO_OPS) / steps == 2 * params["expert"]
    io = wb("prefix", ops=IO_OPS) + wb("expert", ops=IO_OPS) / steps
    assert io == 2 * params["io"]
    assert wb("prefix", ops={"lang_embed_scale"}) == 2 * wl.language_tokens * 960
