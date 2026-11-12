import re

import pytest

from workload_modeling import constants, md_to_html, render
from workload_modeling import smolvla_ops as m


@pytest.fixture(scope="module")
def c():
    return constants.load()


@pytest.fixture(scope="module")
def md(c):
    return render.render_md(c, "libero-2cam")


def test_every_row_has_unit_and_kernel(c):
    for r in m.annotate(m.build(c, "libero-2cam")):
        assert r.unit_today and r.hand_kernel, r.op


def test_kernel_mapping_matches_kernel_repo(c):
    rows = {r.op: r for r in m.annotate(m.build(c, "libero-2cam"))}
    assert rows["gelu"].hand_kernel.startswith("Activation (GELU")   # branch gelu-layernorm-kernels
    assert rows["ln1"].hand_kernel == "LayerNorm"
    assert rows["silu_mul"].hand_kernel.startswith("Activation")
    assert rows["mlp_gateup"].hand_kernel == "SummaGEMM"
    assert rows["state_proj"].unit_today.startswith("host")


def test_md_totals_equal_stage_totals(c, md):
    tot = m.stage_totals(m.build(c, "libero-2cam"))
    section = md.split("## Totals", 1)[1]
    for s, t in tot.items():
        row = re.search(rf"^\| {s} \| ([\d,.]+) \| ([\d,.]+) \|", section, re.MULTILINE)
        assert row, s
        assert float(row.group(1).replace(",", "")) == pytest.approx(t.flop / 1e9, abs=5e-4)
        assert float(row.group(2).replace(",", "")) == pytest.approx(t.matmul_flop / 1e9, abs=5e-4)


def test_md_has_one_table_row_per_op(c, md):
    rows = m.build(c, "libero-2cam")
    for r in rows:
        assert f"| `{r.op}` | {r.layers} |" in md, r.op


def test_html_keeps_every_table(md):
    html = md_to_html.convert(md, "t")
    n_md = len(re.findall(r"^\| ---", md, re.MULTILINE))
    assert html.count("<table>") == n_md


def test_mermaid_fence_becomes_pre(md):
    html = md_to_html.convert(md, "t")
    assert '<pre class="mermaid">' in html
    assert "```mermaid" not in html
    assert "flowchart LR" in html


def test_only_external_url_is_pinned_mermaid(md):
    html = md_to_html.convert(md, "t")
    urls = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', html))
    assert urls == {md_to_html.MERMAID_URL}
    assert "mermaid@10.9.1" in md_to_html.MERMAID_URL


def test_csv_has_all_rows(c):
    text = render.render_csv(c, "libero-2cam")
    assert len(text.strip().splitlines()) == len(m.build(c, "libero-2cam")) + 1


def test_hand_kernel_times_and_stage_sums(c):
    wl = "libero-2cam"
    rows = m.annotate(m.build(c, wl))
    ns = {(r.stage, r.op): m.hand_kernel_ns(c, wl, r) for r in rows}
    # Every operator outside the host prepare step has a simulated kernel or runs inside another row's call.
    for r in rows:
        if r.op in m.HOST_OPS:
            continue
        key = (r.stage, r.op)
        if key in m.INCLUDED_IN:
            assert ns[(r.stage, m.INCLUDED_IN[key])] is not None, r.op
        else:
            assert ns[key] is not None, (r.stage, r.op, m.hand_kernel_key(r))
    sums = render.kernel_sums(c, wl, rows)
    want = sum(r.count * ns[(r.stage, r.op)] for r in rows if ns[(r.stage, r.op)] is not None) / 1e3
    assert sums["total"]["total"] == f"{want:,.1f}"
    vision = next(k for k in sums["stages"] if k["stage"] == "vision")
    want_v = sum(r.count * ns[(r.stage, r.op)] for r in rows
                 if r.stage == "vision" and r.cls == "gemm") / 1e3
    assert vision["GEMM"] == f"{want_v:,.1f}"
    # Times are for libero-2cam shapes only; another workload gets none.
    assert not any(m.hand_kernel_ns(c, "pr1-1cam", r) for r in m.annotate(m.build(c, "pr1-1cam")))


def test_eltwise_and_attention_mapping(c):
    rows = {(r.stage, r.op): r for r in m.annotate(m.build(c, "libero-2cam"))}
    assert rows[("vision", "bias_q")].hand_kernel == "Eltwise (bias)"
    assert rows[("prefix", "rope_q")].hand_kernel == "Eltwise (rope)"
    assert rows[("expert", "euler_update")].hand_kernel == "Eltwise (axpy)"
    assert rows[("connector", "pixel_shuffle")].hand_kernel.startswith("Eltwise (pixel_shuffle")
    assert rows[("vision", "transpose_k")].hand_kernel.startswith("FlatAttention")
    assert m.hand_kernel_key(rows[("expert", "self_qk")]) == "hand_kernels.expert.flatattention_15x50x227x64"
    assert m.hand_kernel_key(rows[("expert", "self_pv")]) is None          # inside the self_qk call
