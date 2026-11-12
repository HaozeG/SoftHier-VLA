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
    assert rows["gelu"].hand_kernel == "none"            # Activation covers Sigmoid, ReLU, SiLU only
    assert rows["ln1"].hand_kernel == "none"             # no LayerNorm kernel
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
