import re

import pytest

from workload_modeling import constants, md_to_html, render
from workload_modeling import smolvla_ops as m


@pytest.fixture(scope="module")
def c():
    return constants.load()


W1 = "vlabench-3cam"


def w1_rows(c):
    return m.build(c, W1, m.kernel_set(W1).lowering)


@pytest.fixture(scope="module")
def md(c):
    return render.render_md(c, W1)


def test_every_row_has_unit_and_kernel(c):
    for r in m.annotate(w1_rows(c)):
        assert r.pr1_unit and r.hand_kernel, r.op


def test_kernel_mapping_matches_kernel_repo(c):
    rows = {r.op: r for r in m.annotate(w1_rows(c))}
    assert rows["gelu"].hand_kernel.startswith("Activation (GELU")   # branch gelu-layernorm-kernels
    assert rows["ln1"].hand_kernel == "LayerNorm"
    assert rows["silu_mul"].hand_kernel.startswith("Activation")
    assert rows["mlp_gateup"].hand_kernel == "SummaGEMM"
    assert rows["state_proj"].pr1_unit.startswith("host")


def test_md_totals_equal_stage_totals(c, md):
    tot = m.stage_totals(w1_rows(c))
    section = md.split("## Totals", 1)[1]
    for s, t in tot.items():
        row = re.search(rf"^\| {s} \| ([\d,.]+) \| ([\d,.]+) \|", section, re.MULTILINE)
        assert row, s
        assert float(row.group(1).replace(",", "")) == pytest.approx(t.flop / 1e9, abs=5e-4)
        assert float(row.group(2).replace(",", "")) == pytest.approx(t.matmul_flop / 1e9, abs=5e-4)


def test_md_has_one_table_row_per_op(c, md):
    rows = w1_rows(c)
    for r in rows:
        assert f"| `{r.op}` | {r.layers} |" in md, r.op


def test_results_at_a_glance_matches_kernel_sums(c, md):
    ks = render.kernel_sums(c, W1, w1_rows(c))
    top = md.split("## Results at a glance", 1)[1].split("## What these numbers assume", 1)[0]
    assert f"**{ks['total_ms']} ms per policy call, MFU {ks['total']['mfu']} %**" in top
    for k in ks["stages"]:
        assert f"| {k['stage']} | {k['ms']} | {k['share']} | {k['gflop']} | {k['mfu']} |" in top
    assert sum(float(k["share"]) for k in ks["stages"]) == pytest.approx(100, abs=2)


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
    text = render.render_csv(c, W1)
    assert len(text.strip().splitlines()) == len(w1_rows(c)) + 1


@pytest.mark.parametrize("wl,low", [("vlabench-3cam", m.STAGE1_LOWERING),
                                    ("vlabench-3cam", m.STAGE1_LOWERING_W1B)])
def test_hand_kernel_times_and_stage_sums(c, wl, low):
    rows = m.annotate(m.build(c, wl, low))
    ns = {(r.stage, r.op): m.hand_kernel_ns(c, wl, r) for r in rows}
    # Every operator outside the host prepare step has a simulated kernel or runs inside another row's call.
    for r in rows:
        if r.op in m.HOST_OPS:
            continue
        key = (r.stage, r.op)
        if key in m.INCLUDED_IN:
            assert ns[(r.stage, m.INCLUDED_IN[key])] is not None, r.op
        else:
            assert ns[key] is not None, (r.stage, r.op, m.hand_kernel_key(r, m.kernel_set(wl).group))
    sums = render.kernel_sums(c, wl, rows)
    want = sum(r.count * ns[(r.stage, r.op)] for r in rows if ns[(r.stage, r.op)] is not None) / 1e3
    assert sums["total"]["total"] == f"{want:,.1f}"
    vision = next(k for k in sums["stages"] if k["stage"] == "vision")
    want_v = sum(r.count * ns[(r.stage, r.op)] for r in rows
                 if r.stage == "vision" and r.cls == "gemm") / 1e3
    assert vision["GEMM"] == f"{want_v:,.1f}"
    # Times are per workload; workloads without a kernel set get none, and cannot be rendered.
    for other in ("pr1-1cam", "libero-2cam"):
        assert not any(m.hand_kernel_ns(c, other, r) for r in m.annotate(m.build(c, other)))
        with pytest.raises(KeyError):
            m.kernel_set(other)


def test_eltwise_and_attention_mapping(c):
    rows = {(r.stage, r.op): r for r in m.annotate(w1_rows(c))}
    assert rows[("vision", "bias_q")].hand_kernel == "Eltwise (bias)"
    assert rows[("prefix", "rope_q")].hand_kernel == "Eltwise (rope)"
    assert rows[("expert", "euler_update")].hand_kernel == "Eltwise (axpy)"
    assert rows[("connector", "pixel_shuffle")].hand_kernel.startswith("Eltwise (pixel_shuffle")
    assert rows[("vision", "transpose_k")].hand_kernel.startswith("FlatAttention")
    qk = rows[("expert", "self_qk")]
    assert m.hand_kernel_key(qk) == f"h1_kernels.expert.flatattention_{'x'.join(map(str, qk.shape))}"
    assert m.hand_kernel_key(qk) in c.entries
    assert m.hand_kernel_key(rows[("expert", "self_pv")]) is None          # inside the self_qk call


@pytest.mark.parametrize("group,n", [("h1_kernels", 21)])
def test_gemm_times_monotonic_in_shape(c, group, n):
    """A GEMM contained in a larger one (every dimension <=) is not slower: catches a starved tile sweep."""
    t = {}
    for key, e in c.entries.items():
        if key.startswith(group + ".") and ".gemm_" in key:
            t[tuple(int(x) for x in key.rsplit("_", 1)[1].split("x"))] = float(e.value)
    assert len(t) == n
    for a, ta in t.items():
        for b, tb in t.items():
            if a != b and all(x <= y for x, y in zip(a, b)):
                assert ta <= tb, (a, ta, b, tb)


def test_h1_entries_provenance(c):
    """Every H1 time is measured on vlabench-3cam and arch h1, and names the kernel build and simulator tree."""
    h1 = {k: e for k, e in c.entries.items() if k.startswith("h1_kernels.")}
    assert len(h1) == 51
    for k, e in h1.items():
        assert (e.label, e.workload, e.arch, e.unit) == ("measured", "vlabench-3cam", "h1", "ns"), k
        # kernel build: the frozen worktree build ("+ build <hash>"), commit de53537 (small-M GEMM picks), or the
        # submodule commit a later pick was re-run on (final_runs.py: "HaozeG/gvsoc@<7 hex> (submodule ...")
        build = ("+ build " in e.source or "HaozeG/gvsoc@de53537" in e.source
                 or re.search(r"HaozeG/gvsoc@[0-9a-f]{7} \(submodule third_party/softhier-kernels\)", e.source))
        assert "HaozeG/gvsoc@" in e.source and build and "GVSoC tree " in e.source, k


def test_w1b_page_states_lowering(c):
    page = render.render_md(c, "vlabench-3cam", m.STAGE1_LOWERING_W1B, "W1-b: cameras batched")
    assert "cameras are batched" in page and "Vision per camera" not in page
    assert "h1_kernels" in page and "arch.h1" in page
