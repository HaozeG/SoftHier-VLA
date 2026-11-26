"""Kernel explorer: group placement and the baseline checks (pick, constants, placement vs trace, sums)."""

from workload_modeling import kernel_explorer as kx


def test_placement_matches_grid_sync_group_init():
    # 2 x 4 groups, 5 busy: groups numbered row-major over 4 x 2 blocks; group 4 sits at x 0-1, y 4-7
    pl = kx.placement(2, 4, 5)
    active = [i for i, p in enumerate(pl) if p["g"] >= 0]
    assert len(active) == 40
    assert pl[0]["g"] == 0 and pl[2]["g"] == 1 and pl[6]["g"] == 3
    assert pl[4 * 8 + 0]["g"] == 4 and pl[4 * 8 + 2]["g"] == -1
    assert (pl[8 + 1]["cx"], pl[8 + 1]["cy"]) == (1, 1)
    # one 8 x 8 group: everyone works
    assert all(p["g"] == 0 for p in kx.placement(8, 8, 1))


def test_gemm_iterations_follow_summagemm():
    s = kx.gemm_setting(dict(scale_x="4", scale_y="2", groups="8", m_tile="32", n_tile="192", k_tile="96",
                             m_size="64", n_size="768", k_size="768", strategy="splitK"), (32, 64, 32), 1024.0)
    assert (s["active"], s["m_iter"], s["n_iter"], s["k_iter"]) == (64, 1, 1, 1)


def test_all_baseline_checks_pass():
    ex = kx.Explorer()
    bad = [c for c in ex.checks if not c["ok"]]
    assert not bad, bad
    kinds = {c["kind"] for c in ex.checks}
    assert kinds == {"pick", "constants", "placement", "sum"}


def test_redmule_constants_match_h1_arch_file():
    # constants.yaml arch.h1 is the one source for the explorer; this guards it against drift from the arch file
    import re
    from workload_modeling import constants
    c = constants.load()
    arch = (kx.KR / "arch" / "arch_h1.py").read_text()
    val = {k: int(re.search(rf"redmule_ce_{k}\s*=\s*(\d+)", arch).group(1)) for k in ("height", "width", "pipe")}
    assert (val["height"], val["width"], val["pipe"]) == tuple(
        int(c.value(f"arch.h1.redmule_{k}")) for k in ("h", "w", "pipe"))
    assert kx.redmule_blocks(c) == (32, 64, 32)


def test_redmule_array_eff_counts_whole_blocks():
    blk = (32, 64, 32)
    assert kx.redmule_array_eff(128, 128, 128, blk) == 1.0
    assert kx.redmule_array_eff(128, 96, 128, blk) == 0.75      # 96 columns -> two 64-column blocks
    assert kx.redmule_array_eff(32, 32, 256, blk) == 0.5
    assert abs(kx.redmule_array_eff(64, 160, 192, blk) - 160 / 192) < 1e-12


def test_f_pad_includes_array_padding():
    ex = kx.Explorer()
    for k in ex.kernels.values():
        if k["kind"] == "gemm":
            assert k["f_pad_array"] == k["selected"]["setting"]["f_array"]
            assert abs(k["f_pad"] - k["f_pad_shape"] * k["f_pad_array"]) < 1e-12
            assert abs(k["f_pad"] * k["f_occ"] * k["f_rest"] - k["mfu"]) < 1e-12
    assert all(k["f_pad_array"] == 1.0 for k in ex.kernels.values() if k["kind"] == "attn")
    # upstream's setting for vision 1024x768x768 under the plan (tile 128x96) loses a quarter of the array to
    # block fill
    c = next(c for c in ex.kernels["vision.gemm_1024x768x768"]["candidates"]
             if c["cand"] == "Nosplit_x8y8g1_t128x96x128")
    assert c["setting"]["f_array"] == 0.75 and c["setting"]["tile_blk"][:2] == [128, 128]
    # 50x32x720 (splitK tile 32x32x32): the 32 output columns fill half of a 64-column block
    k = ex.kernels["gemm_50x32x720"]
    assert k["f_pad_array"] == 0.5
    assert k["f_pad_text"].endswith("x block fill 0.50 (tile 32x32 runs as 32x64)")


def test_mesh_comes_from_constants():
    from workload_modeling import constants
    c = constants.load()
    assert (kx.MESH_X, kx.MESH_Y) == (c.value("arch.h1.clusters_x"), c.value("arch.h1.clusters_y"))
    assert kx.NCL == c.clusters("h1")


def test_kernel_model_is_the_explorer_model():
    # one model: the GEMM sweeps' ranking (gemm_sweep.model_ns) and the explorer's settings use kernel_model
    from workload_modeling import kernel_model as km
    A = km.arch()
    assert (A.blk_m, A.blk_n, A.blk_k, A.cyc_per_k) == (32, 64, 32, 2)
    assert km.call_ns(128, 96, 128, A, overhead=False) == 4 * 2 * 128 * 2     # 96 columns time as two 64-column blocks
    assert km.block_fill(128, 96, 128, A) == 0.75
    G = kx._drivers()[0]
    r = km.gemm_ns(1024, 768, 768, 128, 96, 128, 8, 8, 1, "Nosplit", A)
    assert G.model_ns(1024, 768, 768, 128, 96, 128, 8, 8, 1, "Nosplit") == r["est_ns"]
    ex = kx.Explorer()
    c = next(c for c in ex.kernels["vision.gemm_1024x768x768"]["candidates"]
             if c["cand"] == "Nosplit_x8y8g1_t128x96x128")
    assert c["est_ns"] == round(r["est_ns"])


def test_w1_page_is_the_upstream_plan():
    # W1 reads the deployment-plan group: one kernel per plan run key, selected = fastest checked plan run
    import csv
    ex = kx.Explorer()
    w1 = next(p for p in ex.pages if p["name"] == "W1")
    assert w1["group"] == "h1_plan"
    plan = {k: v for k, v in ex.kernels.items() if v["plan"]}
    assert len(plan) == len(ex.plan_runs)
    # upstream's attention setting failed the check: kept as a failed run next to the recorded fix
    a = ex.kernels["prefix.flatattention_15x241x241x64"]
    assert [f["cand"] for f in a["failed"]] == ["attn_s2x4_b384x256"]
    assert a["selected"]["cand"] == "attn_s2x4_b256x256" and "row tile" in a["selected"]["fix"]
    # the action output projection: upstream's only setting failed, the fallback is labelled as such
    g = ex.kernels["expert.gemm_50x32x720"]
    assert g["selected"]["origin"].startswith("fallback")
    # the page's composed time is plan_runs.py pick's sum
    pick = list(csv.DictReader(open(kx.RES / "plan_pick_h1.csv")))
    assert abs(float(w1["sums"]["total_ms"]) - sum(int(r["total_ns"]) for r in pick) / 1e6) < 0.006
    assert all(c["ok"] for c in ex.checks)


def test_glossary_covers_every_name_on_the_page():
    # every naming element a reader meets in kernel and candidate names has a glossary entry (cold-read feedback)
    import re
    ex = kx.Explorer()
    text = " ".join(e["term"] + " " + e["meaning"] for e in kx.glossary(ex))
    names = set(ex.kernels) | {c["cand"] for k in ex.kernels.values() for c in k["candidates"]}
    for n in names:
        for suf in re.findall(r"_p(?:[A-Z][ws])+$", n.rsplit(".", 1)[-1]):
            assert f"`{suf}`" in text, suf
    for token in ("_zto", "_xfrom", "addi", "head-major", "TALL", "W1-b", "h1_plan", "h1_kernels", "x<X>y<Y>g<G>",
                  "s<S>", "attn_s<sx>x<sy>", "roofline", "f_pad"):
        assert token in text, token
    assert any(re.search(r"_x\d+y\d+g\d+_t", n) for n in names) and any(re.search(r"_s\d+g\d+_t", n) for n in names)


def test_stage_table_expert_split_and_one_mfu():
    ex = kx.Explorer()
    for p in ex.pages:
        rows = ex.stage_tables[p["name"]]
        stage = {r["stage"]: r for r in rows if r["kind"] == "stage"}
        step = next(r for r in rows if r["kind"] == "step")
        once = next(r for r in rows if r["kind"] == "call")
        assert abs(ex.dims.steps * step["ms"] + once["ms"] - stage["expert"]["ms"]) < 1e-6
        assert abs(sum(r["ms"] for r in stage.values()) - float(p["sums"]["total_ms"])) < 0.006
        # MFU: the operator-graph page's definition, to its printed digit
        for s in p["sums"]["stages"]:
            assert f"{100 * stage[s['stage']]['mfu']:.1f}" == s["mfu"]
    # the expert re-reads its weights every step: per-step reads at most the footprint
    w1 = {r["label"]: r for r in ex.stage_tables["W1"]}
    step = next(r for r in ex.stage_tables["W1"] if r["kind"] == "step")
    assert step["weights_mb"] <= w1["expert"]["footprint_mb"]
