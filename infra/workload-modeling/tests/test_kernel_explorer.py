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
    # the stage-1 pick of 1024x768x768 (tile 128x96) loses a quarter of the array to block fill
    c = next(c for c in ex.kernels["gemm_1024x768x768"]["candidates"] if c["cand"] == "Nosplit_s8g1_t128x96x128")
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
