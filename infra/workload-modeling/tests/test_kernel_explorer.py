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
                             m_size="64", n_size="768", k_size="768", strategy="splitK"))
    assert (s["active"], s["m_iter"], s["n_iter"], s["k_iter"]) == (64, 1, 1, 1)


def test_all_baseline_checks_pass():
    ex = kx.Explorer()
    bad = [c for c in ex.checks if not c["ok"]]
    assert not bad, bad
    kinds = {c["kind"] for c in ex.checks}
    assert kinds == {"pick", "constants", "placement", "sum"}
