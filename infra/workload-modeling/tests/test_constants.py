import pytest
import yaml

from workload_modeling import constants
from workload_modeling.constants import ConstantsError


@pytest.fixture(scope="module")
def c():
    return constants.load()


def _load_text(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return constants.load(p)


def test_rejects_missing_source(tmp_path):
    with pytest.raises(ConstantsError, match="missing source"):
        _load_text(tmp_path, "g:\n  x: {value: 1, unit: s, label: measured}\n")


def test_rejects_missing_label(tmp_path):
    with pytest.raises(ConstantsError, match="label"):
        _load_text(tmp_path, "g:\n  x: {value: 1, unit: s, source: here}\n")


def test_rejects_unknown_label(tmp_path):
    with pytest.raises(ConstantsError, match="derived"):
        _load_text(tmp_path, "g:\n  x: {value: 1, unit: s, label: derived, source: here}\n")


def test_rejects_unpinned_checkpoint(tmp_path):
    text = yaml.safe_dump({"workloads": {"w": {
        "checkpoint": "lerobot/smolvla_base", "cameras": 1, "image_size": 256, "language_tokens": 48,
        "state_tokens": 1, "label": "measured", "source": "x"}}})
    with pytest.raises(ConstantsError, match="pinned"):
        _load_text(tmp_path, text)


def test_all_entries_valid(c):
    assert c.entries, "no entries loaded"
    for key, e in c.entries.items():
        assert e.label in constants.LABELS, key
        assert e.source, key


def test_primary_workload(c):
    assert c.primary_workload.id == "libero-2cam"


def test_peaks_exact(c):
    assert c.peak_total("stage1") == 16.384e12
    assert c.peak_total("pr1") == 131.072e12


def test_iso_peak_mesh_is_16x16(c):
    n = c.iso_peak_mesh("stage1")
    assert n == 16
    assert c.peak_total("stage1", n * n) == pytest.approx(262.144e12)
    assert c.peak_total("stage1", 15 * 15) < c.value("thor.fp16_dense") * 1e12


def test_noc_link_bandwidth(c):
    assert c.noc_link_bytes_per_s("stage1") == 128e9
    assert c.noc_link_bytes_per_s("pr1") == 64e9


def test_rq_starting_points_exist(c):
    for key in ("pr1.core_op_share", "pr1.libero_softmax_share", "pr1.libero_rowop_share", "pr1.expert_step_ms",
                "pr1.expert_hbm_read_per_step", "pr1.expert_weight_bytes_per_step", "pr1.gemm_cluster_use",
                "pr1.e2e_ms", "pr1.libero_e2e_ms", "pr1.mfu"):
        e = c.get(key)
        assert e.workload in c.workloads and e.arch == "pr1", key
