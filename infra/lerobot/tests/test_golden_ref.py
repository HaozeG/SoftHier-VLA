"""Tests (a)-(c), (f) of design file 18 for golden_ref.py (published configuration, synthetic mode).

GPU tests skip without CUDA. Run: (cd infra/lerobot && uv run pytest), ideally under
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parent.parent / "golden_ref.py"
CHECKPOINT = "lerobot/smolvla_base"
REVISION = os.environ.get("SHVLA_TEST_REVISION", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def run_golden(*extra: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra], capture_output=True, text=True, timeout=timeout, check=False
    )


def generate(out: Path, obs: str) -> None:
    r = run_golden("--checkpoint", CHECKPOINT, "--revision", REVISION, "--obs", obs, "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def ref_a(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("golden") / "a.npz"
    generate(p, "synthetic:1:256:0")
    return p


@needs_cuda
def test_a_keys_shapes_dtypes(ref_a):
    d = np.load(ref_a, allow_pickle=False)
    expect = {
        "images": ((1, 3, 256, 256), np.float16),
        "vis_pos_ids": ((256,), np.int64),
        "lang_tokens": ((48,), np.int64),
        "lang_mask": ((48,), np.bool_),
        "state": ((32,), np.float32),
        "noise": ((50, 32), np.float32),
        "vis_out_0": ((256, 768), np.float32),
        "img_emb_0": ((16, 960), np.float32),
        "prefix_embs": ((65, 960), np.float32),
        "prefix_valid": ((65,), np.float32),
        "n_valid": ((), np.int64),
        "time_emb": ((10, 720), np.float32),
        "xt": ((11, 50, 32), np.float32),
        "vt": ((10, 50, 32), np.float32),
        "actions": ((50, 32), np.float32),
        "meta": ((5,), np.int64),
    }
    for layer in range(16):
        expect[f"kv_k_{layer}"] = ((65, 320), np.float32)
        expect[f"kv_v_{layer}"] = ((65, 320), np.float32)
    for k, (shape, dtype) in expect.items():
        assert k in d.files, k
        assert d[k].shape == shape, (k, d[k].shape)
        assert d[k].dtype == dtype, (k, d[k].dtype)
    assert d["meta"].tolist() == [50, 32, 10, 65, int(d["n_valid"])]
    assert np.array_equal(d["actions"], d["xt"][10])
    assert np.array_equal(d["xt"][0], d["noise"])
    assert int(d["prefix_valid"].sum()) == int(d["n_valid"])
    info = {k for k in d.files if k.startswith("info_")}
    assert {"info_checkpoint", "info_revision", "info_checkpoint_url", "info_lerobot_version"} <= info
    assert str(d["info_revision"]) == REVISION
    assert float(d["info_selfcheck_maxabs"]) <= 1e-5
    assert all(np.isfinite(d[k]).all() for k in d.files if d[k].dtype.kind == "f")


@needs_cuda
def test_b_byte_identical(ref_a, tmp_path):
    b = tmp_path / "b.npz"
    generate(b, "synthetic:1:256:0")
    assert sha256(ref_a) == sha256(b)


@needs_cuda
def test_c_seed_changes_inputs_and_actions(ref_a, tmp_path):
    c = tmp_path / "c.npz"
    generate(c, "synthetic:1:256:1")
    a, d = np.load(ref_a, allow_pickle=False), np.load(c, allow_pickle=False)
    assert not np.array_equal(a["images"], d["images"])
    assert not np.array_equal(a["actions"], d["actions"])


def test_f_unknown_obs_mode_fails():
    r = run_golden("--checkpoint", CHECKPOINT, "--revision", REVISION, "--obs", "bogus:1", "--out", "/tmp/x.npz")
    assert r.returncode != 0
    assert "unknown --obs mode" in r.stderr


def test_f_libero_not_implemented():
    r = run_golden("--checkpoint", CHECKPOINT, "--revision", REVISION, "--obs", "libero:s:0:0", "--out", "/tmp/x.npz")
    assert r.returncode != 0
    assert "G2.4" in r.stderr


def test_f_nonexistent_revision_fails(tmp_path):
    r = run_golden(
        "--checkpoint", CHECKPOINT, "--revision", "0" * 40, "--obs", "synthetic:1:256:0",
        "--out", str(tmp_path / "x.npz"), "--device", "cpu",
    )  # fmt: skip
    assert r.returncode != 0
    assert "cannot load" in r.stderr
    assert not (tmp_path / "x.npz").exists()
