"""Tests (a)-(f) of design file 18 for golden_ref.py: published configuration (synthetic mode) and LIBERO.

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

LIBERO_CHECKPOINT = "lerobot/smolvla_libero"
LIBERO_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
LIBERO_OBS = "libero:libero_spatial:0:0"

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


@pytest.mark.parametrize("obs", ["libero:s:0", "libero:s:x:0", "libero::0:0", "libero:s:0:0:0"])
def test_f_malformed_libero_obs_fails(obs):
    r = run_golden("--checkpoint", CHECKPOINT, "--revision", REVISION, "--obs", obs, "--out", "/tmp/x.npz")
    assert r.returncode != 0
    assert "--obs libero needs" in r.stderr


def test_f_nonexistent_revision_fails(tmp_path):
    r = run_golden(
        "--checkpoint", CHECKPOINT, "--revision", "0" * 40, "--obs", "synthetic:1:256:0",
        "--out", str(tmp_path / "x.npz"), "--device", "cpu",
    )  # fmt: skip
    assert r.returncode != 0
    assert "cannot load" in r.stderr
    assert not (tmp_path / "x.npz").exists()


# --------------------------------------------------------------------------------------------------------------
# LIBERO (G2.4): tests (a) for the LIBERO shapes, (d), (e)
# --------------------------------------------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT.parent))


@pytest.fixture(scope="session")
def libero_npz(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("golden_libero") / "libero.npz"
    r = run_golden("--checkpoint", LIBERO_CHECKPOINT, "--revision", LIBERO_REVISION, "--obs", LIBERO_OBS, "--out", str(p))
    assert r.returncode == 0, r.stdout + r.stderr
    return p


@needs_cuda
def test_a_libero_keys_shapes_dtypes(libero_npz):
    d = np.load(libero_npz, allow_pickle=False)
    expect = {
        "images": ((2, 3, 512, 512), np.float16),
        "vis_pos_ids": ((1024,), np.int64),
        "lang_tokens": ((48,), np.int64),
        "lang_mask": ((48,), np.bool_),
        "state": ((32,), np.float32),
        "noise": ((50, 32), np.float32),
        "prefix_embs": ((177, 960), np.float32),
        "prefix_valid": ((177,), np.float32),
        "n_valid": ((), np.int64),
        "time_emb": ((10, 720), np.float32),
        "xt": ((11, 50, 32), np.float32),
        "vt": ((10, 50, 32), np.float32),
        "actions": ((50, 32), np.float32),
        "meta": ((5,), np.int64),
        "info_state_raw": ((8,), np.float32),
    }
    for c in range(2):
        expect[f"vis_out_{c}"] = ((1024, 768), np.float32)
        expect[f"img_emb_{c}"] = ((64, 960), np.float32)
    for layer in range(16):
        expect[f"kv_k_{layer}"] = ((177, 320), np.float32)
        expect[f"kv_v_{layer}"] = ((177, 320), np.float32)
    for k, (shape, dtype) in expect.items():
        assert k in d.files, k
        assert d[k].shape == shape, (k, d[k].shape)
        assert d[k].dtype == dtype, (k, d[k].dtype)
    n_valid = int(d["n_valid"])
    assert d["meta"].tolist() == [50, 32, 10, 177, n_valid]
    assert int(d["prefix_valid"].sum()) == n_valid
    assert n_valid == 128 + int(d["lang_mask"].sum()) + 1
    assert np.array_equal(d["actions"], d["xt"][10])
    assert str(d["info_revision"]) == LIBERO_REVISION
    assert str(d["info_obs"]) == LIBERO_OBS
    assert "black bowl" in str(d["info_prompt"])
    assert float(d["info_selfcheck_maxabs"]) <= 1e-5
    assert all(np.isfinite(d[k]).all() for k in d.files if d[k].dtype.kind == "f")


@pytest.fixture(scope="session")
def libero_ctx():
    """Policy plus the LIBERO observation, built once in this process (same functions the CLI uses)."""
    import golden_ref as g

    g.set_deterministic()
    cfg, policy = g.load_policy(LIBERO_CHECKPOINT, LIBERO_REVISION, "cuda")
    lib = g.libero_inputs(policy, cfg, LIBERO_CHECKPOINT, LIBERO_REVISION, "libero_spatial", 0, 0, "cuda")
    yield g, cfg, policy, lib
    del policy
    torch.cuda.empty_cache()


@needs_cuda
def test_d_libero_actions_equal_predict_action_chunk(libero_ctx):
    """Golden `actions` (normalised space, first 7 dims) == policy.predict_action_chunk(batch, noise=noise).

    0.7.0 signature: predict_action_chunk(batch, *, noise=None, **rtc_kwargs) -> [B, n_action_steps, 7]. It runs
    prepare_images/prepare_state on the batch, so a batch with the golden's fp16-rounded images must be fed to
    make both sides see the same values: camera image = (golden_fp16 + 1) / 2 at 512x512 (resize_with_pad is then
    the identity, and `* 2 - 1` gives the rounded values back up to fp32 rounding, asserted below)."""
    g, cfg, policy, lib = libero_ctx
    out = g.run_reference(policy, cfg, lib["images"], lib["lang_tokens"].cpu(), lib["lang_masks"].cpu(),
                          lib["state"].cpu(), lib["noise"], "cuda")  # fmt: skip
    batch = dict(lib["batch"])  # shallow copy: _get_action_chunk overwrites batch keys with queue stacks
    for key, img in zip(("observation.images.camera1", "observation.images.camera2"), lib["images"]):
        batch[key] = (img.to("cuda") + 1) / 2
    policy.reset()
    imgs_back, _ = policy.prepare_images(dict(batch))
    roundtrip = max(float((a.cpu() - b.cpu()).abs().max()) for a, b in zip(imgs_back, lib["images"]))
    print(f"round-trip max|prepare_images - golden images| = {roundtrip:.3e}")
    assert roundtrip <= 1e-6
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch, noise=lib["noise"].to("cuda"))
    assert chunk.shape == (1, 50, 7)
    diff = float((chunk[0].cpu() - torch.from_numpy(out["actions"][:, :7])).abs().max())
    print(f"max|golden actions[:, :7] - predict_action_chunk| = {diff:.3e}")
    assert diff <= 1e-5


@needs_cuda
def test_e_libero_batch_state_and_rotation(libero_ctx):
    """State is 8-D before padding (and 0 after the 8th dim); camera1/2 equal the env's raw frames rotated by
    180 degrees (checked on one pixel row, before resize; exact: the same uint8 / 255 in fp32)."""
    _g, _cfg, _policy, lib = libero_ctx
    assert tuple(lib["state_raw"].shape) == (1, 8)
    assert tuple(lib["batch"]["observation.state"].shape) == (1, 8)
    assert tuple(lib["state"].shape) == (1, 32)
    assert bool((lib["state"][0, 8:] == 0).all())
    raw = lib["raw_obs"]["pixels"]
    for key, cam in (("observation.images.camera1", "image"), ("observation.images.camera2", "image2")):
        frame = np.asarray(raw[cam])  # [H, W, 3] uint8
        assert frame.dtype == np.uint8 and frame.ndim == 3, (frame.dtype, frame.shape)
        got = lib["batch"][key]  # [1, 3, H, W] float32 in [0, 1]
        assert tuple(got.shape) == (1, 3, frame.shape[0], frame.shape[1])
        want = torch.from_numpy(frame[-1, ::-1].copy()).permute(1, 0).float() / 255  # last row, reversed = row 0 rotated
        assert torch.equal(got[0, :, 0, :].cpu(), want), key
        # one more row and the whole image, still exact
        full = torch.from_numpy(frame[::-1, ::-1].copy()).permute(2, 0, 1).float() / 255
        assert torch.equal(got[0].cpu(), full), key
