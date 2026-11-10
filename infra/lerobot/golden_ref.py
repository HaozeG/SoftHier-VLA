#!/usr/bin/env python
"""Golden reference for the SmolVLA end-to-end baseline.

Runs a SmolVLA checkpoint in fp32 through LeRobot's own functions on one observation and writes every
intermediate that softhier-mlir's reference layout names (vision output, connector output, prefix
embeddings, per-layer prefix KV cache, time embeddings, flow states, actions) to one npz.

Independent of softhier-mlir code (decision A4). The layout follows softhier-mlir's
`smolvla_expert_ref.py` docstring; see docs/designs/smolvla-softhier-baseline.md ("Golden reference layout").

    source infra/lerobot/env.sh
    python infra/lerobot/golden_ref.py --checkpoint lerobot/smolvla_base --revision <sha> \
        --obs synthetic:1:256:0 --out $SHVLA_DATA/golden/published.npz

    --obs synthetic:<cams>:<side>:<seed>   seeded images in [-1, 1] at <side>, state, noise (softhier-mlir's RNG order)
    --obs from-npz:<path>                  images, lang_tokens, lang_mask, state, noise read from a reference npz
    --obs libero:<suite>:<task>:<seed>     (G2.4, not implemented yet)
    --crosscheck <ref.npz>                 compare the written keys with another reference npz and exit 1 at the
                                           first key above --tol (relative max error)

Everything runs in fp32 with deterministic settings (no TF32, deterministic algorithms), so two runs on the same
machine and library versions give byte-identical files.
"""

from __future__ import annotations

# Must be set before CUDA is initialised (torch may be imported first, but CUDA is initialised lazily).
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROMPT = "Pick up the red cube and put it in the box"
DEFAULT_REVISION_NOTE = "pass the Hub commit hash; a branch name is accepted but then not reproducible"
CROSSCHECK_TOL = 1e-4


class GoldenRefError(Exception):
    """A user-facing failure: printed without a traceback, exit code 1."""


# --------------------------------------------------------------------------------------------------------------
# Pipeline order of the keys (used by --crosscheck and by the writer)
# --------------------------------------------------------------------------------------------------------------
def key_order(cams: int, layers: int) -> list[str]:
    keys = ["images", "vis_pos_ids", "lang_tokens", "lang_mask", "state", "noise"]
    keys += [f"vis_out_{c}" for c in range(cams)]
    keys += [f"img_emb_{c}" for c in range(cams)]
    keys += ["prefix_embs", "prefix_valid", "n_valid"]
    for layer in range(layers):
        keys += [f"kv_k_{layer}", f"kv_v_{layer}"]
    keys += ["time_emb", "xt", "vt", "actions", "meta"]
    return keys


# --------------------------------------------------------------------------------------------------------------
# Deterministic setup
# --------------------------------------------------------------------------------------------------------------
def set_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


# --------------------------------------------------------------------------------------------------------------
# Observation modes
# --------------------------------------------------------------------------------------------------------------
def parse_obs(obs: str) -> tuple[str, list[str]]:
    mode, _, rest = obs.partition(":")
    if mode not in ("synthetic", "from-npz", "libero"):
        raise GoldenRefError(f"unknown --obs mode {mode!r} (expected synthetic, from-npz or libero)")
    if mode == "synthetic":
        parts = rest.split(":")
        if len(parts) != 3 or not all(p.lstrip("-").isdigit() for p in parts):
            raise GoldenRefError(f"--obs synthetic needs synthetic:<cams>:<side>:<seed>, got {obs!r}")
        return mode, parts
    if mode == "from-npz":
        if not rest:
            raise GoldenRefError("--obs from-npz needs from-npz:<path>")
        return mode, [rest]
    return mode, rest.split(":")


def tokenize(tokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        PROMPT,
        padding="max_length",
        padding_side="right",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"].bool()


def synthetic_inputs(cams: int, side: int, seed: int, tokenizer, cfg):
    """softhier-mlir's RNG order (smolvla_expert_ref.inputs): one CPU generator; per camera rand*2-1 rounded to
    fp16; then state[:6] = randn(6); then noise = randn(1, chunk, max_action_dim)."""
    torch.manual_seed(seed)  # softhier-mlir's main() also seeds the global RNG
    g = torch.Generator().manual_seed(seed)
    images = [(torch.rand(1, 3, side, side, generator=g) * 2 - 1).half().float() for _ in range(cams)]
    lang_tokens, lang_masks = tokenize(tokenizer, cfg.tokenizer_max_length)
    state = torch.zeros(1, cfg.max_state_dim)
    state[0, :6] = torch.randn(6, generator=g)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    return images, lang_tokens, lang_masks, state, noise


def npz_inputs(path: str):
    ref = np.load(path, allow_pickle=False)
    imgs = ref["images"]  # [cams, 3, H, W] fp16
    images = [torch.from_numpy(imgs[c : c + 1].astype(np.float32)) for c in range(imgs.shape[0])]
    lang_tokens = torch.from_numpy(ref["lang_tokens"].astype(np.int64))[None]
    lang_masks = torch.from_numpy(ref["lang_mask"].astype(bool))[None]
    state = torch.from_numpy(ref["state"].astype(np.float32))[None]
    noise = torch.from_numpy(ref["noise"].astype(np.float32))[None]
    return images, lang_tokens, lang_masks, state, noise


def vision_position_ids(side: int, patch: int, num_patches_per_side: int) -> np.ndarray:
    """Position-table rows of SmolVLM's vision embeddings for a full (unpadded) side x side image: the bucketize
    formula of SmolVLMVisionEmbeddings.forward (transformers/models/smolvlm/modeling_smolvlm.py)."""
    n = side // patch
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    frac = torch.clamp(torch.arange(n, dtype=torch.float32) * (1.0 / n), max=1.0 - 1e-6)
    b = torch.bucketize(frac, boundaries, right=True)
    return (b[:, None] * num_patches_per_side + b[None, :]).flatten().numpy().astype(np.int64)


# --------------------------------------------------------------------------------------------------------------
# Policy loading
# --------------------------------------------------------------------------------------------------------------
def load_policy(checkpoint: str, revision: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    try:
        cfg = PreTrainedConfig.from_pretrained(checkpoint, revision=revision)
        cfg.device = device
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg, revision=revision)
    except Exception as e:  # turn any Hub/loader failure into one readable line
        raise GoldenRefError(f"cannot load {checkpoint}@{revision}: {type(e).__name__}: {e}") from e
    policy = policy.to(device=device, dtype=torch.float32).eval()
    return cfg, policy


def check_weights(policy, checkpoint: str, revision: str) -> int:
    """The loaded fp32 weights must equal the checkpoint's tensors exactly (from_pretrained is strict=False and
    the VLM is built in its own dtype first, so verify instead of assuming)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(checkpoint, "model.safetensors", revision=revision)
    sd = policy.state_dict()
    n = 0
    with safe_open(path, "pt") as f:
        for k in f.keys():  # noqa: SIM118 (safetensors handle, not a dict)
            if k not in sd:
                raise GoldenRefError(f"checkpoint tensor {k} is missing from the loaded policy")
            t = f.get_tensor(k).to(torch.float32)
            if not torch.equal(t, sd[k].detach().cpu().to(torch.float32)):
                raise GoldenRefError(f"loaded weight {k} differs from the checkpoint")
            n += 1
    return n


# --------------------------------------------------------------------------------------------------------------
# The reference run
# --------------------------------------------------------------------------------------------------------------
def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().copy()


def run_reference(policy, cfg, images, lang_tokens, lang_masks, state, noise, device: str) -> dict:
    from lerobot.policies.common.vla_utils import (
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
    )

    m = policy.model
    we = m.vlm_with_expert
    vm = we.get_vlm_model()
    dev = torch.device(device)
    images = [im.to(dev) for im in images]
    lang_tokens, lang_masks = lang_tokens.to(dev), lang_masks.to(dev)
    state, noise = state.to(dev), noise.to(dev)
    img_masks = [torch.ones(im.shape[0], dtype=torch.bool, device=dev) for im in images]
    side = images[0].shape[-1]
    out: dict[str, np.ndarray] = {}

    out["images"] = torch.cat(images).half().cpu().numpy()
    out["lang_tokens"] = lang_tokens[0].cpu().numpy().astype(np.int64)
    out["lang_mask"] = lang_masks[0].cpu().numpy().astype(bool)
    out["state"] = to_np(state[0])
    out["noise"] = to_np(noise[0])
    emb = vm.vision_model.embeddings
    out["vis_pos_ids"] = vision_position_ids(side, emb.patch_size, emb.num_patches_per_side)

    with torch.no_grad():
        # 1. vision tower + connector per camera, with the position ids the tower really used
        seen_ids: list[torch.Tensor] = []
        hook = emb.position_embedding.register_forward_hook(lambda _m, i, _o: seen_ids.append(i[0].detach().cpu()))
        for c, img in enumerate(images):
            vis = vm.vision_model(pixel_values=img, patch_attention_mask=None).last_hidden_state
            img_emb = vm.connector(vis)
            out[f"vis_out_{c}"] = to_np(vis[0])
            out[f"img_emb_{c}"] = to_np(img_emb[0])
            assert torch.equal(img_emb, we.embed_image(img)), "manual vision + connector path != embed_image"
        hook.remove()
        for ids in seen_ids:
            assert np.array_equal(ids[0].numpy(), out["vis_pos_ids"]), "vis_pos_ids != position ids used by SigLIP"
        print(f"[golden] vision + connector == embed_image: OK; vis_pos_ids == tower's ids: OK ({len(seen_ids)} calls)")

        # 2. prefix and VLM forward filling the cache
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att2d = make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad, dim=1) - 1
        _, cache = we.forward(
            attention_mask=att2d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=cfg.use_cache,
        )
        lp = prefix_pad.shape[1]
        out["prefix_embs"] = to_np(prefix_embs[0])
        out["prefix_valid"] = to_np(prefix_pad[0].float())
        out["n_valid"] = np.array(int(prefix_pad.sum()), dtype=np.int64)
        kvh, hd = we.num_key_value_heads, vm.config.text_config.head_dim
        for layer in range(we.num_vlm_layers):
            k, v = cache.layers[layer].keys, cache.layers[layer].values  # [B, kv_heads, seq, head_dim]
            assert k.shape == (1, kvh, lp, hd) and v.shape == k.shape, (layer, tuple(k.shape))
            # [1, heads, seq, d] -> [seq, heads, d] -> [seq, heads * d]: head-major columns
            out[f"kv_k_{layer}"] = to_np(k.clone()[0].transpose(0, 1).reshape(lp, -1))
            out[f"kv_v_{layer}"] = to_np(v.clone()[0].transpose(0, 1).reshape(lp, -1))

        # 3. flow: x_t += dt * v_t (denoise_step crops the cache in place; KV was copied above)
        steps = cfg.num_steps
        dt = -1.0 / steps
        x_t = noise
        xt, vt, temb = [to_np(noise[0])], [], []
        for step in range(steps):
            t = torch.full((1,), 1.0 + step * dt, dtype=torch.float32, device=dev)
            temb.append(
                to_np(
                    create_sinusoidal_pos_embedding(
                        t, we.expert_hidden_size, cfg.min_period, cfg.max_period, device=dev
                    )[0]
                )
            )
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=prefix_pad, past_key_values=cache, timestep=t)
            x_t = x_t + dt * v_t
            vt.append(to_np(v_t[0]))
            xt.append(to_np(x_t[0]))
        out["time_emb"] = np.stack(temb)
        out["xt"] = np.stack(xt)
        out["vt"] = np.stack(vt)
        out["actions"] = xt[-1]
        out["meta"] = np.array([cfg.chunk_size, cfg.max_action_dim, steps, lp, int(prefix_pad.sum())], dtype=np.int64)

        # 4. self-check: LeRobot's own sample path on the same inputs and noise gives the same actions
        ref_actions = m.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
        diff = float((ref_actions[0].float().cpu() - torch.from_numpy(out["actions"])).abs().max())
        print(f"[golden] self-check: max|actions - model.sample_actions| = {diff:.3e}")
        if diff > 1e-5:
            raise GoldenRefError(f"self-check failed: captured flow differs from sample_actions by {diff:.3e}")
        out["_selfcheck_maxabs"] = np.array(diff)
    return out


def summarize(out: dict) -> None:
    layers = sum(1 for k in out if k.startswith("kv_k_"))
    kmax = max(float(np.abs(out[f"kv_k_{i}"]).max()) for i in range(layers))
    vmax = max(float(np.abs(out[f"kv_v_{i}"]).max()) for i in range(layers))
    a = out["actions"]
    print(f"[golden] prefix length {int(out['meta'][3])}, valid {int(out['n_valid'])}, KV {out['kv_k_0'].shape} x {layers} layers")
    print(f"[golden] |K| max {kmax:.3f}  |V| max {vmax:.3f}")
    print(f"[golden] actions: mean {a.mean():.5f} std {a.std():.5f} max|a| {np.abs(a).max():.4f}")


# --------------------------------------------------------------------------------------------------------------
# Crosscheck
# --------------------------------------------------------------------------------------------------------------
def crosscheck(mine: dict, ref_path: str, tol: float) -> int:
    ref = np.load(ref_path, allow_pickle=False)
    cams = sum(1 for k in mine if k.startswith("vis_out_"))
    layers = sum(1 for k in mine if k.startswith("kv_k_"))
    first = None
    print(f"[crosscheck] {ref_path}: max|a-b| / max(|b|.max(), 1e-6), tolerance {tol:g}")
    for key in key_order(cams, layers):
        if key not in mine or key not in ref.files:
            continue
        a, b = np.asarray(mine[key]), ref[key]
        if a.shape != b.shape:
            print(f"  {key:14s} SHAPE MISMATCH {a.shape} vs {b.shape}")
            err = float("inf")
        else:
            a, b = a.astype(np.float64), b.astype(np.float64)
            err = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6)) if a.size else 0.0
            print(f"  {key:14s} {err:.3e}{'  <-- above tolerance' if err > tol else ''}")
        if err > tol and first is None:
            first = (key, err)
    if first:
        print(f"[crosscheck] FAIL: first divergence at {first[0]} ({first[1]:.3e} > {tol:g})")
        return 1
    print("[crosscheck] PASS: every shared key within tolerance")
    return 0


# --------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------
def build_info(args, cfg, mode: str, device: str) -> dict[str, np.ndarray]:
    import lerobot
    import transformers

    obs = args.obs
    if mode == "from-npz":
        obs = "from-npz:" + Path(args.obs.partition(":")[2]).name
    info = {
        "info_checkpoint": args.checkpoint,
        "info_revision": args.revision,
        "info_checkpoint_url": f"https://huggingface.co/{args.checkpoint}/tree/{args.revision}",
        "info_vlm_model_name": str(cfg.vlm_model_name),
        "info_lerobot_version": lerobot.__version__,
        "info_transformers_version": transformers.__version__,
        "info_torch_version": torch.__version__.split("+")[0],
        "info_obs": obs,
        "info_prompt": PROMPT if mode == "synthetic" else "from npz",
        "info_device": device,
    }
    return {k: np.str_(v) for k, v in info.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Hub id or local directory")
    ap.add_argument("--revision", required=True, help=DEFAULT_REVISION_NOTE)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", help="output npz (required unless only --crosscheck output is wanted)")
    ap.add_argument("--crosscheck", metavar="REF_NPZ")
    ap.add_argument("--tol", type=float, default=CROSSCHECK_TOL)
    args = ap.parse_args(argv)
    try:
        return run(args)
    except (GoldenRefError, NotImplementedError) as e:
        print(f"golden_ref.py: error: {e}", file=sys.stderr)
        return 1


def run(args) -> int:
    t0 = time.time()
    mode, parts = parse_obs(args.obs)
    if mode == "libero":
        raise NotImplementedError("--obs libero is implemented in step G2.4; use synthetic or from-npz")
    if not args.out and not args.crosscheck:
        raise GoldenRefError("give --out and/or --crosscheck")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise GoldenRefError("--device cuda requested but CUDA is not available")
    if mode == "from-npz" and not Path(parts[0]).is_file():
        raise GoldenRefError(f"from-npz file not found: {parts[0]}")

    set_deterministic()
    cfg, policy = load_policy(args.checkpoint, args.revision, args.device)
    n = check_weights(policy, args.checkpoint, args.revision)
    print(f"[golden] loaded {args.checkpoint}@{args.revision} fp32 on {args.device}; {n} checkpoint tensors equal the loaded weights")

    if mode == "synthetic":
        cams, side, seed = (int(p) for p in parts)
        tok = policy.model.vlm_with_expert.processor.tokenizer
        inputs = synthetic_inputs(cams, side, seed, tok, cfg)
    else:
        inputs = npz_inputs(parts[0])

    out = run_reference(policy, cfg, *inputs, args.device)
    selfcheck = float(out.pop("_selfcheck_maxabs"))
    summarize(out)
    if args.device == "cuda":
        print(f"[golden] peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

    rc = 0
    if args.out:
        out.update(build_info(args, cfg, mode, args.device))
        out["info_selfcheck_maxabs"] = np.array(selfcheck, dtype=np.float64)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **out)  # uncompressed: deterministic bytes
        print(f"[golden] wrote {args.out} ({len(out)} arrays) in {time.time() - t0:.1f} s")
    if args.crosscheck:
        rc = crosscheck(out, args.crosscheck, args.tol)
    return rc


if __name__ == "__main__":
    sys.exit(main())
