# infra/lerobot

LeRobot 0.7.0 and LIBERO for the SmolVLA end-to-end baseline. Two programs:

- `golden_ref.py` runs a SmolVLA checkpoint in fp32 on one observation and writes every intermediate (vision output,
  connector output, prefix, per-layer KV cache, flow states, actions) to one `.npz`. It is the golden reference the
  SoftHier simulation is compared against.
- `libero_eval.py` runs `lerobot-eval` on LIBERO under a memory cap and reports the success rate with Wilson 95 %
  intervals.

Neither imports softhier-mlir.

## Setup

    (cd infra/lerobot && uv sync)        # about 10 minutes the first time, mostly CUDA wheels
    source infra/lerobot/env.sh          # activates .venv, sets the variables below

`uv sync` installs Python 3.12, lerobot 0.7.0, transformers 5.22.2, torch 2.15.0 (cu134 wheels), mujoco, robosuite and
hf-libero. The CUDA index is set in `pyproject.toml`.

`env.sh` sets:

- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`: Mesa's LLVM (osmesa) needs the system libstdc++; an older
  one earlier on the library path breaks it.
- `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`: MuJoCo renders the LIBERO cameras on the CPU with Mesa software
  OpenGL (WSL2 has no usable EGL here). Needs the apt packages `libosmesa6` and `libglfw3`.
- `LIBERO_CONFIG_PATH=$SHVLA_DATA/libero`. The first `import libero.libero` writes `config.yaml` and asks for input()
  when it is missing, which hangs a background job. `env.sh` creates the file once (`echo n | python -c "import
  libero.libero"`). The file holds absolute paths into the current `.venv`: after moving the worktree or rebuilding the
  venv, delete `$SHVLA_DATA/libero` and source `env.sh` again.

## Memory cap and one job at a time

Do not run LeRobot jobs in parallel on this machine (WSL crashed once with an eval and a training run together). Run
GPU jobs under a cap:

    systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 <command>

`libero_eval.py` does this itself (`--no-cap` turns it off). Measured peak: golden reference about 3.8 GiB RSS and
1.8 GiB GPU; one eval process about 5 GiB RSS.

## golden_ref.py

    python infra/lerobot/golden_ref.py --checkpoint <hub id or dir> --revision <commit hash> \
        --obs <mode> --out <file.npz> [--device cuda|cpu] [--crosscheck <ref.npz> [--tol 1e-4]]

`--revision` is required; pass the Hub commit hash so the result is reproducible. The loaded fp32 weights are compared
tensor by tensor with `model.safetensors` before the run (`from_pretrained` is not strict).

`--obs` modes:

| mode | input |
| --- | --- |
| `synthetic:<cams>:<side>:<seed>` | seeded images in [-1, 1] at `<side>`, a fixed prompt, a seeded state and noise; the RNG order of softhier-mlir's `smolvla_expert_ref.py` |
| `from-npz:<path>` | images, language tokens, state and noise read from another reference npz (used by the crosscheck) |
| `libero:<suite>:<task>:<seed>` | the first frame after `env.reset(seed)` of one LIBERO task, through the same chain `lerobot-eval` uses; noise from `torch.Generator().manual_seed(seed)` |

Examples (the two checkpoints used by the baseline):

    # published configuration: 1 camera, 256 x 256
    python infra/lerobot/golden_ref.py --checkpoint lerobot/smolvla_base \
        --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --obs synthetic:1:256:0 --out $SHVLA_DATA/golden/published.npz

    # LIBERO: 2 cameras at 512 x 512, libero_spatial task 0, seed 0
    python infra/lerobot/golden_ref.py --checkpoint lerobot/smolvla_libero \
        --revision a24f1f380461b665158d52acd140152c79e8b36d --obs libero:libero_spatial:0:0 --out $SHVLA_DATA/golden/libero.npz

LIBERO mode, in the order `lerobot-eval` 0.7.0 does it: `env.reset(seed=[seed])` on the env built from
`LiberoEnvConfig(task=<suite>, task_ids=[<task>])` (defaults: 360 x 360 cameras, 20 Hz); the sim already returns
canonical observations (images rotated 180 degrees, 8-D state of eef position, axis-angle and gripper), so
`prepare_canonical_observation` is used, not `preprocess_observation`; the task string from `env.call("task_description")`;
the env preprocessor (empty for LIBERO in 0.7.0); the checkpoint's saved policy preprocessor with the same overrides
as eval (rename `image` to `camera1` and `image2` to `camera2`, tokenizer, normalisation); then the policy's
`prepare_images` (512 x 512 with padding, [-1, 1]) and `prepare_state` (padded to 32). Images are then rounded to fp16
and the rounded values are fed to the model. `info_prompt` holds the env's task text and `info_state_raw` the 8-D state
before normalisation.

Output keys, shapes and the KV layout are in `docs/designs/smolvla-softhier-baseline.md` ("Golden reference layout").
The LIBERO file has `images [2,3,512,512]` fp16, `vis_pos_ids [1024]`, `vis_out_<c> [1024,768]`, `img_emb_<c> [64,960]`,
prefix length 177, `kv_k_<l>`/`kv_v_<l>` `[177,320]` for 16 layers and `meta = [50, 32, 10, 177, n_valid]`.

Self-checks on every run (the run fails if one fails): the hand-written vision + connector path equals
`embed_image`; `vis_pos_ids` equals the ids SigLIP's position embedding actually received (forward hook); the captured
flow equals `model.sample_actions` on the same inputs and noise (max-abs at most 1e-5; it is 0 in practice).

`--crosscheck <ref.npz>` compares every shared numeric key in pipeline order by `max|a-b| / max(|b|.max(), 1e-6)` and
exits 1 if one is above `--tol` (default 1e-4), printing the first diverging key.

### Determinism

fp32 only; `torch.use_deterministic_algorithms(True)`, TF32 off, `cudnn.benchmark` off,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; the npz is written uncompressed with no timestamps. Two runs on the same machine
and library versions give byte-identical files (the LIBERO file too, including the osmesa rendering). Note this
differs from `lerobot-eval`, which turns TF32 and `cudnn.benchmark` on, and golden_ref.py builds the policy with
`from_pretrained` rather than `make_policy`.

### Speed

About 9 s per run once the checkpoint is cached (CPU and GPU alike; the CPU run is exact enough for the crosscheck);
the LIBERO run takes about 20 s including building the env. The first run downloads the checkpoint and
SmolVLM2-500M-Video-Instruct (config and tokenizer, not pinned).

## libero_eval.py

    python infra/lerobot/libero_eval.py --run-name <new name> [--suite libero_spatial] [--task-ids 0 1 ...] \
        [--n-episodes 10] [--batch-size 1] [--policy-path lerobot/smolvla_libero] [--revision <hash>] \
        [--device cuda] [--seed 1000] [--no-cap] [--dry-run]

It builds and runs this `lerobot-eval` command (flags checked against the 0.7.0 source):

    lerobot-eval --env.type=libero --env.task=<suite> --env.task_ids='[0]' --eval.n_episodes=<n> \
        --eval.batch_size=<b> --policy.path=lerobot/smolvla_libero --policy.pretrained_revision=<hash> \
        --policy.device=cuda --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
        --seed=<seed> --output_dir=$SHVLA_DATA/eval/<run-name>

`--n-episodes` is per task (10 tasks x 10 episodes is `--n-episodes 10`). The run name must be new. Results are
`eval_info.json` (from LeRobot) and `summary.json` (success per task and overall with Wilson 95 % intervals computed
from the per-episode outcomes) in `$SHVLA_DATA/eval/<run-name>/`, plus videos.

Speed: about 6.5 steps/s per episode with one env (rendering on the CPU dominates); a 1 task x 1 episode run took 111 s
wall, about 5 GiB largest-process RSS.

## Tests

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 uv run --directory infra/lerobot pytest -q tests

12 tests, about 100 s. The ones that run the model skip without CUDA. They cover keys, shapes and dtypes (published
and LIBERO), byte-identical repeat runs, seed sensitivity, golden actions equal to `policy.predict_action_chunk`,
the LIBERO batch (8-D state, images equal to the env frame rotated by 180 degrees), and error messages for bad input.
