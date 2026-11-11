# SmolVLA on SoftHier: simulator, golden reference and first latency and success-rate baseline

*   **Author:** Haoze
*   **Status:** Proposed (revision 3, after the second implementation-readiness review)
*   **Date:** 2026-11-10

--------------------------------------------------------------------------------

## Overview

SoftHier-VLA (README) has three goals: an MLIR-based deployment toolchain, workload analysis, and high MFU (model FLOPs utilisation: the share of the chip's peak arithmetic rate a model actually uses) through a software scheduling layer. Today the repo holds only knowledge (the `softhier_vla_wiki/` submodule) and working notes. Nothing in it can run a VLA model on SoftHier, check that a run is correct, or measure whether a policy completes its task.

Terms used throughout:

| term | meaning |
| --- | --- |
| SoftHier | a tile-based many-core accelerator model: a 2-D grid of clusters with HBM (high-bandwidth memory) on the edges ([wiki](../../softhier_vla_wiki/knowledge/softhier-and-the-project.md)) |
| GVSoC | the event-based simulator SoftHier runs in; branch `soft_hier_release` of `github.com/gvsoc/gvsoc` |
| `flex_cluster` | the name of the SoftHier chip model and its software kit (SDK) on that branch |
| RedMulE, Spatz, iDMA | per cluster: the matrix-multiply engine, the vector unit, and the DMA engine that moves data between HBM and local memory |
| TCDM | a cluster's local scratchpad memory (L1) |
| HBM placement `(4,0,0,4)` | HBM channels per chip edge in the order west, north, east, south: 4 on the west and 4 on the south edge |
| softhier-mlir | the reference compiler `github.com/runwangdl/softhier-mlir`: an xDSL/MLIR dialect, lowering to C, and an operator library, with a SmolVLA end-to-end flow |
| 4-program chain | softhier-mlir runs one SmolVLA inference as four simulator programs that hand data over through HBM: V (vision tower and connector), Pa (prefix layers 1–8), Pb (prefix layers 9–16), X (action expert, 10 flow steps). Stages in this document: vision = V, prefix = Pa + Pb, expert = X |
| golden reference | the fp32 GPU run of LeRobot's SmolVLA whose intermediate tensors and actions SoftHier runs are compared against |
| softhier-mlir's ledger | its document `docs/SMOLVLA_E2E.md`, which records the published end-to-end runs and costs |
| `wiki://...` | a path inside the wiki submodule `softhier_vla_wiki/` |

The project tracks two metrics for a VLA deployment, both taken from sources already in the wiki (wiki decision recorded in G0):

*   **latency**: the time of one model call (glossary: latency (l)). Jung et al. and VLA-HWBench profile VLA calls per stage (`wiki://knowledge/vla-benchmark-workloads.md`).
*   **success rate**: the share of benchmark episodes in which the task is completed (glossary). SmolVLA reports it on LIBERO (`wiki://knowledge/simulation-benchmarks.md`). Jung et al. §5 show that the two interact in closed loop: more overlap of inference and execution shortens episodes and lowers success (`wiki://resources/serving/vla-workload-characterization-kaist.md`).

This design adds the first executable pieces in two layers:

1.  **Permanent infrastructure on `main`**, reused by later experiments:
    *   the SoftHier simulator: GVSoC `soft_hier_release` as a git submodule, checked with its own kernel tests (`third_party/gvsoc/implementation/README.md` once checked out);
    *   a golden reference and a LIBERO environment: one uv project with LeRobot on the local GPU. It also runs the closed-loop success-rate evaluation. Its setup follows `~/learn/physical_ai`, an earlier local LeRobot exercise on this machine: uv, Python 3.12, MuJoCo rendered with osmesa, the system libstdc++ preloaded, one job at a time under a memory cap.
2.  **One experiment in a git worktree** (`worktree-smolvla-e2e-baseline`): softhier-mlir (latest `main`, recorded as a commit permalink), its hardware configuration, its runs, and their records.

## The Problem

Every performance and accuracy statement in the project so far comes from papers about GPUs or other chips. Nothing says what one SmolVLA inference costs on SoftHier, where that time goes, whether a compiled program computes the right actions, or what success rate the same policy reaches. Without that baseline, toolchain and scheduling decisions rest on paper figures, later work has no reference for speed, correctness or task success, and every experiment rebuilds its environment.

softhier-mlir reports one SmolVLA inference end to end on the SoftHier GVSoC: 119.1 ms simulated for 1 camera at 256 SigLIP tokens, actions within max-abs 0.032 of LeRobot. It was developed on a private simulator fork with unpublished model patches, an aarch64 host and hardcoded paths. Its numbers have not been reproduced in an environment this project controls, and it has never run the configuration of a robot benchmark end to end.

The proposed solution: record why these two metrics in the wiki; build the simulator and the LeRobot environment once on `main`; reproduce softhier-mlir's published run on them; run the same chain in the LIBERO configuration of a fine-tuned checkpoint; and measure that checkpoint's success rate on GPU.

## Goals

### Dependencies

```text
G0 wiki decision + design on main
 |
 +--> G1 simulator + kernel tests ----------------------------+
 |                                                            |
 +--> G6a LIBERO env --> G2 golden reference --> G6b first    |
                         (published + LIBERO)    success rate |
                              |                  (10 x 10)    |
                              v                               v
                         G3 numerical match (published configuration)
                              |
                              v
                         G4 LIBERO latency: G4a staged estimate --> G4b full run
                              |
                              v
                         G5 record          G6c full success-rate sweep (optional)
```

G1 and the G6a → G2 → G6b branch can run at the same time; G3 needs both. G6b early settles two facts the LIBERO configuration needs (state dimension, observation keys). The simulator and the GPU jobs run one at a time on this host.

### Goal list

*   **G0: design needs in the wiki.** A project decision in `memories/decisions/` records:
    *   the two metrics and their sources;
    *   GVSoC `soft_hier_release` and softhier-mlir, and why softhier-sdk is not used;
    *   the layout rule: parts that later experiments reuse (simulator, golden reference, LIBERO evaluation, environment scripts) live on `main` under `third_party/` and `infra/`; everything specific to one experiment (its compiler checkout, arch file, configs, patches, records) lives under `experiments/<name>/` on that experiment's branch;
    *   the LIBERO configuration;
    *   success-rate method SR-A;
    *   SR-B and SR-C as open later options with their requirements.
    This design is saved under `docs/designs/` and linked from the decision.
*   **G1: simulator, checked with its own tests.** A fresh checkout initialises the GVSoC submodule, builds it with one documented procedure, and passes the kernel tests of `implementation/` one at a time: `gemm`, `norm`, `acti`, `attn`.
*   **G2: golden reference.** A small independent program in the LeRobot environment (decision A4) runs `SmolVLAPolicy` in fp32 on a fixed observation and writes the tensors in softhier-mlir's reference layout (key names and shapes of its `smolvla_expert_ref.py`). It covers two configurations: published (`lerobot/smolvla_base`, 1 camera, 256 SigLIP tokens, synthetic seeded input) and LIBERO (`lerobot/smolvla_libero`, the observation from the LIBERO environment). Once, on the synthetic input, its output is cross-checked against softhier-mlir's own reference script `smolvla_expert_ref.py`, ported to the current LeRobot and transformers API (patch 0003) and run on the same inputs in the same environment.
*   **G3: numerical match, published configuration.** softhier-mlir's flow runs on the G1 simulator and is checked rung by rung (the ladder table in the Technical Plan), ending with the full chain compared against G2. Without the private simulator patches, failures are expected. Each failure is traced with the side-note table, then fixed (a commit on the GVSoC fork, or a patch on softhier-mlir) or recorded as a known deviation with its size. G3 ends at the **validity gate**:
    *   total simulated time within **±5 %** of 119.1 ms;
    *   action-chunk max-abs error **≤ 0.10** against G2 (published: 0.032).
    If the gate is missed, G4 still runs, and its results are reported as "environment deviates from published by X", naming the suspected side-note items. G3 has no overall time-box. Every single simulator process (one phase of the chain, or one standalone simulation) is limited to **2 hours** of wall time; a process that reaches the limit is stopped, and the session asks the owner before going on (R2).
*   **G4: LIBERO latency.** The same chain in the LIBERO configuration (decision A1: staged):
    *   **G4a, staged estimate:** each stage kind is run partly at LIBERO shapes: one SigLIP layer at 1024 tokens per camera, prefix layers 1–8 at 177 tokens, a 2-step expert flow at prefix length 177. The per-stage times are measured and the full-chain time is composed (labelled **estimate**). The measured wall-time ratio gives a projection of the full run. G4a measures time only: the vision and prefix partial runs check against softhier-mlir's own fp16 references, and the expert run reads G2's KV and compares its 2 steps with G2 `xt[1..2]`. The full numerical comparison with G2 is in G4b.
    *   **G4b, full run:** the whole 4-program chain at LIBERO shapes, every stage simulated (labelled **measured, simulated**), compared with G2 at every hand-over.
    *   Deliverables: time per stage and total; action error against G2 (LIBERO); RedMulE and iDMA busy time per stage from softhier-mlir's `--e2e-trace` mode, optional if trace mode fails. The setup and the input configuration (which robot each value comes from) are documented.
*   **G5: record.** RESULTS.md in the experiment lists:
    *   the code as commit permalinks (`https://github.com/<owner>/<repo>/tree/<commit>`);
    *   the arch file;
    *   every command;
    *   wall time and peak memory;
    *   every local change;
    *   the G3 gate outcome;
    *   the G4a and G4b numbers with their labels.
*   **G6: success rate on GPU, method SR-A.**
    *   **G6a:** the LIBERO environment is installed.
    *   **G6b:** `lerobot/smolvla_libero` is evaluated closed loop on `libero_spatial`, 10 tasks × 10 episodes, with a synchronous policy (simulation time pauses during inference). The success rate is reported with a 95 % binomial confidence interval and set next to the published `libero_spatial` figure if one exists.
    *   **G6c (optional):** a larger sweep, decided by the owner after G6b.

Labels follow wiki-design decision 0004 (evidence tiers): simulator cycle counts and evaluation success rates are runs, so they are **measured**, qualified as "simulated on GVSoC, ideal HBM" or "GPU fp32, synchronous eval". Composed figures (G4a) and modelled figures are **estimate**.

**What the latency number includes.** It is the sum of the timed segments of the four programs at 1 GHz: the compute of vision, connector, prefix and expert. It excludes data movement between programs (handed over by the simulator's preload path, not simulated), program load and launch, real HBM timing (ideal HBM is used), and host-side preprocessing (image resize, tokenisation, normalisation).

## Non-Goals

*   **pulp-platform/softhier-sdk.** It targets the newer `pulp.chips.softhier` model, which has no RedMulE and none of the runtime headers softhier-mlir uses.
*   **softhier-mlir's private simulator patches** are not applied up front. The side-note table lists what each fixes, so symptoms can be recognised and fixed when they appear.
*   **Success rate with SoftHier effects (SR-B, SR-C).** Recorded in the wiki as later options:
    *   SR-B, latency-injected closed loop after Jung et al. §5: simulation time keeps running during inference, with an action queue, overlap and stale actions;
    *   SR-C, SR-B plus a numerical twin of SoftHier's fp16 arithmetic.
*   **Closed-loop episodes on the SoftHier simulator.** Infeasible at current simulator speed (Risk R1).
*   **Other configurations:** softhier-mlir's 3-camera configurations, other robots or benchmarks, DRAMSys (non-ideal HBM).
*   **softhier-mlir's design-space exploration** (`dse/`), and changes to its algorithms or kernels beyond what the LIBERO configuration needs (listed in the Technical Plan).
*   **Forking softhier-mlir.** Its latest `main` is used, and local changes are patch files (decision D2).
*   **Baseline numbers in the wiki** (wiki-design decision 0013 keeps derived results out of the wiki). They stay in RESULTS.md with a mnemon pointer.

## Risks

| id | risk | trigger | response |
| --- | --- | --- | --- |
| R1 | G4b is infeasible in wall time | the G4a projection puts any G4b phase above the **2 h per-process limit** (expected for the vision phase: 2 cameras × 12 layers at 1024 tokens) | stop and ask the owner before G4b: raise the limit for that run, report G4a only, or reduce scope |
| R2 | a simulator process runs too long or does not converge | a process reaches the 2 h limit, or G3 keeps failing at one rung | consult the reference notes listed under the side-note table (host speed, patch procedures), then stop and ask the owner with a proposed fix; G4 can run with "deviates by X" |
| R3 | softhier-mlir lacks something the LIBERO shapes need | a G4a partial run fails to emit, build or fit (HBM, TCDM, 64 KB instruction memory) | patch per the LIBERO changes list; a change outside that list is designed in LOG.md first and reported to the owner before applying |
| R4 | upstream drift | a softhier-mlir or GVSoC update breaks a patch or changes results | submodules are frozen from G3 to G5; bumps happen only between experiments |

## Open Questions

| id | question | status |
| --- | --- | --- |
| OQ1 | Does lerobot 0.7.0 load `lerobot/smolvla_libero` at its recorded revision, or is 0.6.1 needed? | settled in G6a |
| OQ2 | LIBERO state dimension (8 per the checkpoint config) and the exact observation keys after LeRobot's LIBERO processing | settled in G6b |
| OQ3 | Control rate: `LiberoEnv.fps = 20` in LeRobot ("must match robosuite's control_freq"), while the checkpoint's `train_config.json` says 30 | settled in G6b from the running env; only affects the chunk-duration figure, not latency |
| OQ4 | Published `libero_spatial` success rate for `lerobot/smolvla_libero`, if any exists | looked up in G6b |

Resolved:

| id | decision |
| --- | --- |
| Q1 | GVSoC as a git submodule |
| Q2 | HBM placement `(4,0,0,4)` |
| Q3 | uv environments with tracked `pyproject.toml` and `uv.lock` |
| Q4 | Hugging Face cache plus a path variable; checkpoint revisions recorded |
| Q5 | commit permalinks instead of bare hashes |
| Q6 | softhier-mlir paths: command-line flags where they exist, patch 0001 for the rest |
| Q7 | LIBERO setup as in the configuration table (owner-confirmed) |
| Q8 | latency measured in the same configuration as the robot evaluation |
| Q9 / D1 | one simulator build per worktree, rebuilt per arch, stamp-guarded |
| D2 | simulator fixes in a GVSoC fork; softhier-mlir changes as patch files |
| A1 | G4 staged: estimate first, then the full run (with the R1 stop rule) |
| A2 | softhier-mlir's LIBERO changes designed up front (Technical Plan) |
| A4 | golden reference: thin, independent, on `main`, in softhier-mlir's layout |
| XC | G2 crosscheck (following the owner's suggestion to raise the script to transformers ≥ 5.4; revertible): softhier-mlir's `smolvla_expert_ref.py` is ported to the current LeRobot/transformers API (patch 0003) and run in the same environment, rather than a third environment with the old API or dropping the check |
| SR | success rate by SR-A first; SR-B and SR-C recorded as later options |
| A6 | G3 validity gate: total within ±5 % of 119.1 ms and action max-abs ≤ 0.10 |
| limits | 2 h wall time per simulator process, then stop and ask; no overall G3 time-box; G4b only if every projected phase fits in 2 h, otherwise ask |

## Technical Plan (Architecture)

### Summary

`main` holds the permanent parts: `third_party/gvsoc` (submodule), `infra/softhier/` (build and test scripts), `infra/lerobot/` (golden reference and LIBERO evaluation), `infra/env.sh`, and `docs/designs/`. The experiment branch adds `experiments/smolvla-e2e-baseline/` with softhier-mlir as a submodule, its arch file, configuration files, one run wrapper, patches, and the documents. Large data stays outside git in the Hugging Face cache (`HF_HOME`) and a data folder (`SHVLA_DATA`).

```text
SoftHier-VLA repo
+-- main (permanent) -------------------------------------------------+
| softhier_vla_wiki/   submodule: decision (G0)                       |
| docs/designs/        this design                                    |
| third_party/gvsoc/   submodule: GVSoC soft_hier_release (fork, D2)  |
| infra/env.sh         paths and defaults                             |
| infra/softhier/      setup, build_arch, kernel tests, stamp (G1)    |
| infra/lerobot/       uv env: golden_ref (G2), libero_eval (G6)      |
+---------------------------------------------------------------------+
            | branch worktree-smolvla-e2e-baseline adds
            v
+-- experiments/smolvla-e2e-baseline/ --------------------------------+
| softhier-mlir/       submodule: runwangdl/softhier-mlir main        |
| arch/                softhier-mlir's flex_cluster arch              |
| configs/             published.env, libero.env (paths, shapes)      |
| run.sh               fetch / prepare / sim: stamp, patches, limit   |
| gate.py              G3 validity gate, G4a composition              |
| patches/             changes to softhier-mlir, with README          |
| LOG.md SETUP.md INPUT_CONFIG.md RESULTS.md                          |
+---------------------------------------------------------------------+
outside git:  $HF_HOME (checkpoints)   $SHVLA_DATA (npz, ELFs, logs)
```

### Data flow of one chain run

```text
LIBERO env or seeded synthetic input              (infra/lerobot)
        |
        v
golden_ref.py (GPU fp32) --> ref.npz in softhier-mlir's reference layout
        |                     (images, lang, state, noise, VIS, IMG,
        |                      KV x16, x_t x11, actions)
        v
softhier-mlir (experiment, patched)
  weights per checkpoint: smolvla.py prepare / prepare-vlm /
                          smolvla_expert.py prepare  -> $SHVLA_DATA/<ckpt>/
  smolvla_e2e.py prepare --ref ref.npz               -> e2e.npz
  tests/gvsoc/run.py smolvla-e2e --npz e2e.npz
        |  4 programs V -> Pa -> Pb -> X, HBM hand-over by preload
        |  each hand-over compared with the ref.npz keys (built into run.py)
        v
per-phase logs + summary.json --> run.sh prints a record block; gate.py checks
                                  LOG.md and RESULTS.md are written by hand from these
```

### Components

1.  **GVSoC submodule** (`third_party/gvsoc`, branch `soft_hier_release`, recursive).
    *   Built with its Makefile: `softhier_preparation` (applies its own patches to the nested `core` and `pulp` submodules, builds SystemC and DRAMSys, fetches the RISC-V toolchain), then `make hw` with an arch file.
    *   Build output is untracked; `.gitmodules` sets `ignore = dirty`.
    *   Every worktree has its own checkout and build (D1).
2.  **`infra/softhier/`**:
    *   `setup.sh`: prerequisite check, submodule init, `softhier_preparation`.
    *   `build_arch.sh <arch.py>`: `make hw`, then write the stamp.
    *   `run_kernel_tests.sh`: the four kernel tests. Whether a kernel's make targets rebuild the hardware with `implementation/`'s own arch file is read from `soft_hier_release`'s `implementation/Makefile` in G1.5 (in an older local copy, `<k>-run` neither rebuilds nor preloads, and `<k>-pre` does both). If any target rebuilds, the script rewrites the stamp for that arch at the end, so the next experiment run sees a mismatch and needs `build_arch.sh` first.
    *   `stamp.py`: write, check, permalinks.
    *   A uv project with GVSoC's Python requirements (including `prettytable` and `pyelftools`, which `install/bin/gapy` imports through `#!/usr/bin/env python3`).
3.  **`infra/lerobot/`**: one uv project (Python 3.12, `lerobot[smolvla,libero]`, CUDA torch) with two programs.
    *   `golden_ref.py` (decision A4), independent of softhier-mlir's code:
        *   builds the observation in one of three modes:
            *   `synthetic`: seeded images in [-1, 1] at the given side, bypassing `prepare_images` as softhier-mlir's reference does, a fixed prompt, a seeded 6-of-32 state, seeded noise. The RNG order is softhier-mlir's (`smolvla_expert_ref.inputs`): one CPU `torch.Generator(seed)`; per camera `rand*2-1` then `.half().float()`; then `state[:6] = randn(6)`; then noise `randn(1, 50, 32)`; prompt "Pick up the red cube and put it in the box".
            *   `from-npz`: images, language tokens and mask, state and noise read from an existing reference npz. Used by the crosscheck, so the two programs see identical inputs without relying on RNG order.
            *   `libero`: the LIBERO env is reset with a seed and the first frame is passed through exactly the steps `lerobot-eval` applies, using LeRobot's own functions: `preprocess_observation`, the env preprocessor from `make_env_pre_post_processors` (its `LiberoProcessorStep` rotates both images 180° and builds the 8-D state from end-effector position, axis-angle and gripper), the rename map, then the policy preprocessor (normalisation, tokenizer) and the policy's `prepare_images` and `prepare_state`. Noise is `randn(1, 50, 32)` from a CPU `torch.Generator(seed)`.
            In every mode, images are rounded to fp16 and the fp32 model sees the rounded values, so the device input and the reference input are the same numbers.
        *   runs the VLM and the 10 flow steps in fp32 through LeRobot's own functions: `embed_prefix`; the VLM forward with `use_cache=True, past_key_values=None`, which returns a `DynamicCache`; `denoise_step`;
        *   writes softhier-mlir's keys (below).
        With deterministic settings (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off), two runs on the same machine and library versions give byte-identical files. `--device cpu` runs the same code on the CPU (used by the crosscheck).
    *   `libero_eval.py`: a thin wrapper around `lerobot-eval` with `--env.type=libero`, `--env.task`, `--eval.n_episodes`, `--eval.batch_size`, `--policy.path` and revision, `--rename_map` (`observation.images.image` → `observation.images.camera1`, `observation.images.image2` → `observation.images.camera2`), and `--output_dir`. It reads `eval_info.json` and writes the success rate with a Wilson 95 % interval.
4.  **softhier-mlir submodule** (`experiments/smolvla-e2e-baseline/softhier-mlir`, branch `main`), frozen from G3 to G5 (R4). Its uv project is defined by the experiment (xdsl, numpy, CPU torch, transformers, safetensors, filecheck, lit).
5.  **`run.sh`** (experiment): the only run tool, with four subcommands. `fetch`, `prepare` and `sim` source `infra/env.sh` and a config (`configs/published.env` or `configs/libero.env`), activate the experiment's uv environment, and apply the patches. Applying means `git -C softhier-mlir checkout -- .` (tracked files only, so outputs that softhier-mlir writes inside its own tree are kept), then `git apply patches/*.patch` in order; a patch that fails to apply exits with code 2.
    *   `run.sh <config> fetch`: `huggingface-cli download <checkpoint> --revision <revision>` into `HF_HOME`, then symlinks the snapshot's `model.safetensors` and `config.json` into `$SHVLA_SMOLVLA_DIR`.
    *   `run.sh apply`: only resets and patches the softhier-mlir checkout, activating no environment (used before G2.3, which runs a patched file in the LeRobot environment).
    *   `run.sh <config> prepare`: runs the config's host prepare commands in order (file 26), each with explicit `--ckpt` and `--out`. No simulator, no stamp check.
    *   `run.sh <config> sim <run-name> -- <script> <args>`:
        1. Refuses (exit 2) if `stamp.py check` fails against `arch/softhier_mlir_arch.py`; the fix is to run `build_arch.sh` explicitly.
        2. Runs `python <script> <args>` from the softhier-mlir root. `<script>` is `tests/gvsoc/run.py` or `tests/gvsoc/expert.py`. Every other script or subcommand runs under coreutils `timeout $SHVLA_SIM_TIMEOUT`, since each is a single simulator process. For `run.py smolvla-e2e`, `run.sh` instead appends `--app-dir $SHVLA_DATA/runs/<config>/<run-name>` and `--timeout $SHVLA_SIM_TIMEOUT` (patch 0004). The output is teed to `<app-dir>/run.log`.
        3. **Limit:** each GVSoC process gets `$SHVLA_SIM_TIMEOUT` seconds (default 7200 = 2 h): per chain phase by softhier-mlir's own `run_sim(timeout=...)`, for every other run by `timeout` (exit 124, mapped to 3). A config file may raise it for an owner-approved exception. There is no idle limit: the per-process timeout covers hangs.
        4. **Resume:** the run folder is named by the caller. Calling again with the same `<run-name>` reuses phases that already finished (softhier-mlir's reuse checks `<app>/<name>.log` and `.json`); adding `--fresh` after `--` re-simulates them. A new `<run-name>` starts clean.
        5. After the run, greps `run.log` for `TimeoutExpired` (exit 3) and for the side-note error strings ("No entry found for burst", "Executing illegal instruction", "Received out-of-bound request"), and prints the last 40 lines on any failure. The side-note table is then consulted by hand.
        6. Prints a block for LOG.md: command, run name, permalinks, stamp, patch list, exit code, and per phase the simulated ROI, wall time and peak child RSS, read from `<app>/<phase>.json` (written by softhier-mlir).
    Exit codes: 0 pass, 1 fail, 2 refused (stamp or patch), 3 stopped by the timeout.
6.  **`gate.py`** (experiment): reads run folders' `summary.json` files.
    *   `gate.py g3 <run folder>`: the validity gate. Total simulated time (`segments_ms.total`) against 119.1 ms ± 5 %, and action max-abs (`acc.actions.max_abs`, which `run.py` computes against the `--ref` npz, i.e. G2) against 0.10. Prints both values and PASS or MISS; exit 0 on PASS, 1 on MISS.
    *   `gate.py g4a <vision run> <prefix run> <expert run>`: the G4a composition (formula in the G4 table), written as `g4a_estimate.json` with each term and its source run. It also writes the per-phase wall-time projection: measured wall seconds per simulated ms, times the composed simulated ms of each G4b phase.

### Golden reference layout (softhier-mlir's keys)

The file follows softhier-mlir's `smolvla_expert_ref.py` docstring, so `smolvla_e2e.py prepare --ref` reads it unchanged:

All arrays are float32 unless the dtype column says otherwise.

| key | shape (LIBERO), dtype | where it comes from |
| --- | --- | --- |
| `images` | [2, 3, 512, 512] fp16 | camera inputs after `prepare_images`, in [-1, 1]; the model input is these values cast back to fp32 |
| `vis_pos_ids` | [1024] int64 | SigLIP position rows, computed with the bucketize formula of SmolVLM's vision embeddings (as softhier-mlir's `vision_position_ids`) |
| `lang_tokens`, `lang_mask` | [48] int64, [48] bool | tokenised task, right-padded to 48 |
| `state` | [32] | normalised state, zero-padded |
| `noise` | [50, 32] | flow input x_0 (generator per mode, Component 3) |
| `vis_out_<c>` | [1024, 768] | `vision_model(...).last_hidden_state`, i.e. after SigLIP's post-LayerNorm |
| `img_emb_<c>` | [64, 960] | `connector(vis)` output, before the √960 scaling |
| `prefix_embs`, `prefix_valid`, `n_valid` | [177, 960], [177], int64 scalar | `embed_prefix` outputs: embeddings, `prefix_pad_masks` as 0/1, and its sum |
| `kv_k_<l>`, `kv_v_<l>` | [177, 320] each, l = 0..15 | `DynamicCache.layers[l].keys` / `.values` ([1, 5, 177, 64], keys after RoPE), transposed to [1, 177, 5, 64] and reshaped to [177, 320]: head-major columns, `kv_head * 64 + d` |
| `time_emb` | [10, 720] | `create_sinusoidal_pos_embedding(t)` for t = 1 − s/10, computed with LeRobot's function and the config's min/max period |
| `xt`, `vt`, `actions` | [11, 50, 32], [10, 50, 32], [50, 32] | x_t before each `denoise_step` (and after the last), v_t returned by it; `actions = xt[10]`; normalised action space |
| `meta` | [5] int64 | chunk, action dim, steps, Lp, n_valid |

The published configuration has the same keys with 1 camera, 256 tokens and a 65-token prefix. Metadata (checkpoint permalink, lerobot version, observation descriptor) goes into extra keys prefixed `info_`, which softhier-mlir ignores. They are fixed-width string arrays (`np.str_`) or numbers, never object arrays, because softhier-mlir loads the file without `allow_pickle`. They contain no timestamps, so two runs give byte-identical files.

### LIBERO input configuration (Q7, Q8)

| item | value | source |
| --- | --- | --- |
| checkpoint | `lerobot/smolvla_libero` at Hub revision `a24f1f38…` (fine-tuned from `smolvla_base` on `lerobot/libero`, 25k steps, batch 32) | model card, `train_config.json` |
| architecture | SmolVLM2-500M-Video-Instruct, 16 VLM layers, expert width 0.75 (same as `smolvla_base`) | `config.json` |
| cameras | 2: env keys `image` (agentview) and `image2` (wrist), renamed to the checkpoint's `camera1`, `camera2`; 256×256 policy input | `config.json`, LeRobot `LiberoEnv` |
| render size | 360×360 from the env (`observation_height/width`), resized by LeRobot's processing to the policy's 256×256 | LeRobot `LiberoEnv` defaults |
| vision tokens | resize-with-pad to 512×512: 1024 SigLIP patches per camera, 64 image tokens after the 4×4 pixel shuffle | `resize_imgs_with_padding`, `prepare_images` |
| language | right-padded to 48 tokens | `tokenizer_max_length`, `pad_language_to` |
| state, action | state 8-D (OQ2), action 7-D, both padded to 32 | `config.json` |
| prefix | 2 × 64 + 48 + 1 = 177 tokens, padded to S_pad 256 (multiple of 128) | softhier-mlir `smolvla_e2e.info` |
| expert | chunk 50, 10 flow steps, `n_action_steps = 50` | `config.json` |
| control rate | 20 Hz in the env (OQ3) | LeRobot `LiberoEnv.fps` |
| G2/G4 observation | `libero_spatial` task 0, seed 0, first frame after reset | owner-confirmed |
| G6b evaluation | `libero_spatial`, 10 tasks × 10 episodes | owner-confirmed |

### G3 ladder (published configuration, `lerobot/smolvla_base`)

The host prepare steps run first, through `run.sh published prepare` (file 26 lists them). Every rung then runs as `run.sh published sim <rung name> -- <script> <args>`. Rungs 1–5 use softhier-mlir's own on-device or host references and tolerances, and pass on its PASS line. Rung 6 compares against G2. `$D` stands for `$SHVLA_SMOLVLA_DIR`, written out in full in the commands; `run.py` does not resolve relative npz paths.

| rung | command (`tests/gvsoc/run.py ...` unless noted) | checked against | pass |
| --- | --- | --- | --- |
| 1 | `gemm` | on-device self-check | all shapes PASS |
| 2 | `mlir examples/*.mlir` | on-device self-check | every example that has a self-check PASSes; the others build and run to the end (the count is recorded) |
| 3 | `smolvla --npz $D/vision_s256.npz --layers 1` | HF SigLIP fp32 inside `prepare --seq 256` | PASS; errors recorded next to the README's (EMB 0.0067, layer 1 0.039) |
| 4 | `smolvla-vlm --npz $D/vlm_c1.npz --layers 1` | fp32 LeRobot-semantics reference inside `prepare-vlm` | PASS; errors recorded next to `docs/SMOLVLA.md` |
| 5 | `tests/gvsoc/expert.py flow --npz $D/expert.npz --steps 1` | the expert's numpy twin and G2 `xt[1]` | PASS; step-1 error recorded |
| 6 | `smolvla-e2e --npz $D/e2e_c1_t256.npz --e2e-trace` | G2 at every hand-over (VIS, IMG, K/V, x_t, actions) | `gate.py g3` (the validity gate) |

Rung 6 is the published-configuration chain. G4 is a different configuration (LIBERO) and a different checkpoint.

### G4 staged plan (decision A1)

| step | command | measures |
| --- | --- | --- |
All commands run as `run.sh libero sim <step id> -- tests/gvsoc/run.py ...`, after `run.sh libero prepare`; `$D` is `$SHVLA_SMOLVLA_DIR`, written out in full.

| step | command | measures |
| --- | --- | --- |
| G4a.1 | `smolvla-e2e --vision-only 1024 --layers 1 --e2e-trace` (reads `vision_s1024.npz` of the LIBERO weights) | one SigLIP layer at 1024 tokens: embed, attention, proj/FFN per layer, post-LN, connector |
| G4a.2 | `smolvla-e2e --prefix-only 2 --layers 8 --e2e-trace` (reads `vlm_c2.npz`, 177 tokens) | connector and prefix layers 1–8 at 177 tokens |
| G4a.3 | `smolvla-e2e --expert-only ref:$D/e2e_c2_t1024.npz --steps 2 --e2e-trace` (patch 0005: KV from G2) | expert KV projection and per-step time at Lp 177; steps 1–2 against G2 `xt[1..2]` |
| G4a.4 | `gate.py g4a`: vision = 2 cameras × (embed + 12 × layer + post-LN) + connector; prefix = 2 × layers 1–8; expert = KV projection + 10 × step | the estimate and, from the wall-time ratios, the G4b projection |
| G4b | `smolvla-e2e --npz $D/e2e_c2_t1024.npz --e2e-trace` | the full chain, measured |

The projection is made per phase: G4b runs as four simulator processes (V, Pa, Pb, X), each under the 2 h limit. If any projected phase exceeds 2 h (R1), the session stops and asks the owner before G4b.

### softhier-mlir changes for the LIBERO configuration (decision A2)

Read from softhier-mlir `main` at [b11cd5f](https://github.com/runwangdl/softhier-mlir/tree/b11cd5fa4f3e717374d1e9ba14442ff302d1f970). Already general, no change needed:
*   `smolvla_e2e.emit_vision` loops over cameras in one `scf.for` and takes tokens per camera from the image side.
*   `run.py`'s `run_smolvla_e2e` takes cameras, tokens and S_pad from the npz, splits the prefix 8 + 8, and passes `s_pad` and `kv_stride` to the expert.
*   `smolvla_expert.emit_flow` takes the prefix length from the token-class row.
*   177 tokens pad to S_pad 256, as the 241-token prefix in softhier-mlir's cost runs does.

Changes, as patch files:

| patch | files | change | why |
| --- | --- | --- | --- |
| 0001 | `softhier_mlir/frontend/smolvla_e2e.py` (`MODELS`), `frontend/smolvla.py` (lines 524–525 cache paths; 999, 1012 `--ckpt` defaults), `frontend/smolvla_expert.py` (886–887), `frontend/smolvla_expert_ref.py` (225), `tests/gvsoc/run.py` (803, 838, 1088), `tests/gvsoc/expert.py` (default `--npz`) | read `SHVLA_SMOLVLA_DIR` instead of `/app/models/smolvla_base`, falling back to the original | the paths do not exist here; one variable per run selects the checkpoint folder |
| 0002 | `frontend/smolvla_e2e.py` (`VISION_NPZ`, `VLM_NPZ`, `EXPERT_NPZ` and the default arguments that bind them at import, lines 60, 103–104, 260, 282, 312), `tests/gvsoc/run.py` (748, 751, 755) | replace the three constants with functions `vision_npz()`, `vlm_npz()`, `expert_npz()` that read `SHVLA_SMOLVLA_DIR` when called; default arguments become `None` and are resolved inside the function; the attribute reads in `run.py` call the functions | the same process may prepare both checkpoints, and a default bound at import would keep the first folder |
| 0003 | `frontend/smolvla_expert_ref.py` | port to the LeRobot and transformers versions of `infra/lerobot` (transformers ≥ 5.4): the VLM forward drops `fill_kv_cache=True` (LeRobot fills the cache when `use_cache=True` and `past_key_values=None`); the KV dump reads `DynamicCache.layers[l].keys` / `.values` and transposes to [tokens, heads, d] before the reshape; phase `expert` rebuilds a `DynamicCache` from the npz with `update(k, v, l)` per layer ([1, 5, Lp, 64], the reverse of the dump) before `denoise_step`; the docstring's environment line is updated; `vision_position_ids` takes the formula of the installed transformers (`idx * (1/n)` clamped to `1 - 1e-6`; the old `idx / n * (1 - 1e-6)` puts every row after the first one bucket lower, found in G2.1). Output keys and layout unchanged. This is the known minimum: if the meta-device build (`build_policy`) or the buffer rebuild in `materialise` also breaks under the new versions (for example new buffer names), the extra change is logged in LOG.md first and added to this patch, as part of this plan rather than an R3 change | the crosscheck (G2.3) runs the reference in the same environment as `golden_ref.py`; the old API exists in neither planned environment |
| 0004 | `tests/gvsoc/run.py` (argument parser, the `smolvla-e2e` dispatch) | add `--timeout <seconds>` (default unchanged, 48 h) and pass it to `run_smolvla_e2e`, `run_vision_cost`, `run_prefix_cost` and `run_expert_cost`, which already take `timeout` and hand it to `run_sim` | the 2 h per-process limit, enforced by softhier-mlir's own `subprocess.run(timeout=...)` |
| 0005 | `tests/gvsoc/run.py` (`run_expert_cost`) | a new KV source `ref:<e2e npz>`, checked **before** the existing `kv_src.endswith(".npz") and "e2e_" in kv_src` branch (which `ref:...e2e_....npz` would otherwise match); the `ref:` prefix is stripped, then the expert data is built with `E.expert_data(e2e, {L: (e2e["ref_kv_k_L"], e2e["ref_kv_v_L"])})`, i.e. G2's KV, which `smolvla_e2e.prepare` already copies into the e2e npz (`smolvla_e2e.py:87-89`). Modelled on the `vlm_c1` branch (`run.py:750-761`). The existing e2e source, which reads `chain_outputs.npz` from a fixed folder inside the submodule (`run.py:743`), is left unchanged | G4a.3 must run before G4b, and the existing source needs a finished chain run and ignores `--app-dir` |

Not patched: `--steps` already reaches `run_expert_cost` (`run.py:1148`); `--prefix-only 2` reads `$SHVLA_SMOLVLA_DIR/vlm_c2.npz` once patch 0001 is in, and `prepare-vlm --cams 2` produces the 177-token prefix.

Residual device-side risks, not patched in advance (R3):
*   the vision program's activation buffers at 1024 tokens (scores `12 × 1024 × 1024` fp16 = 24 MiB, in HBM, already exercised by softhier-mlir's 1024-token runs);
*   the expert at Lp 177 with S_pad 256 (same code path as Lp 241);
*   the 64 KB instruction memory (programs do not grow with cameras or layers).
A failure there is handled under R3.

`SOFTHIER_HOME` is already read by `sim/gvsoc.py`; `SOFTHIER_CHROOT=""` is exported by `infra/env.sh`. The `SHARED_HOME` constant only guards `apply_arch`, which this design does not call, so it needs no patch.

### Design decisions

*   **D1: one simulator build per worktree.** `build_arch.sh` rebuilds for an arch file and writes a stamp: arch path and SHA-256, GVSoC and nested submodule commits, and a hash of `git diff` in each. It also records the model folders `run_sim` will pick at run time and are invisible to `git diff`: whether `install/models_fast` and `install/models_fix` exist, with a SHA-256 over their `.so` files, and the values of `SOFTHIER_MODEL_DIR` and `SOFTHIER_STOCK_MODELS`. `run.sh` refuses to run on a mismatch and does not rebuild by itself, so two runs never rebuild under each other. Runs that need different arch files run one after another. The rebuild time is measured in G1.
*   **D2: simulator fixes in a fork, softhier-mlir changes as patch files.**
    *   GVSoC: the submodule points to `HaozeG/gvsoc` (branch `shvla/soft_hier_release`) once the first fix exists. The first expected fix is adding `flex_alloc_api.h`, which softhier-mlir's runtime includes but `soft_hier_release` lacks (present on the GVSoC branch `bowwang-dev/softhier-deeploy`). Nested `core`/`pulp` forks are created only if a fix touches them. Creating the forks waits for the owner.
    *   softhier-mlir: patch files with a header (what, why, evidence), listed in `patches/README.md`, which doubles as the inspection record of this initial implementation.
    *   Boundary: a change goes to the repository that owns the file; the SDK and model files belong to GVSoC.

### Side note: softhier-mlir's simulator patches (not applied; watch for their symptoms)

From softhier-mlir `docs/SIMULATOR_NOTES.md`. The current `soft_hier_release` may already change some of these. #7 and models_fast affect speed; the rest affect correctness.

| # | symptom | cause in the model |
| --- | --- | --- |
| 1 | stale `flw`/`fsw` to HBM or remote TCDM; NoC error "No entry found for burst" | Snitch FP subsystem completes an offloaded access after a fixed latency even when it is pending |
| 2 | `flh`/`fsh` return garbage | Zfh half loads decoded as integer loads; the library avoids scalar fp16 |
| 3 | long fp16 dot products about 10 % low | `light_redmule.cpp` truncates instead of rounding to nearest even |
| 4 | segfault with two outstanding collective broadcasts | NoC collective model; the library waits after each broadcast |
| 5 | DRAMSys used instead of ideal HBM: HBM-bound runs slower (+46-48 % on 16-cluster GEMMs), simulator memory grows with simulated time, `gemm1024_summa` / `siglip_encoder_layer` hit the memory cap | softhier-mlir's `SOFTHIER_IDEAL_HBM` was a private patch (no GVSoC code reads the variable); added on the local branch `shvla/soft_hier_release` (`flex_cluster.py`, ideal `Memory` per HBM channel), see LOG G3.1 |
| 6 | `fcvt.w.s` then `fcvt.s.w` returns a stale integer | FP-to-int result forwarded before it lands; the library avoids the pattern |
| 7 | three cores no faster than one; loop speed depends on address | one shared instruction router, no instruction cache |
| 10 | softmax, LayerNorm, GELU depend on binary layout | scalar FP-to-int instructions lack the `nseq` tag |
| 11 | illegal instruction at a data address late in a long program | 1 KB stack spacing in `flex_start.s`; the library uses private stacks |
| 12 | out-of-bound TCDM request after a RedMulE trigger | register clobber in `flex_redmule_trigger`; the library uses its own trigger |
| 13, 14 | in-network REDADD adds stale data; fp16 REDADD truncates | iDMA burst buffers not cleared; `floonoc.cpp` truncates fp16 |
| (local) | rowops LN/SM/GELU 30-50 % faster than published, rung c layer 1.52 ms vs 2.195 ms, single-cluster GEMMs 5-12 % faster | `cluster_unit.py:209` built `instr_mem` with `width_log2=-1`, so `size >> -1` is undefined and instruction fetch is unlimited; the author's deeploy branch uses the default 2 (32 B line, 8 cycles). Fixed on the local branch by dropping the argument (GVSoC `f99d5a2`; the pulp copy is copied from the same file by `make`), see LOG G3.3-gap, G3.3-fix-instr. This row is an addition to the original table (it has no number: #6 is taken) |
| models_fast | wall time above about 24 s per simulated ms | host-speed rebuild of the Snitch models, not public |

**Reference for fixes (owner's local notes, read when a run hits the 2 h limit or a side-note symptom).** `/home/haoze/SmolVLA/SmolVLA/` holds the notes of the earlier work behind softhier-mlir, in Chinese. They are not part of this repo and may be missing on other machines. What they give:
*   `softhier_vla_summary_2026-10.md` §6: the 15 simulator and SDK pitfalls in softhier-mlir's numbering, with the original patch names (`gvsoc_snitch_fp_ss_pending.patch`, `gvsoc_redmule_fp16_rne.patch`, `gvsoc_snitch_fp_int_nseq.patch`, installed as `install/models_fix`).
*   The same file, host speed: `models_fast` was two changes with unchanged cycle counts. One is the RedMulE functional model vectorised with NEON (190×). That is aarch64-only, so on this x86_64 host an equivalent must be found by profiling first. The other removes by-value copies and idle trace formatting from the ISS (2–4×).
*   `deeploy_softhier_progress.md` §3: the exact fix for #1 (`Iss::handle_event` in `core/models/cpu/iss/src/snitch/snitch_fp_ss.cpp` waits while `exec.is_stalled()`). It also gives the partial rebuild procedure: build only `gen_isa_snitch_fp_ss_rv32imfdva_cpp_*_{optim,debug}` with `SYSTEMC_HOME=third_party/systemc_install`, then copy the `.so` files into `install/models/` and `install/models/debug/`.
*   `repro/softhier_fp_hbm_repro.c`: a reproducer for #1 that prints `out=1.7500` when the fix works.
The `.patch` files themselves (`../SoftHier/dse/patches/`) are not on this machine. Fixes are re-implemented from these descriptions as commits on the GVSoC fork (D2), each citing the note it follows.

## Alternatives Considered

SR-B and SR-C are deferred, not rejected (G0).

### Simulator

*   **softhier-sdk with the new `pulp.chips.softhier` model.** Rejected: that model has no RedMulE and none of the `flex_*` headers, so every softhier-mlir GEMM would need rewriting for Spatz and the published numbers would not apply. Pro: newest public model with more topologies.
*   **Apply softhier-mlir's private patches up front.** Rejected: they are unpublished, would be rewritten from descriptions without knowing which the current branch needs, and effort would precede evidence. Pro: closer numbers from the start. The owner chose to fix along the way.

### Layout and environments

*   **Install in a home folder outside the repo.** Rejected: records no commits, environments or steps, and the next experiment starts from zero. Pro: no repo change.
*   **Reuse earlier installs (`~/softhier`, `~/softhier-mlir`, `~/softhier-mlir-home`).** Rejected: unknown branches, patches and edits make results unattributable. Pro: already built.
*   **GVSoC cloned by a script into a gitignored folder.** Rejected: the commit lives only in a script variable, bumps do not appear in the parent's history, and worktrees get no own checkout (D1 relies on it). A submodule records the exact commit in every parent commit. Its costs (`--recursive`, detached HEAD, nested submodules left modified by GVSoC's preparation) are handled in `setup.sh` and `ignore = dirty`. Pro of the script: smaller `git status`.
*   **conda, or untracked venvs as in `~/learn/physical_ai`.** Rejected: no exact dependency set is recorded in the repo. Pro: already present and working.
*   **A repo-local data folder for weights.** Rejected: checkpoints are large binaries (git would need LFS), and the Hugging Face cache is shared across projects and placed per user. Recording the Hub revision keeps runs reproducible. Pro: one simple path.
*   **One install copy per arch.** Rejected: several GB per copy, refreshed after every simulator fix, with the risk of mixing patched and unpatched copies. Pro: runs with different arch files at once.
*   **G1 only on `main`, experiment build fixed to its arch.** Rejected: two full builds, and fixes in the experiment must reach `main` before the G1 build has them. Pro: no arch switching.
*   **Patch files for the simulator too.** Rejected: each simulator fix would be traced by two pieces (upstream commit plus patch), with no path to an upstream pull request. Pro: no forks.
*   **Forks for softhier-mlir too.** Rejected: up to four forks to sync, and a fork contradicts following softhier-mlir's latest `main` as an implementation under inspection. Pro: one permalink per code state.
*   **Pin softhier-mlir, or record bare hashes.** Rejected: the owner follows latest `main` between experiments (frozen within one, R4), and a bare hash is not a link. Pro: exact reruns.

### Golden reference

*   **softhier-mlir's own `smolvla_expert_ref.py` as the golden reference.** Rejected: the reference would be part of the system under test, so its errors would be copied into the comparison. Pro: least code, matching layout. It is still used once, ported, as the G2.3 crosscheck.
*   **Crosscheck in a third environment with `transformers<5`, or no crosscheck.** Rejected (decision XC): a third environment pins an API LeRobot has left and must be maintained; dropping the check leaves a layout bug in the KV or embedding capture to be found only at rung 6, where it cannot be told apart from a simulator error. Pro of the third environment: softhier-mlir's script runs unmodified.
*   **A compiler-neutral schema plus a converter.** Rejected: an extra layer and an extra source of bugs, with only one compiler to serve. Pro: neutral to future compilers.
*   **Golden reference on CPU instead of GPU.** Rejected: slower for the 16-layer VLM, and LeRobot evaluation runs on the GPU anyway. Pro: no CUDA determinism settings.

### Input configuration and measurement

*   **Latency in the published synthetic configuration only.** Rejected: it matches no robot evaluation, so latency and success rate would describe different setups. Kept as the G3 environment check.
*   **Feed LIBERO's 256×256 images to SigLIP directly.** Rejected: the checkpoint resizes to 512 before SigLIP, so this changes the computation and the numerics. Pro: less vision work.
*   **Full simulation only, no staged estimate.** Rejected after review: the wall time of a full LIBERO run is unknown, and a full run that turns out to take days would block the machine. The staged estimate measures it first (A1). Pro: one step instead of two.
*   **`HuggingFaceVLA/smolvla_libero` or `lerobot/smolvla_base`.** Rejected: the former uses a different VLM (SmolVLM2-500M-Instruct), expert width 0.5 and other layer settings, so softhier-mlir's shapes do not apply; the latter is not fine-tuned on LIBERO. Pro: the former is also first-party; the latter is what softhier-mlir validated.
*   **Another benchmark (Meta-World, VLABench).** Rejected: SmolVLA reports LIBERO, a first-party fine-tuned checkpoint exists, and LeRobot ships the LIBERO environment. Pro: VLABench tests generalisation more strongly (`wiki://knowledge/simulation-benchmarks.md`).
*   **Closed-loop episodes on the SoftHier simulator.** Rejected: one inference takes hours of wall time here (R1). SR-B and SR-C approximate it.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Branches and commits

*   All work happens in this worktree, so the simulator is built once (D1).
*   Every commit is either an **infra commit** or an **experiment commit**, never both:
    *   infra commits touch only `.gitmodules` (GVSoC entry), `.gitignore`, `README.md`, `docs/`, `infra/` and `third_party/`;
    *   experiment commits touch only `experiments/` and the softhier-mlir entry in `.gitmodules`.
*   Infra and experiment commits interleave on this branch (later facts amend files 6, 7 and 13). `infra-baseline` is therefore built at the end, not set at a commit: a new branch from `main`, then `git cherry-pick` of the infra commits in order. They are listed by `git log --reverse --format=%H main..HEAD -- .gitignore README.md docs infra third_party`, plus any commit that touches only the GVSoC entry of `.gitmodules`. The branch is offered to `main` as a pull request.
*   Because `.gitmodules` holds both submodule entries, the GVSoC entry is added in an infra commit (G1.1) and the softhier-mlir entry in a separate experiment commit (G2.2); the cherry-pick of G1.1 then applies cleanly.
*   The wiki commits on its own branch inside the submodule, as the wiki's CLAUDE.md requires. Bumping the parent's wiki pointer is the owner's call, and the bump is offered as a separate commit.
*   Owner approval is needed for: `sudo apt-get install` (if prerequisites are missing), creating the public forks, and pushing or opening the pull request.

### Step log (manual inspection)

`experiments/smolvla-e2e-baseline/LOG.md` is the single trace of the whole effort, infrastructure steps included.
*   It opens with the step list below and a status column (planned, in progress, done, failed, skipped).
*   One entry per step, written when the step ends: step id (for example `G1.3`), time, what and why, exact commands, result in a few lines, files changed, commit hash or "no commit", paths under `$SHVLA_DATA`, and **What to inspect**. Failed attempts and dead ends get entries too.
*   Commits per step: an experiment step is one commit, code and LOG.md entry together. An infra step is two commits: the infra commit (code), then an experiment commit with its LOG.md entry, which names the infra commit. Both subjects carry the step id.

### Ordered steps

Each step ends with its LOG.md entry; the checkpoint is what must hold before the next step. Steps of G1 and of G6a/G2/G6b may interleave; G3 needs both.

| step | files | action | checkpoint |
| --- | --- | --- | --- |
| G0.1 | 19–22 | wiki branch `decision/first-baseline` in the submodule: decision, catalog, glossary row, knowledge link; `wiki.py fmt`, `check`, `commit update` | `wiki.py check` passes; owner told the branch name |
| G0.2 | 3, 4 | `.gitignore`, README layout section | — |
| G1.1 | 1, 2 | add the GVSoC submodule (infra commit with only that entry in `.gitmodules`) | `git submodule status` shows `soft_hier_release` |
| G1.2 | 5 | `infra/env.sh`, `env.local.sh.example` | `env.sh --show` prints every variable |
| G1.3 | 8, 9 | GVSoC uv project, `setup.sh`; run it | `softhier_preparation` finished; toolchain present |
| G1.4 | 10, 12 | `stamp.py`, `build_arch.sh` | `stamp.py check` passes after a build with `implementation/`'s arch |
| G1.5 | 11 | `run_kernel_tests.sh`; read the kernel make targets and pass strings; run | `gemm`, `norm`, `acti`, `attn` PASS |
| G1.6 | 7 | `infra/softhier/README.md` with measured times | — |
| G6a.1 | 14, 15 | LeRobot uv project and env; settle OQ1 | `smolvla_libero` loads at its revision |
| G6a.2 | 17 | `libero_eval.py`; 1 task × 1 episode smoke run | `summary.json` written |
| G2.1 | 16, 18 | `golden_ref.py` synthetic and from-npz modes; tests (a)–(c) | published ref written; tests pass |
| G2.2 | 23, 24, 26, 27, 28 (0001–0005) | softhier-mlir submodule at `b11cd5f` (experiment commit), experiment uv project, configs, `run.sh`, patches | `run.sh published fetch` and `prepare` finish; patches apply |
| G2.3 | — | crosscheck: `run.sh apply`; then in the LeRobot environment `python experiments/smolvla-e2e-baseline/softhier-mlir/softhier_mlir/frontend/smolvla_expert_ref.py --ckpt $SHVLA_DATA/smolvla_base --cams 1 --img-size 256 --out $SHVLA_DATA/golden/xc_ref.npz` (by file path, not `-m`, so the package `__init__` and its xdsl import are not loaded; the script itself imports only stdlib, numpy, torch and LeRobot), then `golden_ref.py --obs from-npz:$SHVLA_DATA/golden/xc_ref.npz --device cpu --crosscheck $SHVLA_DATA/golden/xc_ref.npz` | every key within tolerance (Verification Plan) |
| G2.4 | 16, 18 | libero mode; test (d), (e) | LIBERO ref written; tests pass |
| G2.5 | 13 | `infra/lerobot/README.md` | — |
| G6b.1 | — | `libero_spatial` 10 × 10; settle OQ2–OQ4 | success rate with interval in LOG.md |
| G3.0 | 25 | arch file; `build_arch.sh` with it | `stamp.py check` passes |
| G3.1–G3.6 | — | ladder rungs 1–6; fork commits (33) when a fix is needed, owner first | each rung's pass column |
| G3.7 | 34 | `gate.py g3` on the rung-6 run | gate outcome recorded (PASS or MISS with values) |
| G4a.1–G4a.4 | 31 | `run.sh libero prepare`; G4 table steps; `gate.py g4a`; INPUT_CONFIG.md | projection per phase recorded; R1 decision if any phase > 2 h |
| G4b | — | full LIBERO chain | every phase finished or the R1 stop recorded |
| G5.1 | 6, 30, 32 | RESULTS.md, SETUP.md, design update | Verification Plan passes |
| G5.2 | — | build `infra-baseline` by cherry-pick; offer push and PR to the owner | owner decision recorded |
*   Each goal ends with a short summary, also reported in chat. Work continues unless the owner says stop, except at owner-approval steps and the R1/R2 stop rules.
*   The log's language rule (mixed Chinese and English) is kept in the agent instructions, not here.

### Paths and environment variables

| variable | default | used by |
| --- | --- | --- |
| `SHVLA_ROOT` | `git rev-parse --show-toplevel` from the script's own folder (the worktree root in a worktree) | every script |
| `SHVLA_DATA` | `~/.cache/softhier-vla` | npz files, preload ELFs, logs, run folders, eval outputs |
| `HF_HOME` | Hugging Face default | checkpoints |
| `SHVLA_SMOLVLA_DIR` | set per config: `$SHVLA_DATA/smolvla_base` or `$SHVLA_DATA/smolvla_libero` | softhier-mlir (patches 0001, 0002) |
| `SOFTHIER_HOME` | `$SHVLA_ROOT/third_party/gvsoc` | softhier-mlir `sim/gvsoc.py` |
| `SOFTHIER_CHROOT` | empty | softhier-mlir `sim/gvsoc.py` (x86_64 host) |
| `SHVLA_SIM_TIMEOUT` | `7200` (seconds, per simulator process) | `run.sh sim` → `run.py --timeout` (patch 0004) |
| `SOFTHIER_MODEL_DIR`, `SOFTHIER_STOCK_MODELS` | unset | softhier-mlir `run_sim` (extra model folders; stock models only); recorded in the stamp |

`SOFTHIER_IDEAL_HBM`: `run_sim` sets it to `1` for every run (`ideal_hbm=True`, the default), and that is sufficient only because the local GVSoC branch reads it (side-note item 5; the stock simulator ignores it and always instantiates DRAMSys). It is read when `gapy` generates the platform config, i.e. at run time, not at `make hw`, so it is not in the build stamp; `infra/env.sh` does not export it. The installed `flex_cluster.py` must match the source, which a rebuild (`build_arch.sh`) ensures and the stamp's GVSoC commit covers. `infra/env.sh` sets these defaults and then sources `infra/env.local.sh` if present (gitignored; overrides for one machine or user).

### Target Files to Create/Modify

#### On `main` (via `infra-baseline`)

##### 1. `.gitmodules` (Modify)
*   **Type / Scope**: Git configuration.
*   **Proposed Changes**: Add `third_party/gvsoc` with URL `https://github.com/gvsoc/gvsoc.git` (switched to the fork when the first fix exists, D2), `branch = soft_hier_release`, `ignore = dirty`.

##### 2. `third_party/gvsoc` (Create, submodule)
*   **Type / Scope**: Git submodule.
*   **Proposed Changes**: GVSoC `soft_hier_release` at the current commit, nested submodules initialised recursively.

##### 3. `.gitignore` (Modify)
*   **Type / Scope**: Git configuration.
*   **Proposed Changes**: Add `.venv/`, `__pycache__/`, `infra/env.local.sh`, `egm-review/` (review reports stay local unless the owner wants them committed). `CLAUDE.md` stays ignored, so this design does not modify it.

##### 4. `README.md` (Modify)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   a "Repository layout" section (wiki, `docs/designs/`, `third_party/`, `infra/`, `experiments/` on experiment branches);
    *   the layout rule (as in G0);
    *   the two metrics, linking the wiki decision and this design;
    *   quick-start pointers to the infra READMEs.
    The softhier-sdk link in the first line is replaced by the GVSoC `soft_hier_release` link.

##### 5. `infra/env.sh` and `infra/env.local.sh.example` (Create)
*   **Type / Scope**: Shell environment.
*   **Proposed Changes**: The defaults table above; `--show` prints the resolved values. The example file shows per-user overrides.

##### 6. `docs/designs/smolvla-softhier-baseline.md` (Modify)
*   **Type / Scope**: This design.
*   **Proposed Changes**: Already committed; revised here. Updated with G1–G6 facts that change it (rebuild time, OQ1–OQ4 answers).

##### 7. `infra/softhier/README.md` (Create)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**: prerequisites, the three scripts, the stamp rule, the fact that kernel tests rebuild the hardware, switching to the fork, measured build and rebuild times, disk use, the side-note table by reference.

##### 8. `infra/softhier/pyproject.toml`, `uv.lock` (Create)
*   **Type / Scope**: uv project.
*   **Proposed Changes**: Python 3.12; dependencies from GVSoC's `requirements.txt`, plus `prettytable` and `pyelftools` if that file lacks them (`gapy` imports both).

##### 9. `infra/softhier/setup.sh` (Create)
*   **Type / Scope**: Shell script.
*   **Proposed Changes**:
    1. Check gcc ≥ 11.2, cmake ≥ 3.18.1, uv and the apt packages of the GVSoC README; print the install command for anything missing.
    2. `git submodule update --init --recursive third_party/gvsoc`, then `uv sync`.
    3. `make softhier_preparation` with the uv environment active and stdin from `/dev/null`, so the Makefile's interactive `read -t 30` prompt takes its default instead of waiting; the prompt's default is read in G1.3 and noted in the README.
    4. Print the permalinks.
    Steps already done are skipped: the submodule step if `git submodule status` shows no `-` prefix, the preparation step if `third_party/systemc_install` and the toolchain folder exist.

##### 10. `infra/softhier/build_arch.sh` (Create)
*   **Type / Scope**: Shell script.
*   **Proposed Changes**: Resolve the arch path to an absolute path, run `cfg=<abs path> make hw` in the submodule and time it, then `stamp.py write`. On failure, delete the stamp.

##### 11. `infra/softhier/run_kernel_tests.sh` (Create)
*   **Type / Scope**: Shell script (G1).
*   **Proposed Changes**:
    *   Run `make <k>-pre <k>-run` in `third_party/gvsoc/implementation` for `gemm`, `norm`, `acti`, `attn`, in that order, stopping at the first failure. The target names are confirmed against `soft_hier_release`'s `implementation/Makefile` and its README in G1.5; if they differ, the script follows the README's commands.
    *   Logs go to `$SHVLA_DATA/softhier/g1/<k>.log`.
    *   Pass: exit code 0, a performance-counter line (parsed with the same regular expression as softhier-mlir's `PERF_RE`), none of the side-note error strings, and the kernel's own check output (strings read from each `main.c` in G1.5 and written into the README).
    *   At the end, if a target rebuilt the hardware, `stamp.py write implementation/config/arch/arch.py`.
    *   Print a table of kernel, pass or fail, simulated ns and wall seconds.

##### 12. `infra/softhier/stamp.py` (Create)
*   **Type / Scope**: Python, standard library only.
*   **Proposed Changes**: Three subcommands:
    *   `write <arch>`: arch path (relative to `$SHVLA_ROOT`) and SHA-256; GVSoC and every nested submodule commit (`git submodule status --recursive`); SHA-256 of `git diff HEAD` plus `git status --porcelain --untracked-files=no` in each; the model-folder entries of D1; written to `third_party/gvsoc/.shvla_stamp.json`. Untracked files are not covered; build products are untracked by design.
    *   `check <arch>`: exit 0 on a match, 1 with the difference.
    *   `permalinks`: remote URL plus commit to a GitHub tree link, for every submodule.

##### 13. `infra/lerobot/README.md` (Create)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   setup (`uv sync`, `source env.sh`) and the osmesa and libstdc++ notes from `~/learn/physical_ai`;
    *   `golden_ref.py` and `libero_eval.py` usage;
    *   the key table;
    *   determinism settings;
    *   the memory-cap command (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`);
    *   the measured eval speed.

##### 14. `infra/lerobot/pyproject.toml`, `uv.lock` (Create)
*   **Type / Scope**: uv project.
*   **Proposed Changes**: Python 3.12; `lerobot[smolvla,libero]` pinned per OQ1 (0.7.0, or 0.6.1 if 0.7.0 cannot load the checkpoint); CUDA torch from the PyTorch wheel index whose CUDA version is the highest not above the one `nvidia-smi` reports (declared as a uv index in `pyproject.toml`); numpy; pytest as a dev dependency. Patch 0003 targets the KV cache API of the pinned version; if OQ1 lands on 0.7.0, G2.2 re-reads that API before writing the patch.

##### 15. `infra/lerobot/env.sh` (Create)
*   **Type / Scope**: Shell environment.
*   **Proposed Changes**: Source `infra/env.sh`, activate the environment, set `LD_PRELOAD` to the system `libstdc++.so.6`, `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`.

##### 16. `infra/lerobot/golden_ref.py` (Create)
*   **Type / Scope**: Python program (G2).
*   **Proposed Changes**:
    *   Arguments: `--checkpoint`, `--revision`; `--obs synthetic:<cams>:<side>:<seed>`, `from-npz:<path>` or `libero:<suite>:<task>:<seed>`; `--device cuda|cpu` (default `cuda`); `--out`.
    *   Loads the policy with `from_pretrained` at the revision, in fp32.
    *   Capture points, all in LeRobot's SmolVLA model (where each key comes from is in the key table):
        *   the VLM model's `vision_model` and `connector` outputs per camera;
        *   `embed_prefix` outputs;
        *   the `DynamicCache` returned by the VLM forward with `use_cache=True, past_key_values=None`, read before the first `denoise_step` (which crops it in place);
        *   `x_t` and `v_t` around each `denoise_step`.
    *   Writes the key table, plus `info_*` metadata.
    *   `--crosscheck <ref npz>`: compares every key both files share, in pipeline order, and prints per key `max|a − b| / max(|b|.max(), 1e-6)`. The first key above the tolerance (Verification Plan) is named as the divergence point and the exit code is 1. Used once (G2.3), on the published synthetic input fed through `from-npz`, against the output of the patched `smolvla_expert_ref.py`.

##### 17. `infra/lerobot/libero_eval.py` (Create)
*   **Type / Scope**: Python program (G6).
*   **Proposed Changes**: Builds and runs the `lerobot-eval` command (Component 3) under the memory cap; reads `eval_info.json`; prints success per task and overall with Wilson 95 % intervals; writes `summary.json` next to it. `--eval.n_episodes` counts episodes per task (`lerobot_eval.py`), so 10 tasks × 10 episodes is `--eval.n_episodes=10` over the suite. The revision is passed as `--policy.pretrained_revision`.

##### 18. `infra/lerobot/tests/test_golden_ref.py` (Create)
*   **Type / Scope**: pytest (GPU tests skipped without CUDA).
*   **Proposed Changes**:
    *   (a) every key in the table exists with the expected shape for both configurations;
    *   (b) two runs give the same SHA-256;
    *   (c) a different seed changes `images` and `actions`;
    *   (d) on the LIBERO observation, `actions` equals `policy.predict_action_chunk(batch, noise=noise)[..., :action_dim]` in the normalised space, max-abs ≤ 1e-5 (same device, same code path);
    *   (e) the LIBERO batch matches the observation `lerobot-eval` gives the policy: the state is 8-D before padding, and the images equal the env frame rotated 180° (checked on one pixel row);
    *   (f) failure paths: an unknown `--obs` mode and a missing checkpoint revision exit non-zero with a message.

#### Wiki (submodule branch)

##### 19. `softhier_vla_wiki/memories/decisions/2026-11-10-first-baseline-latency-and-success-rate.md` (Create)
*   **Type / Scope**: Wiki decision (`Context`, `Decision`, `Why`, `Open questions`).
*   **Proposed Changes**: Content as in G0. Cites:
    *   `resources/serving/vla-workload-characterization-kaist.md` §5;
    *   `resources/serving/vla-hwbench-edge-latency.md`;
    *   `resources/models/smolvla.md`;
    *   `knowledge/simulation-benchmarks.md`.
    `Open questions` holds SR-B (needs: a timing wrapper around the LIBERO env, an action queue, overlap, stale actions, the G4 latency) and SR-C (needs: a validated numerical twin of SoftHier's fp16 arithmetic). It links this design by its GitHub URL. It defines "golden reference" (`defines:`).

##### 20. `softhier_vla_wiki/memories/decisions/_overview.md` and `_abstract.md` (Modify)
*   **Type / Scope**: Wiki catalog.
*   **Proposed Changes**: Replace "Empty until populated." in both files with the catalog line and a one-line abstract.

##### 21. `softhier_vla_wiki/glossary/project-terms.md` and `glossary/a-to-z.md` (Modify)
*   **Type / Scope**: Wiki glossary.
*   **Proposed Changes**: A project row for "golden reference", defined in file 19; then regenerate the A-to-Z index with `wiki.py glossary --az`. The existing "success rate" row is left as it is.

##### 22. `softhier_vla_wiki/knowledge/softhier-and-the-project.md` (Modify)
*   **Type / Scope**: Wiki knowledge note.
*   **Proposed Changes**: In "Goals", one sentence linking the new decision for the tracked metrics. No numbers.

Before editing: `git submodule update --init softhier_vla_wiki` if the worktree's wiki checkout is empty, and `softhier_vla_wiki/tools/wiki.py setup` once if the tool environment is missing. Before committing: `wiki.py fmt` and `wiki.py check`; commit with `wiki.py commit update` on the wiki branch.

#### Experiment (worktree branch only)

##### 23. `.gitmodules` (Modify) and `experiments/smolvla-e2e-baseline/softhier-mlir` (Create, submodule)
*   **Type / Scope**: Git submodule.
*   **Proposed Changes**: URL `https://github.com/runwangdl/softhier-mlir.git`, `branch = main`, `ignore = dirty`; pinned to `b11cd5f`, the commit the patches and line numbers in this design were read from, and frozen until G5. If `main` has moved when G2.2 starts, the session reports the new commits and the owner decides whether to bump first (then the patches are re-based by symbol, and line numbers in this design become stale).

##### 24. `experiments/smolvla-e2e-baseline/pyproject.toml`, `uv.lock` (Create)
*   **Type / Scope**: uv project.
*   **Proposed Changes**: Python 3.12; softhier-mlir as an editable path dependency, with `xdsl` at the version its `pyproject.toml` names; numpy; CPU torch, transformers and safetensors (for the `prepare` steps); GVSoC's Python requirements including `prettytable` and `pyelftools`, because `run.sh` activates this environment and `gapy` runs with the active `python3`; filecheck, lit, pytest. `ruff` and `shellcheck` run with default rules.

##### 25. `experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py` (Create)
*   **Type / Scope**: GVSoC arch file.
*   **Proposed Changes**: generated with `python -c "from softhier_mlir.sim.gvsoc import Arch; print(Arch().to_py())"` in the experiment environment, with a header naming the source permalink and the command. Expected values: a 4 × 4 cluster grid, 3 cores per cluster, RedMulE array 128 × 32 (rows × columns of processing elements), NoC link width 512 bits, `hbm_chan_placement = (4, 0, 0, 4)`, 64 KB instruction memory. G3.0 checks that the generated file defines every field of `soft_hier_release`'s `flex_cluster_arch.py` and no unknown ones; a mismatch is reported before building.

##### 26. `experiments/smolvla-e2e-baseline/configs/published.env`, `configs/libero.env` (Create)
*   **Type / Scope**: Shell configuration.
*   **Proposed Changes**: Per configuration:
    *   checkpoint id and revision (`run.sh fetch`);
    *   `SHVLA_SMOLVLA_DIR` (`$SHVLA_DATA/smolvla_base` or `$SHVLA_DATA/smolvla_libero`);
    *   the `golden_ref.py --obs` string and the path of its output, `$SHVLA_DATA/golden/<config>.npz`;
    *   the prepare commands, in order (`D` = `$SHVLA_SMOLVLA_DIR`, `C` = `--ckpt $D/model.safetensors`, `R` = the golden-ref npz):
        *   published:
            1. `smolvla prepare C --seq 256 --out $D/vision_s256.npz` (rung 3)
            2. `smolvla prepare C --seq 1024 --out $D/vision_s1024.npz` (the chain's SigLIP weights; `smolvla_e2e` reads this file at every token count)
            3. `smolvla prepare-vlm C --cams 1 --out $D/vlm_c1.npz`
            4. `smolvla_expert prepare C --ref R --out $D/expert.npz`
            5. `smolvla_e2e prepare C --ref R --out $D/e2e_c1_t256.npz`
        *   libero:
            1. `smolvla prepare C --seq 1024 --out $D/vision_s1024.npz`
            2. `smolvla prepare-vlm C --cams 1 --out $D/vlm_c1.npz` (connector weights read by `emit_vision`)
            3. `smolvla prepare-vlm C --cams 2 --out $D/vlm_c2.npz` (the 177-token prefix, G4a.2)
            4. `smolvla_expert prepare C --ref R --out $D/expert.npz`
            5. `smolvla_e2e prepare C --ref R --out $D/e2e_c2_t1024.npz`
        (each module run as `python -m softhier_mlir.frontend.<module>`);
    *   expected stage times (published values or G4a results) and, only for an owner-approved exception, `SHVLA_SIM_TIMEOUT`.

##### 27. `experiments/smolvla-e2e-baseline/run.sh` (Create)
*   **Type / Scope**: Shell script (the only run tool).
*   **Proposed Changes**: As in Component 5: `run.sh apply`, `run.sh <config> fetch`, `run.sh <config> prepare`, `run.sh <config> sim <run-name> -- <script> <args>`. Exit codes: 0 pass, 1 fail, 2 refused (stamp or patch), 3 stopped by the timeout.

##### 28. `experiments/smolvla-e2e-baseline/patches/0001…0005` and `patches/README.md` (Create)
*   **Type / Scope**: Patches on softhier-mlir and their index.
*   **Proposed Changes**: As in the changes table. Each patch has a header with what, why and evidence. The README has one row per patch (files, what, why, evidence run id, upstream candidate yes or no). New patches found under R3 are added here after the owner is told.

##### 29. `experiments/smolvla-e2e-baseline/LOG.md` (Create)
*   **Type / Scope**: Step log.
*   **Proposed Changes**: As in "Step log". Hand-written; run records are pasted from the block `run.sh` prints.

##### 30. `experiments/smolvla-e2e-baseline/SETUP.md` (Create)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**: From a fresh clone to a G4b run, in order:
    1. `setup.sh`;
    2. `build_arch.sh` with the experiment arch;
    3. `uv sync` in both projects;
    4. `golden_ref.py` per configuration;
    5. the weight-prepare commands per configuration;
    6. `run.sh` for each ladder rung and G4 step.
    Each step comes with its measured time and disk use.

##### 31. `experiments/smolvla-e2e-baseline/INPUT_CONFIG.md` (Create)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**: Both configurations side by side, field by field, each value with the robot it comes from (LIBERO's Franka Panda with agentview and wrist cameras; softhier-mlir's synthetic observation for `smolvla_base`) and its source link, plus the arithmetic of the derived counts and the OQ2/OQ3 answers.

##### 32. `experiments/smolvla-e2e-baseline/RESULTS.md` (Create)
*   **Type / Scope**: Documentation (G5), hand-written from `run.sh` records.
*   **Proposed Changes**:
    *   permalinks;
    *   the arch and stamp;
    *   patches;
    *   G3 ladder outcomes and the validity gate;
    *   the G4a estimate and the G4b measurement per stage, with labels and the latency qualifier;
    *   wall time and RSS;
    *   the G6b success rate with its interval, next to the G4 latency.

##### 34. `experiments/smolvla-e2e-baseline/gate.py` (Create)
*   **Type / Scope**: Python, standard library only.
*   **Proposed Changes**: As in Component 6: `gate.py g3 <run folder>` and `gate.py g4a <vision run> <prefix run> <expert run>`. Tested by hand on the rung-6 and G4a runs; its output is pasted into LOG.md.

#### Outside this repo

##### 33. `HaozeG/gvsoc`, branch `shvla/soft_hier_release` (Create when needed, owner approval)
*   **Type / Scope**: Fork commits.
*   **Proposed Changes**: First commit adds `soft_hier/flex_cluster_sdk/runtime/include/flex_alloc_api.h` from the GVSoC branch `bowwang-dev/softhier-deeploy`, with its origin in the message. Later commits each name a side-note number and a run id.

### Verification Plan

*   **Build**:
    *   `setup.sh` completes;
    *   `build_arch.sh` completes for `implementation/config/arch/arch.py` (G1.4) and `arch/softhier_mlir_arch.py` (G3.0), each followed by a passing `stamp.py check`;
    *   `uv sync --locked` passes in the three uv projects.
*   **G1**: `run_kernel_tests.sh` passes `gemm`, `norm`, `acti`, `attn`.
*   **G2**:
    *   `golden_ref.py` for both configurations;
    *   `uv run pytest` in `infra/lerobot`;
    *   G2.3 crosscheck: both programs on the CPU in fp32, same inputs (`from-npz`). Per key, `max|a − b| / max(|b|.max(), 1e-6)` ≤ 1e-4. Both run LeRobot's own modules with eager attention, so differences should come only from operation order. If a key exceeds it, the session stops at the first such key in pipeline order, records both values, and asks the owner before G3: a divergence there means one of the two capture codes is wrong.
*   **G3**:
    *   ladder rungs 1–6 via `run.sh`;
    *   softhier-mlir's `bash tests/run_filecheck.sh` with patches applied;
    *   `gate.py g3` output recorded.
*   **G4**: G4a.1–G4a.4 with `g4a_estimate.json` recorded; G4b run with every phase under 2 h, or the R1 stop with the owner's decision recorded.
*   **G5**: every RESULTS.md permalink opens after the push; every number carries its label.
*   **G6b**: `libero_eval.py` on `libero_spatial`, 10 tasks × 10 episodes, writes `summary.json` with the interval.
*   **Wiki**: `wiki.py check` passes with no new warnings.
*   **Traceability**: every commit subject carries a step id, and every step in the step list has a LOG.md entry.
*   **Linting**: `ruff check` on the Python files; `shellcheck` on the shell scripts.
