# SmolVLA on SoftHier: simulator, golden reference and first latency and success-rate baseline

*   **Author:** Haoze
*   **Status:** Proposed
*   **Date:** 2026-11-10

--------------------------------------------------------------------------------

## Overview

SoftHier-VLA has three goals (README): an MLIR-based deployment toolchain, workload analysis, and high MFU through a software scheduling layer. Today the repo holds only knowledge: the wiki submodule and `workload-modeling/` notes. Nothing in it can run a VLA model on SoftHier, check that a run is correct, or measure whether a policy completes its task.

The project tracks two metrics for a VLA deployment, both taken from the sources already ingested in the wiki:

*   **latency**: the time of one model call (glossary: latency (l)), here simulated on SoftHier. Jung et al. and VLA-HWBench profile VLA calls per stage (`wiki://knowledge/vla-benchmark-workloads.md`).
*   **success rate**: the share of benchmark episodes in which the task is completed. SmolVLA reports it on LIBERO (`wiki://knowledge/simulation-benchmarks.md`). Jung et al. §5 show that the two interact: in closed loop, more inference overlap shortens episodes and lowers success (`wiki://resources/serving/vla-workload-characterization-kaist.md`).

This design adds the first executable pieces in two layers, and files the design needs in the wiki before any code is written:

1.  **Permanent infrastructure in the SoftHier-VLA repo**, reused by every later experiment:
    *   a SoftHier simulator environment: GVSoC branch `soft_hier_release` as a git submodule (the `flex_cluster` model and its bundled SDK), checked with the simulator's own kernel tests (`implementation/README.md`);
    *   a golden reference: LeRobot's SmolVLA on the local GPU in fp32. It produces the reference tensors that any SoftHier implementation is compared against;
    *   a robotics simulation environment for closed-loop success rate (LeRobot eval on MuJoCo), built from the working setup in `~/learn/physical_ai`.
    All three are uv-managed Python environments whose definitions (`pyproject.toml`, `uv.lock`) are tracked in the repo.
2.  **One experiment in a git worktree** (`worktree-smolvla-e2e-baseline`): the reference compiler `runwangdl/softhier-mlir` (latest `main`, recorded as a commit permalink), the hardware configuration it assumes, the end-to-end SmolVLA run, and its documentation.

## The Problem

Every performance and accuracy statement in the project so far comes from papers about GPUs or other chips. Nothing yet says what one SmolVLA inference costs on SoftHier, where that time goes, whether a compiled program computes the right actions, or what success rate the same policy reaches in simulation. Without that baseline:

*   decisions about the MLIR toolchain (goal 1) and the scheduling layer (goal 3) rest on paper figures rather than on runs on the target;
*   later compiler or scheduling work has no reference for speed, correctness or task success;
*   each experiment would rebuild the simulator, the reference and the robot environment from scratch, and results would differ for reasons unrelated to the change being tested.

A reference compiler already exists. softhier-mlir reports one SmolVLA inference end to end on the SoftHier GVSoC: 119.1 ms simulated for 1 camera at 256 tokens, actions within 0.032 of LeRobot. It was developed on a private simulator fork with unpublished model patches, an aarch64 host and hardcoded paths, and its numbers have not been reproduced in an environment this project controls.

The proposed solution:
1. Record in the wiki why these two metrics, and what each experiment needs.
2. Build the simulator, the golden reference and the robot environment once, as maintained parts of this repo.
3. Reproduce softhier-mlir's SmolVLA run on top of them in a worktree, one checked step at a time.
4. Measure the success rate of the same policy on GPU.

## Goals

Goals are listed in execution order; each starts only when the previous one is done.

*   **G0: design needs in the wiki, before implementation.** The wiki gets a project decision in `memories/decisions/` that records:
    *   the two metrics (latency, success rate) and the sources they come from;
    *   the choice of GVSoC `soft_hier_release` and softhier-mlir, and why softhier-sdk is not used;
    *   the layout rule (permanent infrastructure in the repo, experiments in worktrees);
    *   the success-rate method chosen now (option A, below);
    *   the later options B and C, as open options with their requirements.
    New terms (for example "golden reference") get project glossary rows. The full design document is saved under `docs/designs/` in the repo, and the wiki decision links to it.
*   **G1: simulator environment, checked with its own tests.** A fresh checkout of SoftHier-VLA initialises the GVSoC submodule, builds it with one documented procedure, and passes the kernel tests in `implementation/README.md` one at a time: `gemm`, `norm`, `acti`, `attn` (`make <kernel>-run`, with that directory's own arch file).
*   **G2: golden reference on GPU.** One documented command runs LeRobot's `SmolVLAPolicy` in fp32 on a fixed, seeded observation, using the LIBERO-fine-tuned checkpoint (Q7) and also `lerobot/smolvla_base` for the published-configuration check. It saves the reference tensors (vision output, connector output, prefix KV cache, flow state per step, action chunk) in a format the experiment reads. The same observation always gives the same tensors.
*   **G3: numerical match with the golden reference, after G2 and before G4.** In the worktree, softhier-mlir's flow runs on the G1 simulator with HBM placement `(4,0,0,4)`. Its outputs are compared with the G2 tensors rung by rung:
    1. library GEMM;
    2. the bundled MLIR examples;
    3. one SigLIP layer;
    4. the VLM prefix;
    5. the expert flow;
    6. the full 4-program chain.
    Each rung has a stated tolerance, taken from softhier-mlir's own reported errors, which sit at the fp16 floor. A rung passes when the error is within tolerance. Without the private simulator patches, failures are expected here. RedMulE fp16 truncation (patch #3) makes long dot products about 10 % low, and the missing `nseq` tags (#10) make softmax, LayerNorm and GELU depend on binary layout. Each failure is traced to a cause, using the side note in the Technical Plan. It is then fixed (a simulator patch in our GVSoC submodule, or a glue change) or recorded as a known deviation with its measured size. Timing is not the goal of this step, but every rung runs under the timing watchdog (Technical Plan, component 6): a stage whose simulation runs far past its expected time is stopped, and the investigation goes directly to the matching patch symptoms.
*   **G4: first SmolVLA latency baseline on SoftHier, LIBERO configuration.** Once G3 passes, or its deviations are recorded, the chain runs with the same input configuration as the G6 robot evaluation, the LIBERO setup of the fine-tuned checkpoint. Deliverables:
    *   simulated time per stage (vision, connector, prefix, expert, total);
    *   action-chunk error against G2 on the same LIBERO observation;
    *   RedMulE and iDMA busy time per stage from the chain's trace mode, optional if trace mode fails on this simulator.
    Before the LIBERO run, the published configuration (1 camera, 256 tokens per camera; 27.3 / 0.28 / 14.8 / 76.7 ms, total 119.1 ms) is reproduced once as a check on the environment.
    The setup is documented so someone else can repeat it. So is the **input configuration**: robot and benchmark, number of cameras, image size and tokens per camera, language length, state and action dimensions, chunk size, flow steps and control rate. These depend on the robot, and the document states which robot each setting corresponds to.
*   **G5: reproducible record.** A results document in the worktree lists:
    *   the code used, as commit permalinks (`https://github.com/<owner>/<repo>/tree/<commit>`) for SoftHier-VLA, GVSoC and its submodules, and softhier-mlir;
    *   the hardware configuration and every command run;
    *   wall time and peak memory;
    *   every local change ("glue") needed to make the flow run.
    Any gap against the published numbers is explained or listed as unexplained.
*   **G6: success rate on GPU (option A), after G4 and G5.** The robotics environment runs closed-loop LeRobot evaluation of a SmolVLA checkpoint on a LIBERO suite. The GPU policy runs synchronously, so simulation time pauses during inference. The result is a success rate with episode count and seeds, reported next to the G4 latency as two separate numbers. The environment reuses the `~/learn/physical_ai` approach (uv, MuJoCo with osmesa rendering, the system libstdc++ preload) and adds the LIBERO dependencies.

Labels: simulator cycle counts and evaluation success rates are runs, so they count as **measured** (glossary, wiki-design decision 0004), qualified as "simulated on GVSoC, ideal HBM" or "GPU fp32, synchronous eval". Composed or modeled figures are labeled **estimate**.

## Non-Goals

*   **pulp-platform/softhier-sdk.** It targets the newer `pulp.chips.softhier` model, which has no RedMulE and none of the runtime headers softhier-mlir uses.
*   **softhier-mlir's private simulator patches.** Not applied in this pass. The Technical Plan's side note lists what each one fixes, so their symptoms can be recognised and fixed when they appear.
*   **Success rate with SoftHier effects (options B and C).** Not built in this pass; they are recorded in the wiki as later options:
    *   B, latency-injected closed loop: simulation time keeps running during inference by the SoftHier latency, with an action queue, overlap p and stale actions, after Jung et al. §5;
    *   C, B plus a numerical twin of SoftHier's fp16 arithmetic.
*   **Closed-loop episodes on the SoftHier simulator itself.** One inference takes about 48 minutes of wall time, so an evaluation of tens of episodes is infeasible.
*   **More SoftHier configurations:** the published 3-camera configurations, input configurations of other robots or benchmarks, and DRAMSys (non-ideal HBM) runs. The LIBERO configuration (2 cameras at 512×512) and the published 1-camera check are the only ones in scope.
*   **softhier-mlir's design-space exploration** (`dse/`). Not examined.
*   **Changing softhier-mlir's algorithms or kernels.** Only the glue needed to run it here is in scope, and every glue change is logged.
*   **Pinning or forking softhier-mlir.** The latest `main` is used and recorded by commit permalink. Local changes are patch files (decision D2), not a fork.
*   **Baseline numbers in the wiki.** Per wiki-design decision 0013, numbers stay in the worktree's results document with a mnemon pointer. The wiki records the decision and the method (G0), not results, unless asked.

## Open Questions

*   **Q6: hardcoded paths in softhier-mlir** (`/app/models/smolvla_base`, `/app/install/softhier`). Command-line overrides, environment variables, or a symlink. Not decided yet; to be settled in the Detailed Implementation.
*   **Q9: two architecture files, one simulator build.** The G1 kernel tests use `implementation/config/arch/arch.py`: 5 cores, Spatz, RedMulE 32×16, NoC 1024. softhier-mlir assumes 3 cores, RedMulE 128×32, NoC 512. Both use HBM placement `(4,0,0,4)`. GVSoC's `make hw` rebuilds the model for whichever arch file it is given, so moving from G1 to G3 means a rebuild. Resolved as D1 in the Technical Plan: one build per worktree, rebuilt per arch, guarded by a stamp file.

Resolved by the owner: GVSoC as a git submodule (Q1); HBM placement `(4,0,0,4)` (Q2); uv-managed environments with tracked definitions (Q3); Hugging Face cache plus a documented path variable for weights and large data (Q4); commit permalinks instead of bare hashes (Q5); success rate by option A first, B and C recorded as later options; use a LIBERO-fine-tuned checkpoint (Q7); latency is measured with the same input configuration as the robot evaluation, the LIBERO setup (Q8); the exact LIBERO setup in the Technical Plan, confirmed by the owner (Q7); a timing watchdog that stops stages running too long and sends them to patch triage.

## Technical Plan (Architecture)

### Summary

The work splits into two parts:
- **On `main` (permanent):** infrastructure under `infra/` and `third_party/`, the design under `docs/designs/`, and the decision in the wiki.
- **Only on the experiment branch:** `experiments/smolvla-e2e-baseline/`.

Every external code base enters as a git submodule, so its exact commit is part of the repo state and can be printed as a permalink. Every Python environment is a uv project whose `pyproject.toml` and `uv.lock` are tracked. Large data (weights, preload images, reference tensors, logs) stays outside git under two path variables.

```text
SoftHier-VLA repo
+-- main branch (permanent) ------------------------------------------+
| softhier_vla_wiki/   submodule  decision + design needs (G0)        |
| docs/designs/        design document (G0)                           |
| third_party/gvsoc/   submodule  gvsoc fork of soft_hier_release     |
| infra/softhier/      build + test scripts, uv env (G1)              |
| infra/lerobot/       uv env: lerobot[smolvla,libero], GPU           |
|                      golden reference (G2) + LIBERO eval (G6)       |
+---------------------------------------------------------------------+
            | branch worktree-smolvla-e2e-baseline adds:
            v
+-- experiments/smolvla-e2e-baseline/ --------------------------------+
| softhier-mlir/       submodule  runwangdl/softhier-mlir main        |
| arch/                flex_cluster arch file for softhier-mlir       |
| configs/             input configs: published-c1-t256, libero       |
| scripts/             ladder runner, timing watchdog, converter      |
| patches/             glue on softhier-mlir, logged                  |
| SETUP.md  INPUT_CONFIG.md  RESULTS.md      (G3-G5 documentation)    |
+---------------------------------------------------------------------+

outside git:  $HF_HOME (checkpoints)   $SHVLA_DATA (npz, preload ELFs, logs)
```

### Data flow of one validation or latency run

```text
LIBERO env (infra/lerobot)                    softhier-mlir (experiment)
 fixed observation: libero_spatial task 0,
 seed s, first frame after reset
        |                                              |
        v                                              v
 golden reference (GPU fp32) --- ref.npz --> converter --> softhier-mlir
 SmolVLAPolicy hooks:                         (schema     prepare: weights,
 vision, connector, prefix KV,                 mapping)   images, tables
 flow state x10, actions                                     |
        |                                                    v
        |                                  MLIR -> C -> SDK build -> gvsoc
        |                                  4 programs: V, Pa, Pb, X
        |                                  (HBM hand-over by preload, x4)
        |                                                    |
        |                     timing watchdog: tails [mark] lines, stops a
        |                     stage past its budget -> triage report
        |                                                    |
        +------------> compare (per rung, tolerance) <-------+
                                  |
                                  v
                    RESULTS.md: times per stage, errors, glue, permalinks
```

### Components

1.  **GVSoC submodule** (`third_party/gvsoc`, branch `soft_hier_release`, recursive: its `core` and `pulp` submodules).
    *   The build follows its Makefile: first `softhier_preparation`, which applies its own core and pulp patches, builds SystemC and DRAMSys, and fetches the RISC-V toolchain; then `make hw` with an arch file.
    *   The build output (`install/`, `third_party/`, `sw_build/`) is untracked, and the submodule's working tree is never committed to.
    *   A submodule is checked out once per git worktree, so every worktree has its own GVSoC working tree and build.
2.  **`infra/softhier/`**: thin scripts around the GVSoC Makefile, with no simulator logic of its own.
    *   `setup`: checks the host prerequisites (gcc >= 11.2, cmake >= 3.18.1, Python >= 3.11, the apt packages from the GVSoC README), runs `softhier_preparation`, and records the GVSoC commit permalinks.
    *   `build-arch <arch.py>`: runs `make hw` for one arch file and writes a stamp file naming the active arch. Every run script checks the stamp before simulating.
    *   `test`: runs the G1 kernel tests from `implementation/` in order (`gemm`, `norm`, `acti`, `attn`) with `implementation/config/arch/arch.py`, and reports pass or fail per kernel.
    *   A uv project with GVSoC's Python requirements (from its `requirements.txt`: pyelftools, Mako, PyYAML and others), used by `gapy` and the build.
    *   No patch folder: simulator fixes are commits on the GVSoC fork (decision D2). `setup` prints the fork and nested-submodule permalinks.
3.  **`infra/lerobot/`**: one uv project (Python 3.12, lerobot with the `smolvla` and `libero` extras, torch built for the local CUDA). It follows `~/learn/physical_ai`: MuJoCo with `MUJOCO_GL=osmesa`, and the system libstdc++ preloaded. Two entry points:
    *   `golden-ref`:
        *   loads a checkpoint and builds the fixed observation, from the LIBERO env or synthetic for the published-configuration check;
        *   runs `SmolVLAPolicy` in fp32 with forward hooks;
        *   writes `ref.npz` in a documented, compiler-neutral schema: tensor names and shapes, the checkpoint permalink, the observation seed and the lerobot version.
        The same inputs always give the same file; a rerun checks this with a hash.
    *   `libero-eval`: wraps `lerobot-eval --env.type=libero` with the suite, episode count, seeds and output folder. It writes a summary: success rate per task and suite, episodes, seeds, checkpoint permalink. Runs follow physical_ai's rule: one job at a time, under a memory cap.
4.  **softhier-mlir submodule** (`experiments/smolvla-e2e-baseline/softhier-mlir`, branch `main`).
    *   The submodule commit is the permalink recorded in RESULTS.md.
    *   Its uv project (xdsl, numpy, the dev extras) is defined by the experiment, because softhier-mlir ships no lock file.
    *   softhier-mlir's own reference script (`smolvla_expert_ref.py`) is kept for diagnosis only. When it and the infra golden reference disagree, the converter or the frontend is suspect, not the device.
5.  **Experiment folder** (`experiments/smolvla-e2e-baseline/`):
    *   `arch/softhier_mlir_arch.py`: softhier-mlir's `Arch()` values written out: 16 clusters, 3 cores, RedMulE 128×32, NoC 512, HBM placement `(4,0,0,4)`.
    *   `configs/`: one file per input configuration (`published-c1-t256`, `libero`) with every field of the setup table below, plus a tolerance per rung and the watchdog budget per stage.
    *   `scripts/`:
        *   the ladder runner: G3 rungs 1–6, then G4, each through softhier-mlir's `tests/gvsoc/run.py`;
        *   the converter from `ref.npz` to softhier-mlir's npz inputs;
        *   the timing watchdog;
        *   the RESULTS.md writer.
    *   `patches/`: glue changes on softhier-mlir as patch files, applied on top of the submodule commit and listed in RESULTS.md (decision D2).
    *   Documents:
        *   `SETUP.md`: how to repeat the run;
        *   `INPUT_CONFIG.md`: the robot-dependent configuration, and which robot each value comes from;
        *   `RESULTS.md`: the G5 record.
6.  **Timing watchdog** (experiment `scripts/`).
    *   **Budgets:** each stage of each rung has an expected simulated time. For the published configuration these are the published values; for the LIBERO configuration, composed estimates from softhier-mlir's ledger.
    *   **Calibration:** the wall-time-per-simulated-ms ratio is measured on rung 1. The published host reached about 24 s per simulated ms, using host-speed models we do not have.
    *   **Monitoring:** the watchdog streams the simulator log, follows the `[mark]` stamps that the compiled programs print per layer, and acts on three conditions:
        *   **over budget:** a stage passes k times its expected wall time (k is set in the config, default 3);
        *   **no progress:** no new `[mark]` line within a set interval;
        *   **known error text:** an error message from the patch side note appears in the log, for example "No entry found for burst" or "Executing illegal instruction".
    *   **On a trigger** it stops the simulator, keeps the log, and writes a triage report. The report names the stage, the simulated time reached, and the matching candidates from the side note. For example, a slow stage dominated by softmax loops points to #7 (instruction fetch) for speed and #10 for correctness. The investigation starts there instead of waiting for the run to finish.
7.  **Wiki and design (G0).**
    *   One project decision, `memories/decisions/2026-11-10-first-baseline-latency-and-success-rate.md`, with `Context`, `Decision` and `Why`. It cites the wiki sources for the two metrics (Jung et al., VLA-HWBench, the SmolVLA paper, simulation benchmarks) and records options B and C with their requirements as open later options.
    *   The design document is copied to `docs/designs/` on `main`.
    *   Wiki commits follow the wiki's own rules: on a branch they commit per operation; on `main` they wait for the owner.

### Confirmed LIBERO input configuration (Q7, Q8)

| item              | value                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| checkpoint        | `lerobot/smolvla_libero`, fine-tuned from `smolvla_base` on `lerobot/libero` (25k steps, batch 32)        |
| architecture      | SmolVLM2-500M-Video-Instruct, 16 VLM layers, expert width 0.75 (same as `smolvla_base`)                  |
| cameras           | 2: agentview (`camera1`) and wrist (`camera2`), 256×256 in the dataset                                   |
| vision tokens     | resized with padding to 512×512: 1024 SigLIP patches per camera, 64 image tokens after the pixel shuffle |
| language          | padded to 48 tokens                                                                                      |
| state, action     | state 8-D (to be checked), action 7-D, both padded to 32                                                 |
| prefix            | 2 × 64 + 48 + 1 = 177 tokens (padded length per softhier-mlir's rule, to be checked)                     |
| expert            | chunk 50, 10 flow steps, whole chunk executed (`n_action_steps = 50`)                                    |
| control rate      | 30 fps; chunk duration 50 / 30 ≈ 1.67 s                                                                  |
| G2/G4 observation | `libero_spatial` task 0, fixed seed, first frame after reset                                             |
| G4 simulation     | every stage simulated in full, not composed                                                              |
| G6 evaluation     | first `libero_spatial`, 10 tasks × 10 episodes; then 4 suites × 10 tasks × 50 episodes                   |

softhier-mlir's device runs cover only 65- and 97-token prefixes. Vision at 1024 tokens per camera was simulated for one layer and composed, and the expert at a 177-token prefix was never run. G4 is therefore the first end-to-end device run of this configuration.

### Side note: softhier-mlir's simulator patches (not applied; watch for their symptoms)

From softhier-mlir `docs/SIMULATOR_NOTES.md`. The upstream `soft_hier_release` may already change some of these.

| #           | Symptom                                                                                 | Cause in the model                                                                                         |
| ----------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1           | stale `flw`/`fsw` to HBM or remote TCDM; NoC error "No entry found for burst"           | Snitch FP subsystem completes an offloaded access after a fixed latency even when it is still pending      |
| 2           | `flh`/`fsh` return garbage                                                              | Zfh half loads decoded as integer loads; the library already avoids scalar fp16                            |
| 3           | long fp16 dot products about 10 % low                                                   | `light_redmule.cpp` truncates instead of rounding to nearest even                                          |
| 4           | simulator segfault with two outstanding collective broadcasts                           | NoC collective model; the library already waits after each broadcast                                       |
| 6           | `fcvt.w.s` then `fcvt.s.w` returns a stale integer                                      | FP-to-int result forwarded before it lands; the library avoids the pattern                                 |
| 7           | three cores no faster than one; loop speed depends on its address                       | one shared instruction router, no instruction cache; affects speed and cost, not correctness               |
| 10          | softmax, LayerNorm, GELU results depend on binary layout, mostly with `--cluster all`   | scalar FP-to-int instructions lack the `nseq` tag, so the integer core reads the register too early        |
| 11          | illegal instruction at a data address late in a long program                            | 1 KB stack spacing in `flex_start.s`; the library runs on private 40 KB stacks                             |
| 12          | out-of-bound TCDM request right after a RedMulE trigger                                 | register clobber in `flex_redmule_trigger`; the library uses its own trigger                               |
| 13, 14      | in-network REDADD adds stale data; fp16 REDADD truncates                                | iDMA burst buffers never cleared; `floonoc.cpp` truncates fp16; the library flushes and avoids             |
| models_fast | wall time above about 24 s per simulated ms                                             | host-speed rebuild of the Snitch models, not public                                                        |

### Decisions pending

*   **D1 (Q9), decided: one build per worktree, rebuilt when the arch changes.** Each worktree builds its own GVSoC submodule. `build-arch` rebuilds it for the requested arch file and writes a stamp file: the arch file's path and content hash, plus the GVSoC submodule state, meaning its commit and the patches applied. Every run script compares the stamp with the arch its config asks for and refuses to simulate on a mismatch. Runs that need different arch files in one worktree must run one after another. Two concurrent runs in one worktree are only allowed when they use the same arch. The rebuild time is measured in G1 and recorded in SETUP.md.
*   **D2, decided: simulator fixes in forks, softhier-mlir glue as patch files.**
    *   **Simulator side:** the `third_party/gvsoc` submodule points to a fork on the owner's GitHub (`HaozeG/gvsoc`), project branch based on `soft_hier_release`. When a fix touches the nested `core` or `pulp` submodules, those get forks too, and the fork's `.gitmodules` points at them. Until the first fix lands, the submodule may point at upstream directly; switching to the fork is one `.gitmodules` change. Each fix is one commit, with the matching side-note number and the G3 evidence in its message, so it can be offered upstream as a pull request. Creating the forks publishes them, so it waits for the owner's go-ahead at implementation time.
    *   **Compiler side:** softhier-mlir is treated as an initial implementation under inspection. The submodule follows upstream `main` with no fork. Every local change is a patch file in `experiments/smolvla-e2e-baseline/patches/`, one change per file, with a header stating what it changes, why, and the G3 or G4 evidence. The patch folder doubles as the inspection record: it lists exactly where this environment had to deviate from the published implementation. When an upstream update makes a patch fail to apply, the runner stops and reports which one.
    *   **Boundary rule:** a change belongs to the repo that owns the file. The SDK under `soft_hier/flex_cluster_sdk/` (including the missing `flex_alloc_api.h`) and every model file belong to GVSoC, so they go to the fork. Files under the softhier-mlir tree (runtime, frontend, `sim/gvsoc.py`, tests) are patch files. A change that needs both is split into one fork commit and one patch file that refer to each other.
*   **Q6: hardcoded paths in softhier-mlir.** Settled in the Detailed Implementation.

## Alternatives Considered

Grouped by the question they answer. Options B and C for success rate are deferred, not rejected; they are kept in the wiki as later options (G0).

### Simulator and SDK

#### Alternative A: pulp-platform/softhier-sdk with the new `pulp.chips.softhier` model (gvsoc-pulp master)

*   **Pros**: The newest public SoftHier model and SDK (2026), with more NoC topologies (2D mesh and torus, 3D, rings, hexamesh) and active upstream development.
*   **Cons**: The model has no RedMulE matrix engine (Spatz vector unit, one core per cluster), and its 158-line runtime has none of the `flex_*` headers softhier-mlir includes (`flex_redmule.h`, `flex_group_barrier.h`, `flex_printf.h`, ...).
*   **Reason for Rejection**: softhier-mlir's whole runtime is built on RedMulE GEMMs. Porting it would mean rewriting every GEMM for Spatz, and the published numbers would no longer apply. The owner chose `soft_hier_release` with its bundled SDK and dropped softhier-sdk.

#### Alternative B: apply softhier-mlir's private simulator patches up front

*   **Pros**: Numbers closer to the published ones from the start; fewer G3 failures; host-speed models would cut wall time.
*   **Cons**: The patches are not published. They would have to be rewritten from the descriptions in `SIMULATOR_NOTES.md`, without knowing which ones the current `soft_hier_release` already makes unnecessary, and with effort spent before any evidence that they are needed.
*   **Reason for Rejection**: The owner chose to skip them and fix problems as they appear. G3 and the timing watchdog find out which ones matter, and the side note maps symptoms to candidate patches.

### Repository layout and environments

#### Alternative C: install everything in a new home directory outside the repo (for example `~/softhier-clean/`)

*   **Pros**: No change to the repo; quick to start; nothing to maintain.
*   **Cons**: Treats the work as a throwaway experiment. Nothing records the commits, environments or setup steps, and the next experiment starts again from zero.
*   **Reason for Rejection**: The owner wants this to be the first step of the project's infrastructure. Reused parts belong in the repo, and experiment parts in a worktree.

#### Alternative D: reuse the earlier installs (`~/softhier`, `~/softhier-mlir`, `~/softhier-mlir-home`)

*   **Pros**: Already built; saves the build time.
*   **Cons**: Unknown branches, patches and local edits; results could not be attributed to a known state.
*   **Reason for Rejection**: The owner asked for a new environment that does not depend on earlier implementations.

#### Alternative E: GVSoC cloned by a setup script into a gitignored folder (instead of a submodule)

*   **Pros**: Keeps `git status` and the repo's submodule list small; the clone can be deleted without touching the repo.
*   **Cons**: The exact commit lives only in a script variable, which can drift from what is actually checked out. Changes to the dependency do not show in the parent's history, and worktrees do not get their own checkout automatically.
*   **Reason for Rejection**: A submodule records the dependency's exact commit in every parent commit. Each bump appears in `git log`, permalinks can be generated from the repo state, and each git worktree gets its own checkout, which decision D1 relies on. The owner chose it for this traceability (Q1). The known costs of submodules are accepted and handled in setup: clones need `--recursive`, checkouts are on a detached HEAD, and GVSoC's own preparation leaves its nested submodules modified (`ignore = dirty` in `.gitmodules`).

#### Alternative F: conda environments, or untracked venvs as in `~/learn/physical_ai`

*   **Pros**: conda is already installed; physical_ai's `.venv` already works for LeRobot on this GPU.
*   **Cons**: Neither records an exact, repeatable dependency set in the repo; physical_ai has no `pyproject.toml` or lock file.
*   **Reason for Rejection**: The owner chose uv-managed environments with tracked definitions (Q3). physical_ai is reused for its approach (osmesa rendering, libstdc++ preload, one job at a time), not for its environment.

#### Alternative G: a repo-local `data/` folder for weights and large files

*   **Pros**: One place, simple paths.
*   **Cons**: Duplicates checkpoints already in the Hugging Face cache. Other users keep their caches elsewhere (`HF_HOME`). Checkpoints are large binaries with their own licences and distribution channel.
*   **Reason for Rejection**: Large binaries do not belong in git (they would need Git LFS), and the Hugging Face cache is shared across projects and placed per user. The repo records the exact checkpoint revision instead (a Hub commit permalink, passed as `revision` when loading). Data stays reproducible without being stored, so the owner chose the cache plus a documented path variable (Q4).

#### Alternatives H and I: one shared simulator build rebuilt per arch / one install copy per arch / G1 only on `main` (decision D1)

*   **Pros**: A copy per arch allows runs with different arch files at the same time. Keeping G1 on `main` avoids any arch switching in the experiment worktree.
*   **Cons**: A copy per arch costs several GB each and must be refreshed after every simulator fix. G1 only on `main` needs two full builds, and fixes made in the experiment must reach `main` before the G1 build has them.
*   **Reason for Rejection**: The owner chose one build per worktree, rebuilt per arch and guarded by a stamp file (decision D1).

#### Alternatives J and K: patch files for everything / forks for everything (decision D2)

*   **Pros**: Patch files only: no forks to maintain, and they follow GVSoC's own `gvsoc_core.patch` method. Forks only: one commit permalink describes all the code, and fixes go upstream as pull requests.
*   **Cons**: Patch files only: dirty submodules, and two pieces (upstream commit plus patch) to trace each simulator fix. Forks only: up to four forks to keep in sync, and a softhier-mlir fork contradicts using its latest `main`.
*   **Reason for Rejection**: The owner chose forks for the simulator and patch files for softhier-mlir, because softhier-mlir is an initial implementation under inspection (decision D2).

#### Alternative L: pin softhier-mlir to a fixed commit, or record runs by bare commit hash

*   **Pros**: Exact reruns; a hash is short.
*   **Cons**: Pinning freezes an implementation that is still under inspection; a bare hash is not a link anyone can open.
*   **Reason for Rejection**: The owner chose latest `main`, recorded as commit permalinks (Q5).

### Input configuration and measurement

#### Alternative M: measure latency with the published synthetic configuration (1 camera, 256 tokens per camera)

*   **Pros**: Already validated end to end by softhier-mlir (119.1 ms); the shortest simulation.
*   **Cons**: It is not the configuration of any robot evaluation, so latency and success rate would describe different setups.
*   **Reason for Rejection**: The owner chose to measure latency with the same LIBERO configuration as the robot evaluation (Q8). The published configuration is kept only as an environment check.

#### Alternative N: feed the LIBERO 256×256 images to SigLIP directly (256 patches per camera)

*   **Pros**: About 6 times less vision work and wall time than 1024 patches per camera.
*   **Cons**: The fine-tuned checkpoint resizes images to 512×512 before SigLIP, so this changes the computation and the numerics against the golden reference.
*   **Reason for Rejection**: It would no longer be the configuration the policy was trained and evaluated with.

#### Alternative O: compose the G4 vision time from one simulated layer (as softhier-mlir's ledger does)

*   **Pros**: Hours less wall time.
*   **Cons**: A composed number is an estimate, not a run, and it skips the end-to-end hand-over that G4 is meant to exercise.
*   **Reason for Rejection**: The owner confirmed that G4 simulates every stage in full (Q7).

#### Alternative P: checkpoint `HuggingFaceVLA/smolvla_libero`, or `lerobot/smolvla_base`

*   **Pros**: `HuggingFaceVLA/smolvla_libero` is also first-party and more liked on the Hub. `smolvla_base` is the checkpoint softhier-mlir was validated with.
*   **Cons**: `HuggingFaceVLA/smolvla_libero` uses a different VLM (SmolVLM2-500M-Instruct, not the Video variant), a different expert width (0.5) and different layer settings, so softhier-mlir's frontend shapes do not apply. `smolvla_base` is not fine-tuned on LIBERO.
*   **Reason for Rejection**: `lerobot/smolvla_libero` is fine-tuned on LIBERO (the owner asked for a fine-tuned checkpoint) and has the same architecture as `smolvla_base`.

#### Alternative Q: closed-loop episodes on the SoftHier simulator

*   **Pros**: Success rate with SoftHier's real latency and numerics, with nothing modeled.
*   **Cons**: One inference is about 48 minutes of wall time, so an episode takes hours and an evaluation of tens of episodes takes weeks.
*   **Reason for Rejection**: Infeasible at current simulator speed. Option B (latency injected into a GPU closed loop) and option C (B plus a numerical twin) approximate it, and both are deferred to later work (G0, wiki).

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Branches, commits and order

Everything is implemented in this worktree (`worktree-smolvla-e2e-baseline`), because D1 gives each worktree its own simulator build and building twice would cost hours.

*   **Infrastructure commits first:** G0 (repo side), G1, G2 and the G6 environment. They touch only `.gitmodules` (the GVSoC entry), `.gitignore`, `README.md`, `CLAUDE.md`, `docs/`, `infra/`, `third_party/` and the wiki submodule pointer. A branch `infra-baseline` is set at the last of them and offered to `main` as a pull request; merging is the owner's call.
*   **Experiment commits after it:** they touch only `experiments/` and the softhier-mlir entry in `.gitmodules`. They stay on the worktree branch.
*   **Wiki:** commits go into the wiki submodule on a branch named like the worktree, one commit per finished operation (wiki rules). The parent repo bumps the submodule pointer in the G0 commit.
*   **Execution order:** G0 → G1 → G2 → G3 → G4 → G5 → G6. Each goal ends with a commit and a note in RESULTS.md (from G3 on).

Steps that need the owner, asked at the time they come up:
1. `sudo apt-get install` of GVSoC's prerequisites, if any are missing.
2. Creating the public forks (`HaozeG/gvsoc`, and `gvsoc-core` or `gvsoc-pulp` if a fix needs them). The first fork commit is expected early in G3: the `flex_alloc_api.h` header that softhier-mlir's runtime includes is missing from `soft_hier_release`.
3. Pushing branches and opening the `infra-baseline` pull request.

### Implementation record for manual inspection

The owner inspects the work while it is being done, so every step leaves a trace that can be checked without asking.

*   **Step log.** `experiments/smolvla-e2e-baseline/LOG.md` (file 47) gets one entry per step, written when the step ends, success or not. Each entry has:
    *   step id (for example `G1.3`) and title;
    *   date and time;
    *   what was done and why;
    *   the exact commands run;
    *   the result in a few lines (pass or fail, key numbers, error text);
    *   files changed;
    *   the commit permalink;
    *   log and record paths under `$SHVLA_DATA`;
    *   **What to inspect**: the one or two things the owner should look at to check this step (a file, a diff, a log line, a number against its reference).
    Failed attempts and dead ends get entries too, marked as such, so the path to a fix is visible.
*   **Small commits.** One commit per step, with the step id in the subject (`[G1.3] run gemm kernel test`). A step that changes nothing (a run that only produces logs) still gets its log entry, committed with the next change. Infrastructure steps log here too and point to the infra commits by permalink, so one file follows the whole effort.
*   **Step list.** The log opens with the planned steps per goal (G0.1 … G6.n) and a status column (planned, in progress, done, failed, skipped). It is updated as steps finish, so the current position is visible at a glance.
*   **Goal checkpoints.** At the end of each goal the log gets a short summary, and the session reports it in chat with the inspection pointers. Work continues into the next goal unless the owner says stop. The exceptions are the owner-approval steps listed above, which always wait.
*   **Large outputs** (simulator logs, traces, npz files) stay under `$SHVLA_DATA`. The log names their paths and quotes only the lines that matter.

### Paths and environment variables (resolves Q6)

| variable            | default                                  | used by                                                       |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------- |
| `SHVLA_ROOT`        | the repo or worktree root (detected)     | every script                                                  |
| `SHVLA_DATA`        | `~/.cache/softhier-vla`                  | preload ELFs, npz files, logs, run records, eval outputs      |
| `HF_HOME`           | Hugging Face default (`~/.cache/huggingface`) | checkpoint downloads                                     |
| `SOFTHIER_HOME`     | `$SHVLA_ROOT/third_party/gvsoc`          | softhier-mlir's `sim/gvsoc.py` (it already reads it)          |
| `SOFTHIER_CHROOT`   | empty (x86_64 host, toolchain runs natively) | softhier-mlir's `sim/gvsoc.py`                            |
| `SOFTHIER_IDEAL_HBM`| `1`                                      | every SoftHier run in this design                             |
| `SHVLA_SMOLVLA_DIR` | `$SHVLA_DATA/smolvla`                    | softhier-mlir paths that are constants today (patch 0001)     |

`infra/env.sh` sets the defaults, and `infra/env.local.sh` (gitignored, optional) overrides them per user.

**Q6, hardcoded paths in softhier-mlir.**
*   Command-line flags are used wherever softhier-mlir has them: `--ckpt`, `--ref`, `--out`, `--npz`.
*   The remaining constants are `MODELS` in `frontend/smolvla_e2e.py`, the cache paths in `frontend/smolvla.py`, and `SHARED_HOME` in `sim/gvsoc.py`. Patch 0001 makes them read `SHVLA_SMOLVLA_DIR` and `SOFTHIER_HOME`, falling back to the original values.
*   No `/app` symlink: it would need root, and it would be a machine-level change that the repo cannot show.

### Target Files to Create/Modify

#### Repository root and design (`main`, via `infra-baseline`)

##### 1. `.gitmodules` (Modify)
*   **Type / Scope**: Git configuration.
*   **Proposed Changes**: Add `third_party/gvsoc`: URL `https://github.com/gvsoc/gvsoc.git`, switched to `https://github.com/HaozeG/gvsoc.git` when the fork exists (D2); `branch = soft_hier_release`; `ignore = dirty`, because GVSoC's preparation patches its nested submodules. The experiment branch later adds `experiments/smolvla-e2e-baseline/softhier-mlir` (file 34).

##### 2. `third_party/gvsoc` (Create, submodule)
*   **Type / Scope**: Git submodule (gitlink).
*   **Proposed Changes**: GVSoC `soft_hier_release` at the commit current when added, nested submodules initialised recursively. G1's simulator.

##### 3. `.gitignore` (Modify)
*   **Type / Scope**: Git configuration.
*   **Proposed Changes**: Ignore `.venv/` in any folder, `infra/env.local.sh`, `__pycache__/`, and `experiments/*/build/`. These are the per-run SDK build folders softhier-mlir creates when `SOFTHIER_BUILD_DIR` points inside the experiment.

##### 4. `README.md` (Modify)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**: Add three things:
    *   a "Repository layout" section: wiki, `docs/designs/`, `third_party/`, `infra/softhier`, `infra/lerobot`, `experiments/`;
    *   the rule that reused parts live on `main` and experiments in worktrees;
    *   the two tracked metrics (latency, success rate), linking to the wiki decision.
    Quick-start pointers go to the two infra READMEs. The three goals stay as they are.

##### 5. `CLAUDE.md` (Modify)
*   **Type / Scope**: Agent instructions.
*   **Proposed Changes**: Add a short "Layout" section stating where infrastructure and experiments live, and that RESULTS.md files hold baseline numbers (not the wiki). The memory-routing rules stay unchanged.

##### 6. `infra/env.sh` (Create)
*   **Type / Scope**: Shell environment.
*   **Proposed Changes**: Detects `SHVLA_ROOT`, sets the defaults from the table above, then sources `infra/env.local.sh` if it exists. It prints nothing unless asked (`--show` prints the resolved values). Sourced by every script in `infra/` and `experiments/`.

##### 7. `infra/env.local.sh.example` (Create)
*   **Type / Scope**: Documentation by example.
*   **Proposed Changes**: Commented examples of per-user overrides (`SHVLA_DATA` on another disk, a custom `HF_HOME`).

##### 8. `docs/designs/smolvla-softhier-baseline.md` (Create)
*   **Type / Scope**: Design document.
*   **Proposed Changes**: The final version of this draft (Step 6), linked from the wiki decision and from `README.md`.

#### `infra/softhier/` (`main`)

##### 9. `infra/softhier/README.md` (Create)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   host prerequisites (apt packages, compilers, Python);
    *   the three commands (`setup.sh`, `build_arch.sh`, `run_kernel_tests.sh`) and the stamp rule (D1);
    *   how to switch the submodule to the fork (D2);
    *   the measured build and rebuild times;
    *   disk use;
    *   the known differences from softhier-mlir's original simulator (the side note).

##### 10. `infra/softhier/pyproject.toml` and `infra/softhier/uv.lock` (Create)
*   **Type / Scope**: uv project.
*   **Proposed Changes**: Python 3.12 managed by uv. Dependencies are GVSoC's `requirements.txt` (pyelftools, Mako, PyYAML, hjson, rich, pandas, matplotlib, tabulate, prettytable, pexpect, and the others listed there) with lower bounds as given; the lock fixes the exact versions. No package code; the project exists to provide the environment.

##### 11. `infra/softhier/setup.sh` (Create)
*   **Type / Scope**: Shell script.
*   **Proposed Changes**:
    1. Sources `infra/env.sh` and checks gcc >= 11.2, cmake >= 3.18.1, uv, and the apt packages from the GVSoC README. Missing packages are listed with the install command; nothing is installed without the owner.
    2. Runs `git submodule update --init --recursive third_party/gvsoc` and `uv sync` in `infra/softhier`.
    3. Runs `make softhier_preparation` inside the submodule with the uv environment active.
    4. Calls `stamp.py permalinks`.
    Idempotent: completed steps are detected and skipped.

##### 12. `infra/softhier/build_arch.sh` (Create)
*   **Type / Scope**: Shell script.
*   **Proposed Changes**: `build_arch.sh <arch.py>`:
    1. checks the submodule state;
    2. runs `cfg=<arch.py> make hw` in the submodule and times it;
    3. on success, calls `stamp.py write <arch.py>`.
    On failure the old stamp is removed, so no run script trusts a half-built simulator.

##### 13. `infra/softhier/run_kernel_tests.sh` (Create)
*   **Type / Scope**: Shell script (G1 test runner).
*   **Proposed Changes**:
    *   Builds with `third_party/gvsoc/implementation/config/arch/arch.py` if the stamp differs, then runs `make <k>-run` in `third_party/gvsoc/implementation` for `gemm`, `norm`, `acti`, `attn`, in that order, stopping at the first failure.
    *   Each log goes to `$SHVLA_DATA/softhier/g1/<k>.log`.
    *   Pass criteria per kernel: exit code 0, a "[Performance Counter]" line, and none of the error texts listed in the side note. If a kernel prints its own check result, that result is used as well; the exact strings are read from its `main.c` during G1 and written into the README.
    *   Prints a table: kernel, pass or fail, simulated ns, wall seconds.

##### 14. `infra/softhier/stamp.py` (Create)
*   **Type / Scope**: Python helper (standard library only), shared with the experiment.
*   **Proposed Changes**: Subcommands:
    *   `write <arch.py>`: records the arch path and SHA-256, the GVSoC commit, the nested submodule commits, and whether the tree is dirty. Written to `third_party/gvsoc/.shvla_stamp.json`, untracked and covered by `ignore = dirty`.
    *   `check <arch.py>`: exit 0 on a match, 1 with the difference otherwise.
    *   `permalinks`: turns each submodule's remote URL and commit into `https://github.com/<owner>/<repo>/tree/<commit>`, as text or JSON.
    The experiment's RESULTS.md writer imports the permalink function.

#### `infra/lerobot/` (`main`)

##### 15. `infra/lerobot/README.md` (Create)
*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   environment setup (`uv sync`, `source env.sh`) and the osmesa and libstdc++ notes carried over from `~/learn/physical_ai`;
    *   the `golden-ref` and `libero-eval` commands with examples;
    *   the reference file schema (pointing to `schema.py`);
    *   the one-job-at-a-time memory cap (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`);
    *   the observed eval speed.

##### 16. `infra/lerobot/pyproject.toml` and `infra/lerobot/uv.lock` (Create)
*   **Type / Scope**: uv project with a small package `shvla_lerobot`.
*   **Proposed Changes**:
    *   Python 3.12; `lerobot[smolvla,libero]`, pinned to the newest release (0.7.0 today) that loads `lerobot/smolvla_libero` at its recorded revision, falling back to 0.6.1 (the version proven in physical_ai);
    *   torch from the PyTorch CUDA wheel index matching the driver, set as a uv index; numpy;
    *   console scripts `golden-ref` and `libero-eval`;
    *   dev dependency pytest.

##### 17. `infra/lerobot/env.sh` (Create)
*   **Type / Scope**: Shell environment.
*   **Proposed Changes**: Sources `infra/env.sh`, activates the uv environment, and sets `LD_PRELOAD` to the system `libstdc++.so.6`, `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`, as in physical_ai.

##### 18. `infra/lerobot/shvla_lerobot/schema.py` (Create)
*   **Type / Scope**: Python module.
*   **Proposed Changes**: Defines the compiler-neutral reference file as data:
    *   `meta`: checkpoint id, revision permalink, lerobot version, torch version, observation descriptor, seed, dtype.
    *   `inputs`: camera images after the policy's preprocessing, per camera `[3,512,512]`; language token ids `[48]` and attention mask; state `[32]` after normalisation and padding; initial noise `[50,32]`.
    *   `vision.<cam>`: SigLIP output per camera.
    *   `connector.<cam>`: image tokens after the pixel shuffle and connector.
    *   `prefix.emb`: prefix embeddings `[177,960]`, plus the token-class and padding masks.
    *   `kv.<layer>.k` and `kv.<layer>.v`: the KV cache as lerobot stores it, after RoPE, for layers 0..15.
    *   `flow.x.<step>`: flow state after steps 0..9.
    *   `actions`: before and after unnormalisation.
    Shapes are checked against the checkpoint config by a `validate(path)` function. Names, shapes and the point in lerobot's code where each tensor is taken are documented in the module docstring.

##### 19. `infra/lerobot/shvla_lerobot/golden_ref.py` (Create)
*   **Type / Scope**: Python module, `golden-ref` entry point.
*   **Proposed Changes**: Arguments:
    *   `--checkpoint`, `--revision` (passed to `from_pretrained`);
    *   `--obs`: `libero:<suite>:<task>:<seed>` or `synthetic:<cams>:<img>:<seed>`;
    *   `--out`.
    Flow:
    1. Build the observation: LIBERO env reset with the seed, first frame; or seeded synthetic images like softhier-mlir's reference.
    2. Run the policy's preprocessor.
    3. Draw the initial noise from a seeded generator and pass it in explicitly.
    4. Run `SmolVLAPolicy` in fp32 on the GPU with forward hooks on the vision tower, connector, the VLM layers' KV writes and each flow step.
    5. Write the file and print its SHA-256.
    Deterministic settings: cuDNN deterministic, TF32 off.

##### 20. `infra/lerobot/shvla_lerobot/libero_eval.py` (Create)
*   **Type / Scope**: Python module, `libero-eval` entry point.
*   **Proposed Changes**:
    *   Wraps `lerobot-eval` with `--policy.path`, the revision, `--env.type=libero`, `--env.task=<suite>`, `--eval.n_episodes`, the seed, `--policy.device=cuda` and an output folder under `$SHVLA_DATA/eval/`.
    *   Afterwards it reads lerobot's eval info and writes `summary.json`: success rate per task and per suite, episode count, seeds, checkpoint permalink, lerobot version, wall time.
    *   G6 runs `libero_spatial` with 10 tasks × 10 episodes first, then the 4 suites × 10 tasks × 50 episodes.

##### 21. `infra/lerobot/tests/test_golden_ref.py` (Create)
*   **Type / Scope**: pytest.
*   **Proposed Changes**:
    *   (a) the schema validates on a generated reference;
    *   (b) two runs with the same arguments give byte-identical files (same SHA-256);
    *   (c) a different seed changes `inputs` and `actions`;
    *   (d) the reference's `actions` equal `policy.select_action` output on the same observation and noise.
    GPU tests are marked and skipped without CUDA.

#### Wiki (submodule `softhier_vla_wiki`, branch named like the worktree)

##### 22. `softhier_vla_wiki/memories/decisions/2026-11-10-first-baseline-latency-and-success-rate.md` (Create)
*   **Type / Scope**: Wiki project decision (`type: decision`; sections `Context`, `Decision`, `Why`, `Open questions`).
*   **Proposed Changes**:
    *   **Context:** the project has no executable baseline, and the sources measure VLA deployments by latency and by success rate, which interact in closed loop.
    *   **Decision:**
        *   the two metrics;
        *   GVSoC `soft_hier_release` with its bundled SDK and softhier-mlir (not softhier-sdk);
        *   the layout rule;
        *   the LIBERO configuration with `lerobot/smolvla_libero`;
        *   success rate by option A first.
    *   **Why:** cited sources (`resources/serving/vla-workload-characterization-kaist.md` §5, `resources/serving/vla-hwbench-edge-latency.md`, `resources/models/smolvla.md`, `knowledge/simulation-benchmarks.md`).
    *   **Open questions:** options B and C, with the requirements each needs (B: a timing wrapper around the LIBERO env, an action queue, overlap p, stale actions, a latency from G4; C: a numerical twin of SoftHier's fp16 arithmetic validated against the device).
    *   Links to `docs/designs/smolvla-softhier-baseline.md` by its GitHub URL on `main`.
    *   Lists `defines: [golden reference]`.

##### 23. `softhier_vla_wiki/memories/decisions/_overview.md` (Modify)
*   **Type / Scope**: Wiki catalog.
*   **Proposed Changes**: Replace "Empty until populated." with the catalog line for the new decision.

##### 24. `softhier_vla_wiki/glossary/project-terms.md` (Modify)
*   **Type / Scope**: Wiki glossary.
*   **Proposed Changes**: Add a project row: "golden reference: the fp32 GPU run of a model's original implementation whose intermediate tensors and outputs every SoftHier implementation is compared against", defined in the new decision.

##### 25. `softhier_vla_wiki/glossary/simulation-and-evaluation.md` (Modify)
*   **Type / Scope**: Wiki glossary.
*   **Proposed Changes**: Add a field row for "success rate", the share of episodes in which the task is completed, defined in `knowledge/simulation-benchmarks.md`. Today the term appears only inside the "episode" row.

##### 26. `softhier_vla_wiki/glossary/a-to-z.md` (Modify, generated)
*   **Type / Scope**: Generated index.
*   **Proposed Changes**: Regenerated with `wiki.py glossary --az` after files 24 and 25.

##### 27. `softhier_vla_wiki/knowledge/softhier-and-the-project.md` (Modify)
*   **Type / Scope**: Wiki knowledge note.
*   **Proposed Changes**: In "Goals", one sentence and a link: the metrics the project tracks are recorded in the new decision. No numbers; wiki-design decision 0013 still applies.

Wiki checks before each wiki commit: `wiki.py fmt`, `wiki.py check`.

#### Experiment (`experiments/smolvla-e2e-baseline/`, worktree branch only)

##### 28. `.gitmodules` (Modify, experiment branch)
*   **Type / Scope**: Git configuration.
*   **Proposed Changes**: Add `experiments/smolvla-e2e-baseline/softhier-mlir`, URL `https://github.com/runwangdl/softhier-mlir.git`, `branch = main`, `ignore = dirty` (patches are applied in the working tree).

##### 29. `experiments/smolvla-e2e-baseline/softhier-mlir` (Create, submodule)
*   **Type / Scope**: Git submodule (gitlink).
*   **Proposed Changes**: softhier-mlir `main` at the commit current when added. Updating to a newer `main` is a submodule bump commit, recorded in RESULTS.md.

##### 30. `experiments/smolvla-e2e-baseline/pyproject.toml` and `uv.lock` (Create)
*   **Type / Scope**: uv project.
*   **Proposed Changes**:
    *   Python 3.12; softhier-mlir as an editable path dependency (`xdsl>=0.69`); numpy;
    *   CPU torch, transformers and safetensors, which softhier-mlir's `prepare` steps import;
    *   filecheck, lit and pytest;
    *   pyyaml for the configs;
    *   a small package `shvla_exp` (the scripts below) with console scripts `shvla-ladder`, `shvla-watchdog`, `shvla-results`.

##### 31. `experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py` (Create)
*   **Type / Scope**: GVSoC arch file (Python class `FlexClusterArch`).
*   **Proposed Changes**: The output of softhier-mlir's `Arch().to_py()`, with a header comment naming the softhier-mlir permalink it was taken from. The key values: 4×4 clusters, 3 cores, RedMulE 128×32 (pipe 3), NoC 512, HBM placement `(4,0,0,4)`, instruction memory 64 KB.

##### 32. `experiments/smolvla-e2e-baseline/configs/published-c1-t256.yaml` and `configs/libero.yaml` (Create)
*   **Type / Scope**: Run configuration (YAML).
*   **Proposed Changes**: Same schema in both files:
    *   `robot` and `benchmark`;
    *   `checkpoint` and `revision`;
    *   `cameras`: names and size before and after the resize;
    *   `siglip_tokens_per_camera`, `image_tokens_per_camera`, `language_tokens`;
    *   `state_dim` and `action_dim`, with the padded dims;
    *   `prefix_tokens` and `prefix_padded`;
    *   `chunk`, `flow_steps`, `n_action_steps`, `control_hz`;
    *   `observation`: the `golden-ref --obs` string;
    *   `rungs`: the list, each with its softhier-mlir command template, tolerance (max-abs and median), expected simulated ms per stage, and kind (`redmule`, `rowops`, `attention`, `mixed`);
    *   `watchdog`: k, the no-progress interval, and the calibration rung.
    `published-c1-t256` holds the published values (27.3 / 0.28 / 14.8 / 76.7 ms; action error 0.0317). `libero` holds the confirmed table, with composed estimates as expected times, marked `estimate`.

##### 33. `experiments/smolvla-e2e-baseline/scripts/shvla_exp/ladder.py` (Create)
*   **Type / Scope**: Python module, `shvla-ladder` entry point.
*   **Proposed Changes**: `shvla-ladder <config> [--rung N | --all] [--from N]`. For each rung:
    1. Run `stamp.py check` against `arch/softhier_mlir_arch.py`; rebuild on a mismatch (D1).
    2. Apply `patches/*.patch` to the softhier-mlir submodule with `git apply --check` first. Stop and name the failing patch if one does not apply.
    3. Produce the inputs: `golden-ref`, through the infra env, and the converter.
    4. Run softhier-mlir's `tests/gvsoc/run.py` subcommand from the config, with `SOFTHIER_BUILD_DIR` and its log under `$SHVLA_DATA/runs/<run-id>/`, wrapped by the watchdog.
    5. Compare with `compare.py`.
    6. Write a run record (`record.json`: config, rung, permalinks, stamp, patches applied, times per stage, errors, wall time, peak RSS from `/usr/bin/time -v`, watchdog outcome).
    `--all` stops at the first failing rung unless `--continue` is given.

##### 34. `experiments/smolvla-e2e-baseline/scripts/shvla_exp/convert_ref.py` (Create)
*   **Type / Scope**: Python module.
*   **Proposed Changes**: Maps the infra reference schema (file 18) to the npz files softhier-mlir's `prepare` and `prepare-vlm` steps and its `smolvla_e2e prepare --ref` expect. The mapping covers tensor names, KV layout `[S_pad, 320]` with K then V, the padding rows to `S_pad` (multiples of 128), and fp16 conversion where the device reads fp16. The mapping table is written in the module docstring.
    *   Host-side check `--crosscheck <softhier-mlir ref npz>`: compares the converted tensors with softhier-mlir's own `smolvla_expert_ref.py` output on the synthetic observation, and must agree to fp32 rounding. Any device comparison starts only after this check passes.

##### 35. `experiments/smolvla-e2e-baseline/scripts/shvla_exp/watchdog.py` (Create)
*   **Type / Scope**: Python module, `shvla-watchdog` entry point (also used inside the ladder).
*   **Proposed Changes**: `shvla-watchdog --log <file> --budget <stage:ms,...> --ratio <wall s per sim ms> --k 3 --idle <s> -- <command>`.
    *   Starts the command and follows the log.
    *   Parses `[mark]` lines (tag and cycles, unwrapping the 32-bit counter the way `run.py` does), `[sh_preload_wait]`, `[Performance Counter]` and the error texts from `triage_rules.yaml`.
    *   Stops the process group on: over budget (k × expected wall time for the current stage), no new mark for `--idle` seconds, or an error text.
    *   Writes `triage.md`: stage, last mark, simulated time reached, the trigger, the matching side-note entries for the stage's kind, the last 50 log lines. Exit code 3 marks a watchdog stop.
    *   Calibration mode (`--calibrate`) measures the ratio per stage kind on a rung that completes; the ladder stores it under `$SHVLA_DATA/calibration.json`.

##### 36. `experiments/smolvla-e2e-baseline/scripts/shvla_exp/triage_rules.yaml` (Create)
*   **Type / Scope**: Data file.
*   **Proposed Changes**: The side-note table as rules:
    *   `id` (1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, models_fast);
    *   `error_text` patterns, for example "No entry found for burst", "Executing illegal instruction", "Received out-of-bound request", a segfault exit code;
    *   `stage_kinds` where it shows as slowness or wrong values;
    *   `symptom` and `first_check`, the first thing to inspect.
    The watchdog and `compare.py` (for accuracy failures) both read it.

##### 37. `experiments/smolvla-e2e-baseline/scripts/shvla_exp/compare.py` (Create)
*   **Type / Scope**: Python module.
*   **Proposed Changes**: Reads softhier-mlir's sampled dumps or full hand-over dumps for a rung and the matching reference tensors. It reports max-abs, median-abs, max |ref| and relative L2 per tensor, and pass or fail against the config's tolerance. On a failure it adds the matching `triage_rules.yaml` entries for that rung's kind; for example, values about 10 % low across GEMM outputs point to #3.

##### 38. `experiments/smolvla-e2e-baseline/scripts/shvla_exp/results.py` (Create)
*   **Type / Scope**: Python module, `shvla-results` entry point.
*   **Proposed Changes**: Collects the run records into the generated sections of RESULTS.md, between marker comments, so hand-written analysis is kept:
    *   permalinks;
    *   the stamp;
    *   patches applied;
    *   per rung: pass or fail, errors and times;
    *   the G4 table of ours against published or estimated values;
    *   wall time and RSS;
    *   watchdog stops with links to their triage files.
    Labels follow the evidence glossary: simulated values are "measured, simulated on GVSoC, ideal HBM"; composed values are "estimate".

##### 39. `experiments/smolvla-e2e-baseline/patches/0001-paths-from-env.patch` (Create)
*   **Type / Scope**: Patch on softhier-mlir (Q6).
*   **Proposed Changes**: `MODELS`, `VISION_NPZ` and the cache paths in `frontend/smolvla_e2e.py` and `frontend/smolvla.py`, and the default `--ckpt`/`--npz` values in `frontend/smolvla_expert_ref.py` and `tests/gvsoc/run.py`, read `SHVLA_SMOLVLA_DIR`. `SHARED_HOME` in `sim/gvsoc.py` reads `SOFTHIER_HOME`. Each falls back to the original value. The header states what, why and the evidence.

##### 40. `experiments/smolvla-e2e-baseline/patches/README.md` (Create)
*   **Type / Scope**: Documentation (inspection record).
*   **Proposed Changes**: One row per patch: number, files, what, why, evidence (rung and run id), and whether it is a candidate to propose upstream. Further patches (for example 2-camera support in the chain, the 177-token prefix, the LIBERO observation in `prepare`) are added here as G3 and G4 find them.

##### 41. `experiments/smolvla-e2e-baseline/tests/test_watchdog.py` (Create)
*   **Type / Scope**: pytest.
*   **Proposed Changes**: Synthetic logs replayed through the watchdog:
    *   a stage over budget triggers and names the stage;
    *   a silent process triggers no-progress;
    *   each error text triggers its rule;
    *   a normal log finishes with exit 0;
    *   mark parsing handles the 32-bit wraparound;
    *   `triage.md` contains the expected side-note ids.

##### 42. `experiments/smolvla-e2e-baseline/tests/test_convert_ref.py` (Create)
*   **Type / Scope**: pytest.
*   **Proposed Changes**: Small synthetic reference files (random tensors with the schema's names and reduced sizes):
    *   names and shapes map as documented;
    *   padding rows are zero and placed after the valid rows;
    *   fp16 conversion rounds to nearest;
    *   the crosscheck reports a mismatch when one tensor is perturbed.

##### 43. `experiments/smolvla-e2e-baseline/SETUP.md` (Create)
*   **Type / Scope**: Documentation (G4 setup).
*   **Proposed Changes**: From a fresh clone to a G4 run:
    1. clone with `--recursive`, or check out the branch and update submodules;
    2. `infra/softhier/setup.sh`;
    3. `build_arch.sh` with the experiment arch;
    4. `uv sync` in `infra/lerobot` and the experiment;
    5. the `golden-ref` commands per config;
    6. `shvla-ladder` per config;
    7. `shvla-results`.
    Also covered: the measured times and disk use per step, and how to update softhier-mlir to a newer `main`.

##### 44. `experiments/smolvla-e2e-baseline/INPUT_CONFIG.md` (Create)
*   **Type / Scope**: Documentation (robot-dependent input configuration).
*   **Proposed Changes**:
    *   Both configurations side by side, field by field.
    *   For each value: the robot or benchmark it comes from (LIBERO's Franka Panda with agentview and wrist cameras; the published synthetic observation for `smolvla_base`), and the source (checkpoint `config.json` and `train_config.json` permalinks, softhier-mlir docs).
    *   The derived counts (tokens, prefix, padding) with their arithmetic.
    *   What changes for another robot.
    The two "to be checked" items, state 8-D and prefix padding, are resolved here with evidence.

##### 45. `experiments/smolvla-e2e-baseline/RESULTS.md` (Create)
*   **Type / Scope**: Documentation (G5 record, partly generated).
*   **Proposed Changes**:
    *   hand-written sections: summary, deviations and their causes, open issues, the G6 success-rate table next to the G4 latency (two separate numbers, option A);
    *   generated sections from `shvla-results` between markers.

##### 47. `experiments/smolvla-e2e-baseline/LOG.md` (Create)
*   **Type / Scope**: Documentation (step log for manual inspection).
*   **Proposed Changes**:
    *   opens with the step list per goal and its status column;
    *   then one entry per step, in the format given in "Implementation record for manual inspection", newest last;
    *   ends with goal summaries.
    It is written by hand during implementation, not generated, and referenced from SETUP.md and RESULTS.md. Language: mixed Chinese and English, in the same style as the owner's chat replies. Prose is Chinese with English technical terms; commands, paths, identifiers and quoted log lines stay verbatim in English. The other documents (READMEs, SETUP.md, INPUT_CONFIG.md, RESULTS.md, patch headers, code) stay in English.

#### Outside this repo (GVSoC fork, D2)

##### 46. `HaozeG/gvsoc` branch `shvla/soft_hier_release` (Create when needed, owner approval)
*   **Type / Scope**: Fork commits.
*   **Proposed Changes**: First commit: add `soft_hier/flex_cluster_sdk/runtime/include/flex_alloc_api.h`, taken from GVSoC branch `bowwang-dev/softhier-deeploy` (`runtime/deeploy_include/`), with its origin in the commit message. Later commits: one per simulator fix found in G3, each naming its side-note number and the run id that showed it. If a fix touches `core` or `pulp`, the matching fork is created and the fork's `.gitmodules` updated in the same change.

### Verification Plan

*   **Compilation/Build**:
    *   `infra/softhier/setup.sh` completes;
    *   `infra/softhier/build_arch.sh third_party/gvsoc/implementation/config/arch/arch.py` and `build_arch.sh experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py` both complete, each followed by a passing `stamp.py check`;
    *   `uv sync --locked` passes in all three uv projects.
*   **G1**: `infra/softhier/run_kernel_tests.sh` reports pass for `gemm`, `norm`, `acti`, `attn`.
*   **G2**: `golden-ref` for both configurations; `schema.validate` passes; a second run gives the same SHA-256.
*   **Unit Tests**:
    *   `uv run pytest` in `infra/lerobot` (GPU tests on this host) and in the experiment (`test_watchdog.py`, `test_convert_ref.py`);
    *   softhier-mlir's own `bash tests/run_filecheck.sh` with patches applied.
*   **G3 (numerical match)**:
    *   `shvla-ladder configs/published-c1-t256.yaml --all` reaches rung 6;
    *   each rung passes its tolerance, or its failure is recorded with a triage file and either a fix (fork commit or patch) or a known-deviation entry in RESULTS.md;
    *   the converter crosscheck passes before rung 3.
*   **G4 (latency)**:
    *   the published configuration reproduced: total within a stated band of 119.1 ms, with gaps explained;
    *   then `shvla-ladder configs/libero.yaml --all` completes the 4-program chain;
    *   RESULTS.md has the per-stage table and the action error against the LIBERO golden reference.
*   **G5**: `shvla-results` regenerates RESULTS.md from the records without manual edits to generated sections. Every permalink opens on GitHub, after the push.
*   **G6**: `libero-eval` on `libero_spatial`, 10 × 10 episodes, writes `summary.json`; then the 4 suites × 10 × 50, with the success rates entered in RESULTS.md.
*   **Wiki**: `softhier_vla_wiki/tools/wiki.py check` passes with no new warnings after files 22–27.
*   **Traceability**: every commit subject carries a step id; every step id in the step list has a log entry with a commit permalink or a "no commit" note; every goal has a summary in LOG.md.
*   **Static Analysis & Linting**: `ruff check` on `infra/lerobot`, `infra/softhier/stamp.py` and `experiments/*/scripts`; `shellcheck` on the shell scripts.
*   **Test Coverage Requirement**: no percentage target. Every watchdog trigger type and every converter mapping rule has at least one test.

*Drafted with EGM*
