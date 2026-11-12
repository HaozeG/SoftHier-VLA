# SoftHier-VLA stage 1: problem definition and model/compiler explainer

*   **Author:** Haoze (with Claude)
*   **Status:** Proposed (revision 2, after EGM review)
*   **Date:** 2026-11-12
*   **Predecessor:** [`concepts_brief.md`](../../concepts_brief.md) (brainstorm, one Goldfish round)

--------------------------------------------------------------------------------

## Overview

SoftHier-VLA studies how to run robot policy models (vision-language-action models,
VLAs) on a tile-based many-core accelerator. The accelerator is **SoftHier**, a
configurable mesh of compute clusters simulated in **GVSoC** (an event-based
simulator). Each SoftHier cluster has:

*   **Snitch** cores: small RISC-V cores that run control code and scalar or 4-lane
    fp16 SIMD math;
*   a **RedMulE** matrix engine for matrix products;
*   a **Spatz** vector unit for row math such as exp, max and sums;
*   an **iDMA** engine that moves data between HBM and the cluster;
*   **TCDM**, the cluster's local scratchpad memory.

Spatz is part of the default cluster. The first version of the compiler
(softhier-mlir) does not generate Spatz code.

**PR1** (merged as `e832bfb`) built the infrastructure. SmolVLA runs end to end on a
simulated SoftHier chip and matches the published latency: 120.7 ms against 119.1 ms,
for 1 camera and 256 tokens, measured in simulation with an ideal (unlimited) HBM model.
PR1's profile shows where time goes. The project has not yet stated which problem it
is trying to solve.

This design covers the documents that close project stage 1 ("define the problem"). It
has two parts, done in order:

1.  **Problem definition.** The deliverables are:
    *   a problem statement and four research questions (RQs), drafted in this document;
    *   a constants file that holds every shared number with an evidence label;
    *   per-stage model totals checked against PR1's compiled programs;
    *   a hand-written hardware parameter table that sizes SoftHier against NVIDIA
        Jetson Thor;
    *   a research-plan entry point.
2.  **Model and compiler explainer.** A generated table and diagram of SmolVLA's
    operators with compute and memory needs per operator, and a walk-through of how the
    model goes from a checkpoint to kernels on SoftHier's units.

The operator data comes from a **standalone analytical model** in this repo. It reads
the checkpoints' configs and tensor shapes and is checked against PR1's compiled
programs (Technical Plan §4). The compiler walk-through is a document in this repo that
adds to softhier-mlir's own design document.

Two names are used throughout:

*   **Kernel repo:** the owner's SoftHier checkout `~/softhier` (commit `52859c3`, branch
    `softhier_deeploy`). Its `implementation/` folder holds four hand-written kernels:
    attention (FlatAttention), SUMMA matrix product, RMSNorm, and activations. It also
    holds the arch configuration chosen as stage 1's reference.
*   **Model stage:** one of SmolVLA's parts, namely vision, connector, prefix (the VLM
    layers over image, language and state tokens) and expert (the action expert,
    run once per flow step). Model stages are distinct from project stages.

Readers: the owner, collaborators who read Markdown and HTML on GitHub and in meetings,
and the sessions that plan stage 2 from these documents. Model and level terms (the
VLA, VLM trunk, action expert, S0) follow the wiki glossary
(`wiki://glossary/project-terms.md`).

## The Problem

**What is missing today.** The project has a working simulator setup and one measured
end-to-end number. It has no agreed problem statement, no research questions that a
result can answer, and no fixed reference point to compare against. Three concrete gaps
follow:

*   **Numbers do not line up.** The project's figures come from three input
    configurations:
    *   1 camera, 256 tokens (measured);
    *   2 cameras, 1024 vision tokens each (estimated);
    *   3 cameras at 512x512 (estimated).

    They also come from two SoftHier arches that differ about 8x in peak compute:
    PR1's arch (16 clusters, RedMulE 128x32, ~131 TFLOP/s) and the kernel repo's arch
    (16 clusters, RedMulE 32x16, 16.384 TFLOP/s). PR1's arch file disabled Spatz only
    because the compiler cannot use it, so PR1's measured numbers also exclude Spatz's
    effect on the cluster, for example its memory ports. Without one reference
    workload, one reference arch and one definition of "bytes", an RQ starting point
    cannot be compared with a later result.
*   **Colleagues cannot see the model or the compiler path.** The facts are spread
    across softhier-mlir's code and docs, the PR1 profile and the wiki. Some documents
    describe intended behaviour as if it existed: softhier-mlir's README lists Spatz
    `vfexp` for softmax, but the compiler emits no Spatz code.
*   **No comparison point outside the simulator.** The owner chose Jetson Thor, the edge
    device most VLA benchmark papers measure on. Its published 16-bit peaks vary between
    papers (153 measured to 500 TFLOP/s), and NVIDIA publishes no dense 16-bit figure.

**The research problem behind it.** The long-term question is what a chip or module for
deploying embodied AI should look like, and how to map work spatially (which clusters)
and temporally (when) on a tile-based chip. This includes dividing the cluster mesh
between a VLA and a fast controller (S0) that run at different rates on the same chip.
SmolVLA is the first worked example, not the subject. Stage 1 frames this problem with
numbers. The roadmap after stage 1 is: static mapping of one model, then several
models, then dynamic scheduling under different latency and rate needs.

**Proposed solution in one paragraph.** First fix the reference inputs in one constants
file:

*   workload `libero-2cam`;
*   the kernel-repo arch, scaled by cluster count;
*   Thor at 258 TFLOP/s dense FP16 (inference), with 153 TFLOP/s measured as a second
    reference;
*   two byte columns with a written formula per operator class.

Then compute per-stage totals with a standalone model and check them against PR1's
compiled programs. Write the problem statement and RQs against those totals. Then
generate the per-operator table and write the compiler walk-through.

## Goals

*   **G1. Reference inputs are fixed in one place.** `infra/workload-modeling/constants.yaml`
    defines the following, each entry with a value, unit, evidence label, source,
    workload ID and arch:
    *   the workload IDs (`pr1-1cam`, `libero-2cam`, `release-3cam`), with `libero-2cam`
        primary;
    *   the two arches;
    *   the Thor figures;
    *   the PR1 results.

    Every number in the stage-1 documents cites a constants key or states its workload,
    arch and label.
*   **G2. Problem statement and RQs.** The problem statement and RQ1-RQ4 below, each
    RQ in its own file with its metric, arch, starting points and the measurement that
    would answer it. Collaborators add or revise one RQ without touching the others.
*   **G3. Checked per-stage totals.** Per-stage FLOP, unique bytes and weight bytes for
    `libero-2cam` (and `pr1-1cam`) from the standalone model. The matrix-product FLOP
    per stage agrees within 1 % with the RedMulE FLOP recorded for PR1's compiled
    programs at the same padded shapes.
*   **G4. Hardware parameter table sized against Thor.** A hand-written table with two
    columns for the kernel-repo arch: as configured (16 clusters, 16.384 TFLOP/s), and
    scaled by cluster count to Thor's 258 TFLOP/s (16x16 = 256 clusters,
    262.1 TFLOP/s). It is marked as a proposal, not a validated build. Spatz throughput
    is reported separately, not counted in peak, and bandwidth is given as a
    provisional figure.
*   **G5. SmolVLA operator table and diagram.** One row per operator group of SmolVLA
    on `libero-2cam`. The table is generated by the standalone model, rendered to
    Markdown (committed), and converted to HTML (local, for meetings). It is grouped by
    model stage.
*   **G6. Compiler walk-through.** A short document that follows one action-expert layer
    and one SigLIP layer from checkpoint to execution units, and lists what is not
    built.
*   **G7. Entry point.** `docs/research-plan/README.md`, linked from the repo README:
    the problem statement, the RQ index, the roadmap, and links to the explainer
    documents and the wiki status log.

## Stage-1 exit criterion

Stage 1 is complete when all of the following hold:

1.  The owner has approved the problem statement and RQ1-RQ4.
2.  Every RQ names its arch, at least one labelled starting point with its workload and
    arch, and the measurement that would answer it.
3.  The per-stage totals the RQs cite have passed the cross-check (G3).
4.  The wiki status entry "Analyzing the needs of one model: SmolVLA" records these
    results.

No RQ needs to be answered in stage 1.

## Draft problem statement

> Robots that run vision-language-action policies on board need a chip that serves two
> kinds of work at once. One is a large model at a few hertz (vision, a language-model
> prefix, and an action expert run once per flow step). The other is a small controller
> at hundreds of hertz. Today's edge devices (Jetson Thor) are GPU-style chips, and VLA
> benchmarks on them report low utilization at batch size 1: most stages are limited by
> memory bandwidth or by software overheads, not by compute. Tile-based many-core chips
> keep data in distributed local memories and can split their cluster mesh between
> tasks. Whether this suits embodied AI is open. SoftHier-VLA asks how such a chip
> should be built and how work should be mapped onto it in space (which clusters) and
> time (when), at the same peak FP16 compute as Thor. Stage 1 answers this for one
> model, SmolVLA, on one workload (2 cameras, LIBERO). It measures where time goes
> (row operations on the cores, the bandwidth-bound action expert, under-used clusters)
> and states four questions with starting points. Later stages extend the analysis to
> several models and to scheduling a VLA and a controller on one chip.

## Research questions

All four RQs target `libero-2cam`. The arch column says which arch each RQ is asked
on: "16-cluster" means the kernel-repo arch as configured, and "256-cluster" means the
same arch scaled to Thor's peak. No measured starting point exists yet for
`libero-2cam` on either arch. The starting points come from PR1, and each one states
its workload, arch and label. Measured values on the stage-1 arch come from the
measurement work (the next project step). RQ3 is aimed at the core research topic,
spatial mapping.

| ID  | Question                                                                                                                                                                          | Metric                                                     | Arch                                                             | Measurement that answers it                                                                                          |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| RQ1 | What share of the VLA's latency is spent in operations executed on the cores (elementwise, norm, softmax), and how much of it goes away when they run on each cluster's Spatz unit? | share of latency in core-executed ops; latency per stage   | 16-cluster                                                       | hand-written norm, activation and attention kernels (which use Spatz) on `libero-2cam` shapes, against PR1's per-op times |
| RQ2 | For the action expert, how does time per flow step depend on HBM bandwidth and on keeping the KV cache and per-layer weight tiles on chip?                                         | ms per flow step; achieved GB/s                            | 256-cluster, provisional bandwidth 273 GB/s (Thor's), then swept | expert layers with the SUMMA kernel and KV cache kept in TCDM, simulated with a bandwidth-limited HBM model            |
| RQ3 | Given that vision, prefix and expert differ in FLOP per weight byte, is end-to-end latency lower when the mesh is divided into regions per model stage (pipelined across calls) than when each stage uses the whole mesh? | end-to-end latency; cluster utilization                    | 256-cluster (16-cluster for contrast)                            | the four model stages mapped to cluster regions, against whole-mesh execution                                       |
| RQ4 | At iso-peak FP16 with Thor, what fraction of peak does SoftHier reach per model stage, compared with Thor's per-stage figures?                                                    | per-stage MFU (achieved FLOP/s over peak FP16)              | 256-cluster                                                      | per-stage simulated times on the scaled arch                                                                        |

Starting points. Each gives a constants key, the value, its workload, arch and label.
The source is `experiments/smolvla-e2e-baseline/PROFILE.md` on the experiment branch
`worktree-smolvla-e2e-baseline` unless stated otherwise.

*   **RQ1:**
    *   `pr1.core_op_share`: 64.5 %. This is PROFILE's elementwise + attention + norm
        share; attention includes per-head staging and softmax, elementwise includes RoPE
        and bias adds. `pr1-1cam`,
        PR1 arch, measured.
    *   `pr1.libero_softmax_share`: per-head softmax is 43.8 % of the run. `libero-2cam`,
        PR1 arch, estimate (scaled from partial runs).
    *   `pr1.libero_rowop_share`: softmax, GELU and LayerNorm together are ~60 % of the
        run. `libero-2cam`, PR1 arch, estimate.
*   **RQ2:**
    *   `pr1.expert_step_ms`: 7.761 ms per flow step (step 5). `pr1-1cam`, PR1 arch, ideal
        HBM, measured.
    *   `pr1.expert_hbm_read_per_step`: 335 MB of HBM reads per step. `pr1-1cam`, PR1
        arch, ideal HBM, measured.
    *   `pr1.expert_weight_bytes_per_step`: ~193 MB. `pr1-1cam`, estimate (from op
        shapes, PROFILE finding 1).
    *   The expert's FLOP per weight byte from the standalone model: `libero-2cam`,
        inference, cross-checked.
*   **RQ3:**
    *   `pr1.gemm_cluster_use`: vision and prefix matrix products use 1-10 of 16
        clusters. `pr1-1cam`, PR1 arch, measured.
    *   `pr1.e2e_ms`: 120.7 ms. `pr1-1cam`, PR1 arch, measured.
    *   `pr1.libero_e2e_ms`: 479.1 ms. `libero-2cam`, PR1 arch, estimate.
*   **RQ4:**
    *   `pr1.mfu`: 1.2 % of 131 TFLOP/s. `pr1-1cam`, PR1 arch, measured.
    *   Thor per-stage figures from `wiki://knowledge/mfu-from-measurements.md`. These
        are π0.5 and GR00T runs with FP8/NVFP4 kernels, not SmolVLA at FP16. They are
        converted by dividing each stage's achieved FLOP/s by 258 TFLOP/s, labelled
        inference. The page states that FP8/NVFP4 kernels reach more FLOP/s than FP16
        kernels would, so the converted figures are upper bounds for an FP16 Thor run.

## Non-Goals

*   **No new simulator runs.** The following belong to the measurement work:
    *   hand-written kernels on SmolVLA shapes;
    *   measured values on the stage-1 arch;
    *   a bandwidth-limited HBM model;
    *   fusion estimates.
*   **No compiler changes.** No Spatz lowering in softhier-mlir, no linalg path work.
*   **No validated Thor-sized build.** The scaled arch is a parameter table, not checked
    against TCDM or instruction-memory limits.
*   **No S0 model and no multi-model workload.** VLA/S0 co-location on one chip is on
    the roadmap only.
*   **No full-model weight residency option.** SmolVLA's fp16 weights (the action expert
    alone ~196 MB) exceed the 256-cluster mesh's ~96 MiB of TCDM.
*   **No wiki structure changes.** The status and patch logs, their wiki-design decision
    and the glossary rows were added by a separate wiki session and merged (wiki
    `d74b029`). This work adds one status entry through the normal wiki workflow, as a
    follow-up commit after each pull request merges.
*   **No collaborator sync tooling** beyond GitHub pull requests and HTML built from
    Markdown.

## Risks & Mitigations

| Risk                                                                                                                          | Effect                                                         | Mitigation                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| The Thor dense FP16 figure (258, VLA-HWBench's spec-derived value, FP8 517 / 2) is wrong, or assumes a power mode a robot would not use | the scaled cluster count and every RQ4 ratio shift             | constants hold 153 measured next to 258; the hardware table shows both; one constants edit changes the cluster count                               |
| The 256-cluster arch cannot be built (TCDM banking with Spatz ports, 64 KiB instruction memory, mesh size)                   | RQ2-RQ4 are asked on a configuration the simulator cannot run  | the table is marked "proposal"; the measurement work starts with a build check of the scaled arch, and RQs fall back to the 16-cluster arch if it fails |
| A starting point's arch differs from its RQ's arch (all starting points are PR1-arch, the RQs are stage-1-arch)               | readers compare numbers from different machines                | every starting point states its arch and label; the RQ index has an arch column                                                                    |
| The standalone model drifts from what softhier-mlir emits (fusion, padding, new operators)                                   | operator table and RQ totals disagree with simulated runs      | the cross-check against PR1's programs runs in Part 1 and its per-stage numbers are asserted in the tests                                          |
| The provisional bandwidth (273 GB/s) is not what a SoftHier HBM would deliver                                                 | RQ2's starting framing is off                                  | labelled provisional and inference; RQ2's method is a bandwidth sweep, not one value                                                               |

## Open Questions

*   **Thor power mode.** Whether the spec peaks VLA-HWBench uses (FP8 517, and so FP16
    258) assume the MAXN power mode (the unrestricted power setting) is not
    confirmed. The measured 153-155.5 TFLOP/s were taken at 120 W in one forum report.

## Critical User Journeys (CUJs)

### CUJ 1: Discuss SmolVLA's compute needs in a meeting

*   **Given**: the operator table is generated for `libero-2cam` and the HTML page is
    built locally.
*   **When**: the owner opens the HTML page, or the Markdown on GitHub, and walks
    through vision, connector, prefix and expert.
*   **Then**: for each model stage and operator class, the page shows FLOP, unique
    bytes, weight bytes, FLOP per weight byte, the unit used today, and whether a
    hand-written kernel exists. The stage totals match `stage_totals()` and the
    cross-checked numbers.

### CUJ 2: A collaborator proposes a new research question

*   **Given**: a collaborator has cloned the repo and read `docs/research-plan/README.md`.
*   **When**: they copy `rq/_template.md` to a new file, fill in metric, arch and
    starting points (constants keys or labelled values with workload and arch), add a
    row to the hand-written RQ index, and open a pull request.
*   **Then**: no other RQ changes, and the reviewer can check every cited key against
    `constants.yaml`.

### CUJ 3: A new measurement replaces an estimate

*   **Given**: the measurement work produces a measured value on the stage-1 arch.
*   **When**: the owner adds it to `constants.yaml` with its label and source, and
    regenerates.
*   **Then**: generated pages pick up the new value. Git and the wiki status entry
    (`superseded_by:`) record which value it replaced.

## Technical Plan (Architecture)

### 1. Layout and data flow

Code lives in `infra/workload-modeling/`, a uv project next to `infra/lerobot/`. This
follows the README rule that reusable parts live under `infra/` on `main`. Prose lives
in `docs/analysis/` and `docs/research-plan/`. Generated files are whole files, not
blocks inside hand-written pages:

*   Markdown and CSV are committed, so GitHub shows them;
*   HTML is not committed.

The generator exists for this stage's operator table and for the measurement work and
the later several-models step, where numbers change and must stay reproducible.

```text
 HF Hub (3 pinned revisions:            constants.yaml
 policy config, backbone config,        (workloads, arches, Thor, PR1 results,
 safetensors headers)                    each with label, source, workload, arch)
        | fixtures refresh (online, rare)        |
        v                                        v
 tests/fixtures/  -------------------->  smolvla_ops.py (standalone model)
                                                 |
                      +--------------------------+--------------------------+
                      v                          v                          v
            build.py: stage-totals.md     build.py: smolvla-op-graph   crosscheck.py (Part 1, once)
            + stage-totals.csv            .md/.csv (Part 2)            vs PR1 kept programs
            (docs/analysis/)                     |                     -> crosscheck.md
                                                 v md_to_html.py       + crosscheck.json
                                          build/smolvla-op-graph.html  (docs/analysis/)
                                          (local, not committed)

 Hand-written, citing constants keys: conventions.md, hw-config.md,
 research-plan/README.md (RQ index), rq/*.md, compiler-walkthrough.md
```

### 2. Constants file

`infra/workload-modeling/constants.yaml` holds every number that appears in more than
one document.

*   **Key path:** `group.name`, for example `pr1.expert_step_ms`, `thor.fp16_dense`,
    `arch.stage1.redmule_h`.
*   **Entry fields:** `value`, `unit`, `label`, `source`, and optionally `workload`,
    `arch` and `note`.
*   **Labels:** `measured`, `estimate` and `inference` only (the wiki's set,
    `wiki.py:88`, copied rather than imported), so rows copy into wiki status entries
    unchanged. A community figure that was not independently checked keeps its
    reported label and says "self-reported" (wiki glossary term) in `note`.
*   **Derived numbers:** a figure computed from a source's number is `inference` with
    the computation in `note`. Figures computed from arch parameters (peak per cluster,
    totals) are not stored; `constants.py` computes them.

Workload records are structured entries (not single values) with a `source` and
`label`. Excerpt:

```yaml
workloads:
  libero-2cam:
    checkpoint: lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d
    cameras: 2
    image_size: 512            # per camera, square
    vision_tokens_per_camera: 1024
    language_tokens: 48        # tokenizer padding length (wiki://resources/models/smolvla.md)
    state_tokens: 1
    chunk: 50
    flow_steps: 10
    label: measured            # the configuration PR1's G4a runs used
    source: docs/designs/smolvla-softhier-baseline.md@e832bfb
arch:
  stage1:
    repo: ~/softhier@52859c3
    file: implementation/config/arch/arch.py
    clusters_x: {value: 4, unit: "", label: measured, source: "arch.py:24"}
    redmule_h: {value: 32, unit: CE, label: measured, source: "arch.py:50"}
    redmule_w: {value: 16, unit: CE, label: measured, source: "arch.py:51"}
    tcdm_bytes: {value: 393216, unit: B, label: measured, source: "arch.py:32"}
    noc_link_bits: {value: 1024, unit: bit, label: measured, source: "arch.py:72"}
    clock_hz: {value: 1.0e9, unit: Hz, label: measured, source: "~/softhier@52859c3 pulp/pulp/chips/flex_cluster/flex_cluster.py:338"}
thor:
  fp16_dense: {value: 258, unit: TFLOP/s, label: inference, source: "wiki://resources/serving/vla-hwbench-edge-latency.md (VLA-HWBench Table 3: FP16 258, FP8 517)", note: "spec-derived by the paper; NVIDIA publishes no dense FP16 figure"}
  fp16_measured: {value: 153, unit: TFLOP/s, label: measured, source: "wiki://knowledge/mfu-from-measurements.md (Chalmers 2611.06710)"}
  fp16_forum: {value: 155.5, unit: TFLOP/s, label: measured, source: "NVIDIA developer forum t/344399", note: "self-reported, 120 W"}
  dram_bw: {value: 273, unit: GB/s, label: measured, source: "NVIDIA Jetson Thor developer blog"}
pr1:
  expert_step_ms: {value: 7.761, unit: ms, label: measured, workload: pr1-1cam, arch: pr1, source: "PROFILE.md@<experiment-branch commit>, Expert step anatomy"}
```

The implementation confirms `language_tokens` against the `smolvla_libero` policy
config and the kept LIBERO programs. `arch.pr1` copies PR1's parameters from
`experiments/smolvla-e2e-baseline/profile/hw_spec.csv`, which already gives a source
per row, at the experiment-branch commit.

Thor footnote in the constants file: 500 TFLOP/s (Google/Purdue) refers to DRIVE Thor,
not Jetson Thor; 400 (VLA-Perf) has no stated method. Neither is used. 517 is Thor's FP8
peak (VLA-HWBench Table 3) and is kept as `thor.fp8_dense` for the MFU rows.

### 3. Standalone SmolVLA model (`smolvla_ops.py`)

**Dimension sources.** Every dimension comes from one of three fixtures at a pinned
revision, and the model records which:

| Source                                                                                                              | Dimensions taken from it                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| policy `config.json` (`lerobot/smolvla_base@5e8d12a6…`, `lerobot/smolvla_libero@a24f1f38…`)                        | number of VLM layers kept (`num_vlm_layers`), chunk length, flow steps, expert width multiplier, state dimension, image resize                                                                        |
| backbone `config.json` (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b…`)                                     | text hidden size, attention heads, KV heads, `head_dim` (as `golden_ref.py:330` reads it), text MLP size, vision hidden size, patch size, pixel-shuffle scale factor                                 |
| safetensors header of the policy checkpoint                                                                         | vision layer count and vision MLP size (class defaults that the backbone config does not state, `wiki://resources/models/smolvla.md`), expert layer shapes, connector, state and action projections |

The model checks every config-derived shape against the header tensor of the same name
and fails on a mismatch. Stored dtypes are ignored: the checkpoint holds BF16 and some
FP32 tensors (cross-attention K/V), while all byte columns assume fp16 (2 B), as the
SoftHier library does. The wiki's per-part parameter table
(`wiki://resources/models/smolvla.md`) is the test oracle for parameter counts.

**Rows.** `build(workload_id) -> list[OpRow]`, one row per operator group:

| Column           | Meaning                                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------------------- |
| `stage`          | vision, connector, prefix, expert                                                                           |
| `layers`         | layer index or range (`0-11`)                                                                               |
| `op`             | operator name (`q_proj`, `qk`, `softmax`, `pv`, `o_proj`, `mlp_up`, `gelu`, `layernorm`, `rmsnorm`, `rope`) |
| `class`          | gemm, attn, norm, acti, softmax, elementwise, data (transpose, pixel shuffle)                               |
| `shape`          | (M, N, K) for matrix products; (h, Sq, Skv, d) for attention; (rows, cols) otherwise                        |
| `count`          | instances per call = layers x flow steps that apply (x cameras for vision attention only)                   |
| `flop`           | per call, all instances                                                                                     |
| `bytes_unique`   | weights + inputs + outputs, each counted once per instance as defined below, fp16                          |
| `bytes_weight`   | weight bytes per call as defined below                                                                      |
| `flop_per_wbyte` | `flop / bytes_weight`; empty for operators without weights                                                  |
| `unit_today`     | unit in PR1's compiled path (§8 table)                                                                      |
| `hand_kernel`    | kernel-repo kernel that covers it, or none (§8 table)                                                       |

**How bytes scale with `count`.** This rule fixes the expert's intensity, which would
otherwise be off by about 10x:

*   *Layers:* each layer has its own weights, so `bytes_weight` scales with the number
    of layers.
*   *Heads and batched cameras:* they share one weight matrix and are folded into the
    shape, so `bytes_weight` does not scale with them. Vision cameras are batched into
    the token rows of every per-token operator; attention runs per camera.
*   *Flow steps:* the expert's weights do not fit on chip, so they are read again in
    every step. `bytes_weight` scales with flow steps. The page states this assumption.
*   `bytes_unique` counts inputs, outputs and weights per instance, times `count`.

**FLOP and byte formulas per class** (fp16, 2 B per element; `h` = heads, `hkv` = KV
heads, `d` = head dim, `Sq`/`Skv` = query/key length):

| Class                        | FLOP                                          | `bytes_weight`        | `bytes_unique`                                                               |
| ---------------------------- | --------------------------------------------- | --------------------- | ---------------------------------------------------------------------------- |
| gemm (M x K) . (K x N)       | 2 M N K                                       | 2 K N                 | 2 (M K + M N) + `bytes_weight`                                               |
| attn: QK^T                   | 2 h Sq Skv d                                  | 0                     | 2 (h Sq d + hkv Skv d) (score matrix not counted: intermediate)             |
| attn: P.V                    | 2 h Sq Skv d                                  | 0                     | 2 (hkv Skv d + h Sq d)                                                       |
| softmax (rows x cols)        | 5 per element (max, subtract, exp, sum, divide) | 0                   | 2 x 2 rows cols                                                              |
| norm: LayerNorm (rows x n)   | 8 per element                                 | 2 x 2 n (scale, bias) | 2 x 2 rows n + `bytes_weight`                                                |
| norm: RMSNorm (rows x n)     | 4 per element                                 | 2 n (scale)           | 2 x 2 rows n + `bytes_weight`                                                |
| acti: GELU (tanh form)       | 8 per element                                 | 0                     | 2 x 2 elements                                                               |
| acti: SiLU x up (expert MLP) | 5 per element                                 | 0                     | 2 x 3 elements                                                               |
| elementwise: RoPE            | 3 per element                                 | 0                     | 2 x 2 elements                                                               |
| elementwise: residual add    | 1 per element                                 | 0                     | 2 x 3 elements                                                               |
| elementwise: bias add        | 1 per element                                 | 2 N                   | 2 x 2 elements + `bytes_weight`                                              |

Bias adds are separate rows, as in PR1's programs, so matrix-product FLOP compare
directly with RedMulE work. `docs/analysis/conventions.md` holds the full table (also
SiLU, scale, Euler update, data movement).

The per-element FLOP counts for non-matrix classes are conventions (labelled inference
on the page). They are small next to matrix products. The cost of these operators on
SoftHier comes from core issue rate, not from FLOP, which RQ1 measures.

**Worked example (expert MLP gate/up projection, `pr1-1cam`).**

*   Shape per instance: M = 50 action tokens, K = 720 (expert width), N = 2 x 2048
    (gate and up).
*   Per instance: FLOP = 2 x 50 x 720 x 4096 = 294.9 MFLOP, and weight bytes =
    2 x 720 x 4096 = 5.90 MB.
*   Count: 16 layers x 10 steps = 160 instances (PROFILE lists 160).
*   Totals: `flop` = 47.2 GFLOP and `bytes_weight` = 5.90 MB x 16 x 10 = 943 MB.
*   `flop_per_wbyte` = 50, which equals M, as expected for an M-row product. If weight
    bytes did not scale with steps, the figure would read 500.

The expert width 720 and MLP size 2048 come from the header. The example's numbers are
checked in the tests.

**`stage_totals(rows)`** returns, per model stage: FLOP, matrix-product FLOP,
`bytes_unique`, `bytes_weight` and FLOP per weight byte. Part 1 needs only these.

**Reference computation and `Lowering`.** `build` follows the reference code (lerobot
SmolVLA) at true token counts. A `Lowering` records a compiled program's shape choices
and rewrites. PR1's (`pr1_lowering`) has four, all found while implementing the
cross-check:

*   prefix rows padded for RedMulE (65 to 128, 177 to 256), read from the prefix program;
*   expert attention key length padded to a multiple of 32 inside the runtime
    (softhier-mlir `b11cd5f` `runtime/sh_expert.inc.c:83`), not visible in the program;
*   the cross-attention K/V projection of the prefix done once per call (the reference
    code repeats it every step);
*   the time-MLP input projection split: the time half is one row, the same for all chunk
    tokens, and is folded into a per-step bias table on the host
    (`frontend/smolvla_expert.py:27-28`).

### 4. Cross-check against PR1's compiled programs (Part 1)

The owner chose option (a) of the review: compare against the programs PR1 actually
ran, rather than running softhier-mlir's extractor. The extractor cannot parse these
programs (`arith`, `scf.for`, fused attention).

*   **Inputs:**
    *   For `pr1-1cam`: the kept PR1 programs `vision.mlir`, `prefix_a.mlir`,
        `prefix_b.mlir`, `expert.mlir` and their `ops.csv` in
        `~/.cache/softhier-vla/profile/published-fixi-rung6/`.
    *   For `libero-2cam`: the partial programs in
        `~/.cache/softhier-vla/profile/libero-G4a/` and the scaled
        `libero_ops_scaled.csv`. The partial programs are G4a.1 (vision at 1024 tokens,
        1 layer), G4a.2 (prefix, 8 layers) and G4a.3 (expert, 2 steps).
    *   The parser is PR1's `parse_profile.py` (standard library only; its
        `parse_mlir` and `flatten` expand `scf.for` and label `cross_attention`). It is
        loaded from a path given on the command line, at the experiment-branch commit.
*   **Shapes from the programs:** the script reads vision rows, the padded prefix rows
    and the true prefix token count (cross-attention key rows) from the flattened
    programs, checks the token counts against the model, and evaluates the model three
    times: with `pr1_lowering` ("padded"), with PR1's rewrites at true counts
    ("unpadded"), and as the reference computation.
*   **Fused attention:** each head's QK^T and P.V count as 2 h Sq Skv d each, which
    matches the two RedMulE operations per head that `ops.csv` records.
*   **Pass rule:** for each model stage, the padded model's matrix-product FLOP must be
    within 1 % of the sum of `redmule_flops` in `ops.csv` (G4a runs scaled with PR1's
    recipe for `libero-2cam`). The script exits non-zero otherwise.
*   **Result (run 2026-11-12):** PASS. Every stage of both workloads is within
    0.011 %; the residual is the one-row time half that PR1 runs on the host. Padding
    adds 99 % to PR1's prefix matrix work for `pr1-1cam` and 47 % for `libero-2cam`.
*   **Outputs:**
    *   `docs/analysis/crosscheck.md`: a table per workload of stage, model FLOP padded,
        unpadded and reference, PR1 RedMulE FLOP, difference, and padding overhead;
    *   `docs/analysis/crosscheck.json`: the same numbers, which `test_smolvla_ops.py`
        asserts so later model changes cannot drift silently.

    The `libero-2cam` comparison is labelled estimate because its reference is
    scaled.

The kept programs live outside git. The committed outputs mean collaborators do not need
them; only re-running the cross-check does.

### 5. Rendering (Part 2)

*   `render.py` writes `docs/analysis/smolvla-op-graph.md` and `.csv` from one Jinja
    template (`infra/workload-modeling/workload_modeling/templates/op_graph.md.j2`).
    The page contains:
    *   a Mermaid diagram of the four model stages (GitHub renders Mermaid), each box
        labelled with its FLOP share and FLOP per weight byte;
    *   one table per model stage, with identical layers collapsed;
    *   a totals table;
    *   a limitations paragraph: no tiling re-reads; no intensity for weightless
        operators; weights re-read per flow step.
*   `md_to_html.py` converts that Markdown to
    `infra/workload-modeling/build/smolvla-op-graph.html` (git-ignored):
    *   Python-Markdown with the tables and fenced-code extensions;
    *   a small extension that turns ```` ```mermaid ```` fences into
        `<pre class="mermaid">`;
    *   inline CSS with light and dark themes through `prefers-color-scheme`;
    *   `mermaid@10.9.1` from jsDelivr, the page's only external URL. Tables read
        offline; the diagram needs the network.

### 6. Hardware parameter table (`docs/analysis/hw-config.md`, hand-written)

Each row gives its value, its formula or source, and its label. All inputs are constants
keys.

| Row                         | 16-cluster (as configured)                                    | 256-cluster (Thor-matched)                                         | Formula / source                                                                                    |
| --------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| clusters, mesh              | 16, 4 x 4                                                     | 256, 16 x 16                                                       | smallest n with n^2 x peak per cluster >= `thor.fp16_dense` (15 x 15 gives 230 TFLOP/s)             |
| clock                       | 1 GHz                                                         | 1 GHz                                                              | `arch.stage1.clock_hz` (simulator config, not in `arch.py`)                                         |
| RedMulE peak FP16           | 16.384 TFLOP/s                                                | 262.1 TFLOP/s                                                      | clusters x 32 x 16 MAC x 2 x clock                                                                  |
| Spatz throughput (not in peak) | per formula                                                | per formula                                                        | clusters x cores with Spatz (4) x function units (`spatz_num_function_unit` = 4) x lanes x 2 x clock; lanes read from the simulator's Spatz model at `52859c3`, or the row says "not determined" |
| TCDM per cluster, total     | 384 KiB, 6 MiB                                                | 384 KiB, 96 MiB                                                    | `arch.stage1.tcdm_bytes` x clusters                                                                 |
| HBM capacity                | as configured in `arch.py` (channels x node size)            | next power of two >= fp16 weights + largest activation of `libero-2cam` | model `stage_totals()`; expected 1 GiB                                                     |
| NoC link bandwidth          | 128 GB/s per direction                                        | 128 GB/s per direction                                             | `noc_link_bits` x clock / 8                                                                         |
| HBM bandwidth               | edge ports x 128 GB/s with ideal HBM (as PR1 ran)            | provisional 273 GB/s (Thor's), inference                           | open parameter; RQ2 sweeps it                                                                       |

The page also states:

*   why the kernel-repo arch is the reference: it is the arch the hand-written kernels
    target, and Spatz is enabled in it;
*   how it differs from PR1's arch: RedMulE 32x16 instead of 128x32, 5 cores per
    cluster instead of 3, TCDM 384 KiB instead of 1 MiB, NoC link 1024 bit instead of
    512;
*   that the 256-cluster column is a proposal that has not been built;
*   that PR1's numbers were measured with ideal HBM.

### 7. Research plan and conventions

*   **`docs/research-plan/README.md`** (hand-written) holds:
    *   the problem statement;
    *   the exit criterion;
    *   the roadmap;
    *   a reading order: conventions, hardware table, stage totals and cross-check,
        operator graph, compiler walk-through, wiki status log;
    *   the RQ index, a table with id, title, metric, arch, starting points and status.
*   **`docs/research-plan/rq/rqN-<slug>.md`**: one file per RQ.
    *   Front-matter:
        ```yaml
        id: RQ2
        title: Expert time per flow step against bandwidth and on-chip residency
        status: draft        # draft | active | answered | dropped
        workload: libero-2cam
        arch: stage1-256     # stage1-16 | stage1-256
        metric: ms per flow step; achieved GB/s
        depends_on: []       # list of RQ ids
        ```
    *   Body sections: Question, Why it matters, Starting points (constants keys or
        labelled values with workload and arch), Method, What result would answer it.
    *   `rq/_template.md` holds the same skeleton.
*   **`docs/research-plan/working/`** holds idea-forming material that is not a
    decision. The owner's pattern tables move here in a separate commit.
*   **`docs/analysis/conventions.md`** holds:
    *   the workload IDs and evidence labels;
    *   the byte columns, the count-scaling rule and the formula table from §3;
    *   the Thor figures and the alignment rule (iso-peak FP16 by cluster count; Spatz
        not counted; bandwidth provisional);
    *   the rule that every number cites a constants key or states workload, arch and
        label.

### 8. Compiler walk-through (`docs/analysis/compiler-walkthrough.md`)

The walk-through is a delta on `~/softhier-mlir/docs/DESIGN.md`, linked at the commit
PR1 pinned (`b11cd5f`). It follows one action-expert layer and one SigLIP layer through:

1.  weights loaded from safetensors by softhier-mlir's prepare step and
    `sim/preload.py` into simulated HBM;
2.  the Python/xDSL frontend building softhier-dialect IR directly, with no linalg;
3.  `emit_c.py` turning each softhier op into one `sh_*` runtime call;
4.  the call running on the cluster's units.

Operator class to unit and hand-written kernel:

| Operator                                  | Unit in PR1's compiled path                  | Kernel-repo hand-written kernel (`~/softhier@52859c3`, `implementation/sw/`) |
| ----------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| matrix products (projections, MLP)        | RedMulE, operands moved by iDMA              | `SummaGEMM/` (`gemm`)                                                         |
| attention QK^T and P.V                    | RedMulE                                      | `FlatAttention/` (`attn`), softmax included                                   |
| softmax (vision per head; inside fused attention elsewhere) | Snitch cores, 4-lane fp16 SIMD | inside `FlatAttention/` only                                              |
| RMSNorm                                   | Snitch cores                                 | `RMSNorm/` (`norm`)                                                           |
| LayerNorm                                 | Snitch cores                                 | none                                                                          |
| GELU                                      | Snitch cores                                 | none (`Activation/` covers Sigmoid, ReLU, SiLU)                               |
| SiLU x up                                 | Snitch cores                                 | `Activation/` covers SiLU; whether it includes the multiply is checked in Part 2 |
| RoPE, bias add, residual add              | Snitch cores                                 | none                                                                          |
| transpose, pixel shuffle                  | Snitch cores and iDMA                        | none                                                                          |

Spatz is unused in PR1's path because the compiler does not target it, so PR1's arch file
disabled it. The hand-written kernels use it for row math. The last section lists what
is not built, each item with a code reference:

*   Spatz lowering (`softhier.vexp` is declared but not emitted);
*   the linalg path (three passes, not on the SmolVLA path);
*   the operators with no hand-written kernel (table above).

### 9. Wiki status entry

After each pull request merges, a follow-up commit adds to or extends the status entry
"Analyzing the needs of one model: SmolVLA" through the wiki's normal workflow:

*   a wiki branch;
*   the `status` note type;
*   `wiki.py check` passing;
*   a submodule bump.

Links are pinned to the merge commit on `main`, as the existing entry pins `e832bfb`.
The front-matter carries `follows: [basic-integration-of-components.md]`, `evidence:`
(labels used) and `sources:` (the repo URL at the pinned commit).

*   Part 1 writes the entry with: reference inputs, stage totals and cross-check, the
    hardware table, and the RQs.
*   Part 2 adds rows for the operator table and the walk-through.

## Alternatives Considered

### Alternative A: One large pull request with every stage-1 document

*   **Pros**: one review.
*   **Cons**: too large to review well, and it mixes code, docs and wiki changes.
*   **Reason for Rejection**: two parts, wiki changes in their own session (done), and
    measurement work later.

### Alternative B: Iso-peak FLOPS as the only alignment with Thor, bandwidth matched implicitly

*   **Pros**: one knob, common in architecture papers.
*   **Cons**: the expert is bandwidth-bound at batch size 1, so inheriting SoftHier's
    bandwidth silently would favour SoftHier.
*   **Reason for Rejection (partial)**: iso-peak FP16 by cluster count is kept. On Thor,
    the standalone model gives `libero-2cam` vision 2,489 FLOP per weight byte, above
    Thor's balance of 258 TFLOP/s / 273 GB/s = 945 FLOP/B, while the prefix (183) and the
    expert (55) are below it (inference, cross-checked; the brainstorm's "about 2000 for
    vision and prefix" overstated the prefix). So only vision is compute-bound on Thor;
    the claim is limited to Thor. Bandwidth is a provisional, labelled parameter, never matched
    silently.

### Alternative C: Wiki process alternatives (amend 0013 here; date-ordered status log; patch log mirroring the README)

Settled by the wiki decision `wiki://wiki-design/decisions/2026-11-12-status-and-patch-logs.md`
(merged), which records each alternative and its rejection.

### Alternative D: Research plan inside the wiki, or as its own submodule

*   **Pros**: one knowledge base, or independent versioning.
*   **Cons**: the wiki is lint-gated and limited to what sources say; a third repository
    adds a sync point before a second author exists.
*   **Reason for Rejection**: `docs/research-plan/` in the code repo, linking to the wiki.

### Alternative E: Hand-typed operator tables, or RQs inside the PR1 design doc

*   **Pros**: no generator.
*   **Cons**: not reproducible; the PR1 design is a fixed record and the RQs outlive it.
*   **Reason for Rejection**: generated operator table; RQs as separate files.

### Alternative F: Drop the hardware configuration from stage 1

*   **Pros**: less work.
*   **Cons**: RQs need a fixed comparison arch.
*   **Reason for Rejection**: a hand-written parameter table stays, marked as a proposal.

### Alternative G: Full-model weight residency as a fusion option

*   **Pros**: removes weight streaming.
*   **Cons**: the expert alone is ~196 MB against ~96 MiB of TCDM on 256 clusters.
*   **Reason for Rejection**: only the KV cache and per-layer tiles are candidates.

### Alternative H: Operator data from softhier-mlir (as a submodule, or a tool inside softhier-mlir)

*   **Pros**: numbers match the compiled IR.
*   **Cons**: adds softhier-mlir and xDSL as a dependency on `main`, or requires changes
    in a repo this project does not own. The extractor cannot read the SmolVLA programs
    today.
*   **Reason for Rejection**: standalone model, cross-checked against PR1's programs.

### Alternative I: Byte conventions (unique only; unique plus library tiling traffic)

*   **Unique only**: no figure comparable to published Thor data.
*   **Plus tiling traffic**: shows re-reads, but needs a tiling choice and does not
    describe the hand-written kernels.
*   **Reason for Rejection**: unique bytes plus weight bytes, with FLOP per weight byte.
    Tiling traffic is left to the measurement work.

### Alternative J: Cross-check options not chosen

*   **Extend softhier-mlir's extractor**: changes a repo this project does not own.
*   **Compare against PROFILE's per-phase GFLOP only**: covers `pr1-1cam` only.
*   **Drop the cross-check**: leaves the RQ totals unchecked.
*   **Reason for Rejection**: the owner chose PR1's kept programs with `parse_profile.py`.

### Alternative K: Generated hardware table and RQ index (with `--check` drift detection)

*   **Pros**: numbers in the two pages cannot drift from the constants.
*   **Cons**: two generators and a drift check for two small pages that change rarely.
*   **Reason for Rejection**: hand-written pages citing constants keys. Drift is caught
    by `build && git diff --exit-code` for the generated files only.

### Alternative L: Interactive HTML rendered from the operator rows

*   **Pros**: per-operator detail on hover; works offline.
*   **Cons**: two templates to maintain.
*   **Reason for Rejection**: HTML converted from the committed Markdown.

### Alternative M: Different scope for this design

*   **Problem definition only**: totals for the RQs would come from a later design.
*   **Also the measurement work**: its details are likely to change after the first runs.
*   **Reason for Rejection**: problem definition plus explainer, measurement only
    referenced.

## Non-Functional Requirements

*   **Reproducibility:** the build runs offline from committed fixtures. Network access
    is needed only for `fixtures refresh` and to draw the HTML diagram.
*   **Collaborator setup:** `uv sync` in `infra/workload-modeling/` and one build command.
    The cross-check needs the kept PR1 programs and the experiment branch, but its
    results are committed.
*   **Evidence discipline:** `constants.py` rejects entries without `label` or `source`,
    or with a label outside the allowed set.

## Additional Resources

*   `concepts_brief.md`: brainstorm brief with owner decisions.
*   `docs/designs/smolvla-softhier-baseline.md`: PR1 design, findings F1-F19.
*   `experiments/smolvla-e2e-baseline/PROFILE.md` and `profile/parse_profile.py`,
    `profile/hw_spec.csv` on the experiment branch.
*   `~/softhier-mlir/docs/DESIGN.md`; `~/softhier@52859c3` `implementation/`.
*   Wiki: `knowledge/mfu-from-measurements.md`, `resources/models/smolvla.md`,
    `resources/serving/vla-hwbench-edge-latency.md`, `memories/status/`, glossary.

## Future Improvements

*   **Tiling-traffic column** once the measurement work calibrates the hand-written
    kernels.
*   **Other models:** builders for π0/π0.5 and GR00T for the several-models step.
*   **S0 stand-in** when co-location becomes an RQ.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Branches are named by content. Part 1 is `problem-definition`, branched from
`worktree-stage1-brainstorm`, so this design and `concepts_brief.md` reach `main` with
it. Part 2 is `model-and-compiler-explainer`, branched from Part 1. Each branch opens
one pull request to `main`. Commits use the repo's `[step] area: message` style.

### Target Files to Create/Modify

#### Part 1: problem definition (branch `problem-definition`)

##### 1. `design_draft.md` (Deleted)

*   **Type / Scope**: working file.
*   **Proposed Changes**: removed with this revision on `worktree-stage1-brainstorm`;
    this design file is canonical.

##### 2. `infra/workload-modeling/pyproject.toml` (Create)

*   **Type / Scope**: uv project configuration.
*   **Proposed Changes**: `name = "workload-modeling"`, `requires-python = ">=3.12,<3.13"`,
    a hatchling build backend, and `[tool.uv] package = true`, so
    `python -m workload_modeling.<cmd>` works. Dependencies: `pyyaml`,
    `huggingface_hub` (fixture refresh only), `jinja2` and `markdown` (Part 2). Dev
    dependency: `pytest`.

##### 3. `infra/workload-modeling/uv.lock` (Create, generated by `uv lock`)

*   **Type / Scope**: lock file.
*   **Proposed Changes**: pins the environment.

##### 4. `infra/workload-modeling/README.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: what it generates, the three commands (`build`,
    `fixtures refresh`, `crosscheck`), and the label rule.

##### 5. `infra/workload-modeling/constants.yaml` (Create)

*   **Type / Scope**: data.
*   **Proposed Changes**: as in Technical Plan §2.
    *   Groups: `workloads`, `arch.stage1`, `arch.pr1`, `thor`, `pr1`.
    *   `pr1` keys: `core_op_share`, `libero_softmax_share`, `libero_rowop_share`,
        `expert_step_ms`, `expert_hbm_read_per_step`, `expert_weight_bytes_per_step`,
        `gemm_cluster_use`, `e2e_ms`, `libero_e2e_ms`, `mfu`. Each has `workload`, `arch`,
        and a source at the experiment-branch commit.
    *   `release-3cam` stays as a one-entry workload with a note that no RQ uses it.

##### 6. `infra/workload-modeling/workload_modeling/__init__.py` (Create)

*   **Type / Scope**: package marker.
*   **Proposed Changes**: `__version__ = "0.1.0"`.

##### 7. `infra/workload-modeling/workload_modeling/constants.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   `Entry` (frozen dataclass): `value: float | int | str`, `unit: str`,
        `label: Literal["measured", "estimate", "inference"]`,
        `source: str`, `workload: str | None`, `arch: str | None`, `note: str | None`.
    *   `load(path) -> Constants`; `Constants.get(key) -> Entry`; validation of labels
        and sources.
    *   `peak_per_cluster(arch)` and `peak_total(arch, clusters)`, computed from the
        arch entries.

##### 8. `infra/workload-modeling/workload_modeling/fixtures.py` (Create)

*   **Type / Scope**: Python module and command.
*   **Proposed Changes**: `python -m workload_modeling.fixtures refresh` downloads
    `config.json` (with `huggingface_hub.hf_hub_download`) and the safetensors metadata
    (with `huggingface_hub.get_safetensors_metadata`, no weights) for the three pinned
    revisions. It writes them to `tests/fixtures/<owner>__<name>@<sha>/{config.json,header.json}`.
    `load(repo, sha)` reads only fixtures.

##### 9. `infra/workload-modeling/workload_modeling/smolvla_ops.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   `OpRow` (frozen dataclass) with the §3 columns.
    *   `build(constants, workload_id, lowering: Lowering | None = None) -> list[OpRow]`,
        using the dimension sources, formulas and count rule of §3; `Lowering` and
        `pr1_lowering(prefix_rows)` as in §3.
    *   `stage_totals(rows) -> dict[str, StageTotals]`.
    *   Part 1 fills every column except `unit_today` and `hand_kernel`.

##### 10. `infra/workload-modeling/workload_modeling/crosscheck.py` (Create)

*   **Type / Scope**: command, run by hand.
*   **Proposed Changes**: `python -m workload_modeling.crosscheck --parse-profile <path to parse_profile.py> --pr1 <published-fixi-rung6 dir> --libero <libero-G4a dir> --experiment-commit <sha>`.
    *   It loads `parse_profile.py` by path (standard library only) and reads padded
        token counts from the flattened programs.
    *   It runs the model padded and unpadded, compares against the `redmule_flops`
        sums, and writes `docs/analysis/crosscheck.md` and `crosscheck.json`, as in §4.
    *   It exits 1 if any model stage differs by more than 1 %.

##### 11. `infra/workload-modeling/workload_modeling/build.py` (Create)

*   **Type / Scope**: command.
*   **Proposed Changes**: `python -m workload_modeling.build` validates the constants,
    runs the model for `pr1-1cam` and `libero-2cam`, and writes
    `docs/analysis/stage-totals.md` and `stage-totals.csv`. Part 2 also calls
    `render.py` and `md_to_html.py` from here.

##### 12. `infra/workload-modeling/tests/fixtures/` (Create)

*   **Type / Scope**: test data.
*   **Proposed Changes**: the outputs of `fixtures refresh` for
    `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`,
    `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d` and
    `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

##### 13. `infra/workload-modeling/tests/test_constants.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   rejects a missing label or source, and an unknown label;
    *   `peak_total(stage1, 16)` = 16.384 TFLOP/s (exact) and
        `peak_total(pr1, 16)` = 131.072 TFLOP/s (exact);
    *   the smallest square mesh reaching `thor.fp16_dense` is 16 x 16.

##### 14. `infra/workload-modeling/tests/test_smolvla_ops.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   every header tensor is used or listed as unused (`lm_head`, embedding lookup);
    *   used + unused parameter counts equal the header's total (450.0M for
        `smolvla_base`), and per-part counts match the wiki's table;
    *   every config-derived shape matches its header tensor;
    *   the worked example: expert gate/up `flop` = 47.19 GFLOP and
        `flop_per_wbyte` = 50, exact;
    *   per-stage padded matrix-product FLOP equals `docs/analysis/crosscheck.json`
        within 0.01 % (regression guard).

##### 15. `docs/analysis/conventions.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: content of Technical Plan §7.

##### 16. `docs/analysis/hw-config.md` (Create)

*   **Type / Scope**: documentation, hand-written.
*   **Proposed Changes**: the table and statements of Technical Plan §6, each value
    citing its constants key.

##### 17. `docs/analysis/stage-totals.md`, `stage-totals.csv` (Create, generated)

*   **Type / Scope**: generated documentation.
*   **Proposed Changes**: per-stage totals for `pr1-1cam` and `libero-2cam`, with a
    "generated by" header.

##### 18. `docs/analysis/crosscheck.md`, `crosscheck.json` (Create, generated once)

*   **Type / Scope**: generated documentation and data.
*   **Proposed Changes**: as in §4, with the experiment-branch commit and the cache
    paths used.

##### 19. `docs/research-plan/README.md` (Create)

*   **Type / Scope**: documentation, entry point.
*   **Proposed Changes**: content of §7, including the hand-written RQ index.

##### 20. `docs/research-plan/rq/_template.md`, `rq1-core-ops-on-spatz.md`, `rq2-expert-streaming.md`, `rq3-mesh-partitioning.md`, `rq4-mfu-vs-thor.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: the RQs of this design with the §7 front-matter and body
    sections; starting points copied from the "Research questions" section.

##### 21. `docs/research-plan/working/pattern-table.md`, `working/pattern-table-meeting.md` (Create, separate commit)

*   **Type / Scope**: working material.
*   **Proposed Changes**: copied from the owner's untracked
    `workload-modeling/pattern-table*.md` in the main checkout. They are copied
    unchanged, then a separate commit adds a header line marking them as idea-forming
    material. Confirm with the owner before copying.

##### 22. `README.md` (Modify)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: add `infra/workload-modeling/` and `docs/analysis/`,
    `docs/research-plan/` to the layout table, and a "Start here" link to
    `docs/research-plan/README.md`.

##### 23. `.gitignore` (Modify)

*   **Type / Scope**: configuration.
*   **Proposed Changes**: ignore `infra/workload-modeling/build/`.

##### 24. Follow-up after merge: `softhier_vla_wiki/memories/status/analyzing-the-needs-of-one-model-smolvla.md`, `memories/status/_overview.md` (Create / Modify, wiki), then `softhier_vla_wiki` (submodule bump)

*   **Type / Scope**: wiki status entry and pointer bump, after Part 1 merges.
*   **Proposed Changes**: the entry of §9, from `tools/templates/status.md`, with links
    pinned to the Part 1 merge commit. The catalog line goes after "Basic integration of
    components", and the current-state paragraph is updated. Committed with
    `tools/wiki.py commit update`, merged to wiki `main`, then one parent commit
    `[wiki] bump submodule: …`.

#### Part 2: model and compiler explainer (branch `model-and-compiler-explainer`)

##### 25. `infra/workload-modeling/workload_modeling/smolvla_ops.py` (Modify)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: fill `unit_today` and `hand_kernel` from the §8 table.
    Collapse identical layers.

##### 26. `infra/workload-modeling/workload_modeling/render.py`, `md_to_html.py`, `templates/op_graph.md.j2` (Create) and `build.py` (Modify)

*   **Type / Scope**: Python modules and template.
*   **Proposed Changes**: as in Technical Plan §5. `build.py` calls both after the
    stage totals.

##### 27. `docs/analysis/smolvla-op-graph.md`, `smolvla-op-graph.csv` (Create, generated)

*   **Type / Scope**: generated documentation.
*   **Proposed Changes**: outputs for `libero-2cam` with a "generated by" header.

##### 28. `docs/analysis/compiler-walkthrough.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: content of §8. Every code claim cites a file and line at
    softhier-mlir `b11cd5f` or `~/softhier@52859c3`. The SiLU-multiply question is
    resolved by reading `implementation/sw/Activation/`.

##### 29. `infra/workload-modeling/tests/test_render.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   Markdown stage totals equal `stage_totals()`;
    *   the HTML contains every Markdown table;
    *   the only external URL is the pinned mermaid script;
    *   mermaid fences become `<pre class="mermaid">`.

##### 30. Follow-up after merge: the status entry (Modify, wiki) and `softhier_vla_wiki` (submodule bump)

*   **Type / Scope**: wiki status entry update.
*   **Proposed Changes**: rows for the operator table and walk-through, pinned to the
    Part 2 merge commit; same workflow as file 24.

### Verification Plan

*   **Build and drift**:
    `cd infra/workload-modeling && uv sync && uv run python -m workload_modeling.build && git diff --exit-code`.
    It regenerates offline, and a clean tree shows no diff.
*   **Unit tests**: `uv run pytest` (files 13, 14; plus 29 in Part 2).
*   **Cross-check (Part 1, once)**: `uv run python -m workload_modeling.crosscheck --parse-profile … --pr1 … --libero …`
    exits 0.
*   **Manual review**:
    *   the owner reads `docs/research-plan/README.md` and follows the reading order;
    *   the HTML is opened in light and dark modes;
    *   one RQ is added from the template (CUJ 2).
*   **Wiki follow-ups**: `softhier_vla_wiki/tools/wiki.py check` passes; after the bump,
    the README's `wiki://` links resolve.
