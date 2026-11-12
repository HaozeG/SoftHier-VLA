# SoftHier-VLA stage 1: problem definition and model/compiler explainer

*   **Author:** Haoze (with Claude)
*   **Status:** Draft
*   **Date:** 2026-11-12
*   **Predecessor:** [`concepts_brief.md`](./concepts_brief.md) (brainstorm, one Goldfish round)

--------------------------------------------------------------------------------

## Overview

SoftHier-VLA studies how to run robot policy models (vision-language-action models,
VLAs) on a tile-based many-core accelerator, using SoftHier, a configurable mesh of
compute clusters simulated in GVSoC. PR1 built the infrastructure: SmolVLA runs end to
end on a simulated SoftHier chip and matches the published latency (120.7 ms vs
119.1 ms, 1 camera, 256 tokens, measured in simulation). Its profile shows where time
goes, but the project has not yet stated which problem it is trying to solve.

This design covers the documents that close stage 1 ("define the problem"). It has two
parts, done in order:

1.  **Problem definition.** A short problem statement, 3-5 research questions (RQs)
    with metrics and baselines, one conventions note that fixes units and reference
    figures, one constants file, and a hardware parameter table that sizes SoftHier
    against NVIDIA Jetson Thor.
2.  **Model and compiler explainer.** A generated table and diagram of SmolVLA's
    operators with compute and memory needs per operator, and a walk-through of how the
    model goes from a checkpoint to kernels on SoftHier's execution units, stating
    what is built and what is not.

Readers: the owner, collaborators who read Markdown and HTML on GitHub and in meetings,
and the sessions that plan stage 2 from these documents.

## The Problem

**What is missing today.** The project has a working simulator setup and one measured
end-to-end number, but no agreed problem statement, no research questions that a
result can answer, and no fixed reference point to compare against. Three concrete
gaps follow from that:

*   **Numbers do not line up.** The project's figures come from three input
    configurations (1 camera/256 tokens measured, 2 cameras/1024 tokens estimated,
    3 cameras/512x512 estimated) and two SoftHier machines that differ about 8x in
    peak compute (PR1's 131 TFLOP/s arch without Spatz; the kernel repo's
    16.4 TFLOP/s arch with Spatz). Without one reference workload, one reference arch
    and one definition of "bytes", an RQ baseline means nothing.
*   **Colleagues cannot see the model or the compiler path.** The owner needs to
    discuss SmolVLA's operators and their compute needs, and how the compiler maps them
    to hardware, with people who have not read the code. The facts are spread across
    softhier-mlir's code and docs, the PR1 profile and the wiki, and some documents
    describe intended behaviour (for example, Spatz use for softmax) as if it existed.
*   **No comparison point outside the simulator.** "Is this a good chip for embodied
    AI?" needs a reference device. The owner chose Jetson Thor, the edge device most
    VLA benchmark papers measure on, but its published peaks vary between papers
    (153 measured to 500 TFLOP/s) and NVIDIA publishes no dense 16-bit figure.

**The research problem behind it.** The long-term question is what a chip or module for
deploying embodied AI should look like, and how to map work spatially (which clusters)
and temporally (when) on a tile-based chip. This includes dividing the cluster mesh
between a VLA and a fast controller (S0) that run at different rates on the same
chip. SmolVLA is the first worked example, not the subject. Stage 1 only has to frame
this problem with numbers. The roadmap after stage 1 is: static mapping of one model,
then several models, then dynamic scheduling under different latency and rate needs.

**Proposed solution in one paragraph.** Fix the reference inputs first, in one
conventions note and one constants file:

*   workload `libero-2cam`;
*   the kernel-repo arch, scaled by cluster count;
*   Thor at 258 TFLOP/s dense FP16 (derived), with ~155 TFLOP/s measured as a second
    reference;
*   FLOP/B with a written formula per operator class.

Then write the problem statement and RQs against those inputs. Then generate the
operator table from softhier-mlir's own workload model, so its numbers come from code
rather than hand typing. Finally, write the compiler walk-through as a short addition
to softhier-mlir's existing design document.

## Goals

*   **G1. Reference inputs are fixed in one place.** A conventions note and a constants
    file define:
    *   the workload IDs (`pr1-1cam`, `libero-2cam`, `release-3cam`), with `libero-2cam`
        primary;
    *   the stage-1 arch and its cluster-count knob;
    *   the Thor figures with sources;
    *   the naming of model parts and levels;
    *   the evidence labels.
    Every number in the stage-1 documents cites a workload ID and an evidence label.
*   **G2. Problem statement and RQs.** A problem statement of about 200 words and 3-5
    RQs. Each RQ states a metric, a baseline (measured or labelled estimate), the
    workload ID and the arch it applies to. Each RQ is a separate file, so a
    collaborator can add or revise one without touching the others.
*   **G3. Hardware parameter table sized against Thor.** Two variants of the stage-1
    arch: as in the kernel repo (16 clusters, ~16.4 TFLOP/s), and scaled by cluster
    count to about Thor's 258 TFLOP/s (about 252 clusters; a 16x16 mesh gives
    ~262 TFLOP/s). It also gives HBM capacity large enough for SmolVLA's weights,
    bandwidth listed as an open parameter, and Spatz FLOPs reported separately rather
    than counted in peak. Marked as a proposal, not a validated build.
*   **G4. SmolVLA operator table and diagram.** One row per operator of
    SmolVLA on `libero-2cam`. Columns:
    *   operator class (gemm, attn, norm, acti, softmax, elementwise);
    *   shape;
    *   FLOP;
    *   bytes;
    *   FLOP/B;
    *   the execution unit it runs on today;
    *   whether a hand-tuned kernel exists in `~/softhier/implementation` (with
        commit).
    Generated by a script from softhier-mlir's workload model and the constants file,
    rendered to Markdown, with HTML built from the Markdown (not committed). Grouped
    by stage (vision, connector, prefix, action expert) so it can be shown in a
    meeting.
*   **G5. Compiler walk-through.** A short document that follows one SmolVLA layer from
    checkpoint to execution units: safetensors loading, frontend IR, the
    softhier-dialect ops, the emitted runtime calls, and which unit (RedMulE, Snitch
    cores, Spatz, iDMA) executes each. It links to softhier-mlir's DESIGN.md for general
    design and lists explicitly what is not built (Spatz lowering, the linalg path,
    missing kernels).
*   **G6. Entry point.** A research-plan README, linked from the repo README, that a
    new collaborator can start from. It holds the problem statement, the RQ index with
    each RQ's status, the roadmap, and links to the explainer documents and the wiki
    status log.

## Non-Goals

*   **No new simulator runs.** Hand-tuned kernels on SmolVLA shapes, measured
    baselines on the stage-1 arch, and fusion estimates calibrated against them belong
    to the next (measurement) piece of work.
*   **No compiler changes.** No Spatz lowering in softhier-mlir and no linalg path
    work. The walk-through describes the compiler as it is.
*   **No validated Thor-sized build.** The scaled arch is a parameter table. It is not
    checked against TCDM or instruction-memory limits.
*   **No S0 model and no multi-model workload.** VLA/S0 co-location on one chip is on
    the roadmap only, not an RQ in stage 1.
*   **No full-model weight residency option.** SmolVLA's weights (hundreds of MB) do
    not fit on chip; only the KV cache and per-layer tiles are candidates for staying
    on chip.
*   **No wiki changes in this work.** The new wiki areas (`memories/status/`,
    `memories/patches/`), the wiki-design decision, glossary rows and the recovered
    first-baseline decision come from a separate wiki-development session. This work
    only links to them.
*   **No collaborator sync tooling** beyond GitHub pull requests and HTML built from
    Markdown.

## Open Questions

*   **Wiki session output.** The separate wiki-development session has not reported
    yet. The research-plan README and the status links depend on its folder names and
    the naming glossary entry.
*   **Baselines on the chosen arch.** PR1's measured numbers are on the 131 TFLOP/s
    arch, not the stage-1 arch. Until the measurement work runs, RQ baselines on the
    stage-1 arch are estimates. This design must say how each RQ marks that.
*   **Source of per-operator bytes.** softhier-mlir's `dse/workload.py` counts HBM bytes
    assuming no reuse across tiles. Whether the operator table reports that figure, an
    ideal (unique operand) figure, or both is decided in the Technical Plan.
*   **Thor clock.** Whether the derived 258 TFLOP/s assumes the MAXN power mode is not
    confirmed.
