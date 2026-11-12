# SoftHier-VLA stage 1: define the problem and organize the infrastructure analysis

* **Author:** Haoze (with Claude)
* **Date:** 2026-11-12
* **Status:** brainstorm brief, after one Goldfish review round (6 lenses). Not a decision.

## Overview

Stage 1 of SoftHier-VLA ends when the project can state, in a page, which problem it
solves and how it will measure progress. PR1 (merged) built the infrastructure:
SmolVLA runs end to end on a simulated SoftHier chip (GVSoC) and matches the published
latency (120.7 ms vs 119.1 ms, 1 camera, 256 tokens). This brief covers the next step:
turning that infrastructure and its first profile into

1. a short problem statement and 3-5 research questions (RQs), each with a measurable
   metric and a baseline number;
2. three explanatory documents the owner can discuss with colleagues: the hardware
   configuration (sized against NVIDIA Jetson Thor, with a Spatz vector unit per cluster),
   the SmolVLA operator graph with compute and memory needs per operator, and a
   walk-through of the compiler path from checkpoint to the chip's execution units;
3. bounded estimates of what hand-tuned and fused kernels could gain;
4. a project status log titled by theme, a log of changes made to upstream tools, and
   a research-plan skeleton that collaborators can read and extend.

The research focus is general, not SmolVLA-specific: what a chip or module for
embodied-AI deployment should look like, and how to map work spatially (which clusters)
and temporally (when) onto a tile-based many-core chip, including splitting the cluster
mesh between a slow policy model and a fast controller running on the same chip.
SmolVLA is the first worked example. Roadmap: static mapping of one model, then several
models, then dynamic scheduling under different latency and rate needs.

Audience: the owner, collaborators who read Markdown/HTML on GitHub and in meetings,
and later sessions that plan stage 2 from these documents.

## Analysis

The review found the concept sound in direction but not ready to write. The core
deliverable (problem statement plus RQs with baselines) depends on five inputs that are
not fixed yet. Writing the explanatory documents first would produce numbers that
disagree with each other.

### 1. No single reference workload, so no stable baselines

Three input configurations appear in the project's numbers, and FLOP, bytes and
utilization change by about 4x between them:

| ID (proposed)  | Input                                  | Numbers we have                           | Evidence                   |
|----------------|----------------------------------------|-------------------------------------------|----------------------------|
| `pr1-1cam`     | 1 camera, 256 tokens                   | 189 GFLOP, 120.7 ms on simulated SoftHier | measured (simulator)       |
| `libero-2cam`  | 2 cameras x 1024 tokens                | 479 ms                                    | estimate from partial runs |
| `release-3cam` | 3 cameras 512x512 (checkpoint default) | ~834 GFLOP                                | estimate (analytical)      |

Every number in every document must cite one of these IDs and an evidence label.
One of them must be the primary workload for the RQs.

### 2. Spatz is off today, and the compiler cannot use it

Checked in the code:

- The arch file PR1 used sets `spatz_attaced_core_list = []` (no Spatz).
- softhier-mlir compiles for `rv32imafd_zfh` with no vector extension. Row operations
  (softmax, GELU, norm, SiLU) use the Snitch core's 4-lane fp16 SIMD. The
  `softhier.vexp` op is declared but not emitted. Its README and DESIGN.md describe the
  Spatz path as intent, not as current behaviour.
- The simulator does model Spatz. The hand-written kernels in
  `~/softhier/implementation` use it.

So "a Spatz in every cluster" is a configuration change plus new compiler lowering.
This matters because PR1's profile puts 64.5 % of time on these scalar row operations
(per-head softmax alone ~44 % on `libero-2cam`). The hardware document must show two
variants: without Spatz (measured PR1 baseline) and with Spatz (labelled estimate).

### 3. Two different SoftHier machines are in play, about 8x apart in peak

| Arch                             | Cores/cluster | Spatz      | RedMulE | TCDM    | NoC link | Peak (16 clusters, 1 GHz) |
|----------------------------------|---------------|------------|---------|---------|----------|---------------------------|
| PR1 / softhier-mlir              | 3             | none       | 128x32  | 1 MiB   | 512 bit  | ~131 TFLOP/s              |
| Kernel repo (`softhier_deeploy`) | 5             | on 4 cores | 32x16   | 384 KiB | 1024 bit | ~16.4 TFLOP/s             |

"The existing fused kernels" and "the 131 TFLOP/s profile" describe different machines,
and the kernels sit on a branch the PR1 simulator pin does not include. Stage 1 must
name one arch, and every kernel claim must carry a branch and commit.

### 4. Aligning with Thor by peak FLOPS alone gives a misleading comparison

MFU itself needs no alignment: each chip divides by its own peak. Alignment matters only
when SoftHier is sized "the same as" Thor. Iso-peak-FLOPS is the common choice in
architecture papers, but here:

- The published Thor peak varies from 153 TFLOP/s (measured) to 258, 400 and 500. That
  moves the sizing by up to ~3x. Sparse and FP8/FP4 figures must not be used.
- At Thor's ~273 GB/s, SmolVLA batch 1 is memory-bound. A rough check on `release-3cam`:
  ~2.5 GB of weight traffic per call (expert weights re-read across 10 flow steps) takes
  ~9 ms, while 834 GFLOP at 258 TFLOP/s takes ~3.2 ms (estimate). So the bandwidth and
  SRAM choices decide the result, and those come "for free" with an iso-FLOPS rule.
- PR1 ran with ideal HBM (patch 0004), so no measured SoftHier number responds to
  bandwidth yet. Any bandwidth sweep in this stage is analytical.
- SoftHier has no single GB/s setting. Bandwidth comes from NoC link width times channel
  count at a fixed 1 GHz, and the calibrated aggregate with 16 clusters streaming is
  about 85 GB/s. HBM capacity (512 MiB) is below the fp16 weight size (~674 MiB).

A fairer rule: fix a cited dense BF16 Thor peak (with 153 measured as a second
reference), match machine balance (FLOP per byte) or bandwidth, and sweep on-chip SRAM.
Report peak as a ratio. Name which knob reaches 273 GB/s. State whether Spatz FLOPs
count toward peak.

### 5. The wiki changes conflict with the wiki's own rules

- Wiki-design decision 0013 bans derived estimates. It was adopted after two estimate
  notes went stale.
- The wiki's CLAUDE.md says sessions from the parent repo do not edit `wiki-design/`.
- The wiki lint accepts six note types. A status entry or patch-log entry fits none.
- The 0013 rule is restated in about six files that would all need updating.

The owner approved new `memories/status/` and `memories/patches/` areas. The review
says to do this as a new wiki-design decision, written in a separate wiki-development
session. It would scope the exception to those two folders, add the note types to the
lint, and leave a one-line "revised by" pointer in 0013. Doing it inside this project PR
would break the process rule and fail the lint.

### 6. Roofline-style fusion estimates will overstate gains

PR1's losses are not FLOP or byte limits:

- barrier wait takes 22 % of time;
- vision and prefix GEMMs use only 1-10 of 16 clusters;
- there is no cluster-to-cluster multicast;
- scalar cores issue an instruction in only 6-14 % of cycles.

Keeping all weights on chip is not possible: the action expert alone is 196 MB. Only the
KV cache (4.9 MB) and per-layer tiles fit. Estimates should be bounds on measured time
("removing X recovers at most Y of Z ms"), limited to about three cases tied to the
largest measured costs:

1. row operations on Spatz;
2. expert weight streaming (KV cache kept on chip, weights reused across the 10 flow
   steps, int8/fp8 weights);
3. cluster under-use.

They should use the existing calibrated cost model `softhier_mlir/dse/cost.py`.

### 7. "S1/S0" naming clashes with the glossary

The wiki glossary follows GR00T/Helix:

- System 2 is the slow vision-language model;
- System 1 is the fast action policy;
- System 0 is the controller below that.

SmolVLA spans System 2 and System 1. Calling the whole VLA "S1" contradicts this. The
mapping must be settled in the project glossary before an RQ uses it. The co-location
RQ also has no baseline yet: there is no S0 model, rate or deadline. It needs either a
concrete stand-in (for example a fixed-size MLP at 200-1000 Hz with a deadline) or an
explicit "baseline TBD, method X".

### 8. "Bytes" and "arithmetic intensity" need one written definition

"Bytes" can mean four different things:

- unique operand bytes;
- DRAM traffic under the kernels' real tiling;
- traffic with on-chip reuse;
- traffic counted at L1 rather than DRAM.

`dse/workload.py` assumes no reuse across tiles. Intensity for operations on scalar or
Spatz units must be compared with those units' peak, not RedMulE's. "AI" also collides
with "embodied AI" in the same pages. Use "FLOP/B" and one conventions note with a
formula per operator class.

### 9. Numbers and patch entries will drift unless they have one source

- PR1 numbers and Thor constants would be retyped in four documents. Keep one constants
  file and generate the tables from it.
- Patch files are re-exported, so patch numbers change. Key patch-log entries by
  finding ID (F1-F19 in the baseline design doc) and link with commit-pinned permalinks.
- `infra/softhier/gvsoc-patches/README.md` already has what/why/upstream-worthy columns.
  The wiki log should hold only what the README lacks: upstream status, measured impact
  and links to the RQs.

### 10. The compiler walk-through must describe what exists

The actual path is:

1. weights load from safetensors through `prepare` and `sim/preload.py`;
2. Python/xDSL frontends build softhier-dialect IR directly;
3. `emit_c.py` maps each op to one `sh_*` runtime call.

The three linalg passes are not on the SmolVLA path. Hand kernels exist only for attn,
gemm, norm and acti. There are none for softmax, LayerNorm, GELU, RoPE or residual add,
and SigLIP needs LayerNorm and GELU. softhier-mlir's DESIGN.md already covers the
general design, so this project's document should be a short delta that links to it and
lists what is unbuilt.

### 11. A broken link already exists

The repo README (line 34) links
`memories/decisions/2026-11-10-first-baseline-latency-and-success-rate.md`, which is
absent from the wiki on every branch and in its history. Fix it before adding more links.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  - Fix the stage-1 inputs:
    - primary workload ID;
    - stage-1 arch;
    - Thor peak figure, its source and the alignment rule;
    - S2/S1/S0 naming;
    - the definition of bytes and FLOP/B.
  - Problem statement (~200 words, SmolVLA framed as the example) and 3-5 RQs. Each RQ
    gives a metric, a baseline from measured PR1 numbers or a labelled estimate, and a
    config ID.
  - A hardware parameter table with and without Spatz, sized against Thor. It is a
    proposal, not a validated build.
  - SmolVLA operator table, generated by a script wrapping `dse/workload.py`. Columns:
    FLOP, bytes, FLOP/B, execution unit, and whether a hand kernel exists (with commit).
    Rendered to MD, with HTML built from it.
  - Compiler walk-through as a delta on softhier-mlir DESIGN.md.
  - About three bounded fusion estimates using `dse/cost.py`.
  - A theme-titled status log and an upstream patch log (wiki, after the wiki-design
    session), plus a research-plan skeleton.
* **Out-of-Scope Non-Goals:**
  - New simulator runs, including the hand kernels on SmolVLA shapes (next PR).
  - softhier-mlir Spatz lowering and linalg path work.
  - A validated Thor-sized SoftHier build: TCDM and the 64 KB instruction memory may not
    fit.
  - Modeling S0 or a multi-model workload.
  - Full-model weight residency.
  - Sync tooling for collaborators beyond GitHub PRs and generated HTML.
  - Committing generated HTML.

## Alternatives Weighed

- **One large PR with all four documents:** too big to review, and it crosses the code
  repo, the experiment branch and the wiki's governance boundary. Rejected in favour of
  a split (see Key Decisions).
- **Iso-peak-FLOPS as the only alignment with Thor:** hides the memory-bound regime.
  Kept only as one reported ratio alongside matched bandwidth or machine balance and an
  SRAM sweep.
- **Amending decision 0013 inside this project PR:** breaks the wiki's process rule and
  fails the lint. Replaced by a new wiki-design decision in a separate session.
- **Rewriting 0013 wholesale:** loses the reason the ban exists. A new scoped decision
  plus a pointer keeps both.
- **Status log in date order or one file per PR:** contradicts theme titling. Use theme
  files, catalog order for sequence, `follows:`/`superseded_by:` links, and git for
  dates.
- **Wiki patch log that mirrors the gvsoc-patches README:** fields duplicate and drift.
- **Research plan as its own submodule now:** adds a third place to sync before anyone
  needs it.
- **Research plan inside the wiki:** the wiki is lint-gated and limited to what sources
  say, so collaborators' edits would fail its checks. Proposed home: `docs/research-plan/`
  in the code repo, linking to the wiki in one direction.
- **Hand-typed FLOP and intensity tables:** cannot be reproduced and drift from the
  constants.
- **RQs inside the PR1 baseline design doc:** that doc is a fixed record of PR1; the RQs
  outlive it.
- **Dropping the hardware config from stage 1:** the baselines need a fixed comparison
  point, so a minimal parameter table stays.

## Key Decisions & Staging for Next Steps

Decided by the owner during ideation:

- Target SoC: NVIDIA Jetson Thor.
- Controller (S0) placement: some clusters of the same chip, sized by its compute need;
  MLP or small transformer; not modeled in stage 1.
- This PR: documentation, visualization and analysis only. Hand-kernel measurements on
  SmolVLA shapes go to the next PR.
- Sharing: GitHub plus meeting presentations; Markdown plus HTML as the share format;
  code through PRs.
- Stage-1 exit: concise problem statement, quantifiable RQs, baselines.
- Roadmap: one model static, then several models, then dynamic scheduling.
- Wiki: new `memories/status/` and `memories/patches/` areas approved. Patch files stay
  canonical in the code repo.

Decided by the owner after review round 1:

- **Stage-1 arch:** the kernel-repo arch (5 cores per cluster, Spatz on 4 cores, RedMulE
  32x16, 384 KiB TCDM, ~16.4 TFLOP/s at 16 clusters). All parameters stay configurable
  and the owner may change them later.
- **Thor alignment:** iso-peak-FLOPS at fp16 only. The knob is cluster count. HBM
  capacity grows so the model fits. Spatz FLOPs do not count toward peak. Bandwidth
  stays open for discussion. The Thor figure comes from ingested papers or other cited
  sources (being looked up). If the dense 16-bit figure is 258 TFLOP/s, matching needs
  about 16x the clusters (~256, e.g. a 16x16 mesh); this is arithmetic, not a validated
  build.
  - Caveat kept from the review: the owner's reason is that prefill is compute-bound
    in benchmark papers. That holds on RTX 4090 (OpenVLA prefill 86 % compute-bound),
    but on Jetson the wiki records π0.5 prefill near the ridge and GR00T N1.7 prefill
    memory-bound. For SmolVLA on `libero-2cam`, vision and the ~2k-token prefill reach
    about 2000 FLOP per weight byte (estimate), above Thor's balance of about 950 FLOP/B,
    so they are compute-bound. The action expert stays bandwidth-bound, and it is the
    largest latency share. Any RQ about the expert must state this.
- **Primary workload:** `libero-2cam`.
- **Baselines:** both measured numbers and labelled analytical estimates are allowed.
- **Hand-tuned reference kernels:** only the four in `~/softhier/implementation`
  (attn = FlatAttention, gemm = SUMMA, norm = RMSNorm, acti = Sigmoid/ReLU/SiLU) count
  as hand-tuned optimal. Other kernels seen in local logs are ignored.
- **Pattern table** (`workload-modeling/pattern-table.md`): derived from the wiki, part
  of the owner's idea forming while analyzing workloads. Proposed home: the research
  plan's working material (e.g. `docs/research-plan/working/`), tracked, with its
  existing evidence labels.
- **Naming of levels:** in this brief "S1" means the whole SmolVLA and "S0" a fast
  MLP/transformer controller on the same chip. This conflicts with the wiki glossary
  (System 2 = VLM, System 1 = action policy). The naming used in recent papers is being
  looked up before anything goes into the glossary.
- **Wiki changes:** done in a separate wiki-development session (new wiki-design decision
  for `memories/status/` and `memories/patches/`, lint note types, first status entry,
  patch log keyed by finding ID, recovery of the uncommitted first-baseline decision,
  hardware glossary rows). The missing decision was found as an uncommitted draft in the
  `smolvla-e2e-baseline` worktree's wiki copy.
- **Naming of the next deliverables:** the owner expects the plan to change, so a fixed
  label like "PR2a" is misleading. Proposed instead: name each piece of work by its
  content, the same rule as status-log themes, e.g. branch `problem-definition` for the
  problem statement, conventions, constants, hardware table and RQs; branch
  `model-and-compiler-explainer` for the operator table and compiler walk-through. The
  research plan's README records which pieces exist and their status.

Proposed staging (from the review; labels below are working names, see naming above):

- **Step 0, wiki-development session:**
  - Write a new wiki-design decision allowing labelled estimates in `memories/status/`
    and `memories/patches/` only.
  - Add the note types to the lint and leave a pointer in 0013.
  - Resolve the missing first-baseline decision.
  - Add glossary rows (Spatz, RedMulE, Snitch, iDMA, SUMMA, FlatAttention, GVSoC, F-ids,
    the S2/S1/S0 mapping).
- **PR2a, problem definition:**
  - `docs/analysis/conventions.md` (config IDs, bytes/FLOP-per-byte definitions, Thor
    figure, alignment rule).
  - A constants file.
  - The hardware parameter table.
  - `docs/research-plan/README.md` (problem statement, RQ index, roadmap) with one file
    per RQ.
  - First status entry: "Analyzing the needs of one model: SmolVLA".
- **PR2b, explanation:** generated SmolVLA operator table (MD + HTML) and the compiler
  walk-through delta.
- **PR3, measurement:** hand kernels on SmolVLA shapes on the chosen arch, then the
  bounded fusion estimates calibrated against them.

Still open:

1. The Thor dense fp16 figure and its source (lookup running; wiki has 258, 400 and
   153 measured).
2. The level naming convention from recent papers (lookup running), then one project
   glossary entry.
3. Is co-location of S0 with the policy an RQ in stage 1 (with a stand-in S0 model,
   rate and deadline), or roadmap only?
4. The measured PR1 baseline ran on the 131 TFLOP/s arch, not the chosen 16.4 TFLOP/s
   arch. Measured baselines on the chosen arch need new runs (next measurement PR);
   until then, RQ baselines on the chosen arch are estimates.
