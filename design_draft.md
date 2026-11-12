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
    peak compute (PR1's 131 TFLOP/s arch; the kernel repo's 16.4 TFLOP/s arch). Spatz
    is part of the default SoftHier cluster. PR1's arch file disabled it only because
    softhier-mlir's first implementation does not generate Spatz code, so PR1's
    measured numbers also exclude Spatz's effect on the cluster (for example its memory
    ports). Without one reference workload, one reference arch
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
*   **No wiki structure changes in this work.** The new wiki areas
    (`memories/status/`, `memories/patches/`), the wiki-design decision, glossary rows
    and the recovered first-baseline decision come from a separate wiki-development
    session. This work adds one status entry through the normal wiki workflow (owner's
    choice; see Technical Plan §9) and otherwise only links to the wiki.
*   **No collaborator sync tooling** beyond GitHub pull requests and HTML built from
    Markdown.

## Open Questions

*   **Wiki branch not merged yet.** The wiki-development session finished on wiki
    branch `wiki-status-patches` (status and patch areas, naming entry, recovered
    first-baseline decision). This work links to `wiki://memories/status/` and the
    naming glossary row, which resolve only after the owner merges that branch.
*   **Baselines on the chosen arch.** PR1's measured numbers are on the 131 TFLOP/s
    arch, not the stage-1 arch. Until the measurement work runs, RQ baselines on the
    stage-1 arch are estimates. This design must say how each RQ marks that.
*   **Thor clock.** Whether the derived 258 TFLOP/s assumes the MAXN power mode is not
    confirmed.

## Critical User Journeys (CUJs)

### CUJ 1: Discuss SmolVLA's compute needs in a meeting

*   **Given**: the operator table and diagram are generated for `libero-2cam` and the
    HTML page is built locally.
*   **When**: the owner opens the HTML page (or the Markdown on GitHub) and walks
    through vision, connector, prefix and action expert.
*   **Then**: for each stage and operator class, the page shows FLOP, unique bytes,
    weight bytes, FLOP per weight byte, the execution unit used today, and whether a
    hand-tuned kernel exists. Every number carries its workload ID and evidence label,
    and the stage totals match the constants file.

### CUJ 2: A collaborator proposes a new research question

*   **Given**: a collaborator has cloned the repo and read the research-plan README.
*   **When**: they copy the RQ template into a new file under `docs/research-plan/rq/`,
    fill in metric, baseline key and workload ID, and open a pull request.
*   **Then**: the RQ index in the README is regenerated from the RQ files and the
    constants, without editing any other RQ, and the check reports any baseline key
    that is missing from the constants file.

### CUJ 3: A new measurement replaces an estimate

*   **Given**: the measurement work produces a measured baseline on the stage-1 arch.
*   **When**: the owner updates the value and its evidence label in the constants file.
*   **Then**: regenerating updates every table and RQ index entry that cites that key.
    The superseded estimate stays in git history and in the wiki status log, marked
    superseded.

## Technical Plan (Architecture)

### Parts and data flow

All analysis code lives in `workload-modeling/`, a small uv project like
`infra/lerobot/`. All prose lives in `docs/`. Generated files are committed when they
are Markdown or CSV (so GitHub shows them) and not committed when they are HTML.

```text
 HF Hub: config.json +            constants.yaml
 safetensors header (no weights)  (workloads, arch, Thor, PR1 numbers, labels)
            |                               |
            v                               v
   smolvla_ops.py  ------------------>  ops table (CSV)
   (standalone analytical model:            |
    one row per operator)                   +--> render.py --> docs/analysis/smolvla-op-graph.md
            |                               |                  (tables + Mermaid diagram)
            |                               |                        |
            |                               |                        v  md_to_html (Python-Markdown)
            |                               |                  build/smolvla-op-graph.html
            |                               |                  (local, for meetings; not committed)
            |                               +--> hw_table.py --> docs/analysis/hw-config.md
            |                               +--> rq_index.py --> docs/research-plan/README.md
            |                                                    (RQ index block only)
            v
   crosscheck_softhier_mlir.py  (run once, by hand, needs ~/softhier-mlir)
            --> docs/analysis/crosscheck.md  (FLOP per op vs softhier-mlir workload,
                                             with the softhier-mlir commit id)
```

### 1. Constants file

`workload-modeling/constants.yaml` is the single place where numbers that appear in more
than one document live. Every entry has `value`, `unit`, `label` (measured / estimate /
inference / derived, as the wiki glossary defines) and `source` (a file path with commit,
a wiki URI, or a citation). Groups:

*   **Workloads:** `pr1-1cam`, `libero-2cam` (primary), `release-3cam`. Each entry gives
    camera count, image size, vision tokens per camera, language and state token
    counts, action chunk length, flow steps, and the HF checkpoint the shapes come
    from.
*   **Stage-1 arch:** the kernel-repo arch parameters, copied from
    `~/softhier/implementation/config/arch/arch.py` with its commit:
    *   cores per cluster;
    *   which cores have Spatz;
    *   RedMulE size;
    *   TCDM size;
    *   NoC link width;
    *   clock.
    Peak per cluster is computed, not typed: RedMulE height x width x 2 FLOP x clock.
*   **Thor:** 258 TFLOP/s dense FP16 (derived: NVIDIA's 517 sparse FP16 / 2),
    ~155 TFLOP/s measured (Chalmers 2611.06710; NVIDIA forum benchmark), 273 GB/s,
    128 GB.
*   **PR1 results:** 120.7 ms, 119.1 ms published, 479 ms (estimate), the profile shares,
    each with its arch (the 131 TFLOP/s PR1 arch).

### 2. Standalone SmolVLA model (`smolvla_ops.py`)

Builds the operator list from the checkpoint's `config.json` and the safetensors header
(tensor names and shapes, read from the first bytes of the file through an HTTP range
request; no weights are downloaded). For a workload ID it emits one row per operator
instance group:

| Column               | Meaning                                                                                       |
|----------------------|-----------------------------------------------------------------------------------------------|
| `stage`              | vision, connector, prefix, expert                                                             |
| `layer`              | layer index, or a range when identical layers are collapsed (`x12`)                           |
| `op`                 | operator name (e.g. `q_proj`, `qk`, `softmax`, `pv`, `o_proj`, `ffn_up`, `gelu`, `layernorm`) |
| `class`              | gemm, attn, norm, acti, softmax, elementwise                                                  |
| `shape`              | (M, N, K) for matrix products; (rows, cols) otherwise                                         |
| `count`              | repetitions per call (layers, heads, flow steps)                                              |
| `flop`               | per call, all repetitions                                                                     |
| `bytes_unique`       | weights + inputs + outputs, each counted once, fp16                                           |
| `bytes_weight`       | weight bytes only                                                                             |
| `flop_per_wbyte`     | `flop / bytes_weight`; empty for operators without weights                                    |
| `unit_today`         | execution unit in PR1's compiled path (RedMulE, Snitch SIMD)                                  |
| `hand_kernel`        | which of attn / gemm / norm / acti in `~/softhier/implementation` covers it, with commit      |

Rules that keep it honest:

*   **Attention grouping.** softhier-mlir emits attention as QK product, softmax and PV
    product. The model emits those three rows and tags all three `class = attn`, so the
    table can show attention as one group (matching the hand-written FlatAttention
    kernel) or split.
*   **Byte definition (owner's choice).** Two byte columns: unique bytes and weight
    bytes. FLOP per weight byte is the intensity figure, the same convention as
    VLA-HWBench's `W`, so it can be compared with the published Thor data in the wiki.
    Stated limitation in the generated page: neither column shows re-reads caused by
    tiling (the cause of PR1's iDMA-bound expert products), and activation-heavy
    operators without weights (softmax over long sequences) have no intensity figure.
*   **Expert across flow steps.** Expert rows are per flow step with `count` = steps.
    Weight bytes for the expert are reported once per step (the weights are re-read
    each step when they do not stay on chip). The page states this.
*   **Only real operators.** Operators that the checkpoint does not run (the unused
    `lm_head`, embedding lookup) are listed in a footnote, not in the table.

### 3. Cross-check against softhier-mlir (run once)

`crosscheck_softhier_mlir.py` runs softhier-mlir's frontends for `pr1-1cam` (the
configuration softhier-mlir already supports), extracts its workload with
`softhier_mlir.dse.workload`, and compares, per stage, the multiply-accumulate counts
and operator shapes with the standalone model. It writes `docs/analysis/crosscheck.md`
with the softhier-mlir commit and a per-stage table of differences. It is the only
script that needs softhier-mlir's environment. It is not part of regular regeneration.
The acceptance rule is that per-stage FLOP agrees within 1 %. Any larger difference is
explained in the page (for example padding in the compiled path) before the table is
used for RQs.

### 4. Rendering

*   **Markdown** (`docs/analysis/smolvla-op-graph.md`, committed):
    *   a Mermaid block diagram of the four stages (GitHub renders Mermaid), each box
        labelled with the stage's FLOP share and FLOP per weight byte;
    *   one table per stage with the layer collapsed where layers are identical;
    *   a totals table;
    *   the limitations paragraph.
*   **HTML** (`workload-modeling/build/smolvla-op-graph.html`, git-ignored): converted
    from the committed Markdown by `md_to_html.py` (owner's choice: one template,
    identical content). It uses the Python-Markdown library with the tables and fenced-code
    extensions, so no system binary such as pandoc is needed. Inline CSS adds light and
    dark themes. Mermaid blocks are drawn by mermaid.js loaded from jsDelivr at a pinned
    version; this is the page's only external URL. Tables read offline, and the diagram
    needs network. The page is static: no per-operator hover detail.
*   One command regenerates everything except the cross-check:
    `uv run python -m workload_modeling.build` (writes CSV, Markdown, HTML and the
    generated blocks in the hardware table and RQ index).

### 5. Hardware parameter table (`docs/analysis/hw-config.md`)

The prose is hand-written. The table block between marker comments is generated by
`hw_table.py` from the constants file. Columns are parameter, kernel-repo arch
(16 clusters), Thor-matched (cluster count scaled), Thor, and notes.

*   **Rows:**
    *   clusters and mesh shape;
    *   cores per cluster and which have Spatz;
    *   RedMulE size;
    *   peak FP16 (RedMulE only);
    *   Spatz throughput (reported, not counted in peak);
    *   TCDM per cluster and total on-chip SRAM;
    *   HBM capacity;
    *   bandwidth (open: listed as the current per-link figure, not matched).
*   **Cluster count:** the smallest square mesh whose peak reaches the Thor figure
    (16x16 = 256 clusters, ~262 TFLOP/s).
*   **HBM capacity:** the smallest power of two that holds SmolVLA's fp16 weights plus
    activations for `libero-2cam` (computed from the model).

The page states that the scaled arch has not been built or checked against TCDM and
instruction-memory limits.

### 6. Research plan (`docs/research-plan/`)

*   **`README.md`** is the entry point. It is hand-written except for the RQ index block,
    and holds:
    *   the problem statement (~200 words);
    *   the stage-1 exit criterion;
    *   the roadmap (one model static, several models, dynamic scheduling, VLA/S0
        co-location);
    *   reading order: conventions, hardware table, operator graph, compiler
        walk-through, wiki status log;
    *   the RQ index, generated by `rq_index.py` from the RQ files' front-matter and the
        constants file: id, title, metric, baseline value and label, workload, arch,
        status.
*   **`rq/rqN-<slug>.md`:** one file per RQ. The front-matter has these fields:
    *   `id`
    *   `title`
    *   `status` (draft / active / answered / dropped)
    *   `metric`
    *   `baseline` (a constants key)
    *   `workload`
    *   `arch`
    *   `depends_on`

    The body has: question, why it matters, method, what result would answer it.
    `rq/_template.md` is the template.
*   **`working/`:** idea-forming material that is not a decision. The owner's pattern
    table (`workload-modeling/pattern-table.md`, currently untracked in the main
    checkout) moves here, keeping its evidence labels.
*   **Naming and terms** follow the wiki glossary (the VLA, VLM trunk, action expert,
    S0, "S2+S1 in one model"). The conventions note links to them rather than redefining
    them.

### 7. Conventions note (`docs/analysis/conventions.md`)

One page that every other stage-1 document links to. It defines:

*   the workload IDs;
*   the evidence labels (linking to the wiki glossary);
*   the two byte columns and FLOP per weight byte, with the formula per operator class;
*   the Thor figures and the alignment rule (iso-peak FP16 by cluster count; Spatz not
    counted; bandwidth open);
*   the rule that every number cites a workload ID and a label.

Its numbers are references to constants keys, rendered by the build.

### 8. Compiler walk-through (`docs/analysis/compiler-walkthrough.md`)

Hand-written, short, and a delta on `~/softhier-mlir/docs/DESIGN.md` (linked with its
commit). It follows one action-expert layer and one SigLIP layer through these steps:

1.  weights loaded from safetensors by softhier-mlir's prepare step and `sim/preload.py`
    into the simulated HBM;
2.  the Python/xDSL frontend building softhier-dialect IR directly (no linalg);
3.  `emit_c.py` turning each softhier op into one `sh_*` runtime call;
4.  the runtime call running on the physical units: RedMulE for matrix products,
    Snitch cores with 4-lane fp16 SIMD for row operations, iDMA for HBM-TCDM moves,
    barriers between clusters.

A table maps each operator class to its unit in two columns:

*   PR1's compiled path (Spatz unused, because the compiler does not yet target it;
    PR1's arch file therefore disabled it);
*   the hand-written kernels on the stage-1 arch (FlatAttention, SUMMA GEMM, RMSNorm and
    activations, with Spatz for row math).

A last section lists what is not built, each item with a code reference:

*   Spatz lowering (`softhier.vexp` declared but not emitted);
*   the linalg path (three passes, not on the SmolVLA path);
*   operators with no hand kernel (softmax, LayerNorm, GELU, RoPE, residual add).

### 9. Wiki status entry

The problem-definition work writes the status entry "Analyzing the needs of one model:
SmolVLA" itself (owner's choice). It follows the wiki's own workflow:

*   a wiki branch;
*   the `status` note type with `evidence:` front-matter and `Label`/`Source` columns;
*   `wiki.py check` passing;
*   then a submodule bump in the parent pull request.

The entry `follows:` "Basic integration of components" and summarizes the reference
inputs, the stage totals, the hardware table and the RQs. Links are pinned to commits
on the `problem-definition` branch. The pull request must therefore be merged with a
merge commit, not squashed, so the pinned commits stay reachable. The explainer work
later adds rows for the operator table, the cross-check and the compiler walk-through
to the same entry, the same way.

### Order of work

Part 1 (problem definition):

1.  constants and conventions;
2.  hardware table;
3.  standalone model, enough for stage totals;
4.  RQs and README.

Part 2 (explainer):

1.  full operator table and rendering;
2.  cross-check;
3.  compiler walk-through.

The RQs may need the stage totals from the model before Part 2 is finished. That is
why the model's stage totals come in Part 1 and the per-operator detail in Part 2.

## Alternatives Considered

Rejected during the brainstorm (from `concepts_brief.md`, "Alternatives Weighed") and
during drafting.

### Alternative A: One large pull request with every stage-1 document

*   **Pros**: one review; all documents land together and cannot disagree at merge time.
*   **Cons**: too large to review well. It also crosses the code repo, the experiment
    branch and the wiki's governance boundary (wiki changes need their own session).
*   **Reason for Rejection**: split into the two parts of this design, the separate wiki
    session that has already run, and the later measurement work.

### Alternative B: Iso-peak FLOPS as the only alignment with Thor, bandwidth matched implicitly

*   **Pros**: the most common rule in architecture papers. A single knob (cluster count).
*   **Cons**: SmolVLA's action expert is bandwidth-bound at batch 1, and it is the largest
    latency share. An iso-FLOPS rule that silently inherits SoftHier's HBM bandwidth
    would favour SoftHier.
*   **Reason for Rejection (partial)**: the owner kept iso-peak FP16 by cluster count as
    the alignment rule, because vision and prefill are compute-bound on `libero-2cam`.
    Bandwidth is listed as an open parameter, never silently matched, and every
    expert-related RQ states the bandwidth limit.

### Alternative C: Amend wiki decision 0013 inside this project work

*   **Pros**: one place to change the rule; no extra session.
*   **Cons**: sessions from the parent repo may not edit `wiki-design/`. The wiki lint
    had no note type for status or patch entries.
*   **Reason for Rejection**: done instead as a new scoped wiki-design decision in a
    separate wiki session (branch `wiki-status-patches`), with a "revised by" pointer
    in 0013 that keeps the reason for the original ban.

### Alternative D: Status log in date order, or one entry per pull request

*   **Pros**: easy to append; mirrors git history.
*   **Cons**: contradicts the owner's rule of titling by theme. It grows without bound,
    and titles such as "PR2" carry no meaning for a reader.
*   **Reason for Rejection**: theme-titled entries. Catalog order and
    `follows:`/`superseded_by:` links give the sequence, and git gives the dates.

### Alternative E: Wiki patch log that mirrors the gvsoc-patches README

*   **Pros**: one page in the wiki with everything.
*   **Cons**: the what/why/upstream-worthy columns would exist twice and drift apart.
    Patch numbers change when the series is re-exported.
*   **Reason for Rejection**: patch files and their README stay canonical in the code
    repo. The wiki log holds only upstream status, measured impact and links, and it is
    keyed by finding ID.

### Alternative F: Research plan inside the wiki, or as its own submodule

*   **Pros (wiki)**: one knowledge base. **Pros (submodule)**: independent versioning.
*   **Cons**: the wiki is lint-gated and limited to what sources say, so collaborators'
    edits to RQs would fail its checks. A third repository adds a sync point before a
    second author exists.
*   **Reason for Rejection**: `docs/research-plan/` in the code repo, linking to the wiki
    in one direction.

### Alternative G: Hand-typed FLOP and intensity tables, or RQs inside the PR1 baseline design doc

*   **Pros**: no generator code; RQs next to the numbers they start from.
*   **Cons**: hand-typed tables cannot be reproduced and drift from the constants. The
    PR1 design doc is a fixed record of PR1, and the RQs outlive it.
*   **Reason for Rejection**: generated tables from one constants file; RQs as separate
    files.

### Alternative H: Drop the hardware configuration from stage 1

*   **Pros**: less work; the scaled arch is not validated anyway.
*   **Cons**: RQ baselines need a fixed comparison point.
*   **Reason for Rejection**: a minimal parameter table stays, marked as a proposal.

### Alternative I: Full-model weight residency as a fusion option

*   **Pros**: would remove weight streaming entirely.
*   **Cons**: SmolVLA's fp16 weights are hundreds of MB (the action expert alone is
    ~196 MB), against ~96 MiB of TCDM in total on the 256-cluster mesh (256 x 384 KiB).
*   **Reason for Rejection**: only the KV cache and per-layer tiles are candidates for
    staying on chip.

### Alternative J: Get operator data from softhier-mlir as a pinned submodule

*   **Pros**: numbers match the IR the compiler actually emits and the simulator runs.
*   **Cons**: adds softhier-mlir and its xDSL environment as a dependency on `main`.
    The submodule must follow softhier-mlir's changes, and the frontends support only
    the configurations already built.
*   **Reason for Rejection**: owner chose a standalone model with a one-time cross-check.

### Alternative K: Put the generator inside softhier-mlir and store only its output here

*   **Pros**: no code dependency in SoftHier-VLA; numbers still come from the compiler.
*   **Cons**: needs a change in softhier-mlir during a stage whose Non-Goals exclude
    compiler work. Updating the table requires working in a second repository.
*   **Reason for Rejection**: owner chose the standalone model.

### Alternative L: Bytes as unique operand bytes only

*   **Pros**: one arch-independent column; simplest.
*   **Cons**: gives no figure comparable to the published Thor data (which uses weight
    bytes) and still shows no tiling re-reads.
*   **Reason for Rejection**: owner chose unique bytes plus weight bytes, with FLOP per
    weight byte as the intensity figure.

### Alternative M: Bytes as unique bytes plus bytes the library moves under tiling

*   **Pros**: the gap between the two columns is the headroom for fusion and reuse. It
    would show the re-reads behind PR1's iDMA-bound expert products.
*   **Cons**: needs a tiling choice, and it reproduces library formulas in the model.
    It would not describe the hand-written kernels if they tile differently.
*   **Reason for Rejection**: owner chose weight bytes for comparability with published
    data. Tiling traffic is left to the measurement work.

### Alternative N: Different scope for this design document

*   **Problem definition only**: shorter, but RQ baselines depend on stage totals the
    explainer produces.
*   **Also the measurement work**: the full stage-1 picture, but the measurement details
    are the most likely to change after the first runs.
*   **Reason for Rejection**: owner chose problem definition plus explainer, phased, with
    the measurement work only referenced.

### Alternative O: Interactive HTML rendered from the operator rows

*   **Pros**: per-operator detail on hover and a self-contained page with no network
    needed.
*   **Cons**: two templates (Markdown and HTML) to maintain; the HTML may fall behind
    the Markdown in layout.
*   **Reason for Rejection**: owner chose to convert the HTML from the committed
    Markdown, so there is one template and identical content.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Reproducibility:** regeneration runs offline from committed fixtures (HF config and
    safetensors header cached under `workload-modeling/tests/fixtures/`). Network access
    is only needed to refresh the fixtures.
*   **Collaborator setup:** `uv sync` in `workload-modeling/` and one build command. No
    simulator, GPU or softhier-mlir environment is needed, except for the one-time
    cross-check.
*   **Evidence discipline:** the build fails if a constants entry lacks `label` or
    `source`, or if an RQ cites a missing constants key.
*   Security, privacy and rollout: none critical (documentation and offline scripts).

### Additional Resources

*   [`concepts_brief.md`](./concepts_brief.md): brainstorm brief with owner decisions.
*   [`docs/designs/smolvla-softhier-baseline.md`](./docs/designs/smolvla-softhier-baseline.md):
    PR1 design, findings F1-F19.
*   `~/softhier-mlir/docs/DESIGN.md`, `softhier_mlir/dse/workload.py`, `dse/cost.py`.
*   `~/softhier/implementation/` (README, `config/arch/arch.py`, `sw/`): stage-1 arch and
    hand-tuned kernels.
*   Wiki: `wiki://knowledge/mfu-from-measurements.md`, `wiki://knowledge/smolvla.md`,
    `wiki://resources/serving/vla-hwbench-edge-latency.md`, branch `wiki-status-patches`
    (status and patch logs, naming).

### Future Improvements

*   **Tiling-traffic column:** add bytes moved under each kernel's real tiling once the
    measurement work has calibrated the hand-written kernels.
*   **Other models:** the model builder is written per checkpoint family. π0/π0.5 and
    GR00T builders would feed the "several models" roadmap step.
*   **S0 stand-in:** an MLP/transformer controller entry in the constants file and a
    small builder, when co-location moves from roadmap to an RQ.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Branches are named by content (owner's rule): Part 1 on `problem-definition`, Part 2 on
`model-and-compiler-explainer` (branched from Part 1). Each branch opens one pull request
to `main`. Commits use the repo's `[step] area: message` style.

### Candidate research questions (owner approves with this section)

All four are on `libero-2cam`. Baselines on the PR1 arch are measured; baselines on the
stage-1 arch are estimates until the measurement work runs. RQ3 is the one aimed at the
core research topic (spatial mapping).

| ID  | Question                                                                                                                                                                                     | Metric                                  | Baseline (constants key, label)                                                                                                      |
|-----|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| RQ1 | How much of the VLA's latency is row operations (softmax, norm, activation), and how much is recovered by running them on each cluster's Spatz unit (which the compiler does not use yet)? | row-operation share of latency; latency | `pr1.rowop_share` 64.5 % (measured, PR1 arch, `pr1-1cam`); `pr1.libero_softmax_share` ~44 % (estimate, PR1 arch)                     |
| RQ2 | For the bandwidth-bound action expert, how does time per flow step depend on HBM bandwidth and on keeping the KV cache and per-layer weight tiles on chip?                                   | ms per flow step; achieved GB/s         | `pr1.expert_step_ms` (measured, PR1 arch; value taken from the PR1 profile); FLOP per weight byte from the model (inference)         |
| RQ3 | Given that vision, prefix and expert differ in FLOP per weight byte, does dividing the cluster mesh into regions per stage (pipelined across calls) beat running each stage on the whole mesh? | end-to-end latency; cluster utilization | `pr1.gemm_cluster_use` vision/prefix products use 1-10 of 16 clusters (measured, PR1 arch); `pr1.e2e_ms` 120.7 ms                    |
| RQ4 | At iso-peak FP16 with Thor, what fraction of peak does SoftHier reach per stage, against Thor's per-stage figures?                                                                           | per-stage MFU on SoftHier vs Thor       | `pr1.mfu` 1.2 % of 131 TFLOP/s (measured, PR1 arch); Thor per-stage MFU from `wiki://knowledge/mfu-from-measurements.md` (inference) |

### Target Files to Create/Modify

#### Part 1: problem definition (branch `problem-definition`)

##### 1. `workload-modeling/pyproject.toml` (Create)

*   **Type / Scope**: uv project configuration.
*   **Proposed Changes**: package `workload_modeling`, Python 3.12, dependencies
    `pyyaml`, `jinja2` (Markdown templates) and `markdown` (Markdown to HTML), and
    `pytest` as a dev dependency. No torch,
    xDSL or LeRobot. Same uv conventions as `infra/lerobot/pyproject.toml`.

##### 2. `workload-modeling/uv.lock` (Create, generated by `uv lock`)

*   **Type / Scope**: lock file.
*   **Proposed Changes**: pins dependencies so collaborators get the same environment.

##### 3. `workload-modeling/README.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: what the project generates, the build command, how to refresh
    fixtures, how to run the cross-check, and the evidence-label rule.

##### 4. `workload-modeling/constants.yaml` (Create)

*   **Type / Scope**: data, single source of shared numbers.
*   **Proposed Changes**: the groups `workloads`, `arch.stage1`, `arch.pr1`, `thor`,
    `pr1` as described in Technical Plan §1. Schema per entry:
    `{value, unit, label, source, note?}`. The PR1 profile values (`rowop_share`,
    `libero_softmax_share`, `expert_step_ms`, `gemm_cluster_use`, `mfu`, `e2e_ms`) are
    copied from `experiments/smolvla-e2e-baseline/PROFILE.md` on branch
    `worktree-smolvla-e2e-baseline` with that commit as source.
*   **Arch entries:**
    *   `arch.stage1` copies the parameters of `~/softhier/implementation/config/arch/arch.py`
        with its commit.
    *   `arch.pr1` copies `experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py`.
*   **Workload entries:**
    *   `pr1-1cam` uses `lerobot/smolvla_base`.
    *   `libero-2cam` uses `lerobot/smolvla_libero`.
    *   `release-3cam` uses `lerobot/smolvla_base` with 3 cameras at 512x512.

##### 5. `workload-modeling/workload_modeling/__init__.py` (Create)

*   **Type / Scope**: package marker.
*   **Proposed Changes**: empty except the version string.

##### 6. `workload-modeling/workload_modeling/constants.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `load(path) -> Constants`; `Constants.get(key) -> Entry`;
    validation that every entry has `value`, `unit`, `label` in
    {measured, estimate, inference, derived} and a non-empty `source`. Derived arch
    figures (`peak_per_cluster`, `peak_total(clusters)`) are computed here from arch
    parameters, never stored.

##### 7. `workload-modeling/workload_modeling/hf_header.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `fetch(repo_id, revision) -> (config: dict, tensors: dict[name, shape, dtype])`.
    It reads `config.json` and the safetensors header (8-byte length prefix plus JSON)
    through HTTP range requests using the standard library, with no weights downloaded.
    `--refresh-fixtures` writes them to `tests/fixtures/<repo>@<revision>/`. Normal runs
    read only the fixtures.

##### 8. `workload-modeling/workload_modeling/smolvla_ops.py` (Create; stage totals in Part 1, per-operator detail completed in Part 2)

*   **Type / Scope**: Python module, the standalone analytical model.
*   **Proposed Changes**: `build(workload_id, constants, fixtures) -> list[OpRow]`, with
    the `OpRow` fields from Technical Plan §2.
    *   Shapes come from the config (hidden sizes, heads, KV heads, FFN size, layer
        counts, the prefix-layer truncation SmolVLA uses) and are checked against
        tensor shapes in the header.
    *   Stages: vision (SigLIP), connector (pixel shuffle + projection), prefix (VLM
        layers over image, language and state tokens), expert (self- and cross-attention
        layers over the action chunk, per flow step).
    *   `stage_totals(rows)` returns FLOP, unique bytes, weight bytes and FLOP per weight
        byte per stage. Part 1 needs only these totals.

##### 9. `workload-modeling/workload_modeling/hw_table.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: computes the Thor-matched cluster count and fills the table
    rows described in Technical Plan §5:
    *   cluster count: the smallest square mesh with `peak_total >= thor.fp16_dense`;
    *   HBM capacity: the next power of two at or above weight bytes plus the largest
        activation footprint for `libero-2cam`;
    *   total TCDM.

    It writes the block between `<!-- BEGIN GENERATED hw-table -->` and
    `<!-- END GENERATED hw-table -->` in `docs/analysis/hw-config.md`.

##### 10. `workload-modeling/workload_modeling/rq_index.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: reads the front-matter of `docs/research-plan/rq/rq*.md`,
    resolves `baseline` keys in the constants and writes the RQ index table between
    marker comments in `docs/research-plan/README.md`. It fails on a missing key, a
    duplicate id, or an unknown `status`.

##### 11. `workload-modeling/workload_modeling/build.py` (Create)

*   **Type / Scope**: command-line entry point.
*   **Proposed Changes**: `python -m workload_modeling.build [--check]` runs the
    constants validation, the model, the HW table and the RQ index (and in Part 2 the
    renderer). `--check` regenerates in memory and fails if any committed generated
    file differs, which catches hand edits to generated blocks.

##### 12. `workload-modeling/tests/fixtures/` (Create)

*   **Type / Scope**: test data.
*   **Proposed Changes**: cached `config.json` and safetensors header JSON for
    `lerobot/smolvla_base` and `lerobot/smolvla_libero`, each at a pinned revision.

##### 13. `workload-modeling/tests/test_constants.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   rejects an entry without a label or source;
    *   the stage-1 peak at 16 clusters is ~16.4 TFLOP/s;
    *   the PR1 peak is ~131 TFLOP/s.

##### 14. `workload-modeling/tests/test_smolvla_ops.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   the sum of weight shapes used by the model plus the listed unused tensors
        equals the header's parameter count (450.0M for `smolvla_base`);
    *   every weight tensor in the header is used or listed as unused;
    *   attention rows carry `class = attn`;
    *   expert `count` equals the flow steps;
    *   `release-3cam` stage totals are within 5 % of the earlier estimate (~834 GFLOP
        total), as a regression guard rather than a truth claim.

##### 15. `workload-modeling/tests/test_hw_table.py` and `tests/test_rq_index.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   with Thor at 258 TFLOP/s the cluster count is 256 (16x16);
    *   the RQ index fails on a missing baseline key;
    *   the generated block replaces only the text between the markers.

##### 16. `docs/analysis/conventions.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: the content listed in Technical Plan §7. The FLOP and byte
    formula per operator class is written out (matrix product, attention parts, norm,
    activation, softmax, elementwise).

##### 17. `docs/analysis/hw-config.md` (Create)

*   **Type / Scope**: documentation with one generated block.
*   **Proposed Changes**: prose on the alignment rule, which parameters are fixed or
    open, Spatz reported separately, and the "proposal, not validated" statement; plus
    the generated table.

##### 18. `docs/research-plan/README.md` (Create)

*   **Type / Scope**: documentation, entry point.
*   **Proposed Changes**: the content listed in Technical Plan §6, plus the generated
    RQ index block. Links to the wiki status log are written as `wiki://` URIs with the
    note that they resolve after the wiki branch merge.

##### 19. `docs/research-plan/rq/_template.md` and `rq/rq1-row-operations.md`, `rq/rq2-expert-streaming.md`, `rq/rq3-mesh-partitioning.md`, `rq/rq4-mfu-vs-thor.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: one file per approved RQ, with front-matter as in Technical
    Plan §6 and the body sections question, why it matters, method, and what result
    would answer it. RQ texts come from the table above as approved.

##### 20. `docs/research-plan/working/pattern-table.md` and `working/pattern-table-meeting.md` (Create, moved from untracked files in the main checkout)

*   **Type / Scope**: working material.
*   **Proposed Changes**: copied unchanged from `workload-modeling/pattern-table.md` and
    `workload-modeling/pattern-table-meeting.md` in the main checkout (owner's untracked
    files). A header line marks them as idea-forming material, not decisions.

##### 21. `README.md` (Modify)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: add a "Start here" link to `docs/research-plan/README.md`
    under the goals. The existing line-34 link to the first-baseline decision stays (it
    resolves after the wiki merge).

##### 22. `.gitignore` (Modify)

*   **Type / Scope**: configuration.
*   **Proposed Changes**: ignore `workload-modeling/build/` and
    `workload-modeling/.venv/`.

##### 23. `softhier_vla_wiki` (Modify submodule pointer, after the owner merges `wiki-status-patches`)

*   **Type / Scope**: submodule bump.
*   **Proposed Changes**: a separate `[wiki] bump submodule` commit on
    `problem-definition`, so the status-log and naming links resolve. It points at the
    wiki commit that also contains file 23a.

##### 23a. `softhier_vla_wiki/memories/status/analyzing-the-needs-of-one-model-smolvla.md` and `memories/status/_overview.md` (Create / Modify, in the wiki submodule)

*   **Type / Scope**: wiki status entry, written on a wiki branch based on the merged
    `wiki-status-patches`.
*   **Proposed Changes**: the entry from Technical Plan §9, created from
    `tools/templates/status.md`:
    *   front-matter `type: status`, `evidence:` list, and `follows:
        basic-integration-of-components`;
    *   a table with `Label` and `Source` columns, each row linked to a commit-pinned
        file on `problem-definition`.

    `_overview.md` adds the entry to the catalog after "Basic integration of components"
    and updates the current-state paragraph. Committed with the wiki's commit
    conventions after `tools/wiki.py check` passes. The owner merges the wiki branch.

##### 24. `CLAUDE.md` (Modify, only with the owner's explicit approval)

*   **Type / Scope**: project instructions.
*   **Proposed Changes**: update the "Wiki scope limit" line (line 14) to mention the
    exception for `memories/status/` and `memories/patches/` from wiki-design decision
    `2026-11-12-status-and-patch-logs`, as suggested by the wiki session.

#### Part 2: model and compiler explainer (branch `model-and-compiler-explainer`)

##### 25. `workload-modeling/workload_modeling/smolvla_ops.py` (Modify)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: complete per-operator rows (all columns, including
    `unit_today` and `hand_kernel` with the `~/softhier/implementation` commit).
    Collapse identical layers into one row with a count.

##### 26. `workload-modeling/workload_modeling/render.py`, `workload_modeling/md_to_html.py` and `templates/op_graph.md.j2` (Create)

*   **Type / Scope**: Python modules and one template.
*   **Proposed Changes**:
    *   `render.py` writes `docs/analysis/smolvla-op-graph.csv` and `.md` (Mermaid stage
        diagram, per-stage tables, totals, limitations paragraph) from the single
        template.
    *   `md_to_html.py` converts that Markdown to
        `workload-modeling/build/smolvla-op-graph.html`: Python-Markdown with the tables
        and fenced-code extensions, inline CSS with light and dark themes, and Mermaid
        fences turned into `<pre class="mermaid">` elements drawn by mermaid.js from
        jsDelivr at a pinned version.
    *   `build.py` runs `md_to_html.py` after `render.py`.

##### 27. `docs/analysis/smolvla-op-graph.md` and `docs/analysis/smolvla-op-graph.csv` (Create, generated)

*   **Type / Scope**: generated documentation and data.
*   **Proposed Changes**: committed outputs for `libero-2cam`, with a header saying they
    are generated and by which command.

##### 28. `workload-modeling/workload_modeling/crosscheck_softhier_mlir.py` (Create)

*   **Type / Scope**: Python script, run by hand.
*   **Proposed Changes**: takes `--softhier-mlir <path>`, runs its frontends for
    `pr1-1cam` in that checkout's environment (subprocess, so this project does not
    import xDSL), parses the `dse.workload` output, and compares per-stage
    multiply-accumulate counts and shapes with `smolvla_ops`. It writes
    `docs/analysis/crosscheck.md` with the softhier-mlir commit. It exits non-zero if
    any stage differs by more than 1 %.

##### 29. `docs/analysis/crosscheck.md` (Create, generated once)

*   **Type / Scope**: generated documentation.
*   **Proposed Changes**: the per-stage comparison and explanations for any difference.

##### 30. `docs/analysis/compiler-walkthrough.md` (Create)

*   **Type / Scope**: documentation.
*   **Proposed Changes**: the content listed in Technical Plan §8. Every claim about
    code cites a file and line at a pinned softhier-mlir or `~/softhier` commit.

##### 31. `workload-modeling/tests/test_render.py` (Create)

*   **Type / Scope**: unit tests.
*   **Proposed Changes**:
    *   stage totals in the Markdown equal `stage_totals`;
    *   every row has a workload ID and label in the page header;
    *   the HTML contains every table of the Markdown;
    *   its only external URL is the pinned mermaid.js script.

##### 32. `softhier_vla_wiki/memories/status/analyzing-the-needs-of-one-model-smolvla.md` (Modify, in the wiki submodule) and `softhier_vla_wiki` (submodule bump)

*   **Type / Scope**: wiki status entry update.
*   **Proposed Changes**: add rows for the operator table, the cross-check result and
    the compiler walk-through, linked to commits on `model-and-compiler-explainer`,
    with evidence labels. Same wiki workflow as file 23a, then a submodule bump in the
    explainer pull request (merged with a merge commit).

### Verification Plan

*   **Build**: `cd workload-modeling && uv sync && uv run python -m workload_modeling.build`
    (regenerates all outputs offline from fixtures).
*   **Drift check**: `uv run python -m workload_modeling.build --check` passes on a
    clean tree; run before each commit.
*   **Unit tests**: `uv run pytest` (files 13-15 in Part 1, plus file 31 in Part 2).
*   **Cross-check (Part 2, once)**: `uv run python -m workload_modeling.crosscheck_softhier_mlir --softhier-mlir ~/softhier-mlir`
    exits 0, or `crosscheck.md` explains every difference above 1 %.
*   **Manual review**:
    *   the owner reads `docs/research-plan/README.md` as a newcomer would and follows
        the reading order;
    *   the HTML page is opened in a browser in light and dark modes;
    *   one RQ is added from the template (CUJ 2) to confirm the index regenerates.
*   **Wiki check**: `python3 softhier_vla_wiki/tools/wiki.py check` passes on the wiki
    branch that holds the status entry (files 23a and 32).
*   **Wiki**: after the submodule bump, the `wiki://` links in the README resolve to
    existing files.

--------------------------------------------------------------------------------

*Drafted with EGM*
