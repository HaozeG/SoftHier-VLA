# SoftHier-VLA research plan

Start here. This page holds the problem statement, the research questions (RQs), the roadmap and a
reading order. Each RQ has its own file in [`rq/`](rq/); add one by copying
[`rq/_template.md`](rq/_template.md) and adding a row to the index below in the same pull request.
Material that is still idea-forming, not decided, goes in [`working/`](working/).

## Problem statement (draft, stage 1)

Robots that run vision-language-action policies on board need a chip that serves two kinds of work at
once. One is a large model at a few hertz (vision, a language-model prefix, and an action expert run
once per flow step). The other is a small controller (S0) at hundreds of hertz. Today's edge devices
(Jetson Thor) are GPU-style chips, and VLA benchmarks on them report low utilization at batch size 1:
most stages are limited by memory bandwidth or by software overheads, not by compute. Tile-based
many-core chips keep data in distributed local memories and can split their cluster mesh between
tasks. Whether this suits embodied AI is open. SoftHier-VLA asks how such a chip should be built and
how work should be mapped onto it in space (which clusters) and time (when), at the same peak FP16
compute as Thor. Stage 1 answers this for one model, SmolVLA, on one workload (`libero-2cam`). It
measures where time goes (row operations on the cores, the bandwidth-bound action expert, under-used
clusters) and states four questions with starting points. Later stages extend the analysis to several
models and to scheduling a VLA and a controller on one chip.

## Stage-1 exit criterion

1. The owner has approved the problem statement and RQ1-RQ4.
2. Every RQ names its arch, at least one labelled starting point with its workload and arch, and the
   measurement that would answer it.
3. The stage totals the RQs cite have passed the cross-check ([crosscheck.md](../analysis/crosscheck.md)).
4. The wiki status entry "Analyzing the needs of one model: SmolVLA" records these results.

No RQ needs to be answered in stage 1.

## RQ index

All RQs target `libero-2cam`. "16-cluster" = the kernel-repo arch as configured; "256-cluster" = the
same arch scaled to Thor's peak ([hw-config.md](../analysis/hw-config.md)). Starting points are from
PR1 (`arch: pr1`) unless marked; measured values on the stage-1 arches come from the measurement work.

| ID | Title | Metric | Arch | Starting points (key: value, workload, label) | Status |
| --- | --- | --- | --- | --- | --- |
| [RQ1](rq/rq1-core-ops-on-spatz.md) | Core-executed operations and Spatz | share of latency in core-executed ops; latency per stage | 16-cluster | `pr1.core_op_share` 64.5 %, pr1-1cam, measured; `pr1.libero_rowop_share` ~60 %, libero-2cam, estimate | draft |
| [RQ2](rq/rq2-expert-streaming.md) | Expert time per flow step against bandwidth and on-chip residency | ms per flow step; achieved GB/s | 256-cluster | `pr1.expert_step_ms` 7.761 ms, pr1-1cam, measured; expert 55 FLOP per weight byte, libero-2cam, inference (model) | draft |
| [RQ3](rq/rq3-mesh-partitioning.md) | Dividing the mesh by model stage | end-to-end latency; cluster utilization | 256-cluster (16 for contrast) | `pr1.gemm_cluster_use` 1-10 of 16, pr1-1cam, measured; `pr1.libero_e2e_ms` 479.1 ms, libero-2cam, estimate | draft |
| [RQ4](rq/rq4-mfu-vs-thor.md) | Per-stage MFU at iso-peak with Thor | achieved FLOP/s over peak FP16, per stage | 256-cluster | `pr1.mfu` 1.2 %, pr1-1cam, measured; Thor per-stage figures converted to FP16, inference | draft |

Status values: draft | active | answered | dropped.

## Roadmap

1. **Stage 1 (now):** infrastructure (PR1, done) and the needs of one model, SmolVLA: this plan, the
   constants, the stage totals, the hardware table; then the operator table and the compiler
   walk-through.
2. **Measurement work:** hand-written kernels on `libero-2cam` shapes on the stage-1 arch, a
   bandwidth-limited HBM model, a build check of the 256-cluster arch; fusion estimates.
3. **Several models:** π0/π0.5 and GR00T builders in the same model; static mapping per model.
4. **Dynamic scheduling:** a VLA and an S0 controller on one chip, under different latency and rate
   needs.

## Reading order

1. [Conventions](../analysis/conventions.md): workloads, labels, byte columns, Thor alignment.
2. [Hardware parameters](../analysis/hw-config.md).
3. [Stage totals](../analysis/stage-totals.md) and [cross-check](../analysis/crosscheck.md).
4. Operator graph and compiler walk-through (Part 2 of the stage-1 design).
5. Wiki status log: `softhier_vla_wiki/memories/status/`.

Design: [`docs/designs/stage1-problem-definition.md`](../designs/stage1-problem-definition.md).
