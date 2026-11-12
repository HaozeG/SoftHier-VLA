---
id: RQ3
title: Dividing the mesh by model stage
status: draft
workload: libero-2cam
arch: stage1-256
metric: end-to-end latency (ms); cluster utilization (%)
depends_on: [RQ1, RQ2]
---

# RQ3: Dividing the mesh by model stage

## Question

Given that vision, prefix and expert differ in FLOP per weight byte, is end-to-end latency lower when
the mesh is divided into regions per model stage (pipelined across calls) than when each stage uses the
whole mesh?

## Why it matters

This is the project's core topic, spatial mapping. The stages differ by about 45x in FLOP per weight
byte (vision 2,489, prefix 183, expert 55 on `libero-2cam`), and PR1 leaves most clusters idle in
vision and prefix. A region per stage is also the first step toward placing a controller (S0) next to
the VLA on one chip.

## Starting points

* `pr1.gemm_cluster_use`: vision and prefix matrix products use 1-10 of 16 clusters, `pr1-1cam`, PR1
  arch, measured.
* `pr1.e2e_ms`: 120.7 ms, `pr1-1cam`, PR1 arch, measured.
* `pr1.libero_e2e_ms`: 479.1 ms, `libero-2cam`, PR1 arch, estimate.
* Stage totals and FLOP per weight byte, `libero-2cam`, inference, cross-checked
  ([stage-totals.md](../../analysis/stage-totals.md)).

## Method

Map the four model stages to cluster regions sized by their work, with consecutive calls pipelined,
and compare with whole-mesh execution of each stage in turn, on the 256-cluster arch (16-cluster for
contrast). Inputs: per-stage times from RQ1 and RQ2.

## What result would answer it

End-to-end latency and per-region utilization for the partitioned and the whole-mesh mappings, on the
same arch and workload.
