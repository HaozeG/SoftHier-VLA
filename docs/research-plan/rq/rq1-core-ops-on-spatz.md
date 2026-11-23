---
id: RQ1
title: Core-executed operations and Spatz
status: draft
workload: vlabench-3cam
arch: stage1-16
metric: share of latency in core-executed ops (%); latency per stage (ms)
depends_on: []
---

# RQ1: Core-executed operations and Spatz

## Question

What share of the VLA's latency is spent in operations executed on the cores (elementwise, norm,
softmax), and how much of it goes away when they run on each cluster's Spatz vector unit?

## Why it matters

In PR1 these operations, not matrix products, set most of the run time: the cores issue about one
instruction per cycle per cluster and RedMulE stays idle. If Spatz removes most of this time, the
matrix engine and the memory system become the limits and the later RQs change; if it does not, the
cluster needs a different row-operation path.

## Starting points

* `pr1.core_op_share`: 64.5 %, `pr1-1cam`, PR1 arch, measured (elementwise 26.3 + attention 25.0 +
  norm 12.9; "attention" includes per-head QK/PV staging and softmax, "elementwise" includes RoPE,
  bias and residual adds).

It comes from PR1's arch, where Spatz was disabled because the compiler does not target it.

## Method

Run the kernel repo's hand-written row-operation kernels on `vlabench-3cam` shapes on the 16-cluster
arch. The kernels, their requirements and the measurement protocol are in
[stage1-spec.md §4](../../designs/stage1-spec.md#4-hand-written-kernel-requirements):

*   Eltwise (bias, residual add, RoPE, scale, axpy), Activation (GELU, SiLU, SiLU x gate) and LayerNorm
    run on Spatz;
*   FlatAttention runs softmax on Spatz inside the attention call;
*   RMSNorm is the upstream kernel on one core per cluster, so it is still a core-executed operation.

Compare per-op times with PR1's per-op times scaled to the same shapes, and recompute the share.

## What result would answer it

The share of core-executed time per stage on the 16-cluster arch with the Spatz kernels, next to the
same share without them, both measured on `vlabench-3cam`.
