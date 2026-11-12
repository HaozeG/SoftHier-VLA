---
id: RQ2
title: Expert time per flow step against bandwidth and on-chip residency
status: draft
workload: libero-2cam
arch: stage1-256
metric: ms per flow step; achieved HBM GB/s
depends_on: []
---

# RQ2: Expert time per flow step against bandwidth and on-chip residency

## Question

For the action expert, how does time per flow step depend on HBM bandwidth, and on keeping the KV
cache and per-layer weight tiles on chip?

## Why it matters

The expert runs 10 times per call with 50-row matrix products, so it streams its weights every step.
Its FLOP per weight byte (55) is far below any machine balance (Thor 945), so bandwidth and what stays
on chip set its time, and the two differ in cost.

## Starting points

* `pr1.expert_step_ms`: 7.761 ms per step (step 5), `pr1-1cam`, PR1 arch, ideal HBM, measured.
* `pr1.expert_hbm_read_per_step`: 335 MB of HBM reads per step, `pr1-1cam`, PR1 arch, ideal HBM, measured.
* `pr1.expert_weight_bytes_per_step`: about 193 MB, `pr1-1cam`, estimate (from op shapes).
* Standalone model: expert 11.05 GFLOP and 199.7 MB of weights per step, 55 FLOP per weight byte,
  `libero-2cam`, inference, cross-checked ([stage-totals.md](../../analysis/stage-totals.md)).

## Method

Expert layers with the SUMMA kernel and the KV cache kept in TCDM on the 256-cluster arch (or the
16-cluster arch if the larger build fails), simulated with a bandwidth-limited HBM model: first at the
provisional 273 GB/s (Thor's, inference), then swept.

## What result would answer it

Time per flow step against HBM bandwidth, with and without KV-cache residency, and the achieved
bandwidth as a fraction of the configured one.
