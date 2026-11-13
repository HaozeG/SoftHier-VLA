---
id: RQ4
title: Per-stage MFU at iso-peak with Thor
status: draft
workload: libero-2cam
arch: stage1-256
metric: per-stage MFU (achieved FLOP/s over peak FP16)
depends_on: [RQ1, RQ2]
---

# RQ4: Per-stage MFU at iso-peak with Thor

## Question

At the same peak FP16 as Jetson Thor, what fraction of peak does SoftHier reach per model stage,
compared with Thor's per-stage figures?

## Why it matters

It places the tile-based chip against the edge device VLA papers measure on, per stage rather than as
one number, which separates where the architecture helps from where the software does.

## Starting points

* `pr1.mfu`: 1.2 % of 131.07 TFLOP/s, `pr1-1cam`, PR1 arch, measured.
* Thor per-stage figures from `wiki://knowledge/mfu-from-measurements.md`: π0.5 and GR00T N1.7 with
  FP8/NVFP4 kernels (VLA-HWBench), not SmolVLA at FP16. Converted to an FP16 comparison point by
  dividing each stage's achieved FLOP/s by `thor.fp16_dense` (258 TFLOP/s), labelled inference. FP8 and
  NVFP4 kernels reach more FLOP/s than FP16 kernels would, so the converted figures are upper bounds
  for an FP16 Thor run.

## Method

Per-stage simulated times on the 256-cluster arch (from RQ1 and RQ2 runs), divided into the model's
per-stage FLOP ([stage-totals.md](../../analysis/stage-totals.md)), over 262.1 TFLOP/s.

## What result would answer it

A per-stage table of SoftHier MFU next to Thor's converted per-stage figures, each with its label.
