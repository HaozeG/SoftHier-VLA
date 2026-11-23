# kernel-runs

Stage-1 hand-written kernel runs on H1 (`arch/arch_h1.py`), following the measurement protocol of
[`docs/designs/stage1-spec.md`](../../docs/designs/stage1-spec.md) §4.3. Each run builds the kernel from
the kernel-repo worktree (`KREPO`, default `~/softhier/.claude/worktrees/gelu-layernorm`) and runs it once on
the patched GVSoC tree (`infra/softhier/require_sim.sh`: stamp check, tree lock, `MALLOC_PERTURB_`), one run at
a time. Run folders live under `~/.cache/softhier-vla/stage1-h1/` (outside git); this folder keeps the
drivers and the results tables.

| File | What it does |
| --- | --- |
| `run_kernel.sh <kernel> <cfg.py> <out> [--trace] [--hbm]` | one build and run; out folder gets the config, arch, `sim_stamp.json`, headers, logs, optional Perfetto trace |
| `gemm_sweep.py plan\|run\|pick` | SummaGEMM tile, scale and split sweep per W1 / W1-b shape -> `results/gemm_sweep_h1.csv` |
| `gemm_smallm_sweep.py plan\|run\|pick` | M = 50 expert GEMMs: groups 1 or 2 clusters tall, splitN or splitK (structural 0.02 check) -> `results/gemm_smallm_h1.csv` |
| `row_kernels.py plan\|run\|summary` | Eltwise, Activation, LayerNorm, RMSNorm: one run per shape -> `results/row_kernels_h1.csv` |
| `attn_sweep.py plan\|run\|pick` | FlatAttention group, block and mode sweep (padded, unmasked) -> `results/attn_sweep_h1.csv` |
| `attn_debug.py [case ...]` | FlatAttention correctness bisection (one kernel feature per case); debug only |
| `final_runs.py` | re-runs every picked GEMM setting (fastest over both GEMM sweeps) and row kernel -> `results/final_h1.csv` |
| `trace_picks.py` | Perfetto traces of the picked settings -> `results/traces_h1.csv` |
| `to_constants.py [--print]` | writes the `h1_kernels` section of `infra/workload-modeling/constants.yaml` |
| `hbm_channel_count.py` | requests per HBM channel from an `hbm_ctrl` trace |

Recorded values come from `final_h1.csv` (GEMM, row kernels) and `attn_sweep_h1.csv` (FlatAttention), on one
simulator stamp (patches 0001-0012) and the frozen kernel build (worktree diff kept as
`~/.cache/softhier-vla/stage1-h1/kernel_build.diff`, hash in each row). The nine M = 50 expert GEMM picks from
`gemm_smallm_sweep.py` are recorded on kernel commit `de53537` (build column `de53537`): the frozen build plus
an optional absolute-tolerance GEMM check, the same binary for configs that do not set it (re-run of a
frozen-build pick: 32,353 ns, unchanged); the splitK picks set it. Sweep candidates may span builds (spec §4.3
item 11).
