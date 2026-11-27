# kernel-runs

Stage-1 hand-written kernel runs on H1 (`arch/arch_h1.py`), following the measurement protocol of
[`docs/designs/stage1-spec.md`](../../docs/designs/stage1-spec.md) §4.3. Each run builds the kernel from
the kernel source (`KREPO`, default the submodule `third_party/softhier-kernels`, see below) and runs it once on
the patched GVSoC tree (`infra/softhier/require_sim.sh`: stamp check, tree lock, `MALLOC_PERTURB_`), one run at
a time. Run folders live under `~/.cache/softhier-vla/stage1-h1/` (outside git); this folder keeps the
drivers and the results tables.

## Reproduce the stage-1 numbers

One path from a fresh clone to the 51 kernel times in `constants.yaml` (`h1_kernels`), which the operator-graph
pages use. Host: Linux x86_64, prerequisites as in [`infra/softhier/README.md`](../softhier/README.md) (apt
packages, gcc/g++ >= 11.2, cmake >= 3.18.1, [`uv`](https://docs.astral.sh/uv/)); about 6 GB disk, 19 GB RAM.

```bash
git clone https://github.com/HaozeG/SoftHier-VLA.git && cd SoftHier-VLA
git submodule update --init third_party/softhier-kernels  # kernel source (below); not --recursive: third_party/gvsoc
                                                          # is set up by setup.sh, the wiki submodule uses an SSH URL
source infra/env.sh                    # SHVLA_DATA (default ~/.cache/softhier-vla), SOFTHIER_HOME; `bash infra/env.sh --show`
bash infra/softhier/setup.sh           # once, ~30 min: clones upstream GVSoC soft_hier_release d806a84 into
                                       # $SHVLA_DATA/gvsoc, applies infra/softhier/gvsoc-patches/ (0001-0013),
                                       # builds SystemC, DRAMSys and the RISC-V toolchain; uv venv in infra/softhier/.venv
bash infra/softhier/build_arch.sh infra/kernel-runs/arch/arch_h1.py   # simulator for H1, ~2 min; writes the stamp
python3 infra/kernel-runs/reproduce.py run --max-ns 5000   # spot check, a few minutes
python3 infra/kernel-runs/reproduce.py run                 # all 51 runs, serial; exits 1 unless all are identical
```

*   Reproduced 2026-11-24: 51/51 identical (ns, output checks passed), kernel submodule at `de53537`, GVSoC tree
    `977e08c` (patches 0001-0013; the recorded values were taken on `058a39b`, patches 0001-0012). The simulator tree
    was one already prepared by `setup.sh` on the owner's host, not a fresh install.
*   `reproduce.py` re-runs the recorded settings committed in `configs/h1/` (`recorded.csv`: run name, kernel,
    recorded ns, kernel build, `constants.yaml` keys) and compares the simulated time exactly; the simulation is
    deterministic. Run folders: `$SHVLA_DATA/stage1-h1/repro/<name>/`. `list` shows the runs; `--only`,
    `--kernel`, `--max-ns` select some.
*   Every run refuses a simulator whose stamp does not match the patch series and `arch_h1.py`
    (`infra/softhier/require_sim.sh`). Building for another arch (for example `run_kernel_tests.sh`, which builds
    `arch.py`) invalidates it; run `build_arch.sh` for `arch_h1.py` again.
*   Each git worktree needs its own venv: `(cd infra/softhier && uv sync)`. The simulator tree in `$SHVLA_DATA/gvsoc`
    is shared by all worktrees, under a lock.
*   Owner-only tools: the sweeps and `final_runs.py` read earlier run folders under the owner's `$SHVLA_DATA`;
    `reproduce.py export` regenerates `configs/h1/` from them after a pick changes.

## Kernel source

The hand-written kernels (SummaGEMM, FlatAttention, RMSNorm, LayerNorm, Activation, Eltwise) are in
`implementation/` of the GVSoC fork [`HaozeG/gvsoc`](https://github.com/HaozeG/gvsoc/tree/gelu-layernorm-kernels/implementation),
branch `gelu-layernorm-kernels`. This repo pins it as the submodule `third_party/softhier-kernels`, currently at
`de53537` (the stage-1 kernels: frozen build `aa57690` plus the optional `GEMM_CHECK_ABS_TOL` check).

```bash
git submodule update --init third_party/softhier-kernels   # shallow; its own submodules are not needed
# to use newer kernels: commit them on gelu-layernorm-kernels, push, then
git submodule update --remote third_party/softhier-kernels && git add third_party/softhier-kernels
```

Only `implementation/` is used, against the patched simulator tree (`$SOFTHIER_HOME`); the fork's own simulator
code is not built. Builds write generated headers and `preload.elf` into the submodule's tree (ignored as dirty).
Set `KREPO` to use another checkout. The arch file comes from this folder (`arch/arch_h1.py`), not the kernel
repo.

| File | What it does |
| --- | --- |
| `run_kernel.sh <kernel> <cfg.py> <out> [--trace] [--hbm]` | one build and run; out folder gets the config, arch, `sim_stamp.json`, headers, logs, optional Perfetto trace |
| `gemm_sweep.py plan\|run\|pick` | SummaGEMM tile, scale and split sweep per W1 / W1-b shape, ranked by the shared model `workload_modeling.kernel_model` (parameters from `constants.yaml` arch.h1) -> `results/gemm_sweep_h1.csv` |
| `gemm_smallm_sweep.py plan\|run\|pick` | M = 50 expert GEMMs: groups 1 or 2 clusters tall, splitN or splitK (structural 0.02 check) -> `results/gemm_smallm_h1.csv` |
| `row_kernels.py plan\|run\|summary` | Eltwise, Activation, LayerNorm, RMSNorm: one run per shape -> `results/row_kernels_h1.csv` |
| `attn_sweep.py plan\|run\|pick` | FlatAttention group, block and mode sweep (padded, unmasked), plus upstream `attn_auto.py` `opt()`'s setting per shape -> `results/attn_sweep_h1.csv` |
| `attn_debug.py [case ...]` | FlatAttention correctness bisection (one kernel feature per case); debug only |
| `gemm_transpose.py plan\|run\|pick [--shape MxNxK]` | transposed GEMM (Z^T = W^T X^T run as N x M x K) with gemm_sweep's candidates -> `results/gemm_transpose_h1.csv`; run for the shapes whose stage-1 pick loses RedMulE block fill |
| `gemm_upstream.py plan\|run\|pick [--shape MxNxK]` | upstream's own settings: soft_hier_je `gemm_auto.py` `opt()` ported rule for rule (splitK and splitN, padded where it asserts), on every GEMM shape -> `results/gemm_upstream_h1.csv` |
| `eltwise_pipeline.py [NAME ...]` | the 17 Eltwise configs with `pipeline = 1` (pipelined row schedule) -> `results/eltwise_pipeline_h1.csv` |
| `wrong_runs.py` | every run whose output check failed, with the splitK correctness rule tested against all splitK runs -> `docs/analysis/wrong-output-runs-h1.md` |
| `final_runs.py [--plan]` | the W1 preset: per GEMM shape and row kernel the fastest checked run over every result set (sweeps, upstream settings, transposed GEMM, pipelined Eltwise), re-run once -> `results/final_h1.csv` (stage-1 picks frozen in `results/final_h1_stage1.csv`) |
| `trace_picks.py` | Perfetto traces of the picked settings -> `results/traces_h1.csv` |
| `reproduce.py list\|run\|export` | re-runs the 51 recorded settings (`configs/h1/`) and compares with the recorded times; `export` (owner) refreshes `configs/h1/` |
| `trace_summary.py` | per-cluster RedMulE / iDMA / Spatz busy fractions in the kernel window of each trace -> `results/trace_summary_h1.json` (read by the kernel mapping page) |
| `serve_traces.py [--port 9001]` | serves `infra/workload-modeling/build/kernel-explorer.html` with the traces and run folders on 127.0.0.1, so the page can open traces in Perfetto |
| `to_constants.py [--print]` | writes the `h1_kernels` section of `infra/workload-modeling/constants.yaml` |
| `hbm_channel_count.py` | requests per HBM channel from an `hbm_ctrl` trace |

Recorded values come from `final_h1.csv` (GEMM, row kernels) and `attn_sweep_h1.csv` (FlatAttention), on one
simulator stamp (patches 0001-0012) and the frozen kernel build (worktree diff kept as
`~/.cache/softhier-vla/stage1-h1/kernel_build.diff`, hash in each row). The nine M = 50 expert GEMM picks from
`gemm_smallm_sweep.py` are recorded on kernel commit `de53537` (build column `de53537`): the frozen build plus
an optional absolute-tolerance GEMM check, the same binary for configs that do not set it (re-run of a
frozen-build pick: 32,353 ns, unchanged); the splitK picks set it. Sweep candidates may span builds (spec §4.3
item 11).
