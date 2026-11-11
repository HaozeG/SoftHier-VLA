# Reproducing the experiment from scratch

Everything here was run on one machine: WSL2 on x86_64, 20 cores, about 19 GB RAM (WSL limit around 16 GB usable), one CUDA GPU for LeRobot, 6 GB of disk for GVSoC plus the checkpoints and run folders under `$SHVLA_DATA`. Results: [RESULTS.md](RESULTS.md). Step-by-step trace: [LOG.md](LOG.md).

The two infra READMEs hold the details; this file gives the order and the experiment-specific parts.

*   [infra/softhier/README.md](../../infra/softhier/README.md): prerequisites, scripts, stamp, ideal HBM, kernel tests, measured times.
*   [infra/lerobot/README.md](../../infra/lerobot/README.md): LeRobot setup, memory cap, `golden_ref.py`, `libero_eval.py`, tests.

## 0. Before you start

1.  **GVSoC fork.** All ten GVSoC commits from `4558184` to `f99d5a2` (F17, F18, F13, F5, F14, F8, F15, F3, F1, F16 in RESULTS.md) are commits on a local branch `shvla/soft_hier_release` of `third_party/gvsoc` and are not pushed. `.gitmodules` names the upstream URL `https://github.com/gvsoc/gvsoc.git`, which does not contain them, so a fresh clone cannot check out the pinned commit `f99d5a2`. Until the fork exists (README "Switching the submodule to the fork (D2)" in `infra/softhier/README.md`), reproduce on the machine that has the branch, or apply all ten by hand from the commit list in RESULTS.md (leaving out F5, the ideal-HBM switch, makes the GEMMs about 46 % slow).
2.  **apt packages** (the owner installs them, `setup.sh` prints the exact command if one is missing):
    `build-essential git doxygen python3-pip libsdl2-dev curl cmake gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev wget bzip2 xz-utils`, plus `libosmesa6` and `libglfw3` for the LIBERO camera rendering, and gcc and g++ at least 11.2, cmake at least 3.18.1, `uv`.
3.  **Submodules.** `git submodule update --init third_party/gvsoc experiments/smolvla-e2e-baseline/softhier-mlir` (softhier-mlir is pinned at `b11cd5f`; `run.sh` applies the patches to its working tree, never commits there).
4.  **Hugging Face access** to `lerobot/smolvla_base` and `lerobot/smolvla_libero` (`hf auth login` if the network needs it).

## 1. Environment variables

`source infra/env.sh` sets defaults (a variable that is already set is kept) and then `infra/env.local.sh` if present (gitignored; copy `infra/env.local.sh.example`). `bash infra/env.sh --show` prints all of them.

| variable | default | meaning |
| --- | --- | --- |
| `SHVLA_ROOT` | worktree root | every script |
| `SHVLA_DATA` | `~/.cache/softhier-vla` | npz files, preload ELFs, run folders (`runs/<config>/<name>/`), `golden/`, `eval/`, logs |
| `HF_HOME` | Hugging Face default | checkpoints |
| `SHVLA_SMOLVLA_DIR` | set by `configs/<config>.env` | `$SHVLA_DATA/smolvla_base` or `smolvla_libero` |
| `SOFTHIER_HOME` | `$SHVLA_ROOT/third_party/gvsoc` | read by softhier-mlir |
| `SOFTHIER_CHROOT` | empty | non-empty only on a non-x86_64 host |
| `SHVLA_SIM_TIMEOUT` | `7200` | seconds per simulator process (softhier-mlir's `--timeout`, patch 0004; whole python process for other scripts) |
| `SHVLA_SIM_MEMMAX` | `13G` | `MemoryMax` of the systemd scope each `run.sh sim` runs in (no swap). The runs in RESULTS.md used `8G` |
| `SHVLA_PWR_INTERVAL_PS` | `9000000000000000` | power-capture interval compiled into the build; recorded in the stamp (see F18) |
| `SOFTHIER_MODEL_DIR`, `SOFTHIER_STOCK_MODELS` | unset | extra model folders; recorded in the stamp. Keep unset for these results |
| `SOFTHIER_IDEAL_HBM` | not exported | `run_sim` sets it to `1` at run time; only the local GVSoC branch reads it |
| `SHVLA_SIGLIP_H_DUMPS`, `_REGEMM`, `_FULL` | unset | patch 0006: opt-in diagnostic dumps for `siglip-mlir` (the 150000-point check used `SHVLA_SIGLIP_H_DUMPS=256 SHVLA_SIGLIP_H_FULL=150000`). With them set, ROI and marks are not comparable |

`SHVLA_MEMOUT` appears in `run.sh`; it is an internal variable (the output prefix for the memory-peak files), not a setting.

## 2. Simulator (infra/softhier)

```
source infra/env.sh
bash infra/softhier/setup.sh                    # about 31 min first time; skips when done
bash infra/softhier/build_arch.sh experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py   # about 100 s first, 5-10 s incremental
python3 infra/softhier/stamp.py check experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py   # prints "stamp OK"
```

*   **Optional G1 check:** `bash infra/softhier/run_kernel_tests.sh` builds with `implementation/config/arch/arch.py`, so it replaces the stamp. Rebuild with the experiment arch afterwards. The kernel tests ran on DRAMSys and print no result strings (RESULTS.md).
*   **The stamp** (`third_party/gvsoc/.shvla_stamp.json`) records the arch SHA-256, the GVSoC, `core` and `pulp` commits and diff hashes, `pwr_interval_ps` and the model-folder state. `run.sh sim` refuses to run (exit 2) when `stamp.py check` fails and never rebuilds by itself. After any change to the arch file or to the simulator sources, run `build_arch.sh` again, then `stamp.py check`. One build per worktree: runs that need a different arch run one after another.
*   `make softhier_preparation` rebuilds DRAMSys if its self-test fails. Do not accept that rebuild without checking `LD_LIBRARY_PATH` (setup.sh sets it): the rebuilt library does not match SystemC 2.3.3 (LOG G1.5).

## 3. Python environments

```
(cd infra/lerobot && uv sync)                                   # about 10 min, CUDA wheels
(cd experiments/smolvla-e2e-baseline && uv sync)                # softhier-mlir, CPU torch, transformers, GVSoC Python deps
```

`run.sh` needs `experiments/smolvla-e2e-baseline/.venv`. The experiment environment pins `setuptools<81` (GVSoC's reggen imports `pkg_resources`).

## 4. Run tool: `experiments/smolvla-e2e-baseline/run.sh`

```
run.sh apply                                    # reset softhier-mlir tracked files, apply patches/0001..0006 in order
run.sh <published|libero> fetch                 # hf download <checkpoint> at the pinned revision; link model.safetensors and config.json
run.sh <published|libero> prepare               # host prepare commands (the npz files); no simulator
run.sh <published|libero> sim <name> -- <script> <args...>
```

`sim` runs `python <script> <args>` from the softhier-mlir root inside `systemd-run --user --scope -p MemoryMax=$SHVLA_SIM_MEMMAX -p MemorySwapMax=0`, checks the stamp first, tees output to `$SHVLA_DATA/runs/<config>/<name>/run.log` and prints a record block (command, permalinks, stamp, patch hashes, exit code, peak memory, per-phase simulated and wall time). Calling again with the same run name resumes: finished phases of `smolvla-e2e` are reused; put `--fresh` after `--` to re-simulate them.

Exit codes: 0 pass, 1 fail, 2 refused (usage, stamp or patch), 3 stopped by the timeout, 4 killed at the memory cap.

**Memory and WSL.** Run one simulator job at a time. Peak cgroup memory (includes page cache) was 0.4-1.8 GB for the rungs and 2.6-2.8 GB for the full published chain; a runaway simulation (F13/F14 before they were fixed) grows about 300 MB/s, which is why the cap exists. Check `free -g` for at least 9-11 GB available before a run. On WSL with about 16 GB, do not run LeRobot (3 GB per process, several processes) next to simulators: a batch-5 evaluation was killed this way (LOG G6b.1). `libero_eval.py` has its own 12 GB cap (`--no-cap` removes it).

**Safe mode.** There is no switch of that name in the repo. In LOG.md "safe mode" is the working rule used for the risky diagnostic runs: no `perf`, `gdb` or `strace`, no tracing, one simulator at a time, `SHVLA_SIM_MEMMAX=8G`, and `free -g` showing at least 10 GB available before each run. Use it for any run whose memory behaviour is unknown.

## 5. Published configuration (G2, G3)

```
D=$SHVLA_DATA/smolvla_base
experiments/smolvla-e2e-baseline/run.sh apply
experiments/smolvla-e2e-baseline/run.sh published fetch
(source infra/lerobot/env.sh && python infra/lerobot/golden_ref.py --checkpoint lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
    --obs synthetic:1:256:0 --out $SHVLA_DATA/golden/published.npz)   # subshell: LD_PRELOAD and MUJOCO_GL must not leak into the simulator runs
experiments/smolvla-e2e-baseline/run.sh published prepare        # needs the golden reference
```

Both golden references were written on the GPU (default device, under the 12 GB `systemd-run` cap, see the lerobot README). G2.3 crosscheck: run softhier-mlir's `smolvla_expert_ref.py` by file path in the LeRobot environment, then `golden_ref.py --obs from-npz:... --crosscheck ...`; the commands are in LOG G2.3.

Ladder (`$D` = `$SHVLA_DATA/smolvla_base`, full path; each line is `run.sh published sim <name> -- tests/gvsoc/run.py ...`):

| rung | arguments | wall |
| --- | --- | --- |
| 1 | `gemm` | under 1 min |
| 2 | `mlir examples/*.mlir` (`siglip_encoder_layer` alone with `SHVLA_SIM_TIMEOUT=1200`, about 6 min) | about 6 min |
| c | `siglip-mlir --seq 256 --cluster all` | about 1 min |
| 3 | `smolvla --npz $D/vision_s256.npz --layers 1` | about 1 min |
| 4 | `smolvla-vlm --npz $D/vlm_c1.npz --layers 1` | about 30 s |
| 5 | `tests/gvsoc/expert.py flow --npz $D/expert.npz --steps 1` | about 3 min |
| 6 | `smolvla-e2e --npz $D/e2e_c1_t256.npz --e2e-trace` | 82 min, four processes |

Gate: `python3 experiments/smolvla-e2e-baseline/gate.py g3 $SHVLA_DATA/runs/published/<rung 6 run>`.

## 6. LIBERO configuration (G2.4, G4a)

```
experiments/smolvla-e2e-baseline/run.sh libero fetch
(source infra/lerobot/env.sh && python infra/lerobot/golden_ref.py --checkpoint lerobot/smolvla_libero --revision a24f1f380461b665158d52acd140152c79e8b36d \
    --obs libero:libero_spatial:0:0 --out $SHVLA_DATA/golden/libero.npz)
experiments/smolvla-e2e-baseline/run.sh libero prepare
export SHVLA_SIM_MEMMAX=8G
experiments/smolvla-e2e-baseline/run.sh libero sim G4a.1 -- tests/gvsoc/run.py smolvla-e2e --vision-only 1024 --layers 1 --e2e-trace
experiments/smolvla-e2e-baseline/run.sh libero sim G4a.2 -- tests/gvsoc/run.py smolvla-e2e --prefix-only 2 --layers 8 --e2e-trace
experiments/smolvla-e2e-baseline/run.sh libero sim G4a.3 -- tests/gvsoc/run.py smolvla-e2e --expert-only ref:$SHVLA_DATA/smolvla_libero/e2e_c2_t1024.npz --steps 2 --e2e-trace
python3 experiments/smolvla-e2e-baseline/gate.py g4a $SHVLA_DATA/runs/libero/G4a.1 $SHVLA_DATA/runs/libero/G4a.2 $SHVLA_DATA/runs/libero/G4a.3
```

Each G4a run takes about 10 min. `gate.py g4a` writes `g4a_estimate.json` and exits 1 if a G4a accuracy check misses.

**G4b (not run, untested):**

```
SHVLA_SIM_TIMEOUT=<raised, owner approval> experiments/smolvla-e2e-baseline/run.sh libero sim G4b -- tests/gvsoc/run.py smolvla-e2e --npz $SHVLA_DATA/smolvla_libero/e2e_c2_t1024.npz --e2e-trace
```

The vision phase is projected at 3.72 h (estimate), so the 7200 s default stops it; the rule is to ask the owner first. Nobody has run this command.

## 7. Success rate (G6b)

```
(source infra/lerobot/env.sh && python infra/lerobot/libero_eval.py --run-name <new name> --n-episodes 10 --batch-size 2)
```

10 tasks of `libero_spatial` x 10 episodes, about 63 min under contention, `summary.json` with Wilson 95 % intervals in `$SHVLA_DATA/eval/<run name>/`. Batch 5 ran out of memory on this machine. Run it with nothing else heavy running.

## 8. Tests and lint

```
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 uv run --directory infra/lerobot pytest -q tests     # 12 tests, about 100 s
(cd experiments/smolvla-e2e-baseline/softhier-mlir && bash tests/run_filecheck.sh)                                 # after run.sh apply, in the experiment environment
```
