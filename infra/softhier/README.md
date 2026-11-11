# infra/softhier: build and test the GVSoC SoftHier simulator

This folder sets up the SoftHier simulator (the `soft_hier_release` branch of GVSoC, in the
submodule `third_party/gvsoc`), builds it for one architecture file, and checks it with the four
kernels that ship with GVSoC. Design: `docs/designs/smolvla-softhier-baseline.md` (D1, D2).

## Prerequisites

- Linux x86_64 (no chroot needed). gcc and g++ >= 11.2, cmake >= 3.18.1, `uv`.
- apt packages (the owner installs them; `setup.sh` prints the exact command if any is missing):

  ```
  build-essential git doxygen python3-pip libsdl2-dev curl cmake gtkwave libsndfile1-dev rsync
  autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev wget bzip2 xz-utils
  ```

- About 6 GB of disk and 19 GB of RAM are enough on the host used for the measurements below
  (20 cores).

## Files

| file | what it does |
| --- | --- |
| `setup.sh` | one time: prerequisite check, submodule init, `uv sync`, `make softhier_preparation` (SystemC, DRAMSys, RISC-V toolchain), permalinks |
| `build_arch.sh <arch.py>` | `cfg=<abs arch> make hw` in the submodule, then `stamp.py write`; on failure the stamp is deleted and the exit code is non-zero |
| `run_kernel_tests.sh` | runs the kernels `gemm`, `norm`, `acti`, `attn`; prints a table |
| `stamp.py` | `write <arch>`, `check <arch>` (exit 0 match, 1 with the difference), `permalinks` |
| `gvsoc_env.sh` | sourced by the scripts: PATH, PYTHONPATH, SYSTEMC_HOME, LD_LIBRARY_PATH as GVSoC's `sourceme.sh` sets them, plus the uv venv |
| `pyproject.toml`, `uv.lock` | Python 3.12 environment for GVSoC's tools (not a package); setuptools is pinned `<81` because GVSoC's `reggen` imports `pkg_resources` |

Usage, from the repo root, after `source infra/env.sh`:

```
bash infra/softhier/setup.sh
bash infra/softhier/build_arch.sh third_party/gvsoc/implementation/config/arch/arch.py
bash infra/softhier/run_kernel_tests.sh
python3 infra/softhier/stamp.py check third_party/gvsoc/implementation/config/arch/arch.py
```

Logs go to `$SHVLA_DATA/softhier/` (`setup/`, `build/<arch>.log`, `g1/<kernel>.log`).
`SHVLA_SIM_TIMEOUT` (default 7200 s) limits each simulator process.

## The stamp

`third_party/gvsoc/.shvla_stamp.json` records what the installed simulator was built from:
the arch file path and SHA-256; the commit of GVSoC and of each nested submodule (`core`,
`pulp`); per repository a SHA-256 of `git diff HEAD` plus `git status --porcelain
--untracked-files=no`; and the model folders softhier-mlir's `run_sim` picks at run time
(whether `install/models_fast` and `install/models_fix` exist, a SHA-256 over their `.so` files,
and the values of `SOFTHIER_MODEL_DIR` and `SOFTHIER_STOCK_MODELS`).
Rule: one simulator build per worktree. A run is refused when `stamp.py check` fails, and nothing
rebuilds by itself. After any change to the arch file or the simulator sources, run
`build_arch.sh` explicitly. GVSoC's own preparation patches the nested `core` and `pulp`
checkouts; this shows up as a diff and is part of the stamp. Untracked files (build products) are
not covered.

## Ideal HBM

The local GVSoC branch `shvla/soft_hier_release` has a switch that stock GVSoC lacks: with
`SOFTHIER_IDEAL_HBM=1`, `soft_hier/flex_cluster/flex_cluster.py` instantiates gvsoc's ideal memory
(`Memory`, zero latency, infinite bandwidth, same ports, bindings and address map) for every HBM
channel instead of `memory.dramsys.Dramsys`. softhier-mlir's `run_sim` sets the variable to `1`, and all
its published numbers assume it. The variable is read when `gapy` generates the config, so it is a
run-time switch, not a build option: `build_arch.sh` needs no extra variable and the stamp does not
record it; the source change is covered by the GVSoC commit in the stamp. Kernel tests started from
`run_kernel_tests.sh` (not through `run_sim`) use DRAMSys unless you export `SOFTHIER_IDEAL_HBM=1`.

## Kernel tests rebuild the hardware

The `implementation/Makefile` targets `<k>-pre` (and so `<k>-run`) run `make hs`, which rebuilds
the hardware model with `implementation/config/arch/arch.py` and then the kernel's ELF. So every
kernel target overwrites `install/` with a build for that arch file. `run_kernel_tests.sh` runs
`make <k>-pre` and then the Makefile's `run` recipe separately (so the time limit covers only the
simulation), and at the end rewrites the stamp for `implementation/config/arch/arch.py`. If the
installed build was made for another arch file (for example the softhier-mlir one), run
`build_arch.sh` for that file again afterwards; the stamp check will say so.

Pass rule: simulator exit code 0, the line `[Performance Counter]: Execution period is N ns`
(the same regular expression as softhier-mlir's `PERF_RE`), and none of `No entry found for
burst`, `Executing illegal instruction`, `Received out-of-bound request`. The four kernels'
`main.c` print nothing and nothing compares their output with a golden result, so the kernel tests
check that the simulator runs to the end, not that the results are numerically right.

## Toolchain

The SoftHier SDK is compiled with `riscv32-unknown-elf-gcc` from
`third_party/gvsoc/third_party/toolchain/install/bin` (GCC 14.2.0). The other toolchain in that
folder, `v1.0.16-pulp-riscv-gcc-centos-7` (GCC 7.1.1), is commented out in GVSoC's `sourceme.sh`
and cannot build `-march=rv32imafdv_zfh`, which the SDK uses when the arch file has Spatz cores.

## Notes from setup

- The Makefile's DRAMSys step runs `read -t 30 -r user_input` only if its self-test of the
  prebuilt library fails. Under `/bin/sh` (dash) `read -t` is an error ("Illegal option"); the
  variable stays empty, and the Makefile proceeds as if "y" was typed. `setup.sh` gives it stdin
  from `/dev/null`.
- That self-test dlopens the prebuilt `libDRAMSys_Simulator.so`, which needs
  `libsystemc.so.2.3`. Without `LD_LIBRARY_PATH` pointing at `third_party/systemc_install/lib64` it
  segfaults, and the Makefile then rebuilds DRAMSys (about 40 min) with SystemC 2.3.4 headers; that
  library does not load against the 2.3.3 runtime (`undefined symbol ... sc_api_version_2_3_4`).
  `setup.sh` now exports the path. If a tree was prepared before that fix, copy
  `soft_hier/libDRAMSys_Simulator.so` over `third_party/DRAMSys/libDRAMSys_Simulator.so`.
- `make hw` runs `make config`, which deletes and recreates `pulp/pulp/chips/flex_cluster`; the
  `core`/`pulp` submodules therefore show as modified. Do not commit them.
- Power capture interval: the GVSoC engine stops a power capture and runs `make ice_power` every
  `PWR_INTERVAL_PS` of simulated time (default 100 us), which fails outside the GVSoC root and grows
  memory past 13 GB (LOG G3.1). `build_arch.sh` passes `pwr_interval_ps=$SHVLA_PWR_INTERVAL_PS` to
  `make hw` (default 9000000000000000 ps = 9000 s, set in `env.sh`); `stamp.py` records the build's
  `PWR_INTERVAL_PS` from `build/CMakeCache.txt` and `check` fails if it differs from the variable.

## Switching the submodule to the fork (D2)

When the first simulator fix exists, the submodule moves to `HaozeG/gvsoc`
(branch `shvla/soft_hier_release`): change the URL in `.gitmodules`, run
`git submodule sync third_party/gvsoc`, fetch and check out the fork's commit, commit the new
pointer, then `build_arch.sh` again (the stamp changes with the commit).
Nested `core`/`pulp` forks are created only if a fix touches them. Side-note table of simulator
symptoms and their fixes: see the design document, "Reference for fixes".

## Measured (host: 20 cores, ~19 GB RAM, 2026-11-10)

| item | value |
| --- | --- |
| `make softhier_preparation` (includes the unneeded DRAMSys rebuild, see above) | 1857 s (this run took the DRAMSys rebuild branch, see above; a run with the self-test passing was not measured) |
| `make hw` for `implementation/config/arch/arch.py`, first build | 98 s |
| `<k>-pre` (clean, config, preload, incremental hardware build, SDK compile) | 10 to 12 s per kernel |
| `gemm` simulation | 21708 ns simulated, 67 s wall |
| `norm` simulation | 13177 ns simulated, 4 s wall |
| `acti` simulation | 8060 ns simulated, 4 s wall |
| `attn` simulation | 298126 ns simulated, 1280 s wall |
| disk, `third_party/gvsoc` | 4.8 GB after preparation, 5.4 GB after the build and the kernel runs |
