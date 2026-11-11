# Step log: SmolVLA end-to-end baseline on SoftHier

## Purpose

这个文件是整个 effort 的唯一 trace,包括 infrastructure steps。目标是在 GVSoC `soft_hier_release` 上跑 SmolVLA(softhier-mlir 的 flow),用 golden reference 检查数值,并得到第一份 latency 和 success rate baseline。完整设计见 [docs/designs/smolvla-softhier-baseline.md](../../docs/designs/smolvla-softhier-baseline.md)。每个 step 结束时写一条 entry,失败的尝试和 dead ends 也要写。

## Step list

Status values: planned, in progress, done, failed, skipped, blocked (waiting for an owner action).

| step | files | action | checkpoint | status | commits |
| --- | --- | --- | --- | --- | --- |
| G0.0 | 29 | step log skeleton (this file) | LOG.md committed | done | f48297b |
| G0.1 | 19–22 | wiki branch `decision/first-baseline` in the submodule: decision, catalog, glossary row, knowledge link; `wiki.py fmt`, `check`, `commit update` | `wiki.py check` passes; owner told the branch name | done | wiki branch `decision/first-baseline`: 6f17054 |
| G0.2 | 3, 4 | `.gitignore`, README layout section | — | done | infra 5eb1064 |
| G1.1 | 1, 2 | add the GVSoC submodule (infra commit with only that entry in `.gitmodules`) | `git submodule status` shows `soft_hier_release` | done | infra 5c6a71b |
| G1.2 | 5 | `infra/env.sh`, `env.local.sh.example` | `env.sh --show` prints every variable | done | infra daa21be |
| G1.3 | 8, 9 | GVSoC uv project, `setup.sh`; run it | `softhier_preparation` finished; toolchain present | done | infra 7251468 |
| G1.4 | 10, 12 | `stamp.py`, `build_arch.sh` | `stamp.py check` passes after a build with `implementation/`'s arch | done | infra bce9a99 |
| G1.5 | 11 | `run_kernel_tests.sh`; read the kernel make targets and pass strings; run | `gemm`, `norm`, `acti`, `attn` PASS | done | infra 74fbf46 |
| G1.6 | 7 | `infra/softhier/README.md` with measured times | — | done | infra 89b92d8 |
| G6a.1 | 14, 15 | LeRobot uv project and env; settle OQ1 | `smolvla_libero` loads at its revision | done | infra b3382eb |
| G6a.2 | 17 | `libero_eval.py`; 1 task × 1 episode smoke run | `summary.json` written | done | infra 3d1607c |
| G2.1 | 16, 18 | `golden_ref.py` synthetic and from-npz modes; tests (a)–(c) | published ref written; tests pass | done | infra 6aa2bd2 |
| G2.2 | 23, 24, 26, 27, 28 (0001–0005) | softhier-mlir submodule at `b11cd5f` (experiment commit), experiment uv project, configs, `run.sh`, patches | `run.sh published fetch` and `prepare` finish; patches apply | done | experiment 0c2fefe, ceb1249, e5af70f, ec9c6a7, 8c8efc8, 31795c1 (setuptools pin), the run.sh fixes commit and the log commits (see entry) |
| G2.3 | — | crosscheck: `run.sh apply`; then in the LeRobot environment `python experiments/smolvla-e2e-baseline/softhier-mlir/softhier_mlir/frontend/smolvla_expert_ref.py --ckpt $SHVLA_DATA/smolvla_base --cams 1 --img-size 256 --out $SHVLA_DATA/golden/xc_ref.npz` (by file path, not `-m`, so the package `__init__` and its xdsl import are not loaded; the script itself imports only stdlib, numpy, torch and LeRobot), then `golden_ref.py --obs from-npz:$SHVLA_DATA/golden/xc_ref.npz --device cpu --crosscheck $SHVLA_DATA/golden/xc_ref.npz` | every key within tolerance (Verification Plan) | done | no infra commit (golden_ref.py unchanged); the log commit |
| G2.4 | 16, 18 | libero mode; test (d), (e) | LIBERO ref written; tests pass | done | infra 59218d9; the log commit |
| G2.5 | 13 | `infra/lerobot/README.md` | — | done | infra 84ad868; the log commit |
| G6b.1 | — | `libero_spatial` 10 × 10; settle OQ2–OQ4 | success rate with interval in LOG.md | done | the log commit |
| G3.0 | 25 | arch file; `build_arch.sh` with it | `stamp.py check` passes | done | experiment 0c8e7a2 |
| G3.1–G3.6 | — | ladder rungs 1–6; fork commits (33) when a fix is needed, owner first | each rung's pass column | G3.1 flex_alloc fix done (infra 22d549c); iDMA fix applied (GVSoC 0e4d719) and verified; rung 1 PASS (7/7, timing +46 % vs published on the 16-cluster shapes); ideal-HBM switch added (GVSoC cdb56a2, infra 1d195b0): rung 1 now matches published (1024x3072x768 520114 ns vs 520.5 us); rung 2: 9/9 complete after the second iDMA fix (GVSoC 2910e62, G3.1-diag4; `siglip_encoder_layer` has no self-check, so that only means it terminates); **G3.2-diag: rung c (siglip-mlir seq 256) FAIL, rung 3 FAIL, rung 4 FAIL, rung 5 FAIL; all four fail after the first scalar-FP row op and run 24-41 % too fast; diagnosis = the private Snitch/RedMulE patches (side-note #3, #10, ...) are absent; rung 6 and `gate.py` not run (STOP, owner decision)** **G3.2-fix-redmule: GVSoC RedMulE fp16 subnormal fix (A cd200ad) + RNE (B 7c9cd2f); rung c (QG/KG/VG 0 bad, H/G/OUT PASS), rungs 1/2/rowops/siglip_encoder_layer ROIs unchanged, rungs 3/4/5/6 all PASS (exit 0); rung 6: total 91.139 ms (-23.5 %), actions max-abs 0.0068; timing faster than published on every rung (rung c marks unchanged by the fix) = recorded, continue** **G3.3-fix-instr: `instr_mem` width_log2 default (GVSoC f99d5a2, infra 7d14f70) applied; rung 1 (7/7), rung 2 (8 self-checks + siglip ran), rung c (exit 0, marks 0.324/0.840/1.040, layer 2.204 ms) pass, new baselines; rung 6 PASS exit 0: total 120.716 ms vs 119.1 (+1.4 %), actions max-abs 0.0068; rungs 3-5 not re-run (owner choice)** | experiment (run.sh + LOG), infra 18ad903, infra 2088d59 (GVSoC d42c382); G3.2-fix10: #10 (`nseq`) applied (GVSoC 070936b, infra efec4c4), rung 1/2 ROIs unchanged (siglip_encoder_layer +0.95 %), rung c STILL FAIL (H/G/OUT bit-identical), `rowops` all-cluster PASS, STOPPED; next = so_D iDMA A/B and `run.py mesh` (see LOG G3.2-fix10) ; commits: experiment, infra efec4c4 (GVSoC 070936b) ; infra 83cd69d (GVSoC A), infra f56095c (GVSoC B) |
| G3.7 | 34 | `gate.py g3` on the rung-6 run | gate outcome recorded (PASS or MISS with values) | **done (G3 gate PASS, G3.7-gate: `gate.py g3` on fixi-rung6: total 120.716 ms vs 119.1 +1.4 %, action max-abs 0.0068; see G3.7-gate)** earlier history: blocked (rungs c/3-5 fail accuracy; #10 applied (GVSoC 070936b, infra efec4c4) and did not fix it, see G3.2-fix10: H of rung c is wrong deterministically, next step = stage dumps of AO0/AO1, candidates #8 / #1) **G3.3-fix1: side-note #1 applied (GVSoC fd1124a, infra 8fdce36), rung c marks and every rung 1/2/rowops/siglip_encoder_layer ROI bit-identical, gap unchanged (-24..-40 % non-GEMM segments), STOPPED before rungs 3-6/gate (see G3.3-fix1)** **done: `gate.py g3` on fixrm-rung6: total 91.139 ms vs 119.1 +-5 % = MISS (-23.5 %), action max-abs 0.0068 <= 0.10 = PASS; gate MISS on time only (see G3.2-fix-redmule)** | |
| G4a.1–G4a.4 | 31 | `run.sh libero prepare`; G4 table steps; `gate.py g4a`; INPUT_CONFIG.md | projection per phase recorded; R1 decision if any phase > 2 h | planned | |
| G4b | — | full LIBERO chain | every phase finished or the R1 stop recorded | planned | |
| G5.1 | 6, 30, 32 | RESULTS.md, SETUP.md, design update | Verification Plan passes | planned | |
| G5.2 | — | build `infra-baseline` by cherry-pick; offer push and PR to the owner | owner decision recorded | planned | |

## Entry format

每个 entry 在 step 结束时写,字段如下:

- step id(例如 `G1.3`)和 time;
- what and why:做了什么,为什么做;
- commands:精确的命令;
- result:几行结果;
- files changed;
- commit hash,或 "no commit";
- paths under `$SHVLA_DATA`;
- **What to inspect**:owner 应该打开检查的文件或输出。

Commit 规则:experiment step 是一个 commit(代码和 LOG.md entry 一起);infra step 是两个 commit:先 infra commit(代码),再 experiment commit(LOG.md entry,写明 infra commit)。两个 subject 都带 step id。

## Entries

### G0.0 — step log skeleton

- Time: 2026-11-10。
- What and why: 建立 LOG.md 的骨架:purpose、设计链接、step list(加了 status 和 commits 两列,以及设计里没有的 G0.0 行)、entry format。这样后面每个 step 都有统一的记录位置。
- Commands: `mkdir -p experiments/smolvla-e2e-baseline`,从设计的 "Ordered steps" 表生成 step list。
- Result: 文件已创建,所有 step 为 `planned`,G0.0 为 `done`。
- Files changed: `experiments/smolvla-e2e-baseline/LOG.md`。
- Commit: f48297b(`[G0.0] experiment: step log skeleton`)。
- Paths under `$SHVLA_DATA`: none。
- What to inspect: step list 表的 status 列和 commits 列是否符合你想要的格式。

### G0.1 — wiki decision

- Time: 2026-11-10。
- What and why: 在 wiki submodule 里建分支 `decision/first-baseline`(从 detached HEAD `5dd005f` 出发,没有动 `main`),写 decision,记录两个 metrics、simulator 和 compiler 选择、layout rule、LIBERO configuration、SR-A,以及 SR-B 和 SR-C 两个 open questions。同时加了 catalog 行、`golden reference` glossary 行(`defines:`)和 knowledge 链接。
- Commands: `git checkout -b decision/first-baseline`;`softhier_vla_wiki/tools/wiki.py setup`;`glossary --az`;`fmt`;`check`;`commit update "[G0.1] ..."`;之后 amend 加上 `Co-Authored-By` 行。
- Result: `wiki.py check` 0 error,2 个 STALE warning(glossary 和 knowledge 的 `_overview.md`,内容仍然正确,按 wiki 规则不改)。commit-msg hook 要求 subject 是 `wiki(<op>): ...`,所以 subject 是 `wiki(update): [G0.1] record first-baseline metrics decision`。
- Files changed: wiki 里的 `memories/decisions/2026-11-10-first-baseline-latency-and-success-rate.md`(新),`memories/decisions/_overview.md`,`_abstract.md`,`glossary/project-terms.md`,`glossary/a-to-z.md`,`knowledge/softhier-and-the-project.md`。
- Commit: wiki 分支 `decision/first-baseline` 上的 6f17054。父仓库的 `softhier_vla_wiki` 指针没有 bump,由 owner 决定。
- Paths under `$SHVLA_DATA`: none。
- What to inspect: decision 文件的 Decision 和 Open questions 两节;设计里的 gate 数字(119.1 ms、±5 %、0.10)按 wiki-design decision 0013 没有写进 wiki;decision 里的设计链接在设计合入 `main` 之前会是 404。

### G0.2 — `.gitignore` and README

- Time: 2026-11-10。
- What and why: `.gitignore` 加 `.venv/`、`__pycache__/`、`infra/env.local.sh`、`egm-review/`(`CLAUDE.md` 仍然 ignored)。README 第一行的链接换成 GVSoC `soft_hier_release`,并加了 Repository layout、layout rule、Tracked metrics、Quick start 四部分。
- Commands: 编辑两个文件,`git add .gitignore README.md`。
- Result: infra commit 只含这两个文件。README 里指向 wiki decision 的相对链接,在 owner bump submodule 指针之前,在 pinned wiki commit 里还解析不到。
- Files changed: `.gitignore`,`README.md`。
- Commit: infra 5eb1064(`[G0.2] infra: layout section, metrics, gitignore`);这条 LOG entry 在紧接着的 experiment commit 里。
- Paths under `$SHVLA_DATA`: none。
- What to inspect: `README.md` 的 Repository layout 表和 Tracked metrics 两节。

### G1.1 — GVSoC submodule

- Time: 2026-11-10。
- What and why: 把 GVSoC `soft_hier_release` 加成 `third_party/gvsoc` submodule,`.gitmodules` 里设 `ignore = dirty`(build 输出不污染 `git status`),然后递归 init 嵌套的 `core` 和 `pulp`。
- Commands: `git submodule add -b soft_hier_release https://github.com/gvsoc/gvsoc.git third_party/gvsoc`;手动在 `.gitmodules` 加 `ignore = dirty`(worktree 里 `git config -f` 被拒绝);`git submodule update --init --recursive third_party/gvsoc`。
- Result: `git submodule status` 显示 `third_party/gvsoc (heads/soft_hier_release)`。
  - GVSoC: `d806a84c5785237547f504c2e34a4d9f230cb437`,https://github.com/gvsoc/gvsoc/tree/d806a84c5785237547f504c2e34a4d9f230cb437
  - `core` (gvsoc-core): `bda268a4808708a48eab21981c7d6bbced1577d7`,https://github.com/gvsoc/gvsoc-core/tree/bda268a4808708a48eab21981c7d6bbced1577d7
  - `pulp` (gvsoc-pulp): `3ca2016d056dda72317e27c4319fc6174b5f8c4d`,https://github.com/gvsoc/gvsoc-pulp/tree/3ca2016d056dda72317e27c4319fc6174b5f8c4d
  - checkout 大小 40 MB(build 之前)。
- Files changed: `.gitmodules`,`third_party/gvsoc`(gitlink)。
- Commit: infra 5c6a71b(`[G1.1] infra: GVSoC soft_hier_release submodule`)。
- Paths under `$SHVLA_DATA`: none。
- What to inspect: `.gitmodules` 的 gvsoc 条目(URL、branch、ignore)。

### G1.2 — `infra/env.sh`

- Time: 2026-11-10。
- What and why: 环境变量默认值的单一来源,按设计的变量表:`SHVLA_ROOT`、`SHVLA_DATA`、`SOFTHIER_HOME`、`SOFTHIER_CHROOT`(导出为空)、`SHVLA_SIM_TIMEOUT=7200`。每个默认值只在变量未设置时生效,最后 source `infra/env.local.sh`(如果存在)。`HF_HOME` 不强制设值(用 HF 默认),`SHVLA_SMOLVLA_DIR` 由 experiment config 设,`SOFTHIER_IDEAL_HBM` 不导出。
- Commands: `bash infra/env.sh --show`;`bash -c 'set -u; source infra/env.sh'`。
- Result: `--show` 打印所有变量(未设置的显示 `<unset ...>`);`set -u` 下 source 通过。shellcheck 没有安装,所以没有跑(脚本里只有 `# shellcheck` 注释)。
- Files changed: `infra/env.sh`,`infra/env.local.sh.example`。
- Commit: infra daa21be(`[G1.2] infra: env.sh and env.local.sh.example`)。
- Paths under `$SHVLA_DATA`: none。
- What to inspect: `bash infra/env.sh --show` 的输出。

### G1.3 — GVSoC uv project and `setup.sh` (done)

- Time: 2026-11-10。
- What and why: 写 `infra/softhier/pyproject.toml`(Python 3.12,非 package)和 `setup.sh`(prerequisite check、submodule init、`uv sync`、`make softhier_preparation`、permalinks),然后运行。第一次运行因 apt 前置包缺失 blocked;owner 安装后重新运行,成功。
- Commands: `bash infra/softhier/setup.sh`(第一次:exit 1,缺 apt 包;owner 安装 `doxygen libsdl2-dev gtkwave autoconf automake texinfo libtool libsdl2-ttf-dev` 之后:exit 0)。再运行一次验证 skip。
- Result:
  - `make softhier_preparation` wall time 1857 s(约 31 min;整个脚本约 1860 s)。
  - DRAMSys 分支:预编译库自检**失败**(`Segmentation fault (core dumped)`,日志里 `Test libaray failed, We need to rebuild the library, tasks around 40 min`),走了重编译分支。`read -t 30` 在 `/bin/sh`(dash)里报 `read: Illegal option -t`,同样落到默认 yes(`Go Go Go!`)。重编译实际用时包含在上面的 1857 s 里(比 Makefile 估计的 40 min 短)。
  - `du -sh third_party/gvsoc`:4.8 GB(之前 40 MB)。
  - Toolchain 在 `third_party/gvsoc/third_party/toolchain/`,有两套:`install/bin/riscv32-unknown-elf-gcc` → GCC 14.2.0(`toolchain.tar.xz`,husterZC);`v1.0.16-pulp-riscv-gcc-centos-7/bin/riscv32-unknown-elf-gcc` → GCC 7.1.1 20170509(PULP 官方包)。后续 step 用哪一套要看 GVSoC 的 SoftHier 配置/Makefile(G1.4 时确认)。
  - skip 检查:第二次运行打印 `softhier_preparation already done; skipping.`,用时 0.2 s,没有重新 make,`preparation.log` 没有被覆盖。
  - Permalinks:
    - GVSoC: https://github.com/gvsoc/gvsoc/tree/d806a84c5785237547f504c2e34a4d9f230cb437
    - core: https://github.com/gvsoc/gvsoc-core/tree/bda268a4808708a48eab21981c7d6bbced1577d7
    - pulp: https://github.com/gvsoc/gvsoc-pulp/tree/3ca2016d056dda72317e27c4319fc6174b5f8c4d
- Notes / deviations from the design:
  - `pyproject.toml` 合并了 `requirements.txt`、`gapy/requirements.txt`、`core/requirements.txt`,取较严的下界(`pyelftools>=0.29`,`PyYAML>=6.0.1`)。去掉了 `argparse` 和 `dataclasses`(Python 3.12 标准库已有)。
  - apt 检查在 README 的列表之外加了 `wget`、`bzip2`、`xz-utils`。
  - Makefile 的 `SYSTEMC_INSTALL_DIR := $(PWD)/...` 读 shell 的 `PWD`,所以脚本用 `cd third_party/gvsoc && make`,不用 `make -C`;不传 `-j`。
  - "已完成"的判断比设计更严:检查 `libsystemc.so`、`libDRAMSys_Simulator.so` 和 toolchain 里的 `riscv*-gcc`。
  - 没有对脚本做修改:第二次运行没有需要修的东西。
  - GVSoC submodule 在构建后工作区有改动(构建产物),没有提交;`softhier_vla_wiki` 指针也没有提交。
- Files changed: `infra/softhier/pyproject.toml`,`infra/softhier/uv.lock`,`infra/softhier/setup.sh`(都在 7251468)。本次只改 LOG.md。
- Commit: infra 7251468(`[G1.3] infra: GVSoC uv project and setup.sh`)。
- Paths under `$SHVLA_DATA`: `softhier/setup/preparation.log`(完整 make 输出,含 `Test libaray failed`)。
- What to inspect: `grep -n 'Test libaray\|exit code' $SHVLA_DATA/softhier/setup/preparation.log`;`third_party/gvsoc/third_party/toolchain/install/bin/riscv32-unknown-elf-gcc --version`;第二次运行的 skip 输出。

### G6a.1 — LeRobot uv project and env (OQ1 settled)

- Time: 2026-11-10。
- What and why: 建 `infra/lerobot/` uv 项目(Python 3.12,非 package)和 `env.sh`,然后在 CUDA 上用 fp32 加载 `lerobot/smolvla_libero`,解决 OQ1。
- **OQ1 答案:lerobot 0.7.0 可以加载该 checkpoint,不需要退回 0.6.1。** 证据:`SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero", revision="a24f1f380461b665158d52acd140152c79e8b36d")`,然后 `.to("cuda", dtype=torch.float32)`,成功;`config.pretrained_revision` 等于该 hash;HF cache 里只有 `snapshots/a24f1f380461b665158d52acd140152c79e8b36d`;所有参数 dtype 为 `torch.float32`,device `cuda:0`;峰值显存 1.83 GB。`from_pretrained` 的 kwarg 是 `revision`(`lerobot/policies/pretrained.py:171`)。
- 实际安装的版本(`uv pip list`):lerobot 0.7.0,transformers 5.22.2(**≥ 5.4:是**;lerobot 0.7.0 要求 `transformers>=5.22,<5.23`),torch 2.15.0+cu134,torchvision 0.29.2+cu134,torchcodec 0.17.0,numpy 2.2.6,mujoco 3.8.1,robosuite 1.4.0,hf-libero 0.1.4,pytest 9.2.0。
- CUDA wheel index:`nvidia-smi` 报告 CUDA 13.4,PyTorch 有 cu134 的 cp312 torch 2.15.0 和 torchvision 0.29.2,所以用 `pytorch-cu134`(`explicit = true`,`torch` 和 `torchvision` 显式映射)。lerobot 0.7.0 要求 `torch<2.16`,满足。
- OQ2 输入(加载后的 config,最终答案在 G6b):`observation.state` shape (8,);图像 key `observation.images.camera1`、`observation.images.camera2`,各 (3, 256, 256);`action` shape (7,);`max_state_dim = max_action_dim = 32`;`chunk_size = n_action_steps = 50`;`resize_imgs_with_padding = (512, 512)`;`tokenizer_max_length = 48`。
- `env.sh`:source `infra/env.sh`,激活 `.venv`,`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`,`MUJOCO_GL=osmesa`,`PYOPENGL_PLATFORM=osmesa`。另外加了 `LIBERO_CONFIG_PATH=$SHVLA_DATA/libero`:hf-libero 在 `~/.libero/config.yaml` 不存在时会 `input()` 提问(`libero/libero/__init__.py:104`),后台 job 会卡住或 EOFError。`env.sh` 在 config 不存在时用 `echo n | python -c "import libero.libero"` 创建一次。注意这个 `config.yaml` 里是当前 venv 的绝对路径;换 worktree 或重建 venv 后删掉 `$SHVLA_DATA/libero` 让它重建。`libosmesa6` 和 `libglfw3` 都已安装,不缺系统包。
- Commands: `uv sync`(在 `infra/lerobot`,约 10 分钟,大部分是 CUDA 库下载);`source infra/lerobot/env.sh`;在 `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0` 下运行加载脚本(`systemd-run --user` 在这个 WSL 可用)。
- Deviations: `env.sh` 比设计的 file 15 多了 `LIBERO_CONFIG_PATH` 和首次 source 时创建 config 的一步(原因见上);`from_pretrained` 默认 `strict=False`,所以 key 不匹配不会报错,smoke run 的成功间接说明权重是对的。同一个 experiment commit 里还删掉了 step list 表底部 6 行误拷贝自设计环境变量表的行(`| variable |` 表头、`| --- |` 分隔行和 4 行 `` `SHVLA_*` ``/`` `HF_HOME` ``)。
- Files changed: `infra/lerobot/pyproject.toml`,`infra/lerobot/uv.lock`,`infra/lerobot/env.sh`;`LOG.md`(本 entry,表格修正)。
- Commit: infra b3382eb(`[G6a.1] infra: LeRobot uv project and env.sh`);本 entry 在紧接着的 experiment commit 里。
- Paths under `$SHVLA_DATA`: `libero/config.yaml`;模型在默认 HF cache(`~/.cache/huggingface/hub/models--lerobot--smolvla_libero`)。
- What to inspect: `infra/lerobot/pyproject.toml` 的 index 和 sources;上面 OQ1 的证据。

### G6a.2 — `libero_eval.py` and smoke run

- Time: 2026-11-10。
- What and why: 写 `infra/lerobot/libero_eval.py`:拼出 `lerobot-eval` 命令,放在 `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` 下运行(这个 WSL 里 `systemd-run --user` 可用;`--no-cap` 可关),然后读 `eval_info.json`,打印每个 task 和 overall 的成功率与 Wilson 95 % 区间,写 `summary.json`。区间由 `per_task[].metrics.successes` 的 k、n 自己算(标准库 + numpy),没有从 `pc_success` 反推。
- 在 lerobot 0.7.0 源码里核实的 flag(路径相对 `infra/lerobot/.venv/lib/python3.12/site-packages/lerobot/`):
  - revision:`--policy.pretrained_revision`(`configs/policies.py:86`;CLI 解析 `configs/parser.py:229`;`scripts/lerobot_eval.py:880` 传给 processor 加载,`policies/factory.py:411`、`:424` 传给权重加载;eval 之后 HF cache 里仍然只有 `snapshots/a24f1f38…`,没有 `refs/main`,所以权重没有从 `main` 拉)。
  - `--eval.n_episodes` 是**每个 task** 的 episode 数:`eval_policy_all` 把同一个 `n_episodes` 传给每个 task 的 runner(`scripts/lerobot_eval.py:1150`,runner 在 `:1053` 调用 `eval_one`);所以 10 tasks × 10 episodes 就是 `--eval.n_episodes=10`。
  - `--rename_map`:`EvalPipelineConfig.rename_map: dict[str, str]`(`configs/eval.py:43`),CLI 用 JSON:`--rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'`,作为单个 argv 元素传(Python 列表,不经过 shell)。启动日志里 `rename_map` 被正确解析。
  - 子集:`--env.task_ids='[0]'`(`envs/configs.py:469` 的 `LiberoEnvConfig.task_ids`),所以 1-task smoke run 不需要跑整个 suite。
  - `eval_info.json` 结构:`per_task[]`(`task_group`、`task_id`、`metrics.successes`、`n_success`、`pc_success_ci95` ...)、`per_group`、`overall`(`scripts/lerobot_eval.py:1071`、`:1180–1236`)。
- Command: `source infra/lerobot/env.sh && python infra/lerobot/libero_eval.py --run-name g6a2_smoke_1 --task-ids 0 --n-episodes 1 --batch-size 1`(后台运行)。
- Result: `libero_spatial` task 0,1 episode,seed 1000:成功 1/1,Wilson 95 % [20.7 %, 100 %](和 LeRobot 自己报的 `pc_success_ci95` [20.65, 100.0] 一致)。1 个 episode 在统计上没有意义,只说明 pipeline 通了(env 构建、osmesa 渲染、策略加载、rename、rollout、写文件)。
  - wall time 111 s(`lerobot-eval` 自己报 `eval_s` 91.5 s;rollout 最多 280 步,约 6.5 it/s,含渲染和策略);最大单个进程 RSS 5034 MiB(`RUSAGE_CHILDREN.ru_maxrss`,是进程树里最大的一个进程,不是总和);没有触到 12 GB cap。
  - `EvalPipelineConfig.tf32` 默认为 True,eval 里 TF32 是开的;这只影响 eval,golden reference 会关掉。
  - 日志里只有 robosuite 的 "No private macro file" 警告和 HF 未认证警告。
- Deviations: 无。
- Files changed: `infra/lerobot/libero_eval.py`;`LOG.md`(本 entry,step list 状态)。
- Commit: infra 3d1607c(`[G6a.2] infra: libero_eval.py wrapper around lerobot-eval`);本 entry 在紧接着的 experiment commit 里。
- Paths under `$SHVLA_DATA`: `eval/g6a2_smoke_1/`(`eval_info.json`、`summary.json`、`task_descriptions.json`、`videos/libero_spatial_0/eval_episode_0.mp4`)。
- What to inspect: `$SHVLA_DATA/eval/g6a2_smoke_1/summary.json` 和那段 mp4。

### G2.1 — `golden_ref.py` (synthetic, from-npz) and the published golden reference

- Time: 2026-11-10。
- What and why: 写 `infra/lerobot/golden_ref.py`(`synthetic` 和 `from-npz` 两个 mode;`libero` 是 stub,打印 "implemented in step G2.4" 并以非 0 退出)和 `tests/test_golden_ref.py`(设计的 (a)(b)(c)(f)),并生成 published 配置的 golden ref。
- `lerobot/smolvla_base` revision(Hub 当前 `main`,已 pin):`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`(`HfApi().model_info(...).sha`)。
- lerobot 0.7.0 的 KV cache API(路径相对 `infra/lerobot/.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/`):
  - 填充:`SmolVLMWithExpertModel.forward(..., past_key_values=None, use_cache=True)` 自己建 `DynamicCache()` 并让每层 self-attn 填入(`smolvlm_with_expert.py:441-443`);没有 `fill_kv_cache` 参数了。
  - 布局:`DynamicCache` 存 `[B, heads, seq, head_dim]`,K 在 RoPE 之后(`smolvlm_with_expert.py:277-286`:`update(key_states.transpose(1,2), ...)`)。读取 `cache.layers[l].keys/.values`,形状 `(1, 5, Lp, 64)`;脚本对 16 层都 assert 了形状,然后 `clone()[0].transpose(0,1).reshape(Lp,-1)`,得到 `[Lp, 320]`,列 = `kv_head*64 + d`。
  - crop:`denoise_step` 在每步后对每层 `layer.crop(prefix_len - layer.get_seq_length())`(`modeling_smolvla.py:855-858`),原地修改,所以 KV 在第一个 `denoise_step` 之前就 `.clone()` 出来。
  - 时间步:`euler_integrate` 用 `torch.full((B,), 1.0 + step*dt, float32)`(`common/flow_matching.py`);脚本用同样的 tensor 同时喂 `create_sinusoidal_pos_embedding`(在 `common/vla_utils.py:41`,不在 `modeling_smolvla.py`)和 `denoise_step`。该函数在 CPU/CUDA 上返回 float64,存之前 `.float()`。
- Self-check:同一份输入和 noise 下,`policy.model.sample_actions(...)`(LeRobot 自己的 sample 路径,绕过 `prepare_images`,所以 side 256 不变)和我捕获的 `actions` 的 **max-abs = 0.000e+00**(位相同)。另外在脚本里 assert 了:手写 vision+connector 路径 `torch.equal` `embed_image`;`vis_pos_ids` 等于 SigLIP 的 `position_embedding` 实际收到的 ids(forward hook);加载后的 fp32 权重与 `model.safetensors` 里全部 500 个 tensor `torch.equal`(`from_pretrained` 默认 `strict=False`,所以显式验证)。
- Command: `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python golden_ref.py --checkpoint lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --obs synthetic:1:256:0 --out $SHVLA_DATA/golden/published.npz`(CUDA,RTX 4070 Ti)。
- Output: `$SHVLA_DATA/golden/published.npz`,4 347 810 字节,SHA-256 `75f28e8759bc75fbfe0f6b3b380b38a298bc8bdf1d449a351592f33205cfbcf7`。59 个数组:设计 key 表的全部 key(`images [1,3,256,256] fp16`、`vis_pos_ids [256]`、`lang_tokens/lang_mask [48]`、`state [32]`、`noise [50,32]`、`vis_out_0 [256,768]`、`img_emb_0 [16,960]`、`prefix_embs [65,960]`、`prefix_valid [65]`、`n_valid`、`kv_k_0..15`/`kv_v_0..15` 各 `[65,320]`、`time_emb [10,720]`、`xt [11,50,32]`、`vt [10,50,32]`、`actions [50,32]`、`meta [5] int64`)加 `info_*`(checkpoint、revision、HF permalink、vlm_model_name、lerobot/transformers/torch 版本、obs、prompt、device、selfcheck_maxabs),全部是 `np.str_`/数字,没有 object array,没有时间戳。`meta = [50, 32, 10, 65, 28]`(prefix 65 = 16 图像 + 48 语言 + 1 state,其中 28 个有效)。
- 数值:|K| max 19.576,|V| max 4.523(16 层最大);actions mean 0.03045,std 0.25601,max|a| 1.5764(normalised 空间)。
- Tests:`systemd-run ... uv run pytest -q tests` → 6 passed(43 s):(a) keys/shapes/dtypes,(b) 两次运行 SHA-256 相同,(c) seed 1 的 `images` 和 `actions` 都不同,(f) 未知 `--obs` mode、libero stub、不存在的 revision(`0`*40,`--device cpu`)都非 0 退出并带信息。GPU 测试没有 CUDA 时 skip。`ruff check`(uvx ruff,默认配置)通过。
- Wall time / memory:第一次 golden_ref 运行 103 s(含 `smolvla_base` 的首次下载;SmolVLM2 在 G6a 已经下载过),之后稳定在约 9 s 每次(GPU 和 CPU 都是);GPU 峰值显存 1786 MiB(`max_memory_allocated`);进程最大 RSS 约 3.8 GiB;没有触到 12 GB cap。
- Deviations: (1) 提交的 `Co-Authored-By` 用的是 harness 的 `Claude Sonnet 5.5`,没有用任务文本里的 `Opus 5.5`(没有 CLAUDE.md 规则覆盖)。(2) `vision_position_ids` 用 transformers 5.22.2 里实际的公式(`idx * (1/n)` 再 clamp 到 `1-1e-6`),不是旧 reference 的 `idx/n*(1-1e-6)`;用 hook 验证与 SigLIP 实际用的 ids 相同。(3) `--crosscheck` 只比较两个文件共有的、数值型 key,按固定 pipeline 顺序;shape 不同算 divergence 不抛异常;`--tol` 默认 1e-4。crosscheck 本身 G2.3 才跑,这一步没有用 xc_ref.npz 验证过。(4) 新增 `--tol` 参数。(5) VLM config 和 tokenizer 来自 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 的 `main`(没有 pin),权重来自 checkpoint(已逐个 tensor 验证);tokenizer 对固定 prompt 的输出记录在 npz 的 `lang_tokens`。(6) `pyproject.toml`/`uv.lock` 没有改。
- 补充检查(commit 之后做的):
  - **`vis_pos_ids` 与旧 reference 的公式不同(G2.3 的 crosscheck 预计在这个 key 上 diverge)。** 旧 `smolvla_expert_ref.vision_position_ids` 用 `idx/n*(1-1e-6)`,我的(和 transformers 5.22.2 的 `SmolVLMVisionEmbeddings.forward` 一致,hook 验证过)用 `idx*(1/n)` 再 clamp 到 `1-1e-6`。side 256(n=16)时 `idx/16` 正好落在 `k/32` 的边界上,`bucketize(right=True)` 对"正好在边界"给 bucket k,旧公式因为乘了 `(1-1e-6)` 落在边界下面,给 k-1。numpy 对比:side 256 有 255/256 个元素不同,side 512 有 1023/1024 个不同(idx = 0 的行除外全部不同)。`golden_ref.py` 没有改,它和模型实际用的 ids 一致。G2.3 中 `vis_pos_ids` 是 crosscheck 的第二个 key(在任何 float key 之前),所以预计会先在它上面报 divergence;并且 softhier-mlir 里凡是从 ref npz 读 `vis_pos_ids` 的地方拿到的 position 行,和 transformers 5.22.2 实际用的不同。需要 owner 决定:是 patch 0003 里改旧 reference 的公式,还是 crosscheck 对这个整数 key 单独处理。
  - `from-npz` 和 `--crosscheck` 都实际跑过:CUDA,`--obs from-npz:published.npz --crosscheck published.npz` → 每个 key 都是 0.000e+00,PASS,exit 0(验证了 npz 读取、fp16/bool 往返、crosscheck 输出和 exit code)。
  - CPU 运行(`--device cpu`,同样输入,G2.3 用的配置):`use_deterministic_algorithms(True)` 没有报错,self-check 仍是 0.0,9 s,没有触到 cap。对 GPU published.npz 的相对误差(informational):`vis_out_0` 9.2e-6,`img_emb_0` 1.3e-6,`prefix_embs` 1.3e-6,`kv_k_15` 7.1e-7,`kv_v_15` 3.7e-6,`xt` 4.7e-7,`vt` 1.2e-6,`actions` 9.5e-7;全部 ≤ 1e-4,PASS。
  - SHA 复现:lint 修改之后,用提交的代码再跑一次 `synthetic:1:256:0`,SHA-256 仍是 `75f28e87…bcf7`,与上面记录的一致。
- Files changed: `infra/lerobot/golden_ref.py`,`infra/lerobot/tests/test_golden_ref.py`;`LOG.md`(本 entry,step list 状态)。
- Commit: infra 6aa2bd2(`[G2.1] infra: golden_ref.py synthetic and from-npz modes, tests`);本 entry 在紧接着的 experiment commit 里。
- Paths under `$SHVLA_DATA`: `golden/published.npz`。
- What to inspect: `golden_ref.py` 的 `run_reference`(捕获点和 KV 布局)与 self-check;`published.npz` 的 `info_*` 和 `meta`。

### G1.4 — `stamp.py`, `build_arch.sh`, first hardware build (done)

- Time: 2026-11-10。
- What and why: 写 `stamp.py`(`write`/`check`/`permalinks`,只用标准库)、`build_arch.sh` 和 `gvsoc_env.sh`(被 `build_arch.sh` 和 `run_kernel_tests.sh` source),然后用 `implementation/config/arch/arch.py` 建硬件。
- Toolchain 结论:SoftHier SDK 用 **GCC 14.2.0**(`third_party/toolchain/install/bin`),不是 PULP GCC 7.1.1。依据:(1) `third_party/gvsoc/sourceme.sh` 把 `toolchain/install/bin` 放进 PATH,PULP 那一行是注释掉的;(2) `soft_hier/flex_cluster_sdk/CMakeLists.txt` 直接调 PATH 上的 `riscv32-unknown-elf-gcc`,`-march=rv32imafdv_zfh`(arch 的 `spatz_attaced_core_list` 非空时 Makefile 选这个),`v` 扩展 GCC 7.1.1 编不了。`gvsoc_env.sh` 复制了 `sourceme.sh` 的 PATH/PYTHONPATH/SYSTEMC_HOME/LD_LIBRARY_PATH(不跑它的交互检查和自己的 venv),再激活 `infra/softhier/.venv`。
- Command: `bash infra/softhier/build_arch.sh third_party/gvsoc/implementation/config/arch/arch.py`。Makefile 的做法是 `cfg=<abs> make hw`(`hw` = `make config` + `make TARGETS=pulp.chips.flex_cluster.flex_cluster all`),脚本照做。
- Result:
  - 第一次运行 5 s 失败:`core/models/reggen/gen_dv.py` 里 `from pkg_resources import resource_filename`,uv 装的 setuptools 84.2.1 已经没有 `pkg_resources`(81 起移除)。修:`infra/softhier/pyproject.toml` 加 `setuptools>=59.6.0,<81`,`uv sync` 得到 80.10.2。失败时脚本删了 stamp、退出码非 0(验证了失败路径)。
  - 第二次:`make hw` **98 s**(exit 0),`stamp.py check` 输出 `stamp OK`。
  - stamp 内容:arch SHA-256 `109b8788…2211`;GVSoC d806a84,core bda268a,pulp 3ca2016;`models_fast`/`models_fix` 都不存在;两个环境变量都未设。
  - `du -sh third_party/gvsoc`:5.4 GB(准备后 4.8 GB;`build/` + `install/` 约 0.6 GB)。
- Notes: `make config` 会 `rm -rf` 再 `cp` `pulp/pulp/chips/flex_cluster`,所以 `core`/`pulp` 在 `git status` 里是 modified(构建产物和 patch),被 diff hash 覆盖,没有提交。`.shvla_stamp.json` 是 GVSoC 里的 untracked 文件。没有跑 README 的 hello-world:它用 `cfg=examples/SoftHier/config/arch_test.py ... make hs` 会改硬件,破坏 stamp。
- Files changed: `infra/softhier/stamp.py`、`build_arch.sh`、`gvsoc_env.sh`、`pyproject.toml`、`uv.lock`(都在 bce9a99);本次 LOG.md(本 entry,step list 状态)。
- Commit: infra bce9a99(`[G1.4] infra: stamp.py, build_arch.sh, gvsoc_env.sh; pin setuptools<81 (reggen needs pkg_resources)`)。
- Paths under `$SHVLA_DATA`: `softhier/build/arch.log`(最后一次;第一次失败的日志被覆盖)。
- What to inspect: `third_party/gvsoc/.shvla_stamp.json`;`tail $SHVLA_DATA/softhier/build/arch.log`。

### G2.2 — softhier-mlir submodule, experiment env, patches 0001–0005, configs, `run.sh`; published fetch and prepare (done)

- Time: 2026-11-10。
- What and why:
  - softhier-mlir 作为 submodule 加到 `experiments/smolvla-e2e-baseline/softhier-mlir`,checkout 在 `b11cd5fa4f3e717374d1e9ba14442ff302d1f970`(`.gitmodules`: `branch = main`, `ignore = dirty`)。加的时候 `origin/main` 还是 b11cd5f,**main 没有往前走**,所以没有要 owner 决定的 bump。
  - 实验的 uv 项目(Python 3.12,非 package):softhier-mlir editable、xdsl(其 pyproject 写 `>=0.69`,解析到 0.75.0)、numpy、CPU torch 2.15.0+cpu(PyTorch CPU index)、transformers 5.22.2、safetensors、huggingface_hub、GVSoC 的 Python 依赖(含 prettytable、pyelftools、`setuptools<81`,和 `infra/softhier/pyproject.toml` 一致)、filecheck、lit、pytest。
  - 五个 patch(见 `patches/README.md`),每个由 submodule 里的真实修改生成。header 是 `git apply` 会忽略的前置文字(它跳过第一个 `diff --git` 之前的所有行);已验证 `run.sh apply` 干净应用,连续运行两次也一样(第二次证明 reset 对 index 也有效)。
  - 两个 config、`run.sh`(apply / fetch / prepare / sim)。`sim` 的 stamp 检查:`infra/softhier/stamp.py` 现在已存在,但 `arch/softhier_mlir_arch.py` 要到 G3.0 才有,所以 `sim` 现在 exit 2 并说明缺哪个文件(已试过)。shellcheck 没安装,没跑,只做了 `bash -n`。
- Commands:
  - `git submodule add -b main https://github.com/runwangdl/softhier-mlir.git experiments/smolvla-e2e-baseline/softhier-mlir`;`git -C <path> checkout b11cd5f…`;手动加 `ignore = dirty`。
  - `uv sync`(在 `experiments/smolvla-e2e-baseline`)。
  - `bash run.sh apply`(两次);`bash run.sh published fetch`;`bash run.sh published prepare`;`bash run.sh libero fetch`;`bash run.sh libero prepare`(都在 `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` 下,逐个跑,不并行)。
  - 0003 的运行:`source infra/lerobot/env.sh`,`python experiments/smolvla-e2e-baseline/softhier-mlir/softhier_mlir/frontend/smolvla_expert_ref.py --ckpt $SHVLA_DATA/smolvla_base --cams 1 --img-size 256 --out $SHVLA_DATA/golden/xc_ref.npz`(同样在 `systemd-run` 下)。
  - `bash tests/run_filecheck.sh`(patch 应用之后,实验环境)。
- Result:
  - **Patches**:全部和设计表一致。0001 共 12 行;grep 之后 `/app/models/smolvla_base` 在 `.py` 里只剩 `tests/gvsoc/flow_tables.py:52`、`tests/gvsoc/ttt.py:290`、`tools/e2e/*.py`(不在任何 `run.sh` 路径上,没改,README 里列了)。0002 里 emitter 的参数和模块函数同名(`vision_npz` 等),函数体里会遮蔽,所以加了私有别名 `_vision_npz` 等。`py_compile` 通过。
  - **0003 运行**:在 `infra/lerobot/.venv`(lerobot 0.7.0,transformers 5.22)里 **11.6 s** 跑完,写出 `xc_ref.npz`(132 个数组)。`build_policy`(meta device)和 `materialise`(buffer 重建)**不用改**;只改了设计表里的四个 API 点,外加 `make_att_2d_masks` 和 `create_sinusoidal_pos_embedding` 的 import 位置(`lerobot.policies.common.vla_utils`)。顺带的 sanity(G2.3 才是正式 crosscheck):key 集合覆盖 `published.npz` 的全部 key,`vis_pos_ids` 完全相等,images/lang/state/noise/time_emb 逐位相同,vis_out_0 最大差 2.2e-4,kv_k_0 1.3e-5,xt/actions 1.5e-6(CPU 对 GPU 的差)。
  - **filecheck**:`18 passed, 0 failed`(xdsl 0.75.0,patch 之后)。
  - **fetch**:`hf download`(`huggingface-cli` 在 huggingface_hub 里已经不存在,只有 `hf`,所以用 `hf download --quiet`;这是对设计文字的小偏离)。两个 checkpoint 之前已经在 HF cache 里(G6a/G2.1 下载过),所以 fetch 实际是 0.3 s 的 cache 命中加 symlink,没有测到真实下载时间。
  - **published prepare**(5 步,全部 exit 0):

    | step | 命令 | wall | 输出 |
    | --- | --- | --- | --- |
    | 1 | `smolvla prepare --seq 256` | 16 s | `vision_s256.npz` 193 MB |
    | 2 | `smolvla prepare --seq 1024` | 23 s | `vision_s1024.npz` 282 MB |
    | 3 | `smolvla prepare-vlm --cams 1` | 9 s | `vlm_c1.npz` 348 MB(113 tokens) |
    | 4 | `smolvla_expert prepare --ref published.npz` | 3 s | `expert.npz` 195 MB(n_valid 28) |
    | 5 | `smolvla_e2e prepare --ref published.npz` | 2 s | `e2e_c1_t256.npz` 4.3 MB(1 camera × 256 tokens,prefix 65,28 valid) |

    prepare 自己打印的 reference 误差:step 1(seq 256)fp16-program floor vs HF fp32:EMB 0.0009、L1 0.0026、OUT 0.1155(|ref| max 24.68);step 2(seq 1024):EMB 0.0035、L1 0.0056、OUT 0.1098;step 3:lerobot-semantics 参考对 transformers LlamaModel 3.05e-05,floor 误差 EMB 1.17、L1 1.43、L16 4.81(|ref| max 约 2200–2500)、K1 0.0096、K16 0.0139、OUT 0.0183。注意这是**宿主上 fp16-program floor 对 HF fp32** 的误差;softhier-mlir README 的 0.0067(EMB)/ 0.039(L1)是 rung 3 里**设备**对 HF 的误差(同一个输入,|ref| max 都是 1.23 / 3.6 / 24.7),两者不是同一个量,rung 3 才比较。
  - **libero steps 1–3**(`libero fetch` 之后,`libero prepare` 跑完前三步,第 4 步因为 `libero.npz` 还不存在按设计 exit 1,G2.4 才有):
    - step 1 `--seq 1024`:23 s,`vision_s1024.npz` 282 MB。
    - step 2 `prepare-vlm --cams 1`:7 s,`vlm_c1.npz` 348 MB。
    - step 3 `prepare-vlm --cams 2`:11 s,`vlm_c2.npz` 363 MB,**177 tokens(img 128, lang 48 of which 5 valid, state 1)**,和设计一致。
  - 没有遇到需要新 patch 的问题(没有 R3 变更)。
- Files changed: `.gitmodules`,`experiments/smolvla-e2e-baseline/{softhier-mlir,pyproject.toml,uv.lock,run.sh,configs/*.env,patches/*,LOG.md}`。
- 收尾检查(advisor 之后的修正,见 `run.sh` 的 fixes commit):
  - `sim` 只在本次调用追加到 `run.log` 的那部分里 grep `TimeoutExpired` 和 side-note 字符串(resume 同名 run 时,旧输出不再让新运行报 exit 3);`run_sim` 不捕获 `subprocess.TimeoutExpired`,所以它的 traceback 会进 `run.log`,grep 能匹配。record block 只读有同名 `.log` 的 `<phase>.json`(`--expert-only` 的 `summary.json` 合并了 `expert.json` 的字段,不会再被当成一个 phase)。record block 的 permalinks 用 `stamp.py permalinks`。`fetch` 也先 apply 补丁(Component 5)。
  - `stamp.py check <arch>` 把 arch 路径 `resolve()` 后相对 `$SHVLA_ROOT` 存储,所以 `run.sh` 传 `arch/softhier_mlir_arch.py`(cwd = 实验目录)和任何别的写法等价;**G3.0 调用 `build_arch.sh` 时用同一个文件即可**。只读验证:`stamp.py check third_party/gvsoc/implementation/config/arch/arch.py` → `stamp OK`;对别的文件 → `MISMATCH`,exit 1。
  - 0002 的 host check(实验环境):`SHVLA_SMOLVLA_DIR=smolvla_base` 时 `emit_vision(e2e)`、`expert_data(e2e, {ref KV})` 不带 npz 参数可运行(200 个 preload 数组,`p_kp0` (65, 320)),说明 0005 读的 `ref_kv_k_<L>` 在真实 `e2e_c1_t256.npz` 里存在;同一进程把变量改成 `smolvla_libero`,`vision_npz()` 等随之改变。
  - **留给 G3.0**:`sim` 没有 source `infra/softhier/gvsoc_env.sh`(G1.4 加的,GCC 14.2 toolchain);G3.0 第一次 `sim` 时确认 `build_sw` 能找到编译器,必要时再决定是否由 `run.sh` source 它。
  - 工作区里 `infra/softhier/setup.sh`(modified)和 `run_kernel_tests.sh`(untracked)是另一个 agent 的,没有碰。submodule 目前停在已应用补丁的状态(`ignore = dirty` 隐藏)。
- Commit: 0c2fefe(submodule),ceb1249(uv 项目),e5af70f(patches),ec9c6a7 + 8c8efc8(configs 和 run.sh,后者补可执行位),31795c1(`setuptools<81` pin),b956adf(本条 LOG 的第一版),之后的 fixes commit 和 LOG 修订 commit。
- Paths under `$SHVLA_DATA`: `smolvla_base/`(`model.safetensors` 和 `config.json` 的 symlink,上面五个 npz,共约 1.0 GB),`smolvla_libero/`(symlink,`vision_s1024.npz`,`vlm_c1.npz`,`vlm_c2.npz`,约 1.0 GB),`golden/xc_ref.npz` 和 `golden/xc_ref.npz.kv.npz`(G2.3 的输入)。
- What to inspect: `experiments/smolvla-e2e-baseline/patches/README.md`;`git -C experiments/smolvla-e2e-baseline/softhier-mlir diff` 在 `bash experiments/smolvla-e2e-baseline/run.sh apply` 之后(注意 0003 的 `vision_position_ids` 改动);`run.sh` 的 `sim` 部分(exit code 映射、stamp 拒绝);`configs/*.env` 里的 prepare 列表。

### G2.3 — crosscheck of `golden_ref.py` against softhier-mlir's reference (PASS)

- Time: 2026-11-10。
- What and why: 用提交的 patch 0003 重新生成 reference,再让 `golden_ref.py` 在同样的输入上跑一遍并逐 key 比较,验证 `golden_ref.py` 的捕获点和布局(KV 的 head-major 列、time_emb、xt/vt、n_valid)和 softhier-mlir 的 `smolvla_expert_ref.py` 一致。
- Commands(都在 LeRobot 环境,`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` 下):
  - `bash experiments/smolvla-e2e-baseline/run.sh apply`(`5 patches applied`);
  - `python experiments/smolvla-e2e-baseline/softhier-mlir/softhier_mlir/frontend/smolvla_expert_ref.py --ckpt $SHVLA_DATA/smolvla_base --cams 1 --img-size 256 --out $SHVLA_DATA/golden/xc_ref.npz`(`$SHVLA_DATA/smolvla_base` 里 `model.safetensors` 和 `config.json` 是 symlink,存在;132 个数组,重新生成覆盖了 G2.2 的文件);
  - `python infra/lerobot/golden_ref.py --checkpoint lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --obs from-npz:$SHVLA_DATA/golden/xc_ref.npz --device cpu --out $SHVLA_DATA/golden/xc_golden_cpu.npz --crosscheck $SHVLA_DATA/golden/xc_ref.npz`(10.8 s,self-check 0.0,500 个权重 tensor 相等)。
- Result: **PASS,exit 0。** 比较的 48 个共有数值 key(16 个非 KV key + `kv_k_0..15`/`kv_v_0..15` 共 32 个;`golden_ref.py` 的 npz 共 59 个数组,另有 10 个 `info_*` 字符串和 `info_selfcheck_maxabs` 不参与比较;`xc_ref.npz` 的 132 个数组里其余 84 个不在范围内)的相对误差 `max|a-b| / max(|b|.max(), 1e-6)` **全部是 0.000e+00**(逐位相同,包括 `kv_k_0..15`、`kv_v_0..15` 共 32 个 key,最大 0),容差 1e-4。Per-key 表(相对误差,容差 1e-4):

  | key | rel err |
  | --- | --- |
  | images, vis_pos_ids, lang_tokens, lang_mask, state, noise | 0.000e+00(各自) |
  | vis_out_0, img_emb_0, prefix_embs, prefix_valid, n_valid | 0.000e+00(各自) |
  | time_emb, xt, vt, actions, meta | 0.000e+00(各自) |
  | kv_k_0..15, kv_v_0..15(32 个 key) | max 0.000e+00 |

  结论:PASS。
- 要注意:
  - 逐位相同是因为两边都在 CPU、同一个 torch、同一套 LeRobot 模块上跑,所以这个结果验证的是**捕获点和布局**(KV 的 reshape、cache 读取、flow 的各个量、n_valid),不是两套独立数值实现的一致性。
  - G2.1 的补充检查预计 `vis_pos_ids` 会 diverge,**实际没有**:patch 0003 已经把 `vision_position_ids` 改成 transformers 5 的公式(`idx * (1/n)` 再 clamp),所以和 `golden_ref.py` 的 ids 完全相等。G2.1 entry 里留给 owner 的那个问题("改旧 reference 还是 crosscheck 单独处理")已经由 patch 0003 解决,没有剩余的决定。
- Deviations: 无。没有改 `golden_ref.py`,没有 infra commit。
- Files changed: `LOG.md`(本 entry,step list 状态)。
- Commit: 本 entry 所在的 experiment commit `[G2.3] experiment: log G2.3`。
- Paths under `$SHVLA_DATA`: `golden/xc_ref.npz`、`golden/xc_ref.npz.kv.npz`、`golden/xc_golden_cpu.npz`。
- What to inspect: 上面的 key 表(重跑上面的第三条命令即可看到完整输出)。

### G2.4 — `golden_ref.py` libero mode, tests (d)(e), LIBERO golden reference and `libero prepare` (done)

- Time: 2026-11-10。
- What and why: 实现 `--obs libero:<suite>:<task>:<seed>`(去掉 stub 和 `NotImplementedError` 分支),加测试 (d)(e) 和 LIBERO 版的 (a),生成 `$SHVLA_DATA/golden/libero.npz`,跑完 `run.sh libero prepare`。
- **和任务文字不同的地方(按 lerobot 0.7.0 源码做的,不是按文字):** 在 0.7.0 里 LIBERO 不走 `preprocess_observation` + `LiberoProcessorStep`。`lerobot.sim.LocalVectorEnv` 的 `metadata["canonical_observations"]` 是 True,`env.reset()` 返回的已经是 LeRobot key(`observation.images.image/image2`,`(1,360,360,3)` uint8,**已经旋转 180°**,`observation.state` `(1,8)` float32,eef pos + axis-angle + gripper qpos);旋转和 8-D state 在 `sim/integrations/libero.py:canonical_observation` 里做。`scripts/lerobot_eval.py` 的 `rollout` 因此按 `env.metadata` 选 `prepare_canonical_observation`;`make_env_pre_post_processors` 对 LIBERO 返回**两个空 pipeline**(`LiberoProcessorStep` 只被 XVLA/Cosmos3 的 processor 用)。如果照任务文字再套一遍 `LiberoProcessorStep` 会转两次。所以 `libero_inputs` 的链是:`make_env(LiberoEnvConfig(task=suite, task_ids=[task]), n_envs=1)` → `env.reset(seed=[seed])` → `prepare_canonical_observation`(代码里按 `env.metadata` 选,和 rollout 一样)→ `observation["task"] = env.call("task_description")` → env preprocessor(空)→ checkpoint 保存的 policy preprocessor,overrides 和 eval 相同(`device_processor`、`rename_observations_processor` = `{image→camera1, image2→camera2}`)→ `policy.prepare_images` / `prepare_state`。
- 实现要点:
  - 用 `make_env`、`make_env_pre_post_processors`、`prepare_canonical_observation`、`make_pre_post_processors`,都是 LeRobot 自己的函数;和 `lerobot-eval` 不同的地方:(1) 策略用 `from_pretrained` + 显式 fp32,不是 `make_policy`(权重已逐 tensor 验证);(2) TF32 关、`cudnn.benchmark` 关、`use_deterministic_algorithms`(eval 是 TF32 开、benchmark 开);(3) 没有 `set_seed`,noise 来自独立的 CPU generator。env 配置全是 eval 的默认(360×360 相机、20 Hz、`init_states=True`、`control_mode=relative`)。
  - 原始(旋转前)观测:包住私有的 `env._reset_envs` 记录 sim 返回的 raw obs(断言恰好一次),用于 (e) 测试;**每次 CLI 运行(不只是测试)都会装这个 hook**,它是 LeRobot 私有接口,换版本后 `golden_ref.py --obs libero` 可能在那条断言上失败,到时要改。
  - 断言:`prepare_images` 恰好 2 张图,顺序 camera1、camera2,mask 全 True,形状 `(1,3,512,512)`(360 → 512 的 resize 没有 padding,所以 `vision_position_ids` 的"整幅无 padding"假设成立);`config.image_features` 顺序与此一致。
  - `lang_tokens`/`lang_mask` 取自 preprocessor 输出(不是 `tokenize(PROMPT)`);`info_prompt` = env 的任务字符串,`info_state_raw` = 归一化前的 8-D state。
  - noise:`torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(seed))`(CPU)。图像 round 到 fp16 后再喂模型。
- OQ2 的部分答案(最终答案在 G6b):LIBERO 观测的 key 是 `observation.images.image`、`observation.images.image2`、`observation.state`(sim 输出);**state 是 8 维**([eef pos 3, axis-angle 3, gripper qpos 2]),pad 到 32;env 相机原始 360×360,checkpoint 的 normalizer json 写 256×256 但 VISUAL 是 IDENTITY,真正的图像尺寸由 `resize_imgs_with_padding=(512,512)` 决定(360 → 512,方形,无 padding);语言 48 token 中本任务有 20 个有效;prefix = 2×64 + 48 + 1 = 177,有效 149。
- Tests(`infra/lerobot/tests/test_golden_ref.py`):`(a)` LIBERO 形状(`images [2,3,512,512]` fp16、`vis_pos_ids [1024]`、`vis_out_<c> [1024,768]`、`img_emb_<c> [64,960]`、KV `[177,320]`、`meta [50,32,10,177,n_valid]`、`info_state_raw [8]`);`(d)` golden `actions[:, :7]` 对 `policy.predict_action_chunk(batch, noise=noise)`;`(e)` state 在 pad 前是 `(1,8)`、pad 后 `[8:]` 全 0,camera1/camera2 的 batch 图像与 raw env frame 旋转 180° 逐位相等(检查一行,再检查整幅;同样的 uint8/255 fp32 运算,所以是精确相等);原来的 "libero not implemented" 测试换成了 4 个畸形 `libero:` 字符串必须失败。**12 passed,96.6 s**(`systemd-run` 下,`uv run --directory infra/lerobot pytest`);`ruff check` 通过。
  - (d) 的做法:0.7.0 的 `predict_action_chunk(batch, *, noise=None, **rtc_kwargs)` 有 `noise` kwarg,返回 `[B, 50, 7]`(已经 unpad,在归一化空间,没有 postprocessor)。它自己会跑 `prepare_images`,而 golden 喂的是 fp16 round 过的图像,直接用未 round 的 batch 会差 1e-3 量级。所以测试里把 batch 的相机图像换成 `(golden_fp16 + 1) / 2`(512×512,`resize_with_pad` 此时是恒等),并断言 `prepare_images` 把它们还原成 golden 的图像(实测 max-abs **0.000e+00**),然后比较:max-abs(golden `actions[:, :7]`,`predict_action_chunk`)= **0.000e+00**(要求 ≤ 1e-5)。测试里先 `policy.reset()`、浅拷贝 batch(`_get_action_chunk` 会改写 batch 的 key)、`torch.no_grad()`。
- `libero.npz`:`libero:libero_spatial:0:0`,`lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`,CUDA,22 s,18 060 618 字节,SHA-256 `d89540e9c9288415b640465c39e5d4ec5ecbc55fccf7f818d59a0ca46b47a4a1`,62 个数组。**确定性:同一命令第二次运行 SHA-256 完全相同**(也包含 osmesa 渲染,没有不确定的 key)。`meta = [50, 32, 10, 177, 149]`,self-check max-abs 0.0(`model.sample_actions` 对捕获的 flow;配置里 `num_steps=10`),|K| max 18.682,|V| max 5.025。任务:`pick up the black bowl between the plate and the ramekin and place it on the plate`;raw 8-D state `[-0.2112, -0.0111, 1.1742, 3.1409, -0.0028, -0.0877, 0.0387, -0.0387]`。actions(归一化空间):全部 32 维 mean -0.0143,std 0.458;前 7 维 mean -0.0650,std 0.9781;两者 max|a| 都是 2.5288;第 8–32 维(pad)std 0.0049;第一步前 7 维 `[-0.097, -0.405, 0.273, -0.062, 0.351, 0.067, -0.968]`。图像取值范围 [-0.992, 0.910]。
- `run.sh libero prepare`(`systemd-run` 下,**5 步全 exit 0**):1) `smolvla prepare --seq 1024`:22 s,282 MB;2) `prepare-vlm --cams 1`:10 s,348 MB;3) `prepare-vlm --cams 2`:10 s,363 MB,`tokens=177 (img 128, lang 48 of which 5 valid, state 1)`;4) `smolvla_expert prepare --ref libero.npz`:3 s,208 MB,**`n_valid 149`**,KV 3.5 MiB;5) `smolvla_e2e prepare --ref libero.npz`:1 s,18 MB,输出 `e2e_c2_t1024.npz`,打印:`2 camera(s) x 1024 SigLIP tokens -> 64 image tokens each; prefix 177 tokens (149 valid), expert chunk (50, 32)`。
  - 注意:`prepare-vlm`(步骤 2、3)用自己的合成 prompt,报的是 "lang 48 of which 5 valid"(prefix 177 token,但 valid 只有 5 个语言 token),而 golden reference 的语言有 20 个有效 token(n_valid 149)。这两个量在 G3 之后比较 VLM prefix 时要区分(step 4、5 用的是 golden 的 149)。
- obs 字符串(`libero:libero_spatial:0:0`)和输出路径(`$SHVLA_DATA/golden/libero.npz`)与 `configs/libero.env` 完全一致,没有对齐改动。
- Deviations: (1) 链和任务文字不同,见上(按 0.7.0 源码做);(2) `Co-Authored-By` 用 harness 的 `Claude Sonnet 5.5`;(3) `libero_inputs` 依赖私有的 `env._reset_envs` 来取 raw 帧(只为测试);(4) 没有用 `make_policy`。
- Files changed: `infra/lerobot/golden_ref.py`、`infra/lerobot/tests/test_golden_ref.py`(infra 59218d9);`LOG.md`(本 entry,step list 状态)。
- Commit: infra 59218d9(`[G2.4] infra: golden_ref libero mode, tests (d)(e)`);本 entry 在紧接着的 experiment commit 里。
- Paths under `$SHVLA_DATA`: `golden/libero.npz`;`smolvla_libero/{vision_s1024,vlm_c1,vlm_c2,expert,e2e_c2_t1024}.npz`。
- What to inspect: `golden_ref.py` 的 `libero_inputs`(链的顺序)和 `tests/test_golden_ref.py` 的 (d)(e);`np.load('$SHVLA_DATA/golden/libero.npz')['info_prompt']`、`info_state_raw`、`meta`。

### G2.5 — `infra/lerobot/README.md` (done)

- Time: 2026-11-10。
- What and why: 写 `infra/lerobot/README.md`:setup(`uv sync`、`source env.sh`)、osmesa / libstdc++ 说明、LIBERO `config.yaml` 首次 import 的处理、内存 cap 命令、`golden_ref.py` 三个 mode 的用法(含 LIBERO 链的顺序)、`libero_eval.py` 的用法和已核实的 `lerobot-eval` flag、key 表(指向设计文档的 "Golden reference layout",只写了 LIBERO 的形状)、确定性设置、实测速度(golden ref 稳定约 9 s,eval 约 6.5 it/s)、测试命令。
- Result: README 里的数字都来自上面的 LOG entry(G6a.1、G6a.2、G2.1、G2.4);没有新的测量。
- Deviations: 无。
- Files changed: `infra/lerobot/README.md`(infra 84ad868);`LOG.md`(本 entry,step list 状态)。
- Commit: infra 84ad868(`[G2.5] infra: lerobot README`);本 entry 在紧接着的 experiment commit 里。
- What to inspect: `infra/lerobot/README.md`。

### G1.5 — `run_kernel_tests.sh`, the four kernels (done)

- Time: 2026-11-10。
- What and why: 写 `run_kernel_tests.sh` 并运行 `gemm`、`norm`、`acti`、`attn`。
- Makefile 读出的事实(`third_party/gvsoc/implementation/Makefile`,README 只写了 `make <Kernel>-run`):
  - `<k>-run` 依赖 `<k>-pre`;`<k>-pre` = `<k>-clean <k>-cfg <k>-pld <k>-hs`;`<k>-hs` = `cfg=<implementation/config/arch/arch.py> app=<sw dir> make -C .. hs`,顶层 `hs` = `config` + 硬件 `all` + `sw`。**所以每个 kernel 目标都会重建硬件**(用 `implementation/config/arch/arch.py`,与 G1.4 同一个 arch),而不只是 preload。`<k>-run` 本身只跑 `pld=<preload.elf> make -C .. run`。
  - 因此脚本不是 `make <k>-pre <k>-run`(会重复一次 pre),而是 `make <k>-pre`(不计时)再 `cd third_party/gvsoc && timeout $SHVLA_SIM_TIMEOUT make run pld=<preload.elf>`(与 `<k>-run` 的 recipe 相同),这样 2 h 上限只覆盖模拟器。这是对设计里 `make <k>-pre <k>-run` 的偏离。脚本结尾总是 `stamp.py write implementation/config/arch/arch.py`(失败时也写,因为硬件被重建了)。
  - 软件目录:gemm `SummaGEMM`,norm `RMSNorm`,acti `Activation`,attn `FlatAttention`。
  - **四个 `main.c` 都没有任何结果打印或检查**:只有 `flex_timer_start/end` 和 `flex_eoc(0)`;`*_preload.py` 里的 `O_golden` 只是占位,没有地方比较输出。所以没有"kernel 自己的 check 字符串"。Pass 规则:模拟器 exit 0 + 出现 `[Performance Counter]: Execution period is N ns`(与 softhier-mlir `PERF_RE` 相同的正则)+ 没有 `No entry found for burst`、`Executing illegal instruction`、`Received out-of-bound request`。这是对设计"kernel 自己的 check output"的偏离,没有编造字符串。
- 第一次运行 FAIL(gemm,模拟器启动即报错):`libDRAMSys_Simulator.so: undefined symbol: ..sc_api_version_2_3_4..`。诊断:G1.3 的 `setup.sh` 没有设 `LD_LIBRARY_PATH`,Makefile 的 DRAMSys 自检 dlopen `libsystemc.so.2.3` 找不到,段错误,于是误判预编译库坏了,走了 40 分钟重编译;重编译用 DRAMSys 自带的 SystemC 2.3.4 头文件,与 2.3.3 的运行时不匹配。预编译的 `soft_hier/libDRAMSys_Simulator.so` 引用的是 `sc_api_version_2_3_3`,与我们的 SystemC 2.3.3 匹配。修:把预编译库拷回 `third_party/DRAMSys/`(重编译的留作 `libDRAMSys_Simulator.so.rebuilt-234`),设了 `LD_LIBRARY_PATH` 后自检 `a.out` 退出码 0(`close dram: 0`),没设时 139。`setup.sh` 现在在 make 之前 export 这个路径(以后不会再误重编译)。这不是模拟器模型 bug,不在 side-note 表里。
- Command: `bash infra/softhier/run_kernel_tests.sh`(第二次运行,全部通过)。
- Result(`sim ns` 来自 `[Performance Counter]: Execution period is N ns`;`build s` 是 `<k>-pre`,含 clean、config、preload、硬件增量重建和 SDK 编译;`sim wall s` 是 `make run`):

  | kernel | result | sim ns | build s | sim wall s |
  | --- | --- | --- | --- | --- |
  | gemm | PASS | 21708 | 10 | 67 |
  | norm | PASS | 13177 | 11 | 4 |
  | acti | PASS | 8060 | 11 | 4 |
  | attn | PASS | 298126 | 12 | 1280 |

  - 硬件在未改动的树上增量重建加软件编译,每个 kernel 约 10–12 s;没有接近 2 h 上限(最长的 attn 约 21 min)。
  - 之后 `stamp.py check` 输出 `stamp OK`;`du -sh third_party/gvsoc` 5.4 GB;日志 40–85 KB,不大。
- Files changed: `infra/softhier/run_kernel_tests.sh`、`infra/softhier/setup.sh`(都在 74fbf46);本次 LOG.md(本 entry,step list 状态)。`third_party/gvsoc/third_party/DRAMSys/` 里的库是构建产物,不提交。
- Commit: infra 74fbf46(`[G1.5] infra: run_kernel_tests.sh; setup.sh sets LD_LIBRARY_PATH for the DRAMSys self-test`)。
- Paths under `$SHVLA_DATA`: `softhier/g1/<k>-pre.log`、`softhier/g1/<k>.log`。
- What to inspect: `grep Performance $SHVLA_DATA/softhier/g1/*.log`;`gemm.log` 开头的命令行(`--trace=/chip/cluster_0/redmule`)。

### G1.6 — `infra/softhier/README.md` (done)

- Time: 2026-11-10。
- What and why: 写 `infra/softhier/README.md`:前置条件(含 apt 列表)、各脚本和 `stamp.py`、`gvsoc_env.sh`、stamp 规则、kernel 目标会重建硬件、toolchain 选择(GCC 14.2)、`read -t 30`/dash 说明、DRAMSys 自检需要 `LD_LIBRARY_PATH`、切换到 fork 的步骤(D2)、实测时间和磁盘占用;side-note 表只引用设计文档。
- Result: 实测数字取自 G1.3–G1.5(preparation 1857 s,首次 `make hw` 98 s,`<k>-pre` 10–12 s,gemm 67 s,norm 4 s,acti 4 s,attn 1280 s;磁盘 4.8 GB → 5.4 GB)。没有跑过"自检通过"那条路径的 preparation,README 里如实写了没测。
- Files changed: `infra/softhier/README.md`(infra 提交);本次 LOG.md(本 entry,step list 状态)。
- Commit: infra 89b92d8(`[G1.6] infra: infra/softhier/README.md`)。
- What to inspect: README 的 "Kernel tests rebuild the hardware" 和 "Notes from setup" 两节。

### G6b.1 — `libero_spatial` 10 × 10 closed-loop success rate (OQ2–OQ4 settled)

- Time: 2026-11-10。
- What and why: 对 `lerobot/smolvla_libero`(revision `a24f1f380461b665158d52acd140152c79e8b36d`)在 `libero_spatial` 全部 10 个 task 各跑 10 episode(共 100),GPU fp32,同步策略(SR-A:仿真在推理时暂停),得到基线成功率。
- Command: `source infra/lerobot/env.sh && python infra/lerobot/libero_eval.py --run-name g6b_spatial_10x10_b2 --n-episodes 10 --batch-size 2`(12 GB cap 下,seed 1000)。
- 第一次尝试 `--batch-size 5`(run name `g6b_spatial_10x10`)失败:约 70 s 后 `RuntimeError: Environment process exited with code -9`,即 env worker 进程被 cap 的 OOM kill(并发的模拟器作业占着内存,eval 启动时整机已用约 15 GB)。没有改 `libero_eval.py`,把 batch size 降到 2 并换 run name 重跑(`g6b_spatial_10x10` 目录只含失败的残留,没有 summary)。3 个进程各约 3 GB RSS,没有再触到 cap。
- Results(`summary.json`,Wilson 95 %,k/n 由 `eval_info.json` 的 `successes` 计算):

| task | 成功 | 成功率 | Wilson 95 % |
| --- | --- | --- | --- |
| 0 | 9/10 | 90 % | [59.6, 98.2] |
| 1 | 6/10 | 60 % | [31.3, 83.2] |
| 2 | 7/10 | 70 % | [39.7, 89.2] |
| 3 | 6/10 | 60 % | [31.3, 83.2] |
| 4 | 5/10 | 50 % | [23.7, 76.3] |
| 5 | 1/10 | 10 % | [1.8, 40.4] |
| 6 | 9/10 | 90 % | [59.6, 98.2] |
| 7 | 4/10 | 40 % | [16.8, 68.7] |
| 8 | 4/10 | 40 % | [16.8, 68.7] |
| 9 | 8/10 | 80 % | [49.0, 94.3] |
| overall | **59/100** | **59.0 %** | **[49.2, 68.1]** |

- Wall time / memory:wrapper 报 wall 3781 s(约 63 min;`lerobot-eval` 自己报 `eval_s` 3756 s,37.6 s/episode),`batch_size=2`,最大单个进程 RSS 4145 MiB(`RUSAGE_CHILDREN`),`/usr/bin/time` 的 Maximum RSS 4244020 kB;没有触到 12 GB cap。**这个 wall time 是在争用下测的**:另一个 agent 同时在跑 CPU 密集的模拟器(GVSoC)作业,而 MuJoCo 用 osmesa 在 CPU 上渲染,所以单独跑应当更快。
- 已发表的数字(OQ4):**没有找到 `lerobot/smolvla_libero` 在 `libero_spatial` 上的已发表成功率。** HF cache 里的 model card(`snapshots/a24f1f38…/README.md`)是 LeRobot 自动生成的模板,没有任何评测数字;wiki 里的 `softhier_vla_wiki/resources/models/smolvla.md:18` 只有论文 Table 2 的 LIBERO **平均**(SmolVLA 0.24B 82.75、0.45B 87.3、2.25B 88.75;multi-task,无 robot pretraining),没有 per-suite 的 spatial 数字;`wiki.py find` 也没有 spatial 数字。所以**没有可直接比较的已发表 spatial 数字**。论文的 87.3(0.45B,四个 suite 的平均)对应的是论文自己训练的模型和设置,不是这个 checkpoint,也不是 spatial 单 suite。本次 59 % 低于该平均值很多(区间上限 68 %),但因为模型、suite、设置都不同,不能据此判定是 bug 或不是 bug。**这个差距是一个未解决的疑点**,下一步值得查:checkpoint 的训练配置(`train_config.json`)是不是针对 spatial 以外的设置、eval 的 seed、相机分辨率(env 360 × 360,checkpoint config 写 256 × 256)、max steps 280 是否够用、task 5/7/8 的视频。
- OQ2(state 维度和 observation keys):env 的 observation 是 `pixels` 里的 `image`、`image2`(360 × 360 × 3,`LiberoEnvConfig.observation_height/width = 360`,`envs/configs.py:480`),加 `agent_pos`;`agent_pos` 是 8-D = eef 位置 3 + axis-angle 3 + gripper qpos 2(`lerobot/sim/integrations/libero.py:50–55`)。checkpoint 的 `config.json` 里 `observation.state` shape `[8]`,`observation.images.camera1/camera2` shape `[3,256,256]`,`max_state_dim` 32(模型内部补零到 32)。eval 里用 `--rename_map` 把 `image`/`image2` 改成 `camera1`/`camera2`。与 G2.4 一致。
- OQ3(控制频率):**20 Hz**。`LiberoEnvConfig.fps: int = 20`(`envs/configs.py:470`,注释说必须等于 robosuite 默认 control_freq),`gym_kwargs` 里 `"control_freq": self.fps`(`envs/configs.py:500`);`LiberoEnv.__init__(control_freq=20)`(`sim/integrations/libero.py:164`)存为 `self.control_freq`(`:199`)并传给底层 env(`:307`);robosuite 的 `control_freq=20` 默认值在 `robosuite/environments/base.py:95`,LIBERO 的 `env_wrapper.py:27` 同样默认 20。eval 没有覆盖这个 flag,所以实际是 20 Hz(episode 最多 280 步 ≈ 14 s 仿真时间)。(路径相对 `infra/lerobot/.venv/lib/python3.12/site-packages/`。)
- Deviations: batch size 5 → 2(上面的 OOM);没有改任何代码。
- Files changed: 只有 `LOG.md`(本 entry,step list 状态)。没有 infra 提交。
- Paths under `$SHVLA_DATA`: `eval/g6b_spatial_10x10_b2/`(`eval_info.json`、`summary.json`、`task_descriptions.json`、`videos/libero_spatial_<t>/eval_episode_<e>.mp4`);`eval/g6b_spatial_10x10/` 是失败的 batch-5 尝试。
- What to inspect: `$SHVLA_DATA/eval/g6b_spatial_10x10_b2/summary.json` 和 `eval_info.json`;task 5(1/10)、7、8(4/10)的 mp4。

#### G6b.1 addendum(补充核对)

- OQ3 补充:checkpoint 的 `config.json` 里 `chunk_size = n_action_steps = 50`,所以每次推理后整段 50 个动作都执行,在 20 Hz 下一个 chunk 是 2.5 s 仿真时间(30 Hz 则 1.67 s);`train_config.json` 里 env 段是 `fps: 30`、`task: libero_spatial`、`observation 360 × 360`、`dataset lerobot/libero`、25000 步、batch 32。env 的 20 Hz 与 `train_config.json` 的 30 不一致(设计文档 OQ3 已预期这一点);eval 用的是 20 Hz(没有传 `--env.fps`)。
- 对 59 % 的解读:设计文档 G6b 没有设验收区间,只要求报告成功率和区间并与已发表数字并列。这里没有已发表的 spatial 数字(model card 全文 62 行,没有 results 或 model-index,只有 `datasets: lerobot/libero`)。一个值得先查的疑点是训练配置写 30 Hz 而 eval 默认 20 Hz;另一个是 `n_action_steps = 50` 开环执行 2.5 s。这两点不是已确认的原因,只是没有证伪的假设,本步没有试。
- OOM 的原因没有查清:`journalctl --user` 在那个时间段没有 oom 记录,不能区分 12 G cap 触发(batch 5 时约 6 个进程 × 约 3 GB 本身就可能超过 12 G)和整机 OOM(当时整机可用内存只有约 2 GB)。所以上面"并发作业占着内存"只是可能因素之一,不是定论。
- 内存补充:batch 2 时进程树里有 3 个进程各约 3 GB(合计约 9 GB);"最大单个进程 RSS 4145 MiB"低估了 cap 实际约束的整棵树的占用。

### G3.0 — arch file and build stamp (done)

- What: `arch/softhier_mlir_arch.py` generated in the experiment env with `Arch().to_py()` from softhier-mlir b11cd5f (header names the permalink and the command). Fields: 4x4 clusters, 3 cores, RedMulE 128x32, NoC link 512, `hbm_chan_placement = [4, 0, 0, 4]`, instruction memory 0x10000 (64 KB). The file has every attribute `Arch()` emits; no unknown field. The soft_hier_release template additionally names `cluster_heap_base`, `cluster_heap_end` and `hbm_node_aliase`, which `Arch()` does not emit: `flex_cluster.py` defaults `hbm_node_aliase` to 1 (line 78) and the SDK defaults `ARCH_CLUSTER_HEAP_BASE` to 0 (`flex_runtime.h`); the heap-start code in `flex_runtime.h` is commented out and `flex_alloc_api.h` only declares `flex_hbm_malloc/free`. So the omission is a documented deviation that nothing in the ladder uses.
- Build: the previous agent ran `build_arch.sh` on this file (`$SHVLA_DATA/softhier/build/softhier_mlir_arch.log`: `make hw` exit 0 after 7 s, an incremental build after the 98 s first build of G1.3); `stamp.py check arch/softhier_mlir_arch.py` prints `stamp OK`. No rebuild this session.
- run.sh fix kept: `run.sh sim` exports `SYSTEMC_HOME` and `LD_LIBRARY_PATH` (SystemC lib64 + `third_party/DRAMSys`), without which the simulator segfaults at start (exit 139). `gvsoc_env.sh` is not sourced (it would activate the GVSoC venv). Evidence that the toolchain and gapy are found from the experiment venv's python: rung 1 got through `build_sw`, launched `gvsoc_launcher`, and one shape passed with a ROI.
- Commit: experiment 0c8e7a2.

### G3.1 — flex_alloc_api.h (infra 22d549c) and rung 1 (`gemm`): FAIL, STOPPED (R2)

- Header fix (committed by the previous agent, checked now): GVSoC local branch `shvla/soft_hier_release` commit 4558184 adds `soft_hier/flex_cluster_sdk/runtime/include/flex_alloc_api.h`, copied from `origin/bowwang-dev/softhier-deeploy` (`deeploy_include/`, 99b60c3). Against that blob only the trailing newline differs. softhier-mlir `runtime/sh_ops.c` includes it and rung 1 builds. `.gitmodules` URL unchanged.
- Rung 1: `run.sh published sim rung1-gemm -- tests/gvsoc/run.py gemm` (default 7 shapes), exit 1. Only `256x768x192:256,256,192` PASSes (ROI 32050 ns, samples bad=0, wall 4.1 s). The other six shapes FAIL with `roi=None`, no `GEMM_PASS`, wall 889 / 2377 / 382 / 970 / 1187 / 261 s. Whole run 1:42 h wall, `/usr/bin/time -v` max RSS 18.5 GB, system time 1239 s (the host was swapping). Note: for non-e2e scripts `run.sh` applies the 7200 s timeout to the whole python process, not to each simulator process.
- Diagnosis: reran `256x256x256` alone (`rung1b-gemm256`) under a watchdog that kills `gvsoc_launcher` above 13 GB RSS: RSS reaches 13.4 GB in about 35 s of wall time and no ROI is printed; stdout is full of `make: *** No rule to make target 'ice_power'. Stop.` / `Warning: failed to run 'make ice_power config_file=...arch/softhier_mlir_arch.py ice_geo_file=geo.json ice_power_trace_file=ice_power_trace.txt'`. Source: `third_party/gvsoc/core/engine/src/time/time_engine.cpp` `TimeEngine::power_capture_advance` (GVSoC commits 074b153 .. d806a84 "[POWER]", 2026-06-21): every `PWR_INTERVAL_PS` (default 100000000 ps = 100 us simulated, `core/engine/CMakeLists.txt`, `Makefile pwr_interval_ps`) it stops the power capture, runs `std::system("make ice_power ...")` (fails: cwd is the app dir, not the GVSoC root) and restarts the capture. The only passing shape simulates 32 us, below the first interval; the shapes that cross 100 us fail. This is not in the side-note table or SIMULATOR_NOTES.md; the owner's softhier-mlir apparently expects a GVSoC without this power-capture engine. Whether the memory growth is the power engine's capture state or something else is not separated.
- Not fixed (not a known notes item, so STOP per R2). Proposed fixes for the owner / next agent: (a) cheapest: rebuild the hardware with `pwr_interval_ps` effectively infinite (`pwr_interval_ps=9000000000000000 cfg=<arch> make hw`; build_arch.sh would need to pass it and the stamp record it) and check that memory stays flat; (b) a local GVSoC patch making `TimeEngine::init` skip `start_capture` / `power_capture_advance` unless an env var is set; (c) pin GVSoC to the commit before 074b153 (2026-06-21). (a)/(b) change the simulator build.
- Peak RSS: 13.39 GB (watchdog kill, `rung1b-gemm256`); first full run 18.5 GB (max child RSS, `/usr/bin/time`).
- Run dirs: `$SHVLA_DATA/runs/published/rung1-gemm` and `rung1b-gemm256` (run.log of rung1-gemm is empty: python buffers stdout in the pipe; use `PYTHONUNBUFFERED=1`).
- Rungs 2-5 not run.

### G3.1 follow-up — run.sh memory cap, `pwr_interval_ps` knob, `GVSOC_POWER_CAPTURE` gate: growth NOT fixed, STOPPED

- Step 1, `run.sh` (this commit): `sim` runs the command inside `systemd-run --user --scope -q -p MemoryMax=${SHVLA_SIM_MEMMAX:-13G} -p MemorySwapMax=0`, exports `PYTHONUNBUFFERED=1`, and maps a cgroup OOM kill (exit 137, or `oom_kill` > 0 in the scope's `memory.events`, or "Killed" in the log) to exit code 4 with a `MEMORY CAP` message. The record block prints `memory: peak N MB ... oom_kill events: K` from the scope's `memory.peak` (cgroup-wide, so it includes page cache; it is not the max RSS). Tested: 200 MB allocation under 13G exit 0 / peak 206 MB; 400 MB under `SHVLA_SIM_MEMMAX=100M` exit 4. For non-e2e scripts the 7200 s `timeout` covers the whole python process. Deleted the stale dirs `$SHVLA_DATA/runs/published/rung1-gemm` and `.../rung1b-gemm256` (their paths are in the G3.1 entry above; not in git).
- Step 2, infra 18ad903: `build_arch.sh` passes `pwr_interval_ps=$SHVLA_PWR_INTERVAL_PS` (default 9000000000000000, `infra/env.sh`) to `make hw`; `stamp.py` records `build/CMakeCache.txt` `PWR_INTERVAL_PS` and `check` also fails if it differs from the variable (checked: `SHVLA_PWR_INTERVAL_PS=100000000` gives MISMATCH). Evidence the knob took effect: `CMakeCache.txt` `PWR_INTERVAL_PS:STRING=9000000000000000`, and all four `flags.make` under `build/` carry `-DPWR_INTERVAL_PS=9000000000000000LL`; `time_engine.cpp` was recompiled (19 s build). `stamp.py check` OK.
- Step 3, results (all via `run.sh published sim`, capped):

| build | shape | result | peak | wall |
| --- | --- | --- | --- | --- |
| G3.1 (interval 100 us), earlier | `256x256x256` | killed by watchdog at 13 GB | 13.39 GB | ~35 s |
| knob 9e15 | `256x256x256` (`rung1a-gemm256`) | FAIL roi=None, cap kill, exit 4, no `ice_power` line in run.log | 13312 MB | 31 s |
| knob + gate (GVSoC d42c382, env unset) | `256x256x256` | FAIL, cap kill | 13312 MB | 42 s |
| knob + gate | `256x256x256:256,256,256` | FAIL, cap kill | 13312 MB | 48 s |
| knob + gate | `256x768x192:256,256,192` (`rung1c-gemm768`) | PASS, GEMM_PASS, ROI 32050 ns (identical to the earlier run), samples bad=0 | 482 MB | 4.0 s |
| knob + gate | `256x256x192:256,256,192` | PASS, ROI 11054 ns | 333 MB | 3.5 s |
| knob + gate, cap 4G | `256x256x256:256,256,256` (`rung1d-k256`) | FAIL, cap kill | 4096 MB | ~25 s |
| knob + gate, cap 4G | `256x768x256:256,256,256` | FAIL, cap kill | 4096 MB | 12 s |

- Corrected diagnosis: the power-capture theory in the G3.1 entry is wrong as a cause. Removing the `make ice_power` calls (knob) and the capture itself (gate) leaves the growth unchanged; the `ice_power` warnings appeared only because the hung run kept going past 100 us of simulated time. 256x256x256 is about 16.8 M MACs (cheaper than the 37.7 M-MAC shape that passes in 32 us), so it should finish in tens of us. The data separates the shapes by the K tile: tile K = 192 passes, tile K = 256 (including the default tile of `256x256x256`) never prints a ROI and grows without bound (about 300 MB/s). Working hypothesis: the simulated program does not terminate (hang or livelock) with K tile 256 in this arch/SDK combination, and something in the simulator grows while it runs. Five of the seven default rung-1 shapes use K tile 256. gdb attach is blocked (`ptrace_scope`), so which model spins is unknown.
- GVSoC commit d42c382 (amended from 6710e0a: the first message called it a fix) keeps the env gate in `soft_hier/gvsoc_core.patch` and the `core` working tree; it is harmless and removes the failing `system()` call but is not the fix. Parent pointer: infra 2088d59.
- Proposals (need the owner, they change the simulator or arch): (a) bisect with GVSoC before 074b153 or the upstream release tree to see whether `256x256x256:256,256,256` completes; (b) check the TCDM budget of K-tile-256 against `cluster_tcdm_size = 0x100000` and the owner's real arch file (the arch here is `Arch()` output; `implementation/` arch differs in unknown fields); (c) find the spinning model with `perf`/`gdb` where ptrace is permitted, or run with `--trace=` for a short bounded time. Rungs 1 (full), 2-5 not run.
- Run dirs (`$SHVLA_DATA/runs/published/`): `rung1a-gemm256`, `rung1c-gemm768`, `rung1d-*`.
- **What to inspect**: the table above; `rung1a-gemm256/run.log` (no `ice_power`, no `[gemm]` line); `git -C third_party/gvsoc show d42c382`.

### G3.1-diag — diagnosis of the K-tile-256 hang (no GVSoC change; root cause NOT yet found)

- 摘要:只读诊断,没有改 GVSoC。排除了寄存器位宽问题和 RedMulE 模型差异(deeploy 与 release 的 RedMulE 只差数值舍入和功耗);扫描显示失败条件是 RedMulE 的 M >= 128 且 N(= tile K) >= 256,与 TCDM 容量、preload 无关;进程不是死锁,是模拟变得极慢且内存线性增长(约 480 MB/s)。根因没找到,下面写明已排除的和下一步。

- Scope: diagnosis only. No GVSoC source change, no branch switch, no rebuild. All runs one at a time under `SHVLA_SIM_MEMMAX=4G` and `timeout 120`; GVSoC at d42c382 (`shvla/soft_hier_release`), arch `arch/softhier_mlir_arch.py`.
- **d42c382 (`GVSOC_POWER_CAPTURE` gate): not a fix.** Harmless (it only removes the failing `system("make ice_power")` path), kept for now; revert candidate if the owner wants the GVSoC tree to match upstream release. `patches/README.md` lists only softhier-mlir patches, so it is not touched.
- **Task 1, diff `HEAD..origin/bowwang-dev/softhier-deeploy`** (115 files; the deeploy branch is the older fork: it lacks the power model, `transpose_engine`, `memory.cpp`, fp8e5m2, c2c platform, `implementation/`). Functional hunks relevant to the hang:
  - `soft_hier/flex_cluster/light_redmule.cpp`: release has `gemm_tile_energy` power accounting (lines ~195, 295, 450), an fp8e5m2 path, and `matmul_*` accumulating in fp32 (rounded to fp16 once), while deeploy rounds to fp16 after every FMA. Only numerics and power. Register decode, tile loops, `queue_depth` and the m/n/k handling are unchanged. `light_redmule.py` only removes `tech_node`/`get_gemm_tile_pJ`.
  - `flex_cluster_sdk/runtime/include/flex_redmule.h`, `flex_group_barrier.h`, `flex_cluster_arch.h`: identical on both branches. `flex_dma_pattern.h`: deeploy adds `flex_dma_async_2d` and splits `flex_dma_sync_2d` into 1D transfers for HBM destinations (not used by `sh_gemm`, which calls `bare_dma_start_2d`). `flex_runtime.h` / `flex_alloc.h` / `flex_memory.ld` / `flex_start.s`: the HBM allocator and `.l1`/`.hbm` sections exist only in deeploy (hence `flex_alloc_api.h` was copied in 4558184); deeploy's `hbm_*` macros omit the `(uint64_t)` casts; `flex_start.s` uses a hard-coded stack shift 0xa instead of `ARCH_CLUSTER_STACK_OFFSET`.
  - iDMA backend (`soft_hier/gvsoc_pulp.patch` hunk for `pulp/idma/be/idma_be.cpp`, patch-diff lines ~356-500; live file `third_party/gvsoc/pulp/pulp/idma/be/idma_be.cpp:61-112`): the release tree has a **new transfer-regulation queue** (`enqueue_transfer` -> `regulation_queue` -> `transfer_regulation_handler`, a second `transfer_ack_queue`, `update()` enqueues the regulation event). Deeploy has the original single-transfer backend. `transfer_regulation_handler` returns on "Src Conflict" without re-arming itself. Also differing: the FlooNoc `Router` / `NetworkInterface` / `FlooNoc` hunks (about 100 added lines each in release), `ctrl_registers.cpp` (deeploy: timestamp-queue barrier and sync groups; release: barrier in `cluster_registers.cpp`), `hbm_ctrl.cpp`, `cluster_unit.py`, `flex_cluster_arch.py` (`hbm_node_addr_space` 0x2000000 vs 0x200000, `hbm_node_aliase`), the core patch (power engine, `time_engine.cpp`).
  - softhier-mlir trigger (`runtime/sh_gemm.inc.c:76,99`): `flex_redmule_config(c.tm, c.tk, c.tn)` then `flex_redmule_trigger`, the same SDK header. The config packs `(k<<16)|m` and `n` in a second register; the model unpacks 16-bit fields (`light_redmule.cpp` ~780-790). K=256 and even 1024 fit. The RedMulE 128x32 limits do not bite (`x_buffer = 128 x 2048 x 2`). TCDM need is below `cluster_tcdm_size = 0x100000` in every run below (max 557056 B). Task 1 conclusion: **no register-width problem and no RedMulE-model difference that can hang**; the release-only code paths that remain (iDMA regulation queue, FlooNoc hunks, `memory.cpp` TCDM model) are candidates; see Task 3 for which the traces support.
- **Task 2, sweep.** Shape `128x128xK:128,128,K` (pipeline=1; TCDM bytes = 1024*K + 32768; limit 1048576):

| K = tile K | TCDM B | result | ROI ns | peak MB |
| --- | --- | --- | --- | --- |
| 64 | 98304 | PASS | 3860 | 444 |
| 128 | 163840 | PASS | 4605 | 315 |
| 192 | 229376 | PASS | 5368 | 331 |
| 224 | 262144 | PASS | 5742 | 341 |
| 240 | 278528 | PASS | 6114 | 342 |
| 256 | 294912 | FAIL (cap) | - | 4096 |
| 258 | - | FAIL (cap) | - | 4096 |
| 320 | 360448 | FAIL (cap) | - | 4096 |
| 384 | 425984 | FAIL (cap) | - | 4096 |
| 512 | 557056 | FAIL (cap) | - | 4096 |

  Other dimensions (tile K 256 unless noted): `32x32x256:32,32,256` PASS 3278 ns; `8x128x256:8,128,256` PASS; `32x128x256` PASS; `64x128x256` PASS 4636; `96x128x256` PASS 5352; `128x32x256:128,32,256` FAIL; `128x128x256:128,128,256,0` (no pipeline) FAIL; `128x128x256:128,128,128` (tile K 128, K total 256) PASS 5749; `128x128x384:128,128,192` PASS; `128x128x1024:128,128,128` PASS 12491 ns; `256x256x192:256,256,192` PASS 11054 (TCDM 524288 B). So **it is not the TCDM budget and not K alone**: the hang needs tile M >= 128 together with tile K >= 256 (threshold between 240 and 256 at tile M 128; tile M 96 at tile K 256 passes; tile M 256 at tile K 192 passes). Tile N and pipelining do not matter. At tile M 128 and tile K 256 the X block is 128 rows x 512 B = 64 KiB; the first 2D DMA after the Y-zero DMA is the X (and W) load from HBM with row size 0x200.
- **Task 3, 找 spinning model(结论:没有找到,前面一版的采样点在 hang 之前,已撤回).** gdb 能用,前提是由 gdb 自己启动进程:`gapy` 会 exec `gvsoc_launcher --config=gvsoc_config.json`(`strace -f -e execve` 查到),直接在 gdb 下跑这条命令,另起一个计时脚本给 gdb 发 SIGINT,在 ptrace_scope 1 下可用;`perf record -g` 也可用(只有用户态,模型没有符号)。`$SHVLA_DATA/diag/cap.py` 抓取 `run_sim` 构造的 gapy 命令和环境(写到 `env.sh`),`memgdb.sh` 是按内存触发的 gdb 脚本,`rsspoll.sh` 每 0.5 s 记录 RSS。
  - **Retracted.** An earlier pass interrupted gdb on wall time (simulated time at the interrupts: 22-33 us, before the failing phase) and ran perf from launch; the "engine steps 1 ns at a time", "`Time_engine_stop_event::step_nofree` allocates 104 B per step" and the perf share are normal-operation observations, not hang findings, and are not evidence for anything.
  - **Memory profile (`rsspoll.sh`, no gdb, cap 4G, RSS of `gvsoc_launcher`).** `128x128x256:128,128,256`: 86 MB at 2.0 s after launch, then +480 MB/s linearly (1.1 GB at 4 s, 2.1 GB at 6 s, 4.0 GB at 10 s) until the cap. `128x128x240:128,128,240`: 242-294 MB and the process is done after 1.5 s. Cross test: the 256 binary with the 240 preload image grows the same way (HBM image is not the trigger; the program, i.e. its `shape.h` constants, is). `.text`/`.rodata` of the two ELFs have identical sizes (0x5278 / 0xec4), only immediates differ, so the boot code is the same.
  - **In-growth backtraces (`memgdb.sh`: gdb SIGINT when the inferior RSS passes 1.5 GB, three times).** Simulated time at the three stops: 16.70 us and 21.55 us (plus one stop inside SystemC `sc_event::trigger`). Main-thread stacks: `Exec::exec_instr` -> `Prefetcher::fetch` -> `InsnCache::get_insn` (a core fetching at 0x80003xxx, in `.text`); `sc_thread_process::trigger_dynamic` (SystemC kernel); `Time_engine_stop_event::step_nofree` (operator new). No RedMulE, iDMA, NoC or memory model on any stack, so the sampling does not name the culprit; it says the load is spread over the whole engine (every component is executed far more often per simulated ns than in a passing run, or the engine runs events far ahead). Caveat: those simulated times (16-22 us) are earlier than the Y-zero DMA at 31.5 us, so the growth starts during boot/startup, or the engine reports time differently than I assume; not resolved.
  - **Component traces (stdout must be line-buffered: `stdbuf -oL`; syntax `gapy ... run --trace=<path> --trace-level=trace`).** `/chip/cluster_0/redmule`: `Set MNK addr: 128, 256, 128` at 31199 ns (M = tm, N = tk, K = tn, as in `flex_redmule_config(tm, tk, tn)`), then nothing: the trigger has not happened when the cap kills the run. `/chip/cluster_0/idma/be`: Y-zero 1D DMA done, X rows (`src 0xc0001000 + n*0x200, size 0x200`) queued one burst per row, each "Forward Transfer", 371 queued / 370 finished at 33644 ns, no "Src Conflict" line. The passing 240 run has 1 + 128 + 240 = 369 bursts all finished at 33639 ns, so the 256 run is on the same pace in simulated time up to there (it has 16 more W rows). **This is evidence against the iDMA regulation-queue hypothesis**, not for it (a stall would show the queue stop advancing).
- **Register decode (task 1 follow-up).** `light_redmule.cpp:772-830`: offset 0 reads three `uint16_t` from the 64-bit config write (`m_size = a[0]`, `n_size = a[1]`, `k_size = a[2]`; the SDK writes `(k<<16)|m` and `n`, so tk travels alone as the 16-bit N field); offset 4 takes `uint32_t` x, w, y and the format; zero M/N/K is a `fatal`. 256 and even 1024 fit in 16 bits; no mask, no per-dimension limit. So a register-width bug is excluded.
- **Sweep restated in RedMulE terms** (`flex_redmule_config(tm, tk, tn)` so M = tm, N = tk, K = tn): every failure has M >= 128 (= `ce_height`) and N >= 256, whatever K is (`128x32x256` fails, `128x128x256` fails, M = 96 and N = 256 passes, M = 256 and N = 192 passes). The same preload image passes with `128x128x256:128,128,128` and hangs with `:128,128,256`, so preload and HBM layout are ruled out; the trigger is the program's RedMulE job shape or the DMA geometry that goes with it (X block 64 KiB = 128 rows x 512 B; W block 256 rows). The RedMulE job path is `light_redmule.cpp` -> `HWPEInterleaver` -> TCDM banks, and the TCDM bank model `soft_hier/flex_cluster/memory.cpp` (762 lines, release only, imported by `cluster_unit.py` instead of deeploy's `memory.memory`) sits on it; I did not open `memory.cpp` or `hwpe_interleaver`. That is the next place to look, together with a RedMulE trace that actually reaches the trigger.
- **Root cause: not established.** Open candidates, in order of my current weight: (1) the release-only TCDM model `memory.cpp` / HWPE interleaver under a RedMulE job with M = 128, N >= 256 (the failing shapes share this, the iDMA and register paths look clean); (2) the release-only FlooNoc / `idma_be` regulation changes (iDMA pace matches the passing run up to 33.6 us, so lower weight); (3) a startup-phase effect I cannot explain (growth visible before the first DMA).
- **Fix options.**
  1. Finish the diagnosis cheaply (about 2 h, no new branch): run `--trace=/chip/cluster_0/redmule --trace-level=trace` on `128x128x256` with a cap of 8 G and `timeout -s INT` at the moment RSS passes 2 GB (the earlier runs are killed before the trigger), compare with `128x128x240` (`Set XWY` at 33.7 us, `Finished` 1536 cycles); then break in `LightRedmule::fsm_handler` / `memory.cpp` request path under gdb with the `memgdb.sh` method. The owner must approve the 8 G cap (the rule here is 4 G).
  2. Replace the release `memory.cpp` (and, if needed, the `hwpe_interleaver`) path with the deeploy version of `memory.memory` on the local branch and rebuild hardware (about 100 s), then rerun the sweep: about 0.5 day; changes TCDM timing relative to the release model.
  3. Move G3 to `origin/bowwang-dev/softhier-deeploy` (the owner's published GEMM_PASS simulator): about 1 day; arch file fields change (`hbm_node_addr_space`, no power fields), the stamp and `build_arch.sh` change, and the experiment leaves the release tree, so the owner decides. No-simulator workaround: tile M <= 96 or tile K <= 240 in softhier-mlir's tile policy (rungs would not use the published tile sizes; fallback only).
- Artifacts: `$SHVLA_DATA/diag/` (`cap.py`, `memgdb.sh`, `rsspoll.sh`, `env256.sh`, `env240.sh`, `rss*.txt`, `rm256.txt`, `tr3.txt`, `g6.out` = the three in-growth backtraces); run dirs `$SHVLA_DATA/runs/published/diag-k*`, `diag-v*`.

### G3.1-diag3 — deeploy A/B: root cause FOUND (release `idma_be.cpp` `update()` no longer re-arms the FSM); one-line fix proposed, NOT applied

- 摘要:在 scratch 目录建了 `origin/bowwang-dev/softhier-deeploy` 的完整仿真器(62 s 编译)。A/B:deeploy 仿真器上 `128x128x256` 和 `256x256x256` 都 PASS;把 release 仿真器里**只有 iDMA 后端**的 `.so` 换成 deeploy 的就修好了,反过来把 release 的 iDMA `.so` 放进 deeploy 仿真器就复现挂死。根因:release 的 `IDmaBe::update()` 把 `fsm_event.enqueue()` 改成了 `transfer_regulation_event.enqueue()`,FSM 在 `can_accept_burst()` 为假的那个周期之后没人再唤醒,当前 transfer 永远不结束,DMA wait 永远不返回。在 deeploy 树里(用 release 的 idma 文件)加回一行 `fsm_event.enqueue()` 即 PASS,去掉即复现。没有改 tracked submodule。
- Scope: diagnosis only. Nothing in `third_party/gvsoc` (tracked submodule, its build, its branch) was touched. One simulator at a time, each under `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0` and `timeout 120`; traces piped through `head -c 50000000`; no perf/gdb/strace. Host never pressured (15 GB available before every run). Scratch dir `$HOME/.cache/softhier-vla/gvsoc-deeploy` (outside git), scripts `gvsoc-deeploy-{env,ab,cap,runenv,runtrace,rebuild}.sh` next to it, logs `gvsoc-deeploy-ab-*.log`, `gvsoc-deeploy-build{,2,3}.log`.
- **Deeploy checkout.** GVSoC `origin/bowwang-dev/softhier-deeploy` = 99b60c3 (`[Deeploy] add SoftHier integration`), cloned from the tracked submodule's object store (no network). Nested `core` bda268a and `pulp` 3ca2016 are plain clones of the tracked checkout's `core`/`pulp` checked out at the deeploy gitlinks (not registered submodules; the Makefile's `git submodule update` left them alone, no re-clone, the patches survived). Preparation reuse: `third_party/{systemc_install,DRAMSys,toolchain,systemc}` are symlinks to the tracked checkout's (SystemC 2.3.3, GCC 14.2 toolchain; `soft_hier/libDRAMSys_Simulator.so` md5 is identical on both branches and equals the one in the tracked `third_party/DRAMSys`, so the G1.5 DRAMSys problem does not arise). The `drmasys_apply_patch` step by hand: `git apply soft_hier/gvsoc_core.patch` (core) and `gvsoc_pulp.patch` (pulp), `cp soft_hier/flex_cluster -> pulp/pulp/chips/flex_cluster`, `cp soft_hier/dramsys_configs -> core/models/memory/`. No 31 min preparation. Build: `cfg=<experiment arch/softhier_mlir_arch.py> make hw CMAKE_FLAGS=-j8` with `LD_LIBRARY_PATH` as in `setup.sh`: exit 0, real 62 s (`gvsoc-deeploy-build.log`). The arch file was used unchanged (deeploy's template additionally has `cluster_heap_base/end`, bank 128x32, `hbm_node_addr_space` 0x200000; the SDK defaults the heap macros to 0 and nothing needed them; bank geometry is 64x64 on both simulators, as in every earlier run).
- **Two adaptations on the deeploy SDK (SDK issues, not simulator; record them if the deeploy branch is ever used).** (1) `flex_alloc_api.h` sits in `runtime/deeploy_include/`, not on softhier-mlir's include path: copied to `runtime/include/` (as 4558184 did for the release branch). (2) The natural A/B (deeploy simulator + deeploy SDK) FAILs at once: the program prints nothing, the simulator exits rc 0 at 13628 ns without a ROI. Likely cause (not isolated; `flex_memory.ld`, `flex_runtime.h`, `flex_alloc.h` and `flex_dma_pattern.h` also differ): deeploy's `flex_start.s` has a hard-coded stack shift `sll t0, a0, 0xa` (1 KiB per core) where release uses `ARCH_CLUSTER_STACK_OFFSET`, and softhier-mlir's `sh_gemm` overruns it. The same deeploy-built ELF on the release simulator grows to the 4 GB cap. So a pure branch switch does not work without also fixing the SDK. For the A/B the deeploy SDK was moved to `soft_hier/flex_cluster_sdk.deeploy` and the tracked checkout's SDK (with its generated `flex_cluster_arch.h/.inc`; deeploy's `config.py` emits `ARCH_SPATZ_ATTACED_CORE_LIST []` and drops `ARCH_SPATZ_ATTACED_CORES`) was copied in, so both simulators run byte-identical software. Cross-check: the release-built `softhier.elf` + `preload.elf` for 128x128x240 and 128x128x256 run and PASS on the deeploy simulator (ROI 7213 / 7256 ns); the same ELF hangs on the release simulator for 256.
- **A/B table** (softhier-mlir `tests/gvsoc/run.py gemm --shapes ...`, patches applied by `run.sh apply`, stamp check skipped; peak = cgroup `memory.peak`, includes page cache; wall = simulator wall printed by run.py or, for captured commands, process wall):

| simulator | shape | result | ROI ns | peak MB | wall s |
| --- | --- | --- | --- | --- | --- |
| release (tracked, d42c382) | `128x128x240:128,128,240` | PASS | 6114 | 367 | 3 |
| release | `128x128x256:128,128,256` | FAIL (cap, rc 137) | - | 4096 | ~9 |
| deeploy 99b60c3 + release SDK | `128x128x240:128,128,240` | PASS | 7213 | 431 | 2.7 |
| deeploy + release SDK | `128x128x256:128,128,256` | PASS | 7256 | 445 | 2.8 |
| deeploy + release SDK | `256x256x256` | PASS | 14747 | 453 | 2.9 |
| deeploy + release SDK | default shapes 1-6 | 6 PASS: 14747 / 35471 / 139083 / 14688 / 21143 / 176263; the 7th (`1024x3072x768:...`) not run on deeploy (120 s limit of the whole python process) | | 903 | |
| deeploy + deeploy SDK | `128x128x240` | FAIL (no output, stack) | - | 225 | 1.3 |
| release + deeploy `idma` `.so` only | `128x128x256` | PASS | 6174 | 302 | 2.8 |
| release + deeploy `idma` `.so` only | `256x256x256` | PASS | 13136 | - | 2.9 |
| release + deeploy `idma` `.so` only | default shapes 1-6 | 6 PASS: 13136 / 32050 / 131961 / 13108 / 19595 / 174086; 7th (`1024x3072x768:256,256,256,1,0,all`) cut by the 120 s limit of the whole process | | 559 | |
| release + `idma` `.so` built from release source plus the one line (see below) | `128x128x256` | PASS | 6174 | 303 | 3 |
| same `.so` | `1024x3072x768:256,256,256,1,0,all` (7th default shape, alone) | PASS | 759705 | 607 | 51.6 |
| release + deeploy `floonoc` `.so` only | `128x128x256` | FAIL (cap) | - | 4096 | ~9 |
| release + deeploy `hbm_ctrl` `.so` only | `128x128x256` | FAIL (cap) | - | 4096 | ~9 |
| deeploy + release `idma` `.so` only | `128x128x256` | FAIL (cap) | - | 4096 | ~8 |
| deeploy tree rebuilt with release `idma_be.{cpp,hpp}` unchanged (control) | `128x128x256` | FAIL (cap) | - | 4096 | 8.5 |
| same, plus one line `this->fsm_event.enqueue();` in `IDmaBe::update()` | `128x128x256` | PASS | 7260 | - | 2.7 |
| same, plus the line | `256x256x256` | PASS | 14751 | - | 2.9 |

  The `.so` swaps use `SOFTHIER_MODEL_DIR` (searched first by `run_sim` and by the captured gapy command; the model `.so` names of `idma`, `floonoc`, `hbm_ctrl`, `light_redmule`, ... are identical on both trees; only the TCDM model differs: release `gen_pulp_chips_flex_cluster_memory_cpp_*`, deeploy `gen_memory_memory_cpp_*`). `256x768x192` gives 32050 ns on the release simulator with the release and with deeploy's `idma` `.so`. Deeploy's whole simulator is slower in simulated time than release+deeploy-idma (7256 vs 6174 ns for `128x128x256`); the effect of the iDMA backend alone on ROI was not measured for the other shapes.
- **Trace comparison** (`--trace=<path> --trace-level=trace`, 50 MB cap, `128x128x256`).
  - Release, `/chip/cluster_0/idma/be`: 371 "Forward Transfer", 370 "Finished burst"; the regulation handler is last invoked at 33644 ns with no "Src Conflict" line and no further event. The last forwarded burst is the W row `src 0xc100f100, dst 0x2f100, size 0x100` (1 Y-zero + 128 X + 242 of 256 W rows = 371). The expected total is 385; the remaining 14 W rows are never queued. After that no DMA event occurs but the simulation keeps running (memory growth).
  - Deeploy (same ELF): 513 bursts finished (loads plus write-back), no regulation lines. `/chip/cluster_0/redmule` shows `Set MNK addr: 128, 256, 128`, `Set XWY addr: 0, 131072, 262144`, `redmule configuration (M-N-K)`, then `Preload --- Send TCDM req #n / Receive TCDM resp` for the 128 Y-tile rows and the compute. On release the trigger (`Set XWY`) is never reached: the program is still in `bare_dma_wait_all()` waiting for the stuck load. That explains the earlier "RedMulE configured, nothing followed" observation; RedMulE, TCDM and `memory.cpp` are not involved.
- **Mechanism.** Release `pulp/pulp/idma/be/idma_be.cpp` (in `soft_hier/gvsoc_pulp.patch`): `IDmaBe::update()` ("check if any action should be taken in the next cycle from the FSM handler") enqueues `transfer_regulation_event` instead of `fsm_event`. The regulation handler enqueues `fsm_event` only when it forwards a new transfer from `regulation_queue`; `fsm_handler` re-arms itself only after it has sent a burst. If in some cycle `src_be->can_accept_burst()` or `dst_be->can_accept_burst()` is false (outstanding-burst limits) and the regulation queue is empty (the middle-end pushes the next transfer only when the current one is fully issued), nothing wakes the FSM again: the current 0x100-byte row is not issued, the middle-end waits, the DMA never completes and the cores spin in the wait loop. It is not a simple transfer-count threshold (`256x256x192` passes with at least 449 row transfers per cluster, `256x768x192` too); why tile M 128 with tile K 256 (X rows of 0x200 B) hits the window of empty regulation queue plus back-pressure was not traced. Inferred, not traced: which `can_accept_burst` goes false, and why simulator memory grows while the cores spin (the earlier backtraces were all engine code).
- **Conclusion.** Responsible component: the release iDMA backend regulation change (`idma_be.cpp`/`.hpp`, via `soft_hier/gvsoc_pulp.patch`). Evidence: a single-`.so` swap fixes release and breaks deeploy, `floonoc` and `hbm_ctrl` swaps do nothing, and a one-line source change in a rebuilt scratch tree flips FAIL to PASS with a clean control. The power-capture gate (d42c382) and the TCDM memory-model theory are unrelated.
- **Fix options for the tracked simulator (not applied; they change the simulator, so the owner decides; then `build_arch.sh` + stamp rewrite).**
  1. Preferred, one line: in `third_party/gvsoc/pulp/pulp/idma/be/idma_be.cpp` add `this->fsm_event.enqueue();` after `this->transfer_regulation_event.enqueue();` in `IDmaBe::update()`, and carry it in `soft_hier/gvsoc_pulp.patch` on `shvla/soft_hier_release` (the `core`/`pulp` hunks are working-tree changes of the nested submodules, so the pulp working tree and the patch file both need the edit). Cost about 15 min: incremental `build_arch.sh` (the idma module rebuild took 5 s in the scratch tree) plus the rung-1 sweep (about 2 min). Tested on the release simulator via a model-dir swap (a `.so` built in the deeploy scratch tree from release's `idma_be.{cpp,hpp}` plus the one line: `128x128x256` PASS 6174 ns, and all 7 default rung-1 shapes pass with the swapped idma `.so`, the 7th being `1024x3072x768:256,256,256,1,0,all`, 759705 ns, 51.6 s, 607 MB), not via a rebuild of the tracked tree. Worth reporting upstream (gvsoc-pulp, the regulation-queue commit).
  2. Revert `idma_be.{cpp,hpp}` to deeploy's single-transfer backend (drop the regulation queue): same cost, loses the release regulation feature (ordering by source backend; its effect is untested here), ROI may shift.
  3. No rebuild: `SOFTHIER_MODEL_DIR=<dir holding deeploy's gen_pulp_idma_snitch_dma_cpp_128919885.so>` for runs. Proven for the rung-1 shapes, but mixes `.so`s from two engine versions and bypasses the stamp; fallback only.
  4. Switching the experiment to the deeploy branch is not needed and also needs the SDK fix (`flex_start.s` stack shift), `flex_alloc_api.h` and arch-template changes: about 1 day.
- Safety: no incident; no swap use, 4 GB cap per run (a cap kill takes about 8 s, exit 137), one simulator at a time. Note for later agents: `cap.py` / `run.py gemm` rewrite `tests/gvsoc/gemm/preload.elf` for the shape of the last run (12 MB after the 7-shape run, which made a trace run hit the 120 s limit); regenerate it before a captured-command run.
- Files: this LOG entry only. Scratch (not in git): `$HOME/.cache/softhier-vla/gvsoc-deeploy` and `gvsoc-deeploy-orig-idma/` (deeploy's original `idma_be.*` and `idma_be.cpp.release_plus_fsmkick`; the scratch tree's `idma_be.cpp` now holds the release file plus the one line).

### G3.1-fix — iDMA backend one-line fix applied (GVSoC 0e4d719, parent 46dffe6); PAUSED before rebuild for WSL restart

- 做了什么:按 owner 决定,在 `IDmaBe::update()` 里保留 `transfer_regulation_event.enqueue();` 并加 `this->fsm_event.enqueue();`(带注释)。两处同步:(1) `third_party/gvsoc/soft_hier/gvsoc_pulp.patch` 里 `pulp/idma/be/idma_be.cpp` 的 hunk(`@@ -142,7 +191,10 @@`,后面三个 hunk 的新起始行各 +2);(2) 已打补丁的工作树 `third_party/gvsoc/pulp/pulp/idma/be/idma_be.cpp`。验证:把该文件的 hunk 单独抽出,用 `patch -p1` 打在嵌套 pulp HEAD 的原始 `idma_be.cpp` 上,无 offset/fuzz,结果与改后的工作树文件 `cmp` 相同。GVSoC 分支 `shvla/soft_hier_release` 提交 0e4d719(症状、根因、证据=G3.1-diag3、"candidate for upstream report"),父仓库指针 46dffe6。
- 状态:**尚未重编、尚未跑任何仿真**(owner 要重启 WSL,按指示暂停)。此时 `stamp.py check` 会 MISMATCH(pulp diff 已变),这是预期的。
- 下一步(新 agent 接手):(1) `free -g` 确认 >=14 GB;(2) `infra/softhier/build_arch.sh experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py`(增量,`-j8`),`stamp.py check` 通过,确认没有 `SOFTHIER_MODEL_DIR` 指向 deeploy `.so`;(3) `run.sh published sim fix-check -- tests/gvsoc/run.py gemm --shapes 128x128x256:128,128,256`(预期 PASS,ROI 约 6174 ns),再 `256x256x256`(约 13136 ns);(4) rung 1-5 按设计表顺序。
- What to inspect: 重编后 ROI 是否与 diag3 的 `.so` 交换结果(6174 / 13136 ns)一致;peak memory 应几百 MB。

### G3.1-rebuild — rebuild after the iDMA fix, fix-check, rung 1 (`gemm`): PASS

- 做了什么:`unset SOFTHIER_MODEL_DIR`,`build_arch.sh experiments/smolvla-e2e-baseline/arch/softhier_mlir_arch.py`(增量)用了 **7.4 s**(`make hw` 7 s),`stamp.py check` 通过(stamp OK)。`SHVLA_SIM_MEMMAX=8G` 导出,每次跑之前 `free -g` 可用 >= 11 GB。
- **fix-check(验证修复)**:`128x128x256:128,128,256` PASS,ROI **6174 ns**(与 diag3 的 `.so` 交换结果一致),wall 3.9 s,peak **578 MB**;`256x256x256` PASS,ROI **13136 ns**(一致),wall 5.9 s,peak **350 MB**。重编后的真实 tree 与 `.so` 交换结果逐 ns 相同。
- **Rung 1** (`run.sh published sim rung1-gemm -- tests/gvsoc/run.py gemm`,所有默认 shape):exit 0,**7/7 PASS**,wall 合计约 96 s(各 shape wall 之和,不是 run.sh 总 wall),peak **616 MB**,oom 0。

| shape | ROI ns | wall s | published |
| --- | --- | --- | --- |
| 256x256x256 | 13136 | 3.2 | docs/DSE.md: 14.7-16.7 us(1 cluster) |
| 256x768x192:256,256,192 | 32048 | 3.2 | - |
| 512x768x768:256,256,256 | 131955 | 8.4 | 138.9 us(gemm test) / 150.8 us(ubench) |
| 256x256x256:256,256,256,0 | 13108 | 3.4 | - |
| 256x256x512:256,256,256,1,1 | 19595 | 4.9 | - |
| 1024x768x768:256,256,256,1,0,all | 177462 | 14.5 | 120.1 us |
| 1024x3072x768:256,256,256,1,0,all | **759705** | 58.8 | **520.5-520.8 us** (docs/DSE.md, SIMULATOR_NOTES.md) |

- **Known deviation(记录大小,没有修)**:16 个 cluster 的大 shape 比 published 慢约 **+46 %**(1024x3072x768: 759705 vs 520521 ns;1024x768x768: 177462 vs 120134 ns,+48 %),单 cluster 的 shape 持平或略快(512x768x768 -5 % 至 -12 %,256^3 在 published 范围内或略快)。形态指向多 cluster 的 DMA/互连时序,不是算术。这个偏差在修复之前就存在:diag3 里 `.so` 交换的 759705 ns 完全一致,所以不是重编引入的;deeploy 全套仿真器的 shape 1-6 也比 published 慢,说明与 release 的 GVSoC 补丁集(side-note table 里私有 `.patch` 缺失)有关,不是这次的 iDMA 修复。该偏差会让后面各 rung 的 sim time 同样偏长,应该算 known deviation 而不是 failure(rung 1 的 pass 列只要求所有 shape PASS)。
- What to inspect: (1) 重编后 ROI 6174/13136 与 diag3 完全一致;(2) rung 1 的 7 个 PASS 和 peak 616 MB(不再有内存膨胀);(3) 最后两行 16 cluster 的 +46 % / +48 % 偏差,是否要在 G3 gate 的 timing 判定里按 known deviation 处理。
- 复现:`run.sh published sim rung1-gemm -- tests/gvsoc/run.py gemm`(run dir `$SHVLA_DATA/runs/published/rung1-gemm`)。

### G3.1-rung2 — rung 2 (`mlir examples/*.mlir`): FAIL, STOPPED (R2: memory-cap kills, not a known side-note item)

- 命令:`run.sh published sim rung2-mlir -- tests/gvsoc/run.py mlir examples/{add_test,gemm1024_summa,gemm256,gemm512,gemm512_linalg,gemm_nonconst,mlp,mlp_linalg,siglip_encoder_layer}.mlir`(`examples/*.mlir` 展开的 9 个文件)。exit **4**(内存上限 kill,oom_kill events 2),peak 8192 MB。
- 结果:

| example | result | ROI ns | wall s | self-check |
| --- | --- | --- | --- | --- |
| add_test | PASS | 910393 | 4.3 | ADD_TEST_CHECK ok=65536/65536 |
| gemm1024_summa | **FAIL** (killed at the 8G cap) | - | 21.0 | has GEMM_CHECK in the generated main |
| gemm256 | PASS | 100462 | 6.4 | none (runs to the end) |
| gemm512 | PASS | 86408 | 3.8 | GEMM512_CHECK 262144/262144 |
| gemm512_linalg | PASS | 86408 | 3.9 | 262144/262144 |
| gemm_nonconst | PASS | 86206 | 3.8 | 262144/262144 |
| mlp | PASS | 1064054 | 5.8 | MLP_CHECK 65536/65536 |
| mlp_linalg | PASS | 1064054 | 5.6 | 65536/65536 |
| siglip_encoder_layer | **FAIL** (killed at the 8G cap) | - | 501.8 | - |

  所以 7 个 PASS(6 个有 self-check PASS:add_test, gemm512, gemm512_linalg, gemm_nonconst, mlp, mlp_linalg;1 个没有 self-check 的 gemm256 build-and-run-to-end),2 个 FAIL。
- **gemm1024_summa**:单独重跑(`rung2-summa`,同一个 8G 上限)再次 18 s 内涨到 8 GB 被 kill,没有 ROI,没有任何错误信息,也不是 iDMA 修复前已知的那个 FSM 挂死的同一个形态?症状(模拟器空转并吃内存)与 K-tile-256 一样,但这次 rung 1 / mlp / gemm512 里用到的 iDMA 路径都正常,SUMMA 额外用 cluster 间 multicast DMA(FlooNoc)。**根因未验证**。假设:SUMMA 走的某条 cluster-to-cluster / multicast 路径上还有一个和 `IDmaBe::update()` 同类的漏唤醒(`transfer_regulation` 队列改动的另一处,或 floonoc 的 multicast),待用 `--trace=/chip/cluster_0/idma/be` 和 floonoc 路径的 trace 确认。我想抓 trace 时发现 `cap.py` 只适配 `gemm`,`mlir` 路径要通过 run.sh 的环境(`/opt/x86-ort/run`),没有继续。
- **siglip_encoder_layer**:502 s 后到 8G 上限。published 是 17,851,422 ns 模拟、wall 447 s(docs/SIMULATOR_NOTES.md:320)。注意:rung 1 的内存是 350 MB(6 us)到 616 MB(760 us),随模拟时间大致线性涨(约 0.3 MB/us);外推 17.85 ms 约 6-7 GB 加上程序本身,所以这个 kill 可能是**真实需求略超 8G 的缓慢增长(泄漏),不一定是挂死**。未区分。这个线性增长同样意味着 rung 3(5.36 ms published)约 2 GB、完整链 119 ms 会远超 8G,需要 owner 决定。
- 未执行:rung 3-5(按 "stop at the first unresolved failure")。
- 提议(待 owner):(1) 对 `gemm1024_summa` 抓 iDMA be + floonoc trace(<=50 MB)找最后一个 stall;若同类漏唤醒则在 `gvsoc_pulp.patch` + 补丁后的工作树里补一行;(2) 查 GVSoC 里随模拟时间线性增长的内存(trace/stat buffer 或 event 记录);(3) 或 owner 批准把 `SHVLA_SIM_MEMMAX` 对 siglip_encoder_layer 提高到 13G 做一次区分(违反本次 8G 硬限,所以没做);(4) rung 3-5 可以先跑(它们预计在 8G 内:rung 3 约 2 GB),但按指示没有继续。
- What to inspect: `rung2-mlir` / `rung2-summa` 的 record block(exit 4,oom_kill);siglip 的 peak 8192 MB 与 502 s;上表里 7 PASS / 2 FAIL;summa 在 18 s 内涨 8 GB 的速度(~450 MB/s,远快于线性增长,所以是挂死/空转而不是泄漏)。

### G3.1-ihbm — ideal-HBM switch (SOFTHIER_IDEAL_HBM), rung 1 re-run, rung 2: 6/9 PASS, STOPPED at a DMA hang in `hbm_fill` (R2)

- **Switch (owner decision).** softhier-mlir 的 `run_sim` 设了 `SOFTHIER_IDEAL_HBM=1`,但 GVSoC 里没有任何代码读它,`flex_cluster.py` 一直实例化 `memory.dramsys.Dramsys`(SIMULATOR_NOTES #5 说的是私有 patch)。在本地分支 `shvla/soft_hier_release` 加开关:GVSoC `cdb56a2`,父仓库 infra `1d195b0`(设计文档 side-note 第 5 行 + env 说明: infra `1afd775`)。`soft_hier/flex_cluster/flex_cluster.py` 是 GVSoC 里直接 tracked 的文件(`make config` 把它拷到 `pulp/pulp/chips/flex_cluster` 和 `install/generators`),所以直接改+提交,不需要 patch 文件。`SOFTHIER_IDEAL_HBM=1` 时,每个 HBM channel(placement 不变,(4,0,0,4) 对应的 8 个)用 `flex_cluster.memory.Memory(size=整个 HBM 地址窗口, width_log2=0, latency=0, atomics=False)` 代替 Dramsys:同一个 `input` 口、同样的 `hbm_ctrl` 绑定、同样的地址映射;`width_log2=0` 即 `width_bits==0`,`soft_hier/flex_cluster/memory.cpp` 里不加带宽限制(duration 不设),latency 0 = 零额外延迟、无限带宽。size 取整个窗口是 hbm_ctrl 产生的 channel 内 offset 的安全上界(calloc,没碰的页不占内存)。
- **Build-time vs run-time.** run-time:gapy 生成 config 时读环境变量(`os.environ`),`make hw` 不读。但 `install/generators` 里是拷贝,所以改完要 `build_arch.sh` 一次(4 s,`stamp OK`;stamp 里 GVSoC commit 已覆盖源码变化)。`build_arch.sh` / `stamp.py` 不需要改,没有给 stamp 加字段:构建时没有设该变量,`run_sim` 运行时设了,ROI 从 759705 ns 变成 520114 ns 就是证据。`infra/softhier/README.md` 新增 "Ideal HBM" 一节(注意从 `run_kernel_tests.sh` 起的 kernel 测试不经过 `run_sim`,仍是 DRAMSys,除非自己 export)。
- **Verify.** `ihbm-check`(默认 spec `1024x3072x768`)ROI 1049223 ns,**不可比**(默认 tile/pipeline 不是 published 的那个)。可比的 spec `1024x3072x768:256,256,256,1,0,all`(`ihbm-check2`):**PASS,ROI 520114 ns**(published 520.5 us,-0.07 %),wall 10.4 s,peak 387 MB(DRAMSys: 58.8 s, 607 MB)。
- **Rung 1 re-run** (`rung1-gemm-ihbm`):7/7 PASS,peak 395 MB(DRAMSys 616 MB),oom 0。

| shape | ROI ns (ideal HBM) | wall s | DRAMSys ROI | published |
| --- | --- | --- | --- | --- |
| 256x256x256 | 13102 | 2.0 | 13136 | 14.7-16.7 us |
| 256x768x192:256,256,192 | 31854 | 2.0 | 32048 | - |
| 512x768x768:256,256,256 | 131713 | 3.5 | 131955 | 138.9 / 150.8 us |
| 256x256x256:256,256,256,0 | 13074 | 2.0 | 13108 | - |
| 256x256x512:256,256,256,1,1 | 19536 | 2.3 | 19595 | - |
| 1024x768x768:256,256,256,1,0,all | 119415 | 4.5 | 177462 | 120.1 us |
| 1024x3072x768:256,256,256,1,0,all | 520114 | 10.3 | 759705 | 520.5-520.8 us |

  所以 16 cluster 的 +46-48 % 偏差完全是 DRAMSys 造成的,理想 HBM 下与 published 一致(rung 1 的 known deviation 作废)。
- **Rung 2** (`rung2-mlir-ihbm`, 9 个 example; 之后为了避开 summa 单独跑另外 7 个 `rung2-siglip-ihbm`):`add_test` 910393 ns、`gemm256` 76984、`gemm512` 85800、`gemm512_linalg` 85800、`gemm_nonconst` 85804、`mlp` 1064054、`mlp_linalg` 1064054 都 PASS,self-check 全 ok(`add_test` 65536/65536, gemm512 系 262144/262144, mlp 系 65536/65536);`gemm256` 没有 self-check。`gemm1024_summa` 和 `siglip_encoder_layer` **不终止**。第一次 `rung2-mlir-ihbm` 在 summa 上跑了约 5 min 后我手动 kill(没有 record block);`summa-ihbm`(SHVLA_SIM_TIMEOUT=120)exit 3,peak 264 MB;`rung2-siglip-ihbm` siglip 跑了约 13 min(published wall 447 s,17.85 ms 模拟;mlp 约 4 s/模拟 ms,推算应该 <2 min)后手动 kill,gvsoc_launcher RSS 573 -> 736 MB 后完全不变(100 % CPU)。**内存不再随模拟时间增长**(DRAMSys 下 0.3 MB/us 的增长和 summa 450 MB/s 都来自 DRAMSys 路径;理想 HBM 下 summa 是平的 228 MB)。
- **Diagnosis of summa(有界 trace,管道 `tail -c`/`head -c`,未超过 50 MB)。** 流程不是 pass table 的问题(`run.py` 的 `EXAMPLE_PASSES` 对 summa 是 `linalg-to-softhier,distribute-summa,pipeline-gemm`)。`/chip/cluster_0/.*` trace 的最后 4 MB(约 21.66 ms 模拟时间)里只有 cluster_0 `pe2` 在 `sh_test_fill_const_fp16`(`softhier.hbm_fill`,设备端填充输入)里循环 `dmstati t0,0,0x2; bne t0,0,-4`:iDMA 一直报告 busy,其他组件全空闲。核 0 先用 `sh` 往 TCDM `0xf0000..0xf0800` 填 fp16 常数,然后启动 DMA 把它复制到 HBM;TCDM 最后一次被 DMA 读是约 30.7 us(bank 读的最大时间戳),之后什么都没有。所以是 iDMA 传输永不完成(丢唤醒或死锁),不是 HBM、不是 pass、不是 timeout。DRAMSys 下同一个 hang 表现为内存暴涨,理想 HBM 下表现为内存平坦,症状不同、根相同。这和 K-tile-256 hang 同属 iDMA 后端一类,但 iDMA 修复(0e4d719)已在,所以是另一条路径(猜测:`hbm_fill` 的多维/重复 DMA 描述符,或 fix 的 `fsm_event.enqueue()` 与同步零延迟响应的交互)。注:rung 1/mlp 的 DMA 都能完成,`add_test` 也用 `hbm_fill` 但只是一个小 1D 传输。
- **Not fixed (not a known side-note item: #13/#14 是 REDADD/burst buffer,这里没有 collective)。** 提议:(1) 给 `idma_be.cpp`/`idma_fe` 加临时 trace(或用 `--trace=/chip/cluster_0/idma` 配 `-DTRACE`),看 summa 的 `hbm_fill` DMA 描述符(size/reps/stride)和最后一个未完成的 burst;(2) A/B:在 deeploy scratch tree(`~/.cache/softhier-vla/gvsoc-deeploy`)里跑同一个 summa ELF(`/tmp/shbuild_61379/softhier.elf`,已不在持久目录),看 deeploy 的 iDMA 是否完成;(3) 若是 iDMA 同一类丢唤醒,补一行到 `gvsoc_pulp.patch` + 工作树。siglip 是否卡在同一处没有验证(RSS 平坦、100 % CPU,症状一致,没有取 trace)。
- 未执行:rung 3-5(按 "stop at the first unresolved failure")。工作树状态:所有模拟进程已停止。
- 复现: `run.sh published sim summa-ihbm -- tests/gvsoc/run.py mlir examples/gemm1024_summa.mlir`(设 `SHVLA_SIM_TIMEOUT=120`)。
- What to inspect: (1) ihbm-check2 的 520114 ns 与 rung 1 表;(2) summa 的 trace 结论(pe2 在 `dmstati` 轮询,其他组件空闲);(3) 提议 (1)-(3) 由 owner 选。

### G3.1-diag4 — summa / siglip hang: root cause FOUND and FIXED (release `IDmaBe::can_accept_transfer()` races the regulation queue; GVSoC 2910e62, parent 6338b35); rung 2 now 9/9 PASS

- 摘要:`gemm1024_summa`(以及 siglip)的 `hbm_fill` hang 不是 HBM、不是 `flex_cluster.memory.Memory` 配置,而是 release iDMA 后端的第二个 bug。`regulation_queue` 把"forward 一个 transfer(并设置 `current_transfer_size`)"推迟到 `enqueue_transfer()` 之后一个周期;这一周期里 `can_accept_transfer()`(`idma_be.cpp:115-119`,只看 `current_transfer_size == 0`)仍然返回 true,middle-end 把下一行(2 KiB 一行的 `hbm_fill`)又推进来,第一行在发出 burst 之前被覆盖,永远不会被读、不会被 ack,DMA wait 永不返回。修复一行:`return current_transfer_size == 0 && regulation_queue.empty();`。已应用在本地 GVSoC 分支(infra 提交 2910e62,`gvsoc_pulp.patch` + 已打补丁的 pulp 工作树两处同步),父仓库指针 6338b35。summa 97682 ns PASS(5.7 s,412 MB),siglip PASS,rung 1 7/7 与 ihbm 表逐位一致,其余 7 个 rung-2 例子 ROI 与 ihbm 表逐位一致。
- Scope: diagnosis plus a one-function fix. All runs one at a time under `systemd-run` `MemoryMax=8G`, no swap, `timeout`, output through `head -c 50000000`; no perf/gdb/strace; no sudo; builds `-j8`. Debug trace only through the existing `--trace=/chip/cluster_0/idma/{be,axi_write,tcdm_read}` (no code prints were added, so there is nothing to revert). Scratch (not in git): `~/.cache/softhier-vla/diag4/` (`run.sh`, `w/` logs, `so_{C,D,E,F}/` model dirs) and the deeploy scratch tree `~/.cache/softhier-vla/gvsoc-deeploy` (pulp `idma_be.cpp` is back to `release_plus_fsmkick`, `idma_be.hpp` to release; its `install/models` idma `.so` is therefore stale relative to those files, rebuild before reuse).
- **A/B matrix** (same ELF `~/.cache/softhier-vla/diag/summa-ihbm/softhier.elf`, no preload, run with `gapy` directly; "sim time" = last simulated timestamp seen in the trace):

| simulator tree | iDMA backend | HBM | result | sim time / peak |
| --- | --- | --- | --- | --- |
| ours (shvla/soft_hier_release, 0e4d719 fix) | release + fsm kick | ideal | **HANG** (flat 220 MB, 100 % CPU, 60 s) | last iDMA/AXI activity 38.92 us; last TCDM read 30.75 us; first lost row 21.16 us |
| ours | release + fsm kick | DRAMSys | **HANG** (grows to the 8G cap, rc oom, 27 s) | sim time not captured (cap kill) |
| deeploy scratch (99b60c3, same SDK/ELF) | release + fsm kick (what diag3 left in it) | DRAMSys | **HANG** (8G cap, 21 s) | not captured |
| deeploy scratch | deeploy original (no regulation queue) | DRAMSys | **PASS**, GEMM1024_CHECK 1048576/1048576 | wall 18 s, peak 2.7 GB |
| ours | deeploy original `.so` swapped in (`--model-dir`) | ideal | **PASS** 97684 ns | 5 s, 356 MB |
| ours | release, fsm kick removed (`.so` from scratch) | ideal | **HANG** (same as the fixed one; summa does not need the 0e4d719 line to hang) | flat 219 MB |
| ours | release + fsm kick + `can_accept_transfer` fix (`so_F` swap) | ideal | **PASS** 97682 ns | 6 s, 324 MB |
| ours | same `so_F` | DRAMSys | **PASS** 119202 ns | peak 1678 MB |
| ours, rebuilt with the fix (tracked tree) | release + both fixes | ideal | **PASS** 97682 ns (`run.sh ... diag4-summa`) | 5.7 s, 412 MB |

  So the hang is a property of the release iDMA backend, independent of HBM model and of the SDK: only swapping the iDMA `.so` flips it both ways. Under DRAMSys the lost rows make the simulator run away in memory instead of idling (same root, different symptom). The ideal-HBM-vs-DRAMSys ROI difference for summa (97682 vs 119202 ns) is the memory model, as in rung 1.
- **Locating the stuck transfer** (`--trace=/chip/cluster_0/idma/be`, 1.2 MB). The program is `hbm_fill`: core 0 fills TCDM `0xf0000..0xf0800` and the DMA copies that 2 KiB block 1024 times (src `0xf0000`, dst `0xc0000000 + k*0x800`, 2 MiB, 1D transfers of 0x800) to HBM; the other cores poll `dmstati`. Counts: 1024 "Queueing burst", 1024 "Forward Transfer", but only 741 "Finished burst" (741 x 4 acks of 0x200 = 2964) and 741 "Enqueueing write burst" in `axi_write` and 2964 "Reading line from TCDM" in `tcdm_read`. The 283 rows with no `axi_write`/`tcdm_read` activity are *exactly* the rows queued with `regulation_queue depth = 1` (283 of 283, none of the depth-0 rows after the first one), starting at row 442 (21161 ns) and then every second row. Pattern at 21161-21163 ns: row A queued (depth 0) -> next cycle row B queued (depth 1, because the middle-end FSM, after pushing A, immediately re-ran and `IDmaBe::can_accept_transfer()` was still true) -> handler forwards A, re-arms itself, forwards B in the next cycle, overwriting `current_transfer_*` before the backend FSM issued A's burst. A is lost (it still sits in `transfer_queue`/`transfer_ack_queue`, B's acks are matched to A, the queues get misaligned) and finally the ack queue never drains, the last row never completes, `dmstati` stays busy. Why gemm and mlp did not show this: not traced (inferred: their rows are not pushed in consecutive cycles with an empty backend; `hbm_fill` pushes 2 KiB 1D rows back to back once the TCDM read side keeps up, which happened from row 442 on).
- Code (release `pulp/pulp/idma/be/idma_be.cpp`): `enqueue_transfer()` (line 54-62) only pushes to `regulation_queue` and enqueues `transfer_regulation_event`; `transfer_regulation_handler` (65-104) sets `current_transfer*` when it forwards; `can_accept_transfer()` (115-119) checked only `current_transfer_size == 0`; the caller is `IDmaMe2D::fsm_handler` (`me/idma_me_2d.cpp`, `be->can_accept_transfer()` then `be->enqueue_transfer(burst)` then `fsm_event.enqueue()`). Deeploy's backend set `current_transfer_size` synchronously inside `enqueue_transfer()`, so its check was exact.
- **Memory model check (step 2).** `flex_cluster.memory.Memory(width_log2=0, latency=0)` is legal: `memory.cpp:143-144` read `width_bits`/`latency`, `:238` `req->inc_latency(latency)` (0 adds nothing) and `:241-253` apply the bandwidth limit only if `width_bits != 0`; width 0 is the "no limit" case (the python default for `latency` is also 0, `memory.py:152`). The handlers return `IO_REQ_OK` synchronously for read/write, as the TCDM instances of the same class do. It also did not matter here: the same ELF hangs on the DRAMSys build and passes on ideal HBM once the iDMA is fixed. SIMULATOR_NOTES row 5 (`SOFTHIER_IDEAL_HBM=1` swaps HBM for gvsoc's ideal memory, "HBM-bound numbers are optimistic") is consistent with our switch.
- **Fix (applied).** `can_accept_transfer()`: `return this->current_transfer_size == 0 && this->regulation_queue.empty();` with a comment. Two-place rule: (1) `third_party/gvsoc/pulp/pulp/idma/be/idma_be.cpp` (patched nested pulp working tree); (2) the `idma_be.cpp` hunks of `soft_hier/gvsoc_pulp.patch` regenerated (hunk headers lost their function-name context, so the patch diff is larger than the code change, 22+/14-); checked: the extracted hunks applied with `patch -p1` to the original `idma_be.cpp` of the nested pulp HEAD give no offset/fuzz and a file `cmp`-identical to the working-tree file. Kept 0e4d719 (`fsm_event.enqueue()` in `update()`), it is a separate bug. Stamp: `build_arch.sh` (6 s) + `stamp OK`; no `SOFTHIER_MODEL_DIR` in effect for the regression runs. Candidate for the upstream gvsoc-pulp report together with 0e4d719.
- **Regression on the rebuilt tracked tree** (`SHVLA_SIM_MEMMAX=8G`, one sim at a time; ROI identical to the ihbm tables, i.e. no timing shift):
  - `diag4-summa`: PASS 97682 ns, GEMM1024_CHECK 1048576/1048576, 5.7 s, 412 MB.
  - `diag4-rung1` (7 shapes): PASS 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / **520114** ns (published 520.5 us), peak 394 MB.
  - `diag4-rung2` (7 examples): add_test 910393, gemm256 76984, gemm512 85800, gemm512_linalg 85800, gemm_nonconst 85804, mlp 1064054, mlp_linalg 1064054 ns, all PASS with their self-checks, peak 300 MB.
  - `diag4-siglip` (`SHVLA_SIM_TIMEOUT=1200`): **PASS**, roi **10863627 ns** (10.86 ms), wall 349 s, peak 841 MB (no self-check in this example, it passes when the simulation completes). This is not the published 17.85 ms (sim) / 447 s (wall): the published figure presumably includes HBM timing, and ideal HBM is optimistic (SIMULATOR_NOTES row 5). Not compared further; treat 10.86 ms as the ideal-HBM number.
  - Rung 2 is now 9/9 PASS under ideal HBM.
- 后续:rung 3-5(按 "stop at the first unresolved failure" 之前停在 rung 2)可以继续;注意 DRAMSys 下 summa 仍然很慢/吃内存的历史现象已解释(iDMA 丢行),不再需要 `SHVLA_SIM_MEMMAX` 提高。
- What to inspect: A/B 表(`ours + deeploy-iDMA .so` PASS, `ours + release .so` HANG);trace 里 283/283 的 depth=1 行与丢失行完全重合;修复后 summa 97682 ns 与 deeploy iDMA 的 97684 ns 几乎相同;rung 1 的 520114 ns 未变。

### G3.2-diag — siglip gap explained, rung c / 3 / 4 / 5 all FAIL numerically; cause = side-note #10 (and #3) patches missing from our GVSoC; STOPPED before rung 6

- 摘要:先纠正 G3.1-diag4 的一个说法:`siglip_encoder_layer` "PASS 10.86 ms" 只代表程序跑完,这个 example 没有 self-check。同形状、有 self-check 的 `siglip-mlir --seq 256 --cluster all` 在我们的 simulator 上**数值错误**,而且每一段都比 published 快。rung 3、4、5 同样错(L1 / X0 出现 inf)。所以 10.86 vs 17.85 ms 不是"旧 figure"(a),而是环境差异(b):我们的 GVSoC 没有 softhier-mlir 作者的私有 simulator patch,设计文档本来就预期会这样("Without the private simulator patches, failures are expected")。
- **Scope.** Reading plus 4 sims (each alone, `SHVLA_SIM_MEMMAX=8G`, one at a time, no perf/gdb/strace, outputs not traced). 先 `build_arch.sh` 重新写 stamp(之前的 stamp 是 GVSoC cdb56a2,当前 2910e62,run.sh 拒绝运行;`stamp OK` 之后继续)。注意各 run 顶行的 `PASS ...` 只表示仿真完成,真正的 verdict 是 run.sh 的 exit code(都是 1)和逐 tensor 检查;rung 3 的 `OUT ... nan ... bad=0 PASS` 是 checker 假象(NaN 不算 bad)。
- **17.85 ms 的来源(只读).** `softhier-mlir/docs/SIMULATOR_NOTES.md:320`,表 "models_fast verification":`mlir examples/siglip_encoder_layer.mlir` ROI 17851422 ns,wall 447.5 s(before)/ 174.5 s(after),日期 2026-10-08,commit 861a902(`sim: prefer install/models_fast`)。同一表里 `siglip-mlir --seq 256 --cluster all` 是 4782253 ns。那张表的目的是证明 before/after cycles 相同,不是性能 claim。config:example 是 seq 1024、1 层、per-head attention(heads 钉在 cluster 0..11,`cluster=n`)、256^3 tiles、无 pass(已经是 softhier dialect)、16 cluster、**ideal HBM**(README "16 clusters, ideal HBM",DSE.md §8 "ideal HBM, 1 GHz",SIMULATOR_NOTES #5 说 DRAMSys 在作者的 host 上跑不了)。因此 diag4 里"published 数字可能含 HBM 时序"的猜测是错的,作废。example 文件自 a679410(2026-10-08)起没变。作者的 simulator 用的是 `install/models_fast`(→`models_fix`→`models`)。另有一点:DSE.md §8 说 S=256 的旧 ROI 含 ~2 ms 的 printf sample dump,所以 4.78 ms 的 ROI 不是 layer 时间;可比的是 marks。DSE.md §8.3 "per-head, cost-model tiles" S=256:qkv 0.324 / attention 0.831 / rest 1.040 / layer 2.195 ms;S=1024 同配置 16.67 ms(256^3 tiles 17.15 ms,与 17.85 ms 的 ROI 一致,多出的 ~0.7 ms 是我的猜测:LN 前导等,未核对)。
- **rung c** `rungc-siglip256`(`siglip-mlir --seq 256 --cluster all`):**FAIL (exit 1)**。ROI 1520300 ns(published 4782253 ns, -68 %,但 published ROI 含 dump,不可比),marks:qkv 0.233 / attention 0.493 / rest 0.791 / layer 1.517 ms,对 DSE 的 0.324 / 0.831 / 1.040 / 2.195:-28 / -41 / -24 / -31 %。检查:X、LN1、Q、K、KT、P0、V、O 都 PASS(O maxerr 0.019),**H(attention 之后的 o-proj + bias + residual)bad=59/64 maxerr 16.3,G(GELU)maxerr 8959,OUT maxerr 8174 FAIL**。wall 48.8 s,peak 865 MB。
- **Diagnosis(source reading,没改任何东西).** 我们的 GVSoC(`third_party/gvsoc`)只有 `install/models`,没有 `models_fix` / `models_fast`,源码里:`pulp/pulp/snitch/snitch_isa.py` / `core/models/cpu/iss/isa_gen/isa_riscv_gen.py` 的 `fcvt.w.s` 等标量 FP 指令**没有 `nseq` tag**(`.d` 版本有)= side-note #10 未修;`soft_hier/flex_cluster/light_redmule.cpp:1248` 的 `float_to_fp16` 截断尾数(无 RNE)= 截断 helper 存在,但 #3 的症状不出现(见下),不当作原因。#10 正是 softmax / LayerNorm / GELU(exp2 范围缩减通过 `fcvt.w.s`)在 sequence buffer 非空时读到旧整数寄存器的 bug,结果依赖 cycle 时序和代码对齐;这与现象吻合(GEMM 在 rung 1 逐位对,标量 FP row op 错,出现 inf、-19616 之类的值,时间偏短的解释**不是**"错值缩短了循环":rung c 的 qkv 段(LN1 + 3 个 GEMM + bias)快 28 %,而 LN1/Q/K/V 的值都是对的,GEMM 在 rung 1 又是 cycle 精确,所以变快的是值正确的 row op。推断(未验证):缺 `nseq` tag 时整数核不等 FP 子系统,是 cycle 模型本身偏乐观,打 patch 之后即使值碰巧对也会更慢。所以 -39 % 主要是 timing-model 差异。**#10 是最可能的原因,但没有直接验证**(没打 patch、没跑 insn trace)。#1(pending)、#2(zfh)、#6、#8 的状态没逐项核对。未做的 A/B:`SOFTHIER_MODEL_DIR=diag4/so_D`(deeploy 原始 iDMA)重跑 rung c,用来排除我们两处 iDMA 修复对 timing/数值的影响;stamp 记录 `env.SOFTHIER_MODEL_DIR`,需要先用该变量 `build_arch.sh` 再还原。iDMA 嫌疑低:rung 1 / rung 2 的 6 个有 self-check 的例子(gemm256 没有)在两处修复前后逐位一致,而且坏掉的是 row-op 之后的数值。
- **Rungs 3-5**(全部 FAIL,exit 1,accuracy FAIL + timing 偏快,按规则记录、没有 silently fix):

| rung | command | result | published | wall / peak |
| --- | --- | --- | --- | --- |
| 3 | `smolvla --npz vision_s256.npz --layers 1` | EMB PASS (max abs 0.0012; published 0.0067,更小); **L1 FAIL** (inf, median 44480, 256/256 bad); OUT nan; per layer 1.630 ms (attention 0.832 + mlp/proj 0.798); ROI 3010501 ns | README: layer-1 err 0.039; 2.29 ms/layer (1.21 + 1.07) => -29 %; 5.36 ms ROI total (含 dump,不可比) | 55 s / 1180 MB |
| 4 | `smolvla-vlm --npz vlm_c1.npz --layers 1` | EMB PASS (max abs 4.29 on |ref| max 2213); **L1 FAIL** (inf, 155/155 bad); **K1 FAIL** (nan, 55/157 bad); **V1 FAIL** (81/170 bad); per layer 0.725 ms (attention 0.253 + 0.473); ROI 3367968 ns | docs/SMOLVLA.md: 0.909 ms/layer (mean, 16 layers; 113 tokens) => -20 % (layer 1 比平均慢,不严格可比) | 27 s / 1609 MB |
| 5 | `expert.py flow --npz expert.npz --steps 1` | **FAIL**: X0 (1600 elements) 1600/1600 bad, max abs inf; `x_1` vs lerobot inf (floor vs lerobot 0.0016); 6.181 ms per step + 0.117 ms KV projection; ROI 12619491 ns | docs/SMOLVLA_EXPERT.md: 9.96 ms/flow step + 0.30 ms KV projection => -38 %; G2 xt[1] 未与之比(输出已是 inf) | 168 s / 1721 MB |

  每个 FAIL rung 都是:嵌入 / 第一个 GEMM-only 阶段对,第一个带标量 FP row op(LN / RMSNorm / softmax / rope)的 layer 之后出错,且时间比 published 短 20-41 %。没有 hang、crash、OOM(oom_kill 0)。
- **Not done (STOP).** rung 6(`smolvla-e2e`)和 `gate.py` 没有运行 / 没有写:rung 3-5 的前置条件未满足,继续只会得到 inf,且 phase 之间会传播。按设计文档 R2 / side-note 表给 owner 的提议:(1) 只补 #10 到本地 GVSoC 分支(`nseq` tag + 重新生成 ISA 表,`gvsoc_snitch_fp_int_nseq.patch`,作者笔记 `/home/haoze/SmolVLA/SmolVLA/softhier_vla_summary_2026-10.md` §6)(#3 暂不列入:`light_redmule.cpp:1294` 的 `matmul_fp16` 用 fp32 累加、只在最后一次存储时截断,"768 项点积偏低 10 %"的症状不会出现,rung 3 EMB 的 K=768 误差 0.0012 和 rung 1 全部 self-check 也与之矛盾),再重跑 rung c(48 s)作快速判据;(2) 逐项核对 #1 / #2 / #6 / #8 是否也需要;(3) 可选 A/B:`so_D` 排除 iDMA 修复。这些都是 owner 先批的 fork commit(33),这里没有动。
- Simulators: 全部已停止。Commands: `run.sh published sim rungc-siglip256 -- tests/gvsoc/run.py siglip-mlir --seq 256 --cluster all`;`... rung3-smolvla -- tests/gvsoc/run.py smolvla --npz $D/vision_s256.npz --layers 1`;`... rung4-vlm -- tests/gvsoc/run.py smolvla-vlm --npz $D/vlm_c1.npz --layers 1`;`... rung5-expert -- tests/gvsoc/expert.py flow --npz $D/expert.npz --steps 1`,`D=$HOME/.cache/softhier-vla/smolvla_base`。Logs: `$SHVLA_DATA/runs/published/{rungc-siglip256,rung3-smolvla,rung4-vlm,rung5-expert}/run.log`。
- What to inspect: (1) rungc 的 `run.log`(O PASS、H FAIL,marks);(2) `isa_riscv_gen.py:608`(`fcvt.w.s` 无 `nseq`)对比 `:662`(`fcvt.w.d` 有);(3) `light_redmule.cpp:1248` 的截断;(4) SIMULATOR_NOTES.md:320 的 17851422 行和 DSE.md §8.3。

### G3.2-fix10 proposal — apply SIMULATOR_NOTES #10 (`nseq` tag on scalar FP instructions with an integer operand) to the local GVSoC branch

- 摘要:owner 在任务里批准按 "known side-note item with a direct procedure" 处理:先写 proposal,再在本地 GVSoC 分支 `shvla/soft_hier_release` 上只打 #10。作者的 `gvsoc_snitch_fp_int_nseq.patch` **在本机找不到**(`/home/haoze/SoftHier/dse/patches/` 不存在,全盘 find 无此文件名),所以 tag 集合是按 SIMULATOR_NOTES #10 的文字加上 `Rv32d` 的现有写法**重建**的,不是逐字应用作者的 patch。
- **Procedure(来自 SIMULATOR_NOTES #10 + "Integer-register hazard" 一节)**:ISA 表里给"每个带整数操作数的标量 FP 指令"加 `nseq`(这类指令走 bypass lane,不进 FPU sequence buffer,整数核会等 FP 子系统写回),然后重新生成 `isa_snitch_rv32imfdva.cpp` 并重编 `gen_isa_snitch_rv32imfdva_cpp_*`(`snitch.cpp:check_state` 读 `insn->desc->tags[ISA_TAG_NSEQ_ID]` 决定走哪条 lane;sequencer 看的是整数核的 descriptor,FP-subsystem 的 decoder 不决定行为)。作者自己的描述:`fcvt.w.s flt.s feq.s fmv.x.s fclass.s`(及 fp16/fp8 标量版本)缺 `nseq`,`.d` 版本有。
- **Tag 集合(两份 owner 文件范围不同)**:softhier_vla_summary §6 #6 说 "单精度 fp→int 指令";SIMULATOR_NOTES #10 说 "every scalar FP instruction with an integer operand"(两个方向)。取后者,与 `Rv32d` 完全一致(`.d` 的 `fcvt.d.w/wu`、`fmv.d.x` 也带 `nseq`)。每个子集 10 条:`feq flt fle fcvt.w fcvt.wu fcvt.<fmt>.w fcvt.<fmt>.wu fmv.x fclass fmv.<fmt>.x`。子集:`.s`(`isa_riscv_gen.py`,Rv32f),`.h` / `.ah` / `.b`(`isa_smallfloats.py`)。rv64 行(`fcvt.l*`)不动(`.d` 也没动)。只加 `nseq`,`.ah` 行本来没有 `fp_op` 也不补。
- **Two-place rule 状态**:这两个 `.py` 文件在嵌套 core 里是 tracked 且干净的,**不在**现有 `soft_hier/gvsoc_core.patch` 里(`isa_gen/` 下只有 `isa_rvv.py`)。所以 patch 里是**新增**两个 file section(插在 `isa_riscv_gen.py` / `isa_smallfloats.py` 的字母序位置),同时改已打补丁的 core 工作树。检查方式:对 `git archive HEAD` 的嵌套 core 导出用 `patch -p1` 打整个 `gvsoc_core.patch`,无 fuzz / offset / fail,两个文件与工作树 `cmp` 一致(已做,见下一条 entry)。
- **重新生成风险**:`generate_isa` 的 custom command 没有对 `.py` 的 DEPENDS,所以只跑 `build_arch.sh` 可能不重新生成;需要在 build 前后 grep 生成的 `build/core/isa_snitch_rv32imfdva.cpp` 里 `fcvt.w.s` 的 tags,并看 `.so` mtime。
- **#1 (Snitch FP subsystem pending access) 是否可能产生同样症状(只读代码)**:`snitch_fp_ss.cpp` 的 `Iss::handle_event`(:245)开头**没有** `exec.is_stalled()` 守卫,即 #1 的 `gvsoc_snitch_fp_ss_pending.patch` 同样**没有**在我们的树里(:106 的 `is_stalled` 是 `barrier_sync`,不是它)。#1 的签名是 NoC 报 `No entry found for burst (base: 0x2d68)` 和 `flw` 从 HBM/.rodata/远端 TCDM 读到 0;`$SHVLA_DATA/runs/published/*/run.log` 里没有这个字符串(grep 0 个文件)。证据偏反对:rung c 里 LN1、Q、K、P0、O 全 PASS(LN1 的 rsqrt/常数、softmax 的 exp 多项式常数若读成 0 会在第一个 row op 就确定性出错),而失败的是 H、G(依赖 cycle 时序的 row op 之后)。结论:**#1 未打补丁,但目前没有证据指向它**;按规则只打 #10;如果 #10 之后仍错,再提议 #1 并 STOP。
- **预测(便于检验)**:SIMULATOR_NOTES 实测该 fix 只让 `siglip --seq 256` 变慢 +1.1 % / +2.5 %。所以即使精度修好,rung c 的 marks 很可能仍比 DSE §8.3 快 25-40 %(qkv 0.324 / attention 0.831 / rest 1.040 / layer 2.195 ms),这时要找别的原因(#1 / #2 / #6 /RedMulE),不能把 -30 % 直接归因于 #10。按失败规则:精度 OK、时间偏离 = 记录并继续。
- 判据:rung c exit code 0;H、G、OUT 全 PASS;OUT 无 nan(checker 不把 NaN 算 bad)。

### G3.2-fix10 — #10 (`nseq`) applied (GVSoC 070936b, parent efec4c4); rung c STILL FAILS identically (H / G / OUT); regression ROIs unchanged; STOPPED (failure rule: accuracy still failing)

- 摘要:#10 已按上一条 proposal 应用并确认生效,但 rung c 的数值失败**几乎逐位不变**,所以 #10 **不是**(至少不是唯一的)原因。按失败规则 STOP,rung 3-5 没有重跑(同样的 row-op 栈,只会得到同样的 inf),rung 6 / `gate.py` 没有写。
- **What #10 changed.** `nseq` 加到 `.s`(`isa_riscv_gen.py`)和 `.h/.ah/.b`(`isa_smallfloats.py`)各 10 条带整数操作数的标量 FP 指令(`feq flt fle fcvt.w fcvt.wu fcvt.<fmt>.w fcvt.<fmt>.wu fmv.x fclass fmv.<fmt>.x`),共 40 行,tag 集合与 `Rv32d` 一致;rv64 行不动。作者的 `gvsoc_snitch_fp_int_nseq.patch` 本机没有,所以是重建的。Two-place:两个 `.py` 在嵌套 core 工作树里改了,`soft_hier/gvsoc_core.patch` 里**新增**两个 file section(+126 行);对 `git archive HEAD`(bda268a4)的嵌套 core 用 `patch -p1` 打整个 patch:无 fuzz / offset / fail,两个文件与工作树 `cmp` 一致。GVSoC 提交 070936b(patch only,不含 stamp),父仓库 efec4c4(`git add third_party/gvsoc` only;wiki 指针没提交)。
- **Regeneration 确实发生.** `generate_isa` 没有 DEPENDS,所以先删了 `build/core/isa_snitch{,_fp_ss}_rv32imfdva.{cpp,hpp}` 再 `build_arch.sh`(67 s;GVSoC Makefile 的 `CMAKE_FLAGS ?= -j 6`,并行度 6,没有超过 8)。生成的 `isa_snitch_rv32imfdva.cpp` 里 `fcvt.w.s` 的 tags 数组现在与 `fcvt.w.d` 完全相同(`ISA_TAG_NSEQ_ID`=1 位为 1),`flt.s` / `fmv.s.x` / `fcvt.h.w` 同样;`gen_isa_snitch_rv32imfdva_cpp_60335333.so` 和 `..._fp_ss_..._61161152.so` mtime 更新。提交后再跑一次 `build_arch.sh`(5 s),`stamp OK`;`SOFTHIER_MODEL_DIR` 未设。
- **Rung c** `rungc-fix10`:**FAIL (exit 1)**,ROI 1526876 ns(之前 1520300,+0.43 %,说明 patch 在运行);marks:qkv 0.234 / attention 0.497 / rest 0.793 / layer 1.524 ms vs DSE §8.3 0.324 / 0.831 / 1.040 / 2.195(-28 / -40 / -24 / -31 %,与修前 0.233 / 0.493 / 0.791 / 1.517 几乎相同,预测中的"只慢 1-2 %"成立,所以 -30 % 不是 #10)。检查与修前一致:X LN1 Q K KT P0 V O PASS;**H bad=59/64 maxerr 16.3091,G bad=44/64 maxerr 8959.3789,OUT bad=58/64 maxerr 8174.5742**(三个 maxerr 与修前逐位相同,mismatch 样本值也相同,如 H[138,77] got 2.3984 want 2.0977)。wall 48 s,peak 865 MB,oom 0。**结论:H 的错误是确定性的,与 FP->int 的 cycle 竞争无关。**
- **Diagnosis(未改任何东西).** (1) 第一个错的 tensor 是 H。H 来自 `siglip.mlir:162-164`:`gemm %o,%wo0 -> %ao`(16 cluster,256x768x768,tile 256/384/128)、`add_bias %ao`(就地)、`add %x,%ao -> %h`;这些 `dump_samples` 都在层末执行,所以 H 之前的 O 对、H 错,但 H 内部三步哪一步错没法从现有输出分辨。(2) 相同形状/tile 的 GEMM+add_bias 在 Q/K/V 上是对的(K=768,同一个 add_bias 内核),所以 GEMM 和 add_bias 本身不是显然的嫌疑;新的东西是:(a) 输入 `%o` 是各 head 钉在 cluster 0..11 的 per-head GEMM 写的,然后一个 `group_barrier`;(b) 两输入的 `softhier.add`(`vfadd.h` + 两路 `fld`);(c) 就地 add_bias 后 HBM 写回的完成顺序。错误是 0.3 量级的正负混合,不是 inf,也不是 -10 %(#3 的偏低特征;且 #3 在 Q/K/V 的 K=768 上不出现)。(3) **修正(advisor review 之后)**:H、G、OUT 在 #10 前后逐位相同(含 mismatch 样本值),而 marks 移动了几 us,所以这是**确定性的功能错误**,不是 timing race;#10、#6、#8(两个 master 的顺序竞争)都不会在 timing 变化后给出逐位相同的错误值,不应作为首选候选。#3 不符(误差偏大而不是偏低,且 Q/K/V 在 K=768 通过)。#1 是唯一仍缺的、可能碰 FP 数据的作者 patch,但没有正面证据,**不提议**。(4) **已做的判别实验** `run.py rowops --rows 256 --cluster all`(`fix10-rowops`,exit 0,12.2 s,peak 642 MB):LN / SM / GELU / ADD / BIAS / SCALE / T 在 16 cluster 上对 numpy **全部 PASS**(256 samples,bad=0,maxerr LN 0.0017 / SM 0 / GELU 0.0008 / ADD 0 / BIAS 0 / SCALE 0.0002)。所以孤立的 row-op 和 FP 路径是对的(含 `add` 与 `add_bias`),#1 不再是首选。错误出在**组合**里:per-head GEMM(钉在 cluster 0..11)写出 O 之后,经 `group_barrier` 到 16-cluster o-proj GEMM / add_bias / add 的路径,或 HBM 就地读写的完成顺序。(5) 非 side-note 的候选(G3.2-diag 已提过,从未对 rung c A/B):我们的两处 iDMA 修复(0e4d719 的 FSM re-arm,2910e62 的 `can_accept_transfer`)。(6) 下一步建议(都用已有自检测试,不改前端):(a) `SOFTHIER_MODEL_DIR=~/.cache/softhier-vla/diag4/so_D`(deeploy 原始 iDMA)重跑 rung c,需先用该变量 `build_arch.sh` 写 stamp 再还原;(b) `run.py mesh`(12 cluster 的 64 列 slice,多 cluster 写同一矩阵再被 16-cluster GEMM 读);(c) 若仍不能区分,再在 H 的三步之间加 `dump_samples %ao`(改前端 / 另开 patch,需要 owner 批准)。#1 / #8 / #3 都没有动。
- **Regression(`SHVLA_SIM_MEMMAX=8G`,一次一个 sim,peak ≤ 842 MB,oom 0).**

| item | G3.1-diag4 ROI (ns) | after #10 (ns) | shift |
| --- | --- | --- | --- |
| rung 1: 256^3 / 256x768x192 / 512x768x768 / 256^3,0 / 256x256x512,1,1 / 1024x768x768 all / 1024x3072x768 all | 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / 520114 | 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / **520114** | 0 (7/7 PASS, exit 0) |
| rung 2: gemm1024_summa | 97682 | 97682 | 0 |
| gemm256 / gemm512 / gemm512_linalg / gemm_nonconst | 76984 / 85800 / 85800 / 85804 | same | 0 |
| mlp / mlp_linalg | 1064054 | 1064054 | 0 |
| add_test | 910393 | 910396 | +3 ns |
| siglip_encoder_layer | 10863627 | 10967081 | +0.95 % (published 17851422, unchanged gap) |

  Rung 2 9/9 PASS with all self-checks, exit 0, wall 329 s for the siglip layer. Only the scalar-FP-heavy programs move, by about +1 %, matching SIMULATOR_NOTES (+1.1 %).
- **Rungs 3-6 and the gate:** not re-run / not written. 理由:rung c 的失败在 H 就确定性地出现,rung 3-5 是同一套 LN/softmax/rope row-op + all-cluster add 路径,重跑的信息量为零(G3.2-diag 已有它们的 FAIL 记录)。不会 silently 继续到 rung 6。
- Simulators: 全部已停止。Commands: `run.sh published sim rungc-fix10 -- tests/gvsoc/run.py siglip-mlir --seq 256 --cluster all`;`... fix10-rung1 -- tests/gvsoc/run.py gemm`;`SHVLA_SIM_TIMEOUT=1500 ... fix10-rung2 -- tests/gvsoc/run.py mlir examples/*.mlir`(9 files)。Logs: `$SHVLA_DATA/runs/published/{rungc-fix10,fix10-rung1,fix10-rung2}/run.log`。
- What to inspect: (1) `soft_hier/gvsoc_core.patch` 的两个新 section 和 `isa_riscv_gen.py:604-620`;(2) rungc-fix10 的 run.log(H/G/OUT 数字与 G3.2-diag 逐位相同);(3) `siglip.mlir:162-164` 的 H 三步。

### G3.2-diag5 — iDMA A/B on rung c (so_D = deeploy original iDMA) and `run.py mesh`: iDMA is NOT the cause; rung c H/G/OUT fail bit-identically; STOPPED (next diagnostic needs owner approval)

- 摘要:G3.2-fix10 提出的两个判别实验做完了。(1) 用 `SOFTHIER_MODEL_DIR=~/.cache/softhier-vla/diag4/so_D`(deeploy 原始 iDMA,没有 0e4d719 也没有 2910e62)重跑 rung c:**H / G / OUT 的 bad 数和 maxerr 与当前 build 逐位相同**(H 59/64 16.3091,G 44/64 8959.3789,OUT 58/64 8174.5742,mismatch 样本值也相同,例如 H[138,77] got 2.3984 want 2.0977)。ROI 1494157 ns vs 1526876 ns(-2.1 %),说明 `.so` 确实被换了(timing 变了),数值没变。所以我们的两处 iDMA 修复不是 rung c 数值错误的原因,iDMA 嫌疑排除。(2) `run.py mesh`(默认 modes 0 1 2 5 6,12 head)**5/5 PASS**:slice-store 的 GEMM / DMA / scalar、attention 序列(5)、以及序列后接 16-cluster GEMM 读它(6,z2 自检 + 12 个 head 自检)都对。所以"多 cluster 写同一矩阵再被 16-cluster GEMM 读"这条路径在 GVSoC 里是对的,与 SIMULATOR_NOTES #6 的描述一致。没有改任何 GVSoC / 前端 / 编译器源码,没有 GVSoC 提交。
- Scope: safe mode(无 perf/gdb/strace,无 trace,一次一个 simulator,`SHVLA_SIM_MEMMAX=8G`,每次 run 前 `free -g` available ≥ 10 GB)。
- **so_D 一致性.** `so_*` 目录里只有一个文件 `gen_pulp_idma_snitch_dma_cpp_128919885.so`(iDMA 模块);#10 只改 ISS 的 ISA 表 `.so`(`gen_isa_snitch_*`),`--model-dir` 只覆盖同名 `.so`,所以 so_D 与当前 #10 build 只在 iDMA `.so` 上不同,不需要重建。`stamp.py check` 会因 `env.SOFTHIER_MODEL_DIR` 拒绝运行,所以每次 A/B 前备份 `.shvla_stamp.json`、用该变量 `stamp.py write`(不重建,安装的 `.so` 没动),跑完把原 stamp 拷回并 `stamp.py check` = OK(最终状态已核对)。
- **A/B matrix**(rung c = `siglip-mlir --seq 256 --cluster all`,ideal HBM):

| iDMA variant | 0e4d719 (fsm kick) | 2910e62 (can_accept) | H | G | OUT | ROI (ns) |
| --- | --- | --- | --- | --- | --- | --- |
| current tracked build (G3.2-fix10) | yes | yes | FAIL 59/64 maxerr 16.3091 | FAIL 44/64 8959.3789 | FAIL 58/64 8174.5742 | 1526876 |
| so_D deeploy original | no | no (no regulation queue) | FAIL 59/64 16.3091 | FAIL 44/64 8959.3789 | FAIL 58/64 8174.5742 | 1494157 |
| so_E release, no fixes | no | no (release queue) | 没有结果:3 分钟没出 ROI,CPU 满载,符合 diag4 里 release 无修复版 hang 的行为,手动 TaskStop | | | n/a |

  X LN1 Q K KT P0 V O 在两个有结果的 variant 里都 PASS。so_C(fsm kick only)同样应该 hang(diag4 的 summa 实验已测),没有重跑。
- **Mesh** `diag5-mesh`:mode 0 PASS 28090 ns,1 PASS 147863,2 PASS 197097,5 PASS 439394,6 PASS 473618;exit 0,peak 421 MB,oom 0。
- **Source reading(没有跑 sim,只读)排除的东西.** (a) 帧/种子:`siglip_ref.layer_reference` 的 wo(seed 5)/ bo(11)/ x(1) 与前端 `siglip.py:114-124` 一致,H = r16(x + r16(r16(o@wo)+bo)) 与 `siglip.mlir:162-164` 一致,所以不是 reference bug。(b) HBM buffer 布局(`siglip.mlir:3-17`)无重叠,最高 offset 23.6 MB = preload 的 sentinel 0x168c000。(c) o-proj 的 GEMM 与 Q/K/V 的 GEMM 是同一个 op(256x768x768,tile 256/384/128,`siglip.py:157` vs `:135`),`add_bias` 同一个 kernel;`rowops` 孤立 PASS;mesh 6 PASS。所以现在剩下的只有"全 16 cluster 的 o-proj GEMM 在 per-head 12-cluster 相(P.V 写 o)之后"这个**上下文**,以及 add/add_bias/LN 在同一个 binary 里的相互作用。
- **Hypotheses(都没验证,按我的优先级).** (1) 逐步定位优先:H 的三步(GEMM / add_bias / add)哪一步先错,现有输出无法区分,必须 `dump_samples %ao` 两次(add_bias 前后)。(2) SIMULATOR_NOTES "Host speed of the RedMulE functional model" 一节写道:ISS 用 `fesetround()` 仿真 RISC-V 舍入模式,会把引擎线程留在 RTZ/RDN/RUP,硬件 fp16 FMA 会遵守它,"first NEON build: P.V ~4 % low and H/G/OUT failed"(症状含 H/G/OUT)。我们的 `light_redmule.cpp` 是 scalar 路径(无 `gvsoc_redmule_neon.patch`),fp32 乘加依然受 FPCR 舍入模式影响;如果 o-proj GEMM 恰好在 ISS 把 FPCR 留在非 RNE 之后运行(per-head softmax / group_barrier 之后),GEMM 输出会有系统偏差。但这与 Q/K/V(K=768,同一 GEMM,紧跟 LN1 row-op)通过、O 通过相矛盾,所以只能算弱假设;检验方法:`REDMULE_NO_NEON=1` 不适用,需在 `light_redmule.cpp` 的 matmul 入口强制 `fesetround(FE_TONEAREST)` 做 A/B(改 GVSoC 源码,需 owner 批准)。(3) #1 / #8:`rowops` 与 mesh 都对,不优先。
- **Not done(按任务规则).** item 3 的修复不适用:iDMA variant 没有改变结果,所以没有提出/应用 GVSoC 修复,没有回归运行(rung 1 / 2 没动,仍是 G3.2-fix10 的数字)。
- **Proposed next diagnostic(需要 owner 批准).** 复制 `siglip.py` 的输出,在 H 的三步之间加 `softhier.dump_samples %ao`(GEMM 之后、add_bias 之后),以及 `%o` 在 o-proj 之前(确认 per-head 写回在 group_barrier 后完整),再跑一次 rung c(48 s)。这需要改前端(emit)或手写 mlir,当前 safe-mode 规则禁止。备选(也需批准):在 RedMulE matmul 入口强制 `fesetround(FE_TONEAREST)` 的 `.so` A/B(`SOFTHIER_MODEL_DIR` 临时 swap)。
- Simulators: 全部已停止(`pgrep gapy|gvsoc_launcher` 为空)。Commands: `/tmp/ab.sh D diag5-D tests/gvsoc/run.py siglip-mlir --seq 256 --cluster all`(包装 `run.sh published sim diag5-D -- ...` + stamp 备份/还原),`run.sh published sim diag5-mesh -- tests/gvsoc/run.py mesh`。Logs: `~/.cache/softhier-vla/runs/published/{diag5-D,diag5-mesh}/run.log`。过程事故:第一次 mesh 我用 `timeout 110` 包了,在 mode 6 被杀,孤儿 `run.py mesh` 与第二次启动短暂重叠(共用 `tests/gvsoc/mesh_slices/shape.h`);第二次的 5 个 mode 全部 PASS,孤儿随后自己结束,没有数据污染的迹象(结果与第一次已完成的 mode 0/1/2/5 数字逐位相同)。

### G3.2-diag6 — rung c under DRAMSys (ideal-HBM switch OFF): H / G / OUT fail bit-identically; HBM model cleared; STOPPED

- 摘要:orchestrator 的假设是 ideal-HBM 的 `Memory`(size / 地址 mask / channel offset)有问题,导致 buffer 别名或越界被吞。实验结果:用真正的 DRAMSys 跑 rung c,**H 59/64 maxerr 16.3091、G 44/64 8959.3789、OUT 58/64 8174.5742,mismatch 样本值与 ideal HBM 逐位相同**(H[138,77] got 2.3984 want 2.0977;G[188,355] -0.0908 vs -0.1624;OUT[164,723] 245.75 vs 2.1777)。X/LN1/Q/K/KT/P0/V/O 全 PASS。所以 HBM 模型不是原因,ideal-HBM 开关被排除。没有改任何源码,没有回归运行(没有修复可回归)。
- **怎么关掉 ideal HBM(无源码修改).** `run_sim(ideal_hbm=True)` 默认值会无条件 `env["SOFTHIER_IDEAL_HBM"]="1"`(`softhier-mlir/softhier_mlir/sim/gvsoc.py:171-172`),`tests/gvsoc/run.py` 所有调用都不传该参数,环境变量覆盖不了。干净的做法:`/tmp/diag6/dram_wrap.py`(仓库外)先 `import softhier_mlir.sim.gvsoc`,把 `run_sim` 换成 `functools.partial(run_sim, ideal_hbm=False)`,再 `runpy.run_path("tests/gvsoc/run.py", run_name="__main__")`;`run.sh` 的 `python <script> <args>` 直接接受它。验证:ROI 1788935 ns(ideal HBM 是 1526876),peak 1287 MB(ideal 约 400 MB),确认走的是 DRAMSys。
- **Result.** `run.sh published sim diag6-dram -- /tmp/diag6/dram_wrap.py siglip-mlir --seq 256 --cluster all`:exit 1,ROI 1788935 ns,wall 87.2 s,peak 1287 MB(cap 8G),oom 0;marks qkv 0.282 / attention 0.538 / rest 0.965 ms。
- **地址映射阅读(只读).** (1) `flex_cluster.py:171-175`:ideal 分支 `Memory(size=hbm_total_size, width_log2=0, latency=0, atomics=False)`,`hbm_total_size = hbm_node_addr_space*2*(num_x+num_y)` = 0x4000000*2*8 = 1 GiB **每个 channel**,即整个 HBM 窗口(`virtual_interco` 映射的大小同为此值,flex_cluster.py:~237)。(2) `hbm_ctrl.cpp:131-166`:`bank_id=(offset>>il)&bank_mask`,`bank_offset=((offset>>(stage_bits+il))<<il)+(offset&((1<<il)-1))`,即 channel-local offset ≤ 窗口内偏移 / 通道数,必然 < 1 GiB(arch: `hbm_chan_placement=[4,0,0,4]`,8 个 channel,`hbm_node_aliase` 默认 1)。(3) `pulp/chips/flex_cluster/memory.cpp:282-285`:`offset+size > size` 时 `force_warning` 并返回 `IO_REQ_INVALID`,**不 mask 也不 wrap**,越界不会被静默吞掉。(4) rung c 的 buffer 布局(`siglip.mlir:3-17`)最高 offset 23.6 MB(sentinel 0x168c000,G3.2-diag5 已查无重叠),比 1 GiB 小两个数量级;per-channel local offset ≤ 3 MB。无别名、无越界。`atomics=False` 只影响 HBM 上的 atomic 操作,rung c 不使用。
- **结论.** HBM 模型(ideal 与 DRAMSys 两种)都排除。iDMA(diag5)、HBM(diag6)都排除;剩下的仍是 diag5 的假设:H 的三步(o-proj GEMM / add_bias / residual add)里 RedMulE/FP 路径的问题(如 fesetround 状态、`light_redmule.cpp` 的 fp16 转换),需要 `dump_samples` 中间点才能分辨,需要 owner 批准改前端/手写 mlir。
- Simulators: 全部已停止。Logs: `~/.cache/softhier-vla/runs/published/diag6-dram/run.log`。

### G3.2-diag7 — H-chain dumps (patch 0006): the o-proj chain is NOT the first bad step; the raw Q/K/V GEMM outputs already contain sparse corrupt elements (4 distinct bad elements: Q 1, K 2, V 1, among ~67 k distinct sampled each); location moves with the GEMM tiling; STOPPED (not a minimal softhier-mlir fix: mechanism unknown, suspected RedMulE / GVSoC)

- 摘要:owner 批准后加了 softhier-mlir 补丁 `0006-shvla-siglip-h-chain-dumps`(opt-in)。结论推翻了 diag-diag6 的"H 三步里有一步错"的假设:**H 的三步(o-proj GEMM / add_bias / add)没有问题**。真正的第一个错更早:`siglip-mlir` 里 Q/K/V 的 **GEMM 原始输出**(add_bias 之前)就有稀疏的坏元素(偏差约 ±2.0,V 里一个是 -511);64 个样本的 end-of-program 检查(覆盖 0.03 %)看不到它们,所以 X/LN1/Q/K/KT/P0/V/O 都显示 PASS。坏元素经 attention 放大成 O 的坏列/坏头,再经 o-proj、residual、LN2、MLP 变成 H/G/OUT 的大面积错误。
- **补丁 0006.** 文件:`frontend/siglip.py`(`emit()`)、`testing/siglip_ref.py`(`layer_reference()` 多返回的 key)。这两个文件 0001-0005 都不碰,所以对它们 `git diff` 即 0006;`run.sh apply` 报 `6 patches applied`;README 加了一行。Opt-in:环境变量 `SHVLA_SIGLIP_H_DUMPS=<n>`(未设或 0 时 `emit()` 输出与未打补丁逐字节相同,已验证 layers=1、2 与 `fused_attention=True`);`SHVLA_SIGLIP_H_REGEMM=1`(o-proj GEMM 再跑一次:OIN2/AOR2);`SHVLA_SIGLIP_H_FULL=<m>`(q/k/v/o 的 m 个样本 QF/KF/VF/OF,以及 q/k/v 在 GEMM 后/add_bias 后的 QG/QB 等)。layer 0 内每个 dump 后面跟一个 `group_barrier`(否则 cluster 0 在打印时其他 cluster 已经就地跑下一个 op)。`run.py` 不用改(它对每个设备打印过的 tag 自动查 ref)。注意:`run.sh sim` 每次先 `apply` 重置子模块,子模块树里的临时改动会被抹掉,实验用的改动必须进补丁文件(中途踩过一次)。
- **前置离线检查(零 sim).** 用 `last_run.log` 的 H 样本对 `x+r16(o@wo)`(缺 bias)、`x+bo`、bias 加两次、`o` 只保留前 N 列(N=64..704)等候选公式:全部 bad=56-62/64,没有一个匹配,所以不是"某一步缺失"这类简单形式。
- **Dump 表**(rung c,`siglip-mlir --seq 256 --cluster all`,`SHVLA_SIGLIP_H_DUMPS=256`;H/G/OUT 的数字与默认运行逐位相同:maxerr 16.3091 / 8959 / 8174):

| step / tag | 样本 | bad | maxerr | 模式 |
| --- | --- | --- | --- | --- |
| X, LN1, Q, K, KT, P0, V(4096 点) | 4096 | 0 | ≤0.0098 | 通过 |
| QF / KF / VF(150000 点,约 67 k distinct,约 34 %) | 150000 | 5 / 3 / 1 命中(含重复采样;distinct 坏元素 1 / 2 / 1) | 2.0 / 2.0 / 511 | 孤立单元素:Q[23,273],K[1,167],K[138,278],V[182,111] |
| QG / KG / VG(GEMM 之后、add_bias 之前) | 150000 | 5 / 3 / 1 | 2.0 / 2.0 / 511 | 与 QB/KB/VB、QF/KF/VF 是同一批元素:**GEMM 输出就已经错** |
| 坏元素偏差(raw) | | | | Q -1.9971,K -1.9987,K +1.9941,V -511.15;add_bias 之后同样偏移,add_bias 没有引入 |
| PH0..PH11(softmax 输出,128 点/头) | 128 | 0(PH7:1) | ≤0.0023(PH7 0.98) | PH7[153,95]=0.985 vs 0.0055 |
| OH0..OH11(o 的列片,128 点/头) | 128 | OH1 5, OH2 4, OH5 1, OH7 75, 其余 0 | 209.8 (OH1) | OH1 坏在 col 20/47,OH2 在 col 18(同一列多行,最大 -209);OH7 整个头散布 |
| O(4096 点) | 4096 | 163 | 36.4 | |
| OF(150000 点) | 150000 | 3066 / 67040(4.6 %) | 693 | 坏列集中:global col 84、111、146 各约 88 个坏行(几乎所有行),head 7 的 64 列各约 50-58 个 |
| OIN(o-proj GEMM 读到的 %o) | 256 | 10 | 1.46 | 全在 head 7(col 449-507) |
| OIN2(第二次 o-proj 前的 %o)、OEND(程序末尾,seed 203) | 256 | 10 / 10 | 1.46 | **与 OIN 逐位相同**:O 的错是确定的,不是晚落地 / race |
| WOI、BOI(wo0、bo0 使用前) | 256 | 0 / 0 | 0 | 参数没坏,preload 没问题 |
| AOR(o-proj GEMM 原始输出) | 256 | 243 | 44.3 | 均匀分布在 16 个 row block 与两个 N tile;got std 7.0 vs want 0.42,与任何 `A@W`(A 取 X/LN1/Q/K/V/O,W 取 wq/wk/wv/wo)相关系数 <0.12 |
| AOR2(同一个 GEMM 再跑一次) | 256 | 243 | 44.3 | 与 AOR **逐位相同**:确定的 |
| AOB、HCH、H、G、OUT | 256 / 256 / 4096 / 4096 / 4096 | 245 / 241 / 3858 / 2949 / 3841 | 44.3 / 44.3 / 512 / inf / inf | 只是 O 的错的传播 |

- **解读.** (1) AOR 的大面积错来自 O 里稀疏的巨大坏元素(V 里一个 -511 使 OF 的 col 111 几乎每行都坏)乘 wo 后铺满整行,不是 o-proj GEMM / add_bias / add 本身的错,也不是 group_barrier / 12→16 cluster 交接的问题(OIN = OEND = OIN2:O 在交接时已完整且稳定)。注意 diag4-6 里 "O 对" 只是 64 点采样的盲区。(2) 第一个错的 step 是 Q / K / V 的 `softhier.gemm`(raw 输出)。每个坏元素偏差约 ±2.0(或 -511),单个乘积最大 0.19(`|ln1|≤3`,`|w|≤0.062`),所以不是少加/多加了一个乘积,而是输出元素(或 RedMulE 内部累加)被一个垃圾值污染。(3) 确定性:在试过的扰动下(加 dump/barrier、第二次 o-proj、ideal/DRAMSys、so_D iDMA)H/G/OUT 与 AOR 逐位相同;注意 dump 都在 Q gemm 之后,K/V gemm 之前只多了 barrier,所以这不能证明与 timing 无关,只是比 diag5/6 多一些扰动。(4) 对比(见下面 Addendum,已更新):孤立纯 GEMM 同 cfg(`tm=256 tn=384 tk=128 pipeline=1 acc=0 fmt=FP16`,与 `mlir_app/main.c` 里 Q gemm 的 cfg 逐字段相同)在 600000 个 Z 样本(约 95 % 覆盖)上 0 bad。所以不是该 GEMM 配置本身,而是数据或上下文相关。
- **Root cause / 最佳假设.** 稀疏、确定、单元素、幅度远大于任何单个乘积的 fp16 输出污染,发生在 RedMulE GEMM 的输出路径(累加/回写,或输出 tile 的 TCDM→HBM 写回),而不是 softhier-mlir 的 add_bias / add / barrier 序列。这部分在 GVSoC,按规则 **STOP,不改**。dump 只能定位到 "GEMM 的 raw 输出元素",因此没有 file:line;softhier-mlir 侧的相关位置:`softhier_mlir/frontend/siglip.py:144-146`(Q/K/V GEMM + add_bias),`runtime/sh_gemm.inc.c:53-116` 已读(Addendum),机制未定。
- **没有 fix。** 所以没有跑 rung 2 / rung 1 / rung 3-5 回归;补丁 0006 默认行为不变(`emit()` 对默认参数逐字节相同),不影响这些 rung 的 ROI。
- **未做 / 建议下一步(需要 owner 决定).** (a) 对 Q/K/V GEMM 的输入(LN1、wq/wk/wv)做全覆盖 dump,确认输入没坏(LN1 的一个坏行会让整行输出坏,我们看到的是单元素,所以不太可能);(b) 在纯 GEMM 测试里用 siglip 的真实数据范围复现,区分"数据相关"(fp16 溢出/NaN 处理)和"与前序 op 的状态相关"(例如 layernorm 之后残留的 FP/CSR 状态);(c) 读 `light_redmule.cpp` 输出写回路径相对作者私有 patch 的差异(side-note #1/#3 之外的 RedMulE 部分)。
- Simulators: 全部已停止。Logs: `~/.cache/softhier-vla/runs/published/diag7-{hdump,hdump2,regemm,oend,heads,many,full,qkvg,gemm}/run.log`。memory peak ≤ 0.9 GB(cap 8G),oom 0。

#### G3.2-diag7 Addendum — advisor review: pure-GEMM at 95 % coverage, tile mapping, tiling A/B, Y-zero experiment

- **纯 GEMM 全覆盖.** `run.sh published sim diag7-gemm600 -- tests/gvsoc/run.py gemm --shapes 256x768x768:256,384,128,1,0,all --real --dump-z 600000`:exit 0,600000 个 Z 样本(约 95 % distinct 覆盖)对 numpy **0 bad**,max abs 0.00238。siglip 里同一个 gemm 预期每个 GEMM 约 3-4 个坏元素,所以这个干净结果不是统计巧合(之前 100 k 点只有约 20 % 的把握)。cfg 与 `mlir_app/main.c:75` 的逐字段一致。区别只剩数据(这个测试 X 为 0..0.0156,W ±2;siglip 为 LN1 ±3,W ±0.062)和上下文(前面有 layernorm 等 op)。
- **cluster / tile 映射.** `runtime/sh_gemm.inc.c:80`:输出 tile 按 `(r*NT+col) % P` 分给 cluster,**没有 split-K,没有跨 cluster 规约**。siglip 里 256x768x768、tile 256/384/128:MT=1、NT=2,**只有 cluster 0 和 1 在算**(各一个 256x384 tile,K 方向 6 个 128 的 K-tile,pipeline 双缓冲)。4 个坏元素(Q[23,273]、K[1,167]、K[138,278]、V[182,111])全在 col<384 = **cluster 0 的 tile**(cluster 1 的 tile 里没有观测到)。
- **Tiling A/B(`--tiles 256,256,256`,`diag7-tiles256`).** 此时 NT=3,cluster 0/1/2 各一个 256x256 tile。Q、V 的 raw 输出全干净(QG/VG 0 bad);K 有 2 个坏元素:K[61,519] 得 -2.0000(应为 -0.00055)、K[83,542] 得 -2.2520(应为 -0.2556),列 512-767 = **cluster 2 的 tile**。偏差仍是 ≈ -2.0。所以:(a) 坏元素的集合随 tiling 变,不是固定的数据位置(也不是固定 cluster);(b) 仍是"每个 GEMM 某一个 cluster 的输出 tile 里 0-5 个元素被加了约 ±2 的值",量级和个数在两种 tiling 下相近;(c) 这排除了"某个具体输入元素坏"(坏输入会随数据位置固定)。
- **Y 清零实验(未保留).** 假设:偏差 ≈ ±2.0 恰好是 x 的取值范围(LCG ±16 × 0.125),-511 像 layernorm 留下的 fp32 参数被当成 fp16 读,即 Y tile 里有几个元素没被 iDMA(ZOMEM)清零,带着 TCDM 的残留值进了累加。临时补丁(`sh_gemm.inc.c`:iDMA 清零之后再用所有 core 把 Y 写零并同步)跑 rung c(`diag7-corezero`):QG/KG/VG、OF、H/G/OUT 与不加时**逐位相同**(Q 5、K 3、V 1)。所以不是 Y 清零不完整;该临时补丁已删除(`run.sh apply` 回到 6 patches)。
- **剩下的.** 误差是确定的、加性的(≈ ±2 或 -511)、只在一个 cluster 的输出 tile 内个别元素、不是单个乘积能解释(≤0.19)、不是坏的 X/W 输入元素(那会污染整行/整列)、不是 Y 初值、不是 add_bias。还没有排除:RedMulE 内部累加/写回的瞬态(GVSoC `light_redmule`)、输出 tile 经 `sh_store_block_sync`(256 个 1-D iDMA,`sh_gemm.inc.c:47-51`)写回 HBM 时与其它写的交叠、以及 dump 之外的读取时机。这些都需要 GVSoC 侧的 trace 才能继续,按规则 STOP。推荐下一步(需 owner 批准):对 cluster 0 的 Q gemm 做 `--trace` 的 RedMulE 记录,看坏元素所在输出行块的 RedMulE 事件是否异常;或在 GVSoC 里对该 GEMM 的 Y 读回做 checksum。
- **`layer_reference` 的修正(grep 的结果).** 调用者有两个:`tests/gvsoc/run.py:307`(`run.py siglip`)对 ref 里**每个** key 要求设备打印,缺了就 `MISSING` 并判 FAIL;`:346`(`siglip-mlir`)才跳过没打印的 tag。所以 0006 的初版(无条件返回多余的 key)会破坏 `run.py siglip`。已改:多余的 key 只在 `SHVLA_SIGLIP_H_DUMPS` 设置时返回,默认返回的 key 集合和顺序与未打补丁相同(`X LN1 Q K KT P0 V O0 O6 O H G OUT`,已验证);补丁文件重新生成,`run.sh apply` 6 patches。默认 rung c(`diag7-default`)结果与 diag6 一致:H 16.3091 / G 8959.3789 / OUT 8174.5742,exit 1,wall 49.5 s,ROI 1526876 ns。

### G3.2-diag8 — Q[23,273] 的写者追踪:**RedMulE 模型本身写错**(verdict a)。根因 = GVSoC `light_redmule.cpp` 的 `float_to_fp16` 在 fp16 次正规范围(|v| < 2^-14)的转换 bug;输入 TCDM 正确、没有额外写者、HBM 与 TCDM 一致;未改 GVSoC(STOP,给出 proposal)

- 摘要:owner 批准做 simulator 写 trace。结论:Q[23,273] 的坏值 **-2.0(0xc000)是 cluster 0 的 RedMulE 在第 4 个 K-tile(kk=3)的输出**。该 K-tile 的真实累加结果是 **+4.5657e-5**(落在 fp16 次正规范围,< 6.1035e-5),GVSoC `float_to_fp16` 的 `exponent <= 0` 分支位移写错(`>> (1 - exponent)`,应为 `>> (14 - exponent)`),mantissa 溢出到指数/符号位,得到垃圾 0xc000。bit-exact 复现:Python 按模型逻辑(fp32 顺序 FMA + 该 `float_to_fp16`)算 kk=3,得到 `0xc000`;用修正后的位移得到 `0x02fe`(= 4.5657e-5,等于精确值)。其它三个坏元素(K[1,167]、K[138,278]、V[182,111];以及 tiles256 A/B 的 K[61,519]、K[83,542])同类(次正规/近零的输出被误转),与 diag7 的"±2.0 / -511 的孤立元素、随 tiling 移动、纯 GEMM 干净(数据不同,几乎没有 |v|<6e-5 的累加终值)"全部吻合,但这三个只是推断,没有逐个 peek。
- **地址映射.**
  - HBM:Q = `hb3 = sh_hbm_addr(790528)`,fp16,ld=768。`Q[23,273]` = 0xc0000000 + 790528 + (23*768+273)*2 = **0xc00c9c22**(行 23 起点 0xc00c9a00,iDMA 第 56 个 1D txn:src 0x54500 → dst 0xc00c9a00,size 0x300)。
  - TCDM(cluster 0,`sh_gemm.inc.c` 布局 `X0|X1|W0|W1|Y`,tm=256 tn=384 tk=128 pipeline=1):xb=0x10000,wb=0x18000,x0=0,x1=0x10000,w0=0x20000,w1=0x38000,**y=0x50000**(单缓冲,K-tile 间原地累加)。`Y[23,273]` 的偏移 (23*384+273)*2 = 18210 → **TCDM 0x54722**。
  - bank:bank 宽 **8 B**(不是 64 B:`Building Memory size 0x4008` = 0x100000/64 + 8;`hwpe_interleaver.cpp:55-58` 同一公式),64 bank,`bank = (addr>>3)&63`,`bank_off = ((addr>>9)<<3) | (addr&7)`。所以 0x54722 → **`/chip/cluster_0/tcdm/bank_36`,bank 内 offset 0x1518(8 B 字,元素在字内 +2)**。(第一版我按 64 B 算成 bank_28,advisor 指出后改正;pilot 里的 bank_28 0x1400 = 0x500E0,是 Y 清零的第一个字。)
- **trace 命令与 filter**(无源码改动;`diag8/trace_wrap.py` 在进程内 monkeypatch `softhier_mlir.sim.gvsoc.subprocess`,给 gapy 加 `--trace=`,**流式过滤**,没有被过滤的行不落盘;规则 `[路径子串, 保留 regex, cap, 起, 止(ps)]`;`D8_STOP_PS` 到点 kill gapy;输出上限 50 MB)。run 1(`diag8/run1.sh`,rung c,`SHVLA_SIM_MEMMAX=8G`,看到 187 k 行,保留 265 行 / 72 KB,wall 约 20 s,RSS peak 520 MB):
  - `--trace=/chip/cluster_0/tcdm/bank_36/trace`,保留 `offset: 0x15[0-3][0-9a-f],`(该字所在的 64 B 窗口)。
  - `--trace=/chip/cluster_0/pe[0-2]/insn` 与 `fp_ss[0-2]/insn`:保留访问 TCDM 的 store/amo:(a) `PA:(000|300)5472[0-7]`(该字),(b) 330-363.5 us 窗口里所有 TCDM store(只为判断窗口内有没有核写 TCDM)。insn trace 带 `PA:` 地址和数据寄存器值,所以核的写者可以直接定位。
  - `--trace=/chip/cluster_0/redmule/trace`、`.../idma/fe/trace`:只留 `Finished` 行(默认级别,不开 LEVEL_TRACE,否则每个 TCDM req 都打印)。
  - 注意:gvsoc trace 行首第一个时间戳在个别行有 1 ms 偏移,窗口/停止改用第二个字段(ns)。`Memory access` 行只有 offset/size/is_write/op,**没有数据也没有 initiator**;iDMA/RedMulE 的 trace 也不带数据。所以 trace 只能回答"谁、何时访问",值要靠 peek(见下)。
- **访问时间线(bank_36 offset 0x1518 = TCDM 0x54720,含 Q[23,272..275])**,时间 ns(Q GEMM:iDMA 清零 333.7 us,RedMulE 6 个 op 336.7-358.8 us,iDMA 存储 358.8-362.1 us):
  - 333706:W(8 B) —— iDMA ZOMEM 清零 `Y` 的 1D txn(Finished 334054,src 0x18000000 → dst 0x50000,size 0x20000)。
  - 每个 K-tile 一次 R 一次 W(RedMulE,Y 读-改-写):kk0 R 337508 / W 338968;kk1 R 341209 / W 342668;kk2 R 344914 / W 346374;kk3 R 348616 / W 350075;kk4 R 352321 / W 353781;kk5 R 356004 / W 357412。
  - 359123:R(8 B)—— iDMA 存储读出(txn 33-288 之一,Finished 362149)。
  - 窗口 0x1500-0x153f 共 112 次访问 = 8 (清零) + 6x16 (RedMulE) + 8 (iDMA 读),**没有任何额外访问**;cluster 0 的 3 个核 + 3 个 FP 子系统在 330-363.5 us **没有任何 TCDM store**(363.08 us 起才有 `sh_rowop` 的 `fsw` 到 0x5000-0x58xx,那是下一个 op `add_bias`,而且在 Q 存储完成之后)。别的 cluster 的写会作为额外 bank 访问出现,没有。**所以不是 (c)。**
- **值 peek(临时补丁,`diag8/peek-q23-273.patch`,run.sh 重新 apply 后已清掉,没进 patches/)**:cluster 0、首个 `sh_gemm`(z = hbm 790528)里,每个 `flex_redmule_wait()` 之后由 first core 打印 Y[23,273] 的 fp16 位、并用核在 fp32 里按 TCDM 里的 xcur/wcur 算这个 K-tile 的部分和(kk=3 另外打印 X[23,0..127]、W[0..127,273] 的 fp16 位,`diag8/peek_output.txt`);dm core 在存储后读 TCDM 和 HBM。
  | kk | Y 之后(fp16) | 值 | 核用 TCDM 输入算的部分和 | 差值 |
  | --- | --- | --- | --- | --- |
  | 0 | b647 | -0.39233 | -0.39252 | -0.39233 |
  | 1 | b57c | -0.34277 | +0.04939 | +0.04956 |
  | 2 | b21d | -0.19104 | +0.15173 | +0.15173 |
  | **3** | **c000** | **-2.00000** | +0.19109 | **-1.809(应为 +0.19109)** |
  | 4 | c146 | -2.63672 | -0.63822 | -0.63672 |
  | 5 | c188 | -2.76563 | -0.12992 | -0.12891 |
  "差值" = 本 op 的 Y 变化(fp16 取整,约 1e-3),除 kk=3 外都和部分和一致(kk=0 的部分和 -0.39252 对 Y 变化 -0.39233 同样一致)。kk=3 的精确结果:`y0 + Σ x*w` = -0.19104 + 0.19109 = **+4.5657e-5**,RedMulE 给出 -2.0。之后 kk=4/5 的增量都正确(只是在错的 -2.0 上累加),pre-store TCDM = c188 = post-store HBM(0xc00c9c22 读回 c188):**TCDM 与 HBM 一致,存储没问题,不是 (b)**。
  - 周边证据:inputs 正确(部分和用 TCDM 里的 X/W 算,与 y 的其它 K-tile 增量吻合);kk=3 之前 Y 正确;trace 里没有额外写者。所以只剩 RedMulE 在 kk=3 的 op 内部算错。
- **根因(GVSoC,未改).** `third_party/gvsoc/soft_hier/flex_cluster/light_redmule.cpp:1248-1262` `float_to_fp16`:
  ```
  int32_t exponent = floatBits.parts.exponent - 127 + 15;
  if (exponent <= 0) {                       // fp16 subnormal / underflow
      if (exponent < -10) return sign;
      mantissa = (floatBits.parts.mantissa | 0x800000) >> (1 - exponent);   // <-- 位移错
      return sign | mantissa;                // 且没有掩码,高位溢出到指数/符号
  }
  ```
  fp16 次正规尾数 = ((1<<23)|m23) >> (14 - exponent)(exponent <= 0)。现在的 `1 - exponent` 少移 13 位,结果的高位进入指数和符号位。`matmul_fp16`(:1284-1295)每个输出元素在 fp32 里累加再 `float_to_fp16` 一次,所以任何 |终值| < 2^-14 且非零的输出都会变成垃圾(大小/符号随尾数位而定,本例得到 0xc000 = -2.0,V 里得到 ~-512)。这也解释了为什么只在 siglip 里出现:LN 输出有正有负,K 维累加在部分 K-tile 终点会相消到接近 0,而前面纯 GEMM 的数据(X 为 0..0.0156 非负、W ±2)很少落到 |v|<6e-5。坏元素集合随 tiling 变是因为每个 K-tile 的 Y 终值(RedMulE 每 op 写一次)取决于 tile 边界。同一函数的 `fp16_to_float`(:1267-1275)对 fp16 次正规输入设 exponent=0 把它当成 fp32 次正规(值约 1e-40,等于 0),误差 <6e-5,也该一并改(次正规:`ldexp(m, -24)`)。另外 `float_to_fp16` 是截断而不是 RTNE,与 numpy 差 1 ulp 量级,不是本问题。
  - **Proposal(未应用,GVSoC 源码,按规则 STOP):** (1) `>> (14 - exponent)`(并保持 `exponent < -10` 的下限);(2) `fp16_to_float` 加次正规分支;(3)(可选)改 RTNE。bit 级验证(`diag8/emul.py`):用 kk=3 的真实 X/W/Y,现模型 → 0xc000;改位移后 → 0x02fe = 4.5657e-5(精确值)。**不需要也不能**在 softhier-mlir 侧补(RedMulE 的数值转换在 GVSoC 里;只有用 FP32 acc 之类的 cfg 才能绕过,但 `flex_redmule` 本身是 fp16 输出)。不动 GVSoC 就无法修复,所以这次没有跑回归 / rung 1/2/3-5。
- **没有 fix 应用:** patches 仍是 6 个(`run.sh apply` 确认);0007 是一次性 peek 补丁,存放在 `diag8/` 做记录,没有放进 `patches/`。
- **下一步(需 owner 批准):** 在 GVSoC 私有 patch 上做 (1)(2),重建 RedMulE model(`SOFTHIER_MODEL_DIR` 指向单独的 model 目录,不动共享 install),跑 rung c(SHVLA_SIGLIP_H_FULL),预期 Q/K/V 0 bad,H/G/OUT 恢复;然后全回归(rung 1: 7 shapes,1024x3072x768:256,256,256,1,0,all ≈ 520114 ns;rung 2: 9/9;rung 3-5)。同时这个 bug 影响所有用 RedMulE fp16 的 GEMM(包括之前"通过"的 SmolVLA 阶段:64 点采样看不到)——建议修复后重新验 G3.2 之前的结果。
- 文件:`experiments/smolvla-e2e-baseline/diag8/{trace_wrap.py, run1.sh, emul.py, peek-q23-273.patch, peek_output.txt, run1_bank36_idma_redmule.trace.txt}`。Logs:`~/.cache/softhier-vla/runs/published/diag8-{pilot,pilot2,tl,run1,peek,peek2}/`。运行次数:2 次 pilot(gemm,18 s)+ tl(siglip,trace redmule/idma,73 s)+ run1 + 2 次 peek。Simulators 全部已停止;memory peak ≤ 870 MB(cap 8G),oom 0。
- **diag8 补充(advisor 后)**:(1) 为什么坏值集中在 ±2.0:kk=3 的 fp32 累加终值是 `0x383f8000`(`emul.py` 末行),尾数只有 8 个有效位(低 15 位为 0,因为 W 只有 2-3 个有效位、X 为 fp16),`ex=0` 分支(位移 1)再截到 16 位后只能得到 {±0, ±2.0},位移更大的情况落在别的 2 的幂(V 里的 ~-512)。尾数位恰好为 0 的次正规终值会静默变成 0/看上去正确,所以观测到的坏元素率比"终值落在 |v|<6e-5"的概率低。这个只在一个元素上验证(Q[23,273]),K/V 的坏值是推断。(2) 栈不是嫌疑:pilot2 的 insn trace 里核的 `sp` 是 0x1000ff90 / 0x10017f90 / 0x1001ff90,即 `ARCH_CLUSTER_STACK_BASE`(0x10000000)的独立 `stack_mem`(每核 32 KB),不在 TCDM,栈溢出不会写进 Y tile;和"1 KB stacks"的前提不符(`flex_start.s` 的 sh_call_on_core_stack 是另一回事,没有写 TCDM)。(3) fix proposal 的 (2) `fp16_to_float` 次正规处理是必需的,不只是"可选":(1) 修好后 RedMulE 会写出 fp16 次正规值,下一个 K-tile 读回时若被当成 ~0 会引入新的小误差。(4) `diag8/run1.sh` 的包装脚本路径改为仓库内的 `diag8/trace_wrap.py`(从 `experiments/smolvla-e2e-baseline` 下运行)。

### G3.2-fix-redmule proposal — GVSoC RedMulE fp16 转换修复(side-note #3 同函数 `float_to_fp16`;已知 side-note 项,按失败规则:先写 proposal,再在本地 GVSoC 分支 `shvla/soft_hier_release` 上应用)
- **作者 side-note #3 的做法(读了 `softhier_vla_summary_2026-10.md` §6、`deeploy_softhier_progress.md`、softhier-mlir `docs/SIMULATOR_NOTES.md` #3):** "`float_to_fp16` 在 `light_redmule.cpp` 截断尾数,每次 FMA 的累加都向零舍入 → 768 项 fp16 点积偏小约 10%",补丁 `gvsoc_redmule_fp16_rne.patch`:**IEEE round-to-nearest-even("真实 RedMulE 是会舍入的")**。笔记里**没有**提次正规(subnormal)处理;补丁文件本身在本机找不到(`../SoftHier/dse/patches/` 不存在,全盘 find 无,gvsoc `git log -S nearest` 只有不相关的历史),所以 B 是**按笔记描述重新实现**,不是搬补丁。已发布精度数字是用 RNE 但(推测)带着次正规 bug 跑出来的,修次正规后与发布数字可能有小偏差。
- **为什么是两个提交:** A = diag8 证实的 bug 修复(`1 - exponent` → `14 - exponent` 并掩码,`< -10` 下限保留;`fp16_to_float` 加次正规和 inf/NaN 解码);B = 笔记明确要求的 RNE(`float_to_fp16` 的正规 / 次正规路径都加舍入位,进位自然进入指数)。笔记指定了 RNE,所以 B 做;分开是为了能单独回退、单独看对数值的影响。
- **文件状态:** `third_party/gvsoc/soft_hier/flex_cluster/light_redmule.cpp` 是 GVSoC 分支里的**被跟踪文件**,直接改,不走 patch(两处规则不适用)。唯一的转换路径:该文件 :1248/:1267,`matmul_fp16` 里用(:1289/:1294);没有其它拷贝。`floonoc.cpp`/`gvsoc_pulp.patch`/SDK `flex_libfp16.h` 里各有一份同样有 bug 的 `float_to_fp16`(side-note #14),**不动**(REDADD / 核端,不在本次范围),记在这里。
- **验证计划:** 把两个 C 函数从源文件原样 `sed` 出来编译成主机测试,(1) `fp16_to_float` 对 65536 个 fp16 位型对照 numpy;(2) `float_to_fp16` 对 numpy float16(RNE)/ 向零参考做稠密 float32 扫描 + 命名边界(0、±最小/最大次正规、最小正规、inf/NaN、2^-24、2^-25);(3) `emul.py`(改读 `diag8/peek_output.txt`)Q[23,273]:A → 0x02fe;rung c 全量 dump QG/KG/VG 0 bad;再跑回归 + rung 3–5(6)。

### G3.2-fix-redmule — GVSoC RedMulE fp16 conversion fixed (A: 次正规 bug, B: RNE);rung c / 3 / 4 / 5 / 6 全部 exit 0,**准确度全 PASS**;G3 gate 准确度 PASS、**时间 MISS(91.139 ms,-23.5 %)**;按规则记录并继续,没有 STOP

- **摘要:** diag8 的根因(`float_to_fp16` 次正规位移错)在 GVSoC 本地分支 `shvla/soft_hier_release` 上修了两个可分离提交。修完后,rung c 的 QG/KG/VG(150000 点)0 bad,H/G/OUT 全 PASS;rung 1/2/rowops/`siglip_encoder_layer` 的 ROI 与修前逐 ns 一致(只有 add_test +3 ns,与 #10 之后相同);**rung 3、4、5、6 全部从 FAIL 变为 PASS**(之前的 inf / nan / `X0` 1600/1600 bad 都是这个 bug:cancel 到接近 0 的 K-tile 部分和被写成 ±2.0 / 0,后面的 LN / softmax / 残差把它放大)。端到端:动作 max-abs 0.0068(vs lerobot;published 0.032,gate ≤ 0.10),**总 91.139 ms,published 119.1 ms,-23.5 %,gate 的 ±5 % 窗口 MISS**。时间偏快与修这个 bug 无关(rung c 的 marks 与修前几乎一样),见下。
- **提交.** GVSoC `shvla/soft_hier_release`:**A = `cd200ad`** `[shvla] RedMulE model: fp16 subnormal conversion fix`(`float_to_fp16`:`1 - exponent` → `14 - exponent` 并 `& 0x3FF`,保留 `exponent < -10` 下限,另外 exp==0xFF 时 inf 保持 inf / NaN 保持 quiet NaN;`fp16_to_float`:次正规 = `mant * 2^-24`,exp==31 → inf/NaN);**B = `7c9cd2f`** `[shvla] RedMulE model: float_to_fp16 IEEE round-to-nearest-even (SIMULATOR_NOTES #3)`。父仓库:`83cd69d` `[G3.2] infra: GVSoC RedMulE fp16 subnormal fix`、`f56095c` `[G3.2] infra: GVSoC RedMulE fp16 side-note #3 RNE`。`light_redmule.cpp` 是 GVSoC 分支的被跟踪文件,直接改(两处规则不适用)。
- **为什么做 B(RNE):** SIMULATOR_NOTES #3 明确写作者的补丁是 `gvsoc_redmule_fp16_rne.patch`:IEEE round-to-nearest-even(真实 RedMulE 会舍入),发表的精度数字是用 RNE 跑的。**作者的补丁文件本机找不到,B 是按描述重新实现**,不是搬补丁;笔记没有提次正规处理,所以 A 是我们自己的修复(发表精度数字可能带着次正规 bug 跑出来,与之比较时要考虑)。注意 `matmul_fp16` 在 fp32 里累加整个 K-tile、只在 K-tile 末尾转换一次,所以 #3 描述的"768 项点积低 10 %"在我们的模型里不会出现(早先 G3.2-diag 的判断),B 只影响每个 K-tile 的最后一次舍入(truncate → RNE),对 rung 1 的比特级结果没有可见影响(下表)。剩余未处理:`floonoc.cpp` 的 REDADD `float_to_fp16`、`gvsoc_pulp.patch` 里的同一函数、SDK `flex_libfp16.h` 的同一函数(side-note #14)仍然截断 / 不处理次正规,不在本次范围。
- **离线单元测试**(`experiments/smolvla-e2e-baseline/fix-redmule/{unit_test.sh,unit_test_main.c,unit_check.py,emul_fixed.py}`):把两个 C 函数从 `light_redmule.cpp` 原样 `sed` 出来用 gcc 编成主机程序。`fp16_to_float`:65536 个 fp16 位型对照 numpy 全部逐位一致(含次正规、±0、inf、NaN)。`float_to_fp16`:576021 个 case(float32 指数 2^-30 到 2^17,正负,随机尾数 + fp16 网格 + 正规/次正规 tie / tie±1)—— 提交 A 版对"向零舍入"参考 0 mismatch,提交 B 版对 numpy float32→float16(RNE)0 mismatch、0 NaN 失败。边界(B 版):+0→0x0,-0→0x8000,float32 denormal→0,最小次正规 2^-24→0x1,2^-25(tie)→0x0,2^-25+1ulp→0x1,最大次正规 0x387fc000→0x3ff,最小正规→0x400(进位正确),65504→0x7bff,65520→0x7c00(A 版 0x7bff),65536→0x7c00,inf/NaN→0x7c00 / 0x7e00。`emul_fixed.py`(读 `diag8/peek_output.txt`):Q[23,273] kk=3,现模型 0xc000(复现),**A → 0x02fe = 4.5657e-5,A+B → 0x02fe**(同一个值,RNE 对这个元素无影响)。
- **构建:** `unset SOFTHIER_MODEL_DIR; infra/softhier/build_arch.sh .../softhier_mlir_arch.py` 6 s(只重编 `light_redmule` 的 optim/debug 目标),`stamp OK`(GVSoC 7c9cd2f)。每次跑之前 `free -g` available ≥ 9-11 GB(沿用 G3.2-fix10 的读法:available,不是 free 列);`SHVLA_SIM_MEMMAX=8G`,一次一个 sim。
- **Rung c**(`fixrm-rungc-full`,`SHVLA_SIGLIP_H_DUMPS=256 SHVLA_SIGLIP_H_FULL=150000`):**exit 0**,所有 tag PASS:QG/KG/VG、QB/KB/VB、QF/KF/VF(150000 点)**bad=0**,maxerr 0.0039;H 0.0020、G 0.0049、OUT 0.0059(bad=0,4096 点);OIN/OEND/AOR/AOB/HCH 等 o-proj 链全 PASS;PH0-11、OH0-11 PASS。wall 131.5 s,peak 900 MB。(这个 run 的 ROI / marks 被 dump 打印污染,不可比。)**干净的 rung c**(`fixrm-rungc`,无 dump 环境变量):exit 0,11 个 tag PASS,ROI **1526883 ns**(修前 1526876,+7 ns),marks **qkv 0.234 / attention 0.497 / rest 0.793 / layer 1.524 ms**(修前 0.234 / 0.497 / 0.793 / 1.524,逐位相同到 1 µs),DSE §8.3 的 0.324 / 0.831 / 1.040 / 2.195:**-28 / -40 / -24 / -31 %**。wall 51.6 s,peak 864 MB。
- **回归**(`SHVLA_SIM_MEMMAX=8G`,一次一个 sim,peak ≤ 841 MB,oom 0):

| item | 修前 ROI (ns,G3.2-fix10) | 修后 ROI (ns) | shift |
| --- | --- | --- | --- |
| rung 1 (7 shapes): 256^3 / 256x768x192 / 512x768x768 / 256^3,0 / 256x256x512,1,1 / 1024x768x768 all / 1024x3072x768 all | 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / 520114 | 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / **520114** | 0;7/7 PASS(samples bad=0,maxdiff 0),exit 0,peak 395 MB |
| rung 2: add_test | 910396 | 910396 | 0(相对 diag4 +3 ns,与 #10 之后相同) |
| gemm1024_summa / gemm256 / gemm512 / gemm512_linalg / gemm_nonconst | 97682 / 76984 / 85800 / 85800 / 85804 | same | 0;self-check 全 ok |
| mlp / mlp_linalg | 1064054 | 1064054 | 0 |
| siglip_encoder_layer(无 self-check,只验证跑完;`SHVLA_SIM_TIMEOUT=1200`) | 10967081 | 10967081 | 0(published 17851422:-38.6 %,差距不变);wall 343.6 s,peak 841 MB |
| rowops 256x768 cluster 0 / all | (all-cluster PASS) | 4394446 / 366760 ns,LN/SM/GELU/ADD/BIAS/SCALE/T 全 PASS | n/a |

  Rung 2 的 9 个 example 都 exit 0(8 个一起跑 + siglip 单独跑,为了给 siglip 单独的 1200 s timeout)。**所有 ROI 移位为 0**:GEMM 的时间不依赖值,符合预期。
- **Rungs 3-6**(`SHVLA_SIM_MEMMAX=8G`,命令同 G3.2-diag,run 名 `fixrm-rung{3,4,5,6}`):

| rung | result(exit code,逐 tensor) | 误差 vs published | sim 时间 vs published | wall / peak |
| --- | --- | --- | --- | --- |
| 3 `smolvla --npz vision_s256.npz --layers 1` | **PASS exit 0**。EMB max abs 0.0006,L1 0.0072(|ref| max 3.56),OUT 0.0285(|ref| max 27.3),全 bad=0 | README:EMB 0.0067 / L1 0.039 → 我们的更小(published 带着次正规 bug 的可能性,见上) | 1.625 ms/层(attention 0.832 + mlp+proj 0.793)vs 2.29:**-29 %**;ROI 2974000 ns | 57.2 s / 978 MB |
| 4 `smolvla-vlm --npz vlm_c1.npz --layers 1` | **PASS exit 0**。EMB 0.6219(|ref| max 2213,rel 2.8e-4;修前 4.29),L1 0.4819(|ref| 2256),K1 0.0095,V1 0.0005,全 bad=0 | docs/SMOLVLA.md 的误差未逐项核对(量级:相对 max 2e-4 到 8e-4) | 0.730 ms/层(layer 1;mean over 1 layer)vs 0.909 mean(16 层,不严格可比):**-20 %** | 31.7 s / 1263 MB |
| 5 `expert.py flow --npz expert.npz --steps 1` | **PASS exit 0**。X0 max abs 0.0016(|ref| max 2.865,1600 点,bad=0,fp16 floor 0.0020);x_1(50x32 全元素)vs lerobot 0.0016(floor vs lerobot 0.0016)。`expert.npz` 是用 G2 golden 为 `--ref` prepare 的,所以 "ref" = G2 `xt[1]` 同一条参考链 | published:无误差 claim;只对比时间 | 5.951 ms/step(+KV projection 0.117 ms)vs 9.96:**-40 %** | 178.5 s / 1527 MB |
| 6 `smolvla-e2e --npz e2e_c1_t256.npz --e2e-trace` | **PASS exit 0**(`SHVLA_SIM_TIMEOUT=7200`)。VIS max abs 0.0488(|ref| max 23.6),IMG 0.3986(|ref| max 57.7),prefix K/V 每层 K max abs ≤ 0.097 / V ≤ 0.056,x_t 每步 vs lerobot 0.0018 → 0.0068,**动作 max abs 0.0068**(mean 0.00067,|a| max 1.576;fp16 floor 0.0015,floor vs lerobot 0.0070) | published:0.032 | **total 91.139 ms vs 119.1(-23.5 %)**;vision 19.877(published 27.32)/ connector 0.277(0.28)/ prefix 11.188(14.80)/ expert KV 0.117 + 10 steps 59.797(76.70) | wall 4999 s(vision 995 + prefix_a 459 + prefix_b 435 + expert 3111,各 phase 一个 simulator,串行),peak 2787 MB(cgroup,含 page cache;max child RSS 1486 MB),oom 0 |

  **Rung 6 估计(跑之前,按要求记录):** rung 3-5 peak 0.98 / 1.26 / 1.53 GB,各 phase 是独立 simulator 进程(RSS 单进程 ≤ ~1.5 GB),cap 8G 足够;wall 速率 rung 3 约 19 s / sim-ms、rung 5 约 14 s / sim-ms,119 ms 估计 30-40 min(实际 83 min:`--e2e-trace` 的 hand-over dump + 每个 phase 的 preload 约 190 MB 比单 rung 慢,expert 10 步 3111 s = 约 33 s / sim-ms)。没有超出内存或 timeout 的风险,所以执行了。
- **G3 gate**(`gate.py g3 $SHVLA_DATA/runs/published/fixrm-rung6`,新写的 `experiments/smolvla-e2e-baseline/gate.py`,key 与真实 `summary.json` 一致:`segments_ms.total`、`acc.actions.max_abs`;`g4a` 是未实现的 stub,属于 G4a):`total simulated time : 91.139 ms (published 119.1 ms, window 113.14..125.05 ms, -23.5 %)  MISS`;`action max-abs error : 0.0068 (limit 0.1, published 0.032)  PASS`;`G3 gate: MISS`,exit 1。
- **解读(没有追):** 准确度 PASS 且比 published 更好,时间在所有 rung 都偏快 20-40 %(rung c marks 不随本次修复变化,所以差距不是 RedMulE 转换)。按规则"timing off but accuracy OK:记录并继续"。偏快的来源是别的东西:候选(没验证,留给 owner):作者的 `models_fast` / `models_fix` 里还有别的 timing 相关补丁(side-note #1 `fp_ss_pending` 未装,#2 zfh,#6/#8),或 published 的 119.1 在 DRAM(非 ideal HBM)/ 旧 commit 上测的(DSE §8.3 的 ideal HBM 是作者原话,所以不太可能);rung 1 的 GEMM ROI 与 published 一致说明 RedMulE / iDMA 时间是对的,偏快的是 scalar-FP row-op 段(qkv 的 LN1/bias、attention 的 softmax、rest 的 GELU/LN2):#1(`snitch_fp_ss` 不等 pending 访存)正好会让这些段变快。**下一步建议(需 owner 批):** 只补 #1(`gvsoc_snitch_fp_ss_pending.patch` 的 `exec.is_stalled()` 守卫),重跑 rung c(52 s)看 marks 是否向 DSE 的 0.324 / 0.831 / 1.040 / 2.195 移动。
- **发现的副作用:** G3.2-diag 里 rung 4 EMB 的 max abs 4.29(|ref| max 2213)是同一个 bug(现在 0.62);所有早先"通过"的 SmolVLA 阶段(64 点采样)都受影响,本次 rung 3-6 的结果取代它们。
- Simulators:全部已停止(`pgrep gvsoc_launcher` / `gapy` 为空)。Logs:`$SHVLA_DATA/runs/published/fixrm-{rungc-full,rungc,rung1,rung2,rowops,rowops-all,siglipenc,rung3,rung4,rung5,rung6}/run.log`(`$SHVLA_DATA` = `~/.cache/softhier-vla`)。

### G3.3-fix1 proposal — GVSoC side-note #1(`snitch_fp_ss.cpp` 不等 `IO_REQ_PENDING` 的 FP 访存;已知 side-note 项,按失败规则:先写 proposal,再在本地 GVSoC 分支 `shvla/soft_hier_release` 上应用)
- **作者做法(`softhier_vla_summary_2026-10.md` §6、`deeploy_softhier_progress.md` §3、SIMULATOR_NOTES #1):** offload 到 FP 子系统的 `flw/fsw` 访问 HBM / L3 `.rodata` / 远端 TCDM 时返回 `IO_REQ_PENDING`,但 `Iss::handle_event`(`snitch_fp_ss.cpp`)按固定 latency 就把指令标成完成。补丁 `gvsoc_snitch_fp_ss_pending.patch`:在 `Iss::handle_event` 开头加 `if (exec.is_stalled()) { event->enqueue(1); return; }`,即等 `Lsu::data_response` 到了再完成。补丁文件本机找不到(与 #3/#10 一样),**按描述重新实现**。
- **从代码验证机制(`core/models/cpu/iss/src/lsu.cpp`):** `data_req_aligned` 遇 PENDING 走 `:202-203` `exec.insn_stall()`(`stalled_inc`),`Lsu::data_response`(`:118-123`)里 `exec.stalled_dec()`,然后 `stall_callback`(load resume 写 freg)。fp_ss 的 `Iss` 带自己的 `lsu`(`cores/snitch_fp_ss/class.hpp:79`),所以 `exec.is_stalled()` 在 `handle_notif`(同步执行 `resource_handler`)之后、`handle_event` 之前确实会为 PENDING 访存置位。守卫必须是 `handle_event` 的**第一条语句**,在 `acc_req_ready = true` 之前:否则子系统在访存未完成时就报 ready,下一条 offload 会覆盖 `acc_req` 并复用在途 `io_req`(§3 的失败模式 2,NoC 报 `No entry found for burst`)。
- **方向:** 守卫只会 `enqueue(1)` 重试,所以完成时刻只可能不变或推迟:**时间单调变慢**。只有返回 PENDING 的访问受影响(HBM / L3 / 远端 TCDM);同步返回 OK 的本地 TCDM `flw/fsw` 不受影响。行操作 kernel 多数在 iDMA 之后读本地 TCDM,所以**预期影响可能很小**(准确度已全 PASS 也是弱证据:PENDING 路径不多)。若 rung c marks 几乎不动 → 按规则在回归后 STOP,列剩余候选。
- **文件状态:** `third_party/gvsoc/core/models/cpu/iss/src/snitch/snitch_fp_ss.cpp` 是嵌套 core 里 tracked 且干净的文件,**不在**现有 `soft_hier/gvsoc_core.patch` 里(那里只有 `snitch/{decode,iss,snitch}.cpp`)。所以按 two-place 规则:(1) 改嵌套 core 工作树;(2) 在 patch 里 `snitch/snitch.cpp` 之后**新增**一个 file section。验证:对 `git archive HEAD`(bda268a4)的嵌套 core 打整个 patch,无 fuzz / offset / fail,四个 snitch 文件与工作树 `cmp` 一致(已做)。重编 `build_arch.sh`(`unset SOFTHIER_MODEL_DIR`),`stamp.py check`。
- **步骤(随进展更新):** [x] 1 proposal;[x] 2 apply(patch + tree);[x] build + stamp;[x] 提交(GVSoC fd1124a + infra 8fdce36);[x] 3 rung c;[x] 4 回归;[x] 5 决策 = STOP(见 G3.3-fix1 entry)。

### G3.3-fix1 — side-note #1(fp_ss pending 守卫)已应用并验证:**所有 ROI 逐 ns 不变**,rung c marks 没动;**STOP**(按"marks 几乎没动"分支),没有跑 rung 3-6 / gate
- **提交:** GVSoC `shvla/soft_hier_release` **`fd1124a`** `[shvla] snitch fp_ss: Iss::handle_event waits while exec.is_stalled() ...`(只改 `soft_hier/gvsoc_core.patch`,和 070936b 一样);父仓库 **`8fdce36`** `[G3.3] infra: GVSoC side-note #1 (snitch fp_ss pending access)`。Two-place:嵌套 core 工作树 `models/cpu/iss/src/snitch/snitch_fp_ss.cpp`(`handle_event` 第一条语句 `if (_this->exec.is_stalled()) { _this->event->enqueue(1); return; }`,在 `acc_req_ready = true` 之前)+ `gvsoc_core.patch` 里 `snitch/snitch.cpp` 之后新增的 file section(patch 现 25 个 file section)。对 `git archive HEAD`(bda268a4)的嵌套 core 打整个 patch:无 fuzz / offset / fail;`snitch_fp_ss.cpp / snitch.cpp / iss.cpp / decode.cpp` 与工作树 `cmp` 一致。
- **构建:** `unset SOFTHIER_MODEL_DIR; build_arch.sh` 10 s;`install/models{,/debug}/gen_isa_snitch_fp_ss_rv32imfdva_cpp_61161152.so` mtime 更新(当前 arch 用的就是这个 hash;`..._131782587.so` 没被重编;证据:本 arch 的 build log 和 `g33-rungc` 的配置里只出现 `gen_isa_snitch_fp_ss_rv32imfdva_cpp_61161152`,`install/models_fast` 与 `install/models_fix` 都不存在(`softhier_mlir/sim/gvsoc.py:189-190` 会优先搜它们,但没有被遮蔽),所以加载的就是刚重编的库)。提交之后重新 `stamp.py write`(stamp 记录 GVSoC commit 和 diff hash),`stamp.py check` OK。
- **Rung c**(`g33-rungc`,`siglip-mlir --seq 256 --cluster all`):exit 0,11 个 tag 全 PASS(与修前相同),wall 47 s,peak 865 MB。ROI **1526883 ns(与 fixrm-rungc 逐 ns 相同)**;marks qkv0 483430 / attn0 980399 / layer0 1773273 ns,段时间 **0.234 / 0.497 / 0.793 ms,layer 1.524 ms**,和修前相同;DSE §8.3 0.324 / 0.831 / 1.040 / 2.195(-28 / -40 / -24 / -31 %)**差距不变**。
- **回归(全部 exit 0,peak ≤ 842 MB,oom 0):**

| item | 修前 ROI (ns) | #1 之后 ROI (ns) | shift |
| --- | --- | --- | --- |
| rung 1 (7 shapes) 256^3 / 256x768x192 / 512x768x768 / 256^3,0 / 256x256x512,1,1 / 1024x768x768 all / 1024x3072x768 all | 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / 520114 | 13102 / 31854 / 131713 / 13074 / 19536 / 119415 / **520114** | 0,7/7 PASS |
| rung 2 add_test / gemm1024_summa / gemm256 / gemm512 / gemm512_linalg / gemm_nonconst / mlp / mlp_linalg | 910396 / 97682 / 76984 / 85800 / 85800 / 85804 / 1064054 / 1064054 | same | 0,8/8 PASS(self-check ok) |
| siglip_encoder_layer(published 17851422) | 10967081 | 10967081 | 0(-38.6 %,差距不变) |
| rowops 256x768 all(LN/SM/GELU/ADD/BIAS/SCALE/T) | 83975 / 81138 / 85551 / 21504 / 20882 / 15773 / 57937(合计 366760) | same | 0,全 PASS |

- **解读:** 方向在代码上是对的(守卫只会推迟完成),但**这个工作负载里它从来不触发**:SoftHier 库的 scalar-FP 行操作(LN / softmax / GELU / ...)读写的是 iDMA 搬进来的**本地 TCDM**,返回 IO_REQ_OK(同步),不走 PENDING 路径;全部 ROI 逐 ns 不变正是这个意思,与 SIMULATOR_NOTES #1 的症状(HBM / `.rodata` / 远端 TCDM 的 `flw/fsw`)也一致。所以"未装 #1 导致 FP 段偏乐观"的假设**被否定**(由"库确认已加载 + ROI 逐 ns 不变"推断;"本地 TCDM 同步返回 OK" 是与之一致的解释,没有逐条访问地址验证)。全盘 `find` 没有找到作者的 `gvsoc_snitch_fp_ss_pending.patch`。#1 保留(正确性修复,无副作用),但它不是 -23.5 % 的来源。
- **剩余差距:** rung 1 的纯 GEMM 与 published 一致;非 GEMM 段仍偏快:qkv 0.234 vs 0.324(-28 %)、attention 0.497 vs 0.831(-40 %)、rest 0.793 vs 1.040(-24 %)、layer 1.524 vs 2.195(-31 %);siglip_encoder_layer -38.6 %;rung 6 总 -23.5 %。
- **其它未应用 / 私有的候选(未验证,不再做修复):** (a) **`models_fast`**(`gvsoc_iss_host_speed.patch`):笔记 §"Host speed of the Snitch ISS" 明说是 behaviour-neutral(只改 host 侧的拷贝 / trace / `strstr` 缓存:`iss_insn_t` 按值拷贝次数、`sprintf` trace 格式化、label 谓词缓存),所以按文字不改周期;`gvsoc_iss_host_speed.patch` 不在本机(`softhier-mlir/tools/iss_fast/` 只有 `classify.py gen_makefile.py sample.sh`,全盘无 `*host_speed*`),**没法读补丁,也没法实测**,只能引用笔记;它也没有 #1 的守卫之外的时序语义。(b) #2 zfh(库不用标量 fp16,不影响)、#6 / #8(库用软件 workaround 回避,属正确性)、#4 / #9(崩溃 / hang,不是时序)。(c) **#7 instr_router / 32 B prefetcher / 无 RVC**:笔记说一个 `instr_router`(8 B/cycle)被三个核共用、非 RVC;这是模型 / arch 配置的性质,若 published 的 119.1 ms / DSE 用的 arch(例如指令 cache 或 router 宽度,`softhier_mlir_arch.py`)与我们不同,行操作段会差很多 —— 对 -24…-40 % 且只出现在非 GEMM 段的形态最吻合,**建议 owner 对照作者的 arch 文件**(未验证)。(d) FPU 流水线 / latency 配置(`get_latency`,`flexfloat`)与作者私有分支可能不同;(e) published 的 marks 是否包含我们没算的 host 同步 / 屏障开销。以上都需要 owner 决定,按规则不再动 GVSoC。
- Simulators:全部已停止(`pgrep gvsoc_launcher` / `gapy` 为空)。Logs:`$SHVLA_DATA/runs/published/g33-{rungc,rung1,rung2,rowops,siglipenc}/run.log`。

### G3.3-gap — 时间差距定位:**verdict (a)**,环境参数 = `instr_mem` 的 `width_log2=-1`(我们的 GVSoC)vs 默认 `2`(deeploy / 作者的 GVSoC);用 scratch 生成器 shim 一次性复现 published 的 rowops 逐 op 数值(<1 %)、rung c 的 DSE §8.3 marks、rung 3 的 2.29 ms/层;**未改任何受跟踪文件**,只给出 proposal
- **范围:** 有界定位,6 个短 sim(3 个 ≤ 12 s 的 rowops、rung 1、rung c、rung 3,后三者 ≤ 51 s),`SHVLA_SIM_MEMMAX=8G`,一次一个,`free` available 11 GB,没有 perf / gdb / strace。没有改 GVSoC、arch、softhier-mlir。
- **Q1 逐 op 对比。** published = `docs/SIMULATOR_NOTES.md` "Row-wise / elementwise fp16 ops" 表(`run.py rowops --rows 256 --cols 768`,commit **583b24a**,2026-10-08 14:05)。`runtime/sh_rowops.inc.c` 自 583b24a 之后**没有再改**(`git log -- runtime/sh_rowops.inc.c`:583b24a 是最后一个);从 DSE §8.3 的测量点 8565c57 到 b11cd5f,`git diff --stat -- runtime/` 对 `sh_simd.inc.c` / `sh_rowops.inc.c` / `sh_attention.inc.c` / `sh_l1.inc.c` 为空,`sh_ops.h`(+186)和 `sh_rt.inc.c`(+41)只有**纯插入**(新 kernel 声明、per-core stack),没有删改。所以 rowops 这张表与 b11cd5f 的 kernel 代码相同 —— **编译器 / codegen 进步不能解释 rowops 的差距**。

| op | published all-16 (µs) | 我们(原状) | % | 我们 + instr_mem=2 | % | published cluster 0 (µs) | 我们 cl.0 (原状) | % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN | 149 | 83.98 | -43.6 | **148.7** | -0.2 | 2020 | 1054 | -47.8 |
| SM | 145 | 81.14 | -44.0 | **145.1** | +0.1 | 2140 | 1068 | -50.1 |
| GELU | 123 | 85.55 | -30.4 | **122.2** | -0.7 | 1810 | 1239 | -31.5 |
| ADD | 22 | 21.50 | -2.3 | 22.1 | +0.4 | 145 | 142 | -2.1 |
| BIAS | 27 | 20.88 | -22.7 | 27.1 | +0.5 | 158 | 148 | -6.4 |
| SCALE | 17 | 15.77 | -7.2 | 16.8 | -1.0 | 128 | 125 | -2.7 |
| T(iDMA 转置) | 25 | 57.94 | **+132** | 58.2 | +133 | 231 | 619 | **+168** |
| 合计 | 508 | 366.8 | -27.8 | 541.2(含 T) | +6.5 | 6630 | 4394 | -33.7 |

  形状:**差距集中在取指受限的 op(LN / SM / GELU:-30…-50 %),inner loop 不取指的 ADD / SCALE / BIAS 几乎持平**(SIMULATOR_NOTES #7 的签名:"cluster 是 instruction-fetch bound,~1 instr/cycle")。所以不是均匀的全局 core / FPU 参数,而是**指令取指路径**。(cluster-0 的 per-op 数据取自 `fixrm-rowops/run.log`;all-cluster 的第 4 列来自本次 gap run;published 值 SIMULATOR_NOTES 只给了 ms 三位,故 µs 值含舍入。)
  **T(转置)与取指无关**(`2` 之后仍 58 µs):published 25 µs,我们 2.3× 慢。这是**独立的第二个偏差**:我们的 b11cd5f kernel 是 iDMA 2-D gather(`SH_TRANSPOSE_CORES=0`),所以疑点在 iDMA 的 2-D 路径(我们的 GVSoC 另有 `0e4d719` / `2910e62` 两个 iDMA 修复,以及 ours 多了 `TransposeEngine` 单元)。**没有追**(超出本次预算;不影响 rung c / rung 3:rung c 的 T 只有 kT 转置在 attention 里,marks 吻合),留作待办。
- **Q1 其他已发表数值与 commit。**
  - DSE §8.3 的行 "per-head attention, cost-model tiles"(0.324 / 0.831 / 1.040 / 2.195):`tests/gvsoc/run.py siglip-mlir` 默认就是 per-head + `tiles=model`(`run.py:330` `fused=False`),所以 rung c 与它是同一配置。测于 2026-10-08 19:xx,commit 43d72ec 之后 / **合并 origin/main(8565c57)之前**,DSE.md 末段原话:"stock Snitch ISS, on-device test data"。之后的提交对行操作 kernel 没有 perf 改动(sh_rowops 未动;`sh_gemm` 的 GEMM 部分我们与 published 一致)。siglip frontend 之后只有 8565c57 合并(untimed dumps)与 mark 同一定义 —— 见 Q4。
  - SMOLVLA.md:22706bf 的 "2.29 ms/层"(per-head,256 tokens);rung 6 published 119.1 ms(commit 66679cd)。
  - 没有找到 attention / softmax 单独的、我们能直接复现的新增表(`run.py attention` 的 fused 表 0.556 ms 属于 fused 路径,本次没跑,不需要)。
- **Q2 arch / 模型 diff(所有不同的参数)。**
  1. **`arch/softhier_mlir_arch.py` 与 `Arch()` 默认**:该文件就是 `Arch().to_py()` 生成的(文件头),按构造相同。与 deeploy 的 `soft_hier/flex_cluster/flex_cluster_arch.py` 比较:仅 `hbm_node_addr_space`(deeploy 0x00200000,我们 0x04000000 / 树内默认 0x02000000)、`hbm_node_aliase`(我们有)、`hbm_chan_placement`(deeploy 树内 [4,0,0,0],我们 [4,0,0,4])—— 都是 HBM 相关,G3.2-diag5/6 已排除 HBM 模型;`num_cluster/redmule/idma_outstand/noc_*/instruction_mem_size` 一致。
  2. **生成器 `soft_hier/flex_cluster/cluster_unit.py`(=`pulp/pulp/chips/flex_cluster/cluster_unit.py` 同一行,安装到 `install/generators/pulp/chips/flex_cluster/`)vs deeploy:**
     - **`:209` `instr_mem = Memory(..., width_log2=-1, ...)` vs deeploy 默认 `width_log2=2`(`:203` 无参数)** ← **决定性参数**。Memory 模型 `soft_hier/flex_cluster/memory.cpp:241-247`:`if (width_bits != 0) duration = MAX(size >> width_bits, 1)` 并串行化 `next_packet_start`。`size >> -1` 在 C++ 里是未定义行为;从 run A/B/C 推断,这个 x86 构建里它等效于 duration = 1 周期 / 包(无取指带宽限制);`2` → 32 B 取指行 `32>>2 = 8` 周期 / 包(4 B/cycle)。
     - `:212` `instr_router bandwidth=8*arch.nb_core`(生成的 config 里是 **40**)vs deeploy `bandwidth=8`(SIMULATOR_NOTES #7 的 "8 B/cycle")。单独设成 8 只回来一部分(见 Q3 run A);**设了 instr_mem=2 之后 router 带宽已不是限制**(run B ≈ run C)。
     - `:240` `fpu_sequencer latency=0`(ours 才有,Sequencer);`:206`/`:369` `INST_PREHEAT_DONE` / `FETCH_START`(ours)vs `FETCHEN`(deeploy):启动连线不同,影响的是 ROI 之前的起跑时刻(marks / ROI 的 start 在 barrier 之后),本次没有单独测。
     - 其它只是新功能:`TransposeEngine`、`tech_node`、`spatz_vlsu_bw`、`data_dumpper_ctrl_size 20→64`、`HBM_PRELOAD_DONE`。
  3. **FPU 流水线 / latency(`get_latency`、flexfloat):** `snitch_fp_ss.cpp` 与 deeploy 只差我们的 side-note #1 守卫(`:249-258`);`get_latency`(`:331-392`)逐行相同;`exec_inorder.cpp` 只多两行 trace。`flexfloat.c` / `rv32v` / `spatz` 在 diff 列表里但与 Snitch FP 时序无关(spatz 不用,`spatz_attaced_core_list=[]`)。**没有 FPU latency 差异。** RVC:cores 不执行 compressed 指令(SIMULATOR_NOTES #7 / TTT #12),双方一致。
  4. `gvsoc_core.patch` / `gvsoc_pulp.patch` 的差异:我们多了 #10 / #1 / iDMA / RedMulE 修复与 power 的 CMake 开关;时序相关的已逐一排除(G3.2-fix10、G3.2-diag5、G3.3-fix1)。
  5. 取 `-1` 的来源:两棵树都只经 `a8b40d7` "SoftHier Squashed"(2026-02-23)进入,`git blame` 无法定年代;`-1` 是我们 release 分支比 deeploy 分支新的上游改动,不是我们的任何补丁。
- **Q3 测试(shim,不碰受跟踪文件)。** 方法:`run_sim` 把 `--target-dir=$SOFTHIER_HOME/install/generators` 写死,.so 的 `SOFTHIER_MODEL_DIR` 换不了生成器参数,所以在 scratch(`/tmp/shvla_shim/sitecustomize.py`,通过 `PYTHONPATH` 注入;用完已删除)里用 `importlib` MetaPathFinder 在 `interco.router` / `pulp.chips.flex_cluster.memory` 加载后包装 `__init__`:`name=='instr_router'` 时覆盖 `bandwidth`,`name=='instr_mem'` 时覆盖 `width_log2`。命中记录写进 `hits.log`(run B 起;每次命中写 2 行,64 行 = 32 次命中 = 16 个 cluster 的 router + mem),证明 shim 生效;run A 当时没有 hits.log,它的生效证据是 ROI 变化(LN 84→92.5 µs);run F(rung 1)的 hits.log 112 行。全部经 `run.sh published sim`(stamp OK,cap 8G,peak ≤ 864 MB,oom 0)。

| run | 参数 | 结果 |
| --- | --- | --- |
| A `gap-rowops-all-r8` | router 8,instr_mem -1 | LN 92.5 / SM 91.6 / GELU 88.6 / ADD 21.6 / BIAS 22.5 / SCALE 16.0 / T 58.0 µs(合计 391)—— 只回来一部分 |
| B `gap-rowops-all-r8m2` | router 8 + instr_mem 2 | **LN 148.8 / SM 145.0 / GELU 122.9 / ADD 22.1 / BIAS 27.3 / SCALE 16.8 µs**,T 58.2 —— 与 published 149 / 145 / 123 / 22 / 27 / 17 逐项在 1 % 内 |
| C `gap-rowops-all-m2` | **只** instr_mem 2(router 仍 40) | LN 148.7 / SM 145.1 / GELU 122.2 / ADD 22.1 / BIAS 27.1 / SCALE 16.8 / T 58.2 —— 与 B 相同 ⇒ **instr_mem 是主因,router 宽度不重要** |
| D `gap-rungc-r8m2`(siglip-mlir --seq 256 --cluster all,router 8 + mem 2) | exit 0,11 tag PASS,ROI 2241046 ns,wall 47 s | marks **qkv 0.324 / attention 0.867 / rest 1.044 / layer 2.235 ms**;DSE §8.3 0.324 / 0.831 / 1.040 / 2.195:**0 / +4.3 / +0.4 / +1.8 %**(原状 0.234 / 0.497 / 0.793 / 1.524) |
| E `gap-rung3-m2`(smolvla --layers 1,只 mem 2) | exit 0,EMB / L1 / OUT PASS(误差与 fix-redmule 一致) | **2.265 ms/层**(attention 1.224 + mlp+proj 1.040);published 2.29:**-1.1 %**(原状 1.625,-29 %) |

  | F `gap-rung1-m2`(gemm 默认 7 shapes,只 mem 2) | exit 0,7/7 PASS,peak 395 MB,hits.log 112 行 | **GEMM ROI 并非不变**:256^3 13102→**14720**,256x768x192 31854→35301,512x768x768 131713→**138693**,256^3,0 13074→14647,256x256x512 19536→21065,1024x768x768 all 119415→119870,1024x3072x768 all 520114→520479 ns |

  **Run F 的解读:** 单 cluster 的小 GEMM 慢了 5-12 %(GEMM 控制代码也取指),16 cluster 大 GEMM 几乎不变(+0.4 / +0.07 %)。对照 published(LOG G3.1-ihbm 表 / DSE.md:130-132):256^3 14.7-16.7 µs ⇒ 14.72 在其下限,原状 13.10 在范围外;512x768x768 138.9 µs ⇒ **138.69(-0.15 %)**,原状 131.7(-5.2 %);1024x768x768 120.1 ⇒ 119.87 / 原状 119.4;1024x3072x768 520.5 ⇒ 520.48 / 原状 520.11。即**mem=2 下 rung 1 与 published 吻合得更好**(此前 "rung 1 逐 ns 匹配" 只对 16 cluster 的两个 shape 成立,单 cluster 的 shape 原状偏快 5-11 %),不削弱 verdict (a),是第二组独立证据。

  残余:attention 段 +4.3 %(0.867 vs 0.831)。它是 per-head attention(GEMM + softmax + `kT` 转置),与 T 的偏差(上表)同向(转置我们慢 2.3×)是个合理的候选;没有在本次验证,量级小。
  跑 rung 4 / 5 / 6 / gate 要 > 2 min(sim 2 min 限制),本次没有跑;按 rowops / rung c / rung 3 的一致性,预期 expert / prefix 的偏快也同样消失,但**这是预期,不是测量**。同理,rung 1 / 2 的 ROI 在改参数后会整体改变(run F),proposal 落地后要重做 rung 1/2 的回归基线。
- **Q4 mark 定义。** `softhier.mark` 的位置 / 含义在 b11cd5f 与 DSE §8.3 测量时相同(frontend `siglip.py` 自 1aa6534 / 8565c57 后没有再动 mark 逻辑);DSE §8 说的 "older ROI 含 ~2 ms dumps" 只影响 ROI,不影响 marks(我们本来就用 marks 段)。run D 的 marks 与 published 吻合(qkv / rest < 0.5 %)说明 mark 定义不是差距来源。run.py 里 mark 行的 start 到 qkv0 的 0.324 ms 与 published 的 0.324 完全一致。
- **Verdict (a):特定环境参数。** 我们的 GVSoC release 分支(a8b40d7 squash 起,`soft_hier/flex_cluster/cluster_unit.py:209` 与 `pulp/pulp/chips/flex_cluster/cluster_unit.py:209`)把 `instr_mem` 建成 `width_log2=-1`(= 无取指带宽限制),作者的 deeploy 分支用默认 `2`(32 B 取指行 8 周期)。作者的 published 数字(DSE §8.3、SIMULATOR_NOTES rowops 表、rung 6 的 119.1 ms)是在取指受限的内存上测的。把 instr_mem 设回 2:rowops 6 个 op 与 published 偏差 < 1 %,rung c marks 偏差 0 / +4 / +0.4 / +1.8 %,rung 3 2.265 vs 2.29 ms。
  **不是**(b)(rowops kernel 自 583b24a 没改,且改回 2 就吻合)。
- **Proposal(未应用,需要 owner 批准;按"已知 side-note 之外的 GVSoC 改动先 proposal"规则):** 在**两处**(`third_party/gvsoc/soft_hier/flex_cluster/cluster_unit.py:209` 和 `third_party/gvsoc/pulp/pulp/chips/flex_cluster/cluster_unit.py:209`,GVSoC 与 pulp 子模块各一个提交,同 G3.1-fix 的 two-place 规则;已 grep:`soft_hier/gvsoc_pulp.patch` 里没有 `width_log2` / `instr_mem`,那个补丁文件不用改)去掉 `width_log2=-1`(回到默认 2)。`instr_router` 的 `8*arch.nb_core` 可以不动(run C 表明不重要);要完全对齐 deeploy 也可以改回 8(run B ≈ C)。改了之后:重新 `build_arch.sh`(`stamp.py write`),重跑 rung c(预期 0.324 / 0.867 / 1.044)、rung 1/2、rowops、然后 rung 3-6 + `gate.py g3`(预期总时间落进 119.1 ms ±5 %)。**这会让 G3 gate 的时间项从 MISS 变成(预期)PASS**,也会改变 G4 的所有周期数,所以应该先改再进 G4。
- **T(iDMA 转置)偏差:** 与取指无关,2.3× 慢;未解决,不影响 marks;记为待办(见上)。
- 状态:**unresolved 部分只有** attention +4.3 % 与 T。Simulators:全部已停止(`pgrep gvsoc_launcher` / `gapy` 为空);shim 已删除,`stamp.py check` OK。Logs:`$SHVLA_DATA/runs/published/gap-{rowops-all-r8,rowops-all-r8m2,rowops-all-m2,rungc-r8m2,rung3-m2}/run.log`。

### G3.3-fix-instr — 应用 owner 批准的 `instr_mem` 修复(去掉 `width_log2=-1`,回到默认 2);rung 1/2/c 回归通过,基线重设
- **改动.** 只有一行:`third_party/gvsoc/soft_hier/flex_cluster/cluster_unit.py:209`,`instr_mem = Memory(..., atomics=True, tech_node=...)`(去掉 `width_log2=-1`)。GVSoC 提交 **f99d5a2**(分支 `shvla/soft_hier_release`),infra 提交 **7d14f70**(GVSoC 指针 + 设计文档 side-note 表新增一行,无编号,因为 #6 已被占用;措辞修正见 infra 后续提交)。
  - **patch vs tree:** `soft_hier/flex_cluster/cluster_unit.py` 是 GVSoC 仓库里**直接受跟踪**的文件,所以只有一个 GVSoC 提交。`pulp/pulp/chips/flex_cluster/cluster_unit.py` 在 pulp 子模块里是**未跟踪**的(`?? pulp/chips/flex_cluster/`),由 GVSoC Makefile 的 `cp -rf soft_hier/flex_cluster pulp/pulp/chips/flex_cluster` 从 `soft_hier` 复制得来,所以**没有 pulp 提交**;我把受跟踪文件 cp 过去,三份(`soft_hier`、`pulp/pulp/chips`、`install/generators/pulp/chips`)`cmp` 完全相同。`gvsoc_pulp.patch` / `gvsoc_core.patch` 不含 `instr_mem` / `width_log2`(G3.3-gap 已 grep),所以不需要改。
  - `build_arch.sh`(`MAKEFLAGS=-j8`,6 s,增量)+ `stamp.py check` OK;installed generator 的 :209 已无 `width_log2`。
- **回归**(`SHVLA_SIM_MEMMAX=8G`,一次一个;"旧"= G3.3-gap run F 的值 / g33 / fixrm 的基线,GVSoC fd1124a;"新"= GVSoC f99d5a2,run 名 `fixi-rung1/rung2/rungc`):

| rung | 旧 ROI (ns) | 新 ROI (ns) | Δ |
| --- | --- | --- | --- |
| rung 1: 256^3 | 13102 | 14720 | +12.3 % |
| 256x768x192 | 31854 | 35301 | +10.8 % |
| 512x768x768 | 131713 | 138693 | +5.3 % |
| 256^3,0 | 13074 | 14647 | +12.0 % |
| 256x256x512,1,1 | 19536 | 21065 | +7.8 % |
| 1024x768x768 all | 119415 | 119870 | +0.38 % |
| 1024x3072x768 all | 520114 | 520479 | +0.07 % |
| rung 2: add_test | 910396 | 1306472 | +43.5 % |
| gemm1024_summa | 97682 | 99353 | +1.7 % |
| gemm256 | 76984 | 77471 | +0.6 % |
| gemm512 / gemm512_linalg | 85800 | 90998 | +6.1 % |
| gemm_nonconst | 85804 | 90998 | +6.1 % |
| mlp / mlp_linalg | 1064054 | 1460038 | +37.2 % |
| siglip_encoder_layer(无 self-check) | 10967081 | 18002379 | +64.1 %(published 17851422:**+0.85 %**,旧 -38.6 %) |

  - rung 1:7/7 PASS(bad=0,maxdiff 0),exit 0,peak 400 MB;与 run F(shim)逐 ns 相同 ⇒ shim 与真正改源码等价。单 cluster 形状 +5…12 %,16-cluster 形状 +0.07 / +0.38 %,符合预期。
  - rung 2:**8 个有 self-check 的 example 全 PASS + siglip_encoder_layer 跑完**(无 self-check,只验证终止),exit 0,peak 842 MB。add_test(+43 %)/ mlp(+37 %)的大幅变慢是 scalar 循环取指受限的典型表现,不是失败。
  - **rung c**(`siglip-mlir --seq 256 --cluster all`):exit 0,11 个 tag 全 PASS(X/LN1/Q/K/KT/P0/V/O/H/G/OUT,误差与之前相同,OUT 0.0039),ROI 2209245 ns,wall 71.9 s,peak 864 MB。**marks:qkv 0.324 / attention 0.840 / rest 1.040 ms,layer 2.204 ms**(DSE §8.3:0.324 / 0.831 / 1.040 / 2.195 ⇒ 0 / +1.1 / 0 / +0.4 %;gap run D(router 8 + mem 2)0.324 / 0.867 / 1.044 / 2.235)。只改 mem(router 仍为 40)比 run D 更接近 published,attention 的残余偏差从 +4.3 % 降到 +1.1 %;rung c 的 mark 与 router 宽度的关系这次才第一次直接测到。
  - 老基线 → 新基线:以上"新"列为新的回归基线(`fixi-*` run 名)。rungs 3-5 按 owner 选择(Apply, rung 6 only)没有重跑。
- Simulators:全部已停止。

### G3.7-gate — rung 6 (`fixi-rung6`) 与 `gate.py g3`:**PASS**
- 命令:`SHVLA_SIM_MEMMAX=8G SHVLA_SIM_TIMEOUT=14400 run.sh published sim fixi-rung6 -- tests/gvsoc/run.py smolvla-e2e --npz $D/e2e_c1_t256.npz --e2e-trace`(新 run 名,没有复用 fixrm-rung6 的 phase)。exit 0,4 个 phase 都 PASS,stamp OK。
- 结果(ms,1 GHz,ideal HBM):

| phase | published | 之前(fixrm,mem -1) | 现在(fixi) | vs published |
| --- | --- | --- | --- | --- |
| vision | 27.32 | 19.877 | 27.640 | +1.2 % |
| connector | 0.28 | 0.277 | 0.280 | 0 |
| prefix(16 层) | 14.80 | 11.188 | 15.039 | +1.6 % |
| expert(KV 0.137 + 10 步 × 7.762) | 76.70 | 59.914 | 77.756 | +1.4 % |
| **total** | **119.1** | 91.139 | **120.716** | **+1.4 %** |

- 准确度:**actions max abs 0.0068 vs lerobot**(mean 0.00067,|a| max 1.576;published 0.032;expert fp16 floor 0.0015,floor vs lerobot 0.0070),x_t 每步 0.0018 → 0.0068,与修前逐位一样(时间模型改动不影响数值)。prefix K/V 每层 K max abs ≤ 0.081 / V ≤ 0.056。
- wall:4914 s(= 82 min:vision 1290 + prefix_a 567 + prefix_b 582 + expert 2475),peak 2614 MB(cgroup,含 page cache;max child RSS 1486 MB),oom 0。比 fixrm 的 4999 s 还略快(同一 host 上的波动)。
- **Gate**:`gate.py g3 $SHVLA_DATA/runs/published/fixi-rung6`:`total simulated time : 120.716 ms (window 113.14..125.05, +1.4 %) PASS`;`action max-abs 0.0068 (limit 0.1) PASS`;**`G3 gate: PASS`**。G3 完成。
- 说明:时间差距的根因是 `instr_mem` 的 `width_log2=-1`(G3.3-gap verdict a);修复后 5 个层级(rowops、rung c、rung 3、rung 1、rung 6)都落在 published 的 ±5 % 内。残余(未追):iDMA 转置 T 仍比 published 慢 2.3×(G3.3-gap),attention 段 +1.1 %。G4 的所有周期数从此基于新的取指带宽模型。
- Simulators:全部已停止。Logs:`$SHVLA_DATA/runs/published/fixi-{rung1,rung2,rungc,rung6}/run.log`。
