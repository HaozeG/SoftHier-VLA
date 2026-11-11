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
| G3.1–G3.6 | — | ladder rungs 1–6; fork commits (33) when a fix is needed, owner first | each rung's pass column | G3.1 flex_alloc fix done (infra 22d549c); iDMA fix applied (GVSoC 0e4d719) and verified; rung 1 PASS (7/7, timing +46 % vs published on the 16-cluster shapes); ideal-HBM switch added (GVSoC cdb56a2, infra 1d195b0): rung 1 now matches published (1024x3072x768 520114 ns vs 520.5 us); rung 2: 9/9 complete after the second iDMA fix (GVSoC 2910e62, G3.1-diag4; `siglip_encoder_layer` has no self-check, so that only means it terminates); **G3.2-diag: rung c (siglip-mlir seq 256) FAIL, rung 3 FAIL, rung 4 FAIL, rung 5 FAIL; all four fail after the first scalar-FP row op and run 24-41 % too fast; diagnosis = the private Snitch/RedMulE patches (side-note #3, #10, ...) are absent; rung 6 and `gate.py` not run (STOP, owner decision)** | experiment (run.sh + LOG), infra 18ad903, infra 2088d59 (GVSoC d42c382) |
| G3.7 | 34 | `gate.py g3` on the rung-6 run | gate outcome recorded (PASS or MISS with values) | blocked (rungs 3-5 fail accuracy, see G3.2-diag) | |
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
