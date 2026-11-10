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
| G1.4 | 10, 12 | `stamp.py`, `build_arch.sh` | `stamp.py check` passes after a build with `implementation/`'s arch | planned | |
| G1.5 | 11 | `run_kernel_tests.sh`; read the kernel make targets and pass strings; run | `gemm`, `norm`, `acti`, `attn` PASS | planned | |
| G1.6 | 7 | `infra/softhier/README.md` with measured times | — | planned | |
| G6a.1 | 14, 15 | LeRobot uv project and env; settle OQ1 | `smolvla_libero` loads at its revision | done | infra b3382eb |
| G6a.2 | 17 | `libero_eval.py`; 1 task × 1 episode smoke run | `summary.json` written | done | infra 3d1607c |
| G2.1 | 16, 18 | `golden_ref.py` synthetic and from-npz modes; tests (a)–(c) | published ref written; tests pass | done | infra 6aa2bd2 |
| G2.2 | 23, 24, 26, 27, 28 (0001–0005) | softhier-mlir submodule at `b11cd5f` (experiment commit), experiment uv project, configs, `run.sh`, patches | `run.sh published fetch` and `prepare` finish; patches apply | planned | |
| G2.3 | — | crosscheck: `run.sh apply`; then in the LeRobot environment `python experiments/smolvla-e2e-baseline/softhier-mlir/softhier_mlir/frontend/smolvla_expert_ref.py --ckpt $SHVLA_DATA/smolvla_base --cams 1 --img-size 256 --out $SHVLA_DATA/golden/xc_ref.npz` (by file path, not `-m`, so the package `__init__` and its xdsl import are not loaded; the script itself imports only stdlib, numpy, torch and LeRobot), then `golden_ref.py --obs from-npz:$SHVLA_DATA/golden/xc_ref.npz --device cpu --crosscheck $SHVLA_DATA/golden/xc_ref.npz` | every key within tolerance (Verification Plan) | planned | |
| G2.4 | 16, 18 | libero mode; test (d), (e) | LIBERO ref written; tests pass | planned | |
| G2.5 | 13 | `infra/lerobot/README.md` | — | planned | |
| G6b.1 | — | `libero_spatial` 10 × 10; settle OQ2–OQ4 | success rate with interval in LOG.md | planned | |
| G3.0 | 25 | arch file; `build_arch.sh` with it | `stamp.py check` passes | planned | |
| G3.1–G3.6 | — | ladder rungs 1–6; fork commits (33) when a fix is needed, owner first | each rung's pass column | planned | |
| G3.7 | 34 | `gate.py g3` on the rung-6 run | gate outcome recorded (PASS or MISS with values) | planned | |
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
