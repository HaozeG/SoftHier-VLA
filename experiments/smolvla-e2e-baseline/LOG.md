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
| G1.3 | 8, 9 | GVSoC uv project, `setup.sh`; run it | `softhier_preparation` finished; toolchain present | blocked | infra 7251468 |
| G1.4 | 10, 12 | `stamp.py`, `build_arch.sh` | `stamp.py check` passes after a build with `implementation/`'s arch | planned | |
| G1.5 | 11 | `run_kernel_tests.sh`; read the kernel make targets and pass strings; run | `gemm`, `norm`, `acti`, `attn` PASS | planned | |
| G1.6 | 7 | `infra/softhier/README.md` with measured times | — | planned | |
| G6a.1 | 14, 15 | LeRobot uv project and env; settle OQ1 | `smolvla_libero` loads at its revision | done | infra b3382eb |
| G6a.2 | 17 | `libero_eval.py`; 1 task × 1 episode smoke run | `summary.json` written | done | infra 3d1607c |
| G2.1 | 16, 18 | `golden_ref.py` synthetic and from-npz modes; tests (a)–(c) | published ref written; tests pass | planned | |
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

### G1.3 — GVSoC uv project and `setup.sh` (blocked)

- Time: 2026-11-10。
- What and why: 写 `infra/softhier/pyproject.toml`(Python 3.12,非 package)和 `setup.sh`(prerequisite check、submodule init、`uv sync`、`make softhier_preparation`、permalinks),然后运行。**状态 blocked:** apt 前置包缺失,需要 owner 批准 `sudo apt-get install`。我没有试任何绕过的办法,`softhier_preparation` 没有跑。
- Commands: `uv sync`(在 `infra/softhier`,成功,生成 `uv.lock` 和 `.venv`);`bash infra/softhier/setup.sh` → exit 1。
- Result: gcc/g++ 11.4.0(≥ 11.2)、cmake 3.22.1(≥ 3.18.1)、uv、wget、bzip2、xz 都满足。缺失的 apt 包,owner 需要运行:

  ```
  sudo apt-get install -y doxygen libsdl2-dev gtkwave autoconf automake texinfo libtool libsdl2-ttf-dev
  ```

  装完后再运行 `bash infra/softhier/setup.sh`(它会跳过已经完成的 step)。
- Notes / deviations from the design:
  - `pyproject.toml` 合并了 `requirements.txt`、`gapy/requirements.txt`、`core/requirements.txt`,取较严的下界(`pyelftools>=0.29`,`PyYAML>=6.0.1`)。去掉了 `argparse` 和 `dataclasses`:Python 3.12 的标准库里已有,PyPI 上的 `dataclasses` 只支持 Python < 3.7。`prettytable` 和 `pyelftools` 本来就在文件里。
  - apt 检查在 README 的列表之外加了 `wget`、`bzip2`、`xz-utils`(Makefile 的 toolchain 下载和解压要用);这三个已安装。用 `dpkg-query` 的 Status 字段判断,不用 `dpkg -s` 的 exit code。
  - Makefile 的 `SYSTEMC_INSTALL_DIR := $(PWD)/...` 读 shell 的 `PWD`,所以脚本用 `cd third_party/gvsoc && make`,不用 `make -C`;不传 `-j`(prerequisite 是串行的,内部是裸 `make`)。
  - "已完成"的判断比设计更严:检查 `systemc_install/lib64/libsystemc.so`、`third_party/DRAMSys/libDRAMSys_Simulator.so` 和 toolchain 里的 `riscv*-gcc`,因为 Makefile 先 `mkdir -p` 再下载,下载失败后只检查文件夹会误判。路径都相对 GVSoC checkout。
  - Makefile 的 `read -t 30`:只在 DRAMSys 预编译库的自检失败时才出现。stdin 为 `/dev/null` 时 `read` 立即失败,`user_input` 为空,而 Makefile 只在输入 `n` 时退出,所以默认是 yes:重新编译 DRAMSys(Makefile 估计约 40 分钟)。脚本在 preparation 之后 grep `Test libaray succeeded/failed` 来记录走了哪个分支。
  - DRAMSys 重编译分支的 exit status 是最后一条命令 `make clean` 的,所以脚本在结束后检查 `.so` 是否存在。
  - 没有跑 preparation,所以没有 wall time;disk use 仍是 40 MB(`du -sh third_party/gvsoc`)。shellcheck 没有安装。
- Files changed: `infra/softhier/pyproject.toml`,`infra/softhier/uv.lock`,`infra/softhier/setup.sh`。
- Commit: infra 7251468(`[G1.3] infra: GVSoC uv project and setup.sh`)。
- Paths under `$SHVLA_DATA`: none(`softhier/setup/preparation.log` 还没有创建)。
- What to inspect: `setup.sh` 的 prerequisite 部分和上面的 apt 命令。

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
