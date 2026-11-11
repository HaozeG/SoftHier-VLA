# GVSoC patch series

Ten patches on top of upstream GVSoC `soft_hier_release` at `d806a84c5785237547f504c2e34a4d9f230cb437`
([permalink](https://github.com/gvsoc/gvsoc/tree/d806a84c5785237547f504c2e34a4d9f230cb437)). They replace a fork
(decision "GVSoC fixes shipped as patch series, no fork", owner, 2026-11-11, design D2). The submodule
`third_party/gvsoc` is pinned at the upstream base; `infra/softhier/setup.sh` applies the series.
Findings F1..F19 are in the design's findings table (`docs/designs/smolvla-softhier-baseline.md`); the evidence
for each is in `experiments/smolvla-e2e-baseline/LOG.md` and RESULTS.md.

Files: the ten `NNNN-*.patch` files (`git format-patch --no-signature --zero-commit`, the commit messages are the
rationale), `series` (apply order), `expected` (base commit, expected tree, SHA-256 of the series).

| patch | finding | what | why | upstream-worthy |
| --- | --- | --- | --- | --- |
| `0001-G3.1-sdk-add-flex_alloc_api.h-...` (1.3 KB) | F17 | adds `soft_hier/flex_cluster_sdk/runtime/include/flex_alloc_api.h` (copied from `bowwang-dev/softhier-deeploy`) | softhier-mlir's runtime includes it; the release branch lacks it | yes |
| `0002-G3.1-power-capture-only-...` (4.3 KB) | F18 | power capture only with `GVSOC_POWER_CAPTURE=1` (via `gvsoc_core.patch`) | capture at the compiled interval ran `make ice_power` in long simulations; harmless, not the cause of the memory growth | no |
| `0003-shvla-iDMA-backend-re-arm-FSM-...` (3.5 KB) | F13 | `IDmaBe::update()` also enqueues `fsm_event` (via `gvsoc_pulp.patch`) | `gemm` with K tile 256 never terminated; backend FSM was not re-armed | yes |
| `0004-soft_hier-SOFTHIER_IDEAL_HBM-1-...` (3.9 KB) | F5 (SN #5) | `SOFTHIER_IDEAL_HBM=1` replaces the DRAMSys HBM channels by gvsoc's ideal `Memory` in `flex_cluster.py` | softhier-mlir's published numbers assume ideal HBM; stock tree always instantiates DRAMSys (+46 % on 16-cluster GEMMs) | optional (feature) |
| `0005-shvla-iDMA-backend-can_accept_transfer-...` (5.0 KB) | F14 | `can_accept_transfer()` also requires an empty regulation queue (via `gvsoc_pulp.patch`) | lost transfers (283 of 1024 rows) and `hbm_fill` hangs | yes |
| `0006-shvla-ISS-ISA-tables-nseq-...` (13.4 KB) | F8 (SN #10) | `nseq` tag on scalar FP instructions with an integer operand (via `gvsoc_core.patch`) | softmax, LayerNorm, GELU depended on binary layout; timing +1 %, accuracy unchanged | no effect here; follows the owner's notes |
| `0007-shvla-RedMulE-model-fp16-subnormal-...` (3.3 KB) | F15 | `light_redmule.cpp`: fp16 subnormal conversion in `float_to_fp16` and decode in `fp16_to_float` | sparse corrupt GEMM elements, inf/NaN downstream; rungs c, 3, 4, 5 failed | yes |
| `0008-shvla-RedMulE-model-float_to_fp16-IEEE-...` (3.5 KB) | F3 (SN #3) | `float_to_fp16` rounds to nearest even | truncation | yes, with 0007 |
| `0009-shvla-snitch-fp_ss-Iss-handle_event-...` (2.5 KB) | F1 (SN #1) | `Iss::handle_event` waits while `exec.is_stalled()` (via `gvsoc_core.patch`) | stale `flw`/`fsw` to HBM or remote TCDM; no timing effect on this workload | harmless; follows the owner's notes |
| `0010-G3.3-gvsoc-instr_mem-width_log2-...` (1.2 KB) | F16 | drops `width_log2=-1` of `instr_mem` in `cluster_unit.py` (default 2) | undefined shift, no fetch bandwidth limit: whole chain -23.5 % vs published | yes |

Patches 0002, 0003, 0005, 0006, 0009 edit `soft_hier/gvsoc_core.patch` / `gvsoc_pulp.patch`, ordinary files of the gvsoc
repository. GVSoC's own `make softhier_preparation` applies them to the nested `core` and `pulp` repositories, so
the series must be in place before the preparation (`setup.sh` does this). The nested working trees are never part
of the series.

## Expected result

`expected` holds `BASE` (the upstream commit above), `TREE` = `def18fe6856947bab3009fa551cc43282c5ddbf1` (the tree
of the old local dev commit `f99d5a2`, `git rev-parse f99d5a2^{tree}`) and `SERIES_SHA256`
(`python3 infra/softhier/stamp.py series-hash`: SHA-256 over `series` and each listed patch in order, each as
name, NUL, bytes, NUL). Commit ids after `git am` differ from the dev branch's (committer identity and date are fixed
by `apply_gvsoc_patches.sh`, so they are the same on every clone, but not equal to `f99d5a2`); the trees, and the
tree after every single patch, are identical. Verified on a fresh upstream checkout (LOG G5.2).

## Applying

`bash infra/softhier/apply_gvsoc_patches.sh [gvsoc-dir]` (called by `setup.sh`):

*   `HEAD^{tree}` equals `TREE`: prints "already applied", exits 0. This is the second run, and also the developer's
    tree on the local branch `shvla/soft_hier_release`.
*   `HEAD` is `BASE` and tracked files are clean: `git am` of the series, then asserts the tree.
*   anything else: refuses with the reason (nothing is touched).

`stamp.py check` fails unless the series files still match `SERIES_SHA256`, `HEAD^{tree}` is `TREE` and `BASE` is an
ancestor of `HEAD`. So a tree that is not base + series cannot pass the stamp check, whatever its commit ids.

## Developing a fix (re-exporting the series)

1.  In `third_party/gvsoc` work on a local branch on top of the applied series (the dev branch
    `shvla/soft_hier_release`, or any branch created from the current `HEAD`). Commit with a message that names the
    finding, the evidence and the run id: the message is the rationale shipped in the patch.
2.  Rebuild (`build_arch.sh`) and test as usual. The stamp check will fail because the tree is no longer `TREE`; that
    is expected until the series is re-exported.
3.  Re-export from the GVSoC repository, base to your tip, replacing the old files:

    ```
    rm infra/softhier/gvsoc-patches/0*.patch
    git -C third_party/gvsoc format-patch --no-signature --zero-commit -o <ABSOLUTE>/infra/softhier/gvsoc-patches d806a84c5785237547f504c2e34a4d9f230cb437..HEAD
    ls infra/softhier/gvsoc-patches/0*.patch > infra/softhier/gvsoc-patches/series
    ```

    Then set `TREE` in `expected` to `git -C third_party/gvsoc rev-parse 'HEAD^{tree}'` and `SERIES_SHA256` to
    `python3 infra/softhier/stamp.py series-hash`, update the table above, and commit patches, `series`, `expected`.
4.  Do not stage `third_party/gvsoc` in the superproject: its gitlink stays at `BASE`. After `setup.sh`, and on the dev
    tree, `git status` shows the submodule as modified (new commits); that is expected. A `git add -u` or
    `git commit -a` would re-pin it to a local-only commit. If that happens, undo with
    `git update-index --cacheinfo 160000,d806a84c5785237547f504c2e34a4d9f230cb437,third_party/gvsoc`.
