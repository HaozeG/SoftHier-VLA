# Patches on softhier-mlir

Base: softhier-mlir `main` at [b11cd5f](https://github.com/runwangdl/softhier-mlir/tree/b11cd5fa4f3e717374d1e9ba14442ff302d1f970) (the submodule `../softhier-mlir`, frozen until G5).

`run.sh` applies them in file-name order after resetting the submodule's tracked files (`run.sh apply`; `fetch`, `prepare` and `sim` also apply them). A patch that does not apply makes `run.sh` exit 2. Each file starts with a header (what, why, evidence) that `git apply` ignores (it skips everything before the first `diff --git` line). To see the result: `git -C softhier-mlir diff` after `run.sh apply`.

Design: [docs/designs/smolvla-softhier-baseline.md](../../../docs/designs/smolvla-softhier-baseline.md), "softhier-mlir changes for the LIBERO configuration". New patches found under design rule R3 are added here after the owner is told.

| patch | files | what | why | evidence | upstream candidate |
| --- | --- | --- | --- | --- | --- |
| 0001 `shvla-smolvla-dir` | `frontend/smolvla_e2e.py`, `frontend/smolvla.py`, `frontend/smolvla_expert.py`, `frontend/smolvla_expert_ref.py`, `tests/gvsoc/run.py`, `tests/gvsoc/expert.py` | `/app/models/smolvla_base` becomes `os.environ.get("SHVLA_SMOLVLA_DIR", "/app/models/smolvla_base")` at every site on a run path (12 lines) | the path does not exist here; one variable per run selects the checkpoint folder | applies; `py_compile`; `run.sh published prepare` and `libero prepare` write into the folder `SHVLA_SMOLVLA_DIR` names (G2.2); the `--ckpt` / `--ref` defaults themselves are only exercised by the host check of 0002 | yes |
| 0002 `shvla-npz-functions` | `frontend/smolvla_e2e.py`, `tests/gvsoc/run.py` | `MODELS`, `VISION_NPZ`, `VLM_NPZ`, `EXPERT_NPZ` become `models_dir()`, `vision_npz()`, `vlm_npz()`, `expert_npz()`; default arguments become `None` and resolve inside; `run.py` calls the functions | a default bound at import would keep the first folder when one process prepares both checkpoints | grep finds no remaining use of the constants. Host check in the experiment env (G2.2): with `SHVLA_SMOLVLA_DIR=smolvla_base`, `emit_vision(e2e)` and `expert_data(e2e, {ref KV})` run with no npz arguments (200 preload arrays; `p_kp0` shape (65, 320)); switching the variable to `smolvla_libero` in the same process makes `vision_npz()`, `vlm_npz()`, `expert_npz()` follow. The `ref_kv_k_<L>` keys 0005 reads exist in a real `e2e_c1_t256.npz` | yes |
| 0003 `shvla-expert-ref-lerobot-0.7` | `frontend/smolvla_expert_ref.py` | port to LeRobot 0.7.0 / transformers 5.22: `DynamicCache` for the prefix KV (dump and rebuild), no `fill_kv_cache`, `vla_utils` imports, transformers 5 `vision_position_ids` | the crosscheck (G2.3) runs the reference in the same environment as `golden_ref.py` | G2.2: ran in `infra/lerobot/.venv`, cams 1, 256 px; wrote `xc_ref.npz` (132 arrays), same key set as `golden_ref.py`'s `published.npz`, `vis_pos_ids` identical; the numeric crosscheck is G2.3 | yes. Also report the position-id finding: with transformers 5 SmolVLM computes `idx * (1/n)` clamped to `1 - 1e-6`; the older `idx / n * (1 - 1e-6)` (as in softhier-mlir) puts every row after the first one bucket lower. No other copy of the formula in softhier-mlir (grep `bucketize`, `1 - 1e-6`) |
| 0004 `shvla-timeout-flag` | `tests/gvsoc/run.py` | `--timeout <s>` (default 48 h), passed to `run_smolvla_e2e`, `run_vision_cost`, `run_prefix_cost`, `run_expert_cost` | the 2 h per-process limit through softhier-mlir's own `run_sim(timeout=...)` | none yet (G3 rung 6) | yes |
| 0005 `shvla-expert-only-ref-kv` | `tests/gvsoc/run.py` | `--expert-only ref:<e2e npz>`: KV from the golden reference stored in the e2e npz, checked before the `e2e_` branch | G4a.3 runs before G4b, and the existing source needs a finished chain run | none yet (G4a.3) | yes |
| 0006 `shvla-siglip-h-chain-dumps` | `frontend/siglip.py`, `testing/siglip_ref.py` | opt-in diagnostic dumps for `siglip-mlir` (env `SHVLA_SIGLIP_H_DUMPS=<n>`, plus `_REGEMM`, `_FULL`): inline o-proj chain OIN / WOI / BOI / AOR / AOB / HCH with a barrier after each, per-head OH / PH, QG..VB / QF..OF, matching reference keys | the end-of-program 64-sample dumps cannot localise the first bad step of rung c (G3.2) | emitted module byte-identical to unpatched for layers 1, 2 and `--fused` when the variable is unset (G3.2-diag7); rung c with dumps: LOG G3.2-diag7 | yes (diagnosis aid, default unchanged) |

## Sites on `/app/models` that are not patched

Not on any `run.sh` path (the design lists only the sites above); left as they are:
*   `tests/gvsoc/flow_tables.py:52` and `tests/gvsoc/ttt.py:290`: `--npz` default `/app/models/smolvla_base/expert.npz`;
*   `tools/e2e/connector_fp16_accumulation.py`, `tools/e2e/error_attribution.py`: hard-coded loads;
*   `frontend/wm_rssm.py`, `tests/gvsoc/wm.py`: a different model folder (`/app/models/wm_rssm`);
*   docstrings, comments and `docs/`.

## What 0003 needed beyond the design's list

Nothing: `build_policy` (meta-device build, config fields) and `materialise` (non-persistent buffer rebuild) work unchanged under LeRobot 0.7.0 / transformers 5.22. Only the four API points of the design table changed, plus the import locations of `make_att_2d_masks` and `create_sinusoidal_pos_embedding` (`lerobot.policies.common.vla_utils`).
