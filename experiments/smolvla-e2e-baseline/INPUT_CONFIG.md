# Input configurations: published vs LIBERO

Design: [docs/designs/smolvla-softhier-baseline.md](../../docs/designs/smolvla-softhier-baseline.md) (Component 31, "LIBERO input configuration"). Values below are read from `configs/*.env`, the golden references and LOG.md (G2.4, G6b.1, G4a).

| item | published (G3, `lerobot/smolvla_base`, rev `5e8d12a6`) | LIBERO (G4, `lerobot/smolvla_libero`, rev `a24f1f38`) | source |
| --- | --- | --- | --- |
| robot / env | none: softhier-mlir's synthetic seeded observation | Franka Panda in LIBERO (`libero_spatial` task 0, seed 0, first frame after reset) | golden `info_obs`: `synthetic:1:256:0` / `libero:libero_spatial:0:0` |
| cameras | 1 | 2: agentview (`image`) and wrist (`image2`), renamed `camera1`, `camera2` | `config.json`, LeRobot `LiberoEnv` |
| camera size | 256 x 256 policy input | env renders 360 x 360, resized with padding to 512 x 512 (square, no padding) | `resize_imgs_with_padding = (512, 512)` |
| SigLIP tokens per camera | 256 | 1024 (32 x 32 patches of 16 px) | arithmetic |
| image tokens per camera (after 4 x 4 pixel shuffle) | 16 | 64 | arithmetic |
| language | 48 tokens, right-padded | 48 tokens, 20 valid (`pick up the black bowl between the plate and the ramekin and place it on the plate`) | golden `lang_mask`, `info_prompt` |
| state | synthetic | 8-D (eef pos 3, axis-angle 3, gripper qpos 2), zero-padded to 32 | OQ2, `lerobot/sim/integrations/libero.py` |
| action | 32 (padded) | 7-D, padded to 32 | `config.json` |
| prefix tokens | 16 + 48 + 1 = 65 | 2 x 64 + 48 + 1 = 177 (149 valid) | `smolvla_e2e.info`, meta `[50,32,10,177,149]` |
| S_pad | 128 | 256 | multiple of 128 |
| expert | chunk 50, 10 flow steps | chunk 50, 10 flow steps, `n_action_steps = 50` | `config.json` |
| control rate | not applicable | 20 Hz in the env (`LiberoEnv.fps = 20`; the checkpoint's `train_config.json` says 30), so one 50-action chunk covers 2.5 s of simulated time | OQ3 |

Files under `$SHVLA_DATA/smolvla_libero/` (`run.sh libero prepare`): `vision_s1024.npz` (SigLIP weights and a test image), `vlm_c1.npz`, `vlm_c2.npz` (prefix at 177 tokens), `expert.npz`, `e2e_c2_t1024.npz` (the chain input with the G2 references). Golden: `$SHVLA_DATA/golden/libero.npz`.

## Caveat: `vlm_c2.npz` has 5 valid prompt tokens, the golden has 20

`prepare-vlm` encodes its own default prompt (5 valid tokens of 48), while the golden and `e2e_c2_t1024.npz` use the task prompt (20 valid). Checked on the host (G4a): the prefix program emitted by `smolvla.emit_vlm` for 8 layers is **identical** (MLIR text, preload keys, shapes and data) with `vlm_c2.npz`'s mask and with the golden's 20-token mask. The G4a.2 timing input is therefore unaffected. G4b reads `e2e_c2_t1024.npz`, which carries the golden mask.

## Status at the end of G5

G4b (the full LIBERO chain, `e2e_c2_t1024.npz`) was **not run** (owner decision after R1: the vision phase is projected at 3.72 h against the 2 h limit). The LIBERO latency in [RESULTS.md](RESULTS.md) is the G4a estimate of 479.1 ms, composed from partial runs. The values in the table above (tokens, S_pad, prefix 177 with 149 valid, 50-step chunk at 20 Hz) are the ones the G4a runs used and match RESULTS. The `vlm_c2.npz` caveat applies only to G4a.2, where it was shown to have no effect on the timed program.
