# From checkpoint to SoftHier units: the SmolVLA compiler path

How PR1 takes SmolVLA from a checkpoint to kernels running on SoftHier's units, following one action-expert
layer and one SigLIP layer. This page is a delta on softhier-mlir's own design document
(`docs/DESIGN.md` at commit `b11cd5f`, the checkout PR1 pinned as `experiments/smolvla-e2e-baseline/softhier-mlir`):
it records what the SmolVLA path actually does, which differs from the general pipeline that document
describes. Code references are `file:line` in that checkout unless marked `~/softhier@52859c3` (the kernel repo).
Numbers per operator: [smolvla-op-graph.md](smolvla-op-graph.md).

```text
 checkpoint (safetensors)                         lerobot reference run
        |  prepare: per-model npz in library layout       | e2e_ref npz (inputs, KV, x_t)
        v                                                 v
 Python frontends (siglip.py, smolvla.py, smolvla_expert.py, smolvla_e2e.py)
        |  write softhier-dialect IR text directly (xDSL dialect defs; no linalg)
        v
 4 programs: V (vision+connector), Pa, Pb (prefix 1-8, 9-16), X (expert)
        |  backend/emit_c.py: one softhier op -> one sh_* runtime call
        v
 C program + runtime (runtime/sh_*.inc.c), linked per program
        |  weights + inputs -> preload ELF into simulated HBM (sim/preload.py)
        v
 GVSoC SoftHier: per cluster, RedMulE (matrix products, operands by iDMA),
 3 Snitch cores (everything else); Spatz disabled in PR1's arch
```

## 1. Checkpoint to arrays

* **Prepare (host, torch + safetensors).** `frontend/smolvla_e2e.py:83` (`prepare`) reads the checkpoint and a
  lerobot reference run and writes the chain's inputs and host tables: im2col'd camera images, position rows,
  language rows, the state embedding, RoPE tables, noise. Weights are read at emit time from per-model npz files
  in the library's layout; for the expert, `frontend/smolvla_expert.py:91` (`to_library_layout`) transposes the
  matrices, concatenates gate and up into one `[720, 4096]` matrix, and folds the time half of the time-MLP input
  into a per-step bias table (`:27-28`, `:115`).
* **Host-side operators.** The language embedding lookup and scaling, the state projection and the time-half
  product run on the host in PR1, not on SoftHier.
* **Four programs, not one.** The fp16 weights of the four parts (162 + 22.5 + 300 + 189.5 MiB = 674 MiB) exceed
  the 512 MiB of populated HBM, so the chain runs as four programs; each dumps its HBM state and the next one
  preloads it (`frontend/smolvla_e2e.py:15-25`).
* **Into simulated HBM.** `sim/preload.py:41` (`make_preload_elf`) packs weights and inputs into an ELF that the
  simulator loads into HBM before the program starts.

## 2. Frontend: Python writes softhier IR directly

The SmolVLA frontends do not go through `linalg`. They print softhier-dialect operations (defined with xDSL in
`dialects/softhier.py`) with explicit HBM buffers, shapes, tile sizes and target cluster. One self-attention
layer of the action expert, from `frontend/smolvla_expert.py:423` (`_layer_ops`), as it appears in PR1's
kept `expert.mlir` (types shortened):

```text
softhier.rmsnorm %h, %g1_s -> %xn                 : 50x720
softhier.gemm %xn, %wqkv_s into %qkv {tile_m=50, tile_n=64, tile_k=720}   : 50x720 . 720x1600
softhier.rope %q, %rq_self -> %q                  : 50x960
softhier.rope %k, %rq_self -> %k                  : 50x320
softhier.cross_attention %q, %kp_s, %vp_s own %k, %v mask %tok -> %o
        {heads=15, kv_heads=5}                    : keys 65 prefix + 50 own
softhier.gemm %o, %wo_s into %ao {tile_n=48, tile_k=960}                  : 50x960 . 960x720
softhier.add %h, %ao -> %h
softhier.rmsnorm %h, %g2_s -> %xn
softhier.gemm %xn, %wgu_s into %gu {tile_n=256, tile_k=240}               : 50x720 . 720x4096
softhier.silu_mul %ga, %up -> %ga                 : 50x2048 (gate | up halves)
softhier.gemm ... mlp_down ; softhier.add (residual)
```

A cross-attention layer replaces the QKV product with a query-only product and reads the prefix keys and values
that were projected once per call before the step loop. Loops over layers and steps stay as `scf.for`.

The SigLIP layer comes from `frontend/siglip.py:75` (`emit`). PR1 used its per-head path: LayerNorm, Q/K/V
products with bias adds, a K transpose, then per head a QK^T product, a row softmax and a P.V product, then the
output product, residual, LayerNorm, MLP up, GELU, MLP down, residual. A fused `softhier.attention` option exists
(`fused_attention=True`) and is used by the prefix, not by vision.

## 3. Backend: one op, one runtime call

`backend/emit_c.py:706` (`emit_c`) walks the module and prints one C call per op:

| softhier op | runtime call | emit_c.py |
| --- | --- | --- |
| `gemm` | `sh_gemm` / `sh_gemm_mesh` / `sh_gemm_xmcast_ex` (by tile policy) | 334-342 |
| `layernorm` | `sh_layernorm` | 360 |
| `softmax` | `sh_softmax_rows` / `sh_softmax_masked` | 367-369 |
| `rmsnorm` | `sh_rmsnorm` | 373 |
| `rope` | `sh_rope` | 378 |
| `silu_mul` | `sh_silu_mul` | 385 |
| `gelu` | `sh_gelu` | 401 |
| `attention` (prefix, GQA + mask) | `sh_attention_gqa` | 416-430 |
| `cross_attention` (expert) | `sh_x_attention` | 437-455 |
| `axpy` (Euler update) | `sh_x_axpy` | 432-435 |

## 4. Runtime: which unit does the work

* **Matrix products.** `runtime/sh_gemm.inc.c:53` (`sh_gemm`) splits the output into tiles over clusters; per tile
  the data-mover core drives iDMA to bring X and W panels into TCDM and RedMulE computes. Tiles parallelise over
  output tiles only (no split-K), so small products use few clusters (PR1 PROFILE: 1-10 of 16 in vision/prefix).
* **Attention.** `sh_attention_gqa` (`runtime/sh_llm.inc.c:325`) and `sh_x_attention` (`runtime/sh_expert.inc.c:264`)
  run QK^T and P.V on RedMulE per head and the softmax on the cores. The expert runtime pads the key length to a
  multiple of 32 (`runtime/sh_expert.inc.c:83`).
* **Row and elementwise operators.** `runtime/sh_rowops.inc.c:1-12`: the data-mover core stages row blocks into
  TCDM (double-buffered), all three Snitch cores compute in fp32 through fp16 conversions, and the result is
  streamed back. LayerNorm, softmax, GELU (`sh_rowops.inc.c:383-389`), RMSNorm, RoPE and SiLU x up
  (`sh_llm.inc.c:121-169`) all run this way.
* **No Spatz.** The runtime contains no vector (`vsetvli`, `vfexp`) code; PR1's arch file disables Spatz because
  the compiler does not target it.

## 5. Operator to unit and hand-written kernel

The kernel repo's hand-written kernels (`~/softhier@52859c3`, `implementation/sw/`) target the stage-1 arch with
Spatz enabled. GELU and LayerNorm were added by this project on branch `gelu-layernorm-kernels`
(`HaozeG/gvsoc@580ac85`), with outputs checked on the device against a replay golden; the changes and the
simulator defects found on the way are logged in `wiki://memories/patches/softhier-kernel-repo.md`. Per-call times
of the row-operator kernels at libero-2cam shapes are in [smolvla-op-graph.md](smolvla-op-graph.md), section
"Row operators on hand-written kernels".

| Operator | Unit in PR1's path | Hand-written kernel | Evidence in the kernel repo |
| --- | --- | --- | --- |
| matrix products (projections, MLP) | RedMulE, operands by iDMA | `SummaGEMM` | RedMulE config and trigger, `SummaGEMM/include/SummaGEMM.h:443-444` |
| attention QK^T and P.V | RedMulE | `FlatAttention` | RedMulE per block, `FlatAttention/include/FlatAttention.h:197,316`; GQA by folding query heads into the query length, `FlatAttentionUtil.h:189` |
| softmax | Snitch cores | inside `FlatAttention` only | Spatz `vfredmax` and `vfexp`, `FlatAttentionUtil.h:474,494` |
| attention masks (prefix, expert) | Snitch cores (masked softmax) | none found | no mask or causal handling in `FlatAttention/include/` (grep) |
| RMSNorm | Snitch cores | `RMSNorm` | Spatz `vfredusum`, `RMSNorm/include/RMSNorm.h:159` |
| LayerNorm | Snitch cores | `LayerNorm` (added, `580ac85`) | Spatz, fp32 accumulation, `LayerNorm/include/LayerNorm.h` on branch `gelu-layernorm-kernels` |
| GELU | Snitch cores | `Activation` (GELU, tanh form; added, `580ac85`) | upstream `Activation` covers Sigmoid, ReLU, SiLU (`Activation/include/Activation.h:128-142`); GELU added on branch `gelu-layernorm-kernels` |
| SiLU x up (MLP gate) | Snitch cores | `Activation` (SiLU + gate) | `ACTI_ALGO_SILU` then `vector_lib_gate` multiply, `Activation.h:138-149, 253-269` |
| SiLU (time MLP) | Snitch cores | `Activation` (SiLU) | `Activation.h:225-250` |
| RoPE, bias add, residual add, scale, Euler update | Snitch cores | none | |
| transpose, pixel shuffle | Snitch cores and iDMA | none | |
| language embedding, state projection, time half of the time MLP | host | none | PR1 prepare step |

## 6. What is not built

* **Spatz lowering.** `softhier.vexp` is declared (`dialects/softhier.py:157`, "Spatz vector exponential") and the
  softhier-mlir README lists Spatz `vfexp` for softmax (`README.md:81`), but `emit_c.py` emits nothing for it and
  no runtime call uses Spatz. Every row operator runs on the Snitch cores. The hand-written kernels are
  standalone programs; none is called from softhier-mlir's runtime, so PR1's end-to-end run does not use them.
* **The linalg path.** `transforms/linalg_to_softhier.py` (lowers `linalg.matmul` only), `pipeline_gemm.py` and
  `distribute_summa.py` are the general pipeline of `docs/DESIGN.md` section 4; the SmolVLA frontends bypass them.
* **Kernels for the remaining operators.** RoPE, the elementwise adds, transpose and pixel shuffle have no
  hand-written kernel; FlatAttention has no masking, which the prefix and the expert need.
* **One program.** The chain is four programs because the weights exceed populated HBM; a single program needs
  more HBM in the arch (the 256-cluster proposal assumes 1 GiB, [hw-config.md](hw-config.md)).
