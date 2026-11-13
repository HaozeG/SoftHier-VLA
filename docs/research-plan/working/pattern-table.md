> Working material (idea-forming, not a decision). Copied from the owner's untracked `workload-modeling/pattern-table.md`; RQs do not cite it as a starting point. See [README](README.md).

# VLA workload patterns: a starting table for workload modeling

Working document, not part of the wiki. It splits every model into a few
**compute patterns** (P) and a few **composition axes** (C), then records each
model as "which patterns, how many times, in what order and at what rate".
A workload model can then be written once per pattern and reused for every
combination.

Evidence labels follow the wiki glossary (`estimate, measured, self-reported`,
wiki-design decision 0004): **measured** = a reported run. It is split by who
ran it: **measured, 3rd-party** (Jung et al., VLA-HWBench) or **measured,
authors** (the model's own paper). **inference** = our own derivation here, which
no source states (the glossary's "estimate" is reserved for a figure a source
computes from specs).

Sources not ingested in the wiki, read in session on 2026-11-06:
- † full text: Holo-M 2609.35709, SP-HLM 2610.30044.
- ‡ alphaXiv AI-generated report only, recheck against the paper before use:
  HAF 2608.16837, OpenHLM 2606.22174, ω-0 2608.06375.
- SONIC and TANGO are cited in `wiki://knowledge/dual-system-vla.md` (checked by
  text search) but also not ingested.

Platform: "Bound" values are on the RTX 4090 (Jung et al.) unless stated; the
wiki notes that platform balance moves the ridge (on Jetson Thor every stage is
more bandwidth-sensitive).

---

## Table A. Compute patterns

"Bound" is at batch size 1. Where no source says, it is marked inference.

| ID  | Pattern                                    | Shape that defines it                                                                 | Bound (batch 1)                                                                                                        | Examples                                                       |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| P0  | CPU preprocessing                          | image resize, tokenize, normalise; host-to-device copy                                 | CPU; 39% of the optimized GR00T N1.7 call on Jetson, small for π0.5; caused by serial Python and thread wake-ups, i.e. a software problem (VLA-HWBench **measured, 3rd-party**)                                                    | all                                                            |
| P1  | Vision encode (ViT)                        | one pass; 256 tokens per 224² image (SigLIP), 64 per frame with pixel shuffle         | RTX 4090: compute-bound (OpenVLA ViT 16.0 ms, 76% compute; GR00T N1.6 and π0.5 ViT also in the compute share, Jung). Jetson FP8: memory side, 2.9–3.7× above bound; AMD FP32: misses matrix units, 8–17× above bound (VLA-HWBench) (**measured, 3rd-party**)                                                              | SigLIP, SigLIP+DINOv2, Qwen-VL ViT, V-JEPA                      |
| P2  | LM prefill over a long prefix              | one pass over image + text + state tokens (hundreds), full or truncated depth         | **depends on prefix length and platform**: OpenVLA prefill 86% and OFT backbone 84% compute-bound (RTX 4090); π0.5 (574–712 tokens) near the ridge, compute-bound on AMD, memory side on Jetson; GR00T N1.7 (156 tokens) memory-bound on Jetson. Software-limited: 6–42% of the hardware bound (**measured, 3rd-party**)     | Gemma, Llama 2, Qwen, SmolLM2, Cosmos-Reason                    |
| P3  | Cached autoregressive decode               | 1 token per pass, KV cache grows; passes = action tokens                              | memory-bound, 100% of decode GPU time (OpenVLA, Jung); 20–52 ms per token on Jetson; reasoning tokens take ~91% of a reasoning model's call; for configs whose token count depends on the scene, latency varies 12–122% across 24 frames (VLA-HWBench) (**measured, 3rd-party**)                                                            | OpenVLA, π0-FAST, Holo-M AR                                    |
| P4  | Parallel decode + regression head          | P2 with empty action queries appended, then a tiny MLP                                | compute-bound, backbone 99% (OFT **measured, 3rd-party**)                                                                          | OpenVLA-OFT                                                    |
| P5  | Iterative denoise over a short action seq  | S steps × (expert over ~H action + state tokens, attending to a cached prefix)         | memory-bound on every platform tested (10–41 tokens per step, 20–150 FLOP per weight byte on Jetson). Dominant **in eager runs**: π0.5 75%, GR00T N1.6 65% of E2E (Jung); largest stage in 14 of 19 configs (VLA-HWBench). After fusion and low-bit weights, π0.5's prefill share rises 14% → 34% but denoise is still larger (~10 × 1.92 ms ≈ 19 ms vs 15.2 ms, **inference** from Table 15); in optimized GR00T N1.7, CPU preprocessing (39%) is the largest stage. Not only GEMM: π0.5 head is 44% elementwise (**measured, 3rd-party**)                    | π0/π0.5 expert, GR00T DiT, SmolVLA expert, HAF, SP-VLA          |
| P6  | Masked discrete de-mask over action tokens | groups of action tokens (14–100), 2–8 parallel de-mask passes per group, cached prefix | not measured; **inference**: memory-bound like P5 (few tokens per pass, full LM weights read each pass)                 | Holo-M                                                         |
| P7  | Video DiT over a long latent + VAE         | thousands of latent positions (Cosmos-Policy 9×28×28 = 7056 × 16 ch), S steps; VAE convs | compute-bound: DiT 71% of E2E, 77% of GPU time compute-bound (Cosmos-Policy **measured, 3rd-party**); torch.compile gave 0.97× (slower)                                  | Cosmos-Policy (Wan2.1 VAE + Cosmos-Predict2 DiT)               |
| P8  | Motion generator (full-body motion prior)  | SP-MGM: diffusion transformer over a short motion window, 20 DDIM steps; others vary   | no roofline; SP-MGM 8+8 layers, hidden 1024, 7.8 Hz (11.5 Hz TensorRT); SONIC planner ~12 ms on Jetson Orin, <5 ms on a laptop (**measured, authors**) | SP-MGM †; SONIC kinematic planner (architecture not checked)   |
| P9  | Small control policy (System 0)            | tiny network, one step per tick, proprioception in, joint targets out                 | latency-bound by rate, not by FLOPs (**inference**). Anchor: SONIC policy 1–2 ms per forward pass inside a 20 ms period (Jetson Orin, TensorRT + CUDA Graph, **measured, authors**) | SONIC tracker (50 Hz), Helix 02 S0 (10M params, 1 kHz, first-party), SP-GMT †, microduck policy (50 Hz) |

Notes
- P5 and P6 are both "few tokens, many passes over the same weights", so they fail
  to reuse weights. P2, P4 and P7 are "many tokens, one pass". This split, not the
  backbone family, decides which roofline region a stage lands in.
- A backbone name does not fix the pattern: Qwen3-VL-2B runs P2 + P6 in Holo-M but
  only P2 (as a frozen feature extractor) in ω-0.
- Bound is not time. VLA-HWBench Table 15 shows optimized stages at 1.6× (π0.5
  denoise step, Jetson) to 16.9× (GR00T N1.7 vision, AMD) their roofline bound.
  A model needs a per-pattern efficiency factor for each software stack, not only
  F and W.
- The bound of P1 and P2 moves with prefix length, precision and platform; only the
  few-token patterns (P3, P5) are memory-bound in every measurement so far. Wan appears in ω-0 only as a
  training-time target encoder; it does not run at inference.

---

## Table B. Composition axes

| ID  | Axis                                     | Values seen                                                                                                                         |
| --- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| C1  | How actions are decoded                  | discrete AR tokens (P3) · parallel decode (P4) · flow/diffusion expert (P5) · masked discrete (P6) · actions inside video latent (P7) · reasoning then act (P3 reasoning tokens, then P3 or P5 for actions; DeepThinkVLA, MolmoAct2-Think in VLA-HWBench) |
| C2  | How the action part reads the trunk      | shared self-attention with cached prefix KV (π0) · cross-attention to mid-layer features (GR00T N1, 12th layer) · interleaved cross/self over per-layer KV (SmolVLA) · same LM, same vocabulary (Holo-M) |
| C3  | Passes per call, serial or parallel      | 1 (OFT) · S steps (π0.5 10, GR00T 4, SP-VLA 4) · stages × steps, serial (HAF 3 stages) · groups serial × de-mask steps (Holo-M 4 × 2–8) |
| C4  | KV-cache lifetime                        | prefix cached across denoise steps (π0) · plus cross-stage cache of earlier stage outputs (HAF) · plus completed groups (Holo-M). **Implementation-dependent**: for GR00T N1.7, Isaac-GR00T recomputes the context KV every step, FlashRT once per call (VLA-HWBench) |
| C5  | Rates and loops                          | one rate (SmolVLA, π0.5) · S2 slow / S1 fast (GR00T N1 10/120 Hz, Helix 1 7–9/200 Hz) · four loops (SONIC: input 100, planner 10, policy 50, non-neural command writer 500 Hz) · Helix 02: S1 200 Hz, learned S0 1 kHz, S2 rate not stated |
| C6  | Layer chain and interface below the VLA  | analytic IK/OSC (LIBERO, VLABench) · learned tracker (SONIC) · VLA → motion generator → tracker (SP-HLM) · interface = EE pose, joints, sparse poses, motion tokens |
| C7  | Chunking and async execution             | chunk H, executed prefix, RTC (training-time or inference) · async overlap                                                            |
| C8  | Placement                                | where each loop runs and the network delay inside its deadline: Gemini Robotics: cloud backbone <160 ms query-to-response, local decoder onboard, ~250 ms end to end, 50 Hz effective via chunks (arXiv 2503.20020, **measured, authors**) · TANGO: S2+S1 on an RTX PRO 6000 server, SONIC S0 on onboard Jetson Orin NX, ~20 ms observation uplink, VLA every 0.5 s · π0: 73 ms onboard vs 86 ms off-board, 13 ms network (wiki π0 note) · SONIC: all inference onboard Jetson Orin |

Below S0 sits a hard real-time layer that this table does not model: joint PD /
FOC on per-joint driver MCUs. Example: TI TIDUFG6 humanoid reference design, one
C2000 F28P65 MCU runs 6 FOC loops at a 16 kHz control cycle with 64 kHz PWM, the
position loop coming from the motion controller over EtherCAT
(https://www.ti.com/lit/pdf/tidufg6, vendor document). SONIC's policy sends joint
targets to the G1's built-in joint PD; where that PD runs is not stated.

---

## Table C. Models as pattern compositions

Per call unless stated. ms figures are not comparable across rows (different
hardware, precision and chunk). "×N" = repeated passes.

### Fixed-base manipulation VLAs (wiki-ingested sources)

| Model          | Patterns per call                           | C2 / C5                                   | Chunk                     | Latency (hardware, evidence)                                                     |
| -------------- | ------------------------------------------- | ----------------------------------------- | ------------------------- | -------------------------------------------------------------------------------- |
| OpenVLA        | P1 → P2 → P3 ×6                             | – / one rate                              | 1 action (7 tokens)       | 209.56 ms: ViT 16.0, prefill 43.5, decode 140.9 (RTX 4090 eager, **measured, 3rd-party**)  |
| OpenVLA-OFT    | P1 → P4                                     | – / one rate                              | K = 8                     | 117 ms, MLP head 0.6 (RTX 4090 eager, **measured, 3rd-party**)                              |
| π0             | P1 → P2 → P5 ×10                            | shared attn, cached prefix / one rate     | H = 50                    | 73 ms, 3 cameras (RTX 4090, **measured, authors**)                                   |
| π0.5           | P1 → P2 → P5 ×10                            | as π0 / one rate                          | 10 (Jung) · 50 (HWBench)  | 294 ms: VLM 73, expert 221 (RTX 4090 eager, **measured, 3rd-party**)                        |
| GR00T N1       | P1 → P2 (to layer 12) → P5 ×4               | cross-attn / stated 10 Hz S2, 120 Hz S1   | H = 16                    | 63.9 ms (L40 bf16, **measured, authors**)                                            |
| GR00T N1.6     | P1 → P2 → P5 ×4                             | cross-attn                                | 16                        | 120 ms: VLM 42, DiT 78 (RTX 4090 eager, **measured, 3rd-party**)                            |
| SmolVLA        | P1 → P2 (first L/2 layers) → P5 ×10         | interleaved cross/self / one rate         | n = 50                    | 191 ms (Jetson Thor eager, **measured, 3rd-party**)                                         |
| Cosmos-Policy  | P7 (VAE) → P7 (DiT ×5)                      | – / one rate                              | 16                        | 396 ms: VAE 114, DiT 282 (RTX 4090 eager, **measured, 3rd-party**)                          |
| Helix (1)      | S2: P1 → P2 · S1: small cross-attn policy   | latent / S2 7–9 Hz, S1 200 Hz             | –                         | **measured, authors** rates only (onboard embedded GPUs)                             |

### Humanoid whole-body stacks

| System                 | VLA part (patterns per call)                                                         | Below the VLA (C6)                                                   | Chunk / rate                                                                 | Latency (hardware, evidence)                                                 |
| ---------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| SONIC + GR00T N1.5     | P1 → P2 → P5 (GR00T N1.5)                                                            | P8 planner 10 Hz (3-point mode, skipped in token mode) → P9 50 Hz → 500 Hz non-neural command writer | interface 3-point poses or 78-D token                                        | S0 policy 1–2 ms per pass, planner ~12 ms (Jetson Orin, TensorRT + CUDA Graph, **measured, authors**); GR00T N1.5 VLA latency not given |
| TANGO                  | P1 → P2 → P5 (VL backbone + action expert, remote)                                   | P9 SONIC on Jetson Orin NX; "low-level control loop at approximately 200 Hz" is ambiguous (policy inference or command streaming not stated; SONIC's own policy is 50 Hz), not used as a P9 rate | every 0.5 s: 15 actions at 30 Hz, resampled to 50 Hz                         | VLA latency not given (remote RTX PRO 6000); ~20 ms observation uplink (**measured, authors**) |
| Helix 02 (first-party) | S2 (rate not stated) → S1: 200 Hz transformer conditioned on S2 latents, outputs full-body joint targets | learned S0: 10M params, 1 kHz, full-body joint state + base motion in, joint-level actuator commands out | S1 200 Hz, S0 1 kHz                                                          | hardware per layer not stated (https://www.figure.ai/news/helix-02)          |
| OpenHLM ‡              | P1 → P2 → P5 ×10 (π0.5)                                                              | P9 tracker 50 Hz                                                     | 32-D joint commands at 10 Hz                                                 | not given; 1-step head loses ~20 pp (**measured, authors**)                      |
| HAF ‡                  | P1 → P2 (once) → [P5 ×S] ×3 stages, serial, shared weights                           | humanoid controller (TienKung)                                       | executes first 40 steps                                                      | ~0.12 s for all three stages (RTX 5090, **measured, authors**)                   |
| Holo-M †               | P1 → P2 → P6: 4 groups serial (100/62/32/14 tokens) × 2–8 de-mask passes             | decoupled WBC (RL legs + IK arms, from Ψ0)                           | 30 steps at 30 Hz, new obs every 500 ms                                      | 175.8 / 275.7 / 476.8 ms at 2 / 4 / 8 passes (RTX 5090, **measured, authors**)   |
| SP-HLM †               | P1 → P2 (frozen) → P5 ×4 (GR00T N1.7 head)                                           | P8 SP-MGM 20 DDIM steps → motion VAE encoder → P9 SP-GMT            | 40-step chunk, 34 executed, 6 kept as RTC prefix; 59-D                       | VLA 25 Hz, SP-MGM 10 Hz (TensorRT, **measured, authors**)                        |
| ω-0 ‡                  | P1 + P2 (Qwen3-VL-2B, frozen features) · P1 (V-JEPA 2.1) · T5 text encoder · joint video-action predictor (unmapped, no pattern ID yet) → P5 (DiT) | P9 SONIC                                                             | 25 latents, 8 executed                                                       | > 7 Hz (**measured, authors**)                                                   |

HAF: steps per stage S are not stated in the report read; its ablation compares
against π0.5 at 30 steps, which suggests S = 10 (**inference**).

---

## Gaps to fill before a quantitative model

1. **Token counts per pass.** Needed for every P2/P5/P6 row: prefix length,
   action + state tokens, action dimension. Jung et al. give only above or below
   the ridge. VLA-HWBench gives some: π0.5 prefix 574–712 tokens (about a fifth is
   LeRobot prompt padding), GR00T N1.7 prefix 156, denoise steps 10–41 tokens.
2. **Per-pattern FLOPs and bytes.** Derive them from the model configs (layers,
   hidden size, heads, GQA) for P1, P2, P5 and P6. Calibration target: VLA-HWBench
   Table 15 gives F/P, W/B and measured time per stage (vision, prefill, one
   denoise step) for π0.5 and GR00T N1.7 on Jetson Thor and AMD. Note it counts
   weight bytes only, not activations or KV.
3. **The rows marked "not given".** No public latency for the GR00T N1.5 VLA in
   the SONIC stack, the TANGO VLA or OpenHLM; P6 and P8 have no roofline
   measurement.
4. **Loops (C5) as a model input.** A multi-rate stack needs a per-loop deadline,
   not one end-to-end latency. Candidate: per loop, the sum of its patterns plus
   any transfer time on its path (C8: network uplink, host-device copies) must
   fit inside its period.
5. **P0 and transfers.** CPU preprocessing is up to 39% of an optimized call; it
   stays outside the accelerator unless it is moved on chip.
6. **Coverage beyond the two benchmarks (judged mainstream by us, inference).**
   For one VLA call on a fixed-base arm, P0–P7 cover every stage and action form
   of Jung et al. and VLA-HWBench (S5 postprocessing and S6 residual are not
   listed separately); the humanoid stack is covered by neither benchmark. Not
   covered yet, by us or by either benchmark:
   - π0.5 hierarchical inference (AR subtask text, then 10 denoise steps; the
     benchmarks ran LIBERO checkpoints without it): P3 → P5, cost not measured.
   - Views × frames × tokens per image as an explicit P1/P2 parameter.
   - Batch > 1: candidate-chunk sampling with a critic (Real-Time EXPO-FT, see
     `wiki://knowledge/fast-policy-trained-by-rl.md`) and cloud/fleet serving
     (C8); Jung et al. assume no request concurrency.
   - Co-location of VLA, S0 and perception on one onboard chip (contention).
   - Energy and power caps (only Jung et al. measure it, fixed-base, one model).
   - 1-step distilled heads (P5 with S = 1; excluded by both benchmarks).

Next step candidates: ingest HAF and OpenHLM so their rows rest on wiki source
notes; then fill gap 1 for π0.5 and GR00T N1.6 from their configs.
