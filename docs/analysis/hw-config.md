# SoftHier hardware parameters for stage 1, sized against Jetson Thor

**Status: proposal.** The 16-cluster column is the kernel repo's arch as configured. The 256-cluster
column scales it by cluster count to Thor's peak FP16; it has not been built or simulated. Inputs are
keys in [`constants.yaml`](../../infra/workload-modeling/constants.yaml); peaks are computed by
`constants.py` (`peak_total`, `iso_peak_mesh`) and tested.

## Why this arch

The kernel repo (`~/softhier@52859c3`, branch `softhier_deeploy`, `implementation/config/arch/arch.py`)
is the arch the hand-written kernels (FlatAttention, SUMMA GEMM, RMSNorm, activations) target, and
Spatz is enabled in it. It differs from PR1's arch:

| | Kernel-repo arch (stage 1) | PR1 arch |
| --- | --- | --- |
| RedMulE array | 32 x 16 | 128 x 32 |
| Cores per cluster | 5 (Spatz on cores 0-3) | 3 (Spatz off: the compiler cannot use it) |
| TCDM per cluster | 384 KiB | 1 MiB |
| NoC link | 1024 bit | 512 bit |
| Peak FP16, 16 clusters | 16.384 TFLOP/s | 131.072 TFLOP/s |

PR1's numbers (all `arch: pr1` keys) were measured on PR1's arch with ideal HBM
(`SOFTHIER_IDEAL_HBM=1`: HBM latency and bandwidth never limit).

## Parameter table

| Row | 16-cluster (as configured) | 256-cluster (Thor-matched) | Formula / source | Label |
| --- | --- | --- | --- | --- |
| clusters, mesh | 16, 4 x 4 | 256, 16 x 16 | smallest n with n^2 x peak per cluster >= `thor.fp16_dense`; 15 x 15 gives 230.4 TFLOP/s | measured / inference |
| clock | 1 GHz | 1 GHz | `arch.stage1.clock_hz` (simulator config `flex_cluster.py:338`, not in `arch.py`) | measured |
| RedMulE peak FP16 | 16.384 TFLOP/s | 262.144 TFLOP/s | clusters x `redmule_h` x `redmule_w` MAC x 2 x clock | inference |
| Thor peak FP16 (reference) | | 258 (dense, `thor.fp16_dense`); 153 (`thor.fp16_measured`) | VLA-HWBench Table 3; Chalmers | inference; measured |
| Spatz throughput FP16 (not in peak) | 2.048 TFLOP/s | 32.768 TFLOP/s | clusters x 4 Spatz cores (`spatz_cores`) x 4 FPUs (`spatz_function_units`) x 64/16 fp16 elements per FPU per cycle (`~/softhier@52859c3` `core/models/cpu/iss/include/isa_lib/vint.h:102`, `NUM_FPU = SPATZ_FPU x (64 / SEW)`) x 2 FLOP (one FMA per element, assumed) x clock | inference |
| TCDM per cluster, total | 384 KiB, 6 MiB | 384 KiB, 96 MiB | `arch.stage1.tcdm_bytes` x clusters | measured / inference |
| instruction memory per cluster | 64 KiB | 64 KiB | `arch.stage1.instruction_mem_bytes`; PR1's programs filled 78-86 % of it | measured |
| HBM capacity | not determined from `arch.py` alone | 1 GiB | next power of two >= SmolVLA's fp16 weights (about 0.9 GB for 450M parameters) plus activations | estimate |
| NoC link bandwidth | 128 GB/s per direction | 128 GB/s per direction | `noc_link_bits` x clock / 8 | inference |
| HBM bandwidth | about 1 TB/s ceiling with ideal HBM: 8 edge ports (`hbm_chan_placement` [4,0,0,4]) x 128 GB/s | provisional 273 GB/s (Thor's `thor.dram_bw`) | open parameter; RQ2 sweeps it | inference |

## Notes

* **Weights do not fit on chip.** SmolVLA's fp16 weights (about 0.9 GB; the action expert alone about
  196 MB) exceed 96 MiB of TCDM on 256 clusters. Only the KV cache and per-layer weight tiles are
  candidates for on-chip residency (RQ2).
* **Build risks of the 256-cluster column:** TCDM banking with the Spatz ports, the 64 KiB instruction
  memory, and mesh size in the simulator. The measurement work starts with a build check; if it
  fails, the RQs fall back to the 16-cluster arch.
* **Balance.** Thor: 258 TFLOP/s / 273 GB/s = 945 FLOP per byte. The 256-cluster SoftHier at the
  provisional 273 GB/s: 262 TFLOP/s / 273 GB/s = 960 FLOP per byte. Stage FLOP per weight byte for
  `libero-2cam`: vision 1,244, prefix 183, expert 55 ([stage-totals.md](stage-totals.md)); vision
  reaches 2,489 if the two cameras share one encoder call.
