# SoftHier-VLA

This repo is for exploration on mapping VLA applications for physical AI to tile-based many-PE chip like [SoftHier](https://github.com/gvsoc/gvsoc/tree/soft_hier_release).




## Goals

1. MLIR-based deployment toolchain 
2. Workload analysis
3. Aim for MFU: Scheduling layer on software, ...


## Repository layout

| path | content |
| --- | --- |
| `softhier_vla_wiki/` | git submodule: the project wiki (knowledge notes, glossary, project decisions) |
| `docs/designs/` | design documents |
| `third_party/` | external code used as submodules, such as the GVSoC simulator (`soft_hier_release` branch) |
| `infra/` | reusable parts: environment scripts, simulator build and tests, golden reference, LIBERO evaluation |
| `experiments/<name>/` | one experiment: its compiler checkout, arch file, configs, patches and records; present only on that experiment's branch |

Layout rule: parts that later experiments reuse (simulator, golden reference, LIBERO evaluation, environment scripts) live on `main` under `third_party/` and `infra/`. Everything specific to one experiment lives under `experiments/<name>/` on that experiment's branch.

## Tracked metrics

The first baseline (SmolVLA) tracks two metrics:

*   **Latency:** the time of one policy call on SoftHier, from the GVSoC simulator.
*   **Success rate:** closed-loop task success on the LIBERO benchmark, measured on GPU.

Decision: [first baseline](softhier_vla_wiki/memories/decisions/2026-11-10-first-baseline-latency-and-success-rate.md) (in the wiki submodule). Design: [docs/designs/smolvla-softhier-baseline.md](docs/designs/smolvla-softhier-baseline.md).

## Quick start

*   Simulator (GVSoC): `infra/softhier/README.md` (added in G1).
*   Golden reference and LIBERO evaluation (LeRobot environment): `infra/lerobot/README.md` (added in G2).
