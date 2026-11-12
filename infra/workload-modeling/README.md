# workload-modeling

A standalone analytical model of VLA workloads (SmolVLA first) and the stage-1 constants.
It reads checkpoint configs and safetensors headers (committed as fixtures) and computes, per
operator group, FLOP, unique bytes, weight bytes and FLOP per weight byte. It exists for the
stage-1 problem definition, the measurement work that follows, and the later several-models step,
where numbers change and must stay reproducible.

Design: [`docs/designs/stage1-problem-definition.md`](../../docs/designs/stage1-problem-definition.md).

## Commands

Run from this directory after `uv sync`:

| Command | What it does | Network |
| --- | --- | --- |
| `uv run python -m workload_modeling.build` | validates `constants.yaml`; writes `docs/analysis/stage-totals.md` and `.csv`, `docs/analysis/smolvla-op-graph.md` and `.csv`, and `build/smolvla-op-graph.html` (git-ignored; the Mermaid diagram needs the network, tables read offline) | no |
| `uv run pytest` | unit tests, including the cross-check regression guard | no |
| `uv run python -m workload_modeling.fixtures refresh [--local]` | re-downloads `config.json` and safetensors headers for the pinned revisions into `tests/fixtures/` (`--local`: from the Hugging Face cache) | yes (unless `--local`) |
| `uv run python -m workload_modeling.crosscheck --parse-profile … --pr1 … --libero … --experiment-commit …` | compares the model with PR1's compiled programs; writes `docs/analysis/crosscheck.md` and `.json`; exits 1 above 1 % | no, but needs PR1's kept programs |

Drift check: `uv run python -m workload_modeling.build && git diff --exit-code` shows no diff on a clean tree.

## Files

| Path | Content |
| --- | --- |
| `constants.yaml` | every number shared by stage-1 documents, each with `value`, `unit`, `label`, `source` (and `workload`, `arch`, `note` where they apply) |
| `workload_modeling/constants.py` | loads and validates the constants; computes arch-derived figures (peaks, iso-peak mesh) |
| `workload_modeling/fixtures.py` | pinned checkpoint revisions; fixture refresh and loading |
| `workload_modeling/smolvla_ops.py` | the model: `load_dims`, `build`, `stage_totals`, `Lowering` / `pr1_lowering` |
| `workload_modeling/crosscheck.py` | the one-time cross-check against PR1's programs |
| `workload_modeling/render.py`, `templates/op_graph.md.j2` | the operator-graph page and CSV |
| `workload_modeling/md_to_html.py` | Markdown to a standalone HTML page (light and dark, pinned mermaid@10.9.1) |
| `workload_modeling/build.py` | the page generator |
| `tests/fixtures/` | `config.json` and `header.json` per pinned revision |

## Label rule

Labels are the wiki's: `measured`, `estimate`, `inference` (`softhier_vla_wiki/tools/wiki.py:88`).
A number computed from a source's number is `inference`, with the computation in `note`. A community
figure that was not independently checked says "self-reported" in `note`. `constants.py` rejects an
entry without a label or source, or with any other label.
