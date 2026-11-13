"""Checkpoint fixtures: config.json and the safetensors header (names, dtypes, shapes; no weights).

    python -m workload_modeling.fixtures refresh [--local]

`refresh` downloads config.json (huggingface_hub.hf_hub_download) and the safetensors metadata
(huggingface_hub.get_safetensors_metadata, an HTTP range read, no weights) for every pinned revision.
`--local` reads the same two things from the local Hugging Face cache instead (offline).
The build and tests read only the committed fixtures (`load`).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# (repo, revision). The two policy checkpoints and the VLM backbone they were built from.
PINNED: tuple[tuple[str, str], ...] = (
    ("lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"),
    ("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
)
BACKBONE = PINNED[2]


def fixture_path(repo: str, revision: str) -> Path:
    return FIXTURE_DIR / f"{repo.replace('/', '__')}@{revision}"


def load(repo: str, revision: str) -> tuple[dict, dict[str, dict]]:
    """-> (config, header) where header maps tensor name -> {"dtype": str, "shape": [int, ...]}."""
    d = fixture_path(repo, revision)
    if not d.is_dir():
        raise FileNotFoundError(f"no fixture for {repo}@{revision}; run `python -m workload_modeling.fixtures refresh`")
    config = json.loads((d / "config.json").read_text())
    header = json.loads((d / "header.json").read_text())
    return config, header


def _read_local_header(path: Path) -> dict[str, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = json.loads(f.read(n))
    raw.pop("__metadata__", None)
    return {k: {"dtype": v["dtype"], "shape": list(v["shape"])} for k, v in raw.items()}


def _fetch(repo: str, revision: str, local: bool) -> tuple[dict, dict[str, dict]]:
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    cfg_path = hf_hub_download(repo, "config.json", revision=revision, local_files_only=local)
    config = json.loads(Path(cfg_path).read_text())
    if local:
        st = hf_hub_download(repo, "model.safetensors", revision=revision, local_files_only=True)
        header = _read_local_header(Path(st))
    else:
        meta = get_safetensors_metadata(repo, revision=revision)
        header = {}
        for fm in meta.files_metadata.values():
            for name, t in fm.tensors.items():
                header[name] = {"dtype": t.dtype, "shape": list(t.shape)}
    return config, header


def refresh(local: bool = False) -> None:
    for repo, rev in PINNED:
        config, header = _fetch(repo, rev, local)
        d = fixture_path(repo, rev)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (d / "header.json").write_text(json.dumps(dict(sorted(header.items())), indent=1) + "\n")
        print(f"{repo}@{rev[:8]}: {len(header)} tensors -> {d.relative_to(FIXTURE_DIR.parent.parent)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m workload_modeling.fixtures")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh", help="download config.json and safetensors headers for the pinned revisions")
    r.add_argument("--local", action="store_true", help="read from the local Hugging Face cache (offline)")
    a = p.parse_args(argv)
    if a.cmd == "refresh":
        refresh(local=a.local)
    return 0


if __name__ == "__main__":
    sys.exit(main())
