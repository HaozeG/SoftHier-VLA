"""Load and validate constants.yaml; compute arch-derived figures (never stored)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

# Copied from softhier_vla_wiki/tools/wiki.py:88 (EVIDENCE_LABELS), not imported:
# the wiki has its own venv and governance.
LABELS = ("measured", "estimate", "inference")
Label = Literal["measured", "estimate", "inference"]

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "constants.yaml"
WORKLOAD_FIELDS = ("checkpoint", "cameras", "image_size", "language_tokens", "state_tokens", "label", "source")


class ConstantsError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    value: float | int | str
    unit: str
    label: Label
    source: str
    workload: str | None = None
    arch: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Workload:
    id: str
    repo: str
    revision: str
    cameras: int
    image_size: int
    language_tokens: int
    state_tokens: int
    label: Label
    source: str
    primary: bool = False
    note: str | None = None


def _check_label(key: str, d: dict) -> None:
    if not d.get("source"):
        raise ConstantsError(f"{key}: missing source")
    if d.get("label") not in LABELS:
        raise ConstantsError(f"{key}: label {d.get('label')!r} not in {LABELS}")


def _is_entry(d: Any) -> bool:
    return isinstance(d, dict) and "value" in d


class Constants:
    def __init__(self, raw: dict):
        self.raw = raw
        self.entries: dict[str, Entry] = {}
        self.workloads: dict[str, Workload] = {}
        self._walk(raw, "")
        for wid, w in (raw.get("workloads") or {}).items():
            missing = [f for f in WORKLOAD_FIELDS if f not in w]
            if missing:
                raise ConstantsError(f"workloads.{wid}: missing {missing}")
            _check_label(f"workloads.{wid}", w)
            repo, _, rev = w["checkpoint"].partition("@")
            if not rev:
                raise ConstantsError(f"workloads.{wid}: checkpoint must be pinned as repo@sha")
            self.workloads[wid] = Workload(
                id=wid, repo=repo, revision=rev, cameras=int(w["cameras"]), image_size=int(w["image_size"]),
                language_tokens=int(w["language_tokens"]), state_tokens=int(w["state_tokens"]),
                label=w["label"], source=w["source"], primary=bool(w.get("primary", False)), note=w.get("note"))
        primaries = [w for w in self.workloads.values() if w.primary]
        if len(primaries) > 1:
            raise ConstantsError("more than one primary workload")

    def _walk(self, node: dict, prefix: str) -> None:
        for k, v in node.items():
            key = f"{prefix}{k}"
            if key == "workloads":
                continue
            if _is_entry(v):
                _check_label(key, v)
                extra = set(v) - {"value", "unit", "label", "source", "workload", "arch", "note"}
                if extra:
                    raise ConstantsError(f"{key}: unknown fields {sorted(extra)}")
                if v.get("workload") and v["workload"] not in (self.raw.get("workloads") or {}):
                    raise ConstantsError(f"{key}: unknown workload {v['workload']!r}")
                self.entries[key] = Entry(
                    value=v["value"], unit=str(v.get("unit", "")), label=v["label"], source=str(v["source"]),
                    workload=v.get("workload"), arch=v.get("arch"), note=v.get("note"))
            elif isinstance(v, dict):
                self._walk(v, key + ".")

    def get(self, key: str) -> Entry:
        try:
            return self.entries[key]
        except KeyError:
            raise KeyError(f"no constants entry {key!r}") from None

    def value(self, key: str) -> float:
        return float(self.get(key).value)

    @property
    def primary_workload(self) -> Workload:
        return next(w for w in self.workloads.values() if w.primary)

    # ------------------------------------------------------------------ arch-derived figures (computed, not stored)
    def peak_per_cluster(self, arch: str) -> float:
        """RedMulE peak FP16 per cluster in FLOP/s: h x w MAC/cycle x 2 FLOP x clock."""
        a = f"arch.{arch}."
        return self.value(a + "redmule_h") * self.value(a + "redmule_w") * 2 * self.value(a + "clock_hz")

    def clusters(self, arch: str) -> int:
        a = f"arch.{arch}."
        return int(self.value(a + "clusters_x") * self.value(a + "clusters_y"))

    def peak_total(self, arch: str, clusters: int | None = None) -> float:
        return self.peak_per_cluster(arch) * (self.clusters(arch) if clusters is None else clusters)

    def iso_peak_mesh(self, arch: str, target_key: str = "thor.fp16_dense") -> int:
        """Smallest n such that an n x n mesh reaches the target peak (TFLOP/s entry)."""
        target = self.value(target_key) * 1e12
        return math.ceil(math.sqrt(target / self.peak_per_cluster(arch)))

    def noc_link_bytes_per_s(self, arch: str) -> float:
        a = f"arch.{arch}."
        return self.value(a + "noc_link_bits") / 8 * self.value(a + "clock_hz")


def load(path: str | Path = DEFAULT_PATH) -> Constants:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConstantsError(f"{path}: top level must be a mapping")
    return Constants(raw)
