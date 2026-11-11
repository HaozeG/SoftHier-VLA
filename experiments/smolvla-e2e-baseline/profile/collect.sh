#!/usr/bin/env bash
# PROF.1: build the kept trace directory ~/.cache/softhier-vla/profile/ from the run directories (no simulation).
# Re-runnable: everything is regenerated from the logs. Needs zstd and python3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PP="python3 -I $HERE/parse_profile.py"
R="${SHVLA_DATA:-$HOME/.cache/softhier-vla}/runs"
P="${PROF_OUT:-$HOME/.cache/softhier-vla/profile}"
mkdir -p "$P"

keep() {   # keep <run dir> <dest dir> <phase...>: filtered+compressed logs, mlir, json, run.log
  local src="$1" dst="$2"; shift 2
  mkdir -p "$dst"
  for ph in "$@"; do
    [ -f "$dst/$ph.log.zst" ] || $PP compact "$src/$ph.log" "$dst/$ph.log.zst"
    cp -f "$src/$ph.mlir" "$src/$ph.json" "$dst/" 2>/dev/null || true
  done
  cp -f "$src/summary.json" "$src/run.log" "$dst/" 2>/dev/null || true
}

# 1. the published-config rung-6 run (fixi-rung6): all four phases, full run -> MEASURED
keep "$R/published/fixi-rung6" "$P/published-fixi-rung6" vision prefix_a prefix_b expert
$PP ops "$P/published-fixi-rung6" --out "$P/published-fixi-rung6/ops.csv"
$PP summary "$P/published-fixi-rung6/ops.csv" --top 30 --ref > "$P/published-fixi-rung6/summary.md"

# 2. core-level traced runs (cluster-0 instruction bins): vision 1 layer, prefix_a layer 1, expert step 0
for spec in "prof-vision-L1:vision" "prof-prefix-L1:prefix_a" "prof-expert-S1:expert"; do
  name="${spec%%:*}"; ph="${spec##*:}"
  keep "$R/published/$name" "$P/$name" "$ph"
  $PP ops "$P/$name" --out "$P/$name/ops.csv" --phases "$ph" > "$P/$name/ops.log"
  $PP insn "$P/$name/ops.csv" "$P/$name/insn_bins.csv.gz" --out "$P/$name/insn_ops.csv" > "$P/$name/insn_summary.md"
done

# 3. LIBERO G4a partial runs -> ESTIMATE (scaled to the 2-camera / 1024-token composition)
L="$P/libero-G4a"
keep "$R/libero/G4a.1" "$L/G4a.1" vision
keep "$R/libero/G4a.2" "$L/G4a.2" prefix_a
keep "$R/libero/G4a.3" "$L/G4a.3" expert
for i in 1 2 3; do $PP ops "$L/G4a.$i" --out "$L/G4a.$i/ops.csv" > "$L/G4a.$i/ops.log"; done
$PP libero --vision "$L/G4a.1/ops.csv" --prefix "$L/G4a.2/ops.csv" --expert "$L/G4a.3/ops.csv" --out "$L/libero_ops_scaled.csv"
$PP summary "$L/libero_ops_scaled.csv" --top 25 > "$L/summary.md"
du -sh "$P"/*/ | sed 's/^/  /'
find "$P" -type f -size +50M -print | sed 's/^/OVER 50 MB: /'
