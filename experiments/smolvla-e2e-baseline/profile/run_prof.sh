#!/usr/bin/env bash
# PROF.1 traced runs. SAFE MODE: one sim at a time through run.sh with SHVLA_SIM_MEMMAX=8G; needs >= 10 GB available.
#   profile/run_prof.sh vision <name>     --vision-only 256 --layers 1   (the chain's vision program, 1 layer)
#   profile/run_prof.sh expert <name> [stop_ns]   --expert-only ref:e2e_c1_t256.npz --steps 1
#   profile/run_prof.sh prefix <name> [stop_ns]   the chain with the vision phase reused from fixi-rung6; prefix_a simulated until stop_ns
# Extra env: PROF_TRACES (insn paths), PROF_INSN_WINDOW, PROF_SAMPLE, PROF_STOP_NS. The wrapper (prof_wrap.py) filters
# every trace line in the pipe; the aggregated insn bins land in $OUT/<name>/insn_bins.csv.gz.
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:?vision|expert}"; name="${2:?run name}"
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 10 ]; then echo "run_prof: only ${avail} GB available (< 10): refused" >&2; exit 2; fi
OUT="${PROF_OUT:-$HOME/.cache/softhier-vla/profile}/$name"
mkdir -p "$OUT"
export SHVLA_SIM_MEMMAX=8G
export PROF_INSN_OUT="$OUT/insn_bins.csv.gz" PROF_SAMPLE_OUT="$OUT/insn_sample_pe0.txt.gz"
export PROF_TRACES="${PROF_TRACES:-/chip/cluster_0/pe[0-2]/insn,/chip/cluster_0/fp_ss[0-2]/insn}"
APP="$HOME/.cache/softhier-vla/runs/published/$name"
case "$what" in
  vision) ARGS=(smolvla-e2e --vision-only 256 --layers 1 --e2e-trace --app-dir "$APP") ;;
  expert) ARGS=(smolvla-e2e --expert-only "ref:$HOME/.cache/softhier-vla/smolvla_base/e2e_c1_t256.npz" --steps 1 --e2e-trace --app-dir "$APP")
          [ -n "${3:-}" ] && export PROF_STOP_NS="$3" ;;
  prefix) # the chain's prefix_a program (published config): needs the vision hand-over, which is reused from fixi-rung6 (symlinks)
          mkdir -p "$APP"; for f in vision.log vision.json; do ln -sf "$HOME/.cache/softhier-vla/runs/published/fixi-rung6/$f" "$APP/$f"; done
          ARGS=(smolvla-e2e --npz "$HOME/.cache/softhier-vla/smolvla_base/e2e_c1_t256.npz" --steps 1 --e2e-trace --app-dir "$APP")
          export PROF_STOP_NS="${3:-3600000}" ;;
  *) echo "unknown $what" >&2; exit 2 ;;
esac
./run.sh published sim "$name" -- "$PWD/profile/prof_wrap.py" "${ARGS[@]}" 2>&1 | grep -a "prof\]\|record block\|exit code\|memory:\|phase \|PASS\|FAIL\|Error" | cut -c1-300 || true
ls -la "$OUT" "$APP" | head -30
