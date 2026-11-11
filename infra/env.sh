#!/usr/bin/env bash
# Environment defaults for the SoftHier-VLA scripts.
#
# Source it:   source infra/env.sh
# Show values: bash infra/env.sh --show
#
# Every default applies only if the variable is not already set. Per-machine
# overrides go in infra/env.local.sh (gitignored; see env.local.sh.example),
# which is sourced last.
#
# HF_HOME is deliberately not given a default: when unset, Hugging Face uses
# its own default (~/.cache/huggingface).
# SHVLA_SMOLVLA_DIR is set by the experiment configs, not here.
# SOFTHIER_IDEAL_HBM is not exported: softhier-mlir's run_sim sets it itself.

_shvla_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SHVLA_ROOT:-}" ]; then
    SHVLA_ROOT="$(git -C "$_shvla_env_dir" rev-parse --show-toplevel)"
fi
export SHVLA_ROOT

: "${SHVLA_DATA:=$HOME/.cache/softhier-vla}"
export SHVLA_DATA

: "${SOFTHIER_HOME:=$SHVLA_ROOT/third_party/gvsoc}"
export SOFTHIER_HOME

# Empty on an x86_64 host (no chroot). An explicitly set value, even empty, is kept.
: "${SOFTHIER_CHROOT=}"
export SOFTHIER_CHROOT

# Seconds per simulator process.
: "${SHVLA_SIM_TIMEOUT:=7200}"
export SHVLA_SIM_TIMEOUT

# Power-capture interval compiled into the GVSoC build (picoseconds of simulated time; build variable
# pwr_interval_ps, used by infra/softhier/build_arch.sh). GVSoC's default, 100 us, makes every simulation
# longer than 100 us run `make ice_power` and grow past 13 GB (experiments/smolvla-e2e-baseline/LOG.md,
# G3.1). 9e15 ps = 9000 s of simulated time, longer than any run. stamp.py check fails if the build differs.
: "${SHVLA_PWR_INTERVAL_PS:=9000000000000000}"
export SHVLA_PWR_INTERVAL_PS

# shellcheck source=/dev/null
[ -f "$_shvla_env_dir/env.local.sh" ] && . "$_shvla_env_dir/env.local.sh"
unset _shvla_env_dir

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "${1:-}" = "--show" ]; then
        echo "SHVLA_ROOT=$SHVLA_ROOT"
        echo "SHVLA_DATA=$SHVLA_DATA"
        echo "HF_HOME=${HF_HOME:-<unset: Hugging Face default>}"
        echo "SHVLA_SMOLVLA_DIR=${SHVLA_SMOLVLA_DIR:-<unset: set by the experiment config>}"
        echo "SOFTHIER_HOME=$SOFTHIER_HOME"
        echo "SOFTHIER_CHROOT='${SOFTHIER_CHROOT}'"
        echo "SHVLA_SIM_TIMEOUT=$SHVLA_SIM_TIMEOUT"
        echo "SHVLA_PWR_INTERVAL_PS=$SHVLA_PWR_INTERVAL_PS"
        echo "SOFTHIER_MODEL_DIR=${SOFTHIER_MODEL_DIR:-<unset>}"
        echo "SOFTHIER_STOCK_MODELS=${SOFTHIER_STOCK_MODELS:-<unset>}"
    else
        echo "usage: source infra/env.sh   |   bash infra/env.sh --show" >&2
        exit 2
    fi
fi
