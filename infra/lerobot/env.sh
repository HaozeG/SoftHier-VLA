#!/usr/bin/env bash
# Environment for the LeRobot / LIBERO programs (golden_ref.py, libero_eval.py).
#
# Source it:  source infra/lerobot/env.sh
# Create the environment first:  (cd infra/lerobot && uv sync)
#
# Do not run LeRobot jobs in parallel on this machine: WSL crashed once when an
# eval and a training run were launched together. libero_eval.py runs its job
# under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`.

_shvla_lerobot_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
. "$_shvla_lerobot_dir/../env.sh"

if [ ! -f "$_shvla_lerobot_dir/.venv/bin/activate" ]; then
    echo "infra/lerobot/.venv is missing: run 'uv sync' in infra/lerobot" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=/dev/null
. "$_shvla_lerobot_dir/.venv/bin/activate"

# Preload the system libstdc++ so Mesa's LLVM (osmesa) finds the symbols it
# needs; an older libstdc++ earlier on the library path breaks it. (The earlier
# physical_ai setup needed this because conda's python shipped an old one; kept
# here as a safeguard.)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# MuJoCo renders the LIBERO cameras on the CPU through Mesa's software OpenGL
# (WSL2 has no usable EGL/Vulkan path here). Needs apt packages libosmesa6 and libglfw3.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# LIBERO writes ~/.libero/config.yaml on first import and asks for input() when
# it is missing, which hangs or fails in a background job. Keep its config under
# $SHVLA_DATA and create it once here, answering "n" to the custom-dataset question.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$SHVLA_DATA/libero}"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    echo "initialising $LIBERO_CONFIG_PATH/config.yaml" >&2
    echo n | python -c "import libero.libero" >&2
fi

unset _shvla_lerobot_dir
