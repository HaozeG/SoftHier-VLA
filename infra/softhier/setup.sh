#!/usr/bin/env bash
# One-time setup of the GVSoC soft_hier_release checkout:
#   1. prerequisite check (prints the exact apt command if anything is missing)
#   2. submodule init (recursive) and `uv sync` of infra/softhier
#   3. `make softhier_preparation` (SystemC, DRAMSys, RISC-V toolchain), logged and timed
#   4. permalinks of the GVSoC submodule and its nested submodules
#
# Usage: bash infra/softhier/setup.sh
#
# The Makefile's DRAMSys step runs `read -t 30 -r user_input` only if its
# prebuilt-library self-test fails. With stdin from /dev/null, read fails at
# once, user_input stays empty, and the script proceeds as if "y" was typed
# (it only aborts on "n"): a ~40 min DRAMSys rebuild.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
. "$here/../env.sh"

gvsoc="$SOFTHIER_HOME"
log_dir="$SHVLA_DATA/softhier/setup"
rel_gvsoc="${gvsoc#"$SHVLA_ROOT"/}"

# ---------------------------------------------------------------- 1. prerequisites
missing_pkgs=()
problems=()

# README list, plus wget, bzip2 and xz-utils (the Makefile's toolchain download and untar).
apt_pkgs=(build-essential git doxygen python3-pip libsdl2-dev curl cmake gtkwave
          libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config
          libsdl2-ttf-dev wget bzip2 xz-utils)
for p in "${apt_pkgs[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed'; then
        missing_pkgs+=("$p")
    fi
done

version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }  # $1 >= $2

for c in gcc g++; do
    if command -v "$c" >/dev/null 2>&1; then
        v="$("$c" -dumpfullversion)"
        version_ge "$v" 11.2.0 || problems+=("$c $v < 11.2.0")
    else
        problems+=("$c not found")
    fi
done
if command -v cmake >/dev/null 2>&1; then
    v="$(cmake --version | head -n1 | awk '{print $3}')"
    version_ge "$v" 3.18.1 || problems+=("cmake $v < 3.18.1")
else
    problems+=("cmake not found")
fi
command -v uv >/dev/null 2>&1 || problems+=("uv not found (https://docs.astral.sh/uv/)")

if [ ${#missing_pkgs[@]} -gt 0 ] || [ ${#problems[@]} -gt 0 ]; then
    echo "Prerequisites missing." >&2
    for pr in "${problems[@]}"; do echo "  - $pr" >&2; done
    if [ ${#missing_pkgs[@]} -gt 0 ]; then
        echo "Install the missing apt packages (needs the owner's approval):" >&2
        echo "  sudo apt-get install -y ${missing_pkgs[*]}" >&2
    fi
    exit 1
fi
echo "Prerequisites OK."

# ---------------------------------------------------------------- 2. submodule, uv
if git -C "$SHVLA_ROOT" submodule status "$rel_gvsoc" | grep -q '^-'; then
    git -C "$SHVLA_ROOT" submodule update --init --recursive "$rel_gvsoc"
else
    echo "Submodule $rel_gvsoc already initialised."
fi
(cd "$here" && uv sync)

# ---------------------------------------------------------------- 3. preparation
# Paths are relative to the GVSoC checkout. The directory targets are created before
# their downloads run, so check for real files, not just the folders.
prep_done() {
    [ -f "$gvsoc/third_party/systemc_install/lib64/libsystemc.so" ] \
    && [ -f "$gvsoc/third_party/DRAMSys/libDRAMSys_Simulator.so" ] \
    && [ -d "$gvsoc/third_party/toolchain" ] \
    && [ -n "$(find "$gvsoc/third_party/toolchain" -name 'riscv*-gcc' -print -quit 2>/dev/null)" ]
}

if prep_done; then
    echo "softhier_preparation already done; skipping."
else
    mkdir -p "$log_dir"
    # shellcheck disable=SC1091
    . "$here/.venv/bin/activate"
    echo "Running make softhier_preparation (log: $log_dir/preparation.log)"
    start=$SECONDS
    rc=0
    # cd, not `make -C`: the Makefile builds SYSTEMC_INSTALL_DIR from $(PWD), which make -C leaves unchanged.
    # No -j: the Makefile's recipes call plain `make` serially.
    (cd "$gvsoc" && make softhier_preparation </dev/null) >"$log_dir/preparation.log" 2>&1 || rc=$?
    echo "softhier_preparation exit code $rc after $((SECONDS - start)) s" | tee -a "$log_dir/preparation.log"
    if [ "$rc" -ne 0 ]; then
        tail -n 40 "$log_dir/preparation.log" >&2
        echo "Failed. If the toolchain download was interrupted, remove $gvsoc/third_party/toolchain before retrying." >&2
        exit "$rc"
    fi
    # The DRAMSys recipe's exit status is that of its last command, so verify the result.
    grep -E 'Test libaray (succeeded|failed)' "$log_dir/preparation.log" || true
    prep_done || { echo "Preparation finished but the expected files are missing." >&2; exit 1; }
fi

# ---------------------------------------------------------------- 4. permalinks
echo "GVSoC: https://github.com/gvsoc/gvsoc/tree/$(git -C "$gvsoc" rev-parse HEAD)"
git -C "$gvsoc" submodule foreach --quiet \
    'echo "  $name: $(git config --get remote.origin.url | sed "s/\.git$//")/tree/$(git rev-parse HEAD)"'
