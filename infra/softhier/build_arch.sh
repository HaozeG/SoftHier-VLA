#!/usr/bin/env bash
# Build the GVSoC SoftHier hardware model for one arch file, then write the build stamp.
#
# Usage: bash infra/softhier/build_arch.sh <arch.py>
# Log:   $SHVLA_DATA/softhier/build/<arch-basename>.log
# Runs `cfg=<abs arch> make hw` in the GVSoC checkout (as its README documents). On failure the
# stamp is deleted and the exit code is non-zero, so `stamp.py check` cannot pass on a half build.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
. "$here/../env.sh"
# shellcheck source=gvsoc_env.sh
. "$here/gvsoc_env.sh"

[ $# -eq 1 ] || { echo "usage: $0 <arch.py>" >&2; exit 2; }
[ -f "$1" ] || { echo "arch file not found: $1" >&2; exit 2; }
arch="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

log_dir="$SHVLA_DATA/softhier/build"
mkdir -p "$log_dir"
log="$log_dir/$(basename "$arch" .py).log"
stamp="$SOFTHIER_HOME/.shvla_stamp.json"

rm -f "$stamp"
echo "Building hardware for $arch (log: $log)"
start=$SECONDS
rc=0
# cd, not `make -C`: the Makefile uses $(PWD) in places.
(cd "$SOFTHIER_HOME" && cfg="$arch" make hw </dev/null) >"$log" 2>&1 || rc=$?
echo "make hw exit code $rc after $((SECONDS - start)) s" | tee -a "$log"
if [ "$rc" -ne 0 ]; then
    tail -n 40 "$log" >&2
    rm -f "$stamp"
    exit "$rc"
fi
python3 "$here/stamp.py" write "$arch"
python3 "$here/stamp.py" check "$arch"
