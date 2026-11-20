#!/usr/bin/env bash
# Guard for every simulator run: source it, then call require_sim <arch.py> [run_dir].
#
#   . infra/softhier/require_sim.sh
#   require_sim infra/kernel-runs/arch/arch_h1.py "$out_dir"   # exits non-zero if the tree is not right
#   sim_lock                                                   # holds the tree's lock until the shell exits
#
# require_sim:
#   - takes the GVSoC tree only from $SOFTHIER_HOME (infra/env.sh: the worktree's own third_party/gvsoc if
#     initialised, else the shared patched tree $SHVLA_DATA/gvsoc); never a hard-coded path;
#   - runs `stamp.py check <arch>`: the tree must be the upstream base plus the GVSoC patch series
#     (gvsoc-patches/expected TREE), built for this exact arch file, with no unexpected edits in core/pulp.
#     On failure it prints the build_arch.sh command and returns 1; it never builds by itself;
#   - copies the stamp into run_dir (if given), so each result records which simulator produced it;
#   - unsets SOFTHIER_IDEAL_HBM (patch 0004): stage-1 runs use the DRAMSys HBM model;
#   - exports MALLOC_PERTURB_=85 (unless set): glibc fills new and freed heap with a fixed byte, so a model that
#     reads uninitialised heap (patch 0011) fails loudly and the same way every run instead of passing on zero pages.
# sim_lock: flock on $SOFTHIER_HOME/.shvla_lock, so two worktrees never build or run on the shared
#   tree at once (also the host rule: one heavy job at a time).

_rs_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
# shellcheck source=../env.sh
. "$_rs_here/../env.sh"
# shellcheck source=gvsoc_env.sh
. "$_rs_here/gvsoc_env.sh"
set -u
unset SOFTHIER_IDEAL_HBM
export MALLOC_PERTURB_="${MALLOC_PERTURB_:-85}"

require_sim() {
    local arch="$1" run_dir="${2:-}"
    if [ ! -f "$SOFTHIER_HOME/install/bin/gvsoc" ]; then
        echo "require_sim: no GVSoC build in $SOFTHIER_HOME (run infra/softhier/setup.sh, then build_arch.sh $arch)" >&2
        return 1
    fi
    if ! python3 "$_rs_here/stamp.py" check "$arch" >&2; then
        echo "require_sim: refusing to run. Rebuild with: bash infra/softhier/build_arch.sh $arch" >&2
        return 1
    fi
    if [ -n "$run_dir" ]; then
        mkdir -p "$run_dir"
        cp "$SOFTHIER_HOME/.shvla_stamp.json" "$run_dir/sim_stamp.json"
    fi
}

sim_lock() {
    exec 9>"$SOFTHIER_HOME/.shvla_lock"
    if ! flock -n 9; then
        echo "sim_lock: $SOFTHIER_HOME is in use by another build or run; waiting" >&2
        flock 9
    fi
}
