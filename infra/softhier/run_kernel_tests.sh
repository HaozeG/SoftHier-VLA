#!/usr/bin/env bash
# Run the four GVSoC implementation kernels (gemm, norm, acti, attn) on the simulator.
#
# Usage: bash infra/softhier/run_kernel_tests.sh
# Logs:  $SHVLA_DATA/softhier/g1/<k>-pre.log (build) and <k>.log (simulation)
#
# Per kernel this runs the implementation/Makefile's own chain, split in two so that the time
# limit covers only the simulator:
#   1. `make <k>-pre`   (clean, config, preload, `make hs`: REBUILDS THE HARDWARE with
#                        implementation/config/arch/arch.py, then builds the kernel ELF)
#   2. `make run pld=<preload.elf>` in the GVSoC checkout (what `<k>-run` runs after `<k>-pre`),
#      under coreutils `timeout ${SHVLA_SIM_TIMEOUT:-7200}`.
# `make <k>-run` alone would run `<k>-pre` as a prerequisite; calling `<k>-pre` and `<k>-run`
# separately would do it twice.
#
# Pass = simulator exit 0, a "[Performance Counter]: Execution period is N ns" line (same regex as
# softhier-mlir's PERF_RE), and none of the side-note error strings. The kernels' main.c print no
# result and nothing compares the output with a golden result (O_golden in the *_preload.py is
# only a placeholder), so there is no kernel check string to require.
# Stops at the first failure. Because `<k>-pre` rebuilds the hardware, the stamp is rewritten at
# the end for implementation/config/arch/arch.py (also after a failure, since the build changed).
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
. "$here/../env.sh"
# shellcheck source=gvsoc_env.sh
. "$here/gvsoc_env.sh"

impl="$SOFTHIER_HOME/implementation"
arch="$impl/config/arch/arch.py"
log_dir="$SHVLA_DATA/softhier/g1"
mkdir -p "$log_dir"
limit="${SHVLA_SIM_TIMEOUT:-7200}"

declare -A swdir=([gemm]=SummaGEMM [norm]=RMSNorm [acti]=Activation [attn]=FlatAttention)
perf_re='\[Performance Counter\]: Execution period is ([0-9]+) ns'
bad_re='No entry found for burst|Executing illegal instruction|Received out-of-bound request'

rows=()
overall=0
for k in gemm norm acti attn; do
    pre_log="$log_dir/$k-pre.log"
    sim_log="$log_dir/$k.log"
    result=FAIL; ns="-"; pre_s="-"; sim_s="-"

    echo "== $k: make $k-pre (log $pre_log)"
    t0=$SECONDS
    (cd "$impl" && make "$k-pre" </dev/null) >"$pre_log" 2>&1
    rc=$?
    pre_s=$((SECONDS - t0))
    if [ "$rc" -ne 0 ]; then
        echo "   $k-pre failed (exit $rc); last lines:"; tail -n 20 "$pre_log"
        rows+=("$k|FAIL (build, exit $rc)|-|$pre_s|-")
        overall=1; break
    fi

    echo "== $k: simulate (log $sim_log, limit ${limit} s)"
    t0=$SECONDS
    (cd "$SOFTHIER_HOME" && timeout "$limit" make run "pld=$impl/sw/${swdir[$k]}/preload.elf" </dev/null) >"$sim_log" 2>&1
    rc=$?
    sim_s=$((SECONDS - t0))
    ns="$(grep -oE "$perf_re" "$sim_log" | head -n1 | grep -oE '[0-9]+')"
    ns="${ns:--}"
    if [ "$rc" -eq 124 ]; then
        result="FAIL (timeout ${limit} s)"
    elif [ "$rc" -ne 0 ]; then
        result="FAIL (exit $rc)"
    elif [ "$ns" = "-" ]; then
        result="FAIL (no perf line)"
    elif grep -qE "$bad_re" "$sim_log"; then
        result="FAIL ($(grep -oE "$bad_re" "$sim_log" | head -n1))"
    else
        result=PASS
    fi
    rows+=("$k|$result|$ns|$pre_s|$sim_s")
    if [ "$result" != PASS ]; then
        echo "   $k: $result; last lines:"; tail -n 20 "$sim_log"
        overall=1; break
    fi
done

# The -pre targets rebuilt the hardware with the implementation arch: refresh the stamp.
if [ -f "$SOFTHIER_HOME/install/bin/gvsoc" ]; then
    python3 "$here/stamp.py" write "$arch"
fi

echo
printf '%-6s %-28s %12s %10s %10s\n' kernel result "sim ns" "build s" "sim wall s"
for r in "${rows[@]}"; do
    IFS='|' read -r a b c d e <<<"$r"
    printf '%-6s %-28s %12s %10s %10s\n' "$a" "$b" "$c" "$d" "$e"
done
exit "$overall"
