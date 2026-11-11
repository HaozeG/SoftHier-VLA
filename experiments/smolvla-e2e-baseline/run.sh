#!/usr/bin/env bash
# The only run tool of the SmolVLA-on-SoftHier experiment.
#
#   run.sh apply
#   run.sh <config> fetch
#   run.sh <config> prepare
#   run.sh <config> sim <run-name> -- <script> <args...>
#
# <config> is published or libero (configs/<config>.env).
#
#   apply    reset the softhier-mlir checkout (tracked files only) and apply patches/*.patch in order.
#            Activates no environment (used before G2.3, which runs a patched file in the LeRobot env).
#   fetch    hf download <checkpoint> --revision <revision>, then link model.safetensors and config.json
#            into $SHVLA_SMOLVLA_DIR.
#   prepare  the config's host prepare commands, in order. No simulator, no stamp check.
#   sim      refuse (exit 2) unless the simulator build stamp matches arch/softhier_mlir_arch.py; run
#            `python <script> <args>` from the softhier-mlir root. tests/gvsoc/run.py smolvla-e2e gets
#            --app-dir and --timeout (softhier-mlir's own per-process limit, patch 0004); every other run
#            goes under coreutils `timeout $SHVLA_SIM_TIMEOUT`. Output is teed to <app-dir>/run.log. Calling
#            again with the same run name resumes (finished phases are reused); put --fresh after `--` to
#            re-simulate them.
#
# Every sim runs in a systemd user scope with MemoryMax=$SHVLA_SIM_MEMMAX (default 13G) and no swap, so the
# kernel kills it at the cap instead of the host swapping. The record block reports the scope's peak memory
# (cgroup memory.peak: includes page cache). For scripts other than smolvla-e2e the timeout covers the whole
# python process, not each simulator process.
#
# Exit codes: 0 pass, 1 fail, 2 refused (usage, stamp or patch), 3 stopped by the timeout,
# 4 killed at the memory cap.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHMLIR="$EXP/softhier-mlir"
PATCHES="$EXP/patches"

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

# shellcheck source=/dev/null
. "$EXP/../../infra/env.sh"

# ---------------------------------------------------------------------------------------------- apply
apply_patches() {
    if [ ! -e "$SHMLIR/.git" ]; then
        echo "run.sh: $SHMLIR is not checked out: git submodule update --init experiments/smolvla-e2e-baseline/softhier-mlir" >&2
        exit 2
    fi
    # Tracked files only, so outputs softhier-mlir writes inside its own tree are kept. `reset` first: checkout
    # restores from the index, which must be HEAD's.
    git -C "$SHMLIR" reset -q
    git -C "$SHMLIR" checkout -- .
    local p
    for p in "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch; do
        if ! git -C "$SHMLIR" apply "$p"; then
            echo "run.sh: patch $(basename "$p") does not apply to softhier-mlir $(git -C "$SHMLIR" rev-parse --short HEAD)" >&2
            exit 2
        fi
    done
    echo "[apply] softhier-mlir $(git -C "$SHMLIR" rev-parse --short HEAD): $(find "$PATCHES" -name '[0-9][0-9][0-9][0-9]-*.patch' | wc -l) patches applied"
}

# ---------------------------------------------------------------------------------------------- arguments
[ $# -ge 1 ] || usage
if [ "$1" = "apply" ]; then
    [ $# -eq 1 ] || usage
    apply_patches
    exit 0
fi

[ $# -ge 2 ] || usage
CONFIG="$1"
CMD="$2"
shift 2
CONFIG_FILE="$EXP/configs/$CONFIG.env"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "run.sh: no configuration '$CONFIG' ($CONFIG_FILE)" >&2
    exit 2
fi
# shellcheck source=/dev/null
. "$CONFIG_FILE"

if [ ! -f "$EXP/.venv/bin/activate" ]; then
    echo "run.sh: $EXP/.venv is missing: run 'uv sync' in experiments/smolvla-e2e-baseline" >&2
    exit 2
fi
# shellcheck source=/dev/null
. "$EXP/.venv/bin/activate"

# ---------------------------------------------------------------------------------------------- fetch
do_fetch() {
    [ $# -eq 0 ] || usage
    apply_patches
    local snap
    echo "[fetch] $SHVLA_CKPT_ID @ $SHVLA_CKPT_REV"
    # `hf download --quiet` prints only the snapshot folder (huggingface-cli no longer exists in huggingface_hub 1.x)
    snap="$(hf download "$SHVLA_CKPT_ID" --revision "$SHVLA_CKPT_REV" --quiet | tail -n 1)"
    if [ ! -f "$snap/model.safetensors" ] || [ ! -f "$snap/config.json" ]; then
        echo "run.sh: snapshot '$snap' lacks model.safetensors or config.json" >&2
        exit 1
    fi
    case "$snap" in
        *"$SHVLA_CKPT_REV"*) ;;
        *) echo "run.sh: snapshot '$snap' is not revision $SHVLA_CKPT_REV" >&2; exit 1 ;;
    esac
    mkdir -p "$SHVLA_SMOLVLA_DIR"
    ln -sfn "$snap/model.safetensors" "$SHVLA_SMOLVLA_DIR/model.safetensors"
    ln -sfn "$snap/config.json" "$SHVLA_SMOLVLA_DIR/config.json"
    echo "[fetch] $SHVLA_SMOLVLA_DIR/model.safetensors -> $(readlink -f "$SHVLA_SMOLVLA_DIR/model.safetensors")"
    echo "[fetch] $SHVLA_SMOLVLA_DIR/config.json -> $(readlink -f "$SHVLA_SMOLVLA_DIR/config.json")"
}

# ---------------------------------------------------------------------------------------------- prepare
do_prepare() {
    [ $# -eq 0 ] || usage
    apply_patches
    if [ ! -f "$SHVLA_SMOLVLA_DIR/model.safetensors" ]; then
        echo "run.sh: $SHVLA_SMOLVLA_DIR/model.safetensors missing: run 'run.sh $CONFIG fetch' first" >&2
        exit 1
    fi
    local n=${#SHVLA_PREPARE_STEPS[@]} i step out t0 t1 words rc
    for ((i = 0; i < n; i++)); do
        step="${SHVLA_PREPARE_STEPS[$i]}"
        out="${SHVLA_PREPARE_OUT[$i]}"
        if [[ "$step" == *"$SHVLA_GOLDEN"* ]] && [ ! -f "$SHVLA_GOLDEN" ]; then
            echo "run.sh: step $((i + 1)) needs the golden reference $SHVLA_GOLDEN (infra/lerobot/golden_ref.py --obs $SHVLA_GOLDEN_OBS)" >&2
            exit 1
        fi
        read -ra words <<<"$step"
        echo "[prepare] step $((i + 1))/$n: python -m softhier_mlir.frontend.${words[0]} ${words[*]:1}"
        t0=$(date +%s)
        rc=0
        python -m "softhier_mlir.frontend.${words[0]}" "${words[@]:1}" || rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "[prepare] step $((i + 1))/$n FAILED (exit $rc)" >&2
            exit 1
        fi
        t1=$(date +%s)
        echo "[prepare] step $((i + 1))/$n done: $((t1 - t0)) s wall, $(du -h "$out" | cut -f1) $out"
    done
    echo "[prepare] $CONFIG: $n steps done"
}

# ---------------------------------------------------------------------------------------------- sim
SIDE_NOTE_RE='No entry found for burst|Executing illegal instruction|Received out-of-bound request'

print_record() {   # app-dir, exit code
    local app="$1" rc="$2" f
    echo "---- record block (paste into LOG.md) ----"
    echo "command:    run.sh $CONFIG sim $RUN_NAME -- ${SIM_ARGS[*]}"
    echo "run name:   $RUN_NAME   app dir: $app"
    echo "permalinks:"
    python "$SHVLA_ROOT/infra/softhier/stamp.py" permalinks | sed 's/^/  /'
    echo "  softhier-mlir: https://github.com/runwangdl/softhier-mlir/tree/$(git -C "$SHMLIR" rev-parse HEAD) (+ patches)"
    echo "stamp:      $STAMP_LINE"
    echo "patches:    $(for f in "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch; do printf '%s:%s ' "$(basename "$f")" "$(sha256sum "$f" | cut -c1-8)"; done)"
    echo "exit code:  $rc"
    echo "memory:     peak $(cat "$app/.mem_peak_mb" 2>/dev/null || echo '?') MB (scope cap ${SHVLA_SIM_MEMMAX:-13G}), oom_kill events: $(cat "$app/.mem_oom" 2>/dev/null || echo '?')"
    python - "$app" <<'PY'
import json, pathlib, sys
app = pathlib.Path(sys.argv[1])
for f in sorted(app.glob("*.json"), key=lambda p: p.stat().st_mtime):
    try:
        d = json.loads(f.read_text())
    except ValueError:
        continue
    if "roi_ns" in d and "wall_s" in d and f.with_suffix(".log").exists():
        roi = d["roi_ns"]
        print(f"phase {f.stem:<10} ok={d.get('ok')} roi={roi / 1e6 if roi is not None else float('nan'):.3f} ms simulated, "
              f"wall={d['wall_s']} s, max child RSS={d.get('max_child_rss_mb', float('nan')):.0f} MB")
PY
    echo "------------------------------------------"
}

do_sim() {
    [ $# -ge 3 ] || usage
    RUN_NAME="$1"
    shift
    [ "$1" = "--" ] || usage
    shift
    SIM_ARGS=("$@")
    SCRIPT="${SIM_ARGS[0]}"
    local stamp_py="$SHVLA_ROOT/infra/softhier/stamp.py"
    local arch="arch/softhier_mlir_arch.py"

    # 1. refuse on a missing or mismatching simulator stamp
    if [ ! -f "$stamp_py" ] || [ ! -f "$EXP/$arch" ]; then
        echo "run.sh: refused: $( [ -f "$stamp_py" ] || echo "infra/softhier/stamp.py" ) $( [ -f "$EXP/$arch" ] || echo "experiments/smolvla-e2e-baseline/$arch" ) not present yet (G1.4 / G3.0); no simulator stamp to check" >&2
        exit 2
    fi
    if ! STAMP_LINE="$(cd "$EXP" && python "$stamp_py" check "$arch" 2>&1)"; then
        echo "run.sh: refused: simulator stamp check failed:" >&2
        echo "$STAMP_LINE" >&2
        echo "run.sh: run infra/softhier/build_arch.sh $EXP/$arch explicitly, then retry" >&2
        exit 2
    fi
    STAMP_LINE="$(echo "$STAMP_LINE" | tail -n 1)"
    apply_patches

    # 2. simulator runtime library path. softhier-mlir's build_sw finds the GCC 14.2 toolchain under
    # $SOFTHIER_HOME itself, and run_sim sets PATH, PYTHONPATH, SYSTEMC_HOME and the install/ library path for
    # gapy; what it does not set is the DRAMSys directory, without which the simulator segfaults at start
    # (exit 139, G3.1). gvsoc_env.sh is not sourced: it would activate the GVSoC venv, but the experiment
    # venv's python must run softhier-mlir.
    export SYSTEMC_HOME="$SOFTHIER_HOME/third_party/systemc_install"
    export LD_LIBRARY_PATH="$SYSTEMC_HOME/lib64:$SOFTHIER_HOME/third_party/DRAMSys${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    local app="$SHVLA_DATA/runs/$CONFIG/$RUN_NAME"
    mkdir -p "$app"
    local cmd=(python "${SIM_ARGS[@]}")
    if [ "$SCRIPT" = "tests/gvsoc/run.py" ] && [ "${SIM_ARGS[1]:-}" = "smolvla-e2e" ]; then
        # softhier-mlir's own subprocess timeout per simulator process (patch 0004)
        cmd+=(--app-dir "$app" --timeout "$SHVLA_SIM_TIMEOUT")
    else
        cmd=(timeout "$SHVLA_SIM_TIMEOUT" "${cmd[@]}")
    fi

    export PYTHONUNBUFFERED=1
    rm -f "$app/.mem_peak_mb" "$app/.mem_oom"
    # The wrapper runs inside the scope and writes the scope's peak memory and OOM-kill count when the command ends.
    local wrap='"$@"; rc=$?; d=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup);
        echo $(( $(cat $d/memory.peak 2>/dev/null || echo 0) / 1048576 )) >"$SHVLA_MEMOUT.mem_peak_mb";
        awk "/^oom_kill /{print \$2}" $d/memory.events >"$SHVLA_MEMOUT.mem_oom" 2>/dev/null; exit $rc'
    cmd=(systemd-run --user --scope -q -p "MemoryMax=${SHVLA_SIM_MEMMAX:-13G}" -p MemorySwapMax=0
         env "SHVLA_MEMOUT=$app/" bash -c "$wrap" wrap "${cmd[@]}")

    echo "[sim] $CONFIG/$RUN_NAME: ${cmd[*]}   (limit ${SHVLA_SIM_TIMEOUT} s; memory cap ${SHVLA_SIM_MEMMAX:-13G}, no swap)"
    free -g | sed -n 2p
    local rc=0 start=0
    [ -f "$app/run.log" ] && start=$(stat -c %s "$app/run.log")
    set +e
    ( cd "$SHMLIR" && "${cmd[@]}" ) 2>&1 | tee -a "$app/run.log"
    rc="${PIPESTATUS[0]}"
    set -e

    # 5. classify
    # only the text this invocation appended (a resumed run keeps the earlier output in the same file)
    tail -c +$((start + 1)) "$app/run.log" >"$app/.this_run.log"
    local code=0
    local oom=0
    [ -f "$app/.mem_oom" ] && oom=$(cat "$app/.mem_oom")
    if [ "$rc" -eq 137 ] || [ "${oom:-0}" -gt 0 ] || grep -Eq "oom-kill|Out of memory|^Killed$|Killed +.*gvsoc" "$app/.this_run.log"; then
        echo "[sim] MEMORY CAP: killed at ${SHVLA_SIM_MEMMAX:-13G} (exit $rc, oom_kill events ${oom:-0}); peak $(cat "$app/.mem_peak_mb" 2>/dev/null || echo '?') MB" >&2
        code=4
    elif [ "$rc" -eq 124 ] || grep -q "TimeoutExpired" "$app/.this_run.log"; then
        code=3
    elif [ "$rc" -ne 0 ]; then
        code=1
    fi
    if grep -Eq "$SIDE_NOTE_RE" "$app/.this_run.log"; then
        echo "[sim] side-note error strings in run.log (see the side-note table in the design):"
        grep -E "$SIDE_NOTE_RE" "$app/.this_run.log" | sort | uniq -c | head -10
    fi
    if [ "$code" -ne 0 ]; then
        echo "[sim] last 40 lines of $app/run.log:"
        tail -n 40 "$app/.this_run.log"
    fi
    rm -f "$app/.this_run.log"
    print_record "$app" "$code"
    exit "$code"
}

case "$CMD" in
    fetch) do_fetch "$@" ;;
    prepare) do_prepare "$@" ;;
    sim) do_sim "$@" ;;
    *) usage ;;
esac
