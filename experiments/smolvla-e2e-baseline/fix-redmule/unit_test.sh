#!/usr/bin/env bash
# Host unit test of the two conversion functions, extracted verbatim from light_redmule.cpp.
# usage: unit_test.sh <light_redmule.cpp> <trunc|rne>
set -euo pipefail
SRC="$1"; MODE="$2"; D="$(cd "$(dirname "$0")" && pwd)"; T="$(mktemp -d)"
{
    printf '%s\n' '#include <stdint.h>' '#include <stdio.h>' '#include <string.h>'
    sed -n '/^typedef union/,/^} FloatBits;/p' "$SRC"
    echo 'typedef uint16_t fp16;'
    sed -n '/^\/\/ Convert float to FP16/,/^\/\/ Fused multiply-add for FP16/p' "$SRC" | sed '$d'
    cat "$D/unit_test_main.c"
} > "$T/t.c"
gcc -O1 -o "$T/t" "$T/t.c"
"$T/t" > "$T/out.bin"
"${PYTHON:-python3}" -I "$D/unit_check.py" "$T/out.bin" "$MODE"
