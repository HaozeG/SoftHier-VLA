#!/usr/bin/env bash
# diag8 run 1: bank_36 (TCDM 0x54722) + core stores + iDMA + RedMulE summary, cluster 0, until 363.5 us
cd /home/haoze/SoftHier-VLA/.claude/worktrees/smolvla-e2e-baseline/experiments/smolvla-e2e-baseline || exit 2
export SHVLA_SIM_MEMMAX=8G D8_STOP_PS=363500000 D8_OUT=$HOME/.cache/softhier-vla/diag8/run1.trace
export D8_TRACES='/chip/cluster_0/tcdm/bank_36/trace,/chip/cluster_0/pe[0-2]/insn,/chip/cluster_0/fp_ss[0-2]/insn,/chip/cluster_0/redmule/trace,/chip/cluster_0/idma/fe/trace'
export D8_RULES='[["bank_36","offset: 0x15[0-3][0-9a-f],",300000],["insn","(sw|sh|sb|fsw|fsd|fsh|sd|amo[a-z]*) .*PA:(000|300)5472[0-7]",2000],["insn","(sw|sh|sb|fsw|fsd|fsh|sd|amo[a-z]*) .*PA:(000|300)",3000,330000000,363500000],["redmule","Finished",200],["idma","Finished",2000]]'
free -g | sed -n 2p
./run.sh published sim "${1:-diag8-run1}" -- "$HOME/.cache/softhier-vla/diag8/trace_wrap.py" siglip-mlir --seq 256 --cluster all 2>&1 | grep "d8\|exit code\|memory:" | cut -c1-300 | tail -5
ls -la "$D8_OUT"
