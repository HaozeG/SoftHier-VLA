#!/usr/bin/env bash
# Sourced by build_arch.sh and run_kernel_tests.sh: the environment third_party/gvsoc/sourceme.sh
# sets (without its interactive checks and its own venv), plus infra/softhier/.venv.
# Requires infra/env.sh to be sourced first (SOFTHIER_HOME, SHVLA_ROOT).
# Toolchain: sourceme.sh puts third_party/toolchain/install/bin (GCC 14.2) on PATH; the PULP GCC
# 7.1.1 line in it is commented out. The same choice is made here.
export PATH="$SOFTHIER_HOME/install/bin:$SOFTHIER_HOME/gapy/bin:$SOFTHIER_HOME/third_party/toolchain/install/bin:$PATH"
export PYTHONPATH="$SOFTHIER_HOME/install/python:$SOFTHIER_HOME/soft_hier/flex_cluster_utilities${PYTHONPATH:+:$PYTHONPATH}"
export SYSTEMC_HOME="$SOFTHIER_HOME/third_party/systemc_install"
export LD_LIBRARY_PATH="$SYSTEMC_HOME/lib64:$SOFTHIER_HOME/third_party/DRAMSys${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# shellcheck source=/dev/null
. "$SHVLA_ROOT/infra/softhier/.venv/bin/activate"
