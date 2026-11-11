#!/usr/bin/env bash
# Apply the GVSoC patch series (infra/softhier/gvsoc-patches/) to a GVSoC checkout.
#
# Usage: bash infra/softhier/apply_gvsoc_patches.sh [gvsoc-dir]     (default: $SOFTHIER_HOME)
#
# The series goes on top of the upstream base commit BASE and must give exactly the tree TREE
# (both in gvsoc-patches/expected). States:
#   HEAD^{tree} == TREE                    -> "already applied", exit 0 (second run, or the dev branch
#                                              shvla/soft_hier_release, which has the same tree)
#   HEAD == BASE, tracked files clean      -> git am the series, then assert the tree
#   anything else                          -> refuse (exit 1) with the reason
# Run it after the submodule checkout and before `make softhier_preparation`: two of the patches change
# soft_hier/gvsoc_core.patch and gvsoc_pulp.patch, which the preparation applies to the nested repos.
# Committer identity and date are fixed (date = author date), so the commit ids are the same everywhere.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pdir="$here/gvsoc-patches"
if [ $# -ge 1 ]; then
    gv="$1"
else
    # shellcheck source=../env.sh
    . "$here/../env.sh"
    gv="$SOFTHIER_HOME"
fi
BASE="$(sed -n 's/^BASE=//p' "$pdir/expected")"
TREE="$(sed -n 's/^TREE=//p' "$pdir/expected")"
[ -n "$BASE" ] && [ -n "$TREE" ] || { echo "apply_gvsoc_patches: $pdir/expected lacks BASE or TREE" >&2; exit 1; }

g() { git -C "$gv" "$@"; }
g rev-parse --git-dir >/dev/null 2>&1 || { echo "apply_gvsoc_patches: $gv is not a git checkout" >&2; exit 1; }

head_tree="$(g rev-parse 'HEAD^{tree}')"
if [ "$head_tree" = "$TREE" ]; then
    echo "GVSoC patch series already applied (tree ${TREE:0:12}); nothing to do."
    exit 0
fi
if [ -d "$(g rev-parse --absolute-git-dir)/rebase-apply" ]; then
    echo "apply_gvsoc_patches: a git am is half done in $gv. Run: git -C $gv am --abort" >&2
    exit 1
fi
head="$(g rev-parse HEAD)"
if [ "$head" != "$BASE" ]; then
    echo "apply_gvsoc_patches: $gv is at ${head:0:12}, neither the base ${BASE:0:12} nor the series result (tree ${TREE:0:12})." >&2
    echo "  If you changed the dev branch, re-export the series (infra/softhier/README.md, 'Developing a fix')." >&2
    echo "  If the submodule was just re-initialised, check it out at the base: git -C $gv checkout ${BASE:0:12}" >&2
    exit 1
fi
if ! g diff --quiet --ignore-submodules=all HEAD -- || ! g diff --cached --quiet --ignore-submodules=all --; then
    echo "apply_gvsoc_patches: tracked files in $gv are modified; refusing to apply on top." >&2
    exit 1
fi

files=()
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    [ -f "$pdir/$line" ] || { echo "apply_gvsoc_patches: $pdir/$line listed in series but missing" >&2; exit 1; }
    files+=("$pdir/$line")
done <"$pdir/series"

echo "Applying ${#files[@]} patches from $pdir to $gv (base ${BASE:0:12})"
# --keep-non-patch: keep the "[shvla]" / "[G3.1]" prefixes of the subjects. mailinfo.scissors off: a scissors line in a
# body must not cut the message. Whitespace and CRLF handling neutral: the nested .patch files stay byte-exact.
if ! GIT_COMMITTER_NAME="SoftHier-VLA patch series" GIT_COMMITTER_EMAIL="noreply@softhier-vla.invalid" \
     git -C "$gv" -c commit.gpgsign=false -c mailinfo.scissors=false -c core.autocrlf=false \
        -c apply.whitespace=nowarn \
        am --keep-non-patch --committer-date-is-author-date --whitespace=nowarn "${files[@]}"; then
    g am --abort || true
    echo "apply_gvsoc_patches: git am failed; aborted, the checkout is back at the base." >&2
    exit 1
fi
new_tree="$(g rev-parse 'HEAD^{tree}')"
if [ "$new_tree" != "$TREE" ]; then
    echo "apply_gvsoc_patches: applied, but the tree ${new_tree:0:12} differs from the expected ${TREE:0:12}." >&2
    echo "  Undo with: git -C $gv reset --hard ${BASE:0:12}" >&2
    exit 1
fi
echo "GVSoC patch series applied: HEAD $(g rev-parse --short=12 HEAD), tree ${TREE:0:12}."
