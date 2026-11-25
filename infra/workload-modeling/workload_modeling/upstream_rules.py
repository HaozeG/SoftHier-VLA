"""Upstream's kernel-selection rules (gvsoc soft_hier_je 4610405), ported for the stage-1 baseline.

The stage-1 baseline selects each kernel setting by upstream's rule (stage1-spec §4.3 item 3). Where the rule
gives no valid run on H1, the smallest change to it is made and recorded in the returned `fix` field.

attn_opt: implementation/scripts/kernels/auto_config/attn_auto.py opt() (:40-98).
"""
from __future__ import annotations

from dataclasses import dataclass

TILE = 128          # 4 * redmule_ce_height (H1: RedMulE 32 high)
MESH = (8, 8)       # H1 clusters (x, y)


def _opt_axis(n: int) -> tuple[int, int] | None:
    """attn_auto opt() along one axis, as written: tile = min(128, n), assert n % tile == 0; scale = n / tile,
    rounded up to a power of two with tile = n / scale (assert n % scale == 0); scale capped at the mesh side.
    None where one of opt()'s asserts would fail."""
    tile = min(TILE, n)
    if n % tile:
        return None
    s = n // tile
    if s & (s - 1):
        s = 1 << (s - 1).bit_length()
        if n % s:
            return None
        tile = n // s
    return min(s, MESH[0]), tile


@dataclass(frozen=True)
class AttnSetting:
    sx: int
    sy: int
    Br: int
    Bc: int
    q_pad: int          # query rows per head after padding (the head-block row count of Q and O)
    kv_pad: int         # key rows per head after padding (the head-block row count of K and V)
    async_: int
    fix: str            # "" or the recorded changes to upstream's rule


def attn_opt(hq: int, hkv: int, q: int, kv: int) -> AttnSetting:
    """Rows Y = q * hq / hkv (the query heads of one KV head stacked), columns X = kv; block Br = sy * tile_y,
    Bc = sx * tile_x; async only when the batch (1 here) exceeds the free groups, so collective mode.

    Fixes (stage-1), each recorded in `fix`:
    - padding: upstream's callers pass lengths its asserts accept; SmolVLA's (50, 241, 291) are not. Each length
      is padded to the smallest value for which opt() runs as written (q per head, so Y stays q_pad * hq / hkv).
    - head coverage: upstream does not check it. With G = 64 / (sx * sy) groups, a KV-head count above G and not
      a multiple of it leaves heads unprocessed (seen: 15x241x241x64 fails the output check), so the larger scale
      is halved until the heads fit (hkv <= G or hkv % G == 0)."""
    r = hq // hkv
    fix = []
    qp = q
    while _opt_axis(qp * r) is None:
        qp += 1
    kvp = kv
    while _opt_axis(kvp) is None:
        kvp += 1
    if (qp, kvp) != (q, kv):
        fix.append(f"padded q {q}->{qp}, kv {kv}->{kvp} so opt() runs as written")
    sy, ty = _opt_axis(qp * r)
    sx, tx = _opt_axis(kvp)
    while True:
        g = (MESH[0] * MESH[1]) // (sx * sy)
        if hkv <= g or hkv % g == 0:
            break
        if sy >= sx:
            sy //= 2
        else:
            sx //= 2
    if (sx, sy) != _scales(qp * r, kvp):
        fix.append(f"scale halved to {sx}x{sy} so all {hkv} KV heads are processed")
    # block = scale * tile; the halved axis keeps opt()'s tile, so the block must still divide the padded length
    Br, Bc = sy * ty, sx * tx
    assert (qp * r) % Br == 0 and kvp % Bc == 0, (qp, kvp, Br, Bc)
    return AttnSetting(sx=sx, sy=sy, Br=Br, Bc=Bc, q_pad=qp, kv_pad=kvp, async_=0, fix="; ".join(fix))


def _scales(y: int, x: int) -> tuple[int, int]:
    return _opt_axis(x)[0], _opt_axis(y)[0]
