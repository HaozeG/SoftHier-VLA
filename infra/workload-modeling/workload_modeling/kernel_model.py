"""SummaGEMM cost model on a SoftHier arch (estimate): the one model the GEMM sweeps rank candidates with and the
kernel explorer splits MFU with. Every parameter comes from constants.yaml (arch.<arch>.*).

RedMulE call time, GVSoC LightRedmule (soft_hier/flex_cluster/light_redmule.cpp init_redmule_meta_data,
get_redmule_array_runtime; identical upstream on soft_hier_je): a call of an m x n output tile with reduction k is
timed in output blocks of redmule_h rows x redmule_w * (redmule_pipe + 1) columns, redmule_pipe + 1 cycles per
reduction element per block, whatever part of the block holds data; the last reduction chunk is rounded up to
redmule_w elements. Plus redmule_call_overhead cycles per call (estimate, fitted to 9 traced GEMM shapes).

GEMM time = max(compute, HBM): compute = RedMulE calls per cluster x call time (+ splitk_reduce_step per N step for
splitK); HBM = SUMMA traffic (per group and (m, n) step an X band and a W band of the group, plus Z written once) at
hbm_channels x hbm_bw_per_channel. NoC, synchronisation and pipeline fill are not modelled; the model ranks
candidates for simulation, the simulated time decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import constants


def _up(x: int, a: int) -> int:
    return -(-x // a) * a


@dataclass(frozen=True)
class Arch:
    clusters_x: int
    clusters_y: int
    blk_m: int          # output block rows (redmule_h)
    blk_n: int          # output block columns (redmule_w * (redmule_pipe + 1))
    blk_k: int          # reduction granule (redmule_w)
    cyc_per_k: int      # cycles per reduction element per block (redmule_pipe + 1)
    clock_ghz: float
    call_overhead_cyc: float
    splitk_step_ns: float
    hbm_bytes_per_ns: float

    @property
    def mac_per_ns(self) -> float:
        """RedMulE peak MAC per ns per cluster: one block per cyc_per_k cycles."""
        return self.blk_m * self.blk_n / self.cyc_per_k * self.clock_ghz


def arch(c: constants.Constants | None = None, name: str = "h1") -> Arch:
    c = c or constants.load()
    a = f"arch.{name}."
    v = lambda k: c.value(a + k)  # noqa: E731
    h, w, pipe = int(v("redmule_h")), int(v("redmule_w")), int(v("redmule_pipe"))
    return Arch(clusters_x=int(v("clusters_x")), clusters_y=int(v("clusters_y")), blk_m=h, blk_n=w * (pipe + 1),
                blk_k=w, cyc_per_k=pipe + 1, clock_ghz=v("clock_hz") / 1e9,
                call_overhead_cyc=v("redmule_call_overhead"), splitk_step_ns=v("splitk_reduce_step"),
                hbm_bytes_per_ns=v("hbm_channels") * v("hbm_bw_per_channel"))


def block_fill(m: int, n: int, k: int, A: Arch) -> float:
    """Share of the RedMulE array time of one m x n x k call that does real MACs (a partial block costs a full one)."""
    return (m * n * k) / (_up(m, A.blk_m) * _up(n, A.blk_n) * _up(k, A.blk_k))


def tile_as_timed(m: int, n: int, k: int, A: Arch) -> tuple[int, int, int]:
    return _up(m, A.blk_m), _up(n, A.blk_n), _up(k, A.blk_k)


def call_ns(m: int, n: int, k: int, A: Arch, overhead: bool = True) -> float:
    """Time of one RedMulE call (array time from the simulator's rule, plus the per-call overhead if asked)."""
    cyc = (_up(m, A.blk_m) // A.blk_m) * (_up(n, A.blk_n) // A.blk_n) * _up(k, A.blk_k) * A.cyc_per_k
    return (cyc + (A.call_overhead_cyc if overhead else 0.0)) / A.clock_ghz


def iterations(Mp: int, Np: int, Kp: int, mt: int, nt: int, kt: int, sx: int, sy: int, g: int,
               strategy: str) -> tuple[int, int, int]:
    """(M, N, K) iterations per cluster, SummaGEMM.h:142-144 (splitN: groups own N blocks; splitK: K blocks)."""
    m_it = -(-Mp // (sy * mt))
    n_it = -(-(Np // g) // (sx * nt)) if strategy == "splitN" else -(-Np // (sx * nt))
    k_it = -(-(Kp // g) // kt) if strategy == "splitK" else -(-Kp // kt)
    return m_it, n_it, k_it


def gemm_ns(Mp: int, Np: int, Kp: int, mt: int, nt: int, kt: int, sx: int, sy: int, g: int, strategy: str,
            A: Arch) -> dict:
    """Model time of a padded SummaGEMM setting: compute, HBM and their max (ns, estimate)."""
    m_it, n_it, k_it = iterations(Mp, Np, Kp, mt, nt, kt, sx, sy, g, strategy)
    t_comp = m_it * n_it * k_it * call_ns(mt, nt, kt, A)
    if strategy == "splitK":
        t_comp += m_it * n_it * A.splitk_step_ns
    Kg = Kp // g if strategy == "splitK" else Kp
    hbm = g * m_it * n_it * 2 * Kg * (sy * mt + sx * nt) + 2 * Mp * Np   # fp16: X and W bands, Z written once
    t_hbm = hbm / A.hbm_bytes_per_ns
    return dict(t_comp=t_comp, t_hbm=t_hbm, est_ns=max(t_comp, t_hbm), calls=m_it * n_it * k_it,
                m_iter=m_it, n_iter=n_it, k_iter=k_it)
