"""Stage-1 deployment plan: where every tensor lives and how every operator runs (stage1-spec §4.3 item 3).

Upstream's deployment method (gvsoc soft_hier_je 4610405, implementation/scripts/llm/normal_llm_plan.py), applied
to SmolVLA:

- one place per tensor, in space A (activations, scratch; H1 west) or space B (weights, residual stream, K/V,
  tables; H1 south); producer and consumer use the same address;
- attention operands head-major, (heads, rows per head, d): q/k/v projections write them through the GEMM option
  RESHA_Z_TO (TALL, chunk d) and o_proj (and the cross-attention K/V projections) read them through RESHA_X_FROM;
- KV concatenation for expert self-attention is not modelled (as upstream);
- kernels as upstream's plan names them: residual add = Activation 'none' with a full-size bias ("addi"), RoPE
  kernel on head-major data, separate gate / up and q / k / v products.

The plan is organised by tensor, as upstream's hbm_plan + kernel_flow are: a tensor carries its space and, if
head-major, its row count per head, which every kernel touching it takes (GEMM padded M, RoPE and addi rows,
attention q_pad / kv_pad). The per-run kernel config and HBM_PLACE are derived from it, so drivers and the op graph
read one source. Kernel settings (tiles, blocks) are the selection, not the plan: upstream_rules.

    python -m workload_modeling.deploy_plan [--workload vlabench-3cam]   # run list and checks
"""
from __future__ import annotations

import argparse
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass, field

from . import constants, smolvla_ops
from .upstream_rules import attn_opt

A, B = "west", "south"          # upstream: two HBM edges -> space A on the first, space B on the second (H1)
D = 64                          # head dimension

# preload defaults (kernel repo scripts/kernels/*_preload.py, upstream plan roles); the key suffix lists overrides
DEFAULTS = {
    "gemm": {"X_addr": A, "W_addr": B, "Z_eaddr": A},
    "attn": {"Q_addr": A, "K_addr": B, "V_addr": B, "O_eaddr": A},
    "elt": {"I_addr": A, "Z_addr": B, "O_eaddr": A},
    "acti": {"I_addr": A, "G_addr": A, "B_addr": B, "O_eaddr": A},
    "lnorm": {"I_addr": B, "G_addr": B, "B_addr": B, "O_eaddr": A},
    "norm": {"I_addr": B, "O_eaddr": A},
    "rope": {"I_addr": A, "C_addr": B, "S_addr": B, "P_addr": A, "O_eaddr": A},
}


@dataclass
class Tensor:
    name: str
    space: str                       # A (west) or B (south)
    heads: int | None = None         # head-major tensors: (heads, rows, D)
    rows: int | None = None          # rows per head (padded); None: free (no consumer constrains it)


@dataclass
class Deploy:
    stage: str
    op: str
    kernel: str                      # gemm | attn | elt | acti | lnorm | norm | rope
    key: str                         # constants key (without group / stage): base _ shape [_zto|_xfrom] [_p<overrides>]
    cfg: dict                        # kernel config fields fixed by the plan (shape, reshape, op / algo)
    place: dict                      # HBM_PLACE: every array of the kernel -> edge
    count: int
    tensors: dict = field(default_factory=dict)   # array -> tensor name (provenance)
    fix: str = ""


def _sig(kernel: str, place: dict) -> str:
    over = {k: v for k, v in place.items() if DEFAULTS[kernel][k] != v}
    return "_p" + "".join(f"{k[0]}{v[0]}" for k, v in sorted(over.items())) if over else ""


class Plan:
    def __init__(self, workload: str = "vlabench-3cam", c=None, lowering=None, attn: dict | None = None):
        """lowering: default PLAN_LOWERING (W1); PLAN_LOWERING_W1B batches the cameras. attn: (stage, op) ->
        AttnSetting overriding upstream's attn_opt for that attention call (the plan-constrained sweep); the
        tensors' head padding follows the attention settings, so every kernel touching them changes with it."""
        c = c or constants.load()
        self.workload = workload
        # upstream's plan has separate q / k / v and gate / up products (stage1-spec §3.3, split_fused)
        low = lowering or smolvla_ops.PLAN_LOWERING
        assert low.split_fused, "the plan has separate q / k / v and gate / up products"
        self.lowering = low
        self.dims = smolvla_ops.load_dims(c.workloads[workload])
        self.cameras = c.workloads[workload].cameras
        self.rows = smolvla_ops.annotate(smolvla_ops.build(c, workload, low))
        self.T: dict[str, Tensor] = {}
        self.attn = {}                      # (stage, op) -> AttnSetting
        self._override = dict(attn or {})
        self._tensors()
        self.deploys: list[Deploy] = [d for r in self.rows for d in [self._deploy(r)] if d]

    # ------------------------------------------------------------------ tensors
    def _t(self, name, space, heads=None, rows=None):
        self.T[name] = Tensor(name, space, heads, rows)

    def _row(self, stage, op):
        return next(r for r in self.rows if r.stage == stage and r.op == op)

    def _tensors(self):
        dm = self.dims
        for stage, op in (("vision", "qk"), ("prefix", "qk"), ("expert", "self_qk"), ("expert", "cross_qk")):
            h, q, kv, _ = self._row(stage, op).shape
            hkv = h if stage == "vision" else dm.t_kv_heads        # SigLIP: MHA; VLM / expert: GQA
            self.attn[(stage, op)] = self._override.get((stage, op)) or attn_opt(h, hkv, q, kv)
        v, p = self.attn[("vision", "qk")], self.attn[("prefix", "qk")]
        es, ec = self.attn[("expert", "self_qk")], self.attn[("expert", "cross_qk")]
        assert es.q_pad == ec.q_pad, "expert self and cross Q share rope_q and o_proj"
        hv, ht, hk = dm.v_heads, dm.t_heads, dm.t_kv_heads
        if self.lowering.attn_cameras_as_heads:
            # (heads, cameras x rows, d) = (heads x cameras, rows, d) blocks: no padding between cameras
            N = self._row("vision", "qk").shape[1]
            assert v.q_pad == N and v.kv_pad == N, ("cameras-as-heads needs unpadded vision rows", v)
            vq = vkv = self.cameras * N
        else:
            vq, vkv = v.q_pad, v.kv_pad
        self._t("vision.q", A, hv, vq); self._t("vision.k", B, hv, vkv); self._t("vision.v", B, hv, vkv)
        self._t("vision.o", A, hv, vq)
        self._t("prefix.q", A, ht, p.q_pad); self._t("prefix.k", B, hk, p.kv_pad); self._t("prefix.v", B, hk, p.kv_pad)
        self._t("prefix.o", A, ht, p.q_pad)
        self._t("expert.q", A, ht, es.q_pad); self._t("expert.o", A, ht, es.q_pad)
        # self-attention keys: prefix cache + own 50 tokens; the concatenation is not modelled, so the projection
        # writes its own head-major tensor and no consumer fixes its row count
        self._t("expert.k_self", B, hk, None); self._t("expert.v_self", B, hk, None)
        self._t("expert.k_cross", B, hk, ec.kv_pad); self._t("expert.v_cross", B, hk, ec.kv_pad)
        assert p.kv_pad == ec.kv_pad, "cross K/V projections read prefix K/V and write cross K/V at the same M"
        for s in ("vision", "connector", "prefix", "expert"):
            self._t(f"{s}.res", B)          # residual stream
            self._t(f"{s}.act", A)          # any other activation (scratch)
        self._t("w", B)                     # weights, biases, tables
        self._t("expert.action", B)         # flow-matching action state x_t

    # ------------------------------------------------------------------ operators
    def _deploy(self, r) -> Deploy | None:
        if r.op in smolvla_ops.HOST_OPS or (r.stage, r.op) in smolvla_ops.INCLUDED_IN:
            return None
        s, op, shp = r.stage, r.op, r.shape
        act, res, w = f"{s}.act", f"{s}.res", "w"
        hm = lambda n: f"{s}.{n}"
        if r.cls == "gemm":
            return self._gemm(r, act, res, w, hm)
        if op in smolvla_ops.ATTN_CALL_OPS:
            st = self.attn[(s, op)]
            if s == "expert":
                kk, vv = ("expert.k_self", "expert.v_self") if op == "self_qk" else ("expert.k_cross", "expert.v_cross")
            else:
                kk, vv = hm("k"), hm("v")
            arr = {"Q_addr": hm("q"), "K_addr": kk, "V_addr": vv, "O_eaddr": hm("o")}
            h, q, kv, d = shp
            cfg = dict(q_pad=st.q_pad, kv_pad=st.kv_pad, sx=st.sx, sy=st.sy, Br=st.Br, Bc=st.Bc, async_=st.async_)
            return self._mk(r, "attn", f"flatattention_{'x'.join(map(str, shp))}", cfg, arr, fix=st.fix)
        rr, cc = shp
        name = "x".join(map(str, shp))
        if op.startswith("rope_"):
            t = hm("q") if op == "rope_q" else (hm("k") if s == "prefix" else "expert.k_self")
            T = self.T[t]
            rows = T.rows or rr
            cfg = dict(m_size=rows, n_size=cc, view_enable=1, view_n=D)
            arr = {"I_addr": t, "O_eaddr": t, "C_addr": "w" if T.space == A else act, "S_addr": "w" if T.space == A else act,
                   "P_addr": act}
            return self._mk(r, "rope", f"rope_{name}", cfg, arr)
        if s == "vision" and op in ("bias_q", "bias_k", "bias_v"):
            t = hm(op[-1])
            T = self.T[t]
            cfg = dict(algo="'none'", gate_enable=0, bias_enable=1, m_size=T.rows, n_size=cc)
            return self._mk(r, "acti", f"addi_{name}", cfg, {"I_addr": t, "B_addr": w, "O_eaddr": t, "G_addr": act})
        if op.startswith("residual_"):
            cfg = dict(algo="'none'", gate_enable=0, bias_enable=1, m_size=rr, n_size=cc)
            return self._mk(r, "acti", f"addi_{name}", cfg, {"I_addr": res, "B_addr": act, "O_eaddr": res, "G_addr": act})
        if op in ("ln1", "ln2", "ln_post"):
            return self._mk(r, "lnorm", f"layernorm_{name}", dict(m_size=rr, n_size=cc),
                            {"I_addr": res, "G_addr": w, "B_addr": w, "O_eaddr": act})
        if op.startswith("rmsnorm_"):
            return self._mk(r, "norm", f"rmsnorm_{name}", dict(m_size=rr, n_size=cc), {"I_addr": res, "O_eaddr": act})
        if op == "gelu":
            return self._mk(r, "acti", f"gelu_{name}", dict(algo="'gelu'", gate_enable=0, bias_enable=0, m_size=rr,
                            n_size=cc), {"I_addr": act, "G_addr": act, "B_addr": w, "O_eaddr": act})
        if op == "time_silu":
            return self._mk(r, "acti", f"silu_{name}", dict(algo="'silu'", gate_enable=0, bias_enable=0, m_size=rr,
                            n_size=cc), {"I_addr": act, "G_addr": act, "B_addr": w, "O_eaddr": act})
        if op == "silu_mul":
            return self._mk(r, "acti", f"silu_gate_{name}", dict(algo="'silu'", gate_enable=1, bias_enable=0, m_size=rr,
                            n_size=cc), {"I_addr": act, "G_addr": act, "B_addr": w, "O_eaddr": act})
        kind = smolvla_ops.eltwise_kind(op)
        assert kind, (s, op)
        cfg = dict(op=f"'{kind}'", m_size=rr, n_size=cc)
        if kind == "pixel_shuffle":           # 3 cameras of 1024 x 768 -> 192 x 12288 (config rows / cols)
            cfg.update(m_size=192, n_size=12288)
        out = act
        if op in ("bias_patch",) or op == "embed_scale" or op == "bias_time_out":
            out = res                         # these write the stage's residual stream (ext rows, spec §4.3)
        if op == "pos_embed_add":
            arr = {"I_addr": res, "Z_addr": w, "O_eaddr": res}
        elif op == "euler_update":
            arr = {"I_addr": "expert.action", "Z_addr": act, "O_eaddr": "expert.action"}
        else:
            arr = {"I_addr": act, "Z_addr": w, "O_eaddr": out}
        if op == "embed_scale":
            out_t = "prefix.res"              # the connector writes the prefix's residual stream
            arr["O_eaddr"] = out_t
        return self._mk(r, "elt", f"eltwise_{kind}_{name}", cfg, arr)

    def _gemm(self, r, act, res, w, hm):
        s, op = r.stage, r.op
        M, N, K = r.shape
        x, z, cfg = act, act, dict(m_size=M, n_size=N, k_size=K)
        suf = ""
        zt = xt = None
        if op in ("q_proj", "self_q_proj"):
            zt = hm("q")
        elif op in ("k_proj", "v_proj"):
            zt = hm(op[0])
        elif op in ("self_k_proj", "self_v_proj"):
            zt = f"expert.{op[5]}_self"
        elif op in ("cross_k_proj", "cross_v_proj"):
            xt, zt = f"prefix.{op[6]}", f"expert.{op[6]}_cross"
        elif op == "o_proj":
            xt = hm("o")
        if op == "action_in_proj":
            x = "expert.action"
        if op == "time_mlp_out":
            z = act                           # its bias (bias_time_out) writes the expert residual stream
        mrows = None
        if xt:
            T = self.T[xt]
            mrows = T.rows or M
            cfg.update(resha_x_from_enable=1, resha_x_from_m=T.heads * mrows)
            x = xt
            suf += "_xfrom"
        if zt:
            T = self.T[zt]
            zr = T.rows or M
            assert mrows is None or mrows == zr, (s, op, mrows, zr)
            mrows = zr
            cfg.update(resha_z_to_enable=1, resha_z_to_m=T.heads * zr)
            z = zt
            suf += "_zto"
        if mrows is not None:
            assert mrows >= M, (s, op, mrows, M)
            cfg["m_size"] = mrows             # padded M = rows per head of the head-major tensor
        arr = {"X_addr": x, "W_addr": w, "Z_eaddr": z}
        return self._mk(r, "gemm", f"gemm_{M}x{N}x{K}{suf}", cfg, arr)

    def _mk(self, r, kernel, base, cfg, arr, fix=""):
        place = {a: self.T[t].space for a, t in arr.items()}
        assert set(place) == set(DEFAULTS[kernel]), (r.op, sorted(place), sorted(DEFAULTS[kernel]))
        return Deploy(r.stage, r.op, kernel, base + _sig(kernel, place), cfg, place, r.count, arr, fix)

    # ------------------------------------------------------------------ run list
    def runs(self) -> "OrderedDict[str, dict]":
        """Distinct runs (one per stage-qualified key): kernel, cfg, place, operators and total count. Operators
        sharing a key must share cfg and place."""
        out: OrderedDict[str, dict] = OrderedDict()
        for d in self.deploys:
            k = f"{d.stage}.{d.key}"
            if k in out:
                assert out[k]["cfg"] == d.cfg and out[k]["place"] == d.place, (k, d.op)
                out[k]["ops"].append(d.op)
                out[k]["count"] += d.count
            else:
                out[k] = dict(kernel=d.kernel, cfg=d.cfg, place=d.place, ops=[d.op], count=d.count, fix=d.fix)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="vlabench-3cam")
    a = ap.parse_args()
    p = Plan(a.workload)
    mapped = {(d.stage, d.op) for d in p.deploys}
    missing = [(r.stage, r.op) for r in p.rows if r.op not in smolvla_ops.HOST_OPS
               and (r.stage, r.op) not in smolvla_ops.INCLUDED_IN and (r.stage, r.op) not in mapped]
    assert not missing, missing
    runs = p.runs()
    for name, t in p.T.items():
        if t.heads:
            print(f"tensor {name:16} space {'A' if t.space == A else 'B'}  ({t.heads}, {t.rows or 'free'}, {D})")
    print()
    for k, v in runs.items():
        cfg = " ".join(f"{a}={b}" for a, b in v["cfg"].items())
        print(f"{k:42} {v['kernel']:5} x{v['count']:<4} {','.join(v['ops']):38} {cfg}" + (f"  [fix: {v['fix']}]" if v["fix"] else ""))
    by = {}
    for v in runs.values():
        by[v["kernel"]] = by.get(v["kernel"], 0) + 1
    print(f"\n{len(runs)} distinct runs: {by}; every device operator mapped")


if __name__ == "__main__":
    main()
