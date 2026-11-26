"""Standalone analytical model of one SmolVLA policy call.

Dimensions come from three pinned fixtures (see DIMENSION_SOURCES); every config-derived shape is
checked against the safetensors header tensor of the same name. Stored dtypes are ignored: all bytes
assume fp16 (2 B per element), as the SoftHier library does.

Byte columns and the count rule (docs/analysis/conventions.md):
  - one row per operator group; `shape` is one instance; `count` = instances per call
    (layers x flow steps x cameras that apply; heads are folded into the shape, because they share
    one weight matrix; Lowering.batch_cameras folds cameras into the shape too);
  - weights_read_bytes = weight bytes of one instance x count. Each layer has its own weights; the expert's
    weights do not fit on chip, so they are read again in every flow step;
  - memory_traffic_bytes = (inputs + outputs + weights) of one instance x count;
  - flop_per_weight_byte = flop / weights_read_bytes, None for operators without weights.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace

from . import fixtures
from .constants import Constants, Workload

FP16 = 2  # bytes per element

# Per-element FLOP conventions for non-matrix operators (label: inference).
EW_FLOP = {
    "softmax": 5,     # max, subtract, exp, sum, divide (the 1/sqrt(d) scale is folded in)
    "layernorm": 8,
    "rmsnorm": 4,
    "gelu": 8,        # tanh form
    "silu": 4,
    "silu_mul": 5,
    "rope": 3,
    "add": 1,
    "bias": 1,
    "scale": 1,
    "axpy": 2,
}

STAGES = ("vision", "connector", "prefix", "expert")


@dataclass(frozen=True)
class OpRow:
    stage: str
    layers: str
    op: str
    cls: str                     # gemm | attn | softmax | norm | acti | elementwise | data
    shape: tuple[int, ...]       # (M, N, K) for gemm; (h, Sq, Skv, d) for attn; (rows, cols) otherwise
    count: int
    flop: float
    memory_traffic_bytes: float
    weights_read_bytes: float
    pr1_unit: str = ""           # unit that ran the operator in PR1's compiled path (filled by annotate)
    hand_kernel: str = ""        # stage-1 hand-written kernel that does the operator (filled by annotate)
    hand_kernel_units: str = ""  # units that kernel uses for this operator (filled by annotate)
    note: str = ""

    @property
    def flop_per_weight_byte(self) -> float | None:
        return self.flop / self.weights_read_bytes if self.weights_read_bytes else None

    @property
    def is_matmul(self) -> bool:
        return self.cls in ("gemm", "attn")


@dataclass(frozen=True)
class StageTotals:
    flop: float
    matmul_flop: float
    memory_traffic_bytes: float
    weights_read_bytes: float

    @property
    def flop_per_weight_byte(self) -> float | None:
        return self.flop / self.weights_read_bytes if self.weights_read_bytes else None


@dataclass(frozen=True)
class Lowering:
    """Shape choices and rewrites of a compiled program, to compare the model with what a compiler ran.

    The default is the reference computation (lerobot SmolVLA code) with true token counts.
    PR1's program (see pr1_lowering and crosscheck.py) differs in four ways:
      - prefix token rows padded for RedMulE (65 -> 128, 177 -> 256; read from the prefix program);
      - expert attention key length padded to a multiple of 32 by the runtime
        (softhier-mlir@b11cd5f runtime/sh_expert.inc.c:83, `(Lp + So + 31u) & ~31u`);
      - cross-attention K/V projection of the prefix done once per call, not in every step;
      - time MLP input projection split: concat(action, time) @ W = action @ W_a + time @ W_t, and the
        time half (one row, the same for all chunk tokens) is folded into a per-step bias table on the
        host (softhier-mlir@b11cd5f frontend/smolvla_expert.py:27-28, :115).
    """

    prefix_rows: int | None = None   # pad prefix token rows to this (None: true count)
    kv_align: int = 1                # pad expert attention key length to a multiple of this
    kv_proj_per_step: bool = True    # False: cross-attention K/V projection once per call
    time_in_split: bool = False      # True: time half of time_mlp_in as a one-row product
    batch_cameras: bool = False      # True: all cameras in one vision-encoder call (weights read once)
    split_fused: bool = False        # True: expert qkv_proj as q, k, v products and mlp_gateup as gate, up
    #                                  (upstream's deployment plan; stage1-spec §3.3)
    attn_cameras_as_heads: bool = False  # with batch_cameras: vision attention as one call per layer over
    #                                  cameras x heads (head-major (heads, cameras x rows, d) read as
    #                                  (heads x cameras, rows, d) blocks; MHA, so every block is its own head)


def pr1_lowering(prefix_rows: int) -> Lowering:
    return Lowering(prefix_rows=prefix_rows, kv_align=32, kv_proj_per_step=False, time_in_split=True)


@dataclass(frozen=True)
class Dims:
    # vision (SigLIP encoder)
    v_layers: int
    v_hidden: int
    v_heads: int
    v_mlp: int
    patch: int
    shuffle: int
    # connector
    conn_in: int
    # VLM text layers kept
    t_layers: int
    t_hidden: int
    t_heads: int
    t_kv_heads: int
    head_dim: int
    t_mlp: int
    # expert
    e_layers: int
    e_hidden: int
    e_mlp: int
    self_every: int
    chunk: int
    steps: int
    state_dim: int
    action_dim: int
    sources: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def kv_width(self) -> int:
        return self.t_kv_heads * self.head_dim


def _shape(header: dict, name: str) -> list[int]:
    try:
        return header[name]["shape"]
    except KeyError:
        raise KeyError(f"tensor {name!r} not in the checkpoint header") from None


def _count_layers(header: dict, prefix: str) -> int:
    idx = {int(k[len(prefix):].split(".", 1)[0]) for k in header if k.startswith(prefix)}
    return max(idx) + 1 if idx else 0


V = "model.vlm_with_expert.vlm.model.vision_model."
T = "model.vlm_with_expert.vlm.model.text_model."
E = "model.vlm_with_expert.lm_expert."


def load_dims(wl: Workload) -> Dims:
    policy, header = fixtures.load(wl.repo, wl.revision)
    backbone, _ = fixtures.load(*fixtures.BACKBONE)
    if policy.get("vlm_model_name") != fixtures.BACKBONE[0]:
        raise ValueError(f"{wl.repo}: backbone {policy.get('vlm_model_name')} is not the pinned {fixtures.BACKBONE[0]}")
    tc, vc = backbone["text_config"], backbone["vision_config"]
    hd = tc["head_dim"]
    d = Dims(
        v_layers=_count_layers(header, V + "encoder.layers."),
        v_hidden=vc["hidden_size"],
        v_heads=vc["num_attention_heads"],
        v_mlp=_shape(header, V + "encoder.layers.0.mlp.fc1.weight")[0],
        patch=vc["patch_size"],
        shuffle=backbone["scale_factor"],
        conn_in=vc["hidden_size"] * backbone["scale_factor"] ** 2,
        t_layers=policy["num_vlm_layers"],
        t_hidden=tc["hidden_size"],
        t_heads=tc["num_attention_heads"],
        t_kv_heads=tc["num_key_value_heads"],
        head_dim=hd,
        t_mlp=tc["intermediate_size"],
        e_layers=policy["num_expert_layers"] or policy["num_vlm_layers"],
        e_hidden=int(tc["hidden_size"] * policy["expert_width_multiplier"]),
        e_mlp=_shape(header, E + "layers.0.mlp.gate_proj.weight")[0],
        self_every=policy["self_attn_every_n_layers"],
        chunk=policy["chunk_size"],
        steps=policy["num_steps"],
        state_dim=policy["max_state_dim"],
        action_dim=policy["max_action_dim"],
        sources={
            "v_layers": "header (class default, not in backbone config)",
            "v_mlp": "header (class default, not in backbone config)",
            "e_mlp": "header",
            "v_hidden v_heads patch shuffle t_hidden t_heads t_kv_heads head_dim t_mlp": "backbone config.json",
            "t_layers e_layers e_hidden self_every chunk steps state_dim action_dim": "policy config.json",
        },
    )
    check_header(d, header)
    return d


def expected_shapes(d: Dims) -> dict[str, list[int]]:
    """Config-derived shape for every header tensor the model uses, keyed by tensor name (layer 0 / 1)."""
    kv = d.kv_width
    s = {
        V + "embeddings.patch_embedding.weight": [d.v_hidden, 3, d.patch, d.patch],
        V + "encoder.layers.0.self_attn.q_proj.weight": [d.v_hidden, d.v_hidden],
        V + "encoder.layers.0.mlp.fc2.weight": [d.v_hidden, d.v_mlp],
        "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight": [d.t_hidden, d.conn_in],
        T + "layers.0.self_attn.q_proj.weight": [d.t_heads * d.head_dim, d.t_hidden],
        T + "layers.0.self_attn.k_proj.weight": [kv, d.t_hidden],
        T + "layers.0.mlp.gate_proj.weight": [d.t_mlp, d.t_hidden],
        E + "layers.0.self_attn.q_proj.weight": [d.t_heads * d.head_dim, d.e_hidden],
        E + "layers.0.self_attn.k_proj.weight": [kv, d.e_hidden],
        E + "layers.1.self_attn.k_proj.weight": [kv, kv],
        E + "layers.0.self_attn.o_proj.weight": [d.e_hidden, d.t_heads * d.head_dim],
        E + "layers.0.mlp.down_proj.weight": [d.e_hidden, d.e_mlp],
        "model.state_proj.weight": [d.t_hidden, d.state_dim],
        "model.action_in_proj.weight": [d.e_hidden, d.action_dim],
        "model.action_time_mlp_in.weight": [d.e_hidden, 2 * d.e_hidden],
        "model.action_out_proj.weight": [d.action_dim, d.e_hidden],
    }
    return s


def check_header(d: Dims, header: dict) -> None:
    for name, shape in expected_shapes(d).items():
        got = _shape(header, name)
        if list(got) != shape:
            raise ValueError(f"{name}: header shape {got} != config-derived {shape}")
    if _count_layers(header, E + "layers.") != d.e_layers:
        raise ValueError("expert layer count in header differs from the policy config")
    if _count_layers(header, T + "layers.") != d.t_layers:
        raise ValueError("VLM layer count in header differs from num_vlm_layers")


# ---------------------------------------------------------------------------------------------- row builders
def _gemm(stage, layers, op, M, N, K, count, note="") -> OpRow:
    w = FP16 * K * N
    return OpRow(stage, layers, op, "gemm", (M, N, K), count, 2.0 * M * N * K * count,
                 (FP16 * (M * K + M * N) + w) * count, w * count, note=note)


def _attn(stage, layers, op, h, hkv, Sq, Skv, d, count, note="") -> OpRow:
    # QK^T reads Q (h Sq d) and K (hkv Skv d); P.V reads V (hkv Skv d) and writes O (h Sq d).
    # The score matrix is an intermediate and is not counted.
    unique = FP16 * (h * Sq * d + hkv * Skv * d)
    return OpRow(stage, layers, op, "attn", (h, Sq, Skv, d), count, 2.0 * h * Sq * Skv * d * count, unique * count, 0.0,
                 note=note)


def _ew(stage, layers, op, cls, kind, rows, cols, count, n_in=1, n_out=1, weight_elems=0, note="") -> OpRow:
    elems = rows * cols
    w = FP16 * weight_elems
    return OpRow(stage, layers, op, cls, (rows, cols), count, float(EW_FLOP[kind] * elems * count),
                 (FP16 * elems * (n_in + n_out) + w) * count, w * count, note=note)


def _data(stage, layers, op, rows, cols, count, note="") -> OpRow:
    return OpRow(stage, layers, op, "data", (rows, cols), count, 0.0, FP16 * 2 * rows * cols * count, 0.0, note=note)


def _pad(n: int, a: int) -> int:
    return -(-n // a) * a


# ---------------------------------------------------------------------------------------------- the model
def vision_tokens(wl: Workload, d: Dims) -> int:
    return (wl.image_size // d.patch) ** 2


def prefix_tokens(wl: Workload, d: Dims) -> int:
    return wl.cameras * vision_tokens(wl, d) // d.shuffle ** 2 + wl.language_tokens + wl.state_tokens


def build(c: Constants, workload_id: str, lowering: Lowering | None = None) -> list[OpRow]:
    lowering = lowering or Lowering()
    wl = c.workloads[workload_id]
    d = load_dims(wl)
    rows: list[OpRow] = []
    add = rows.append

    # ---- vision: SigLIP encoder, one call per camera as in the reference code (lerobot embed_prefix
    # loops over cameras), so weights are read once per camera. With lowering.batch_cameras the cameras
    # are batched into the token rows of every per-token operator and weights are read once per call.
    # Attention runs per camera either way.
    N, Hv, cam = vision_tokens(wl, d), d.v_hidden, wl.cameras
    k = 1 if lowering.batch_cameras else cam      # encoder calls per policy call
    M = N * cam // k                              # token rows per call
    L = f"0-{d.v_layers - 1}"
    n = d.v_layers * k
    add(_gemm("vision", "embed", "patch_embed", M, Hv, 3 * d.patch * d.patch, k))
    add(_ew("vision", "embed", "bias_patch", "elementwise", "bias", M, Hv, k, weight_elems=Hv))
    add(_ew("vision", "embed", "pos_embed_add", "elementwise", "add", M, Hv, k, weight_elems=N * Hv))
    add(_ew("vision", L, "ln1", "norm", "layernorm", M, Hv, n, weight_elems=2 * Hv))
    for p in ("q", "k", "v"):
        add(_gemm("vision", L, f"{p}_proj", M, Hv, Hv, n))
        add(_ew("vision", L, f"bias_{p}", "elementwise", "bias", M, Hv, n, weight_elems=Hv))
    add(_data("vision", L, "transpose_k", M, Hv, n))
    hd_v = Hv // d.v_heads
    if lowering.attn_cameras_as_heads:
        if not lowering.batch_cameras:
            raise ValueError("attn_cameras_as_heads needs batch_cameras")
        hv, na, note = d.v_heads * cam, d.v_layers, "cameras x heads in one call"
    else:
        hv, na, note = d.v_heads, d.v_layers * cam, "per camera"
    add(_attn("vision", L, "qk", hv, hv, N, N, hd_v, na, note=note))
    add(_ew("vision", L, "softmax", "softmax", "softmax", hv * N, N, na))
    add(_attn("vision", L, "pv", hv, hv, N, N, hd_v, na, note=note))
    add(_gemm("vision", L, "o_proj", M, Hv, Hv, n))
    add(_ew("vision", L, "bias_o", "elementwise", "bias", M, Hv, n, weight_elems=Hv))
    add(_ew("vision", L, "residual_attn", "elementwise", "add", M, Hv, n, n_in=2))
    add(_ew("vision", L, "ln2", "norm", "layernorm", M, Hv, n, weight_elems=2 * Hv))
    add(_gemm("vision", L, "mlp_up", M, d.v_mlp, Hv, n))
    add(_ew("vision", L, "bias_up", "elementwise", "bias", M, d.v_mlp, n, weight_elems=d.v_mlp))
    add(_ew("vision", L, "gelu", "acti", "gelu", M, d.v_mlp, n))
    add(_gemm("vision", L, "mlp_down", M, Hv, d.v_mlp, n))
    add(_ew("vision", L, "bias_down", "elementwise", "bias", M, Hv, n, weight_elems=Hv))
    add(_ew("vision", L, "residual_mlp", "elementwise", "add", M, Hv, n, n_in=2))
    add(_ew("vision", "post", "ln_post", "norm", "layernorm", M, Hv, k, weight_elems=2 * Hv))

    # ---- connector: pixel shuffle then one projection, all cameras together
    img_tok = cam * N // d.shuffle ** 2
    add(_data("connector", "-", "pixel_shuffle", cam * N, Hv, 1))
    add(_gemm("connector", "-", "connector_proj", img_tok, d.t_hidden, d.conn_in, 1))
    add(_ew("connector", "-", "embed_scale", "elementwise", "scale", img_tok, d.t_hidden, 1))

    # ---- prefix: image + language + state tokens through the kept VLM layers (KV cache kept)
    P = prefix_tokens(wl, d)
    Pr = lowering.prefix_rows or P
    if Pr < P:
        raise ValueError(f"prefix_rows {Pr} < prefix tokens {P}")
    Ht, kv, hd = d.t_hidden, d.kv_width, d.head_dim
    L = f"0-{d.t_layers - 1}"
    n = d.t_layers
    add(_ew("prefix", "embed", "lang_embed_scale", "elementwise", "scale", wl.language_tokens, Ht, 1,
            weight_elems=wl.language_tokens * Ht, note="embedding lookup + scale; host-side in PR1"))
    add(_gemm("prefix", "embed", "state_proj", wl.state_tokens, Ht, d.state_dim, 1, note="host-side in PR1"))
    add(_ew("prefix", "embed", "bias_state", "elementwise", "bias", wl.state_tokens, Ht, 1, weight_elems=Ht))
    add(_ew("prefix", L, "rmsnorm_attn", "norm", "rmsnorm", Pr, Ht, n, weight_elems=Ht))
    add(_gemm("prefix", L, "q_proj", Pr, d.t_heads * hd, Ht, n))
    add(_gemm("prefix", L, "k_proj", Pr, kv, Ht, n))
    add(_gemm("prefix", L, "v_proj", Pr, kv, Ht, n))
    add(_ew("prefix", L, "rope_q", "elementwise", "rope", Pr, d.t_heads * hd, n))
    add(_ew("prefix", L, "rope_k", "elementwise", "rope", Pr, kv, n))
    add(_attn("prefix", L, "qk", d.t_heads, d.t_kv_heads, Pr, Pr, hd, n))
    add(_ew("prefix", L, "softmax", "softmax", "softmax", d.t_heads * Pr, Pr, n))
    add(_attn("prefix", L, "pv", d.t_heads, d.t_kv_heads, Pr, Pr, hd, n))
    add(_gemm("prefix", L, "o_proj", Pr, Ht, d.t_heads * hd, n))
    add(_ew("prefix", L, "residual_attn", "elementwise", "add", Pr, Ht, n, n_in=2))
    add(_ew("prefix", L, "rmsnorm_mlp", "norm", "rmsnorm", Pr, Ht, n, weight_elems=Ht))
    add(_gemm("prefix", L, "mlp_gate", Pr, d.t_mlp, Ht, n))
    add(_gemm("prefix", L, "mlp_up", Pr, d.t_mlp, Ht, n))
    add(_ew("prefix", L, "silu_mul", "acti", "silu_mul", Pr, d.t_mlp, n, n_in=2))
    add(_gemm("prefix", L, "mlp_down", Pr, Ht, d.t_mlp, n))
    add(_ew("prefix", L, "residual_mlp", "elementwise", "add", Pr, Ht, n, n_in=2))
    add(_ew("prefix", "tail", "rmsnorm_final", "norm", "rmsnorm", Pr, Ht, 1, weight_elems=Ht,
            note="output unused (only the KV cache is kept); PR1 runs it"))

    # ---- expert: flow-matching transformer, `steps` Euler steps over a `chunk`-token action chunk
    S, He, Q = d.steps, d.e_hidden, d.chunk
    n_self = len(range(0, d.e_layers, d.self_every))
    n_cross = d.e_layers - n_self
    Ls, Lc, La = f"self (0,{d.self_every}..{d.e_layers - 2})", f"cross (1,3..{d.e_layers - 1})", f"0-{d.e_layers - 1}"
    kv_steps = S if lowering.kv_proj_per_step else 1
    for p in ("k", "v"):
        add(_gemm("expert", Lc, f"cross_{p}_proj", P, kv, kv, n_cross * kv_steps,
                  note="prefix K/V projected once per call" if kv_steps == 1 else "inside every step (reference code)"))
    add(_gemm("expert", "embed", "action_in_proj", Q, He, d.action_dim, S))
    add(_ew("expert", "embed", "bias_action_in", "elementwise", "bias", Q, He, S, weight_elems=He))
    if lowering.time_in_split:
        add(_gemm("expert", "embed", "time_mlp_in", Q, He, He, S, note="action half"))
        add(_gemm("expert", "embed", "time_mlp_in_time", 1, He, He, S, note="time half, one row; host-side in PR1"))
    else:
        add(_gemm("expert", "embed", "time_mlp_in", Q, He, 2 * He, S))
    add(_ew("expert", "embed", "bias_time_in", "elementwise", "bias", Q, He, S, weight_elems=He))
    add(_ew("expert", "embed", "time_silu", "acti", "silu", Q, He, S))
    add(_gemm("expert", "embed", "time_mlp_out", Q, He, He, S))
    add(_ew("expert", "embed", "bias_time_out", "elementwise", "bias", Q, He, S, weight_elems=He))
    add(_ew("expert", La, "rmsnorm_attn", "norm", "rmsnorm", Q, He, d.e_layers * S, weight_elems=He))
    qw = d.t_heads * hd
    if lowering.split_fused:
        add(_gemm("expert", Ls, "self_q_proj", Q, qw, He, n_self * S))
        add(_gemm("expert", Ls, "self_k_proj", Q, kv, He, n_self * S))
        add(_gemm("expert", Ls, "self_v_proj", Q, kv, He, n_self * S))
    else:
        add(_gemm("expert", Ls, "qkv_proj", Q, qw + 2 * kv, He, n_self * S))
    add(_gemm("expert", Lc, "q_proj", Q, qw, He, n_cross * S))
    add(_ew("expert", La, "rope_q", "elementwise", "rope", Q, qw, d.e_layers * S))
    add(_ew("expert", Ls, "rope_k", "elementwise", "rope", Q, kv, n_self * S))
    skv_self = _pad(P + Q, lowering.kv_align)
    skv_cross = _pad(P, lowering.kv_align)
    for op in ("qk", "pv"):
        add(_attn("expert", Ls, f"self_{op}", d.t_heads, d.t_kv_heads, Q, skv_self, hd, n_self * S,
                  note="keys: prefix cache + own action tokens"))
    add(_ew("expert", Ls, "self_softmax", "softmax", "softmax", d.t_heads * Q, skv_self, n_self * S))
    for op in ("qk", "pv"):
        add(_attn("expert", Lc, f"cross_{op}", d.t_heads, d.t_kv_heads, Q, skv_cross, hd, n_cross * S,
                  note="keys: projected prefix cache"))
    add(_ew("expert", Lc, "cross_softmax", "softmax", "softmax", d.t_heads * Q, skv_cross, n_cross * S))
    add(_gemm("expert", La, "o_proj", Q, He, qw, d.e_layers * S))
    add(_ew("expert", La, "residual_attn", "elementwise", "add", Q, He, d.e_layers * S, n_in=2))
    add(_ew("expert", La, "rmsnorm_mlp", "norm", "rmsnorm", Q, He, d.e_layers * S, weight_elems=He))
    if lowering.split_fused:
        add(_gemm("expert", La, "mlp_gate", Q, d.e_mlp, He, d.e_layers * S))
        add(_gemm("expert", La, "mlp_up", Q, d.e_mlp, He, d.e_layers * S))
    else:
        add(_gemm("expert", La, "mlp_gateup", Q, 2 * d.e_mlp, He, d.e_layers * S))
    add(_ew("expert", La, "silu_mul", "acti", "silu_mul", Q, d.e_mlp, d.e_layers * S, n_in=2))
    add(_gemm("expert", La, "mlp_down", Q, He, d.e_mlp, d.e_layers * S))
    add(_ew("expert", La, "residual_mlp", "elementwise", "add", Q, He, d.e_layers * S, n_in=2))
    add(_ew("expert", "out", "rmsnorm_final", "norm", "rmsnorm", Q, He, S, weight_elems=He))
    add(_gemm("expert", "out", "action_out_proj", Q, d.action_dim, He, S))
    add(_ew("expert", "out", "bias_action_out", "elementwise", "bias", Q, d.action_dim, S, weight_elems=d.action_dim))
    add(_ew("expert", "out", "euler_update", "elementwise", "axpy", Q, d.action_dim, S, n_in=2))
    return rows


# ---------------------------------------------------------------------------------------------- units and kernels
# Where each operator runs in PR1's compiled path (softhier-mlir b11cd5f) and which hand-written kernel of the
# kernel repo covers it: ~/softhier@52859c3 implementation/sw/, plus GELU and LayerNorm added on branch
# gelu-layernorm-kernels (HaozeG/gvsoc@580ac85). Code references: docs/analysis/compiler-walkthrough.md.
CORES = "Snitch cores (scalar; Spatz off in PR1)"
HOST = "host (PR1 prepare step)"
HOST_OPS = {"lang_embed_scale", "state_proj", "bias_state", "time_mlp_in_time"}
KERNEL_BY_OP = {
    "softmax": "FlatAttention (part of the QK^T call)",
    "self_softmax": "FlatAttention (part of the QK^T call)",
    "cross_softmax": "FlatAttention (part of the QK^T call)",
    "rmsnorm_attn": "RMSNorm", "rmsnorm_mlp": "RMSNorm", "rmsnorm_final": "RMSNorm",
    "silu_mul": "Activation (SiLU + gate)",
    "time_silu": "Activation (SiLU)",
    "gelu": "Activation (GELU, tanh form)",
    "ln1": "LayerNorm", "ln2": "LayerNorm", "ln_post": "LayerNorm",
}

# Rows whose work runs inside another row's kernel call: FlatAttention does QK^T, softmax and P.V in one
# call (time on the qk row) and transposes K on chip with the transpose engine.
INCLUDED_IN = {
    ("vision", "softmax"): "qk", ("vision", "pv"): "qk", ("vision", "transpose_k"): "qk",
    ("prefix", "softmax"): "qk", ("prefix", "pv"): "qk",
    ("expert", "self_softmax"): "self_qk", ("expert", "self_pv"): "self_qk",
    ("expert", "cross_softmax"): "cross_qk", ("expert", "cross_pv"): "cross_qk",
}
ATTN_CALL_OPS = {"qk", "self_qk", "cross_qk"}


def eltwise_kind(op: str) -> str | None:
    """Operator of the Eltwise kernel that implements an elementwise or data row."""
    if op.startswith("bias_"):
        return "bias"
    if op.startswith("residual_") or op == "pos_embed_add":
        return "add"
    if op.startswith("rope_"):
        return "rope"
    if op == "embed_scale":
        return "scale"
    if op == "euler_update":
        return "axpy"
    if op == "pixel_shuffle":
        return "pixel_shuffle"
    return None


# Row operators by kernel name in the constants key (h1_kernels).
ROW_KERNEL = {
    "ln1": "layernorm", "ln2": "layernorm", "ln_post": "layernorm", "gelu": "gelu",
    "rmsnorm_attn": "rmsnorm", "rmsnorm_mlp": "rmsnorm", "rmsnorm_final": "rmsnorm",
    "silu_mul": "silu_gate", "time_silu": "silu",
}


@dataclass(frozen=True)
class KernelSet:
    """Where a workload's hand-kernel times live: constants group and the arch section they ran on."""
    group: str
    arch: str
    lowering: Lowering


# Stage-1 lowering (docs/designs/stage1-spec.md §3.3): reference computation plus cross K/V once per call.
STAGE1_LOWERING = Lowering(kv_proj_per_step=False)
STAGE1_LOWERING_W1B = Lowering(kv_proj_per_step=False, batch_cameras=True)
# The stage-1 baseline (decision 2026-11-25): upstream's deployment plan, which has separate q / k / v and
# gate / up products (deploy_plan.Plan); its kernel times are the h1_plan group (plan_runs.py, to_constants.py).
PLAN_LOWERING = Lowering(kv_proj_per_step=False, split_fused=True)
# W1-b under the same plan: cameras batched through the vision encoder, attention over cameras x heads
PLAN_LOWERING_W1B = Lowering(kv_proj_per_step=False, split_fused=True, batch_cameras=True,
                             attn_cameras_as_heads=True)
KERNEL_SETS = {
    "vlabench-3cam": KernelSet("h1_plan", "h1", PLAN_LOWERING),  # H1, upstream deployment plan (stage-1 spec §4)
}
# Every page number comes from runs under the deployment plan: group -> (lowering, selection). "upstream": upstream's
# kernel-selection rule (plan_runs.py, the baseline); "constrained": the plan-constrained sweep (plan_sweep.py: the
# plan's placement, reshapes and head padding kept, free knobs searched, attention padding chosen for the graph).
PLAN_GROUPS = {
    "h1_plan": (PLAN_LOWERING, "upstream"),
    "h1_psweep": (PLAN_LOWERING, "constrained"),
    "h1_plan_w1b": (PLAN_LOWERING_W1B, "upstream"),
    "h1_psweep_w1b": (PLAN_LOWERING_W1B, "constrained"),
}
# The sweep preset (per kernel shape the fastest checked run over the stage-1 sweeps, placement per kernel, inter-
# kernel layout free): not a deployable plan; kept for the cost-model fit and history, for the unsplit stage-1
# lowerings only. No page reports it as performance.
SWEEP_GROUP = "h1_kernels"


def kernel_set(workload_id: str, lowering: Lowering | None = None, selection: str = "upstream") -> KernelSet:
    """Kernel times of a workload under a lowering (default: the workload's own) and selection ("upstream" rule or
    "constrained" sweep); KeyError for workloads without simulated kernels (pr1-1cam, libero-2cam, ...)."""
    if workload_id not in KERNEL_SETS:
        raise KeyError(f"no hand-kernel times for workload {workload_id!r}; kernel sets: {sorted(KERNEL_SETS)}")
    ks = KERNEL_SETS[workload_id]
    low = lowering or ks.lowering
    for g, (l, sel) in PLAN_GROUPS.items():
        if l == low and sel == selection:
            return KernelSet(g, ks.arch, low)
    if low.split_fused:
        raise KeyError(f"no plan group for lowering {low} / {selection}")
    return KernelSet(SWEEP_GROUP, ks.arch, low)        # unsplit stage-1 lowerings: the sweep preset (history)


@functools.lru_cache(maxsize=None)
def plan_keys(workload_id: str = "vlabench-3cam", lowering: Lowering | None = None) -> dict[tuple[str, str], str]:
    """(stage, op) -> run key of the upstream deployment plan (deploy_plan.Plan), e.g. gemm_1024x768x768_zto_pZs."""
    from .deploy_plan import Plan          # deploy_plan imports this module
    return {(d.stage, d.op): d.key for d in Plan(workload_id, lowering=lowering).deploys}


def hand_kernel_key(row: OpRow, group: str = "h1_kernels") -> str | None:
    """constants key of the row's simulated kernel: <group>.<stage>.<plan run key> for the deployment-plan groups
    (PLAN_GROUPS), or <group>.<stage>.<kernel>_<row shape> for the sweep preset."""
    if row.op in HOST_OPS or (row.stage, row.op) in INCLUDED_IN:
        return None
    if group in PLAN_GROUPS:
        k = plan_keys("vlabench-3cam", PLAN_GROUPS[group][0]).get((row.stage, row.op))
        return f"{group}.{row.stage}.{k}" if k else None
    shape = "x".join(map(str, row.shape))
    if row.cls == "gemm":
        return f"{group}.{row.stage}.gemm_{shape}"
    if row.op in ATTN_CALL_OPS:
        return f"{group}.{row.stage}.flatattention_{shape}"
    if row.op in ROW_KERNEL:
        return f"{group}.{row.stage}.{ROW_KERNEL[row.op]}_{shape}"
    kind = eltwise_kind(row.op)
    return f"{group}.{row.stage}.eltwise_{kind}_{shape}" if kind else None


def hand_kernel_ns(c, workload_id: str, row: OpRow, group: str | None = None) -> float | None:
    """Measured ns per call of the row's hand kernel, if one was simulated at this workload and shape. group: the
    constants group (kernel_set(workload, lowering).group); default the workload's own."""
    if workload_id not in KERNEL_SETS:
        return None
    group = group or kernel_set(workload_id).group
    key = hand_kernel_key(row, group)
    if key is None or key not in c.entries:
        return None
    e = c.get(key)
    if e.workload != workload_id:
        return None
    if group == SWEEP_GROUP and key.rsplit("_", 1)[1] != "x".join(map(str, row.shape)):
        return None
    return float(e.value)


SPATZ = "Spatz on 4 cores, data by iDMA"


def unit_and_kernel(row: OpRow) -> tuple[str, str, str]:
    """(unit in PR1's compiled path, stage-1 hand-written kernel, units that kernel uses for this operator).

    Kernel units: the busy unit tracks (redmule, idma, pe<n>/spatz) in the H1 kernel traces
    (~/.cache/softhier-vla/stage1-h1/traces/); NoC broadcast and the transpose engine have no track and come
    from the kernel source (SummaGEMM, FlatAttention). PR1 units: compiler-walkthrough.md and arch.pr1 (no Spatz)."""
    if row.op in HOST_OPS:
        return HOST, "none", ""
    if row.cls == "gemm":
        return "RedMulE (operands moved by iDMA)", "SummaGEMM", "RedMulE; operands by iDMA and NoC (SUMMA broadcast)"
    if row.cls == "attn":
        mask = "" if row.stage == "vision" else "; masking not supported"
        return "RedMulE", f"FlatAttention{mask}", "RedMulE (QK^T, P.V), Spatz (softmax); data by iDMA"
    if row.op == "transpose_k":
        return "Snitch cores and iDMA", "FlatAttention (part of the QK^T call)", "transpose engine, in L1"
    kind = eltwise_kind(row.op)
    if kind == "pixel_shuffle":
        return "Snitch cores and iDMA", "Eltwise (pixel_shuffle)", "iDMA only (one 2D copy per output row)"
    if kind:
        return CORES, f"Eltwise ({kind})", SPATZ
    if row.cls == "data":
        return "Snitch cores and iDMA", "none", ""
    if row.cls == "softmax":
        return CORES, KERNEL_BY_OP.get(row.op, "none"), "Spatz (vfredmax, vfexp), inside FlatAttention"
    kernel = KERNEL_BY_OP.get(row.op, "none")
    if kernel == "RMSNorm":
        return CORES, kernel, "Spatz on 1 core (upstream kernel), data by iDMA"
    return CORES, kernel, SPATZ if kernel != "none" else ""


def annotate(rows: list[OpRow]) -> list[OpRow]:
    """Fill pr1_unit, hand_kernel and hand_kernel_units."""
    out = []
    for r in rows:
        unit, kernel, kunits = unit_and_kernel(r)
        out.append(replace(r, pr1_unit=unit, hand_kernel=kernel, hand_kernel_units=kunits))
    return out


def stage_totals(rows: list[OpRow]) -> dict[str, StageTotals]:
    out = {}
    for s in STAGES:
        rs = [r for r in rows if r.stage == s]
        out[s] = StageTotals(
            flop=sum(r.flop for r in rs),
            matmul_flop=sum(r.flop for r in rs if r.is_matmul),
            memory_traffic_bytes=sum(r.memory_traffic_bytes for r in rs),
            weights_read_bytes=sum(r.weights_read_bytes for r in rs),
        )
    return out


# Header tensor name prefix -> model part (wiki://resources/models/smolvla.md per-part table).
PARTS: tuple[tuple[str, str], ...] = (
    (V, "vision"),
    ("model.vlm_with_expert.vlm.model.connector.", "connector"),
    (T + "embed_tokens.", "token_embedding"),
    (T, "llm"),
    (E, "expert"),
    ("model.vlm_with_expert.vlm.lm_head.", "lm_head"),
    ("model.action_", "io"),
    ("model.state_proj.", "io"),
)
NOT_CALLED = {"lm_head"}   # stored but never called (wiki://resources/models/smolvla.md)


def part_of(name: str) -> str:
    for prefix, part in PARTS:
        if name.startswith(prefix):
            return part
    raise KeyError(f"header tensor {name!r} belongs to no known part")


def param_counts(header: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, t in header.items():
        part = part_of(name)
        out[part] = out.get(part, 0) + math.prod(t["shape"])
    return out
