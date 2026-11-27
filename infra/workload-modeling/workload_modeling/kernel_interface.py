"""Kernel parameter interfaces, parsed from the kernel sources, and the parameters each run used.

    python -m workload_modeling.kernel_interface              # print the interfaces of our kernel tree
    python -m workload_modeling.kernel_interface --freeze DIR # parse another tree (e.g. an upstream clone) to JSON

Convention the kernels follow (descriptor style): a config template config/kernels/<cfg>.py sets fields
(self.<field> = default); scripts/kernels/<cfg>_config.py writes them to the generated header <hdr>.h as macros
<PREFIX>_<FIELD>; sw/<Kernel>/main.c passes them to the kernel entry call as `<PREFIX>_<FIELD>/*name*/`, and the kernel
headers test compile-time switches with #if / #ifdef. The parser reads every <PREFIX>_* token the kernel C code uses
(main.c and include/*.h, comments stripped) and classifies it by use:
  descriptor: passed to the entry call with a /*name*/ comment;
  switch:     tested in #if / #ifdef / #ifndef / #elif;
  value:      used in any other expression.
What a parameter means (operator, mapping, schedule, fusion, check, ...) is not in the code: ANNOT below says it, and
tests/test_kernel_interface.py fails if a parsed parameter has no entry or an entry matches no parsed parameter.
The parameters a run used are read from the generated header kept in its run folder (ground truth of what ran).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KREPO = REPO / "third_party" / "softhier-kernels" / "implementation"
UPSTREAM_JSON = REPO / "infra" / "kernel-runs" / "results" / "kernel_interface_upstream.json"

# kernel directory -> (macro prefix, generated header, config template)
KERNELS = {
    "SummaGEMM": ("GEMM", "gemm.h", "gemm"),
    "FlatAttention": ("ATTN", "attn.h", "attn"),
    "Activation": ("ACTI", "acti.h", "acti"),
    "Eltwise": ("ELT", "elt.h", "elt"),
    "LayerNorm": ("LNORM", "lnorm.h", "lnorm"),
    "RMSNorm": ("NORM", "norm.h", "norm"),
    "RoPE": ("ROPE", "rope.h", "rope"),
}

# What each parameter is. kind: shape (operator size), operator (which computation), mapping (how the work is split
# over clusters), schedule (order / overlap inside a cluster), fusion (extra operator folded in), layout (data
# arrangement in HBM), precision, check (output verification only). "dep": meaningful only when that switch is on. "field": the template field it comes from when
# that is not <prefix>_<FIELD> (derived in <cfg>_config.py).
ANNOT: dict[str, dict[str, dict]] = {
    "SummaGEMM": {
        "GEMM_M_SIZE": dict(kind="shape", note="padded M"), "GEMM_N_SIZE": dict(kind="shape", note="padded N"),
        "GEMM_K_SIZE": dict(kind="shape", note="padded K"),
        "GEMM_M_TILE": dict(kind="mapping", note="output tile rows per cluster"),
        "GEMM_N_TILE": dict(kind="mapping", note="output tile columns per cluster"),
        "GEMM_K_TILE": dict(kind="mapping", note="reduction step per RedMulE call"),
        "GEMM_SUMMA_SCALE_X": dict(kind="mapping", note="SUMMA group width in clusters"),
        "GEMM_SUMMA_SCALE_Y": dict(kind="mapping", note="SUMMA group height; non-square with M_iter > 1 is wrong "
                                   "(SummaGEMM.h:165)"),
        "GEMM_SUMMA_GROUP_NUMBER": dict(kind="mapping", note="number of SUMMA groups"),
        "GEMM_SUMMA_GROUP_REDUCE": dict(kind="mapping", dep="GEMM_SUMMA_GROUP_SPLITK",
                                        note="NoC reduction of group partial sums (with splitK)"),
        "GEMM_SUMMA_GROUP_SPLITK": dict(kind="mapping", note="groups own K blocks; correct only with 64 active "
                                        "clusters and N_iter <= 2 (wrong-output-runs-h1.md)"),
        "GEMM_SUMMA_GROUP_SPLITN": dict(kind="mapping", note="groups own N blocks"),
        "GEMM_SUMMA_GROUP_GAP_X": dict(kind="mapping", dep="GEMM_SUMMA_GROUP_SPLITK",
                                       note="X address offset between groups (bytes; splitK)"),
        "GEMM_SUMMA_GROUP_GAP_W": dict(kind="mapping", note="W address offset between groups (bytes)"),
        "GEMM_SUMMA_GROUP_GAP_Z": dict(kind="mapping", note="Z address offset between groups (bytes)"),
        "GEMM_RESHA_X_FROM_ENABLE": dict(kind="layout", note="reshape X on load"),
        "GEMM_RESHA_X_FROM_M": dict(kind="layout", dep="GEMM_RESHA_X_FROM_ENABLE", note="row count of the X reshape"),
        "GEMM_RESHA_X_FROM_TALL": dict(kind="layout", dep="GEMM_RESHA_X_FROM_ENABLE", note="X reshape direction", field="resha_x_from_m"),
        "GEMM_RESHA_X_FROM_THIN": dict(kind="layout", dep="GEMM_RESHA_X_FROM_ENABLE", note="X reshape direction", field="resha_x_from_m"),
        "GEMM_RESHA_Z_TO_ENABLE": dict(kind="layout", note="reshape Z on store"),
        "GEMM_RESHA_Z_TO_M": dict(kind="layout", dep="GEMM_RESHA_Z_TO_ENABLE", note="row count of the Z reshape"),
        "GEMM_RESHA_Z_TO_TALL": dict(kind="layout", dep="GEMM_RESHA_Z_TO_ENABLE", note="Z reshape direction", field="resha_z_to_m"),
        "GEMM_RESHA_Z_TO_THIN": dict(kind="layout", dep="GEMM_RESHA_Z_TO_ENABLE", note="Z reshape direction", field="resha_z_to_m"),
        "GEMM_CHECK_ABS_TOL": dict(kind="check", note="splitK output check tolerance", field="check_abs_tol"),
        "GEMM_CHECK_TOL_ULP": dict(kind="check", note="output check tolerance (ULP)", field=None),
        "GEMM_M_VALID": dict(kind="check", note="unpadded M: the check reads only the valid region", field="m_valid"),
        "GEMM_N_VALID": dict(kind="check", note="unpadded N: the check reads only the valid region", field="n_valid"),
        "GEMM_RESHAPE_X_FROM_K": dict(kind="layout", dep="GEMM_RESHA_X_FROM_ENABLE", note="K of the X reshape (derived in gemm_config.py)",
                                      field="resha_x_from_m"),
        "GEMM_RESHAPE_Z_TO_N": dict(kind="layout", dep="GEMM_RESHA_Z_TO_ENABLE", note="N of the Z reshape (derived in gemm_config.py)",
                                    field="resha_z_to_m"),
    },
    "FlatAttention": {
        "ATTN_KV_SEQUENCE_LENGTH": dict(kind="shape", note="padded keys"),
        "ATTN_Q_SEQUENCE_LENGTH": dict(kind="shape", note="padded queries"),
        "ATTN_SPECULATIVE_LENGTH": dict(kind="shape", note="speculative tokens per query"),
        "ATTN_HEAD_DIMEMSION": dict(kind="shape", note="head dimension"),
        "ATTN_NUM_HEAD": dict(kind="shape", note="query heads"),
        "ATTN_NUM_HEAD_GROUP": dict(kind="shape", note="KV heads (GQA)"),
        "ATTN_BATCH_SIZE": dict(kind="shape", note="batch"),
        "ATTN_FLATTEN_SCALE_X": dict(kind="mapping", note="group width in clusters (KV direction)"),
        "ATTN_FLATTEN_SCALE_Y": dict(kind="mapping", note="group height in clusters (query direction)"),
        "ATTN_FLATTEN_SHAPE_X": dict(kind="mapping", note="block Bc"),
        "ATTN_FLATTEN_SHAPE_Y": dict(kind="mapping", note="block Br"),
        "ATTN_FLATTEN_ASYNC": dict(kind="schedule", note="asynchronous overlap of heads (even head count per group)"),
        "ATTN_FP16": dict(kind="precision", note="fp16 data", field="dtype"),
        "ATTN_FP8": dict(kind="precision", note="fp8 data", field="dtype"),
        "ATTN_CHECK_ABS_TOL": dict(kind="check", note="output check tolerance", field=None),
    },
    "Activation": {
        "ACTI_M_SIZE": dict(kind="shape", note="rows"), "ACTI_N_SIZE": dict(kind="shape", note="columns"),
        "ACTI_GATE_ENABLE": dict(kind="operator", note="multiply by a gate input (SwiGLU: silu(a) * b)"),
        "ACTI_BIAS_ENABLE": dict(kind="fusion", note="add a bias row before the activation"),
        "ACTI_ALGO_SIGMOID": dict(kind="operator", note="activation function", field="algo"),
        "ACTI_ALGO_RELU": dict(kind="operator", note="activation function", field="algo"),
        "ACTI_ALGO_SILU": dict(kind="operator", note="activation function", field="algo"),
        "ACTI_ALGO_GELU": dict(kind="operator", note="activation function (tanh form)", field="algo"),
        "ACTI_FP16": dict(kind="precision", note="fp16 data", field="dtype"),
        "ACTI_FP8": dict(kind="precision", note="fp8 data", field="dtype"),
        "ACTI_CHECK_TOL_ULP": dict(kind="check", note="output check tolerance", field=None),
    },
    "Eltwise": {
        "ELT_M_SIZE": dict(kind="shape", note="rows"), "ELT_N_SIZE": dict(kind="shape", note="columns"),
        "ELT_OP_ADD": dict(kind="operator", note="elementwise op", field="op"),
        "ELT_OP_AXPY": dict(kind="operator", note="elementwise op", field="op"),
        "ELT_OP_BIAS": dict(kind="operator", note="elementwise op", field="op"),
        "ELT_OP_PIXEL_SHUFFLE": dict(kind="operator", note="elementwise op", field="op"),
        "ELT_OP_ROPE": dict(kind="operator", note="elementwise op", field="op"),
        "ELT_OP_SCALE": dict(kind="operator", note="elementwise op", field="op"),
        "ELT_ALPHA_H": dict(kind="operator", note="scalar of axpy / scale, as fp16 bits (derived in elt_config.py)",
                            field="alpha"),
        "ELT_FP16": dict(kind="precision", note="fp16 data", field="dtype"),
        "ELT_HEAD_DIM": dict(kind="shape", note="rope head dimension"),
        "ELT_GRID": dict(kind="shape", note="pixel-shuffle grid"),
        "ELT_PIX_SCALE": dict(kind="shape", note="pixel-shuffle factor"),
        "ELT_PIPELINE": dict(kind="schedule", note="pipelined rows: store j-1 and load j+1 while computing j",
                             field="pipeline"),
        "ELT_CHECK_TOL_ULP": dict(kind="check", note="output check tolerance", field=None),
    },
    "LayerNorm": {
        "LNORM_M_SIZE": dict(kind="shape", note="rows"), "LNORM_N_SIZE": dict(kind="shape", note="columns"),
        "LNORM_EPS_F": dict(kind="operator", note="epsilon (derived in lnorm_config.py)", field="eps"),
        "LNORM_CHECK_TOL_ULP": dict(kind="check", note="output check tolerance", field=None),
    },
    "RMSNorm": {
        "NORM_M_SIZE": dict(kind="shape", note="rows"), "NORM_N_SIZE": dict(kind="shape", note="columns"),
        "NORM_FP16": dict(kind="precision", note="fp16 data", field="dtype"),
        "NORM_FP8": dict(kind="precision", note="fp8 data", field="dtype"),
        "NORM_CHECK_TOL_ULP": dict(kind="check", note="output check tolerance", field=None),
    },
    "RoPE": {
        "ROPE_M_SIZE": dict(kind="shape", note="tokens (rows per head in view mode)"),
        "ROPE_N_SIZE": dict(kind="shape", note="values per token (heads x head dimension)"),
        "ROPE_VIEW_ENABLE": dict(kind="layout", note="head-major data (heads, rows, view_n), as attention reads it"),
        "ROPE_VIEW_N": dict(kind="layout", dep="ROPE_VIEW_ENABLE", note="head dimension of the head-major view"),
        "ROPE_FP16": dict(kind="precision", note="fp16 data", field="dtype"),
        "ROPE_CHECK_TOL_ULP": dict(kind="check", note="output check tolerance"),
    },
}
# upstream parameters our kernels do not have (frozen interface, kernel_interface_upstream.json)
ANNOT_UPSTREAM: dict[str, dict[str, dict]] = {
    "SummaGEMM": {"GEMM_SUMMA_NUMER": dict(kind="check", note="output check on / off"),
                  "GEMM_SUMMA_NUMER_CHUNK": dict(kind="check", note="output check chunk size")},
    "FlatAttention": {"ATTN_FLATTEN_NUMER": dict(kind="check", note="output check on / off"),
                      "ATTN_FLATTEN_NUMER_CHUNK": dict(kind="check", note="output check chunk size"),
                      "ATTN_FLATTEN_SWCOLL": dict(kind="schedule", note="software collectives for the row broadcast "
                                                  "and reductions (SoftwareCollectives.h; e8 vector ops, fp8)"),
                      "ATTN_FLATTEN_SWCOLL_PIPELINE": dict(kind="schedule", note="pipelined software collectives")},
    "Activation": {"ACTI_ACTI_NUMER": dict(kind="check", note="output check on / off"),
                   "ACTI_ACTI_NUMER_CHUNK": dict(kind="check", note="output check chunk size")},
    "RMSNorm": {"NORM_NORM_NUMER": dict(kind="check", note="output check on / off"),
                "NORM_NORM_NUMER_CHUNK": dict(kind="check", note="output check chunk size")},
    "RoPE": {"ROPE_ROPE_NUMER": dict(kind="check", note="output dump on / off (our port checks on chip instead)"),
             "ROPE_ROPE_NUMER_CHUNK": dict(kind="check", note="output dump chunk size")},
}
TUNABLE = ("mapping", "schedule", "fusion", "layout")      # kinds a mapping choice can change ("never swept" applies)

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _strip(text: str) -> str:
    """Comments removed, line breaks kept (so line numbers stay right)."""
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def parse_kernel(root: Path, kern: str) -> dict:
    """{macro: {uses: {descriptor, switch, value}, name, where: [file:line], defined_default}} for one kernel."""
    prefix = KERNELS[kern][0]
    tok = re.compile(rf"\b{prefix}_[A-Z0-9_]+\b")
    desc = re.compile(rf"\b({prefix}_[A-Z0-9_]+)\s*/\*\s*(\w+)\s*\*/")
    d = root / "sw" / kern
    gen = {KERNELS[kern][1], "preload.h"}             # generated by the build (git-ignored), not kernel source
    files = [d / "main.c"] + sorted(f for f in (d / "include").glob("*.h") if f.name not in gen)
    out: dict[str, dict] = {}

    def add(m, use, where, name=None):
        e = out.setdefault(m, dict(uses=set(), name=None, where=[], defined_default=None))
        e["uses"].add(use)
        e["where"].append(where)
        if name:
            e["name"] = name

    for f in files:
        if not f.exists():
            continue
        raw = f.read_text(errors="ignore")
        rel = f.relative_to(root)
        for m in desc.finditer(raw):                       # the /*name*/ comments are the descriptor convention
            add(m.group(1), "descriptor", f"{rel}:{raw.count(chr(10), 0, m.start()) + 1}", m.group(2))
        for i, line in enumerate(_strip(raw).split("\n"), 1):
            s = line.strip()
            if s.startswith("#"):
                parts = s[1:].split(None, 2)
                if not parts:
                    continue
                dr = parts[0]
                if dr in ("if", "ifdef", "ifndef", "elif"):
                    for m in tok.findall(s):
                        add(m, "switch", f"{rel}:{i}")
                elif dr == "define" and len(parts) > 1 and tok.fullmatch(parts[1].split("(")[0]):
                    e = out.setdefault(parts[1], dict(uses=set(), name=None, where=[], defined_default=None))
                    e["defined_default"] = parts[2] if len(parts) > 2 else ""
                    for m in tok.findall(parts[2] if len(parts) > 2 else ""):
                        add(m, "value", f"{rel}:{i}")
                continue
            for m in tok.findall(s):
                if not any(w == f"{rel}:{i}" for w in out.get(m, {}).get("where", [])):
                    add(m, "value", f"{rel}:{i}")
    return {m: e for m, e in out.items() if e["uses"]}


def template(root: Path, kern: str) -> dict[str, str]:
    p = root / "config" / "kernels" / f"{KERNELS[kern][2]}.py"
    if not p.exists():
        return {}
    return {f: v.strip() for f, v in re.findall(r"self\.(\w+)\s*=\s*([^#\n]+)", p.read_text())}


def interface(root: Path = KREPO) -> dict[str, list[dict]]:
    """Per kernel: one row per parameter (parsed macro), with kind, note, template field and default."""
    out = {}
    for kern, (prefix, _, _) in KERNELS.items():
        if not (root / "sw" / kern).exists():
            continue
        parsed = parse_kernel(root, kern)
        tmpl = template(root, kern)
        ann = ANNOT.get(kern, {})
        rows = []
        for m in sorted(parsed):
            e = parsed[m]
            a = ann.get(m, {})
            field = a["field"] if "field" in a else m[len(prefix) + 1:].lower()
            use = "switch" if "switch" in e["uses"] else ("descriptor" if "descriptor" in e["uses"] else "value")
            default = tmpl.get(field) if field else None
            if default is None and e["defined_default"] is not None:
                default = f"{e['defined_default']} (kernel #define)"
            rows.append(dict(param=m, use=use, name=e["name"], kind=a.get("kind"), note=a.get("note", ""), dep=a.get("dep"),
                             field=field, template=field in tmpl if field else False, default=default,
                             where=e["where"][0]))
        out[kern] = rows
    return out


def run_values(run_dir: Path, kern: str) -> dict[str, str] | None:
    """Macros of the generated header kept in a run folder: {name: value} ('' for a bare #define)."""
    h = Path(run_dir) / KERNELS[kern][1]
    if not h.exists():
        return None
    out = {}
    for m in re.finditer(r"^#define[ \t]+(\w+)(?:[ \t]+(.*?))?[ \t]*$", h.read_text(), re.M):
        v = (m.group(2) or "").strip()
        mo = re.fullmatch(r"\(\(\w+\)\s*(.*?)\)", v)
        out[m.group(1)] = mo.group(1) if mo else v
    return out


def freeze(root: Path, commit: str, out: Path = UPSTREAM_JSON) -> None:
    """Parse another tree (an upstream clone) once and keep the result: build never depends on the clone."""
    data = dict(source=str(root), commit=commit, interface=interface(root),
                kernels=sorted(p.name for p in (root / "sw").iterdir() if p.is_dir()))
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        i = sys.argv.index("--freeze")
        freeze(Path(sys.argv[i + 1]), sys.argv[i + 2])
    else:
        for kern, rows in interface().items():
            print(f"== {kern}")
            for r in rows:
                print(f"  {r['param']:28} {r['use']:10} {str(r['kind']):9} default {r['default']!s:12} {r['where']}")
