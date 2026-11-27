"""Kernel parameter interfaces: the parsed parameters and the annotation table must match both ways, so a kernel
parameter cannot be missed (as Activation's bias_enable once was) and a stale annotation cannot linger."""

import json

from workload_modeling import kernel_interface as ki


def test_every_parsed_parameter_is_annotated_and_back():
    for kern, rows in ki.interface().items():
        parsed = {r["param"] for r in rows}
        ann = set(ki.ANNOT[kern])
        assert parsed - ann == set(), (kern, "parsed but not in ANNOT", sorted(parsed - ann))
        assert ann - parsed == set(), (kern, "in ANNOT but not in the kernel code", sorted(ann - parsed))
        assert all(r["kind"] for r in rows), kern


def test_every_template_field_reaches_the_kernel():
    # a config field the kernel never reads would be a silent no-op knob
    for kern, rows in ki.interface().items():
        used = {r["field"] for r in rows if r["field"]}
        # dtype also reaches every kernel through the shared DATA_TYPE_* / REDMULE_COMPUTE_TYPE macros (no prefix)
        unused = set(ki.template(ki.KREPO, kern)) - used - {"dtype"}
        assert unused == set(), (kern, sorted(unused))


def test_descriptor_fields_carry_their_name():
    # main.c passes the descriptor as PREFIX_FIELD/*name*/ (the convention the parser relies on)
    for kern, rows in ki.interface().items():
        d = [r for r in rows if r["use"] == "descriptor"]
        if kern == "Eltwise":          # EltwiseAnalyze(ELT_M_SIZE, ELT_N_SIZE, ...) has no /*name*/ comments yet
            continue
        assert d and all(r["name"] for r in d), kern


def test_upstream_only_parameters_are_annotated():
    up = json.loads(ki.UPSTREAM_JSON.read_text())
    ours = ki.interface()
    for kern, rows in up["interface"].items():
        if kern not in ours:
            continue
        only = {r["param"] for r in rows} - {r["param"] for r in ours[kern]}
        assert only == set(ki.ANNOT_UPSTREAM.get(kern, {})), (kern, sorted(only))


def test_run_values_read_bare_defines():
    # a bare #define (ELT_FP16) must not swallow the next line
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "elt.h").write_text("#define ELT_FP16\n#define ELT_PIPELINE ((uint64_t)1)\n#define ELT_M_SIZE ((uint64_t)8)\n")
        v = ki.run_values(Path(d), "Eltwise")
    assert v == {"ELT_FP16": "", "ELT_PIPELINE": "1", "ELT_M_SIZE": "8"}
