"""Every generated lab notebook follows the notebook contract (CONTRIBUTING.md): the five
generated cells, stub -> solution -> checkpoint for every exercise, parseable headings, the
first checkpoint before any model fit, the badge and install at repo.ref, and no drift
between the sources and the committed notebooks and includes."""

import json
import re

import pytest

import common
import gen_notebooks
import gen_tables
import lab_steps

V = common.load_variables()
ENTRIES = common.existing_notebooks(V)
IDS = [e["id"] for e in ENTRIES]
REF = str(V["repo"]["ref"])
FIT = re.compile(r"\.fit\(|causal_forest\(|\bpnbd\(|\bbgnbd\(|sample\(")


def load(entry):
    return json.loads((common.ROOT / entry["path"]).read_text(encoding="utf-8"))


def text(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def tags(cell):
    return cell.get("metadata", {}).get("tags", [])


def test_there_are_notebooks_to_check():
    assert ENTRIES, "no generated notebooks found"


def test_generated_notebooks_have_not_drifted():
    assert gen_notebooks.main(["--check"]) == 0


def test_generated_includes_have_not_drifted():
    assert gen_tables.main(["--check"]) == 0


def test_notebook_layout_matches_the_variables():
    assert gen_notebooks.check_layout(V) == []


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_generated_cells_frame_the_lab(entry):
    cells = load(entry)["cells"]
    ids = [c["id"] for c in cells]
    assert ids[:3] == [common.HEADER_ID, common.INSTALL_ID, common.HARNESS_ID]
    assert ids[-2:] == [common.RECORD_ID, common.FOOTER_ID]
    assert len(set(ids)) == len(ids), "cell ids must be unique"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_kernelspec_and_no_outputs(entry):
    nb = load(entry)
    assert nb["metadata"]["kernelspec"] == common.KERNELS[entry["kernel"]]["kernelspec"]
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_badge_and_install_point_at_the_ref(entry):
    cells = {c["id"]: text(c) for c in load(entry)["cells"]}
    badge = f"{V['repo']['colab_base']}/{REF}/{entry['path']}"
    assert badge in cells[common.HEADER_ID]
    install = cells[common.INSTALL_ID]
    if entry["kernel"] == "python3":
        assert f'MKTSTATS_REF = "{REF}"' in install
        assert '_os.environ["MKTSTATS_REF"] = MKTSTATS_REF' in install  # data/ at the same ref
        assert f"{V['repo']['raw']}/{REF}/environment/requirements.txt" in install
        assert repr(common.python_pins(V, entry)) in install
    else:
        assert f'MKTSTATS_REF <- "{REF}"' in install
        assert V["environment"]["p3m_url"] in install
    assert (
        f"ref={REF!r}" in cells[common.HARNESS_ID] or f'ref = "{REF}"' in cells[common.HARNESS_ID]
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_harness_carries_the_current_hashes(entry):
    nb = load(entry)
    harness = next(text(c) for c in nb["cells"] if c["id"] == common.HARNESS_ID)
    assert common.sha_of_cells(nb["cells"]) in harness
    assert common.deps_sha(V, entry) in harness


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_exercise_has_stub_then_solution_then_checkpoint(entry):
    cells = load(entry)["cells"]
    python = entry["kernel"] == "python3"
    heads = [
        (i, int(m[1]))
        for i, c in enumerate(cells)
        if c["cell_type"] == "markdown"
        for m in [re.search(r"^## Exercise (\d+) ·", text(c), re.M)]
        if m
    ]
    assert [n for _, n in heads] == list(range(1, len(heads) + 1)), "exercises number 1, 2, ..."
    bounds = [i for i, _ in heads] + [len(cells)]
    for (start, n), end in zip(heads, bounds[1:], strict=True):
        code = [c for c in cells[start:end] if c["cell_type"] == "code"]
        kinds = [
            next((t for t in ("exercise", "solution", "checkpoint") if t in tags(c)), None)
            for c in code
        ]
        found = [k for k in kinds if k]
        assert found[:3] == ["exercise", "solution", "checkpoint"], f"exercise {n}: {found}"
        stub = text(code[kinds.index("exercise")])
        solution = text(code[kinds.index("solution")])
        check = text(code[kinds.index("checkpoint")])
        if python:
            assert f'raise NotImplementedError("TODO {n}")' in stub
            assert f"@workshop.solution({n})" in solution
            assert solution.startswith(f"# @title Solution {n}")
            assert re.search(rf"with workshop\.checkpoint\({n}\b", check)
        else:
            assert f'stop("TODO {n}")' in stub
            assert re.search(rf'^solution\({n}, "', solution, re.M)
            assert solution.startswith(f"#@title Solution {n}")
            assert re.search(rf"^checkpoint\({n},", check, re.M)


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_solution_cells_are_folded(entry):
    for c in load(entry)["cells"]:
        if "solution" in tags(c):
            assert c["metadata"]["cellView"] == "form"
            assert c["metadata"]["jupyter"] == {"source_hidden": True}


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_headings_parse(entry):
    rows = lab_steps.rows(entry)
    headings = lab_steps.headings(entry)
    for line in headings:
        if re.match(r"^##\s+(Exercise|Stretch|Decision)\b", line) or re.match(r"^#\s+Part\b", line):
            assert any(
                p.match(line)
                for p in (lab_steps.PART, lab_steps.EXERCISE, lab_steps.STRETCH, lab_steps.DECISION)
            ), line
    exercises = [r for r in rows if r["kind"] == "exercise"]
    assert exercises and all(r["minutes"] for r in exercises), "each exercise states its minutes"
    decisions = [r for r in rows if r["kind"] == "decision"]
    assert len(decisions) == 1, "every lab has one Decision section"
    kinds = [r["kind"] for r in rows]
    if "stretch" in kinds:
        assert kinds.index("stretch") > kinds.index("decision"), "the stretch comes last"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_first_checkpoint_comes_before_any_model_fit(entry):
    cells = load(entry)["cells"]
    code = [c for c in cells if c["cell_type"] == "code" and c["id"] not in common.GENERATED_IDS]
    first = next(i for i, c in enumerate(code) if "checkpoint" in tags(c))
    for c in code[:first]:
        assert not FIT.search(text(c)), f"a model fit before the first checkpoint: {c['id']}"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_the_lab_never_downloads_files_through_colab(entry):
    """Rich outputs (files.download, progress bars) move a cell's output into an iframe the
    lead cannot read on Colab: the record cell must stay plain text."""
    for c in load(entry)["cells"]:
        assert "files.download" not in text(c)
        if entry["kernel"] == "python3" and ".fit(" in text(c):
            assert "progressbar=False" in text(c), c["id"]
