"""scripts/test_notebooks.py: verify-cell injection, the learner verdict, notebook selection
and the CLI contracts the CI workflows rely on."""

import json

import nbformat
import pytest

import common
import test_notebooks as tn

V = common.load_variables()


def code(source, tags=None):
    cell = nbformat.v4.new_code_cell(source)
    if tags:
        cell.metadata["tags"] = tags
    return cell


def notebook(cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


def test_python_verification_cells_follow_each_exercises_first_checkpoint():
    nb = notebook(
        [
            code("def f(): raise NotImplementedError('TODO 1')", ["exercise"]),
            code("with workshop.checkpoint(1):\n    assert f()", ["checkpoint"]),
            code("with workshop.checkpoint(1, label='1b'):\n    assert f()", ["checkpoint"]),
            code("with workshop.checkpoint(2):\n    assert g()", ["checkpoint", "no-verify"]),
        ]
    )
    assert tn.inject_verification(nb, "python3") == 1
    sources = [c.source for c in nb.cells]
    assert sources[2] == "workshop._verify_begin(1)"
    assert sources[3] == sources[1] and "raises-exception" in nb.cells[3].metadata["tags"]
    assert sources[4] == "workshop._verify_end(1)"
    assert len(nb.cells) == 7


def test_r_verification_cells():
    nb = notebook([code("checkpoint(3, {\n  check_close(1, 1, abs = 0)\n})", ["checkpoint"])])
    assert tn.inject_verification(nb, "ir") == 1
    assert [c.source for c in nb.cells][1::2] == [".ws_verify_begin(3)", ".ws_verify_end(3)"]


def error_output():
    return nbformat.v4.new_output("error", ename="NotImplementedError", evalue="TODO 1")


def stream(text):
    return nbformat.v4.new_output("stream", name="stdout", text=text)


def test_the_learner_verdict():
    provided = code("x = 1")
    checkpoint = code("with workshop.checkpoint(1):\n    f()", ["checkpoint"])
    checkpoint.outputs = [
        stream("[workshop] Checkpoint 1: TODO 1 is not written yet."),
        error_output(),
    ]
    ok, why = tn.learner_verdict(notebook([provided, checkpoint]))
    assert ok, why
    provided.outputs = [error_output()]
    ok, why = tn.learner_verdict(notebook([provided, checkpoint]))
    assert not ok and "not at checkpoint 1" in why
    provided.outputs = []
    checkpoint.outputs = [error_output()]
    ok, why = tn.learner_verdict(notebook([provided, checkpoint]))
    assert not ok and "recovery message" in why


def test_selection_by_id_stem_or_path():
    ids = {e["id"] for e in tn.select(V, ["00-setup-warmup"])}
    assert ids == {"python/00-setup-warmup", "r/00-setup-warmup"}
    assert [e["id"] for e in tn.select(V, ["r/00-setup-warmup"])] == ["r/00-setup-warmup"]
    assert [e["id"] for e in tn.select(V, ["labs/python/00-setup-warmup.ipynb"])] == [
        "python/00-setup-warmup"
    ]
    with pytest.raises(SystemExit):
        tn.select(V, ["99-nothing"])


def test_list_matrix_gives_ids_and_artifact_safe_slugs(capsys):
    assert tn.main(["--list-matrix"]) == 0
    matrix = json.loads(capsys.readouterr().out)
    assert matrix and all(set(m) >= {"id", "slug"} for m in matrix)
    for m in matrix:
        assert "/" not in m["slug"]
        assert tn.select(V, [m["id"]])


def test_learner_and_verify_flags_are_separate_runs():
    with pytest.raises(SystemExit):
        tn.main(["--learner", "--verify-checkpoints", "--notebooks", "python/00-setup-warmup"])
