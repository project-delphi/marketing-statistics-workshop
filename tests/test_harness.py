"""The Python exercise harness (src/mktstats/harness.py), run in a real IPython shell."""

import contextlib
import io
import json

import pytest
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config

START = (
    "from mktstats.harness import start\n"
    "workshop = start('python/toy', '0' * 16, '1' * 16, 'main', {1: ('double',), 2: ('half',)})"
)
STUB = "def double(x):\n    raise NotImplementedError('TODO 1')"
MINE = "def double(x):\n    return x + x + 1"
REFERENCE = "@workshop.solution(1)\ndef double(x):\n    return 2 * x"
CHECK = "with workshop.checkpoint(1):\n    assert double(2) == 4, 'double(2) should be 4'"


@pytest.fixture
def ip():
    InteractiveShell.clear_instance()
    config = Config()
    config.HistoryManager.enabled = False  # no history database in ~/.ipython
    shell = InteractiveShell.instance(config=config)
    yield shell
    InteractiveShell.clear_instance()


def cell(ip, code):
    return ip.run_cell(code, store_history=False)


def printed(ip, code):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = cell(ip, code)
    return out.getvalue(), result


def ws(ip):
    return ip.user_ns["workshop"]


def test_other_hooks_survive_reruns_of_the_harness(ip):
    calls = []

    def other(*args):
        calls.append(args)

    ip.events.register("post_run_cell", other)
    cell(ip, START)
    cell(ip, START)
    callbacks = ip.events.callbacks["post_run_cell"]
    assert other in callbacks
    ours = [c for c in callbacks if type(getattr(c, "__self__", None)).__name__ == "Workshop"]
    assert len(ours) == 1


def test_a_solution_does_not_replace_your_code(ip):
    cell(ip, START)
    cell(ip, MINE)
    cell(ip, REFERENCE)
    assert ip.user_ns["double"](3) == 7
    text, _ = printed(ip, "workshop.use_reference(1)")
    assert "REFERENCE" in text
    assert ip.user_ns["double"](3) == 6
    cell(ip, MINE)  # rerunning your TODO cell goes back to your code
    cell(ip, REFERENCE)
    assert ip.user_ns["double"](3) == 7


def test_worked_mode_binds_the_reference(ip, monkeypatch):
    monkeypatch.setenv("MKTSTATS_WORKED", "1")
    cell(ip, START)
    assert ip.user_ns["WORKED_EXAMPLE"] is True
    cell(ip, STUB)
    cell(ip, REFERENCE)
    assert ip.user_ns["double"](3) == 6


def test_the_worked_example_switch(ip):
    cell(ip, START)
    cell(ip, MINE)
    cell(ip, "WORKED_EXAMPLE = True")
    cell(ip, REFERENCE)
    assert ip.user_ns["double"](3) == 6
    cell(ip, "WORKED_EXAMPLE = False")
    cell(ip, REFERENCE)
    assert ip.user_ns["double"](3) == 7


def test_a_skipped_todo_is_not_replaced_by_the_reference(ip):
    cell(ip, START)
    cell(ip, REFERENCE)
    result = cell(ip, "double(2)")
    assert isinstance(result.error_in_exec, NotImplementedError)
    assert "TODO 1" in str(result.error_in_exec)


def test_checkpoints_say_whose_code_they_checked(ip):
    cell(ip, START)
    cell(ip, "def double(x):\n    return 2 * x")
    cell(ip, REFERENCE)
    text, result = printed(ip, CHECK)
    assert result.success and "passed on your code" in text
    assert ws(ip).results["1"] == (True, "your code")
    cell(ip, "workshop.use_reference(1)")
    text, _ = printed(ip, CHECK.replace("checkpoint(1)", "checkpoint(1, label='1b')"))
    assert "passed on the REFERENCE solution" in text
    assert ws(ip).results["1b"] == (True, "the REFERENCE solution")


def test_an_unwritten_todo_stops_with_the_recovery_message(ip):
    cell(ip, START)
    cell(ip, STUB)
    cell(ip, REFERENCE)
    text, result = printed(ip, CHECK)
    assert isinstance(result.error_in_exec, NotImplementedError)  # re-raised: Run all stops
    assert "TODO 1 is not written yet" in text and "workshop.use_reference(1)" in text
    assert ws(ip).results["1"] == (False, "your code")


def test_a_wrong_answer_points_at_the_todo(ip):
    cell(ip, START)
    cell(ip, MINE)
    cell(ip, REFERENCE)
    text, result = printed(ip, CHECK)
    assert isinstance(result.error_in_exec, AssertionError)
    assert "failed on your code" in text and "fix TODO 1" in text


def test_a_failure_on_the_reference_is_not_the_learners_fault(ip, monkeypatch):
    monkeypatch.setenv("MKTSTATS_WORKED", "1")
    cell(ip, START)
    cell(ip, "@workshop.solution(1)\ndef double(x):\n    return 0")
    text, result = printed(ip, CHECK)
    assert not result.success and "Tell the instructor" in text


def test_a_check_of_provided_code_names_no_owner(ip):
    cell(ip, START)
    text, result = printed(ip, "with workshop.checkpoint(label='data'):\n    assert 1 == 1")
    assert result.success and ws(ip).results["data"] == (True, None)
    assert "Checkpoint data passed." in text


def test_verify_mode_runs_a_checkpoint_on_the_stub(ip, monkeypatch):
    monkeypatch.setenv("MKTSTATS_WORKED", "1")
    cell(ip, START)
    cell(ip, STUB)
    cell(ip, REFERENCE)
    assert cell(ip, CHECK).success
    cell(ip, "workshop._verify_begin(1)")
    text, copy = printed(ip, CHECK)
    assert isinstance(copy.error_in_exec, NotImplementedError)
    assert "failed on the stub, as it should" in text
    assert cell(ip, "workshop._verify_end(1)").success
    assert ip.user_ns["double"](3) == 6  # the reference is back
    assert ws(ip).cell_errors == 0  # the verification copy failed on purpose
    assert ws(ip).status() == "pass"


def test_verify_mode_fails_a_checkpoint_that_passes_on_the_stub(ip, monkeypatch):
    monkeypatch.setenv("MKTSTATS_WORKED", "1")
    cell(ip, START)
    cell(ip, "def double(x):\n    return 4")  # a stub that happens to pass
    cell(ip, REFERENCE)
    cell(ip, "workshop._verify_begin(1)")
    cell(ip, CHECK)
    result = cell(ip, "workshop._verify_end(1)")
    assert isinstance(result.error_in_exec, AssertionError)
    assert "cannot tell a finished exercise" in str(result.error_in_exec)
    assert ws(ip).status() == "fail"


def test_sabotage_keeps_the_stub_in_worked_mode(ip, monkeypatch):
    monkeypatch.setenv("MKTSTATS_WORKED", "1")
    monkeypatch.setenv("MKTSTATS_SABOTAGE", "1")
    cell(ip, START)
    cell(ip, STUB)
    cell(ip, REFERENCE)
    text, result = printed(ip, CHECK)
    assert isinstance(result.error_in_exec, NotImplementedError)
    assert ws(ip).settings() == {"MKTSTATS_SABOTAGE": "1"}


def test_quick_comes_from_the_environment(ip, monkeypatch):
    monkeypatch.setenv("MKTSTATS_QUICK", "1")
    cell(ip, "QUICK = False")
    cell(ip, START)
    assert ip.user_ns["QUICK"] is True
    assert ws(ip).settings() == {"MKTSTATS_QUICK": "1"}


def test_rerunning_the_harness_starts_a_new_run_but_keeps_solutions(ip):
    cell(ip, START)
    cell(ip, MINE)
    cell(ip, REFERENCE)
    cell(ip, "with workshop.checkpoint(label='x'):\n    assert False")
    cell(ip, START)
    assert ws(ip).results == {} and ws(ip).cell_errors == 0
    assert 1 in ws(ip).ref and 1 in ws(ip).stub


def test_cells_are_timed(ip):
    cell(ip, START)
    cell(ip, "x = 1")
    cell(ip, "y = 2")
    assert len(ws(ip).cell_seconds) == 2


def test_a_plain_scalar_cannot_be_a_solution_value(ip):
    cell(ip, START)
    result = cell(ip, "K = workshop.solution_value(2, 'K', 5)")
    assert isinstance(result.error_in_exec, TypeError)


def test_the_run_record_is_printed_and_saved(ip, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MKTSTATS_INSTALL_SECONDS", "31.5")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-test")
    cell(ip, START)
    cell(ip, "def double(x):\n    return 2 * x")
    cell(ip, REFERENCE)
    cell(ip, CHECK)
    text, _ = printed(ip, "workshop.run_record()")
    begin = text.index("----- run record")
    assert "----- end of run record -----" in text
    shown = json.loads(text[text.index("{", begin) : text.rindex("}") + 1])
    saved = json.loads((tmp_path / "run_record.json").read_text())
    assert shown == saved
    for field in (
        "notebook",
        "content_sha",
        "deps_sha",
        "ref",
        "mktstats_commit",
        "mode",
        "settings",
        "status",
        "seconds",
        "install_seconds",
        "cell_seconds",
        "python",
        "colab_release",
        "cpus",
        "packages",
        "checkpoints",
        "kernel",
        "date",
    ):
        assert field in saved, field
    assert saved["status"] == "pass" and saved["mode"] == "learner"
    assert saved["install_seconds"] == 31.5
    assert saved["colab_release"] == "release-colab-test"
    assert saved["checkpoints"] == {"1": {"passed": True, "whose": "your code"}}
    assert saved["deps_sha"] == "1" * 16 and saved["ref"] == "main"
    assert "mktstats" in saved["packages"]


def test_a_run_without_checkpoints_or_with_errors_does_not_pass(ip):
    cell(ip, START)
    assert ws(ip).status() == "fail"
    cell(ip, "with workshop.checkpoint(label='x'):\n    pass")
    cell(ip, "raise ValueError('a provided cell failed')")
    assert ws(ip).status() == "fail"
