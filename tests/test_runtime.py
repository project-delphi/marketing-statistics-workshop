"""mktstats.runtime: where a lab runs, installs only on Colab/SageMaker, restart guard."""

import sys
import types

import pytest

from mktstats import runtime


def test_detect():
    assert runtime.detect({"COLAB_RELEASE_TAG": "release-colab_20261001-060000"}) == "colab"
    assert runtime.detect({"SAGEMAKER_APP_TYPE": "JupyterLab"}) == "sagemaker"
    assert (
        runtime.detect({"MKTSTATS_RUNTIME": "sagemaker", "COLAB_RELEASE_TAG": "x"}) == "sagemaker"
    )
    if not runtime.os.path.exists(runtime.SAGEMAKER_METADATA):
        assert runtime.detect({}) == "local"


def test_pins_of():
    assert runtime.pins_of(["Scikit_Learn==1.6.1", "pandas==2.2.3"]) == {
        "scikit-learn": "1.6.1",
        "pandas": "2.2.3",
    }
    with pytest.raises(ValueError, match="exact pin"):
        runtime.pins_of(["pandas>=2"])


def test_import_names():
    assert runtime.import_name("scikit-learn") == "sklearn"
    assert runtime.import_name("pymc-marketing") == "pymc_marketing"


def no_pip(*args, **kwargs):
    raise AssertionError("pip must not run here")


def test_local_runs_install_nothing_and_warn(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "_pip", no_pip)
    out = runtime.ensure({"pandas": "0.0.1"}, "https://example.invalid/req.txt", runtime="local")
    assert out == {"runtime": "local", "installed": [], "mismatched": ["pandas"]}
    printed = capsys.readouterr().out
    assert "pandas" in printed and "tested with pandas==0.0.1" in printed


def test_met_pins_install_nothing_anywhere(monkeypatch):
    monkeypatch.setattr(runtime, "_pip", no_pip)
    pandas = runtime.installed_version("pandas")
    out = runtime.ensure({"pandas": pandas}, runtime="colab")
    assert out["installed"] == [] and out["mismatched"] == []


def test_colab_installs_mismatched_pins_with_constraints(monkeypatch):
    calls = []
    versions = {"duckdb": "0.0.1"}

    def fake_pip(specs, constraints_url):
        calls.append((specs, constraints_url))
        versions["duckdb"] = "1.3.2"

    monkeypatch.setattr(runtime, "_pip", fake_pip)
    monkeypatch.setattr(runtime, "installed_version", lambda d: versions.get(d))
    monkeypatch.setattr(runtime, "loaded_version", lambda d: None)
    out = runtime.ensure({"duckdb": "1.3.2"}, "https://x/requirements.txt", runtime="colab")
    assert calls == [(["duckdb==1.3.2"], "https://x/requirements.txt")]
    assert out["installed"] == ["duckdb==1.3.2"]


def test_restart_guard_stops_when_pymc_was_preloaded(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "_pip", no_pip)
    monkeypatch.setitem(sys.modules, "pymc", types.SimpleNamespace(__version__="5.28.5"))
    monkeypatch.setattr(runtime, "installed_version", lambda d: "6.3.2" if d == "pymc" else None)
    with pytest.raises(runtime.RestartRequired, match="Restart session, then Run all"):
        runtime.ensure({}, runtime="local")
    assert "pymc: 5.28.5 in memory, 6.3.2 installed" in capsys.readouterr().out


def test_restart_guard_after_installing_a_loaded_package(monkeypatch):
    versions = {"pandas": "2.2.2"}
    monkeypatch.setattr(runtime, "installed_version", lambda d: versions.get(d))
    monkeypatch.setitem(sys.modules, "pandas", types.SimpleNamespace(__version__="2.2.2"))

    def fake_pip(specs, constraints_url):
        versions["pandas"] = "2.2.3"

    monkeypatch.setattr(runtime, "_pip", fake_pip)
    with pytest.raises(runtime.RestartRequired):
        runtime.ensure({"pandas": "2.2.3"}, runtime="colab", guarded=())


def test_info_reports_the_runtime(monkeypatch):
    monkeypatch.setenv("MKTSTATS_RUNTIME", "local")
    i = runtime.info()
    assert i["runtime"] == "local" and i["cpus"] >= 1 and i["python"].startswith("3.")
