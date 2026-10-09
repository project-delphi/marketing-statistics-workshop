"""The committed synthetic files: drift gate, size budget, truth.json structure, and (slow) the
multi-seed recovery sweep the tolerances in truth.json were derived from."""

import importlib.util
import json
from pathlib import Path

import pytest

from mktstats import data, recovery, synth

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"


def _make_synthetic():
    spec = importlib.util.spec_from_file_location("make_synthetic",
                                                  ROOT / "scripts" / "make_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_files_match_the_generators():
    assert _make_synthetic().main(["--check"]) == 0


def test_files_are_small():
    for p in SYNTH.iterdir():
        assert p.stat().st_size < 5 * 1024 * 1024, p.name


def test_truth_json_structure():
    truth = json.loads((SYNTH / "truth.json").read_text())
    for name, files in truth["files"].items():
        assert name in truth
        for f in files:
            assert (SYNTH / f).exists(), f
    assert set(truth["tolerances"]) >= {"btyd_bgnbd_map", "retailer_pnbd_covariates_map",
                                        "retailer_gamma_gamma_map", "geo_did", "mmm_mcmc"}
    vbc = truth["retailer"]["value_by_channel"]
    assert set(vbc["existing_customers_at_cal_end"]) == {"search", "social", "referral"}
    assert "optimal_allocation" in truth["mmm"]["true_optimal_allocation"]


# ----------------------------------------------------------------------------- slow sweeps
@pytest.mark.slow
@pytest.mark.parametrize("seed", range(30))
def test_bgnbd_recovery_across_seeds(seed):
    pytest.importorskip("pymc_marketing")
    res = synth.btyd_bgnbd(seed)
    _, est = recovery.fit_bgnbd_map(res.rfm)
    tol = data.load_truth()["tolerances"]["btyd_bgnbd_map"]
    table = recovery.compare(res.truth["bgnbd"], est, tol)
    assert table["within"].all(), table.to_string()


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(100, 130))
def test_retailer_recovery_across_seeds(seed):
    pytest.importorskip("pymc_marketing")
    res = synth.retailer(seed)
    rfm = data.retailer_rfm(res.transactions, res.customers, res.truth)
    cols = list(res.truth["covariates"]["columns"].values())
    by_name = {v: k for k, v in res.truth["covariates"]["columns"].items()}
    _, est = recovery.fit_pnbd_map(rfm, covariate_cols=cols)
    est = {**{k: est[k] for k in ("r", "alpha", "s", "beta")},
           "gamma_purchase": {by_name[c]: v for c, v in est["gamma_purchase"].items()},
           "gamma_dropout": {by_name[c]: v for c, v in est["gamma_dropout"].items()}}
    t = {**res.truth["pnbd"], "gamma_purchase": res.truth["covariates"]["gamma_purchase"],
         "gamma_dropout": res.truth["covariates"]["gamma_dropout"]}
    tol = data.load_truth()["tolerances"]
    table = recovery.compare(t, est, tol["retailer_pnbd_covariates_map"])
    table = table[~table["parameter"].str.endswith(".search")]
    assert table["within"].all(), table.to_string()
    _, gg = recovery.fit_gamma_gamma_map(rfm)
    table = recovery.compare(res.truth["gamma_gamma"], gg, tol["retailer_gamma_gamma_map"])
    assert table["within"].all(), table.to_string()
