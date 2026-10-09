"""Recovery: fits on the committed (default-seed) synthetic data land within the tolerances stored
in truth.json. MAP fits take seconds; the MMM MCMC fit is marked slow."""

import math

import numpy as np
import pandas as pd
import pytest

from mktstats import data, recovery, synth, uplift


@pytest.fixture(scope="module")
def truth():
    return data.load_truth()


@pytest.fixture(scope="module")
def tol(truth):
    return truth["tolerances"]


def _assert_within(table: pd.DataFrame):
    bad = table.loc[table["within"] != True]  # noqa: E712 - NaN must fail too
    assert bad.empty, "\n" + table.to_string()


# ----------------------------------------------------------------------------- helpers
def test_compare_relative_absolute_and_nested():
    truth = {"r": 1.0, "g": {"social": -0.4}, "skip": "text"}
    est = {"r": 1.1, "g": {"social": -0.2}, "extra": 3.0}
    t = recovery.compare(truth, est, {"r": {"rel": 0.15}, "g": {"social": {"abs": 0.1}}})
    assert list(t["parameter"]) == ["r", "g.social"]
    row = t.set_index("parameter")
    assert row.loc["r", "rel_error"] == pytest.approx(0.1) and row.loc["r", "within"]
    assert row.loc["g.social", "abs_error"] == pytest.approx(0.2)
    assert not row.loc["g.social", "within"]
    flat = recovery.compare({"a": 2.0}, {"a": 3.0}, 0.4)
    assert not flat["within"].iloc[0]
    untoleranced = recovery.compare({"a": 2.0}, {"a": 3.0}, {"b": {"rel": 1}})
    assert math.isnan(untoleranced["within"].iloc[0])


def test_hdi_is_the_shortest_interval():
    x = np.r_[np.zeros(50), np.arange(1, 51)]  # half the mass piled at 0
    lo, hi = recovery.hdi(x, prob=0.5)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = recovery.hdi(np.arange(100), prob=0.9)
    assert hi - lo == 89
    with pytest.raises(ValueError):
        recovery.hdi([1.0, 2.0], prob=1.0)


# ----------------------------------------------------------------------------- BTYD (MAP)
def test_bgnbd_map_recovers_truth(truth, tol):
    pytest.importorskip("pymc_marketing")
    rfm = data.load_synthetic("bgnbd_rfm")
    _, est = recovery.fit_bgnbd_map(rfm)
    _assert_within(recovery.compare(truth["btyd_bgnbd"]["bgnbd"], est, tol["btyd_bgnbd_map"]))


def test_bgnbd_map_fails_on_swapped_columns(truth, tol):
    """The tolerances are tight enough to catch a broken RFM table."""
    pytest.importorskip("pymc_marketing")
    rfm = data.load_synthetic("bgnbd_rfm")
    bad = rfm.assign(T=rfm["T"] * 7, recency=rfm["recency"] * 7)  # days instead of weeks
    _, est = recovery.fit_bgnbd_map(bad)
    table = recovery.compare(truth["btyd_bgnbd"]["bgnbd"], est, tol["btyd_bgnbd_map"])
    assert not table["within"].all()


@pytest.fixture(scope="module")
def retailer_rfm(truth):
    return data.retailer_rfm()


def test_pnbd_map_with_covariates_recovers_truth(truth, tol, retailer_rfm):
    pytest.importorskip("pymc_marketing")
    cols = list(truth["retailer"]["covariates"]["columns"].values())
    _, est = recovery.fit_pnbd_map(retailer_rfm, covariate_cols=cols)
    by_name = {v: k for k, v in truth["retailer"]["covariates"]["columns"].items()}
    est = {**{k: est[k] for k in ("r", "alpha", "s", "beta")},
           "gamma_purchase": {by_name[c]: v for c, v in est["gamma_purchase"].items()},
           "gamma_dropout": {by_name[c]: v for c, v in est["gamma_dropout"].items()}}
    t = {**truth["retailer"]["pnbd"],
         "gamma_purchase": truth["retailer"]["covariates"]["gamma_purchase"],
         "gamma_dropout": truth["retailer"]["covariates"]["gamma_dropout"]}
    table = recovery.compare(t, est, tol["retailer_pnbd_covariates_map"])
    table = table[~table["parameter"].str.endswith(".search")]  # reference level, fixed at 0
    _assert_within(table)


def test_pnbd_map_without_covariates_matches_population_means(truth, tol, retailer_rfm):
    pytest.importorskip("pymc_marketing")
    _, est = recovery.fit_pnbd_map(retailer_rfm)
    implied = {"mean_purchase_rate": est["r"] / est["alpha"],
               "mean_dropout_rate": est["s"] / est["beta"]}
    sample = truth["retailer"]["sample"]
    t = {"mean_purchase_rate": sample["mean_true_lambda"],
         "mean_dropout_rate": sample["mean_true_mu"]}
    _assert_within(recovery.compare(t, implied, tol["retailer_pnbd_nocovariates_map"]))


def test_gamma_gamma_map_recovers_truth(truth, tol, retailer_rfm):
    pytest.importorskip("pymc_marketing")
    _, est = recovery.fit_gamma_gamma_map(retailer_rfm)
    est["population_mean_spend"] = est["p"] * est["v"] / (est["q"] - 1)
    t = {**truth["retailer"]["gamma_gamma"],
         "population_mean_spend": truth["retailer"]["population_mean_spend"]}
    _assert_within(recovery.compare(t, est, tol["retailer_gamma_gamma_map"]))


# ----------------------------------------------------------------------------- geo and uplift
def test_geo_did_recovers_lift(truth, tol):
    panel = data.load_synthetic("geo_panel")
    est = recovery.did_log_lift(panel)
    _assert_within(recovery.compare(truth["geo_panel"], {"lift_pct": est["lift_pct"]},
                                    tol["geo_did"]))
    rel = abs(est["incremental_sales"] - truth["geo_panel"]["incremental_sales"])
    assert rel / truth["geo_panel"]["incremental_sales"] < 0.6


def test_uplift_oracle_beats_random():
    d = data.load_synthetic("email_experiment")
    y, t = d["spend"].to_numpy(), d["treatment"].to_numpy()
    oracle = uplift.qini_coefficient(y, t, d["true_cate"].to_numpy())
    # a score unrelated to the data (seed differs from the generator's)
    rnd = np.random.default_rng(12345).uniform(size=len(d))
    assert oracle > uplift.qini_coefficient(y, t, rnd)
    assert oracle > 0
    assert uplift.qini_coefficient(y, t, d["true_cate"].to_numpy(), normalize=True) > 0


def test_oracle_policy_value_close_to_truth(truth):
    d = data.load_synthetic("email_experiment")
    tr = truth["email_experiment"]
    pol = uplift.targeting_rule(d["true_cate"], tr["margin"], tr["offer_cost"])
    exact = uplift.policy_value_true(d["true_cate"], pol, tr["margin"], tr["offer_cost"])
    assert exact == pytest.approx(tr["policy_value_per_customer"]["oracle"], rel=1e-4)
    ipw = uplift.policy_value(d["spend"], d["treatment"], pol, tr["margin"], tr["offer_cost"])
    assert ipw == pytest.approx(exact, abs=0.4)


# ----------------------------------------------------------------------------- MMM (slow)
@pytest.mark.slow
def test_mmm_mcmc_recovers_roas_and_adstock(truth, tol):
    pytest.importorskip("pymc_marketing")
    pytest.importorskip("nutpie")
    weekly = data.load_synthetic("mmm_weekly")
    model = recovery.fit_mmm(weekly, draws=500, tune=500, chains=2, nuts_sampler="nutpie")
    table = recovery.mmm_recovery_table(recovery.mmm_draws(model, weekly), truth["mmm"],
                                        prob=0.94)
    divergences = int(model.idata.sample_stats["diverging"].sum())
    print(f"\ndivergences: {divergences}\n" + table.round(4).to_string())
    roas = table[table["quantity"] == "roas"]
    assert roas["inside"].sum() >= tol["mmm_mcmc"]["roas_inside_min_channels"], roas
    tv_alpha = table[(table["channel"] == "tv") & (table["quantity"] == "adstock_alpha")]
    assert tv_alpha["inside"].all(), tv_alpha


def test_synth_mmm_matches_committed_file():
    w = data.load_synthetic("mmm_weekly")
    pd.testing.assert_frame_equal(w, synth.mmm().weekly, check_dtype=False)
