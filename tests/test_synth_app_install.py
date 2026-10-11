"""Module 15's app-install generator (mktstats.synth.app_install): determinism, shapes, truth
keys, that the truth's estimands are the weighted means of the saved true_* columns, the
committed files, and the claims the lab's checkpoints rely on, checked with the reference
solution on the committed data and (marked slow) over the 30 seeds behind the tolerances in
truth_app_install.json."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from mktstats import dml
from mktstats.synth import app_install as gen

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
SEEDS = range(2035, 2065)


@pytest.fixture(scope="module")
def res():
    return gen.app_install()


def test_deterministic(res):
    again = gen.app_install()
    pd.testing.assert_frame_equal(res.customers, again.customers)
    assert res.truth == again.truth
    assert not gen.app_install(1, n=2_000).customers.equals(gen.app_install(2, n=2_000).customers)
    with pytest.raises(TypeError, match="unknown parameters"):
        gen.app_install(t_zero=1)


def test_customers_table(res):
    df = res.customers
    assert list(df.columns) == list(gen.COLUMNS) and len(df) == gen.N == 20_000
    assert df["customer_id"].tolist() == list(range(1, 20_001))
    assert set(df["acquisition_channel"]) == set(gen.CHANNELS)
    assert df["tenure_months"].between(3, 96).all()
    assert (df["orders_12m"] >= 1).all()
    assert df["days_since_last_order"].between(1, 365).all()
    assert (df["avg_order_value"] > 0).all()
    for c in ("email_open_rate", "mobile_share", "true_propensity"):
        assert df[c].between(0, 1).all(), c
    assert set(df["loyalty_member"]) == {0, 1} and set(df["app_installed"]) == {0, 1}
    assert (df["spend_next_26w"] >= 0).all()
    assert 0.55 < (df["spend_next_26w"] == 0).mean() < 0.65  # most customers buy nothing
    # The covariates the lab uses never include the truth columns.
    assert not [c for c in dml.FEATURES if c.startswith("true_")]
    assert dml.features(df).shape == (20_000, 9)


def test_truth_keys(res):
    t = res.truth
    want = {"seed", "n", "description", "columns", "covariates", "truth_columns", "parameters",
            "formulas", "estimands", "incentive", "decision", "confounders", "install_rate",
            "spend", "propensity", "tolerances"}
    assert set(t) == want
    assert t["seed"] == 2035 and t["n"] == 20_000
    assert set(t["estimands"]) == {"ate", "att", "atc", "effect_overlap", "naive_expected"}
    assert all(set(v) == {"value", "definition"} for v in t["estimands"].values())
    assert set(t["parameters"]) == set(gen.PARAMS)
    assert {r["odds_multiplier"] for r in t["incentive"]["table"]} == {1.25, 1.5, 2, 3, 5, 10}
    d = t["decision"]
    assert (d["margin"], d["credit"], d["horizon_weeks"]) == (0.30, 2.50, 26)
    assert d["break_even_effect"] == pytest.approx(2.50 / 0.30, rel=1e-6)
    assert t["confounders"]["mobile_share"] == "both"
    assert t["confounders"]["email_open_rate"] == "install"
    json.dumps(t)  # JSON-serialisable


def test_estimands_are_weighted_means_of_the_true_columns(res):
    df, est = res.customers, res.truth["estimands"]
    e, tau, mu0 = (df[c].to_numpy() for c in ("true_propensity", "true_effect", "true_baseline"))
    assert est["ate"]["value"] == pytest.approx(tau.mean(), rel=1e-5)
    assert est["att"]["value"] == pytest.approx(np.sum(e * tau) / e.sum(), rel=1e-5)
    assert est["atc"]["value"] == pytest.approx(np.sum((1 - e) * tau) / (1 - e).sum(), rel=1e-5)
    assert est["effect_overlap"]["value"] == pytest.approx(dml.overlap_truth(e, tau), rel=1e-5)
    naive = np.sum(e * (mu0 + tau)) / e.sum() - np.sum((1 - e) * mu0) / (1 - e).sum()
    assert est["naive_expected"]["value"] == pytest.approx(naive, rel=1e-5)
    # The averages differ, which is why each checkpoint names its target.
    v = {k: est[k]["value"] for k in est}
    assert v["atc"] < v["ate"] < v["effect_overlap"] < v["att"] < v["naive_expected"]


def test_the_truth_columns_follow_the_formulas(res):
    df = res.customers
    e = df["true_propensity"].to_numpy()
    assert np.allclose(e, expit(gen.install_logit(df)), atol=5e-7)
    mu0 = df["true_baseline"].to_numpy()
    assert np.allclose(mu0, gen.baseline_spend(df), atol=5e-5)
    assert np.allclose(df["true_effect"], 8 + 0.12 * gen.baseline_spend(df), atol=5e-5)
    # E[Y | X, T] = mu0 + T tau: the average residual is within sampling noise of 0.
    resid = df["spend_next_26w"] - mu0 - df["app_installed"] * df["true_effect"]
    assert abs(resid.mean()) < 3 * resid.std() / np.sqrt(len(resid))


def test_the_incentive_table(res):
    df, inc = res.customers, res.truth["incentive"]
    table = inc["table"]
    effects = [r["effect_persuaded"] for r in table]
    shares = [r["share_persuaded"] for r in table]
    assert effects == sorted(effects, reverse=True)  # bigger credits reach customers who gain less
    assert shares == sorted(shares)
    main = next(r for r in table if r["odds_multiplier"] == 2)
    assert inc["effect_persuaded"] == main["effect_persuaded"]
    e, tau = df["true_propensity"].to_numpy(), df["true_effect"].to_numpy()
    # As k -> 1 the persuaded customers are weighted e (1 - e): the overlap-weighted effect.
    near_one = gen.persuaded(logit(e), e, tau, 1.0001)["effect_persuaded"]
    assert near_one == pytest.approx(res.truth["estimands"]["effect_overlap"]["value"], rel=1e-3)
    assert gen.persuaded(logit(e), e, tau, 2.0)["effect_persuaded"] == pytest.approx(
        main["effect_persuaded"], rel=1e-4)


def test_committed_files_match_the_generator(res):
    spec = importlib.util.spec_from_file_location("make_synthetic",
                                                  ROOT / "scripts" / "make_synthetic.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    assert "app_install" in ms.EXTENSIONS
    frames, truth = ms.build_extension("app_install")
    assert list(frames) == ["app_install_customers.csv"]
    committed = pd.read_csv(SYNTH / "app_install_customers.csv")
    pd.testing.assert_frame_equal(committed, res.customers, check_dtype=False)
    assert json.loads((SYNTH / "truth_app_install.json").read_text()) == json.loads(
        json.dumps(truth))
    assert (SYNTH / "app_install_customers.csv").stat().st_size < 1.5e6
    assert truth["tolerances"] == gen.TOLERANCES


@pytest.mark.parametrize("quick", [False, True], ids=["FULL", "QUICK"])
def test_checkpoint_1_claim_on_the_committed_data(quick):
    """Fast (no boosting): both intervals above the true ATT, and the regression in the middle
    option of Exercise 1's Predict ($25 to $50), not within $3 of either boundary."""
    df = pd.read_csv(SYNTH / "app_install_customers.csv")
    if quick:
        df = df.iloc[dml.quick_rows(len(df), 2035)].reset_index(drop=True)
    e, tau = df["true_propensity"].to_numpy(), df["true_effect"].to_numpy()
    base = dml.baseline_estimates(df["spend_next_26w"].to_numpy(dtype=float),
                                  df["app_installed"].to_numpy(), dml.features(df))
    assert (base["ci_low"] > dml.true_averages(e, tau)["ATT"]).all()
    assert 28 < base.loc["regression", "estimate"] < 47


# ---------------------------------------------------------------------- slow: model fits
@pytest.mark.slow
@pytest.mark.parametrize("quick", [False, True], ids=["FULL", "QUICK"])
def test_the_committed_seed_passes_every_check(quick):
    """The lab's own configuration (seed 2035): every checkpoint's truth comparison and both
    demonstration checks hold in both modes."""
    m = dml.measure_app(2035, quick)
    assert m["baselines_above_att"] and m["dml_covers_overlap"]
    assert m["trimmed"]["0.05-0.95"]["covers"]
    assert m["linear"]["covers"] and m["linear"]["rel_diff"] < 1e-6
    assert m["overfit"]["se_ratio"] >= gen.TOLERANCES["no_cross_fitting"]["min_ratio"]
    assert m["hidden"]["shift"] > gen.TOLERANCES["hidden_confounder"]["min_shift"]
    assert m["decision"]["offer"] and not m["decision"]["offer_after_shift"]


@pytest.fixture(scope="module")
def sweep():
    return {q: [dml.measure_app(s, q) for s in SEEDS] for q in (False, True)}


@pytest.mark.slow
@pytest.mark.parametrize("quick", [False, True], ids=["FULL", "QUICK"])
def test_the_demonstration_checks_hold_on_every_seed(sweep, quick):
    """The rule for a labeled check of a demonstration (brief, Pedagogy review item 3): it
    stays only if it holds on every one of at least 30 seeds in both modes."""
    rs = sweep[quick]
    assert len(rs) >= 30
    tol = gen.TOLERANCES
    assert min(r["overfit"]["se_ratio"] for r in rs) >= tol["no_cross_fitting"]["min_ratio"]
    assert min(r["hidden"]["shift"] for r in rs) > tol["hidden_confounder"]["min_shift"]
    assert all(r["baselines_above_att"] for r in rs)


@pytest.mark.slow
@pytest.mark.parametrize("quick", [False, True], ids=["FULL", "QUICK"])
def test_coverage_over_30_seeds(sweep, quick):
    """95% intervals: measured 27/30 (FULL) and 29/30 (QUICK) for Checkpoints 3, 4 and 5 on
    macOS; at least 26 of 30 allows for float differences between thread counts."""
    rs = sweep[quick]
    assert sum(r["dml_covers_overlap"] for r in rs) >= 26
    assert sum(r["trimmed"]["0.05-0.95"]["covers"] for r in rs) >= 26
    assert sum(r["linear"]["covers"] for r in rs) >= 26
    assert max(r["linear"]["rel_diff"] for r in rs) < 1e-6
    assert all(r["trimmed"]["0.05-0.95"]["weight_share_dropped"]
               < 1 - r["trimmed"]["0.05-0.95"]["share_kept"] for r in rs)


@pytest.fixture(scope="module")
def hill_arms():
    from mktstats.data import load_hillstrom

    return dml.hillstrom_arms(load_hillstrom())


@pytest.mark.slow
def test_the_hillstrom_check_over_spoiling_seeds(hill_arms):
    full = [dml.measure_hillstrom(hill_arms, s, False) for s in SEEDS]
    assert all(r["kept_ok"] and r["naive_excludes"] and r["linear_covers"] for r in full)
    half = [dml.measure_hillstrom(hill_arms, s, True, half_seed=2035) for s in SEEDS]
    assert all(r["kept_ok"] and r["linear_covers"] for r in half)
    assert sum(r["naive_excludes"] for r in half) >= 28  # measured 29 of 30
    lab = next(r for r in half if r["seed"] == 2035)  # the lab's own QUICK configuration
    assert lab["naive_excludes"] and lab["linear_covers"] and lab["kept_ok"]
