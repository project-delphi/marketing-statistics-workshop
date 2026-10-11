"""The subscription generator (Module 14): determinism, shape, the analytic truth, and that the
reference solutions recover the truth the checkpoints rely on, on the committed data (fast)
and over other seeds (slow, the sweep behind truth_subscriptions.json's tolerances)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from mktstats.synth import subscriptions as sub

ROOT = Path(__file__).resolve().parents[1]
TRUTH_FILE = ROOT / "data" / "synthetic" / "truth_subscriptions.json"
COLUMNS = ["customer_id", "start_date", "cancel_date", "plan", "onboarded", "intro_offer",
           "true_tenure_months", "true_rmst24_gain"]


@pytest.fixture(scope="module")
def res():
    return sub.subscriptions()


@pytest.fixture(scope="module")
def subs(res):
    from mktstats import survival as sv

    return sv.add_plan_dummies(sv.tenure_and_churn(res.subscribers))


def test_deterministic(res):
    again = sub.subscriptions()
    pd.testing.assert_frame_equal(res.subscribers, again.subscribers)
    assert res.truth == again.truth
    assert not sub.subscriptions(seed=1).subscribers.equals(res.subscribers)


def test_shape_and_dates(res):
    s, t = res.subscribers, res.truth
    assert list(s.columns) == COLUMNS and len(s) == t["n"] == 8000
    assert s["customer_id"].tolist() == list(range(1, 8001))
    assert s["start_date"].is_monotonic_increasing
    assert s["start_date"].min() >= pd.Timestamp("2024-01-01")
    assert s["start_date"].max() <= pd.Timestamp("2025-12-31")
    observed = s["cancel_date"].notna()
    assert (s.loc[observed, "cancel_date"] > s.loc[observed, "start_date"]).all()
    assert (s.loc[observed, "cancel_date"] <= pd.Timestamp(t["observation_end"])).all()
    assert set(s["plan"]) == set(sub.PLANS)
    assert set(s["onboarded"]) == {0, 1} and set(s["intro_offer"]) == {0, 1}
    # sign-ups grow: the rate doubles over two years, so year 2 has 1.4 times year 1
    first = (s["start_date"] < pd.Timestamp("2025-01-01")).sum()
    assert 1.25 < (len(s) - first) / first < 1.6  # expected 0.875 / 0.625 = 1.4


def test_true_tenure_agrees_with_the_dates(res):
    s = res.subscribers
    end = pd.Timestamp(res.truth["observation_end"])
    observed = s["cancel_date"].notna()
    days = (s.loc[observed, "cancel_date"] - s.loc[observed, "start_date"]).dt.days
    np.testing.assert_allclose(s.loc[observed, "true_tenure_months"], days / 30.4375, atol=1e-6)
    follow = (end - s.loc[~observed, "start_date"]).dt.days
    assert (s.loc[~observed, "true_tenure_months"] * 30.4375 > follow - 1e-6).all()


def test_truth_has_what_the_lab_reads(res):
    t = res.truth
    for key in ("seed", "n", "signup_window", "signup_growth", "observation_end",
                "days_per_month", "time_zero", "model", "covariates", "survival", "rmst",
                "decision", "sample"):
        assert key in t, key
    assert set(t["survival"]["overall"]) == {"3", "6", "12", "18", "24"}
    assert set(t["model"]["hazard_ratios"]) == {"plan_standard", "plan_premium", "onboarded"}
    assert set(t["model"]["intro_offer"]) >= {"landmark_months", "hr_before", "hr_after"}
    for key in ("gain_target", "gain_target_discounted", "target_plan_weights",
                "observed_groups", "by_plan", "gain_target_sample_mean"):
        assert key in t["rmst"], key
    committed = json.loads(TRUTH_FILE.read_text())
    assert committed["files"] == ["subscriptions_subscribers.csv"]
    assert committed["tolerances"]["seed_sets"]


# The analytic values in the brief's truth table (briefs/14-survival-churn.md).
@pytest.mark.parametrize(
    "path, value, tol",
    [
        (("survival", "overall", "3"), 0.7665, 5e-5),
        (("survival", "overall", "6"), 0.6595, 5e-5),
        (("survival", "overall", "12"), 0.5107, 5e-5),
        (("survival", "overall", "18"), 0.4084, 5e-5),
        (("survival", "overall", "24"), 0.3333, 5e-5),
        (("survival", "by_plan", "basic", "12"), 0.424, 5e-4),
        (("survival", "by_plan", "premium", "24"), 0.522, 5e-4),
        (("survival", "by_observed_onboarding", "onboarded", "12"), 0.611, 5e-4),
        (("survival", "by_observed_onboarding", "not_onboarded", "12"), 0.420, 5e-4),
        (("model", "hazard_ratios", "onboarded"), 0.670, 5e-4),
        (("model", "intro_offer", "hr_before"), 2.01375, 1e-5),
        (("rmst", "by_plan", "basic", "not_onboarded"), 10.41, 5e-3),
        (("rmst", "by_plan", "premium", "onboarded"), 17.23, 5e-3),
        (("rmst", "target_plan_weights", "basic"), 0.667, 5e-4),
        (("rmst", "gain_target"), 2.907, 5e-4),
        (("rmst", "gain_target_discounted"), 2.614, 5e-4),
        (("rmst", "observed_groups", "unadjusted_gain"), 4.061, 5e-4),
        (("decision", "true_break_even_q"), 0.430, 5e-4),
        (("decision", "true_value_per_call", "0.35"), 12.21, 5e-3),
    ],
)
def test_truth_matches_the_brief(res, path, value, tol):
    x = res.truth
    for key in path:
        x = x[key]
    assert x == pytest.approx(value, abs=tol)


def test_lifetimes_invert_the_cumulative_hazard():
    rng = np.random.default_rng(0)
    e = rng.exponential(size=2000)
    eta = rng.normal(0, 1.5, 2000)
    offer = rng.integers(0, 2, 2000)
    t = sub.lifetimes(e, eta, offer)
    np.testing.assert_allclose(sub.cumulative_hazard(t, eta, offer), e, rtol=1e-10)


@pytest.mark.parametrize("eta, offer, horizon", [(0.0, 0, 24), (-0.85, 1, 24), (0.4, 1, 2),
                                                 (-0.25, 0, 10)])
def test_rmst_closed_form_matches_numerical_integration(eta, offer, horizon):
    numeric = sum(quad(lambda x: float(sub.survival(x, eta, offer)), a, b, limit=200)[0]
                  for a, b in ((0, min(3, horizon)), (min(3, horizon), horizon)))
    assert float(sub.rmst(horizon, eta, offer)) == pytest.approx(numeric, rel=1e-9)
    assert float(sub.discounted_rmst(horizon, eta, offer, 0.0)[0]) == pytest.approx(numeric,
                                                                                 rel=1e-8)
    assert float(sub.discounted_rmst(horizon, eta, offer, 0.008)[0]) < numeric


def test_true_survival_matches_the_truth(res):
    t = res.truth["survival"]
    np.testing.assert_allclose(sub.true_survival([3, 6, 12, 18, 24]),
                               [t["overall"][k] for k in ("3", "6", "12", "18", "24")],
                               rtol=1e-5)
    assert float(sub.true_survival(0.0)[0]) == 1.0


def test_gain_columns_agree_with_the_truth(res):
    s, t = res.subscribers, res.truth
    target = s["onboarded"] == 0
    assert s.loc[target, "true_rmst24_gain"].mean() == pytest.approx(
        t["rmst"]["gain_target"], abs=0.02)
    assert t["rmst"]["gain_target_sample_mean"] == pytest.approx(
        s.loc[target, "true_rmst24_gain"].mean(), rel=1e-5)
    assert (s["true_rmst24_gain"] > 0).all()


def test_strong_hidden_confounding_builds_without_overflow(res):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = sub.subscriptions(hidden_confounding=1.0)
    s = h.subscribers
    assert (s["plan"].to_numpy() == res.subscribers["plan"].to_numpy()).all()
    assert s["true_tenure_months"].max() > 1000  # centuries: dates exist only if observed
    assert s["cancel_date"].max() <= pd.Timestamp("2026-06-30")
    target = s["onboarded"] == 0
    assert h.truth["rmst"]["gain_target"] == pytest.approx(
        s.loc[target, "true_rmst24_gain"].mean(), abs=0.05)
    # motivation makes onboarders look better than they are
    assert h.truth["rmst"]["observed_groups"]["unadjusted_gain"] > res.truth["rmst"][
        "observed_groups"]["unadjusted_gain"]


def test_reference_solutions_recover_the_truth_on_the_committed_data(subs, res):
    pytest.importorskip("lifelines")
    from mktstats import survival as sv

    committed = pd.read_csv(ROOT / "data" / "synthetic" / "subscriptions_subscribers.csv",
                            parse_dates=["start_date", "cancel_date"])
    pd.testing.assert_frame_equal(committed, res.subscribers, check_dtype=False)
    out = sv.checkpoint_outcomes(subs, res.truth, B=100)
    rules = {k: bool(v) for k, v in out.items() if k.endswith("_rule")}
    assert all(rules.values()), rules
    assert out["km_inside"] == 4 and out["cox_inside"] == 3
    assert out["ph_offer_flagged"] and out["ph_others_flagged"] == 0


# ----------------------------------------------------------------------------- slow sweeps
@pytest.mark.slow
def test_reference_passes_with_the_full_bootstrap(subs, res):
    pytest.importorskip("lifelines")
    from mktstats import survival as sv

    out = sv.checkpoint_outcomes(subs, res.truth, B=200)
    assert out["rmst_rule"] and out["rmst_unadjusted_rule"]


@pytest.mark.slow
def test_checkpoint_coverage_across_seeds():
    """Re-measure the first seed set of the tolerances block (seeds 1000-1039, B = 100)."""
    pytest.importorskip("lifelines")
    from mktstats import survival as sv

    recorded = json.loads(TRUTH_FILE.read_text())["tolerances"]["seed_sets"]["1000-1039"]
    counts = dict.fromkeys(["km_rule", "dashboards_rule", "cox_rule", "cox_unadjusted_rule",
                            "landmark_rule", "rmst_rule", "rmst_unadjusted_rule"], 0)
    for seed in range(1000, 1040):
        r = sub.subscriptions(seed=seed)
        s = sv.add_plan_dummies(sv.tenure_and_churn(r.subscribers))
        out = sv.checkpoint_outcomes(s, r.truth, B=100)
        for k in counts:
            counts[k] += bool(out[k])
    expected = {
        "km_rule": recorded["km_truth_3_of_4"],
        "dashboards_rule": recorded["dashboards_outside"],
        "cox_rule": recorded["cox_onboarded_inside_and_2_of_3"],
        "cox_unadjusted_rule": recorded["cox_unadjusted_outside_and_lower"],
        "landmark_rule": recorded["landmark_3_of_4"],
        "rmst_rule": recorded["rmst_truth_inside_B100"],
        "rmst_unadjusted_rule": recorded["rmst_unadjusted_outside_B100"],
    }
    # One seed of slack for floating-point differences between platforms.
    short = {k: (counts[k], v) for k, v in expected.items() if counts[k] < v - 1}
    assert not short, f"rules held in fewer seeds than recorded: {short}"
    assert all(c >= 36 for c in counts.values()), counts  # at least 90% of 40 seeds
