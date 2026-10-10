"""mktstats.survival: the interval checks fail on broken input, the stratified bootstrap keeps
its strata and is reproducible, and the reference solutions give the hand-worked answers."""

import numpy as np
import pandas as pd
import pytest

from mktstats import survival as sv

lifelines = pytest.importorskip("lifelines")


# ---- ci_contains ---------------------------------------------------------------------------


def test_ci_contains_passes_inside_and_on_the_ends():
    assert sv.ci_contains(0.67, 0.64, 0.73, level=0.95, name="HR")
    assert sv.ci_contains(0.64, 0.64, 0.73)
    assert sv.ci_contains(0.73, 0.64, 0.73)


def test_ci_contains_fails_outside_and_names_the_interval():
    with pytest.raises(AssertionError) as err:
        sv.ci_contains(0.67, 0.535, 0.603, level=0.95, name="the onboarding HR",
                       hint="Adjust for plan.")
    msg = str(err.value)
    assert "0.67" in msg and "above" in msg and "95% confidence interval" in msg
    assert "the onboarding HR" in msg and msg.endswith("Adjust for plan.")
    with pytest.raises(AssertionError, match="below its 90% bootstrap percentile interval"):
        sv.ci_contains(1.0, 2.0, 3.0, level=0.9, kind="bootstrap percentile interval")
    with pytest.raises(AssertionError, match=r"^Checkpoint 5: the true value of x, 1, "):
        sv.ci_contains(1.0, 2.0, 3.0, name="x", label="Checkpoint 5")


@pytest.mark.parametrize(
    "truth, lower, upper, match",
    [
        (0.5, 0.6, 0.4, "lower end is above"),
        (0.5, float("nan"), 0.7, "finite"),
        (float("inf"), 0.1, 0.7, "finite"),
        ("x", 0.1, 0.7, "number"),
    ],
)
def test_ci_contains_rejects_broken_input(truth, lower, upper, match):
    with pytest.raises(AssertionError, match=match):
        sv.ci_contains(truth, lower, upper)


def test_ci_contains_needs_a_level_between_0_and_1():
    with pytest.raises(ValueError):
        sv.ci_contains(0.5, 0.4, 0.6, level=95)


# ---- k_of_K_ci_contains ----------------------------------------------------------------------


TRUTHS = {"standard": 0.779, "premium": 0.549, "onboarded": 0.670}


def test_k_of_K_passes_with_enough_inside():
    out = sv.k_of_K_ci_contains(
        TRUTHS, {"standard": (0.69, 0.79), "premium": (0.46, 0.56), "onboarded": (0.70, 0.80)},
        k=2)
    assert [v[2] for v in out.values()] == [True, True, False]


def test_k_of_K_fails_with_too_few_and_lists_the_misses():
    with pytest.raises(AssertionError) as err:
        sv.k_of_K_ci_contains(
            TRUTHS, {"standard": (0.69, 0.79), "premium": (0.60, 0.70),
                     "onboarded": (0.70, 0.80)},
            k=2, name="the hazard ratios", hint="Use the dummy columns.")
    msg = str(err.value)
    assert "Only 1 of 3" in msg and "premium" in msg and "onboarded" in msg
    assert "standard (" not in msg and msg.endswith("Use the dummy columns.")
    with pytest.raises(AssertionError, match=r"^Checkpoint 3: only 0 of 3 "):
        sv.k_of_K_ci_contains(TRUTHS, {k: (2.0, 3.0) for k in TRUTHS}, k=1,
                              label="Checkpoint 3")


@pytest.mark.parametrize(
    "intervals, match",
    [
        ({"standard": (0.7, 0.8), "premium": (0.5, 0.6)}, "same keys"),
        ({"standard": (0.8, 0.7), "premium": (0.5, 0.6), "onboarded": (0.6, 0.7)}, "above"),
        ({"standard": (0.7,), "premium": (0.5, 0.6), "onboarded": (0.6, 0.7)}, "lower, upper"),
    ],
)
def test_k_of_K_rejects_broken_input(intervals, match):
    with pytest.raises(AssertionError, match=match):
        sv.k_of_K_ci_contains(TRUTHS, intervals, k=2)


def test_k_of_K_needs_a_sensible_k():
    with pytest.raises(ValueError):
        sv.k_of_K_ci_contains(TRUTHS, {k: (0, 1) for k in TRUTHS}, k=4)


# ---- bootstrap_interval ----------------------------------------------------------------------


@pytest.fixture
def grouped():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"g": np.repeat(["a", "b", "c"], [50, 30, 20]),
                         "x": rng.normal(size=100)})


def test_bootstrap_keeps_every_stratum_size(grouped):
    seen = []

    def fn(d):
        seen.append(d["g"].value_counts().to_dict())
        return d["x"].mean()

    sv.bootstrap_interval(fn, grouped, by="g", B=20, seed=1)
    assert all(s == {"a": 50, "b": 30, "c": 20} for s in seen)


def test_bootstrap_without_strata_lets_the_group_sizes_move(grouped):
    sizes = []
    sv.bootstrap_interval(lambda d: sizes.append((d["g"] == "a").sum()) or 0.0, grouped,
                          by=None, B=20, seed=1)
    assert len(set(sizes[1:])) > 1  # sizes[0] is the full data


def test_bootstrap_is_reproducible_and_seeded(grouped):
    a = sv.bootstrap_interval(lambda d: d["x"].mean(), grouped, by="g", B=50, seed=7)
    b = sv.bootstrap_interval(lambda d: d["x"].mean(), grouped, by="g", B=50, seed=7)
    c = sv.bootstrap_interval(lambda d: d["x"].mean(), grouped, by="g", B=50, seed=8)
    np.testing.assert_array_equal(a.draws, b.draws)
    assert not np.array_equal(a.draws, c.draws)
    assert a.estimate == pytest.approx(grouped["x"].mean())
    assert a.lower < a.estimate < a.upper and a.label == "95% bootstrap percentile interval"


def test_bootstrap_interval_of_a_mean_has_the_textbook_width():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"x": rng.normal(size=2000)})
    res = sv.bootstrap_interval(lambda f: f["x"].mean(), d, by=None, B=400, seed=2)
    assert (res.upper - res.lower) / 2 == pytest.approx(1.96 / np.sqrt(2000), rel=0.15)
    np.testing.assert_allclose([res.lower, res.upper],
                               np.quantile(res.draws, [0.025, 0.975]))


def test_bootstrap_refuses_a_statistic_that_is_not_a_number(grouped):
    with pytest.raises(ValueError, match="not finite"):
        sv.bootstrap_interval(lambda d: float("nan"), grouped, by="g", B=5)
    with pytest.raises(ValueError):
        sv.bootstrap_interval(lambda d: 0.0, grouped, by="g", B=5, prob=1.5)
    with pytest.raises(ValueError):
        sv.bootstrap_interval(lambda d: 0.0, grouped.iloc[:0], by="g", B=5)


# ---- reference solutions ----------------------------------------------------------------------


def test_kaplan_meier_on_the_hand_worked_toys():
    km = sv.kaplan_meier([2, 3, 3, 5, 6, 6, 8, 9], [1, 1, 0, 1, 0, 1, 1, 0])
    expected = pd.Series({2.0: 0.875, 3.0: 0.75, 5.0: 0.6, 6.0: 0.45, 8.0: 0.225})
    np.testing.assert_allclose(km.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert list(km.index) == list(expected.index) and km.name == "survival"
    tie = sv.kaplan_meier([1, 1, 2], [1, 1, 0])
    assert list(tie.index) == [1.0] and tie.iloc[0] == pytest.approx(1 / 3, abs=1e-12)


def test_kaplan_meier_matches_lifelines_with_ties():
    rng = np.random.default_rng(5)
    t = rng.integers(1, 60, size=500) / 7.0
    e = rng.random(500) < 0.6
    km = sv.kaplan_meier(t, e)
    kmf = lifelines.KaplanMeierFitter().fit(t, e)
    lib = kmf.survival_function_["KM_estimate"].reindex(km.index)
    np.testing.assert_allclose(km.to_numpy(), lib.to_numpy(), atol=1e-12)
    assert set(km.index) == set(np.unique(t[e]))


def test_expected_periods_sums_from_period_0_and_refuses_a_short_curve():
    assert sv.expected_periods([1, 0.5, 0.25], 3) == pytest.approx(1.75)
    assert sv.expected_periods([1, 0.5, 0.25], 2) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="periods 0 to 2"):
        sv.expected_periods([1, 0.5, 0.25], 4)


def toy_subs():
    return pd.DataFrame({
        "plan": ["basic", "basic", "basic", "premium", "premium", "basic", "premium", "basic"],
        "onboarded": [1, 0, 1, 0, 1, 0, 1, 0],
        "tenure_months": [1.0, 2.5, 4.0, 3.5, 7.0, 0.5, 2.0, 6.0],
        "churned": [1, 1, 0, 1, 0, 1, 1, 0],
    })


def test_landmark_split_stops_the_clock_and_restarts_it():
    subs = toy_subs()
    copy = subs.copy()
    before, after = sv.landmark_split(subs, 3.0)
    pd.testing.assert_frame_equal(subs, copy)
    assert len(before) == len(subs) and before["tenure_months"].max() == 3.0
    assert not ((before["churned"] == 1) & (subs["tenure_months"] > 3)).any()
    assert list(after.index) == list(subs.index[subs["tenure_months"] > 3])
    np.testing.assert_allclose(after["tenure_months"], [1.0, 0.5, 4.0, 3.0])


def test_rmst_gain_asserts_its_weights_and_weights_the_plans():
    subs = toy_subs()
    with pytest.raises(AssertionError, match="sum to 1"):
        sv.rmst_gain(subs, 6, {"basic": 0.25, "premium": 0.25})
    basic = sv.rmst_gain(subs, 6, {"basic": 1.0})
    premium = sv.rmst_gain(subs, 6, {"premium": 1.0})
    assert sv.rmst_gain(subs, 6, {"basic": 0.5, "premium": 0.5}) == pytest.approx(
        (basic + premium) / 2)
    # basic, onboarded: S = 0.5 from t = 1 (censored at 4) -> area 1 + 0.5 * 5 = 3.5;
    # not onboarded: three at risk, S = 2/3 from 0.5, 1/3 from 2.5 (censored at 6) -> 0.5 +
    # 2/3 * 2 + 1/3 * 3.5 = 3.0
    assert basic == pytest.approx(3.5 - 3.0)


def test_tenure_churn_dashboards_and_monthly_rates():
    raw = pd.DataFrame({
        "start_date": pd.to_datetime(["2026-01-01", "2025-01-01", "2026-05-01"]),
        "cancel_date": pd.to_datetime(["2026-03-01", None, None]),
    })
    subs = sv.tenure_and_churn(raw, "2026-06-30")
    assert subs["churned"].tolist() == [1, 0, 0]
    np.testing.assert_allclose(subs["tenure_months"], np.array([59, 545, 60]) / 30.4375)
    a, b = sv.dashboard_rates(subs, 12)
    assert a == pytest.approx(2 / 3) and b == pytest.approx(1 / 2)
    subs["intro_offer"] = [1, 0, 0]
    rates = sv.monthly_churn_rate(subs, months=3)
    month2 = rates[(rates["intro_offer"] == 1) & (rates["month"] == 2)].iloc[0]
    assert month2["active"] == 1 and month2["cancellations"] == 1 and month2["rate"] == 1.0
    assert rates.loc[rates["intro_offer"] == 0, "cancellations"].sum() == 0


def test_cox_hazard_ratios_shape_and_scale():
    rng = np.random.default_rng(11)
    x = rng.integers(0, 2, 2000)
    t = rng.exponential(1 / np.exp(-0.5 * x))
    d = pd.DataFrame({"x": x, "z": rng.normal(size=2000), "tenure_months": t,
                      "churned": np.ones(2000, int)})
    out = sv.cox_hazard_ratios(d, ["x"])
    assert list(out.index) == ["x"] and list(out.columns) == ["hr", "lower", "upper"]
    assert out.loc["x", "lower"] < np.exp(-0.5) < out.loc["x", "upper"]
    two = sv.cox_hazard_ratios(d, ["z", "x"], alpha=0.1)
    assert list(two.index) == ["z", "x"]
    assert (two["upper"] - two["lower"]).loc["x"] < (out["upper"] - out["lower"]).loc["x"]
