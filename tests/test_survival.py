"""mktstats.survival: the interval checks fail on broken input, the stratified bootstrap keeps
its strata and is reproducible, and the reference solutions give the hand-worked answers.

The last section runs the Module 14 lab's own checkpoint cells (labs/src/14-survival-churn.py)
on wrong answers a learner plausibly writes, and checks the message each one gets."""

import contextlib
import re
import warnings
from pathlib import Path

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


# ---- the lab's checkpoints on wrong answers ----------------------------------------------------

LAB = Path(__file__).resolve().parent.parent / "labs" / "src" / "14-survival-churn.py"
CHECKPOINT = re.compile(r"workshop\.checkpoint\((\d+)\)")


class _Workshop:
    """Enough of the harness to run the lab's cells outside a notebook: a solution cell binds
    the reference as it is defined, and a checkpoint is a plain block whose error propagates."""

    def solution(self, n):
        return lambda obj: obj

    def checkpoint(self, n=None, label=None):
        return contextlib.nullcontext()


def _lab_cells():
    jupytext = pytest.importorskip("jupytext")
    return jupytext.read(LAB, fmt="py:percent").cells


@pytest.fixture(scope="module")
def lab():
    """(namespace after every code cell before Checkpoint 5, run on the reference solutions;
    {n: source of checkpoint n})."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mktstats import checks

    code = [c.source for c in _lab_cells() if c.cell_type == "code" and c.source.strip()]
    checkpoints = {int(m[1]): c for c in code if (m := CHECKPOINT.search(c))}
    ns = {"QUICK": True, "workshop": _Workshop(), "checks": checks}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        for source in code:
            if source is checkpoints[5]:
                break
            exec(compile(source, str(LAB), "exec"), ns)
            plt.close("all")
    return ns, checkpoints


def run_checkpoint(lab, n, **names):
    """Run checkpoint n on a copy of the lab's namespace with ``names`` replaced; returns the
    copy. Pass ``subs=...`` to a function that changes subs, so the fixture's stays intact."""
    ns, checkpoints = lab
    scope = {**ns, **names}
    exec(compile(checkpoints[n], f"<checkpoint {n}>", "exec"), scope)
    return scope


def checkpoint_message(lab, n, **names) -> str:
    with pytest.raises(AssertionError) as err:
        run_checkpoint(lab, n, **names)
    return str(err.value)


# Exercise 1: wrong Kaplan-Meier estimators that pass the two toys or fail them in a new way.


def km_rounded(reference, decimals):
    def km(duration, event):
        t = np.asarray(duration, dtype=float)
        return reference(t.astype(int) if decimals is None else t.round(decimals), event)

    return km


def km_in_order_of_appearance(sort_at_the_end):
    """A loop over the cancellation times in the order they appear; the toys are in order."""

    def km(duration, event):
        t, d = np.asarray(duration, dtype=float), np.asarray(event, dtype=float)
        s, out = 1.0, {}
        for tj in pd.unique(t[d == 1]):
            s *= 1 - d[t == tj].sum() / (t >= tj).sum()
            out[float(tj)] = s
        km = pd.Series(out, name="survival")
        return km.sort_index() if sort_at_the_end else km

    return km


def km_just_before(reference):
    def km(duration, event):
        s = reference(duration, event)
        return pd.Series(np.r_[1.0, s.to_numpy()[:-1]], index=s.index, name="survival")

    return km


def test_checkpoint_1_passes_on_the_reference(lab):
    run_checkpoint(lab, 1)


def test_checkpoint_1_says_not_to_cast_the_tenures(lab):
    msg = checkpoint_message(lab, 1, kaplan_meier=km_rounded(lab[0]["kaplan_meier"], None))
    assert re.search(r"your curve has \d+ times; there are \d+ distinct cancellation", msg)
    assert "Your times are whole numbers." in msg
    assert "without rounding them or casting them (no round(), no astype(int))" in msg
    assert "Pass subs" not in msg  # the checkpoint passes the columns, not the learner


def test_checkpoint_1_says_not_to_round_the_tenures(lab):
    msg = checkpoint_message(lab, 1, kaplan_meier=km_rounded(lab[0]["kaplan_meier"], 1))
    assert "whole numbers" not in msg and "without rounding them or casting them" in msg


def test_checkpoint_1_says_to_sort_before_the_product(lab):
    msg = checkpoint_message(lab, 1, kaplan_meier=km_in_order_of_appearance(False))
    assert "all there but not in ascending order: sort by time before the cumulative" in msg
    msg = checkpoint_message(lab, 1, kaplan_meier=km_in_order_of_appearance(True))
    assert "right times but differs from lifelines'" in msg
    assert "sort before .cumprod(), not after" in msg and "Pass subs" not in msg


def test_checkpoint_1_names_survival_just_before_each_time(lab):
    msg = checkpoint_message(lab, 1, kaplan_meier=km_just_before(lab[0]["kaplan_meier"]))
    assert "returned {2.0: 1.0, 3.0: 0.875, 5.0: 0.75" in msg
    assert "Ŝ just before each time" in msg and "Ŝ(2) = 1 − 1/8 = 0.875" in msg


# Exercise 2: length checks that use assert, or refuse a curve that is just long enough.


def ep_assert(survival, horizon):
    assert len(survival) >= horizon
    return float(np.asarray(survival, dtype=float)[:horizon].sum())


def ep_off_by_one(survival, horizon):
    if len(survival) <= horizon:
        raise ValueError(f"the curve has only {len(survival)} values")
    return float(np.asarray(survival, dtype=float)[:horizon].sum())


def ep_assert_off_by_one(survival, horizon):
    assert len(survival) > horizon, "too short"
    return float(np.asarray(survival, dtype=float)[:horizon].sum())


def test_checkpoint_2_passes_on_the_reference(lab):
    run_checkpoint(lab, 2)


def test_checkpoint_2_says_to_raise_valueerror_not_assert(lab):
    msg = checkpoint_message(lab, 2, expected_periods=ep_assert)
    assert msg.startswith("Checkpoint 2: expected_periods([1, 0.5, 0.25], 4) raised"
                          " AssertionError, not ValueError")
    assert "if len(survival) < horizon: raise ValueError" in msg


def test_checkpoint_2_says_when_to_raise(lab):
    msg = checkpoint_message(lab, 2, expected_periods=ep_off_by_one)
    assert msg.startswith(
        "Checkpoint 2: expected_periods([1, 0.5, 0.25], 3) raised ValueError, but its 3 values,"
        " S(0) to S(2), are enough for a horizon of 3: raise only when len(survival) < horizon."
    )
    assert "not an assert" not in msg
    msg = checkpoint_message(lab, 2, expected_periods=ep_assert_off_by_one)
    assert "raised AssertionError, but its 3 values" in msg
    assert "And raise ValueError, not an assert." in msg


# Exercise 3: the printout before Exercise 4 must not answer Exercise 4's Predict.


def test_checkpoint_3_does_not_give_away_the_offers_truth(lab, capsys):
    capsys.readouterr()
    run_checkpoint(lab, 3)
    out = capsys.readouterr().out
    assert "no single true hazard ratio" in out and "Exercise 4 shows why" in out
    offer = lab[0]["truth"]["model"]["intro_offer"]
    for value in (offer["hr_before"], offer["hr_after"]):
        assert f"{value:.2f}" not in out


# Exercise 4: a split that changes subs in place, and a rerun that does not rebuild it.


def ls_in_place(subs, landmark=3.0):
    late = subs["tenure_months"] > landmark
    after = subs[late].assign(tenure_months=subs.loc[late, "tenure_months"] - landmark)
    before = subs
    before["churned"] = before["churned"].where(~late, 0)
    before["tenure_months"] = before["tenure_months"].clip(upper=landmark)
    return before, after


def _code_after(cells, heading):
    """The first code cell after the markdown cell that starts with ``heading``."""
    at = next(i for i, c in enumerate(cells)
              if c.cell_type == "markdown" and c.source.startswith(heading))
    return next(c.source for c in cells[at:] if c.cell_type == "code")


def test_checkpoint_4_in_place_names_the_cells_that_rebuild_subs(lab):
    ns = lab[0]
    msg = checkpoint_message(lab, 4, landmark_split=ls_in_place, subs=ns["subs"].copy())
    assert msg.startswith("Checkpoint 4: your function changed `subs` in place")
    assert ('rerun the cell under "Run · Tenure, churn and two dashboard numbers" in Part A,'
            " then Part B's first cell") in msg
    assert "Part A's first provided cell" not in msg
    # The cells it names are the ones that build subs.
    cells = _lab_cells()
    assert "subs = add_tenure(" in _code_after(cells, "### Run · Tenure, churn and two dashboard")
    assert "subs = add_plan_dummies(subs)" in _code_after(cells, "# Part B ·")


def test_checkpoint_4_stops_a_rerun_on_a_changed_subs_and_its_advice_works(lab):
    ns = lab[0]
    changed = ls_in_place(ns["subs"].copy())[0]
    msg = checkpoint_message(lab, 4, subs=changed)  # the reference, subs not rebuilt
    assert msg.startswith("Checkpoint 4: `subs` is no longer the table Parts A and B built")
    assert 'rerun the cell under "Run · Tenure, churn and two dashboard numbers"' in msg
    # Following the advice: the two named cells, then the checkpoint, which now passes.
    cells = _lab_cells()
    scope = {**ns, "subs": changed}
    for heading in ("### Run · Tenure, churn and two dashboard", "# Part B ·"):
        exec(compile(_code_after(cells, heading), str(LAB), "exec"), scope)
    run_checkpoint((scope, lab[1]), 4)


# Exercise 5: restricted_mean_survival_time without t=horizon.


def test_checkpoint_5_explains_a_gain_that_is_not_a_number(lab):
    ns = lab[0]
    rmst = ns["restricted_mean_survival_time"]

    def rmst_gain_without_t(subs, horizon, weights):
        w = pd.Series(weights, dtype=float)
        assert abs(w.sum() - 1) < 1e-9
        total = 0.0
        for plan, weight in w.items():
            area = {onb: rmst(ns["fit_km"](subs[(subs["plan"] == plan)
                                                & (subs["onboarded"] == onb)]))
                    for onb in (1, 0)}
            total += weight * (area[1] - area[0])
        return float(total)

    msg = checkpoint_message(lab, 5, rmst_gain=rmst_gain_without_t)
    assert "your gain is nan months" in msg
    assert "restricted_mean_survival_time(kmf) without t" in msg and "Pass t=horizon." in msg
