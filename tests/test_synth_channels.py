"""Shared channel vocabulary and the long-run (CLV-weighted) allocation answer key."""

import numpy as np
import pytest
from scipy.integrate import quad

from mktstats import synth
from mktstats.synth import channels
from mktstats.synth.mmm import (
    long_run_value,
    optimal_allocation,
    optimal_allocation_value,
    steady_state_response,
)


@pytest.fixture(scope="module")
def block():
    return synth.channel_value(synth.retailer().truth,
                               {"mmm": synth.mmm().truth,
                                "mmm_confounded": synth.mmm_confounded().truth})


def _arrays(mmm_truth):
    ch = list(mmm_truth["channels"])
    al = mmm_truth["true_optimal_allocation"]
    get = lambda k: np.array([mmm_truth["channels"][c][k] for c in ch])  # noqa: E731
    return (ch, get("saturation_beta_sales_units"), get("saturation_lam"), get("channel_scale"),
            np.array([al["bounds"][c][0] for c in ch]), np.array([al["bounds"][c][1] for c in ch]),
            al["weekly_budget"])


def test_vocabulary_and_mapping(block):
    assert block["media_channels"] == list(synth.mmm().truth["channels"])
    assert block["acquisition_channels"] == list(synth.retailer().truth["channel_probs"])
    for m, shares in block["media_to_acquisition"].items():
        assert m in block["media_channels"]
        assert set(shares) == set(block["acquisition_channels"])
        assert sum(shares.values()) == pytest.approx(1.0)
    for m, n in block["new_customers_per_dollar"].items():
        assert n == pytest.approx(1 / block["cost_per_new_customer"][m], rel=1e-5)
    clv_a = block["clv_margin_by_acquisition_channel"]
    for m, v in block["clv_margin_by_media_channel"].items():
        mix = block["media_to_acquisition"][m]
        assert v == pytest.approx(sum(w * clv_a[a] for a, w in mix.items()), rel=1e-5)
    new = synth.retailer().truth["value_by_channel"]["new_customer"]
    for a, v in clv_a.items():
        assert v == pytest.approx(0.3 * new[a]["discounted_clv_excluding_first_purchase"],
                                  rel=1e-5)


@pytest.mark.parametrize("name", ["mmm", "mmm_confounded"])
def test_long_run_optimum_is_optimal(block, name):
    mt = synth.mmm().truth if name == "mmm" else synth.mmm_confounded().truth
    lr = block["long_run_optimal_allocation"][name]
    ch, bb, lam, sc, lo, hi, budget = _arrays(mt)
    lin = np.array([lr["long_run_value_per_dollar_of_new_customers"][c] for c in ch])
    x = np.array([lr["optimal_allocation"][c] for c in ch])
    assert x.sum() == pytest.approx(budget, rel=1e-5)
    assert (x >= lo * (1 - 1e-6)).all() and (x <= hi * (1 + 1e-6)).all()

    def value(plan):
        return long_run_value(plan, bb, lam, sc, margin=0.3, linear=lin).sum()

    best = value(x)
    assert best == pytest.approx(lr["weekly_value_optimal"], rel=1e-5)
    assert lr["weekly_value_optimal"] >= lr["weekly_value_short_run_optimum"]
    assert lr["weekly_value_short_run_optimum"] > lr["weekly_value_current"]
    rng = np.random.default_rng(1)
    for _ in range(2000):
        cand = lo + rng.dirichlet(np.ones(len(ch))) * (hi - lo)
        cand = cand / cand.sum() * budget
        if (cand < lo).any() or (cand > hi).any():
            continue
        assert value(cand) <= best + 1e-6
    interior = [c for c in ch if lr["at_bound"][c] == "interior"]
    m = [lr["marginal_value_at_optimum"][c] for c in interior]
    assert interior and np.ptp(m) < 1e-4 * max(m)
    # the CLV term moves budget towards tv, whose new customers are worth most
    assert lr["change_vs_short_run_optimum"]["tv"] > 0


def test_value_solver_reduces_to_the_short_run_solver():
    mt = synth.mmm().truth
    _, bb, lam, sc, lo, hi, budget = _arrays(mt)
    x0, eta0 = optimal_allocation(budget, bb, lam, sc, lo, hi)
    x1, eta1 = optimal_allocation_value(budget, bb, lam, sc, lo, hi, margin=1.0)
    np.testing.assert_allclose(x1, x0, rtol=1e-6)
    x2, eta2 = optimal_allocation_value(budget, bb, lam, sc, lo, hi, margin=0.3)
    np.testing.assert_allclose(x2, x0, rtol=1e-6)  # a common margin does not move the optimum
    assert eta2 == pytest.approx(0.3 * eta0, rel=1e-6)


def test_raising_one_clv_never_lowers_its_spend():
    """Module 10 Exercise 6's checkpoint, on the true curves."""
    mt = synth.mmm().truth
    ch, bb, lam, sc, lo, hi, budget = _arrays(mt)
    base = np.array([0.2, 0.2, 0.2, 0.2])
    x_base, _ = optimal_allocation_value(budget, bb, lam, sc, lo, hi, margin=0.3, linear=base)
    for j in range(len(ch)):
        for bump in (0.05, 0.2, 1.0):
            lin = base.copy()
            lin[j] += bump
            x, _ = optimal_allocation_value(budget, bb, lam, sc, lo, hi, margin=0.3, linear=lin)
            assert x[j] >= x_base[j] - 1e-6 * budget, (ch[j], bump)
            assert x.sum() == pytest.approx(budget, rel=1e-9)


def test_long_run_value_with_zero_clv_is_margin_times_response():
    mt = synth.mmm().truth
    _, bb, lam, sc, *_ = _arrays(mt)
    x = np.array([30_000.0, 20_000.0, 10_000.0, 5_000.0])
    np.testing.assert_allclose(long_run_value(x, bb, lam, sc, margin=0.3),
                               0.3 * steady_state_response(x, bb, lam, sc))


def test_new_customer_value_monthly_matches_quadrature():
    r, a, s, b, m = 0.8, 5.0, 0.5, 10.0, 30.0
    v = channels.new_customer_value_monthly(r, a, s, b, m, months=36, monthly_rate=0.01)
    wpm = channels.WEEKS_PER_MONTH
    rate = lambda u: r / a * (b / (b + u)) ** s  # noqa: E731  (purchases per week at age u)
    direct = sum(m * quad(rate, (k - 1) * wpm, k * wpm)[0] / 1.01**k for k in range(1, 37))
    assert v["value_excluding_first_purchase"] == pytest.approx(direct, rel=1e-8)
    assert v["value_including_first_purchase"] == pytest.approx(direct + m, rel=1e-8)
    assert v["expected_repeat_purchases"] == pytest.approx(quad(rate, 0, 36 * wpm)[0], rel=1e-8)
    # s = 1 branch
    v1 = channels.pnbd_expected_purchases_new(52.0, r, a, 1.0, b)
    assert v1 == pytest.approx(quad(lambda u: r / a * b / (b + u), 0, 52)[0], rel=1e-8)
    # no discounting: value = spend * expected repeat purchases
    v0 = channels.new_customer_value_monthly(r, a, s, b, m, months=12, monthly_rate=0.0)
    assert v0["value_excluding_first_purchase"] == pytest.approx(
        m * v0["expected_repeat_purchases"], rel=1e-10)
