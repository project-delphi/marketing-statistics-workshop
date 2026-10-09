"""MMM generator: determinism, shape, the truth recomputed from the data, agreement of the numpy
transforms with pymc-marketing's, lift-test format and the true optimal allocation."""

import numpy as np
import pandas as pd
import pytest

from mktstats import synth
from mktstats.synth.mmm import optimal_allocation, steady_state_response


@pytest.fixture(scope="module")
def res():
    return synth.mmm()


def _true_parts(res):
    tr, w = res.truth, res.weekly
    out = {}
    for c, ct in tr["channels"].items():
        ad = synth.geometric_adstock(w[c].to_numpy(float), ct["adstock_alpha"], tr["l_max"])
        out[c] = ct["saturation_beta_sales_units"] * synth.logistic_saturation(
            ad / w[c].max(), ct["saturation_lam"])
    return out


def test_deterministic(res):
    again = synth.mmm()
    pd.testing.assert_frame_equal(res.weekly, again.weekly)
    pd.testing.assert_frame_equal(res.lift_tests, again.lift_tests)
    assert res.truth == again.truth
    assert not synth.mmm(seed=1).weekly.equals(res.weekly)


def test_shape_and_spend_patterns(res):
    w = res.weekly
    assert list(w.columns) == ["date_week", "tv", "search", "social", "display", "price_index",
                               "holiday", "t", "y"]
    assert len(w) == 156 and (w["date_week"].diff().dropna() == pd.Timedelta(days=7)).all()
    assert (w[["tv", "search", "social", "display"]] >= 0).all().all()
    assert 0.2 < (w["tv"] == 0).mean() < 0.7  # flighting: tv is off part of the time
    assert (w[["search", "social", "display"]] > 0).all().all()
    assert w.groupby(w["date_week"].dt.year)["holiday"].sum().between(1, 3).all()


def test_truth_recomputes_from_data(res):
    tr, w = res.truth, res.weekly
    parts = _true_parts(res)
    for c, ct in tr["channels"].items():
        assert parts[c].sum() / w[c].sum() == pytest.approx(ct["roas"], rel=1e-5)
        assert ct["channel_scale"] == pytest.approx(w[c].max(), rel=1e-6)
        assert ct["saturation_beta_model_units"] == pytest.approx(
            ct["saturation_beta_sales_units"] / w["y"].max(), rel=1e-5)
        assert ct["roas_with_carryover"] >= ct["roas"]
    shares = [ct["contribution_share"] for ct in tr["channels"].values()]
    assert sum(shares) == pytest.approx(1.0, abs=1e-5)
    media = sum(parts.values())
    assert media.sum() / w["y"].sum() == pytest.approx(tr["media_share_of_sales"], rel=1e-4)
    assert 0.15 < tr["media_share_of_sales"] < 0.5


def test_numpy_transforms_match_pymc_marketing():
    pytest.importorskip("pymc_marketing")
    import pytensor.tensor as pt
    import pytensor.xtensor as ptx
    from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

    x = np.random.default_rng(0).uniform(size=40)
    xt = ptx.as_xtensor(pt.as_tensor_variable(x), dims=("date",))
    ref = np.asarray(geometric_adstock(xt, alpha=0.6, l_max=8, normalize=True, dim="date").eval())
    np.testing.assert_allclose(synth.geometric_adstock(x, 0.6, 8), ref, atol=1e-12)
    ref = np.asarray(logistic_saturation(xt, lam=2.0).eval())
    np.testing.assert_allclose(synth.logistic_saturation(x, 2.0), ref, atol=1e-12)


def test_lift_tests_format_and_truth(res):
    lt, tr = res.lift_tests, res.truth
    assert {"channel", "x", "delta_x", "delta_y", "sigma"} <= set(lt.columns)
    assert set(lt["channel"]) == set(tr["channels"])
    assert (lt[["x", "delta_x", "delta_y", "sigma"]] > 0).all().all()
    for row in lt.itertuples():
        ct = tr["channels"][row.channel]
        curve = steady_state_response([row.x, row.x + row.delta_x],
                                      ct["saturation_beta_sales_units"], ct["saturation_lam"],
                                      ct["channel_scale"])
        assert row.true_delta_y == pytest.approx(curve[1] - curve[0], abs=1.0)
        assert abs(row.delta_y - row.true_delta_y) < 4 * row.sigma


def test_true_optimal_allocation(res):
    tr = res.truth
    al = tr["true_optimal_allocation"]
    ch = list(tr["channels"])
    x = np.array([al["optimal_allocation"][c] for c in ch])
    lo = np.array([al["bounds"][c][0] for c in ch])
    hi = np.array([al["bounds"][c][1] for c in ch])
    assert x.sum() == pytest.approx(al["weekly_budget"], rel=1e-5)
    assert (x >= lo * (1 - 1e-6)).all() and (x <= hi * (1 + 1e-6)).all()
    bb = np.array([tr["channels"][c]["saturation_beta_sales_units"] for c in ch])
    lam = np.array([tr["channels"][c]["saturation_lam"] for c in ch])
    sc = np.array([tr["channels"][c]["channel_scale"] for c in ch])
    best = steady_state_response(x, bb, lam, sc).sum()
    assert best == pytest.approx(al["weekly_contribution_optimal"], rel=1e-5)
    assert al["weekly_contribution_optimal"] > al["weekly_contribution_current"]
    # no random feasible reallocation does better
    rng = np.random.default_rng(0)
    for _ in range(2000):
        cand = lo + rng.dirichlet(np.ones(len(ch))) * (hi - lo)
        cand = cand / cand.sum() * al["weekly_budget"]
        if (cand < lo).any() or (cand > hi).any():
            continue
        assert steady_state_response(cand, bb, lam, sc).sum() <= best + 1e-6
    # interior channels share the same marginal return
    interior = [c for c in ch if al["at_bound"][c] == "interior"]
    m = [al["marginal_roas_at_optimum"][c] for c in interior]
    assert np.ptp(m) < 1e-4 * max(m)


def test_optimal_allocation_solver_on_a_symmetric_case():
    x, eta = optimal_allocation(100.0, [1, 1], [2, 2], [50, 50], [0, 0], [100, 100])
    np.testing.assert_allclose(x, [50, 50], rtol=1e-6)
    with pytest.raises(ValueError):
        optimal_allocation(500.0, [1, 1], [2, 2], [50, 50], [0, 0], [100, 100])
