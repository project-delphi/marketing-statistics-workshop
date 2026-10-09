"""Confounded MMM (Module 9): determinism, what changes against the clean MMM, the latent demand
shock, the omitted-variable-bias mechanism in the truth, the dated lift tests, and (slow) that an
uncalibrated pymc-marketing MMM overstates search while the search lift tests bring it back."""

import numpy as np
import pandas as pd
import pytest

from mktstats import synth
from mktstats.synth.mmm import LIFT_TEST_PLAN, steady_state_response


@pytest.fixture(scope="module")
def conf():
    return synth.mmm_confounded()


@pytest.fixture(scope="module")
def clean():
    return synth.mmm()


def test_deterministic(conf):
    again = synth.mmm(confounded=True)
    for name in conf.data:
        pd.testing.assert_frame_equal(conf.data[name], again.data[name])
    assert conf.truth == again.truth
    assert not synth.mmm_confounded(seed=1).weekly.equals(conf.weekly)


def test_only_search_and_sales_differ_from_the_clean_mmm(conf, clean):
    w, c = conf.weekly, clean.weekly
    assert list(w.columns) == list(c.columns)
    for col in ["date_week", "tv", "social", "display", "price_index", "holiday", "t"]:
        pd.testing.assert_series_equal(w[col], c[col])
    assert not w["search"].equals(c["search"]) and not w["y"].equals(c["y"])
    assert (w["search"] > 0).all()
    # the clean generator's outputs are untouched by the new option
    assert set(clean.data) == {"weekly", "lift_tests"}
    assert "confounding" not in clean.truth


def test_latent_demand_shock(conf):
    lat, w, tr = conf.latent, conf.weekly, conf.truth
    assert list(lat.columns) == ["date_week", "demand_shock", "demand_effect",
                                 "media_contribution_tv", "media_contribution_search",
                                 "media_contribution_social", "media_contribution_display"]
    d = lat["demand_shock"].to_numpy()
    assert abs(d.mean()) < 1e-5 and d.std() == pytest.approx(1.0, abs=1e-5)
    assert np.corrcoef(d[1:], d[:-1])[0, 1] > 0.5  # persistent, AR(1) with phi 0.7
    sales_effect = tr["confounding"]["demand_shock"]["sales_effect_per_sd"]
    np.testing.assert_allclose(lat["demand_effect"], sales_effect * d, atol=0.01)
    corr = np.corrcoef(d, w["search"])[0, 1]
    assert corr == pytest.approx(tr["confounding"]["corr_demand_spend"], rel=1e-4)
    for ch in ["tv", "social", "display"]:
        assert abs(np.corrcoef(d, w[ch])[0, 1]) < 0.25, ch
    # the true ROAS is causal: noiseless media contribution / spend, demand effect excluded
    for ch, ct in tr["channels"].items():
        roas = lat[f"media_contribution_{ch}"].sum() / w[ch].sum()
        assert roas == pytest.approx(ct["roas"], rel=1e-5), ch
    assert tr["channels"]["search"]["roas"] == pytest.approx(3.0)


def test_omitted_variable_bias_in_truth(conf):
    c = conf.truth["confounding"]
    r = c["roas"]
    # omitting the demand shock inflates search's ROAS a lot, the others hardly
    assert r["search"]["omitted_variable_bias"] > 0.5 * r["search"]["true"]
    assert r["search"]["naive_ols"] > 1.5 * r["search"]["true"]
    for ch in ["tv", "social", "display"]:
        assert abs(r[ch]["omitted_variable_bias"]) < r["search"]["omitted_variable_bias"] / 4
    # OLS identity: naive = oracle + coef_d * delta, in ROAS units for search
    ovb = c["ovb_identity"]
    ct = conf.truth["channels"]["search"]
    per_b = r["search"]["true"] / ct["saturation_beta_sales_units"]  # ROAS per unit of B
    implied = ovb["oracle_coef_d"] * ovb["delta"]["search"] * per_b
    assert implied == pytest.approx(r["search"]["omitted_variable_bias"], rel=1e-3)


def test_dated_lift_tests(conf):
    lt, tr = conf.lift_tests, conf.truth
    assert list(lt.columns) == ["channel", "x", "delta_x", "delta_y", "sigma", "date",
                                "test_start", "test_weeks", "true_delta_y"]
    assert len(lt) == len(LIFT_TEST_PLAN)
    assert sorted(lt["channel"].unique()) == sorted(tr["channels"])
    assert (lt["channel"] == "search").sum() == 2
    assert lt["date"].is_monotonic_increasing
    assert (lt["delta_x"] * lt["delta_y"] > 0).all()  # add_lift_test_measurements' monotonicity
    assert (lt["sigma"] > 0).all() and (lt["x"] > 0).all()
    dates = conf.weekly["date_week"]
    assert lt["date"].isin(dates).all() and lt["test_start"].isin(dates).all()
    assert ((lt["date"] - lt["test_start"]).dt.days == 7 * (lt["test_weeks"] - 1)).all()
    off = lt[lt["delta_x"] < 0]
    assert len(off) == 1 and off["channel"].item() == "search"
    assert (off["x"] + off["delta_x"]).item() == 0  # search switched off
    for row in lt.itertuples():
        ct = tr["channels"][row.channel]
        curve = steady_state_response([row.x, row.x + row.delta_x],
                                      ct["saturation_beta_sales_units"], ct["saturation_lam"],
                                      ct["channel_scale"])
        assert row.true_delta_y == pytest.approx(curve[1] - curve[0], abs=1.0)
        assert row.sigma == pytest.approx(tr["lift_tests"]["rel_sigma"] * abs(row.true_delta_y),
                                          abs=1.0)
        assert abs(row.delta_y - row.true_delta_y) < 4 * row.sigma
    # both search tests are known before week 101, so CV folds with n_init >= 101 keep them
    week = {d: i for i, d in enumerate(dates)}
    assert max(week[d] for d in lt.loc[lt["channel"] == "search", "date"]) <= 100
    assert len(tr["lift_tests"]["rows"]) == len(lt)


def test_lift_tests_are_accepted_by_pymc_marketing(conf):
    pytest.importorskip("pymc_marketing")
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    model = MMM(date_column="date_week", channel_columns=["tv", "search", "social", "display"],
                target_column="y", control_columns=["price_index", "holiday", "t"],
                adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
                yearly_seasonality=2)
    w = conf.weekly
    model.build_model(w.drop(columns=["y"]), w["y"])
    # the whole table, extra columns (date, test_start, test_weeks, true_delta_y) included
    model.add_lift_test_measurements(conf.lift_tests)
    assert "lift_measurements" in model.model.named_vars


def test_confounded_allocation_answer_key(conf):
    al = conf.truth["true_optimal_allocation"]
    x = np.array(list(al["optimal_allocation"].values()))
    assert x.sum() == pytest.approx(al["weekly_budget"], rel=1e-5)
    interior = [c for c, b in al["at_bound"].items() if b == "interior"]
    m = [al["marginal_roas_at_optimum"][c] for c in interior]
    assert len(interior) >= 1 and np.ptp(m) < 1e-4 * max(m)


def test_rejects_unknown_confounded_channel():
    with pytest.raises(ValueError, match="confounded_channel"):
        synth.mmm(confounded=True, confounded_channel="radio")


# ----------------------------------------------------------------------------- slow (MCMC)
@pytest.mark.slow
@pytest.mark.parametrize("draws", [300, 1000])
def test_calibration_fixes_search(draws):
    """Module 9 Exercises 2-3 on the committed data, QUICK (300) and FULL (1000) draws: the
    uncalibrated 94% HDI for search ROAS lies above the truth; the other channels are covered; with
    the two search lift tests the truth is inside a narrower HDI."""
    pytest.importorskip("pymc_marketing")
    pytest.importorskip("nutpie")
    from mktstats import data, recovery

    truth = data.load_truth()["mmm_confounded"]
    res = synth.mmm_confounded()  # the committed CSVs are its default-seed output
    weekly, lift = res.weekly, res.lift_tests
    out = {}
    for label, lt in [("uncalibrated", None), ("calibrated", lift[lift["channel"] == "search"])]:
        model = recovery.fit_mmm(weekly, draws=draws, tune=draws, chains=2, lift_tests=lt)
        table = recovery.mmm_recovery_table(recovery.mmm_draws(model, weekly), truth, prob=0.94)
        out[label] = table[table["quantity"] == "roas"].set_index("channel")
        print(f"\n{label} ({draws} draws)\n" + out[label].round(3).to_string())
    unc, cal = out["uncalibrated"], out["calibrated"]
    assert unc.loc["search", "hdi_low"] > truth["channels"]["search"]["roas"]
    assert unc.drop("search")["inside"].sum() >= 2
    assert cal.loc["search", "inside"]
    width = cal["hdi_high"] - cal["hdi_low"]
    assert width["search"] < (unc["hdi_high"] - unc["hdi_low"])["search"]
