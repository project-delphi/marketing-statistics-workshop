"""Capstone scenario (Module 12): determinism, one timeline and one channel vocabulary, the answers
recomputed from the parts, and recovery of each stage's answer with the methods the stages use
(synthetic control, BG/NBD MAP, a T-learner; the calibrated MMM is slow)."""

import numpy as np
import pandas as pd
import pytest

from mktstats import recovery, synth
from mktstats.synth import channels
from mktstats.synth.capstone import capstone
from mktstats.synth.mmm import steady_state_response


@pytest.fixture(scope="module")
def cap():
    return capstone()


def _lift_row_from_readout(cap, readout):
    """Stage 2 -> Stage 3: the national-equivalent lift-test row, sigma from placebos that fit
    (relative placebo effects times the treated synthetic total)."""
    sc = cap.truth["scenario"]["geo_test"]
    panel = cap.panel
    pre = panel["date"] < pd.Timestamp(sc["test_start"])
    by_geo = panel[pre].groupby("geo")["sales"].sum()
    f = by_geo[sc["treated_geos"]].sum() / by_geo.sum()
    good = readout["placebo_pre_rmspe"] <= 2 * np.median(readout["placebo_pre_rmspe"])
    sigma = readout["placebo_relative_effects"][good].std(ddof=1) * readout["synthetic_post_total"]
    scale = f * sc["test_weeks"]
    return f, pd.DataFrame([{"channel": "search", "x": sc["base_weekly_spend_national"],
                             "delta_x": sc["spend_change_total"] / scale,
                             "delta_y": readout["incremental"] / scale, "sigma": sigma / scale}])


def test_deterministic(cap):
    again = capstone()
    for name in cap.data:
        pd.testing.assert_frame_equal(cap.data[name], again.data[name])
    assert cap.truth == again.truth
    assert not capstone(seed=1).weekly.equals(cap.weekly)


def test_tables_match_the_module_formats(cap):
    assert list(cap.transactions.columns) == list(synth.retailer().transactions.columns)
    assert list(cap.customers.columns) == list(synth.retailer().customers.columns)
    assert list(cap.weekly.columns) == list(synth.mmm().weekly.columns)
    assert list(cap.panel.columns) == list(synth.geo_panel().panel.columns)
    assert list(cap.experiment.columns) == list(synth.email_experiment().experiment.columns) + [
        "split"]
    # answers differ from the module datasets
    assert cap.truth["retailer"]["pnbd"] != synth.retailer().truth["pnbd"]
    assert not cap.weekly["y"].equals(synth.mmm().weekly["y"])


def test_one_timeline_and_one_vocabulary(cap):
    tr, sc = cap.truth, cap.truth["scenario"]
    gt = sc["geo_test"]
    start = pd.Timestamp(gt["test_start"])
    assert cap.weekly["date_week"].max() == start - pd.Timedelta(weeks=1)  # MMM ends before test
    post_dates = cap.panel.loc[cap.panel["post"] == 1, "date"].unique()
    assert len(post_dates) == gt["test_weeks"] and pd.Timestamp(post_dates.min()) == start
    assert cap.panel["date"].max() == pd.Timestamp(gt["test_end"])
    assert sorted(cap.panel.loc[cap.panel["treated"] == 1, "geo"].unique()) == gt["treated_geos"]
    media = list(tr["mmm"]["channels"])
    assert media == sc["mmm"]["channels"] == list(channels.MEDIA_CHANNELS)
    assert set(cap.customers["acquisition_channel"]) == set(channels.ACQUISITION_CHANNELS)
    assert gt["channel"] in media
    assert set(sc["channel_link"]["media_to_acquisition"]) == set(media)
    # regional sales add up to about the national level
    regional = cap.panel.groupby("date")["sales"].sum().mean()
    assert 0.8 < regional / cap.weekly["y"].iloc[-52:].mean() < 1.25


def test_geo_test_agrees_with_the_mmm_curve(cap):
    tr = cap.truth
    ans, gt = tr["answers"]["stage2_geo_test"], tr["scenario"]["geo_test"]
    row = ans["national_equivalent_lift_test"]
    ct = tr["mmm"]["channels"]["search"]
    curve = steady_state_response([row["x"], row["x"] + row["delta_x"]],
                                  ct["saturation_beta_sales_units"], ct["saturation_lam"],
                                  ct["channel_scale"])
    assert row["delta_y_curve"] == pytest.approx(curve[1] - curve[0], rel=1e-5)
    f = ans["treated_share_of_sales"]
    # realized increments are rounded to whole dollars per geo-week
    assert row["delta_y"] == pytest.approx(row["delta_y_curve"], abs=8 * 0.5 / f + 1)
    assert row["delta_x"] == pytest.approx(gt["spend_change"] * gt["base_weekly_spend_national"])
    assert gt["spend_change_total"] == pytest.approx(
        round(f * row["delta_x"] * gt["test_weeks"], -2))
    inc = sum(tr["geo_panel"]["weekly_increment_by_geo"].values()) * gt["test_weeks"]
    assert ans["incremental_sales"] == pytest.approx(inc)
    camp = ans["campaign"]
    assert camp["incremental_roas"] == pytest.approx(inc / gt["spend_change_total"], rel=1e-5)
    # the share is computable from the panel
    pre = cap.panel["post"] == 0
    by_geo = cap.panel[pre].groupby("geo")["sales"].sum()
    assert by_geo[gt["treated_geos"]].sum() / by_geo.sum() == pytest.approx(f, rel=1e-5)
    # the default test switches search off: sales fall, spend saved, below break-even
    assert gt["spend_change"] == -1.0 and ans["incremental_sales"] < 0
    assert camp["incremental_roas"] < camp["break_even_roas"]


def test_stage1_values_recompute(cap):
    rt = cap.truth["retailer"]
    ans = cap.truth["answers"]["stage1_customer_value"]
    cv = cap.truth["scenario"]["customer_value"]
    for c, v in ans["by_acquisition_channel"].items():
        a = rt["pnbd"]["alpha"] * np.exp(-rt["covariates"]["gamma_purchase"][c])
        b = rt["pnbd"]["beta"] * np.exp(-rt["covariates"]["gamma_dropout"][c])
        ref = channels.new_customer_value_monthly(rt["pnbd"]["r"], a, rt["pnbd"]["s"], b,
                                                  rt["population_mean_spend"], cv["months"],
                                                  cv["monthly_discount_rate"])
        assert v["margin_including_first_purchase"] == pytest.approx(
            0.3 * ref["value_including_first_purchase"], rel=1e-5)
    assert ans["top_channel"] == "referral"


def test_stage4_allocation(cap):
    tr = cap.truth
    a = tr["answers"]["stage4_allocation"]
    clv = a["clv_margin_by_media_channel"]
    ref = channels.long_run_allocation(tr["mmm"], clv)  # from the rounded CLVs in the truth
    for c, v in a["optimal_allocation"].items():
        assert ref["optimal_allocation"][c] == pytest.approx(v, rel=1e-4)
    x = np.array(list(a["optimal_allocation"].values()))
    assert x.sum() == pytest.approx(tr["scenario"]["budget"]["weekly_budget"], rel=1e-6)
    lo = np.array([b[0] for b in a["bounds"].values()])
    hi = np.array([b[1] for b in a["bounds"].values()])
    assert (x >= lo * (1 - 1e-6)).all() and (x <= hi * (1 + 1e-6)).all()
    # the long-run objective moves budget from search to tv by more than the spend tolerance
    tol = a["checkpoint"]["spend_by_channel"]["abs_share_of_budget"] * a["weekly_budget"]
    assert a["change_vs_short_run_optimum"]["tv"] > tol
    assert a["change_vs_short_run_optimum"]["search"] < -tol
    vs = a["value_sensitivity"]
    assert vs["current_share_of_optimal_value"] < 1 - a["checkpoint"]["value"]["rel"]


def test_stage5_policy_values(cap):
    d = cap.experiment
    ans = cap.truth["answers"]["stage5_targeting"]
    assert (d["split"] == "test").sum() == (d["split"] == "train").sum() == 10_000
    test = d[d["split"] == "test"]
    gain = 0.3 * test["true_cate"].to_numpy() - 0.75
    assert ans["test_split"]["treat_all"] == pytest.approx(gain.mean(), rel=1e-4)
    assert ans["test_split"]["oracle"] == pytest.approx(np.maximum(gain, 0).mean(), rel=1e-4)
    assert 0 < ans["test_split"]["treat_all"] < ans["test_split"]["oracle"] / 2


# ----------------------------------------------------------------------------- recovery
def test_stage2_synthetic_control_recovers_the_test(cap):
    sc = cap.truth["scenario"]["geo_test"]
    ans = cap.truth["answers"]["stage2_geo_test"]
    r = recovery.synthetic_control(cap.panel, sc["treated_geos"], sc["test_start"])
    rel = abs(r["incremental"] / ans["incremental_sales"] - 1)
    assert rel < ans["checkpoint"]["incremental_sales"]["rel"] / 2, rel
    assert r["p_value"] <= ans["checkpoint"]["placebo_p_value_max"]
    w = np.array(list(r["weights"].values()))
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)


def test_sc_finds_no_effect_without_one():
    panel = synth.geo_panel(lift_pct=0.0).panel
    tr = sorted(panel.loc[panel["treated"] == 1, "geo"].unique())
    start = panel.loc[panel["post"] == 1, "date"].min()
    r = recovery.synthetic_control(panel, tr, start)
    assert r["p_value"] > 0.1 and abs(r["relative_effect"]) < 0.02


def test_stage1_bgnbd_map_recovers_new_customer_value(cap):
    pytest.importorskip("pymc_marketing")
    import warnings

    from pymc_marketing import clv

    from mktstats import data

    rt = cap.truth["retailer"]
    rfm = data.retailer_rfm(cap.transactions, cap.customers, rt)
    cols = list(rt["covariates"]["columns"].values())
    model = clv.BetaGeoModel(model_config={"purchase_covariate_cols": cols,
                                           "dropout_covariate_cols": cols})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=rfm[["customer_id", "frequency", "recency", "T", *cols]], method="map",
                  progressbar=False)
    _, gg = recovery.fit_gamma_gamma_map(rfm)
    spend = gg["p"] * gg["v"] / (gg["q"] - 1)
    rows = pd.DataFrame({"customer_id": ["search", "social", "referral"],
                         "channel_social": [0, 1, 0], "channel_referral": [0, 0, 1]})
    cv = cap.truth["scenario"]["customer_value"]
    cum = np.array([np.asarray(model.expected_purchases_new_customer(
        data=rows, t=k * cv["weeks_per_month"]).mean(("chain", "draw")))
        for k in range(cv["months"] + 1)])
    disc = (1 + cv["monthly_discount_rate"]) ** -np.arange(1, cv["months"] + 1)
    est = 0.3 * spend * (1 + (np.diff(cum, axis=0) * disc[:, None]).sum(axis=0))
    ans = cap.truth["answers"]["stage1_customer_value"]
    for j, c in enumerate(rows["customer_id"]):
        truth = ans["by_acquisition_channel"][c]["margin_including_first_purchase"]
        assert abs(est[j] / truth - 1) < ans["checkpoint"]["rel"], (c, est[j], truth)
    assert rows["customer_id"][int(np.argmax(est))] == ans["top_channel"]


def test_stage5_learned_rule_beats_mailing_everyone(cap):
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = cap.experiment
    x = pd.get_dummies(d[["recency", "history", "mens", "womens", "zip_code", "newbie",
                          "channel"]], dtype=float)
    train = (d["split"] == "train").to_numpy()
    treated = (d["treatment"] == 1).to_numpy()
    y = d["spend"].to_numpy()

    def fit(mask):
        return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, min_samples_leaf=100,
                                             random_state=0).fit(x[mask], y[mask])

    m1, m0 = fit(train & treated), fit(train & ~treated)
    cate = m1.predict(x[~train]) - m0.predict(x[~train])
    gain = 0.3 * d.loc[~train, "true_cate"].to_numpy() - 0.75
    value = float(((0.3 * cate > 0.75) * gain).mean())
    ans = cap.truth["answers"]["stage5_targeting"]["test_split"]
    assert value > 0 and value >= ans["treat_all"]
    assert value <= ans["oracle"]


@pytest.mark.slow
@pytest.mark.parametrize("draws", [300, 1000])
def test_stage3_calibrated_mmm(cap, draws):
    """Stage 2's readout as a lift test: the uncalibrated MMM overstates search; calibrated, the
    truth is inside the 94% HDI for at least 3 of 4 channels and search moves down."""
    pytest.importorskip("pymc_marketing")
    pytest.importorskip("nutpie")
    sc = cap.truth["scenario"]["geo_test"]
    readout = recovery.synthetic_control(cap.panel, sc["treated_geos"], sc["test_start"])
    _, row = _lift_row_from_readout(cap, readout)
    out = {}
    for label, lt in [("uncalibrated", None), ("calibrated", row)]:
        model = recovery.fit_mmm(cap.weekly, draws=draws, tune=draws, chains=2, lift_tests=lt)
        table = recovery.mmm_recovery_table(recovery.mmm_draws(model, cap.weekly),
                                            cap.truth["mmm"], prob=0.94)
        out[label] = table[table["quantity"] == "roas"].set_index("channel")
        print(f"\n{label} ({draws} draws)\n" + out[label].round(3).to_string())
    print("lift row\n" + row.round(1).to_string())
    unc, cal = out["uncalibrated"], out["calibrated"]
    truth = cap.truth["mmm"]["channels"]["search"]["roas"]
    assert unc.loc["search", "mean"] > 1.3 * truth
    assert cal["inside"].sum() >= 3
    assert cal.loc["search", "mean"] < unc.loc["search", "mean"] - 0.5
