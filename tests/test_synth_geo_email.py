"""Geo panel and e-mail experiment generators: determinism, shape and invariants."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mktstats import synth
from mktstats.synth.email import HISTORY_LABELS, _f_tau_m


@pytest.fixture(scope="module")
def geo():
    return synth.geo_panel()


@pytest.fixture(scope="module")
def email():
    return synth.email_experiment()


def test_geo_deterministic(geo):
    again = synth.geo_panel()
    pd.testing.assert_frame_equal(geo.panel, again.panel)
    assert geo.truth == again.truth
    assert not synth.geo_panel(seed=1).panel.equals(geo.panel)


def test_geo_shape_and_design(geo):
    p, tr = geo.panel, geo.truth
    assert list(p.columns) == ["date", "geo", "region", "sales", "treated", "post"]
    assert len(p) == 40 * 104 and p.groupby("geo").size().eq(104).all()
    assert (p["sales"] > 0).all()
    treated = sorted(p.loc[p["treated"] == 1, "geo"].unique())
    assert treated == tr["treated_geos"] and len(treated) == 8
    post_dates = p.loc[p["post"] == 1, "date"].unique()
    assert len(post_dates) == tr["test_weeks"] == 10
    assert pd.Timestamp(post_dates.min()) == pd.Timestamp(tr["test_start"])
    assert (p.groupby("geo")["treated"].nunique() == 1).all()  # treatment is a geo attribute


def test_geo_incremental_sales_consistent_with_data(geo):
    p, tr = geo.panel, geo.truth
    window = p.loc[(p["treated"] == 1) & (p["post"] == 1), "sales"].sum()
    lift = tr["lift_pct"] / 100
    # observed = round(y0 * (1 + lift)), so sum(y0 * lift) = window * lift / (1 + lift) up to
    # rounding (80 treated geo-weeks, at most 0.5 each)
    assert tr["incremental_sales"] == pytest.approx(window * lift / (1 + lift), abs=80 * 0.5 + 1)
    assert tr["counterfactual_sales_treated_window"] == pytest.approx(window / (1 + lift),
                                                                      rel=1e-4)


def test_geo_campaign_cost_and_return(geo):
    p, tr = geo.panel, geo.truth
    camp = tr["campaign"]
    pre = p[(p["treated"] == 1) & (p["post"] == 0)].groupby("geo")["sales"].mean()
    rate = camp["spend_rate_of_pre_period_sales"]
    for g, s in camp["weekly_spend_by_geo"].items():
        assert s == pytest.approx(round(rate * pre[g], -1))
    assert sorted(camp["weekly_spend_by_geo"]) == tr["treated_geos"]
    assert camp["cost"] == pytest.approx(sum(camp["weekly_spend_by_geo"].values())
                                         * tr["test_weeks"])
    assert camp["incremental_sales"] == tr["incremental_sales"]
    assert camp["incremental_roas"] == pytest.approx(tr["incremental_sales"] / camp["cost"],
                                                     rel=1e-5)
    assert camp["net_return"] == pytest.approx(
        camp["margin"] * tr["incremental_sales"] - camp["cost"], rel=1e-5)
    assert camp["break_even_roas"] == pytest.approx(1 / camp["margin"], rel=1e-5)
    # a decision with tension: profitable at the truth, but not by much
    assert camp["break_even_roas"] < camp["incremental_roas"] < 2 * camp["break_even_roas"]


def test_email_deterministic(email):
    again = synth.email_experiment()
    pd.testing.assert_frame_equal(email.experiment, again.experiment)
    assert email.truth == again.truth


def test_email_columns_and_randomization(email):
    d = email.experiment
    assert list(d.columns) == ["customer_id", "recency", "history_segment", "history", "mens",
                               "womens", "zip_code", "newbie", "channel", "treatment",
                               "conversion", "spend", "true_cate"]
    assert len(d) == 20000
    assert abs(d["treatment"].mean() - 0.5) < 0.02
    assert d["recency"].between(1, 12).all() and (d["history"] >= 29.99).all()
    assert ((d["mens"] == 1) | (d["womens"] == 1)).all()
    assert set(d["zip_code"]) == {"Urban", "Surburban", "Rural"}
    assert set(d["channel"]) == {"Phone", "Web", "Multichannel"}
    assert set(d["history_segment"]) <= set(HISTORY_LABELS)
    assert ((d["spend"] > 0) == (d["conversion"] == 1)).all()
    # covariates are balanced across arms (randomized)
    for col in ["recency", "womens", "newbie"]:
        diff = d.groupby("treatment")[col].mean().diff().iloc[-1]
        assert abs(diff) < 0.1 * d[col].std() + 0.02, col


def test_email_true_cate_recomputes(email):
    d, tr = email.experiment, email.truth
    f, tau, m = _f_tau_m(d["recency"].to_numpy(), d["history"].to_numpy(), d["mens"].to_numpy(),
                         d["womens"].to_numpy(), d["zip_code"].to_numpy(), d["newbie"].to_numpy(),
                         d["channel"].to_numpy())
    cate = (expit(f + tau) - expit(f)) * m
    np.testing.assert_allclose(d["true_cate"], cate, atol=5e-5)
    assert tr["ate"] == pytest.approx(d["true_cate"].mean(), rel=1e-5)
    assert 0.05 < tr["share_negative_cate"] < 0.4  # some customers should not get the offer
    pv = tr["policy_value_per_customer"]
    assert pv["oracle"] >= max(pv["treat_all"], pv["treat_none"])
    # the experiment shows the effect: treated mean spend minus control mean spend near the ATE
    diff = d.groupby("treatment")["spend"].mean().diff().iloc[-1]
    assert diff == pytest.approx(tr["ate"], abs=1.0)
