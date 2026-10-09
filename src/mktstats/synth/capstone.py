"""Module 12: one coherent retailer scenario with hidden answers for every stage.

``capstone(seed)`` runs the workshop's generators with their own seeds and parameters (so module
answers cannot be copied) and ties them together:

* **customers** (``retailer``): acquisition channels search / social / referral. Stage 1 values a
  new customer by acquisition channel over 36 months at 1% a month (Module 5's convention:
  acquisition purchase undiscounted, then monthly discounted repeat purchases), in margin.
* **weekly MMM** (``mmm(confounded=True)``): media channels tv / search / social / display, with
  search spend following a hidden demand shock (Module 9's mechanism), so an uncalibrated MMM
  overcredits search. Its window ends the week before the geo test starts.
* **geo test** on the MMM's **search** channel: in ``n_treated`` of ``n_geos`` regions search spend
  was raised by ``spend_increase`` (e.g. +100%) for ``test_weeks`` weeks. Each region's sales
  respond as a scaled copy of the national curve (region g with share w_g of national sales has
  response ``w_g * B * logistic(lam * x_g / (w_g * S))``, and spend is proportional to sales), so
  the treated regions gain ``f * (r(x * (1 + rho)) - r(x))`` per week, where ``f`` is their share
  of sales, ``x`` the national weekly search spend and ``r`` the MMM's true response curve. The
  national-equivalent lift test is therefore ``(x, rho * x, r(x (1 + rho)) - r(x))``: what
  ``add_lift_test_measurements`` needs, and exactly on the MMM's true curve.
* **CLV-weighted budget** (``channels``): the shared vocabulary maps media channels to acquisition
  channels and gives the cost per new customer; the true long-run optimal weekly plan uses the
  MMM's true curves and Stage 1's true CLV (excluding the first purchase, which the MMM counts).
* **retention offer** (``email_experiment``): randomized e-mail offer with true CATE, margin 30%,
  offer cost chosen so that targeting is worth several times mailing everyone; a ``split`` column
  fixes the held-out half used for scoring.

The truth separates ``scenario`` (what the notebook tells pairs: dates, spend, budget, bounds,
costs) from ``answers`` (hidden until the final cell), each answer with the checkpoint tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mktstats.synth._common import SynthResult, make_rng, sig
from mktstats.synth.btyd import RETAILER_CHANNELS, retailer
from mktstats.synth.channels import (
    COST_PER_NEW_CUSTOMER,
    GROSS_MARGIN,
    MEDIA_TO_ACQUISITION,
    WEEKS_PER_MONTH,
    long_run_allocation,
    media_clv,
    new_customer_value_monthly,
)
from mktstats.synth.email import email_experiment
from mktstats.synth.geo import _ar1, campaign_economics
from mktstats.synth.mmm import logistic_saturation, mmm

CAPSTONE_RETAILER = {
    "r": 0.9, "alpha": 5.5, "s": 0.6, "beta": 12.0,
    "gamma_purchase": {"search": 0.0, "social": -0.35, "referral": 0.25},
    "gamma_dropout": {"search": 0.0, "social": 0.5, "referral": -0.7},
    "p": 6.0, "q": 4.0, "v": 16.0,
}
CAPSTONE_ROAS = {"tv": 1.8, "search": 2.6, "social": 1.2, "display": 0.8}
CAPSTONE_MMM = {"confounded": True, "lift_test_plan": ()}


def _geo_panel_additive(seed, *, dates, n_geos, n_regions, n_treated, test_weeks, base_median,
                        base_log_sd, growth_per_week, slope_sd, national_sd, regional_sd,
                        noise_sd, national_lift_per_week):
    """Geo panel like ``geo_panel`` (same counterfactual model) with an additive effect: each
    treated geo gains its share of pre-period sales times ``national_lift_per_week`` in every test
    week. Returns the panel, per-geo shares, treated geos and the realized increments."""
    rng = make_rng(seed)
    n_weeks = len(dates)
    t = np.arange(n_weeks)
    geos = np.array([f"G{g + 1:02d}" for g in range(n_geos)])
    region_of = rng.permutation(np.arange(n_geos) % n_regions)
    regions = np.array([f"R{r + 1}" for r in range(n_regions)])
    base = base_median * np.exp(rng.normal(0, base_log_sd, n_geos))
    amp = rng.uniform(0.7, 1.3, n_regions)
    slope = rng.normal(0, slope_sd, n_geos)
    ph = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    season = 0.08 * np.sin(ph) + 0.05 * np.cos(ph)
    national = _ar1(rng, n_weeks, 0.6, national_sd)
    regional = _ar1(rng, n_weeks, 0.5, regional_sd, size=n_regions)
    eps = rng.normal(0, noise_sd, size=(n_geos, n_weeks))
    treated_idx = np.sort(rng.choice(n_geos, size=n_treated, replace=False))
    log_y0 = (np.log(base)[:, None] + amp[region_of][:, None] * season[None, :]
              + growth_per_week * t[None, :] + slope[:, None] * t[None, :]
              + national[None, :] + regional[region_of] + eps)
    y0 = np.round(np.exp(log_y0), 0)
    post = (t >= n_weeks - test_weeks).astype(np.int64)
    share = y0[:, post == 0].sum(axis=1) / y0[:, post == 0].sum()  # observed pre-period shares
    treated = np.zeros(n_geos, dtype=np.int64)
    treated[treated_idx] = 1
    inc_geo_week = np.round(share * national_lift_per_week, 0) * treated  # dollars per week
    y = y0 + inc_geo_week[:, None] * post[None, :]
    panel = pd.DataFrame({
        "date": np.tile(dates, n_geos),
        "geo": np.repeat(geos, n_weeks),
        "region": np.repeat(regions[region_of], n_weeks),
        "sales": y.ravel().astype(np.int64),
        "treated": np.repeat(treated, n_weeks),
        "post": np.tile(post, n_geos),
    })
    return {"panel": panel, "geos": geos, "share": share, "treated_idx": treated_idx,
            "inc_geo_week": inc_geo_week, "y0": y0, "post": post}


def capstone(
    seed: int = 2032,
    *,
    n_customers: int = 4000,
    n_geos: int = 40,
    n_treated: int = 8,
    geo_weeks: int = 104,
    test_weeks: int = 10,
    test_end: str = "2025-12-22",
    spend_increase: float = 1.0,
    mmm_weeks: int = 156,
    margin: float = GROSS_MARGIN,
    clv_months: int = 36,
    monthly_rate: float = 0.01,
    email_n: int = 20_000,
    offer_cost: float = 0.75,
    geo_noise_sd: float = 0.02,
    geo_regional_sd: float = 0.01,
    geo_national_sd: float = 0.015,
    mmm_options: dict | None = None,
) -> SynthResult:
    """The capstone scenario (module docstring).

    Tables
    ------
    transactions, customers : as ``retailer`` (``n_customers`` customers).
    weekly : as ``mmm`` (date_week, tv, search, social, display, price_index, holiday, t, y),
        ``mmm_weeks`` weeks ending the week before the geo test.
    panel : as ``geo_panel`` (date, geo, region, sales, treated, post), ``geo_weeks`` weeks ending
        ``test_end``; the last ``test_weeks`` weeks are the search test.
    experiment : as ``email_experiment`` plus ``split`` ("train" / "test", half each, random).
    """
    test_end_ts = pd.Timestamp(test_end)
    geo_dates = pd.date_range(end=test_end_ts, periods=geo_weeks, freq="7D")
    test_start_ts = geo_dates[geo_weeks - test_weeks]
    mmm_end = test_start_ts - pd.Timedelta(weeks=1)
    mmm_start = mmm_end - pd.Timedelta(weeks=mmm_weeks - 1)

    # Stage 1: customers -------------------------------------------------------------------
    ret = retailer(seed, n_customers=n_customers, start=str(geo_dates[0].date()),
                   **CAPSTONE_RETAILER)
    rt = ret.truth
    mean_spend = rt["population_mean_spend"]
    value = {}
    for c in RETAILER_CHANNELS:
        a_c = rt["pnbd"]["alpha"] * np.exp(-rt["covariates"]["gamma_purchase"][c])
        b_c = rt["pnbd"]["beta"] * np.exp(-rt["covariates"]["gamma_dropout"][c])
        v = new_customer_value_monthly(rt["pnbd"]["r"], a_c, rt["pnbd"]["s"], b_c, mean_spend,
                                       clv_months, monthly_rate)
        value[c] = {
            "expected_repeat_purchases": sig(v["expected_repeat_purchases"]),
            "revenue_including_first_purchase": sig(v["value_including_first_purchase"]),
            "revenue_excluding_first_purchase": sig(v["value_excluding_first_purchase"]),
            "margin_including_first_purchase": sig(margin * v["value_including_first_purchase"]),
            "margin_excluding_first_purchase": sig(margin * v["value_excluding_first_purchase"]),
        }
    ranking = sorted(value, key=lambda c: -value[c]["margin_including_first_purchase"])

    # Stage 3 data: the national MMM -------------------------------------------------------
    mm = mmm(seed + 1, n_weeks=mmm_weeks, start=str(mmm_start.date()), roas=CAPSTONE_ROAS,
             **{**CAPSTONE_MMM, **(mmm_options or {})})
    mt = mm.truth
    sch = mt["channels"]["search"]
    big_b, lam, scale = (sch["saturation_beta_sales_units"], sch["saturation_lam"],
                         sch["channel_scale"])
    x_base = float(np.round(mm.weekly["search"].iloc[-8:].mean(), -2))
    dx = spend_increase * x_base
    lift_nat = float(big_b * (logistic_saturation((x_base + dx) / scale, lam)
                              - logistic_saturation(x_base / scale, lam)))

    # Stage 2: the geo test ----------------------------------------------------------------
    y_level = float(mm.weekly["y"].iloc[-52:].mean())
    base_median = float(np.round(y_level / (n_geos * np.exp(0.6**2 / 2)), -2))
    geo = _geo_panel_additive(
        seed + 2, dates=geo_dates, n_geos=n_geos, n_regions=4, n_treated=n_treated,
        test_weeks=test_weeks, base_median=base_median, base_log_sd=0.6, growth_per_week=0.001,
        slope_sd=0.0002, national_sd=geo_national_sd, regional_sd=geo_regional_sd,
        noise_sd=geo_noise_sd, national_lift_per_week=lift_nat)
    tidx = geo["treated_idx"]
    f = float(geo["share"][tidx].sum())
    incremental = float(geo["inc_geo_week"].sum() * test_weeks)
    cost = float(np.round(f * dx * test_weeks, -2))
    cf_window = float(geo["y0"][tidx][:, geo["post"] == 1].sum())
    camp = campaign_economics(incremental, cost, margin)
    lift_row = {"channel": "search", "x": x_base, "delta_x": sig(dx),
                "delta_y": sig(incremental / (f * test_weeks)),
                "delta_y_curve": sig(lift_nat), "date": str(test_end_ts.date())}

    # Stage 4: CLV-weighted allocation -----------------------------------------------------
    clv_acq = {c: value[c]["margin_excluding_first_purchase"] for c in RETAILER_CHANNELS}
    clv_media = media_clv(clv_acq)
    alloc = long_run_allocation(mt, clv_media, margin=margin)

    # Stage 5: retention offer -------------------------------------------------------------
    em = email_experiment(seed + 3, n=email_n, margin=margin, offer_cost=offer_cost)
    exp = em.experiment.copy()
    split = make_rng(seed + 4).permutation(np.r_[np.zeros(email_n // 2, dtype=int),
                                                 np.ones(email_n - email_n // 2, dtype=int)])
    exp["split"] = np.where(split == 1, "test", "train")
    gain = margin * exp["true_cate"].to_numpy() - offer_cost
    test = exp["split"].to_numpy() == "test"

    def policy_values(mask):
        g = gain[mask]
        return {"treat_none": 0.0, "treat_all": sig(g.mean()),
                "oracle": sig(np.maximum(g, 0).mean()), "share_should_treat": sig((g > 0).mean()),
                "n": int(mask.sum())}

    # truth ---------------------------------------------------------------------------------
    scenario = {
        "story": (
            "One retailer. Its customers arrive through search, social or referral (acquisition "
            "channels); it spends on tv, search, social and display (media channels). Last "
            f"autumn it doubled search spend in {n_treated} of its {n_geos} regions for "
            f"{test_weeks} weeks. Next quarter's weekly media budget must be split, and a "
            "retention e-mail offer can be sent to some customers."
        ),
        "margin": margin,
        "customer_value": {"months": clv_months, "monthly_discount_rate": monthly_rate,
                           "weeks_per_month": sig(WEEKS_PER_MONTH),
                           "convention": "acquisition purchase (month 0, undiscounted) plus "
                                         "monthly repeat purchases discounted by "
                                         "(1 + monthly_discount_rate)**k, k = 1..months"},
        "geo_test": {
            "channel": "search", "treated_geos": geo["geos"][tidx].tolist(),
            "test_start": str(test_start_ts.date()), "test_end": str(test_end_ts.date()),
            "test_weeks": test_weeks, "spend_increase": spend_increase,
            "base_weekly_spend_national": x_base,
            "extra_spend_total": cost,
            "treated_share_of_sales_rule": "treated geos' share of all geos' sales in the weeks "
                                           "before test_start (computable from the panel)",
            "national_equivalent_rule": (
                "lift-test row for the national MMM: x = base_weekly_spend_national, delta_x = "
                "spend_increase * x = extra_spend_total / (treated_share * test_weeks), delta_y = "
                "incremental sales in the treated geos / (treated_share * test_weeks), sigma = "
                "its standard error on the same scale"
            ),
        },
        "mmm": {"window_start": mt["start_date"], "window_end": mt["end_date"],
                "channels": list(mt["channels"])},
        "budget": {"weekly_budget": mt["true_optimal_allocation"]["weekly_budget"],
                   "budget_rule": mt["true_optimal_allocation"]["budget_rule"],
                   "bounds": mt["true_optimal_allocation"]["bounds"],
                   "bounds_rule": mt["true_optimal_allocation"]["bounds_rule"],
                   "current_allocation": mt["true_optimal_allocation"]["current_allocation"],
                   "planning_weeks": 13},
        "channel_link": {"media_to_acquisition": MEDIA_TO_ACQUISITION,
                         "cost_per_new_customer": COST_PER_NEW_CUSTOMER,
                         "new_customers_per_dollar": {m: sig(1 / c) for m, c in
                                                      COST_PER_NEW_CUSTOMER.items()}},
        "retention_offer": {"offer_cost": offer_cost, "margin": margin,
                            "rule": "send when predicted CATE (spend) * margin > offer_cost",
                            "held_out": "rows with split == 'test'"},
    }
    answers = {
        "stage1_customer_value": {
            "by_acquisition_channel": value,
            "ranking_by_margin_value": ranking,
            "top_channel": ranking[0],
            "checkpoint": {"compare": "margin_including_first_purchase", "rel": 0.20,
                           "also": "the top channel is the true top channel"},
        },
        "stage2_geo_test": {
            "incremental_sales": sig(incremental),
            "incremental_sales_per_week": sig(incremental / test_weeks),
            "lift_pct": sig(100 * incremental / cf_window),
            "counterfactual_sales_treated_window": sig(cf_window),
            "treated_share_of_sales": sig(f),
            "campaign": camp,
            "national_equivalent_lift_test": lift_row,
            "lift_test_note": (
                "delta_y is the realized increment (rounded per geo-week) scaled to the nation; "
                "delta_y_curve is the MMM's true curve r(x + delta_x) - r(x); they agree to "
                "rounding."
            ),
            "checkpoint": {"incremental_sales": {"rel": 0.15}, "placebo_p_value_max": 0.10},
        },
        "stage3_mmm": {
            "true_roas": {c: v["roas"] for c, v in mt["channels"].items()},
            "confounded_channel": "search",
            "naive_ols_search_roas": mt["confounding"]["roas"]["search"]["naive_ols"],
            "checkpoint": {"k_of_K_inside": [3, 4], "interval": "94% HDI"},
        },
        "stage4_allocation": {
            **alloc,
            "clv_margin_by_acquisition_channel": {c: sig(v) for c, v in clv_acq.items()},
            "clv_margin_by_media_channel": {m: sig(v) for m, v in clv_media.items()},
            "short_run_optimum_details": "capstone.mmm.true_optimal_allocation",
            "checkpoint": {
                "budget_and_bounds": "sum equals weekly_budget (1e-6 relative), bounds hold",
                "value": {"rel": 0.03, "of": "weekly_value_optimal",
                          "note": "see value_sensitivity: even the current plan is within "
                                  "a few percent, so also compare spend by channel"},
                "spend_by_channel": {"abs_share_of_budget": 0.05},
            },
        },
        "stage5_targeting": {
            "all_rows": policy_values(np.ones(len(exp), dtype=bool)),
            "test_split": policy_values(test),
            "definition": "mean over rows of policy * (margin * true_cate - offer_cost), "
                          "relative to sending nobody the offer; oracle sends exactly the rows "
                          "with margin * true_cate > offer_cost",
            "checkpoint": "on the test split, the learned rule's true value (from true_cate) is "
                          "positive and at least treat_all",
        },
    }
    truth = {
        "description": "Module 12 scenario: one retailer, hidden answers per stage "
                       "(mktstats.synth.capstone).",
        "seed": seed,
        "seeds": {"retailer": seed, "mmm": seed + 1, "geo_panel": seed + 2,
                  "email_experiment": seed + 3, "email_split": seed + 4},
        "scenario": scenario,
        "answers": answers,
        "retailer": rt,
        "mmm": mt,
        "geo_panel": {
            "n_geos": n_geos, "n_weeks": geo_weeks, "base_median": base_median,
            "noise": {"geo_noise_sd": geo_noise_sd, "regional_sd": geo_regional_sd,
                      "national_sd": geo_national_sd},
            "effect": "additive: treated geo g gains round(share_g * delta_y_curve) dollars in "
                      "every test week (share_g: its share of pre-period sales)",
            "weekly_increment_by_geo": {g: float(v) for g, v in
                                        zip(geo["geos"][tidx], geo["inc_geo_week"][tidx],
                                            strict=True)},
        },
        "email_experiment": em.truth,
    }
    data = {"transactions": ret.transactions, "customers": ret.customers, "weekly": mm.weekly,
            "panel": geo["panel"], "experiment": exp}
    return SynthResult("capstone", data, truth)
