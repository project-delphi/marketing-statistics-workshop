"""Weekly geo panel with a campaign of known multiplicative lift in a few treated geos.

log counterfactual sales of geo g in week t:

    log y0[g,t] = log base[g] + amp[region] * season[t] + growth * t + slope[g] * t
                  + national[t] + regional[region, t] + eps[g,t]

with AR(1) national and regional shocks (correlated across geos) and i.i.d. noise. In the test
window the treated geos sell ``y = y0 * (1 + lift_pct / 100)``. The true incremental sales are
``sum(y0 * lift_pct / 100)`` over treated geos and test weeks, computed on the unrounded
counterfactual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mktstats.synth._common import SynthResult, make_rng, sig


def _ar1(rng, n, phi, sd, size=None):
    shape = (n,) if size is None else (size, n)
    e = rng.normal(0, sd, size=shape)
    out = np.empty(shape)
    prev = np.zeros(shape[:-1]) if size is not None else 0.0
    for t in range(n):
        prev = phi * prev + e[..., t]
        out[..., t] = prev
    return out


def geo_panel(
    seed: int = 2029,
    *,
    n_geos: int = 40,
    n_weeks: int = 104,
    start: str = "2024-01-01",
    n_regions: int = 4,
    n_treated: int = 8,
    test_weeks: int = 10,
    lift_pct: float = 5.0,
    base_median: float = 20_000.0,
    base_log_sd: float = 0.6,
    growth_per_week: float = 0.001,
    slope_sd: float = 0.0002,
    national_sd: float = 0.02,
    regional_sd: float = 0.015,
    noise_sd: float = 0.04,
    campaign_spend_rate: float = 0.0125,
    margin: float = 0.30,
) -> SynthResult:
    """Sales for ``n_geos`` geos over ``n_weeks`` weeks; campaign in the last ``test_weeks`` weeks.

    Tables
    ------
    panel : date, geo, region, sales, treated (1 for the treated geos in every week),
        post (1 in the test window for every geo).

    The campaign's cost (truth ``campaign``) does not change the panel: each treated geo received
    ``campaign_spend_rate`` times its mean weekly pre-period sales (observed, rounded to $10) in
    every test week. With the gross ``margin`` this gives the incremental ROAS, the incremental
    margin and the net return of the campaign.
    """
    if not 0 < n_treated < n_geos:
        raise ValueError("need 0 < n_treated < n_geos")
    if not 0 < test_weeks < n_weeks:
        raise ValueError("need 0 < test_weeks < n_weeks")
    rng = make_rng(seed)
    dates = pd.date_range(start, periods=n_weeks, freq="7D")
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
    y0 = np.exp(log_y0)
    treated = np.zeros(n_geos, dtype=np.int64)
    treated[treated_idx] = 1
    post = (t >= n_weeks - test_weeks).astype(np.int64)
    effect = 1.0 + (lift_pct / 100.0) * treated[:, None] * post[None, :]
    y = np.round(y0 * effect, 0)
    incremental = float((y0 * (effect - 1.0)).sum())

    panel = pd.DataFrame(
        {
            "date": np.tile(dates, n_geos),
            "geo": np.repeat(geos, n_weeks),
            "region": np.repeat(regions[region_of], n_weeks),
            "sales": y.ravel().astype(np.int64),
            "treated": np.repeat(treated, n_weeks),
            "post": np.tile(post, n_geos),
        }
    )
    test_start = dates[n_weeks - test_weeks]
    cf_window = float(y0[treated_idx][:, post == 1].sum())
    truth = {
        "description": (
            "Geo panel with shared seasonality, national and regional AR(1) shocks; campaign "
            "multiplies counterfactual sales of treated geos by (1 + lift_pct/100) in the test "
            "window."
        ),
        "seed": seed,
        "n_geos": n_geos,
        "n_weeks": n_weeks,
        "lift_pct": sig(lift_pct),
        "log_lift": sig(np.log1p(lift_pct / 100.0)),
        "incremental_sales": sig(incremental),
        "counterfactual_sales_treated_window": sig(cf_window),
        "treated_geos": geos[treated_idx].tolist(),
        "test_start": str(test_start.date()),
        "test_end": str(dates[-1].date()),
        "test_weeks": test_weeks,
        "pre_weeks": n_weeks - test_weeks,
        "incremental_definition": (
            "sum over treated geos and test weeks of y0 * lift_pct / 100, where y0 is the "
            "unrounded counterfactual (no-campaign) sales of this realization."
        ),
    }
    pre_mean = y[treated_idx][:, post == 0].mean(axis=1)  # observed pre-period weekly sales
    weekly_spend = np.round(campaign_spend_rate * pre_mean, -1)
    cost = float(weekly_spend.sum() * test_weeks)
    truth["campaign"] = campaign_economics(incremental, cost, margin)
    truth["campaign"].update({
        "cost_rule": (
            f"each treated geo spent {campaign_spend_rate:.2%} of its mean weekly pre-period "
            "sales (observed sales, rounded to $10) in every test week"
        ),
        "weekly_spend_by_geo": {g: float(s) for g, s in zip(geos[treated_idx], weekly_spend,
                                                            strict=True)},
        "spend_rate_of_pre_period_sales": campaign_spend_rate,
    })
    return SynthResult("geo_panel", {"panel": panel}, truth)


def campaign_economics(incremental_sales: float, cost: float, margin: float) -> dict:
    """Return on a campaign from its true incremental sales, its cost and the gross margin."""
    inc_margin = margin * incremental_sales
    return {
        "cost": sig(cost),
        "margin": margin,
        "incremental_sales": sig(incremental_sales),
        "incremental_roas": sig(incremental_sales / cost),
        "break_even_roas": sig(1 / margin),
        "incremental_margin": sig(inc_margin),
        "net_return": sig(inc_margin - cost),
        "return_on_spend": sig((inc_margin - cost) / cost),
        "definition": (
            "incremental_roas = true incremental sales / cost; incremental_margin = margin * "
            "incremental sales; net_return = incremental_margin - cost; return_on_spend = "
            "net_return / cost. The campaign pays back when incremental_roas > break_even_roas "
            "= 1 / margin."
        ),
    }
