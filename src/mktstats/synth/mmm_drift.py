"""Weekly marketing-mix data in which one channel wears out (Module 17).

Not exported from ``mktstats.synth``: the lab reads the committed files
(``mmm_drift_weekly.csv``, ``mmm_drift_latent.csv``, ``mmm_drift_lift_tests.csv``,
``truth_mmm_drift.json``), written by ``scripts/make_synthetic.py`` (``EXTENSIONS["mmm_drift"]``).

The functional form is Module 8's (:mod:`mktstats.synth.mmm`, pymc-marketing 1.2.0's ``MMM`` with
``GeometricAdstock(l_max=8)`` and ``LogisticSaturation()`` and default max scaling) with one
change: an **effectiveness multiplier** ``m_c(t) > 0`` on the week the effect lands,

    contribution_c(t) = B_c * m_c(t) * logistic(lam_c * adstock(x_c)(t) / S_c),

as pymc-marketing's ``time_varying_media`` applies its ``media_temporal_latent_multiplier``
(``mmm/mmm.py``: ``baseline_channel_contribution * media_latent_process``). Only social fades:
``m_social(t) = 1 - (1 - end) / (1 + exp(-(t - mid) / width))`` for ``t = 0 .. n_weeks - 1``
(defaults end 0.35, mid 104, width 20: about 1.0 in the first year, 0.40 in the last week); the
other channels have ``m_c(t) = 1``. ``B_c`` is solved so that the channel's ROAS over all weeks,
multiplier included, equals the requested ``roas``. pymc-marketing normalizes its multiplier to
mean 1 over the fitted weeks, so the comparable truth is the **relative path**
``m_c(t) / mean_t m_c(t)`` (``multiplier_rel_<c>`` in the latent file).

ROAS over a window W (a set of weeks) is the window definition of the brief: the channel's
noiseless contribution in those weeks over its spend in those weeks (carry-in from earlier spend
counts, carry-out after the window does not). ``roas_last_quarter_counterfactual`` is
pymc-marketing's definition instead (``mmm/incrementality.py``): spend set to zero in the window,
carry-in kept, the difference in contribution summed from the window's first week to the end of
the data (the window is clamped to the fitted dates), divided by the window's spend.

Random draws, all from one stream ``make_rng(seed)``, in this order: tv flights (start offset,
then alternating on/off lengths), tv levels (n), search noise (n), social noise (n), social push
jitter (one per push), display noise (n), price AR(1) shocks (n), promotion weeks (10),
promotion depths (10), sales noise (n), lift-test noise (one per test). Changing a default that
adds or removes draws (``n_weeks``, ``push_every``) changes every later draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mktstats.synth._common import SynthResult, make_rng, sig
from mktstats.synth.mmm import (
    _thanksgiving,
    _tv_flighting,
    geometric_adstock,
    logistic_saturation,
)

CHANNELS = ("tv", "search", "social", "display")
ADSTOCK_ALPHA = {"tv": 0.6, "search": 0.2, "social": 0.4, "display": 0.3}  # Module 8's
SATURATION_LAM = {"tv": 1.5, "search": 2.5, "social": 2.0, "display": 3.0}  # Module 8's
ROAS = {"tv": 1.6, "search": 3.0, "social": 4.0, "display": 0.6}
FATIGUE = {"channel": "social", "end": 0.35, "mid": 104.0, "width": 20.0}
GROSS_MARGIN = 0.30  # Modules 8 and 9
LIFT_TEST = {"channel": "social", "end_week": 150, "weeks": 8, "change": 0.3, "rel_sigma": 0.08,
             "base_weeks": 26}
FIRST_YEAR_WEEKS = 52
LAST_QUARTER_WEEKS = 13


def fatigue_multiplier(t, end: float = 0.35, mid: float = 104.0, width: float = 20.0):
    """``1 - (1 - end) / (1 + exp(-(t - mid) / width))``: 1 early, ``end`` late."""
    t = np.asarray(t, dtype=float)
    return 1.0 - (1.0 - end) / (1.0 + np.exp(-(t - mid) / width))


def social_push_starts(rng: np.random.Generator, n: int, every: int = 13, first: int = 4,
                       jitter: int = 2) -> np.ndarray:
    """Start week (0-based) of each four-week social push: ``first, first + every, ...`` while
    the push fits before week ``n - 4``, each moved by a jitter drawn from ``-jitter..jitter``."""
    base = np.arange(first, n - 4, every)
    return base + rng.integers(-jitter, jitter + 1, size=len(base))


def _window_index(dates: pd.DatetimeIndex, start, end) -> np.ndarray:
    return np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def mmm_drift(
    seed: int = 2037,
    *,
    n_weeks: int = 156,
    start: str = "2023-01-02",
    l_max: int = 8,
    roas: dict[str, float] | None = None,
    fatigue_end: float = FATIGUE["end"],
    fatigue_mid: float = FATIGUE["mid"],
    fatigue_width: float = FATIGUE["width"],
    intercept: float = 250_000.0,
    trend_per_week: float = 500.0,
    fourier: tuple[float, float, float, float] = (15_000.0, 25_000.0, 5_000.0, -8_000.0),
    price_coef: float = -400_000.0,
    holiday_effect: float = 40_000.0,
    noise_sd: float = 8_000.0,
    tv_level: float = 60_000.0,
    search_level: float = 20_000.0,
    search_sd: float = 0.12,
    social_level: float = 10_000.0,
    social_sd: float = 0.3,
    push_every: int = 13,
    push_mult: float = 2.5,
    display_level: float = 6_000.0,
    display_sd: float = 0.3,
    gross_margin: float = GROSS_MARGIN,
) -> SynthResult:
    """Weekly sales and spend for tv, search, social and display, with social wearing out.

    Tables
    ------
    weekly : date_week, tv, search, social, display (spend, dollars), price_index, holiday (0/1),
        t (0..n_weeks-1), y (sales, dollars). The columns of Module 8's ``mmm_weekly.csv``.
    latent : date_week, baseline (everything but media, noiseless), baseline_trend (intercept +
        trend), seasonality (the Fourier part), and per channel ``multiplier_<c>`` (m_c(t)),
        ``multiplier_rel_<c>`` (m_c(t) / its mean over the weeks) and ``contribution_<c>``
        (noiseless, dollars). Truth only; never a model input.
    lift_tests : one randomized social test (+30% weekly spend for 8 weeks ending in week 151),
        Module 9's columns: channel, x, delta_x, delta_y, sigma, date (last week of the test),
        test_start, test_weeks, true_delta_y.

    Spend: tv in flights (on 4-8 weeks, off 3-8, ``tv_level`` x Uniform(0.7, 1.3) when on);
    search follows demand, ``search_level * d_t * LogNormal(0, search_sd)`` with
    ``d_t = (intercept + trend_t + seasonality_t) / intercept``; social
    ``social_level * LogNormal(0, social_sd)`` times ``push_mult`` in a four-week push starting
    every ``push_every`` weeks (weeks 5, 18, 31, ... counting from 1, each moved by -2..+2 weeks);
    display ``display_level * LogNormal(0, display_sd)``. All rounded to dollars.
    """
    roas_c = dict(ROAS if roas is None else roas)
    rng = make_rng(seed)
    n = n_weeks
    dates = pd.date_range(start, periods=n, freq="7D")
    t = np.arange(n)
    ph = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    season = (fourier[0] * np.sin(ph) + fourier[1] * np.cos(ph)
              + fourier[2] * np.sin(2 * ph) + fourier[3] * np.cos(2 * ph))
    trend = trend_per_week * t
    demand_index = (intercept + trend + season) / intercept

    # spend (draw order: see the module docstring)
    tv_on = _tv_flighting(rng, n)
    tv = np.where(tv_on, tv_level * rng.uniform(0.7, 1.3, n), 0.0)
    search = search_level * demand_index * rng.lognormal(0, search_sd, n)
    social = social_level * rng.lognormal(0, social_sd, n)
    pushes = social_push_starts(rng, n, push_every)
    for b0 in pushes:
        social[b0:b0 + 4] *= push_mult
    display = display_level * rng.lognormal(0, display_sd, n)
    spend = {c: np.round(v, 0) for c, v in
             {"tv": tv, "search": search, "social": social, "display": display}.items()}

    # controls (Module 8's rules)
    price = np.empty(n)
    e = 0.0
    for i in range(n):
        e = 0.7 * e + rng.normal(0, 0.012)
        price[i] = 1.0 + e
    promo = rng.choice(n, size=10, replace=False)
    price[promo] -= rng.uniform(0.04, 0.10, size=10)
    price = np.round(price, 4)
    holiday = np.zeros(n, dtype=np.int64)
    for year in range(dates[0].year, dates[-1].year + 1):
        black_friday = _thanksgiving(year) + pd.Timedelta(days=1)
        for day in [black_friday, pd.Timestamp(year=year, month=12, day=18),
                    pd.Timestamp(year=year, month=12, day=24)]:
            hit = (dates <= day) & (day < dates + pd.Timedelta(days=7))
            holiday[hit] = 1

    # media
    mult = {c: np.ones(n) for c in CHANNELS}
    mult[FATIGUE["channel"]] = fatigue_multiplier(t, fatigue_end, fatigue_mid, fatigue_width)
    scale = {c: float(spend[c].max()) for c in CHANNELS}
    sat = {c: logistic_saturation(geometric_adstock(spend[c], ADSTOCK_ALPHA[c], l_max) / scale[c],
                                  SATURATION_LAM[c]) for c in CHANNELS}
    big_b = {c: roas_c[c] * spend[c].sum() / (mult[c] * sat[c]).sum() for c in CHANNELS}
    contrib_exact = {c: big_b[c] * mult[c] * sat[c] for c in CHANNELS}
    media = sum(contrib_exact.values())

    baseline = (intercept + trend + season + price_coef * (price - 1.0)
                + holiday_effect * holiday)
    noise = rng.normal(0, noise_sd, n)
    y = np.round(baseline + media + noise, 0)

    weekly = pd.DataFrame({"date_week": dates})
    for c in CHANNELS:
        weekly[c] = spend[c].astype(np.int64)
    weekly["price_index"] = price
    weekly["holiday"] = holiday
    weekly["t"] = t
    weekly["y"] = y.astype(np.int64)

    # latent truth, rounded as written to the CSV; every ROAS in the truth is computed from these
    # rounded values, so the lab's window ROAS on the latent file reproduces it to rounding error
    latent = pd.DataFrame({"date_week": dates, "baseline": np.round(baseline, 2),
                           "baseline_trend": np.round(intercept + trend, 2),
                           "seasonality": np.round(season, 2)})
    contrib = {}
    for c in CHANNELS:
        latent[f"multiplier_{c}"] = np.round(mult[c], 8)
        latent[f"multiplier_rel_{c}"] = np.round(mult[c] / mult[c].mean(), 8)
        contrib[c] = np.round(contrib_exact[c], 2)
        latent[f"contribution_{c}"] = contrib[c]

    # windows
    windows = {
        "all": (dates[0], dates[-1]),
        "first_year": (dates[0], dates[FIRST_YEAR_WEEKS - 1]),
        "last_quarter": (dates[n - LAST_QUARTER_WEEKS], dates[-1]),
    }
    idx = {k: _window_index(dates, *w) for k, w in windows.items()}

    def window_roas(c, k):
        return contrib[c][idx[k]].sum() / spend[c][idx[k]].sum()

    def counterfactual_roas(c, k):
        """pymc-marketing's definition: zero the window's spend, keep carry-in, sum the change
        in contribution from the window's first week to the end of the data."""
        w = idx[k]
        x_cf = spend[c].copy()
        x_cf[w] = 0.0
        sat_cf = logistic_saturation(
            geometric_adstock(x_cf, ADSTOCK_ALPHA[c], l_max) / scale[c], SATURATION_LAM[c])
        diff = big_b[c] * mult[c] * (sat[c] - sat_cf)
        return diff[w[0]:].sum() / spend[c][w].sum()

    def drift_ratio(m):
        return m[idx["last_quarter"]].mean() / m[idx["first_year"]].mean()

    # one dated lift test (Module 9's form): a randomized geo test scaled to the national level;
    # the national spend series is unchanged
    lt = LIFT_TEST
    c_lt = lt["channel"]
    end_w = lt["end_week"]
    start_w = end_w - lt["weeks"] + 1
    before = spend[c_lt][start_w - lt["base_weeks"]:start_w]
    x0 = float(np.round(before.mean(), -2))
    dx = float(np.round(lt["change"] * x0, -2))
    m_bar = float(mult[c_lt][start_w:end_w + 1].mean())
    true_dy = big_b[c_lt] * m_bar * (logistic_saturation((x0 + dx) / scale[c_lt],
                                                         SATURATION_LAM[c_lt])
                                     - logistic_saturation(x0 / scale[c_lt],
                                                           SATURATION_LAM[c_lt]))
    sigma = lt["rel_sigma"] * abs(true_dy)
    dy = true_dy + rng.normal(0, sigma)
    lift_row = {"channel": c_lt, "x": x0, "delta_x": dx, "delta_y": float(np.round(dy, 0)),
                "sigma": float(np.round(sigma, 0)), "date": dates[end_w],
                "test_start": dates[start_w], "test_weeks": lt["weeks"],
                "true_delta_y": float(np.round(true_dy, 0))}
    lift_tests = pd.DataFrame([lift_row])

    break_even = 1.0 / gross_margin
    channel_truth = {}
    for c in CHANNELS:
        channel_truth[c] = {
            "adstock_alpha": sig(ADSTOCK_ALPHA[c]),
            "saturation_lam": sig(SATURATION_LAM[c]),
            "saturation_beta_sales_units": sig(big_b[c]),
            "channel_scale": sig(scale[c]),
            "roas": sig(window_roas(c, "all"), 12),
            "roas_first_year": sig(window_roas(c, "first_year"), 12),
            "roas_last_quarter": sig(window_roas(c, "last_quarter"), 12),
            "roas_last_quarter_counterfactual": sig(counterfactual_roas(c, "last_quarter")),
            "roas_target": sig(roas_c[c]),
            "spend": {k: sig(spend[c][idx[k]].sum(), 12) for k in idx},
            "contribution": {k: sig(contrib[c][idx[k]].sum(), 12) for k in idx},
            "multiplier_mean": sig(mult[c].mean()),
            "drift_ratio": sig(drift_ratio(mult[c])),
        }
    soc = channel_truth[FATIGUE["channel"]]
    truth = {
        "description": (
            "Weekly sales = intercept + trend + yearly seasonality + price and holiday effects + "
            "sum of channel contributions + Normal noise; channel contribution = B_c * m_c(t) * "
            "logistic(lam_c * normalized_geometric_adstock(spend_c) / max(spend_c)). Only social "
            "wears out: m_social(t) falls smoothly from about 1 to about 0.4 (logistic in t); the "
            "other channels have m_c(t) = 1."
        ),
        "seed": seed,
        "n_weeks": n,
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "l_max": l_max,
        "adstock": "GeometricAdstock(l_max=8, normalize=True), zero spend before the first week",
        "saturation": "LogisticSaturation: beta * (1 - exp(-lam x)) / (1 + exp(-lam x))",
        "units": (
            "adstock_alpha and saturation_lam are scale-free (lam applies to adstocked spend / "
            "channel_scale, the channel's maximum weekly spend); saturation_beta_sales_units is "
            "B_c in dollars per week at m_c(t) = 1; spend, contribution and sales are dollars."
        ),
        "roas_definition": (
            "Window definition: ROAS_c(W) = sum over the weeks in W of the channel's noiseless "
            "contribution (multiplier included) / its spend in those weeks; carry-in from earlier "
            "spend counts, carry-out after the window does not. Computed from the contributions "
            "as written to mmm_drift_latent.csv (rounded to cents). "
            "roas_last_quarter_counterfactual is pymc-marketing's contribution_over_spend "
            "definition: spend zeroed in the window, carry-in kept, the change in contribution "
            "summed from the window's first week to the end of the data, over the window's spend."
        ),
        "windows": {k: {"start": str(a.date()), "end": str(b.date()), "weeks": int(len(idx[k]))}
                    for k, (a, b) in windows.items()},
        "channels": channel_truth,
        "multiplier": {
            "acts_on": "the week the effect lands (after adstock and saturation), as "
                       "pymc-marketing's time_varying_media multiplier",
            "relative_path": "multiplier_rel_<c> = m_c(t) / mean over all weeks of m_c(t): the "
                             "quantity pymc-marketing's SoftPlusHSGP multiplier estimates (it is "
                             "normalized to mean 1 over the fitted weeks)",
            "drift_ratio": "mean of m_c over the last quarter / mean over the first year (the "
                           "same for m_c and its relative path)",
        },
        "fatigue": {
            "channel": FATIGUE["channel"],
            "form": "m(t) = 1 - (1 - end) / (1 + exp(-(t - mid) / width)), t = 0 .. n_weeks - 1",
            "end": sig(fatigue_end), "mid": sig(fatigue_mid), "width": sig(fatigue_width),
            "first_week": sig(mult[FATIGUE["channel"]][0]),
            "last_week": sig(mult[FATIGUE["channel"]][-1]),
        },
        "spend_process": {
            "tv": f"flights on 4-8 weeks, off 3-8; {tv_level:g} x Uniform(0.7, 1.3) when on",
            "search": f"{search_level:g} x d_t x LogNormal(0, {search_sd}); d_t = (intercept + "
                      "trend_t + seasonality_t) / intercept (follows demand)",
            "social": f"{social_level:g} x LogNormal(0, {social_sd}), x {push_mult} in a "
                      f"four-week push starting every {push_every} weeks (jitter -2..+2)",
            "display": f"{display_level:g} x LogNormal(0, {display_sd})",
            "social_push_start_weeks": [int(b) + 1 for b in pushes],
        },
        "baseline": {
            "intercept": sig(intercept),
            "trend_per_week": sig(trend_per_week),
            "fourier_sin1_cos1_sin2_cos2": [sig(f) for f in fourier],
            "fourier_phase": "2 pi x day of year / 365.25",
            "price_coef_per_unit_index": sig(price_coef),
            "price_effect_form": "price_coef * (price_index - 1)",
            "price_process": "AR(1) around 1 with coefficient 0.7 and sd 0.012; ten promotion "
                             "weeks 4 to 10% lower; rounded to 4 decimals",
            "holiday_effect": sig(holiday_effect),
            "holiday_rule": "weeks containing Black Friday, December 18 or December 24",
            "noise_sd": sig(noise_sd),
        },
        "y_max": sig(float(y.max())),
        "mean_weekly_spend": {c: sig(spend[c].mean()) for c in CHANNELS},
        "mean_weekly_sales": sig(y.mean()),
        "media_share_of_sales": sig(media.sum() / y.sum()),
        "economics": {
            "gross_margin": gross_margin,
            "break_even_roas": sig(break_even),
            "rule": "cut a channel's budget next quarter if P(current-quarter ROAS < break-even) "
                    ">= 0.9; keep if P(> break-even) >= 0.9; otherwise run a lift test first",
            FATIGUE["channel"]: {
                "pays_back_all_period": bool(soc["roas"] > break_even),
                "pays_back_first_year": bool(soc["roas_first_year"] > break_even),
                "pays_back_last_quarter": bool(soc["roas_last_quarter"] > break_even),
                "spend_last_quarter": soc["spend"]["last_quarter"],
                "net_return_last_quarter": sig(
                    soc["spend"]["last_quarter"] * (gross_margin * soc["roas_last_quarter"] - 1)),
                "net_return_definition": "spend x (gross_margin x ROAS - 1): the margin on the "
                                         "sales the spend produced, net of the spend",
            },
        },
        "lift_tests": {
            "definition": (
                "One randomized geo test on social, scaled to the national level (the weekly "
                "spend series is unchanged): weekly spend raised from x (mean social spend over "
                "the 26 weeks before the test, rounded to $100) to x + delta_x (delta_x = 0.3 x, "
                "rounded to $100) for test_weeks weeks ending in the week `date`. true_delta_y = "
                "B_social x mean of m_social over the test weeks x [s((x + delta_x) / S) - "
                "s(x / S)] (steady state: the saturation curve only, as "
                "add_lift_test_measurements models it, times the multiplier as "
                "pymc-marketing's time-varying lift test does at `date`); delta_y = true_delta_y "
                "+ Normal(0, sigma), sigma = 8% of the true lift (Module 9's relative error)."
            ),
            "rel_sigma": lt["rel_sigma"],
            "test_weeks": lt["weeks"],
            "multiplier_mean_over_test": sig(m_bar),
            "rows": [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                      for k, v in lift_row.items()}],
        },
    }
    return SynthResult("mmm_drift", {"weekly": weekly, "latent": latent,
                                     "lift_tests": lift_tests}, truth)


__all__ = ["CHANNELS", "FATIGUE", "GROSS_MARGIN", "fatigue_multiplier", "mmm_drift",
           "social_push_starts"]
