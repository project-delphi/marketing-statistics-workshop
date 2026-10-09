"""Weekly marketing-mix data with known adstock, saturation and ROAS.

The data-generating process is written in the same functional form as pymc-marketing 1.2.0's
``MMM`` with ``GeometricAdstock(l_max)`` and ``LogisticSaturation()`` (checked in the installed
source, 2026-10-09):

* ``GeometricAdstock``: weights ``alpha**l`` for lags ``l = 0 .. l_max-1``, **normalized** to sum to
  1 (``AdstockTransformation`` default ``normalize=True``), convolution mode ``After``, zero spend
  assumed before the first week.
* ``LogisticSaturation``: ``beta * (1 - exp(-lam * x)) / (1 + exp(-lam * x))``.
* ``MMM`` default scaling: each channel is divided by its maximum over the fitted dates and the
  target by its maximum (``DataDerivedScaling(method="max")``); adstock comes before saturation.

So for channel c with spend ``x_c`` and ``S_c = max_t x_c(t)``:

    contribution_c(t) = B_c * logistic(lam_c * adstock(x_c)(t) / S_c)

and a model fitted on all ``n_weeks`` rows with default scaling should report
``adstock_alpha = alpha_c``, ``saturation_lam = lam_c`` and ``saturation_beta = B_c / max_t y(t)``
(the ``*_model_units`` truth). ``B_c`` itself is in sales units. ``B_c`` is solved so that the
channel's ROAS over all weeks equals the requested ``roas``.

Baseline: intercept + linear trend + yearly Fourier seasonality (two orders on day of year /
365.25, as pymc-marketing's ``YearlyFourier``) + price-index effect + holiday effect + Gaussian
noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mktstats.synth._common import SynthResult, make_rng, sig, sig_dict

MMM_CHANNELS = ("tv", "search", "social", "display")
MMM_ADSTOCK_ALPHA = {"tv": 0.6, "search": 0.2, "social": 0.4, "display": 0.3}
MMM_SATURATION_LAM = {"tv": 1.5, "search": 2.5, "social": 2.0, "display": 3.0}
MMM_ROAS = {"tv": 1.6, "search": 3.0, "social": 1.4, "display": 0.6}

# confounded variant (Module 9): hidden demand shock shared by search spend and sales
# Chosen by fitting pymc-marketing 1.2.0 (see data/README.md): strong enough that an uncalibrated
# MMM's 94% HDI for search ROAS excludes the truth, weak enough that the two search lift tests
# bring it back inside.
CONFOUNDED_DEMAND_PHI = 0.7
CONFOUNDED_SPEND_ELASTICITY = 0.15
CONFOUNDED_SALES_EFFECT = 9_000.0
CONFOUNDED_LIFT_REL_SIGMA = 0.08
# dated lift tests of the confounded variant: (channel, end of the test as a fraction of the
# window, relative change in weekly spend during the test; -1 = channel switched off). Noise is
# drawn in this order; the table is sorted by date.
LIFT_TEST_PLAN = (("search", 0.40, 0.3), ("tv", 0.55, 0.3), ("social", 0.70, 0.3),
                  ("display", 0.85, 0.3), ("search", 0.65, -1.0))
CONFOUNDED_STREAM_OFFSET = 1_000_003  # seed of the demand stream = seed + this


def _demand_shock(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """Stationary AR(1) with coefficient ``phi``, standardized to sample mean 0 and sd 1."""
    e = rng.normal(0.0, 1.0, n)
    d = np.empty(n)
    d[0] = e[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        d[i] = phi * d[i - 1] + e[i]
    return (d - d.mean()) / d.std()


def geometric_adstock(x, alpha: float, l_max: int = 8, normalize: bool = True) -> np.ndarray:
    """Geometric adstock along axis 0, zero history before the first row (numpy)."""
    x = np.asarray(x, dtype=float)
    w = alpha ** np.arange(l_max)
    if normalize:
        w = w / w.sum()
    out = np.zeros_like(x)
    for lag in range(l_max):
        if lag == 0:
            out += w[0] * x
        else:
            out[lag:] += w[lag] * x[:-lag]
    return out


def logistic_saturation(x, lam: float) -> np.ndarray:
    """``(1 - exp(-lam x)) / (1 + exp(-lam x))`` (pymc-marketing's ``logistic_saturation``)."""
    e = np.exp(-lam * np.asarray(x, dtype=float))
    return (1 - e) / (1 + e)


def steady_state_response(x, big_b: float, lam: float, scale: float) -> np.ndarray:
    """Weekly contribution of constant weekly spend ``x`` (normalized adstock of a constant is the
    constant): ``B * logistic(lam * x / S)``."""
    return big_b * logistic_saturation(np.asarray(x, dtype=float) / scale, lam)


def optimal_allocation(budget: float, big_b, lam, scale, lower, upper, tol: float = 1e-9):
    """Split a weekly ``budget`` across channels to maximize total steady-state contribution
    ``sum_c B_c * logistic(lam_c * x_c / S_c)`` subject to ``sum x_c = budget`` and
    ``lower_c <= x_c <= upper_c``.

    ``logistic(z) = tanh(z / 2)`` is concave for z >= 0, so the optimum is unique and satisfies
    the KKT conditions: every channel strictly inside its bounds has the same marginal return
    ``eta``. Solved exactly by bisection on ``eta`` (no random starts). Returns ``(x, eta)``.
    """
    big_b, lam, scale, lower, upper = (np.asarray(a, dtype=float)
                                       for a in (big_b, lam, scale, lower, upper))
    if not lower.sum() - 1e-9 <= budget <= upper.sum() + 1e-9:
        raise ValueError("budget is outside the range allowed by the bounds")
    k = big_b * lam / (2 * scale)  # marginal return at zero spend

    def spend_at(eta):
        ratio = np.clip(eta / k, 0.0, 1.0)
        x = np.where(ratio < 1, (2 * scale / lam) * np.arctanh(np.sqrt(1 - ratio)), 0.0)
        return np.clip(x, lower, upper)

    lo, hi = 0.0, float(k.max())
    for _ in range(200):
        mid = (lo + hi) / 2
        if spend_at(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    eta = (lo + hi) / 2
    return spend_at(eta), eta


def optimal_allocation_value(budget: float, big_b, lam, scale, lower, upper, *,
                             margin: float = 1.0, linear=None, tol: float = 1e-12):
    """Like :func:`optimal_allocation`, for the long-run objective

        sum_c margin * B_c * logistic(lam_c * x_c / S_c) + linear_c * x_c

    (``linear_c``: value per dollar that does not saturate, e.g. new customers per dollar times
    their future margin). Concave plus linear is concave, so the optimum is unique and is found
    exactly by bisection on the common marginal value ``eta`` of the channels inside their bounds.
    Marginal value of channel c at spend x: ``margin * k_c * (1 - tanh(lam_c x / (2 S_c))**2) +
    linear_c`` with ``k_c = B_c lam_c / (2 S_c)``. Returns ``(x, eta)``.
    """
    big_b, lam, scale, lower, upper = (np.asarray(a, dtype=float)
                                       for a in (big_b, lam, scale, lower, upper))
    lin = np.zeros_like(big_b) if linear is None else np.asarray(linear, dtype=float)
    if not lower.sum() - 1e-9 <= budget <= upper.sum() + 1e-9:
        raise ValueError("budget is outside the range allowed by the bounds")
    k = margin * big_b * lam / (2 * scale)  # short-run marginal value at zero spend

    def spend_at(eta):
        ratio = np.clip((eta - lin) / k, 0.0, 1.0)
        with np.errstate(divide="ignore"):
            x = np.where(ratio <= 0, np.inf,
                         np.where(ratio < 1, (2 * scale / lam) * np.arctanh(np.sqrt(1 - ratio)),
                                  0.0))
        return np.clip(x, lower, upper)

    lo, hi = min(0.0, float(lin.min())), float((k + lin).max())
    for _ in range(400):
        mid = (lo + hi) / 2
        if spend_at(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(hi)):
            break
    eta = (lo + hi) / 2
    x = spend_at(eta)
    # put the bisection's rounding residue (< 1e-6 of the budget) on an interior channel
    inside = (x > lower + 1e-9) & (x < upper - 1e-9)
    if inside.any():
        j = int(np.flatnonzero(inside)[0])
        x[j] += budget - x.sum()
    return x, eta


def long_run_value(x, big_b, lam, scale, *, margin: float = 1.0, linear=None) -> np.ndarray:
    """Per-channel weekly long-run value ``margin * B_c * logistic(lam_c x_c / S_c) + linear_c *
    x_c`` (the objective of :func:`optimal_allocation_value`)."""
    x = np.asarray(x, dtype=float)
    lin = 0.0 if linear is None else np.asarray(linear, dtype=float)
    return margin * steady_state_response(x, big_b, lam, scale) + lin * x


def true_response(spend, channel_truth: dict) -> np.ndarray:
    """True steady-state weekly sales response of constant weekly ``spend`` for one channel, from
    its truth block (``truth["mmm"]["channels"][c]``): ``saturation_beta_sales_units *
    logistic(saturation_lam * spend / channel_scale)``."""
    return steady_state_response(spend, channel_truth["saturation_beta_sales_units"],
                                 channel_truth["saturation_lam"], channel_truth["channel_scale"])


def _thanksgiving(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + pd.Timedelta(weeks=3)


def _tv_flighting(rng, n):
    """On/off bursts: on for 4-8 weeks, off for 3-8 weeks."""
    on = np.zeros(n, dtype=bool)
    t = int(rng.integers(0, 4))
    state = True
    while t < n:
        length = int(rng.integers(4, 9) if state else rng.integers(3, 9))
        on[t : t + length] = state
        t += length
        state = not state
    return on


def mmm(
    seed: int = 2028,
    *,
    n_weeks: int = 156,
    start: str = "2023-01-02",
    l_max: int = 8,
    adstock_alpha: dict[str, float] | None = None,
    saturation_lam: dict[str, float] | None = None,
    roas: dict[str, float] | None = None,
    intercept: float = 250_000.0,
    trend_per_week: float = 250.0,
    fourier: tuple[float, float, float, float] = (15_000.0, 25_000.0, 5_000.0, -8_000.0),
    price_coef: float = -400_000.0,
    holiday_effect: float = 40_000.0,
    noise_sd: float = 8_000.0,
    lift_delta_frac: float = 0.3,
    lift_rel_sigma: float | None = None,
    alloc_bounds: tuple[float, float] = (0.5, 2.0),
    confounded: bool = False,
    confounded_channel: str = "search",
    demand_phi: float = CONFOUNDED_DEMAND_PHI,
    demand_spend_elasticity: float = CONFOUNDED_SPEND_ELASTICITY,
    demand_sales_effect: float = CONFOUNDED_SALES_EFFECT,
    lift_test_weeks: int = 8,
    lift_test_plan: tuple[tuple[str, float, float], ...] | None = None,
) -> SynthResult:
    """Weekly sales and spend for tv, search, social and display.

    Tables
    ------
    weekly : date_week, tv, search, social, display (spend, dollars), price_index, holiday (0/1),
        t (0..n_weeks-1), y (sales).
    lift_tests : channel, x, delta_x, delta_y, sigma — the format of
        ``MMM.add_lift_test_measurements`` (pymc-marketing 1.2.0): steady-state weekly sales lift
        ``delta_y`` from raising weekly spend from ``x`` to ``x + delta_x``, measured with standard
        error ``sigma`` (``delta_y`` = true lift + Normal(0, sigma) noise); ``true_delta_y`` is the
        noiseless value on the true curve (an extra column the method ignores).

    ``confounded=True`` (Module 9, "what an MMM cannot identify") adds an unobserved weekly demand
    shock ``d_t`` (stationary AR(1) with coefficient ``demand_phi``, standardized to mean 0 and
    standard deviation 1, drawn from its own random stream so every other draw is the same as with
    ``confounded=False``). Spend on ``confounded_channel`` is multiplied by
    ``exp(demand_spend_elasticity * d_t)`` (paid search is bought per click, so spend rises when
    more people search) and sales rise by ``demand_sales_effect * d_t``. The channel's causal
    contribution is unchanged, so its true ROAS is still the requested one, but a model that does
    not see ``d_t`` credits the channel with the demand effect. The variant also returns

    * ``lift_tests`` with dates: one row per test in ``lift_test_plan`` (``(channel, end,
      change)``: the test lasts ``lift_test_weeks`` weeks, ends at fraction ``end`` of the window
      and changes weekly spend by ``change`` times ``x``; default ``LIFT_TEST_PLAN``: one +30% test
      per channel plus a second search test that switches search off); extra columns ``date``
      (last week of the test: the result is known from then on; the column for
      ``TimeSliceCrossValidator.run(..., lift_test_date_column="date")``), ``test_start``,
      ``test_weeks``; ``x`` is the channel's mean positive weekly spend over the 26 weeks before
      the test; ``sigma`` is ``lift_rel_sigma`` (default 8% here) of the true lift;
    * ``latent`` : date_week, demand_shock (``d_t``), demand_effect (``demand_sales_effect *
      d_t``, dollars), media_contribution_<channel> (noiseless, dollars) — never shown to a model,
      only for revealing the mechanism;
    * truth ``confounding``: the mechanism and the omitted-variable bias of a regression that uses
      the true media transforms but omits ``d_t`` (``roas[c].naive_ols``), next to the same
      regression with ``d_t`` (``roas[c].oracle_ols``).
    """
    if lift_rel_sigma is None:  # relative standard error of the lift tests
        lift_rel_sigma = CONFOUNDED_LIFT_REL_SIGMA if confounded else 0.15
    alpha_c = dict(MMM_ADSTOCK_ALPHA if adstock_alpha is None else adstock_alpha)
    lam_c = dict(MMM_SATURATION_LAM if saturation_lam is None else saturation_lam)
    roas_c = dict(MMM_ROAS if roas is None else roas)
    channels = MMM_CHANNELS
    if confounded and confounded_channel not in channels:
        raise ValueError(f"confounded_channel must be one of {channels}")
    rng = make_rng(seed)
    n = n_weeks
    dates = pd.date_range(start, periods=n, freq="7D")
    t = np.arange(n)
    doy = dates.dayofyear.to_numpy()
    ph = 2 * np.pi * doy / 365.25
    season = (fourier[0] * np.sin(ph) + fourier[1] * np.cos(ph)
              + fourier[2] * np.sin(2 * ph) + fourier[3] * np.cos(2 * ph))
    demand = season / max(np.abs(season).max(), 1.0)  # in [-1, 1]

    # spend patterns (dollars per week)
    tv_on = _tv_flighting(rng, n)
    tv = np.where(tv_on, 60_000 * rng.uniform(0.7, 1.3, n), 0.0)
    search = 20_000 * (1 + 0.25 * demand) * rng.lognormal(0, 0.12, n)
    social = 12_000 * rng.lognormal(0, 0.25, n)
    for b0 in rng.choice(np.arange(4, n - 4), size=3, replace=False):
        social[b0 : b0 + 4] *= 1.8
    display = 6_000 * np.linspace(0.6, 1.3, n) * rng.lognormal(0, 0.3, n)
    spend = {"tv": tv, "search": search, "social": social, "display": display}
    if confounded:
        # own stream, so the main stream (and every other column) matches confounded=False
        demand_shock = _demand_shock(make_rng(seed + CONFOUNDED_STREAM_OFFSET), n, demand_phi)
        spend[confounded_channel] = (spend[confounded_channel]
                                     * np.exp(demand_spend_elasticity * demand_shock))
    spend = {c: np.round(v, 0) for c, v in spend.items()}

    # controls
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
    scale = {c: float(spend[c].max()) for c in channels}
    sat = {c: logistic_saturation(geometric_adstock(spend[c], alpha_c[c], l_max) / scale[c],
                                  lam_c[c]) for c in channels}
    big_b = {c: roas_c[c] * spend[c].sum() / sat[c].sum() for c in channels}
    contrib = {c: big_b[c] * sat[c] for c in channels}
    media = sum(contrib.values())

    baseline = (intercept + trend_per_week * t + season + price_coef * (price - 1.0)
                + holiday_effect * holiday)
    noise = rng.normal(0, noise_sd, n)
    demand_effect = demand_sales_effect * demand_shock if confounded else 0.0
    y = np.round(baseline + media + noise + demand_effect, 0)

    weekly = pd.DataFrame({"date_week": dates})
    for c in channels:
        weekly[c] = spend[c].astype(np.int64)
    weekly["price_index"] = price
    weekly["holiday"] = holiday
    weekly["t"] = t
    weekly["y"] = y.astype(np.int64)

    # ROAS with carry-out: extend spend with l_max - 1 zero weeks and sum the tail as well
    roas_carry = {}
    for c in channels:
        ext = np.r_[spend[c], np.zeros(l_max - 1)]
        sat_ext = logistic_saturation(geometric_adstock(ext, alpha_c[c], l_max) / scale[c],
                                      lam_c[c])
        roas_carry[c] = big_b[c] * sat_ext.sum() / spend[c].sum()

    # lift tests on the true curve (steady state: normalized adstock of a constant is the constant)
    last52 = slice(max(0, n - 52), n)
    if confounded:
        rows = _dated_lift_tests(rng, dates, spend, big_b, lam_c, scale, lift_test_weeks,
                                 LIFT_TEST_PLAN if lift_test_plan is None else lift_test_plan,
                                 lift_rel_sigma)
    else:
        rows = []
        for c in channels:
            recent = spend[c][last52]
            x0 = float(np.round(recent[recent > 0].mean(), -2))
            dx = float(np.round(lift_delta_frac * x0, -2))
            true_dy = big_b[c] * (logistic_saturation((x0 + dx) / scale[c], lam_c[c])
                                  - logistic_saturation(x0 / scale[c], lam_c[c]))
            sigma = lift_rel_sigma * true_dy
            dy = true_dy + rng.normal(0, sigma)
            rows.append({"channel": c, "x": x0, "delta_x": dx,
                         "delta_y": float(np.round(dy, 0)), "sigma": float(np.round(sigma, 0)),
                         "true_delta_y": float(np.round(true_dy, 0))})
    lift_tests = pd.DataFrame(rows)

    # answer key for budget allocation: steady-state weekly split of the recent average budget
    cur = np.array([spend[c][last52].mean() for c in channels])
    budget = float(np.round(cur.sum(), -3))
    cur_scaled = cur * budget / cur.sum()
    lower, upper = alloc_bounds[0] * cur, alloc_bounds[1] * cur
    bb = np.array([big_b[c] for c in channels])
    ll = np.array([lam_c[c] for c in channels])
    ss = np.array([scale[c] for c in channels])
    x_opt, eta = optimal_allocation(budget, bb, ll, ss, lower, upper)
    resp_opt = steady_state_response(x_opt, bb, ll, ss)
    resp_cur = steady_state_response(cur_scaled, bb, ll, ss)
    marginal = bb * ll / (2 * ss) * (1 - np.tanh(ll * x_opt / (2 * ss)) ** 2)
    allocation = {
        "definition": (
            "Steady-state weekly allocation (the same spend every week, so normalized adstock "
            "equals spend) maximizing the sum of true weekly channel contributions "
            "B_c * logistic(lam_c * x_c / S_c) subject to sum(x_c) = budget and per-channel "
            "bounds. The problem is concave (logistic = tanh(z/2)), so the optimum is unique; it "
            "is solved exactly by equalizing marginal returns (bisection on the Lagrange "
            "multiplier, mktstats.synth.mmm.optimal_allocation), no random starts."
        ),
        "weekly_budget": budget,
        "budget_rule": "mean weekly total spend over the last 52 weeks, rounded to $1,000",
        "bounds_rule": (f"{alloc_bounds[0]} to {alloc_bounds[1]} times each channel's mean weekly "
                        "spend over the last 52 weeks"),
        "bounds": {c: [sig(lower[j]), sig(upper[j])] for j, c in enumerate(channels)},
        "current_allocation": {c: sig(cur_scaled[j]) for j, c in enumerate(channels)},
        "optimal_allocation": {c: sig(x_opt[j]) for j, c in enumerate(channels)},
        "optimal_share": {c: sig(x_opt[j] / budget) for j, c in enumerate(channels)},
        "weekly_contribution_current": sig(resp_cur.sum()),
        "weekly_contribution_optimal": sig(resp_opt.sum()),
        "weekly_gain": sig(resp_opt.sum() - resp_cur.sum()),
        "marginal_roas_at_optimum": {c: sig(marginal[j]) for j, c in enumerate(channels)},
        "lagrange_multiplier": sig(eta),
        "at_bound": {c: ("lower" if np.isclose(x_opt[j], lower[j]) else
                         "upper" if np.isclose(x_opt[j], upper[j]) else "interior")
                     for j, c in enumerate(channels)},
    }

    y_scale = float(y.max())
    total_media = float(media.sum())
    total_sales = float(y.sum())
    channel_truth = {}
    for c in channels:
        channel_truth[c] = {
            "adstock_alpha": sig(alpha_c[c]),
            "saturation_lam": sig(lam_c[c]),
            "saturation_beta_model_units": sig(big_b[c] / y_scale),
            "saturation_beta_sales_units": sig(big_b[c]),
            "channel_scale": sig(scale[c]),
            "roas": sig(contrib[c].sum() / spend[c].sum()),
            "roas_with_carryover": sig(roas_carry[c]),
            "contribution": sig(contrib[c].sum()),
            "spend": sig(spend[c].sum()),
            "contribution_share": sig(contrib[c].sum() / total_media),
            "share_of_sales": sig(contrib[c].sum() / total_sales),
        }
    truth = {
        "description": (
            "Weekly sales = intercept + trend + yearly seasonality + price and holiday effects + "
            "sum of channel contributions + Normal noise; channel contribution = B_c * "
            "logistic(lam_c * normalized_geometric_adstock(spend_c) / max(spend_c))."
        ),
        "seed": seed,
        "n_weeks": n,
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "l_max": l_max,
        "adstock": "GeometricAdstock(l_max=8, normalize=True), zero spend before the first week",
        "saturation": "LogisticSaturation: beta * (1 - exp(-lam x)) / (1 + exp(-lam x))",
        "units": (
            "adstock_alpha and saturation_lam are scale-free (lam applies to spend / max spend "
            "over all weeks); saturation_beta_model_units = B_c / max(y) is what "
            "pymc-marketing's MMM reports when fitted on all rows with default max scaling; "
            "*_sales_units are dollars."
        ),
        "roas_definition": (
            "roas = sum over all weeks of the channel's noiseless contribution / sum of its spend "
            "over all weeks (zero spend before week 1; carry-over past the last week not counted). "
            "roas_with_carryover adds the contribution of in-window spend in the l_max - 1 weeks "
            "after the last week."
        ),
        "roas_window": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        "y_scale": sig(y_scale),
        "channels": channel_truth,
        "media_share_of_sales": sig(total_media / total_sales),
        "baseline": {
            "intercept": sig(intercept),
            "trend_per_week": sig(trend_per_week),
            "fourier_sin1_cos1_sin2_cos2": [sig(f) for f in fourier],
            "price_coef_per_unit_index": sig(price_coef),
            "price_effect_form": "price_coef * (price_index - 1)",
            "holiday_effect": sig(holiday_effect),
            "noise_sd": sig(noise_sd),
        },
        "lift_tests": {
            "definition": (
                "steady-state weekly sales lift when weekly spend rises from x to x + delta_x "
                "(the saturation curve only, as add_lift_test_measurements models it); "
                "delta_y = true_delta_y + Normal(0, sigma)."
            ),
            "true_delta_y": {r["channel"]: r["true_delta_y"] for r in rows},
        },
        "true_roas_target": sig_dict(roas_c),
        "true_optimal_allocation": allocation,
    }
    if not confounded:
        return SynthResult("mmm", {"weekly": weekly, "lift_tests": lift_tests}, truth)

    # ------------------------------------------------------------------ confounded variant only
    latent = pd.DataFrame({"date_week": dates, "demand_shock": np.round(demand_shock, 6),
                           "demand_effect": np.round(demand_effect, 2)})
    for c in channels:
        latent[f"media_contribution_{c}"] = np.round(contrib[c], 2)
    truth["description"] += (
        f" Confounded variant: an unobserved demand shock d_t (AR(1), phi = {demand_phi}, "
        f"standardized) multiplies {confounded_channel} spend by exp({demand_spend_elasticity} * "
        f"d_t) and adds {demand_sales_effect:g} * d_t to sales. Channel contributions, and so the "
        "true ROAS, are causal and do not include the demand effect."
    )
    truth["lift_tests"] = {
        "definition": (
            "Randomized geo tests, each lasting test_weeks weeks and ending in the week `date`. "
            "A test changed weekly spend from x (the channel's mean positive weekly spend over "
            "the 26 weeks before the test) to x + delta_x (delta_x = -x: the channel was switched "
            "off in the test regions, scaled to the national level); true_delta_y is the "
            "steady-state weekly sales change on the true curve (saturation only, as "
            "add_lift_test_measurements models it), delta_y = true_delta_y + Normal(0, sigma), "
            "sigma = rel_sigma * |true_delta_y|. Randomized tests measure the causal effect, so "
            "the demand shock does not bias them. Use `date` as lift_test_date_column in "
            "TimeSliceCrossValidator.run: a fold may use a test only if the test ended on or "
            "before the fold's last training week."
        ),
        "test_weeks": lift_test_weeks,
        "rel_sigma": lift_rel_sigma,
        "plan": [{"channel": c, "end_fraction": e, "spend_change": ch}
                 for c, e, ch in (LIFT_TEST_PLAN if lift_test_plan is None else lift_test_plan)],
        "rows": [{k: (v if not isinstance(v, pd.Timestamp) else str(v.date()))
                  for k, v in r.items()} for r in rows],
    }
    truth["confounding"] = _confounding_truth(
        weekly, dates, sat, spend, big_b, demand_shock, channels, confounded_channel,
        demand_phi, demand_spend_elasticity, demand_sales_effect, fourier_phase=ph)
    return SynthResult("mmm_confounded",
                       {"weekly": weekly, "lift_tests": lift_tests, "latent": latent}, truth)


def mmm_confounded(seed: int = 2028, **kwargs) -> SynthResult:
    """``mmm(seed, confounded=True, **kwargs)``: Module 9's data (see :func:`mmm`).

    With the default seed, tv, social and display spend, prices and holidays are identical to
    Module 8's ``mmm()``; search spend and sales differ by the hidden demand shock.
    """
    return mmm(seed, confounded=True, **kwargs)


def _dated_lift_tests(rng, dates, spend, big_b, lam_c, scale, weeks, plan, rel_sigma):
    """Randomized lift tests with their dates (see ``mmm(confounded=True)``).

    A negative change is a spend cut (``change = -1`` turns the channel off in the test regions);
    ``delta_x`` and ``delta_y`` are then negative, which ``add_lift_test_measurements`` accepts
    (it compares absolute values and requires ``delta_x * delta_y >= 0``)."""
    n = len(dates)
    rows = []
    for c, end_frac, change in plan:
        end = int(round(end_frac * n)) - 1
        start = end - weeks + 1
        if start < 26:
            raise ValueError(f"lift test for {c} starts before week 26")
        before = spend[c][start - 26:start]
        x0 = float(np.round(before[before > 0].mean(), -2))
        dx = float(np.round(change * x0, -2))
        true_dy = big_b[c] * (logistic_saturation((x0 + dx) / scale[c], lam_c[c])
                              - logistic_saturation(x0 / scale[c], lam_c[c]))
        sigma = rel_sigma * abs(true_dy)
        dy = true_dy + rng.normal(0, sigma)
        rows.append({"channel": c, "x": x0, "delta_x": dx, "delta_y": float(np.round(dy, 0)),
                     "sigma": float(np.round(sigma, 0)), "date": dates[end],
                     "test_start": dates[start], "test_weeks": weeks,
                     "true_delta_y": float(np.round(true_dy, 0))})
    return sorted(rows, key=lambda row: row["date"])


def _ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _confounding_truth(weekly, dates, sat, spend, big_b, demand_shock, channels, conf_channel,
                       phi, elasticity, sales_effect, fourier_phase):
    """The omitted-variable-bias mechanism, computed exactly on this realization.

    Both regressions use the true media transforms (true adstock alpha and saturation lam), so only
    the media coefficients B_c are estimated; that isolates confounding from the harder problem of
    learning the curves. ``naive`` omits d_t (what an MMM sees), ``oracle`` includes it. OLS
    identity: naive_B_c = oracle_B_c + oracle_coef_d * delta_c, where delta_c is the coefficient on
    channel c in the regression of d_t on the naive regressors.
    """
    y = weekly["y"].to_numpy(dtype=float)
    ph = fourier_phase
    base = np.column_stack([
        np.ones(len(y)), weekly["t"].to_numpy(float), np.sin(ph), np.cos(ph), np.sin(2 * ph),
        np.cos(2 * ph), weekly["price_index"].to_numpy(float), weekly["holiday"].to_numpy(float)])
    media_x = np.column_stack([sat[c] for c in channels])
    naive_x = np.column_stack([base, media_x])
    oracle_x = np.column_stack([naive_x, demand_shock])
    k = base.shape[1]
    b_naive = _ols(naive_x, y)[k:k + len(channels)]
    b_oracle_all = _ols(oracle_x, y)
    b_oracle, coef_d = b_oracle_all[k:k + len(channels)], b_oracle_all[-1]
    delta = _ols(naive_x, demand_shock)[k:k + len(channels)]
    roas = {}
    for j, c in enumerate(channels):
        per_b = sat[c].sum() / spend[c].sum()  # ROAS per unit of B_c
        roas[c] = {"true": sig(big_b[c] * per_b), "naive_ols": sig(b_naive[j] * per_b),
                   "oracle_ols": sig(b_oracle[j] * per_b),
                   "omitted_variable_bias": sig((b_naive[j] - b_oracle[j]) * per_b)}
    log_spend = np.log(weekly[conf_channel].to_numpy(dtype=float))
    corr_spend = float(np.corrcoef(demand_shock, weekly[conf_channel].to_numpy(float))[0, 1])
    return {
        "channel": conf_channel,
        "mechanism": (
            f"Unobserved demand shock d_t raises {conf_channel} spend (log spend + "
            f"{elasticity} * d_t) and sales (+ {sales_effect:g} * d_t) in the same weeks. A model "
            f"without d_t attributes the demand effect to {conf_channel}, so its ROAS is "
            "overstated; the other channels' spend does not depend on d_t."
        ),
        "demand_shock": {"process": "AR(1), standardized to sample mean 0 and sd 1",
                         "phi": phi, "spend_elasticity": elasticity,
                         "sales_effect_per_sd": sig(sales_effect),
                         "observed": False,
                         "weekly_series": "table `latent`, column demand_shock"},
        "corr_demand_spend": sig(corr_spend),
        "corr_demand_log_spend": sig(float(np.corrcoef(demand_shock, log_spend)[0, 1])),
        "roas": roas,
        "ols_definition": (
            "OLS of y on intercept, t, the generator's yearly Fourier terms (order 2), "
            "price_index, holiday and each channel's true transformed spend "
            "logistic(lam_c * adstock(spend_c) / S_c) (true alpha and lam); naive omits d_t, "
            "oracle adds it. ROAS = coefficient * sum(transformed spend) / sum(spend); "
            "omitted_variable_bias = naive_ols - oracle_ols. It shows the direction and size of "
            "the bias an MMM fitted without d_t inherits; a Bayesian MMM also estimates alpha and "
            "lam and uses priors, so its posterior need not equal it. OLS values for a small "
            "channel can be far from the truth for reasons unrelated to confounding (display "
            "spend trends with t); compare naive with oracle, not with the truth."
        ),
        "ovb_identity": {
            "formula": "naive_B_c = oracle_B_c + oracle_coef_d * delta_c",
            "oracle_coef_d": sig(coef_d),
            "delta": {c: sig(delta[j]) for j, c in enumerate(channels)},
        },
    }
