"""Module 17 (MMM with drifting effects and a Prophet baseline): reference implementations of the
lab's functions, the checkpoint checks (each failure message says what failed, the likely cause,
what to check and how to recover), the three model specifications, and the reference pipeline the
tests and the seed sweep run.

Notation (briefs/17-mmm-time-varying.md): contribution_c(t) = beta_c * m_c(t) * s(a_c(t) / S_c);
ROAS over a window W is the channel's contribution in W over its spend in W; the drift ratio is the
mean of the multiplier over the last quarter over its mean over the first year. Data:
``mktstats.synth.mmm_drift`` (seed 2037), committed as ``data/synthetic/mmm_drift_*.csv`` and
``truth_mmm_drift.json``.

pymc-marketing 1.2.0, nutpie 0.16.11, prophet 1.4.0 and xarray 2025.12.0 are imported lazily, so
importing this module costs nothing (the lab imports it for its checks).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from mktstats import checks

CHANNELS = ["tv", "search", "social", "display"]
GROSS_MARGIN = 0.30
BREAK_EVEN = 1 / GROSS_MARGIN
PROB = 0.94
FIT_SETTINGS = {"nuts_sampler": "nutpie", "chains": 2, "cores": 2, "target_accept": 0.98,
                "progressbar": False}
# The drifting model's multiplier prior (SoftPlusHSGP.parameterize_from_data): lengthscale
# InverseGamma with 90% of its mass in [20, 104] weeks; amplitude Exponential with
# P(eta > 2) = 0.05; Matern 5/2 kernel, the choice pymc-marketing makes for its own
# time_varying_media=True (mmm/tvp.py; parameterize_from_data's default is ExpQuad).
HSGP_SETTINGS = {"ls_lower": 20, "ls_upper": 104, "eta_upper": 2.0, "cov_func": "matern52"}
PROPHET_CONTROLS = ["price_index", "prophet_trend", "prophet_season", "prophet_holiday"]


def _recover(n: int | str, rerun: bool = False) -> str:
    if rerun:
        return (f" To move on now, run workshop.use_reference({n}) and rerun from Checkpoint"
                f" {n}.")
    return f" To move on now, run workshop.use_reference({n}) and come back to this later."


def _fmt(x, digits: int = 4) -> str:
    return f"{float(x):.{digits}g}"


# ------------------------------------------------------------------------------- references
def window_roas(contrib: xr.DataArray, spend: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """ROAS over the weeks from ``start`` to ``end`` (both included): the window's contribution
    over the window's spend. ``contrib`` (..., date, channel), ``spend`` (date, channel)."""
    sales = contrib.sel(date=slice(start, end)).sum("date")
    cost = spend.sel(date=slice(start, end)).sum("date")
    return sales / cost


def residual_spend_corr(resid: pd.Series, spend: pd.DataFrame, year: pd.Series) -> pd.DataFrame:
    """Pearson correlation of the residuals with each channel's weekly spend, within each year:
    index year, one column per channel."""
    rows = {}
    for yr in sorted(year.unique()):
        rows[yr] = spend[year == yr].corrwith(resid[year == yr])
    return pd.DataFrame(rows).T.rename_axis("year")


def drift_ratio(mult: xr.DataArray, early: tuple[str, str], late: tuple[str, str]) -> xr.DataArray:
    """Mean of ``mult`` over the ``late`` weeks over its mean over the ``early`` weeks."""
    return mult.sel(date=slice(*late)).mean("date") / mult.sel(date=slice(*early)).mean("date")


def path_coverage(mult_draws: xr.DataArray, true_path, prob: float = PROB) -> float:
    """Share of weeks whose true value lies inside that week's ``prob`` HDI of the draws.
    ``mult_draws`` has dims (chain, draw, date) for one channel."""
    true_path = np.asarray(true_path, dtype=float)
    inside = []
    for k in range(mult_draws.sizes["date"]):
        lo, hi = checks.interval(mult_draws.isel(date=k).values, prob, "hdi")
        inside.append(lo <= true_path[k] <= hi)
    return float(np.mean(inside))


def baseline_comparison(roas_own: xr.DataArray, roas_prophet: xr.DataArray, true_roas: Mapping,
                        prob: float = PROB) -> pd.DataFrame:
    """Per channel and baseline: posterior mean and ``prob`` HDI of all-period ROAS, the truth,
    the absolute error of the mean, and whether the HDI covers the truth. Long form, index
    (channel, baseline) with baseline in {"own", "prophet"}."""
    rows = []
    for baseline, draws in {"own": roas_own, "prophet": roas_prophet}.items():
        for c in draws.coords["channel"].values:
            d = draws.sel(channel=c).values.ravel()
            lo, hi = checks.interval(d, prob, "hdi")
            t = float(true_roas[str(c)])
            rows.append({"channel": str(c), "baseline": baseline, "mean": d.mean(),
                         "hdi_low": lo, "hdi_high": hi, "true_roas": t,
                         "abs_error": abs(d.mean() - t), "covered": lo <= t <= hi})
    return pd.DataFrame(rows).set_index(["channel", "baseline"])


def quiet_prophet() -> None:
    """Prophet logs every fit through cmdstanpy at INFO level, and off Colab, without plotly,
    an error that interactive plots will not work; neither matters here."""
    log = logging.getLogger("cmdstanpy")
    if not any(isinstance(h, logging.NullHandler) for h in log.handlers):
        log.addHandler(logging.NullHandler())  # else cmdstanpy resets the level to DEBUG
    log.setLevel(logging.WARNING)
    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)


def prophet_components(weekly: pd.DataFrame, holidays: pd.DataFrame,
                       regressors: tuple[str, ...] | list[str] = ()) -> pd.DataFrame:
    """Prophet's trend, yearly season and holiday components of sales (dollars), one row per week
    in the order of ``weekly``, and its fitted ``yhat``. ``regressors``: columns of ``weekly`` to
    add with ``add_regressor`` (stretch B)."""
    quiet_prophet()
    from prophet import Prophet

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    holidays=holidays, uncertainty_samples=0)
    df = pd.DataFrame({"ds": weekly["date_week"], "y": weekly["y"].astype(float)})
    for c in regressors:
        model.add_regressor(c)
        df[c] = weekly[c].astype(float)
    model.fit(df)
    fc = model.predict(df)
    return pd.DataFrame({"trend": fc["trend"].to_numpy(), "season": fc["yearly"].to_numpy(),
                         "holiday": fc["holidays"].to_numpy(), "yhat": fc["yhat"].to_numpy()},
                        index=weekly.index)


def holidays_table(weekly: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"holiday": "peak_week",
                         "ds": weekly.loc[weekly["holiday"] == 1, "date_week"]})


# --------------------------------------------------------------------------- data as arrays
def spend_array(weekly: pd.DataFrame, channels=CHANNELS) -> xr.DataArray:
    return xr.DataArray(weekly[channels].to_numpy(dtype=float), dims=("date", "channel"),
                        coords={"date": weekly["date_week"].to_numpy(), "channel": channels})


def true_contrib_array(latent: pd.DataFrame, channels=CHANNELS) -> xr.DataArray:
    cols = [f"contribution_{c}" for c in channels]
    return xr.DataArray(latent[cols].to_numpy(dtype=float), dims=("date", "channel"),
                        coords={"date": latent["date_week"].to_numpy(), "channel": channels})


def windows_of(truth: Mapping) -> dict[str, tuple[str, str]]:
    return {k: (w["start"], w["end"]) for k, w in truth["windows"].items()}


def target_scale(idata) -> float:
    return float(np.asarray(idata.constant_data["target_scale"]).squeeze())


def contribution_dollars(idata) -> xr.DataArray:
    """channel_contribution in dollars, dims (chain, draw, date, channel)."""
    return idata.posterior["channel_contribution"] * target_scale(idata)


# ------------------------------------------------------------------------ the three models
def beta_sigma(weekly: pd.DataFrame, channels=CHANNELS) -> xr.DataArray:
    """Module 8's spend-share scales for the saturation_beta HalfNormal priors."""
    totals = weekly[channels].sum().to_numpy(dtype=float)
    return xr.DataArray(len(channels) * totals / totals.sum(), dims="channel",
                        coords={"channel": channels})


def multiplier_prior(n_weeks: int):
    """A fresh SoftPlusHSGP for the drifting model: one multiplier path per channel."""
    from pymc_marketing.mmm import SoftPlusHSGP

    return SoftPlusHSGP.parameterize_from_data(X=np.arange(n_weeks), dims=("date", "channel"),
                                               **HSGP_SETTINGS)


def static_mmm(weekly: pd.DataFrame, time_varying: bool = False):
    """Fit 1 (``time_varying=False``) or fit 2: Module 8's specification, plus a time-varying
    multiplier per channel for fit 2."""
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    return MMM(
        date_column="date_week", channel_columns=CHANNELS,
        control_columns=["price_index", "holiday", "t"], target_column="y",
        adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
        yearly_seasonality=2,
        time_varying_media=multiplier_prior(len(weekly)) if time_varying else False,
        model_config={"saturation_beta": Prior("HalfNormal", sigma=beta_sigma(weekly),
                                               dims="channel")})


def prophet_inputs(weekly: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """X for fit 3: dates, spend, price and Prophet's components divided by max sales."""
    y_max = float(weekly["y"].max())
    return weekly[["date_week", *CHANNELS, "price_index"]].assign(
        prophet_trend=components["trend"] / y_max, prophet_season=components["season"] / y_max,
        prophet_holiday=components["holiday"] / y_max)


def prophet_mmm(weekly: pd.DataFrame):
    """Fit 3: static media; controls price_index and Prophet's three components with an informed
    prior (coefficient Normal(1, 0.1) on each component, Normal(0, 2) on price)."""
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    coords = {"control": PROPHET_CONTROLS}
    mu = xr.DataArray([0.0, 1.0, 1.0, 1.0], dims="control", coords=coords)
    sd = xr.DataArray([2.0, 0.1, 0.1, 0.1], dims="control", coords=coords)
    return MMM(
        date_column="date_week", channel_columns=CHANNELS, control_columns=PROPHET_CONTROLS,
        target_column="y", adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
        yearly_seasonality=None,
        model_config={"saturation_beta": Prior("HalfNormal", sigma=beta_sigma(weekly),
                                               dims="channel"),
                      "gamma_control": Prior("Normal", mu=mu, sigma=sd, dims="control")})


def fit(model, X: pd.DataFrame, y: pd.Series, draws: int, seed: int):
    return model.fit(X, y, draws=draws, tune=draws, random_seed=seed, **FIT_SETTINGS)


def divergences(idata) -> int:
    return int(idata["/sample_stats"]["diverging"].sum())


# ------------------------------------------------------------------- reference pipeline
def _hdi(d, prob=PROB):
    return checks.interval(np.asarray(d).ravel(), prob, "hdi")


def _summ(d, digits=4):
    d = np.asarray(d).ravel()
    lo, hi = _hdi(d)
    return [round(float(d.mean()), digits), round(lo, digits), round(hi, digits)]


def reference_results(weekly: pd.DataFrame, latent: pd.DataFrame, truth: Mapping, draws: int,
                      seed: int, *, stretch: bool = False, lift_tests: pd.DataFrame | None = None,
                      ) -> dict:
    """Fit the lab's three models with the reference solutions and return every number the
    checkpoints, the Predict bins and the decision depend on (rounded, JSON-serialisable)."""
    import warnings

    warnings.filterwarnings("ignore", message=".*merge_dataset.*")
    out: dict = {"seed": seed, "draws": draws, "seconds": {}}
    W = windows_of(truth)
    X, y = weekly.drop(columns=["y"]), weekly["y"]
    spend = spend_array(weekly)
    tc = true_contrib_array(latent)
    tr = truth["channels"]
    soc = "social"

    # fit 1: static
    t0 = time.time()
    m_static = static_mmm(weekly)
    id_static = fit(m_static, X, y, draws, seed)
    out["seconds"]["static"] = round(time.time() - t0, 1)
    out["divergences_static"] = divergences(id_static)
    c_static = contribution_dollars(id_static)
    roas_static = {k: window_roas(c_static, spend, *W[k]) for k in W}
    cos = m_static.incrementality.contribution_over_spend(frequency="all_time")
    out["static_vs_contribution_over_spend"] = float(np.abs(
        cos.transpose("chain", "draw", "channel").values
        - roas_static["all"].transpose("chain", "draw", "channel").values).max())
    out["static_roas"] = {k: {c: _summ(roas_static[k].sel(channel=c)) for c in CHANNELS}
                          for k in W}
    lq = roas_static["last_quarter"].sel(channel=soc).values
    out["static_social_lq_hdi_above_truth"] = _hdi(lq)[0] > tr[soc]["roas_last_quarter"]
    out["static_social_p_above_all"] = round(float(
        (roas_static["all"].sel(channel=soc).values > BREAK_EVEN).mean()), 4)

    # Exercise 2: residuals of the static model
    mu = id_static.posterior["mu"].mean(("chain", "draw")).values * target_scale(id_static)
    resid = pd.Series(y.to_numpy(dtype=float) - mu, index=weekly.index)
    year = weekly["date_week"].dt.year
    corr = residual_spend_corr(resid, weekly[CHANNELS], year)
    out["resid_corr"] = {str(k): {c: round(float(v), 4) for c, v in row.items()}
                         for k, row in corr.iterrows()}

    # fit 2: drifting, prior first
    m_drift = static_mmm(weekly, time_varying=True)
    out["hsgp_m_L"] = [m_drift.time_varying_media.m, round(float(m_drift.time_varying_media.L), 1)]
    t0 = time.time()
    m_drift.sample_prior_predictive(X, y, samples=1000, random_seed=seed)
    prior_mult = m_drift.idata["prior"]["media_temporal_latent_multiplier"]
    out["seconds"]["prior"] = round(time.time() - t0, 1)
    pr = drift_ratio(prior_mult, W["first_year"], W["last_quarter"]).sel(channel=soc).values
    out["prior_social_ratio"] = {
        "p_le_0.5": round(float((pr <= 0.5).mean()), 4),
        "p_le_truth": round(float((pr <= tr[soc]["drift_ratio"]).mean()), 4),
        "eti94": [round(v, 3) for v in checks.interval(pr, 0.94, "eti")]}
    t0 = time.time()
    id_drift = fit(m_drift, X, y, draws, seed)
    out["seconds"]["drift"] = round(time.time() - t0, 1)
    out["divergences_drift"] = divergences(id_drift)
    mult = id_drift.posterior["media_temporal_latent_multiplier"]
    out["coverage"] = {c: round(path_coverage(mult.sel(channel=c), latent[f"multiplier_rel_{c}"]),
                                4) for c in CHANNELS}
    ratio = drift_ratio(mult, W["first_year"], W["last_quarter"])
    out["drift_ratio"] = {c: _summ(ratio.sel(channel=c)) for c in CHANNELS}
    ratio_hdi = {c: _hdi(ratio.sel(channel=c).values) for c in CHANNELS}
    out["fading"] = [c for c in CHANNELS if ratio_hdi[c][1] < 1]
    out["constant_ratio_hdis_contain_1"] = all(
        ratio_hdi[c][0] <= 1 <= ratio_hdi[c][1] for c in CHANNELS if c != soc)
    sm = mult.sel(channel=soc).stack(s=("chain", "draw")).transpose("date", "s").values
    band = np.array([_hdi(row) for row in sm])
    true_rel = latent[f"multiplier_rel_{soc}"].to_numpy()
    miss = np.flatnonzero((true_rel < band[:, 0]) | (true_rel > band[:, 1]))
    out["social_band_miss_weeks"] = (miss + 1).tolist()  # counting from 1
    out["social_band_miss_side"] = {
        "below_truth": (np.flatnonzero(band[:, 1] < true_rel) + 1).tolist(),
        "above_truth": (np.flatnonzero(band[:, 0] > true_rel) + 1).tolist()}
    out["social_mean_path_week144"] = [round(float(sm[143].mean()), 3),
                                       round(float(true_rel[143]), 3)]
    c_drift = contribution_dollars(id_drift)
    roas_drift = {k: window_roas(c_drift, spend, *W[k]) for k in W}
    out["drift_roas"] = {k: {c: _summ(roas_drift[k].sel(channel=c)) for c in CHANNELS}
                         for k in W}
    d_lq = roas_drift["last_quarter"].sel(channel=soc).values.ravel()
    out["drift_social_lq_p_below"] = round(float((d_lq < BREAK_EVEN).mean()), 4)
    out["drift_social_lq_above_truth"] = float(d_lq.mean()) > tr[soc]["roas_last_quarter"]
    sp_lq = float(weekly.set_index("date_week").loc[W["last_quarter"][0]:W["last_quarter"][1],
                                                    soc].sum())
    net = sp_lq * (GROSS_MARGIN * d_lq - 1)
    out["decision"] = {
        "verdict_drift_lq": _verdict(d_lq), "verdict_static_all": _verdict(
            roas_static["all"].sel(channel=soc).values.ravel()),
        "spend_last_quarter": sp_lq, "net_return": _summ(net, 1)}
    cos_lq = m_drift.incrementality.contribution_over_spend(
        frequency="all_time", start_date=W["last_quarter"][0], end_date=W["last_quarter"][1])
    out["drift_social_lq_counterfactual"] = _summ(cos_lq.sel(channel=soc))
    out["counterfactual_formula_max_gap"] = _counterfactual_gap(id_drift, weekly, cos_lq, W)

    # Prophet and fit 3
    hol = holidays_table(weekly)
    t0 = time.time()
    comp = prophet_components(weekly, hol)
    out["seconds"]["prophet"] = round(time.time() - t0, 2)
    out["prophet_sum_gap"] = float(np.abs(comp[["trend", "season", "holiday"]].sum(axis=1)
                                          - comp["yhat"]).max())
    out["prophet_trend_per_week"] = round(float(
        (comp["trend"].iloc[-1] - comp["trend"].iloc[0]) / (len(weekly) - 1)), 1)
    out["prophet_holiday_effect"] = round(float(comp.loc[weekly["holiday"] == 1,
                                                         "holiday"].mean()), 0)
    gap = comp["trend"].to_numpy() - latent["baseline_trend"].to_numpy()
    out["trend_gap_corr_social"] = round(float(np.corrcoef(
        gap, latent[f"contribution_{soc}"])[0, 1]), 3)
    t0 = time.time()
    m_p = prophet_mmm(weekly)
    id_p = fit(m_p, prophet_inputs(weekly, comp), y, draws, seed)
    out["seconds"]["prophet_mmm"] = round(time.time() - t0, 1)
    out["divergences_prophet"] = divergences(id_p)
    roas_p = window_roas(contribution_dollars(id_p), spend, *W["all"])
    true_all = {c: tr[c]["roas"] for c in CHANNELS}
    table = baseline_comparison(roas_static["all"], roas_p, true_all)
    out["baseline_comparison"] = {f"{c}/{b}": [round(float(r["mean"]), 4),
                                               round(float(r["hdi_low"]), 4),
                                               round(float(r["hdi_high"]), 4), bool(r["covered"])]
                                  for (c, b), r in table.iterrows()}
    out["prophet_social_change"] = round(float(table.loc[(soc, "prophet"), "mean"]
                                               - table.loc[(soc, "own"), "mean"]), 4)
    out["own_closer"] = {c: bool(table.loc[(c, "own"), "abs_error"]
                                 < table.loc[(c, "prophet"), "abs_error"]) for c in CHANNELS}
    gamma = id_p.posterior["gamma_control"]
    out["prophet_gamma_control"] = {str(k): _summ(gamma.sel(control=k))
                                    for k in gamma.coords["control"].values}
    _ = tc  # the truth's window ROAS is checked by the tests, not here

    if stretch:
        # A: a dated lift test now
        t0 = time.time()
        m_lift = static_mmm(weekly, time_varying=True)
        m_lift.build_model(X, y)
        m_lift.add_lift_test_measurements(lift_tests[["channel", "x", "delta_x", "delta_y",
                                                      "sigma", "date"]])
        id_lift = fit(m_lift, X, y, draws, seed)
        out["seconds"]["stretch_a"] = round(time.time() - t0, 1)
        r = window_roas(contribution_dollars(id_lift), spend, *W["last_quarter"]).sel(channel=soc)
        out["stretch_a_social_lq"] = _summ(r) + [round(float((r.values < BREAK_EVEN).mean()), 4)]
        out["divergences_stretch_a"] = divergences(id_lift)
        # B: Prophet told about spend
        t0 = time.time()
        comp_b = prophet_components(weekly, hol, regressors=CHANNELS)
        id_b = fit(prophet_mmm(weekly), prophet_inputs(weekly, comp_b), y, draws, seed)
        out["seconds"]["stretch_b"] = round(time.time() - t0, 1)
        rb = window_roas(contribution_dollars(id_b), spend, *W["all"])
        out["stretch_b_roas_all"] = {c: _summ(rb.sel(channel=c)) for c in CHANNELS}
        out["divergences_stretch_b"] = divergences(id_b)
    return out


def _verdict(d) -> str:
    p_below = float((np.asarray(d) < BREAK_EVEN).mean())
    if p_below >= 0.9:
        return f"cut ({p_below:.3f})"
    if 1 - p_below >= 0.9:
        return f"keep ({1 - p_below:.3f})"
    return f"lift test first ({p_below:.3f})"


def counterfactual_window_roas(spend: np.ndarray, alpha, lam, beta, mult, w0: int, w1: int,
                               l_max: int = 8) -> np.ndarray:
    """pymc-marketing's window ROAS for one channel on draws: zero the spend in weeks w0..w1,
    keep carry-in, sum the change in contribution from w0 to the end of the data, divide by the
    window's spend. ``spend`` (T,) in dollars; ``alpha``, ``lam`` (D,); ``beta`` (D,) in dollars;
    ``mult`` (D, T); returns (D,)."""
    spend = np.asarray(spend, dtype=float)
    scale = spend.max()
    x = spend / scale
    x_cf = x.copy()
    x_cf[w0:w1 + 1] = 0.0
    alpha = np.asarray(alpha, dtype=float)[:, None]
    w = alpha ** np.arange(l_max)[None, :]
    w = w / w.sum(axis=1, keepdims=True)

    def adstock(v):
        out = np.zeros((alpha.shape[0], v.size))
        for lag in range(l_max):
            out[:, lag:] += w[:, lag:lag + 1] * v[None, :v.size - lag]
        return out

    lam = np.asarray(lam, dtype=float)[:, None]

    def sat(a):
        e = np.exp(-lam * a)
        return (1 - e) / (1 + e)

    diff = np.asarray(beta)[:, None] * np.asarray(mult) * (sat(adstock(x)) - sat(adstock(x_cf)))
    return diff[:, w0:].sum(axis=1) / spend[w0:w1 + 1].sum()


def _counterfactual_gap(idata, weekly, cos_lq, W) -> float:
    """Largest gap, over draws, between pymc-marketing's contribution_over_spend for social's
    last quarter and :func:`counterfactual_window_roas` (the truth file's definition)."""
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    soc = "social"
    dates = weekly["date_week"]
    w0 = int(np.flatnonzero(dates == pd.Timestamp(W["last_quarter"][0]))[0])
    w1 = int(np.flatnonzero(dates == pd.Timestamp(W["last_quarter"][1]))[0])
    ts = target_scale(idata)
    mine = counterfactual_window_roas(
        weekly[soc].to_numpy(dtype=float), post["adstock_alpha"].sel(channel=soc).values,
        post["saturation_lam"].sel(channel=soc).values,
        post["saturation_beta"].sel(channel=soc).values * ts,
        post["media_temporal_latent_multiplier"].sel(channel=soc).transpose("s", "date").values,
        w0, w1)
    theirs = cos_lq.sel(channel=soc).stack(s=("chain", "draw")).values
    return float(np.abs(mine - theirs).max())


# ------------------------------------------------------------------------------ checks
def _windows_toy_dates(n: int = 4) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-02", periods=n, freq="7D")


def check_window_roas(fn: Callable, true_contrib: xr.DataArray, spend: xr.DataArray,
                      windows: Mapping, truth: Mapping) -> dict[str, xr.DataArray]:
    """Checkpoint 1: three toys, then the truth's three windows. Returns {window: ROAS}.

    The four-week toy runs first: it tells the likely mistakes apart (a mean of weekly ratios,
    a window that loses an end, no window, the wrong spend), while the other two toys would catch
    some of them with a less specific message."""
    dates = spend.coords["date"].values
    first = windows["first_year"]
    # (1) one channel, four weeks: the likely mistakes give different numbers
    d4 = _windows_toy_dates()
    s4 = xr.DataArray([[4.0], [3.0], [2.0], [2.0]], dims=("date", "channel"),
                      coords={"date": d4, "channel": ["toy"]})
    c4 = xr.DataArray([[1.0], [1.0], [1.0], [5.0]], dims=("date", "channel"),
                      coords={"date": d4, "channel": ["toy"]})
    g = float(np.asarray(fn(c4, s4, "2023-01-09", "2023-01-23"), dtype=float).ravel()[0])
    if abs(g - 1.0) > 1e-9:
        known = {
            10 / 9: "you averaged the weekly ratios. ROAS over a window is total sales over total"
                    " spend: sum each over the window's weeks, then divide. (On the real data the"
                    " weekly ratio is infinite or undefined in tv's weeks off air, when its spend"
                    " is 0.)",
            0.4: "the window's last week is missing. .sel(date=slice(start, end)) includes both"
                 " ends; a < comparison on the end date drops it.",
            1.5: "the window's first week is missing. .sel(date=slice(start, end)) includes both"
                 " ends; a > comparison on the start date drops it.",
            8 / 11: "your function ignores start and end; select the window's weeks first.",
            7 / 11: "you divided the window's sales by spend over all weeks; select the same"
                    " weeks in spend.",
        }
        cause = next((msg for v, msg in known.items() if abs(g - v) < 1e-6),
                     "check that you select the weeks from start to end (both included) in both"
                     " arrays, sum each over \"date\", then divide.")
        raise AssertionError(
            "Checkpoint 1: a four-week toy (spend 4, 3, 2, 2; contribution 1, 1, 1, 5) over the"
            " window 2023-01-09 to 2023-01-23 must give (1 + 1 + 5) / (3 + 2 + 2) = 1.0; your"
            f" function gives {g:.4f}: " + cause + _recover(1))
    # (2) contribution = 2 x spend
    got = fn(2 * spend, spend, *first)
    if not isinstance(got, xr.DataArray):
        raise AssertionError(
            f"Checkpoint 1: window_roas returned a {type(got).__name__}, not an xarray DataArray."
            " Keep the arrays as DataArrays: select with .sel(date=slice(start, end)), sum with"
            " .sum(\"date\") and divide; xarray keeps the channel names." + _recover(1))
    vals = np.asarray(got, dtype=float)
    n_ch = spend.sizes["channel"]
    if vals.size == 1 and n_ch > 1:
        raise AssertionError(
            "Checkpoint 1: with a contribution of exactly twice the spend in every week, ROAS"
            f" over the first year must be 2 for each of the {n_ch} channels, one value per"
            f" channel; your function returned one number ({float(vals.ravel()[0]):.4g})"
            " instead of one per channel. .sum() with no argument sums over every dim, channel"
            " included; sum over \"date\" only: .sum(\"date\")." + _recover(1))
    if vals.shape != (n_ch,) or not np.allclose(vals, 2.0, rtol=1e-12):
        hint = (" You divided spend by contribution; divide the contribution by the spend."
                if vals.size and np.allclose(vals, 0.5) else
                " Divide the window's total contribution by the window's total spend, per"
                " channel.")
        raise AssertionError(
            "Checkpoint 1: with a contribution of exactly twice the spend in every week, ROAS"
            f" over the first year must be 2 for every channel; your function gives"
            f" {np.round(vals, 4).tolist()}." + hint + _recover(1))
    # (3) leading dims survive
    lead = xr.DataArray(np.ones((2, 3, len(dates), n_ch)),
                        dims=("chain", "draw", "date", "channel"),
                        coords={"date": dates, "channel": spend.coords["channel"].values})
    try:
        got = fn(lead * spend, spend, *first)
    except (TypeError, KeyError, ValueError, IndexError) as e:
        if "cannot be interpreted as an integer" in str(e):
            cause = (" .loc[start:end] selects along the first dim, which here is chain, not"
                     " date; on the toys the first dim happened to be date. Select by name:"
                     " .sel(date=slice(start, end)).")
        else:
            cause = (" Select and sum by name, .sel(date=slice(start, end)) and .sum(\"date\"),"
                     " so that leading dims such as chain and draw do not matter.")
        raise AssertionError(
            "Checkpoint 1: your function passes the toys with dims (date, channel), but on draws"
            " with dims (chain, draw, date, channel) it raised"
            f" {type(e).__name__}: {e}." + cause + _recover(1)) from None
    shape = getattr(got, "shape", ())
    if not isinstance(got, xr.DataArray) or tuple(shape) != (2, 3, n_ch):
        raise AssertionError(
            "Checkpoint 1: on draws with dims (chain, draw, date, channel) of sizes (2, 3, "
            f"{len(dates)}, {n_ch}), window_roas must return dims (chain, draw,"
            f" channel), shape (2, 3, {n_ch}); yours has shape {tuple(shape)}."
            " Sum over \"date\" by name (.sum(\"date\")), not over an axis number, so that"
            " chain and draw stay." + _recover(1))
    # (4) the truth
    keys = {"all": "roas", "first_year": "roas_first_year", "last_quarter": "roas_last_quarter"}
    out = {}
    for k, key in keys.items():
        got = fn(true_contrib, spend, *windows[k])
        out[k] = got
        for c in spend.coords["channel"].values:
            want = float(truth["channels"][str(c)][key])
            have = float(got.sel(channel=c))
            if not abs(have - want) <= 1e-9 * abs(want):
                raise AssertionError(
                    f"Checkpoint 1: your toys pass, but on the truth the {k} window"
                    f" ({windows[k][0]} to {windows[k][1]}) gives {c} a ROAS of {have:.6f};"
                    f" truth_mmm_drift.json says {want:.6f}. Check that you pass WINDOWS[...]'s"
                    " dates unchanged, as strings, and the arrays true_contrib and spend."
                    + _recover(1))
    return out


def check_static_model(window_fn: Callable, contrib: xr.DataArray, spend: xr.DataArray,
                       roas_cos: xr.DataArray, windows: Mapping, truth: Mapping,
                       channel: str = "social") -> dict[str, xr.DataArray]:
    """Labeled check "static model": (a) all weeks equals contribution_over_spend to 1e-8;
    (b) the fading channel's last-quarter 94% HDI lies above its true last-quarter ROAS."""
    roas = {k: window_fn(contrib, spend, *windows[k]) for k in windows}
    mine = roas["all"].transpose("chain", "draw", "channel")
    gap = float(np.abs(mine.values - roas_cos.transpose("chain", "draw", "channel").values).max())
    if not gap <= 1e-8:
        raise AssertionError(
            "Your window_roas passed Checkpoint 1 but differs from PyMC-Marketing's"
            f" contribution_over_spend on the posterior draws by {gap:.3g}. Check that it sums"
            " over \"date\" by name and keeps chain and draw." + _recover(1, rerun=True))
    lq = roas["last_quarter"].sel(channel=channel).values
    lo, hi = checks.interval(lq, PROB, "hdi")
    t = float(truth["channels"][channel]["roas_last_quarter"])
    if not lo > t:
        raise AssertionError(
            f"The static model's 94% HDI for {channel}'s last-quarter ROAS is [{lo:.2f}, {hi:.2f}]"
            f" and contains or lies below the true {t:.2f}; the lab expects it entirely above. This"
            " compares the fit with the truth, not your code. With QUICK on the interval is wider;"
            " if it fails in a FULL run, note the message and tell the instructor.")
    return roas


def _toy_corr_inputs():
    year = pd.Series([2023] * 4 + [2024] * 4)
    spend = pd.DataFrame({"a": [1, 2, 3, 4, 1, 2, 3, 4], "b": [4, 1, 3, 2, 2, 2, 1, 5]},
                         dtype=float)
    resid = pd.Series([1, 2, 3, 10, 4, 3, 2, 1], dtype=float)
    return resid, spend, year


def check_residual_spend_corr(fn: Callable, resid: pd.Series, spend: pd.DataFrame,
                              year: pd.Series, channel: str = "social") -> pd.DataFrame:
    """Checkpoint 2: an exact toy, the reference on the static fit, then the fading channel's
    correlation falls most from the first year to the last and ends below -0.3."""
    r_t, s_t, y_t = _toy_corr_inputs()
    want = pd.DataFrame({c: [np.corrcoef(s_t.loc[y_t == yr, c], r_t[y_t == yr])[0, 1]
                             for yr in (2023, 2024)] for c in ("a", "b")}, index=[2023, 2024])
    got = fn(r_t, s_t, y_t)
    if not isinstance(got, pd.DataFrame):
        raise AssertionError(
            f"Checkpoint 2: residual_spend_corr returned a {type(got).__name__}, not a pandas"
            " DataFrame with one row per year and one column per channel. Collect one"
            " corrwith result per year and build the DataFrame from them." + _recover(2))
    if set(map(str, got.index)) == {"a", "b"}:
        raise AssertionError(
            "Checkpoint 2: on the toy your table has the rows a and b; rows are years and columns"
            " are channels; transpose (.T)." + _recover(2))
    if len(got) == 1 or set(got.index) != {2023, 2024}:
        allweeks = s_t.corrwith(r_t)
        if len(got) == 1 and np.allclose(got.iloc[0].reindex(["a", "b"]).to_numpy(dtype=float),
                                         allweeks.to_numpy(), atol=1e-3):
            cause = (" you correlated over all weeks at once. Compute one correlation per year,"
                     " from that year's rows only.")
        else:
            cause = (" index the rows by year (2023, 2024 in the toy): select one year's rows"
                     " with year == yr.")
        raise AssertionError(
            f"Checkpoint 2: on the toy your table has the rows {list(got.index)}; it needs one"
            " row per year (2023, 2024) and one column per channel (a, b):" + cause
            + _recover(2))
    if not set(got.columns) >= {"a", "b"}:
        raise AssertionError(
            f"Checkpoint 2: on the toy your table has the columns {list(got.columns)}; it needs"
            " one column per channel (a, b)." + _recover(2))
    g = got.loc[[2023, 2024], ["a", "b"]].astype(float)
    if not np.allclose(g.to_numpy(), want.to_numpy(), atol=1e-12, rtol=0):
        a23, b23 = float(g.loc[2023, "a"]), float(g.loc[2023, "b"])
        if abs(a23 - 1.0) < 1e-9 and abs(b23 + 0.4) < 1e-9:
            cause = (" that is a rank (Spearman) correlation. Use Pearson, the default of corrwith"
                     " and of np.corrcoef.")
        elif abs(a23 - 14 / 3) < 1e-3:
            cause = (" that is a covariance. A correlation divides it by both standard"
                     " deviations; corrwith does that for you.")
        else:
            cause = (" compute, for each year, the Pearson correlation between that year's"
                     " residuals and each channel's spend (spend_part.corrwith(resid_part)).")
        raise AssertionError(
            f"Checkpoint 2: on the toy, 2023 should give a {want.loc[2023, 'a']:.4f} and b"
            f" {want.loc[2023, 'b']:.4f} (2024: a {want.loc[2024, 'a']:.4f}, b"
            f" {want.loc[2024, 'b']:.4f}); yours gives a {a23:.4f} and b {b23:.4f} in 2023:"
            + cause + _recover(2))
    table = fn(resid, spend, year)
    ref = residual_spend_corr(resid, spend, year)
    if not (isinstance(table, pd.DataFrame) and table.shape == ref.shape and np.allclose(
            table.loc[ref.index, ref.columns].to_numpy(dtype=float), ref.to_numpy(),
            atol=1e-12, rtol=0)):
        raise AssertionError(
            "Checkpoint 2: your function passes the toy but differs from the reference on the"
            " static fit's residuals. Check that it uses the arguments it is given, with rows"
            " for every year in `year` and a column for every channel in `spend`." + _recover(2))
    first, last = ref.index.min(), ref.index.max()
    change = ref.loc[last] - ref.loc[first]
    if change.idxmin() != channel or not ref.loc[last, channel] < -0.3:
        raise AssertionError(
            f"Your function is right (it matched np.corrcoef and the reference). {channel}'s"
            f" correlation goes from {ref.loc[first, channel]:+.2f} in {first} to"
            f" {ref.loc[last, channel]:+.2f} in {last}; the lab expects the largest fall of any"
            " channel, ending below -0.3. This compares the static fit with the drift built into"
            " the data, not your code. With QUICK on the numbers move; if it fails in a FULL run,"
            " note the message and tell the instructor.")
    return table


def linear_toy(dates) -> xr.DataArray:
    """m(t) = 2 - t / 155 over the weeks, one channel."""
    t = np.arange(len(dates))
    return xr.DataArray((2 - t / 155)[:, None], dims=("date", "channel"),
                        coords={"date": np.asarray(dates), "channel": ["toy"]})


def spike_toy(dates, late: tuple[str, str]) -> xr.DataArray:
    """1 in every week except 14 in the last week of ``late``, one channel: its late mean is 2,
    its late median 1, so it tells a mean from a median."""
    dates = pd.DatetimeIndex(np.asarray(dates))
    vals = np.where(dates == pd.Timestamp(late[1]), 14.0, 1.0)
    return xr.DataArray(vals[:, None], dims=("date", "channel"),
                        coords={"date": np.asarray(dates), "channel": ["toy"]})


def check_drift_ratio(fn: Callable, dates, windows: Mapping, prior_mult: xr.DataArray
                      ) -> xr.DataArray:
    """Checkpoint 3: a constant multiplier, a linear toy, a one-week spike (mean, not median),
    then the prior draws' dims."""
    early, late = windows["first_year"], windows["last_quarter"]
    ones = xr.DataArray(np.ones((len(dates), 2)), dims=("date", "channel"),
                        coords={"date": np.asarray(dates), "channel": ["a", "b"]})
    g1 = np.asarray(fn(ones, early, late), dtype=float)
    if np.isnan(g1).any():
        raise AssertionError(
            "Checkpoint 3: a constant multiplier gives NaN: a window selected no weeks; pass the"
            " dates in WINDOWS unchanged." + _recover(3))
    if not np.allclose(g1, 1.0, rtol=1e-12):
        cause = (" you used sums. The windows have 52 and 13 weeks, so their sums differ by a"
                 " factor of 4 even when nothing changes; use means." if np.allclose(g1, 0.25)
                 else " divide the mean over the late weeks by the mean over the early weeks.")
        raise AssertionError(
            "Checkpoint 3: a multiplier that is 1 in every week must give a drift ratio of 1;"
            f" yours gives {np.round(g1, 4).tolist()}:" + cause + _recover(3))
    want = float(drift_ratio(linear_toy(dates), early, late).squeeze())
    g2 = float(np.asarray(fn(linear_toy(dates), early, late), dtype=float).ravel()[0])
    if abs(g2 - want) > 1e-9:
        if abs(g2 - 1 / want) < 1e-3:
            cause = "early and late are swapped: divide the late mean by the early mean."
        elif abs(g2 - 0.56667) < 1e-4:
            cause = ("each window lost its last week. .sel(date=slice(start, end)) includes both"
                     " ends.")
        else:
            cause = ("take the mean over \"date\" of the late weeks, divided by the mean over"
                     " \"date\" of the early weeks.")
        raise AssertionError(
            "Checkpoint 3: the linear path m(t) = 2 - t/155 has a first-year mean of 1.8355 and a"
            f" last-quarter mean of 1.0387, a drift ratio of {want:.5f}; yours gives {g2:.5f}: "
            + cause + _recover(3))
    g3 = float(np.asarray(fn(spike_toy(dates, late), early, late), dtype=float).ravel()[0])
    if abs(g3 - 2.0) > 1e-9:
        cause = ("that is a ratio of medians (the late weeks' median is 1); use means:"
                 " .mean(\"date\")." if abs(g3 - 1.0) < 1e-9 else
                 "divide the mean over the late weeks by the mean over the early weeks.")
        raise AssertionError(
            "Checkpoint 3: a multiplier of 1 in every week except 14 in the last week has a"
            " last-quarter mean of (12 + 14) / 13 = 2 and a first-year mean of 1, a drift ratio"
            f" of 2.0; yours gives {g3:.5f}: " + cause + _recover(3))
    got = fn(prior_mult, early, late)
    dims = tuple(getattr(got, "dims", ()))
    if set(dims) != {"chain", "draw", "channel"}:
        raise AssertionError(
            f"Checkpoint 3: on the prior draws your result has dims {dims}; it needs (chain,"
            " draw, channel). Average over \"date\" only, so that every prior draw keeps its own"
            " ratio." + _recover(3))
    vals = np.asarray(got, dtype=float)
    if not (np.isfinite(vals).all() and (vals > 0).all()):
        raise AssertionError(
            "Checkpoint 3: some prior drift ratios are not positive and finite (NaN means a"
            " window selected no weeks); pass the dates in WINDOWS unchanged." + _recover(3))
    return got


TOY_C_TRUE = (0.01, 2.5, -0.5)  # toy (c)'s true values, in turn week by week


def _exp_quantiles(k: int) -> np.ndarray:
    """The k exact quantiles of Exponential(1) at (i + 0.5) / k, ascending."""
    return -np.log(1 - (np.arange(k) + 0.5) / k)


def coverage_toys(dates, seed: int = 2037) -> dict:
    """The four toys of Checkpoint 4 (2 chains x 500 draws x the weeks): see the lab. Toy (c)'s
    chains hold the lower and the upper half of the draws; "c_same" is toy (c) with both chains
    holding the same 500 quantiles, which tells a function that reads one chain from a wrong
    interval."""
    n = len(dates)
    coords = {"date": np.asarray(dates)}
    dims = ("chain", "draw", "date")
    line = np.linspace(1.0, 0.4, n)
    rng = np.random.default_rng(seed)
    tight = line + rng.normal(0, 0.01, (2, 500, n))
    q = _exp_quantiles(1000)
    skew = np.broadcast_to(q.reshape(2, 500)[:, :, None], (2, 500, n)).copy()
    same = np.broadcast_to(_exp_quantiles(500)[None, :, None], (2, 500, n)).copy()
    alt = np.asarray(TOY_C_TRUE)[np.arange(n) % len(TOY_C_TRUE)]
    return {
        "a": (xr.DataArray(tight, dims=dims, coords=coords), line),
        "b": (xr.DataArray(tight + 5, dims=dims, coords=coords), line),
        "c": (xr.DataArray(skew, dims=dims, coords=coords), alt),
        "c_same": (xr.DataArray(same, dims=dims, coords=coords), alt),
        "d": (xr.DataArray(tight, dims=dims, coords=coords), line[::-1].copy()),
    }


def _coverage_share(got, toy: str) -> float:
    """``got`` as a float, or a checkpoint message if it is one value per week."""
    try:
        return float(got)
    except (TypeError, ValueError):
        size = f" of {len(got)} values" if hasattr(got, "__len__") else ""
        raise AssertionError(
            f"Checkpoint 4, toy ({toy}): path_coverage returned a {type(got).__name__}{size},"
            " one per week, not their share. Return the share of weeks whose true value lies"
            " inside, a float between 0 and 1: float(np.mean(inside))." + _recover(4)) from None


def check_path_coverage(fn: Callable, mult: xr.DataArray, latent: pd.DataFrame,
                        channel: str = "social", seed: int = 2037) -> dict[str, float]:
    """Checkpoint 4: four toys, then the fit (equal to the reference; coverage bars)."""
    dates = mult.coords["date"].values
    toys = coverage_toys(dates, seed)
    n = len(dates)
    ga = _coverage_share(fn(*toys["a"]), "a")
    if abs(ga - 1.0) > 1e-12:
        if abs(ga) < 1e-12:
            cause = "you returned the share of weeks outside the band; return the share inside."
        elif abs(ga - n) < 1e-9:
            cause = "you returned a count of weeks; divide by the number of weeks."
        elif abs(ga - 100) < 1e-9:
            cause = ("you returned a percentage. A share lies between 0 and 1: return the share"
                     " of weeks inside, without multiplying by 100.")
        elif 0.8 < ga < 1:
            cause = ("you used one interval for all weeks together. Compute an interval for each"
                     " week from that week's draws only.")
        elif ga > 1:
            cause = ("a share lies between 0 and 1. For each week, compute the 94% HDI of that"
                     " week's draws and check whether the true value lies inside; return the"
                     " share of weeks where it does.")
        else:
            cause = ("for each week, compute the 94% HDI of that week's draws and check whether"
                     " the true value lies inside; return the share of weeks where it does.")
        raise AssertionError(
            "Checkpoint 4, toy (a): draws tightly around a known path (sd 0.01) must cover it in"
            f" every week (1.0); yours gives {ga:.4g}: " + cause + _recover(4))
    gb = _coverage_share(fn(*toys["b"]), "b")
    if abs(gb) > 1e-12:
        raise AssertionError(
            "Checkpoint 4, toy (b): the same draws shifted by +5 cover the path in no week (0.0);"
            f" yours gives {gb:.4g}. Compare each week's true value with that week's interval."
            + _recover(4))
    want_c = path_coverage(*toys["c"])
    gc = _coverage_share(fn(*toys["c"]), "c")
    if abs(gc - want_c) > 1e-12:
        same_ok = abs(_coverage_share(fn(*toys["c_same"]), "c") - want_c) <= 1e-12
        if same_ok:
            cause = ("your function does not use every draw of both chains (it reads one chain,"
                     " or averages the chains). In this toy chain 0 holds the lower half of the"
                     " draws and chain 1 the upper half; when both chains hold the same draws"
                     " your function is right. mult_draws.isel(date=k).values holds week k's"
                     " draws from both chains; pass them all to checks.interval, which pools"
                     " every axis.")
        elif abs(gc - 1.0) < 1e-12:
            cause = ("your interval contains -0.5, though every draw is above 0. Mean ± 1.88 sd"
                     " (a Normal approximation) is symmetric, so in skewed draws it reaches below"
                     " 0. Use checks.interval(draws, prob, \"hdi\"), built from the draws"
                     " themselves.")
        elif abs(gc - 1 / 3) < 1e-12:
            cause = ("the interval is not the 94% HDI. An equal-tailed interval (3rd to 97th"
                     " percentile, as np.percentile or np.quantile give) misses 0.01, and an"
                     " interval at a different probability, such as 89% (ArviZ's default),"
                     " misses 2.5. Use checks.interval(draws, prob, \"hdi\") and pass prob.")
        else:
            cause = "use checks.interval(draws, prob, \"hdi\") for each week's draws."
        raise AssertionError(
            "Checkpoint 4, toy (c): skewed draws (the same 1,000 exact quantiles of an"
            " Exponential in every week, all above 0) against true values 0.01, 2.5 and -0.5 in"
            " turn: the 94% HDI [0.0005, 2.805] contains 0.01 and 2.5 but not -0.5, a coverage"
            f" of 2/3 ({want_c:.4f}); yours gives {gc:.4g}: " + cause + _recover(4))
    want_d = path_coverage(*toys["d"])
    gd = _coverage_share(fn(*toys["d"]), "d")
    if abs(gd - want_d) > 1e-12:
        cause = ("you used one interval for all weeks together. Compute an interval for each"
                 " week from that week's draws only." if gd > 0.5 else
                 "compute an interval for each week from that week's draws only.")
        raise AssertionError(
            "Checkpoint 4, toy (d): toy (a)'s draws against the path reversed in time cover it"
            f" only where the two cross ({want_d:.4f}); yours gives {gd:.4f}: " + cause
            + _recover(4))
    out = {}
    for c in mult.coords["channel"].values:
        c = str(c)
        draws = mult.sel(channel=c).transpose("chain", "draw", "date")
        got = float(fn(draws, latent[f"multiplier_rel_{c}"].to_numpy()))
        ref = path_coverage(draws, latent[f"multiplier_rel_{c}"].to_numpy())
        if got != ref:
            raise AssertionError(
                f"Checkpoint 4: your toys pass, but on the fit {c}'s coverage is {got:.4f} and"
                f" the reference gives {ref:.4f}. Check that you use the prob argument and each"
                " week's draws from both chains." + _recover(4))
        out[c] = got
    low = {c: v for c, v in out.items() if (v < 0.5 if c == channel else v < 0.9)}
    if low:
        shares = ", ".join(f"{c} {v:.2f}" for c, v in out.items())
        raise AssertionError(
            f"Your function is right. Coverage: {shares};"
            f" the lab expects {channel} at least 0.5 and the other channels at least 0.9. This"
            " compares the fit with the truth, not your code. With QUICK on, the band is rougher;"
            " if it fails in a FULL run, note the message and tell the instructor.")
    return out


def check_drifting_model(ratio: xr.DataArray, truth: Mapping, channel: str = "social"
                         ) -> pd.DataFrame:
    """Labeled check "drifting model": the fading channel's drift-ratio 94% HDI lies below 1 and
    the other channels' HDIs contain 1. Returns the table it prints."""
    rows = {}
    for c in ratio.coords["channel"].values:
        c = str(c)
        d = ratio.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, PROB, "hdi")
        rows[c] = {"mean": d.mean(), "hdi_low": lo, "hdi_high": hi,
                   "true_ratio": float(truth["channels"][c]["drift_ratio"])}
    table = pd.DataFrame(rows).T.rename_axis("channel")
    bad = []
    if not table.loc[channel, "hdi_high"] < 1:
        bad.append(f"{channel}'s HDI [{table.loc[channel, 'hdi_low']:.2f},"
                   f" {table.loc[channel, 'hdi_high']:.2f}] does not lie entirely below 1")
    for c in table.index:
        if c != channel and not table.loc[c, "hdi_low"] <= 1 <= table.loc[c, "hdi_high"]:
            bad.append(f"{c}'s HDI [{table.loc[c, 'hdi_low']:.2f},"
                       f" {table.loc[c, 'hdi_high']:.2f}] does not contain 1")
    if bad:
        raise AssertionError(
            "Drift ratios (94% HDI): " + "; ".join(bad) + ". This compares the fit with the"
            " truth, not your code. If it fails in a FULL run, note the message and tell the"
            " instructor.")
    return table


def check_prophet(components: pd.DataFrame, weekly: pd.DataFrame) -> None:
    """Labeled check "prophet" (provided code): columns, rows, finite; holiday zero outside the
    holiday weeks; the three components add up to Prophet's yhat."""
    msg = " This checks the provided code, not yours. If it fails, note the message and tell the"
    msg += " instructor."
    need = ["trend", "season", "holiday"]
    if not set(need) <= set(components.columns) or len(components) != len(weekly):
        raise AssertionError(
            f"prophet_components returned columns {list(components.columns)} and"
            f" {len(components)} rows; it should return {need} (and yhat) for {len(weekly)}"
            " weeks." + msg)
    if not np.isfinite(components[need].to_numpy(dtype=float)).all():
        raise AssertionError("Prophet's components contain NaN or infinite values." + msg)
    off = components.loc[weekly["holiday"].to_numpy() == 0, "holiday"].abs().max()
    if off > 1e-9:
        raise AssertionError(
            f"Prophet's holiday component is {off:.3g} in a week that is not a holiday week; it"
            " should be 0 there." + msg)
    gap = float(np.abs(components[need].sum(axis=1) - components["yhat"]).max())
    if gap > 1e-6:
        raise AssertionError(
            f"Prophet's trend, season and holiday add up to its yhat only to within ${gap:,.2f}"
            " (expected to 1e-6): another component is in the model." + msg)


def comparison_toys() -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Checkpoint 5's toy: 1,000 exact lognormal quantiles per channel (dims (2, 500))."""
    from scipy import stats

    z = stats.norm.ppf((np.arange(1000) + 0.5) / 1000)

    def arr(median):
        d = (median * np.exp(0.3 * z)).reshape(2, 500)
        return xr.DataArray(np.stack([d, d], axis=-1), dims=("chain", "draw", "channel"),
                            coords={"channel": ["a", "b"]})

    return arr(2.0), arr(3.0), {"a": 2.5, "b": 1.10}


def check_baseline_comparison(fn: Callable, roas_own: xr.DataArray, roas_prophet: xr.DataArray,
                              true_roas: Mapping, channel: str = "social") -> pd.DataFrame:
    """Checkpoint 5: the toy, then on the fits: the Prophet baseline lowers the fading channel's
    all-period ROAS, its 94% HDI excludes the truth and the own-baseline HDI covers it."""
    own, pro, t = comparison_toys()
    want = baseline_comparison(own, pro, t)
    got = fn(own, pro, t)
    cols = ["mean", "hdi_low", "hdi_high", "true_roas", "abs_error", "covered"]
    ok_shape = (isinstance(got, pd.DataFrame) and list(got.index.names) == ["channel", "baseline"]
                and set(cols) <= set(got.columns))
    if not ok_shape:
        idx = list(getattr(getattr(got, "index", None), "names", []) or [])
        have = list(getattr(got, "columns", []))
        raise AssertionError(
            f"Checkpoint 5: on the toy your table has index {idx} and columns {have}; it needs"
            " one row per channel and baseline, index (channel, baseline), with the six columns"
            f" {cols}. Build a list of dicts and finish with .set_index([\"channel\","
            " \"baseline\"])." + _recover(5))
    if set(got.index) != set(want.index):
        bases = sorted({str(b) for b in got.index.get_level_values("baseline")})
        chans = sorted({str(c) for c in got.index.get_level_values("channel")})
        if bases != ["own", "prophet"]:
            what = (f"the baseline labels in your index are {bases}; they must be exactly"
                    " \"own\" and \"prophet\", in lowercase: the lab looks rows up by these"
                    " labels.")
        elif chans != ["a", "b"]:
            what = (f"the channels in your index are {chans}; the toy's are ['a', 'b']: take each"
                    " row's channel from draws.coords[\"channel\"].values, as a string.")
        else:
            what = "it needs one row for each channel and baseline, each exactly once."
        raise AssertionError(
            "Checkpoint 5: your table has the right index names and columns, but on the toy "
            + what + _recover(5))
    g = got.loc[want.index]
    if abs(float(g.loc[("a", "own"), "mean"]) - 3.1379) < 1e-3:
        raise AssertionError(
            "Checkpoint 5: on the toy your \"own\" rows hold means near 3.14, the Prophet draws':"
            " the two baselines are swapped; check the order of the arguments." + _recover(5))
    if abs(float(g.loc[("a", "own"), "abs_error"]) + 0.4081) < 1e-3:
        raise AssertionError(
            "Checkpoint 5: on the toy own a's abs_error is -0.408: that is a signed error (mean"
            " minus truth); take its absolute value." + _recover(5))
    lo_b, hi_b = float(g.loc[("b", "own"), "hdi_low"]), float(g.loc[("b", "own"), "hdi_high"])
    w_lo, w_hi = float(want.loc[("b", "own"), "hdi_low"]), float(want.loc[("b", "own"), "hdi_high"])
    tol = 0.02 * (w_hi - w_lo)
    if abs(lo_b - w_lo) > tol or abs(hi_b - w_hi) > tol:
        eti = checks.interval(own.sel(channel="b").values, PROB, "eti")
        if abs(lo_b - eti[0]) < 1e-3 and abs(hi_b - eti[1]) < 1e-3:
            what = (" That is the equal-tailed interval (3rd to 97th percentile): in these skewed"
                    " draws it starts at 1.14 and misses 1.10, while the HDI reaches down to"
                    " 1.02. Use checks.interval(d, prob, \"hdi\").")
        else:
            what = (" That is the equal-tailed interval, or an interval at a different probability"
                    " (89% is ArviZ's default). Use checks.interval(d, prob, \"hdi\") and pass"
                    " prob=0.94, the function's default.")
        raise AssertionError(
            f"Checkpoint 5: on the toy own b's interval is [{lo_b:.4f}, {hi_b:.4f}]; the 94% HDI"
            f" is [{w_lo:.4f}, {w_hi:.4f}] and covers the true 1.10." + what + _recover(5))
    if not bool(g.loc[("b", "own"), "covered"]):
        raise AssertionError(
            f"Checkpoint 5: on the toy your interval for own b is right, [{lo_b:.4f},"
            f" {hi_b:.4f}], but covered is False although the true 1.10 lies inside it. covered"
            " is hdi_low <= true_roas <= hdi_high, the same comparison in every row." + _recover(5))
    hints = {
        "mean": "the mean of the row's draws, d.mean()",
        "hdi_low": "the lower end of the 94% HDI from checks.interval(d, prob, \"hdi\")",
        "hdi_high": "the upper end of the 94% HDI from checks.interval(d, prob, \"hdi\")",
        "true_roas": "true_roas[channel] for the row's channel",
        "abs_error": "the absolute error of the mean, abs(mean - true_roas)",
        "covered": "hdi_low <= true_roas <= hdi_high",
    }

    def show(c, v):
        return str(bool(v)) if c == "covered" else f"{float(v):.4f}"

    for key in want.index:
        w, h = want.loc[key], g.loc[key]
        width = w["hdi_high"] - w["hdi_low"]
        tols = {"mean": 1e-9, "hdi_low": 0.02 * width, "hdi_high": 0.02 * width,
                "true_roas": 1e-12, "abs_error": 1e-9}
        diff = [c for c in cols if (bool(h[c]) != bool(w[c]) if c == "covered"
                                    else not abs(float(h[c]) - w[c]) <= tols[c])]
        if diff:
            shown = "; ".join(f"{c} {show(c, h[c])}, the reference {show(c, w[c])}"
                              for c in diff)
            first = diff[0]
            extra = ""
            draws = (own if key[1] == "own" else pro).sel(channel=key[0]).values.ravel()
            if first == "mean" and abs(float(h["mean"]) - float(np.median(draws))) < 1e-9:
                extra = " Yours is the median of the draws."
            elif first == "abs_error" and abs(
                    float(h["abs_error"]) - w["abs_error"] / w["true_roas"]) < 1e-9:
                extra = " Yours is divided by the truth (a relative error)."
            raise AssertionError(
                f"Checkpoint 5: on the toy, row ({key[0]}, {key[1]}) differs from the reference"
                f" in {', '.join(diff)}: {shown}. {first} is {hints[first]}." + extra
                + _recover(5))
    table = fn(roas_own, roas_prophet, true_roas)
    own_r, pro_r = table.loc[(channel, "own")], table.loc[(channel, "prophet")]
    t_c = float(true_roas[channel])
    fails = []
    if not pro_r["mean"] < own_r["mean"]:
        fails.append("the Prophet baseline did not lower its mean")
    if pro_r["hdi_low"] <= t_c <= pro_r["hdi_high"]:
        fails.append("the Prophet-baseline HDI covers the truth")
    if not own_r["hdi_low"] <= t_c <= own_r["hdi_high"]:
        fails.append("the own-baseline HDI does not cover the truth")
    if fails:
        raise AssertionError(
            f"Your function is right. For {channel} (true all-period ROAS {t_c:.2f}): own"
            f" {own_r['mean']:.2f} [{own_r['hdi_low']:.2f}, {own_r['hdi_high']:.2f}], Prophet"
            f" {pro_r['mean']:.2f} [{pro_r['hdi_low']:.2f}, {pro_r['hdi_high']:.2f}]; "
            + "; ".join(fails) + ". This compares the fits with the truth, not your code. If it"
            " fails in a FULL run, note the message and tell the instructor.")
    return table


__all__ = [
    "BREAK_EVEN", "CHANNELS", "FIT_SETTINGS", "GROSS_MARGIN", "HSGP_SETTINGS",
    "PROPHET_CONTROLS", "TOY_C_TRUE", "baseline_comparison", "beta_sigma",
    "check_baseline_comparison", "check_drift_ratio", "check_drifting_model",
    "check_path_coverage", "check_prophet", "check_residual_spend_corr", "check_static_model",
    "check_window_roas", "comparison_toys", "contribution_dollars", "counterfactual_window_roas",
    "coverage_toys", "divergences", "drift_ratio", "fit", "holidays_table", "linear_toy",
    "multiplier_prior", "path_coverage", "prophet_components", "prophet_inputs", "prophet_mmm",
    "quiet_prophet", "reference_results", "residual_spend_corr", "spend_array", "spike_toy",
    "static_mmm", "target_scale", "true_contrib_array", "window_roas", "windows_of",
]
