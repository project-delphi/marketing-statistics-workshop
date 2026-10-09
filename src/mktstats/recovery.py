"""Fit a model to synthetic data and compare the estimates with the truth.

Thin helpers used by the tests and the labs. pymc-marketing is imported inside the functions, so
importing ``mktstats.recovery`` stays cheap. Estimates come back as plain dicts keyed like the
truth in ``data/synthetic/truth.json``, so :func:`compare` can line them up.

Priors: BG/NBD and Pareto/NBD fits use pymc-marketing 1.2.0's default priors unless
``model_config`` is given. The Gamma-Gamma fit defaults to weakly informative priors
(``WEAK_GAMMA_GAMMA_PRIORS``) because the package defaults (``p, q ~ Weibull(2, 1)``,
``v ~ Weibull(2, 10)``) pull ``p`` far below a true value of 6 in a MAP fit with ~2,700 repeat
customers (measured over 30 seeds: mean MAP p = 4.55 with the defaults, 5.79 with the weak
priors; truth 6). The population mean spend ``p v / (q - 1)`` is recovered either way.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

WEAK_GAMMA_GAMMA_PRIORS = {
    "p": ("HalfNormal", {"sigma": 10}),
    "q": ("HalfNormal", {"sigma": 10}),
    "v": ("HalfNormal", {"sigma": 100}),
}


def _priors(spec: Mapping[str, tuple[str, dict]]):
    from pymc_extras.prior import Prior

    return {k: Prior(dist, **kw) for k, (dist, kw) in spec.items()}


def _scalars(summary: pd.Series) -> dict[str, float]:
    """Drop per-customer deterministics such as ``alpha[17]`` from a MAP fit summary."""
    out = {}
    for k, v in summary.items():
        if "[" in k and not (k.startswith("purchase_coefficient") or
                             k.startswith("dropout_coefficient")):
            continue
        out[k] = float(v)
    return out


def fit_bgnbd_map(rfm: pd.DataFrame, model_config: dict | None = None):
    """MAP fit of pymc-marketing's ``BetaGeoModel``; returns ``(model, {r, alpha, a, b})``."""
    from pymc_marketing import clv

    model = clv.BetaGeoModel(model_config=model_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=rfm[["customer_id", "frequency", "recency", "T"]], method="map",
                  progressbar=False)
    s = _scalars(model.fit_summary())
    return model, {k: s[k] for k in ("r", "alpha", "a", "b")}


def fit_pnbd_map(rfm: pd.DataFrame, covariate_cols: list[str] | None = None,
                 model_config: dict | None = None):
    """MAP fit of pymc-marketing's ``ParetoNBDModel``, optionally with the same static covariates
    on purchase and dropout. Returns ``(model, estimates)`` with keys r, alpha, s, beta and, with
    covariates, ``gamma_purchase`` / ``gamma_dropout`` dicts keyed by covariate column."""
    from pymc_marketing import clv

    cfg = dict(model_config or {})
    cols = list(covariate_cols or [])
    if cols:
        cfg.setdefault("purchase_covariate_cols", cols)
        cfg.setdefault("dropout_covariate_cols", cols)
    model = clv.ParetoNBDModel(model_config=cfg or None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=rfm[["customer_id", "frequency", "recency", "T", *cols]], method="map",
                  progressbar=False)
    s = _scalars(model.fit_summary())
    if not cols:
        return model, {k: s[k] for k in ("r", "alpha", "s", "beta")}
    est = {"r": s["r"], "alpha": s["alpha_scale"], "s": s["s"], "beta": s["beta_scale"],
           "gamma_purchase": {c: s[f"purchase_coefficient[{c}]"] for c in cols},
           "gamma_dropout": {c: s[f"dropout_coefficient[{c}]"] for c in cols}}
    return model, est


def fit_gamma_gamma_map(rfm: pd.DataFrame, model_config: dict | str | None = "weak"):
    """MAP fit of ``GammaGammaModel`` on customers with ``frequency > 0``.

    ``model_config="weak"`` (default) uses ``WEAK_GAMMA_GAMMA_PRIORS``; ``None`` uses the package
    defaults. Returns ``(model, {"p", "q", "v"})``.
    """
    from pymc_marketing import clv

    cfg = _priors(WEAK_GAMMA_GAMMA_PRIORS) if model_config == "weak" else model_config
    data = rfm.loc[rfm["frequency"] > 0, ["customer_id", "frequency", "monetary_value"]]
    model = clv.GammaGammaModel(model_config=cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=data, method="map", progressbar=False)
    s = _scalars(model.fit_summary())
    return model, {k: s[k] for k in ("p", "q", "v")}


def _flatten(d: Mapping, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, int | float | np.floating | np.integer) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def compare(truth: Mapping, estimate: Mapping, tol: Mapping | float = 0.2) -> pd.DataFrame:
    """Tidy truth-versus-estimate table, one row per parameter present in both.

    Nested dicts are flattened with dots (``gamma_purchase.social``). ``tol`` is either a relative
    tolerance for every parameter, or a dict mapping a parameter to ``{"rel": x}`` or
    ``{"abs": y}`` (parameters without an entry get no tolerance and ``within`` = NaN).
    Columns: parameter, truth, estimate, abs_error, rel_error, tolerance, tolerance_kind, within.
    """
    t = _flatten(truth)
    e = _flatten(estimate)
    tols = _flatten(tol) if isinstance(tol, Mapping) else {}
    rows = []
    for name in [k for k in e if k in t]:
        tv, ev = t[name], e[name]
        abs_err = abs(ev - tv)
        rel_err = abs_err / abs(tv) if tv != 0 else math.inf
        kind, bound = None, math.nan
        if isinstance(tol, Mapping):
            if f"{name}.rel" in tols:
                kind, bound = "rel", tols[f"{name}.rel"]
            elif f"{name}.abs" in tols:
                kind, bound = "abs", tols[f"{name}.abs"]
        else:
            kind, bound = "rel", float(tol)
        within = (rel_err <= bound if kind == "rel" else abs_err <= bound) if kind else math.nan
        rows.append({"parameter": name, "truth": tv, "estimate": ev, "abs_error": abs_err,
                     "rel_error": rel_err, "tolerance": bound, "tolerance_kind": kind,
                     "within": within})
    return pd.DataFrame(rows, columns=["parameter", "truth", "estimate", "abs_error", "rel_error",
                                       "tolerance", "tolerance_kind", "within"])


def did_log_lift(panel: pd.DataFrame, outcome: str = "sales", treated: str = "treated",
                 post: str = "post") -> dict[str, float]:
    """Difference-in-differences on log outcome for a balanced panel with one treatment start.

    With unit and period fixed effects and a single adoption date, the two-way fixed-effects
    estimate equals the 2x2 difference of group means of ``log(outcome)``. Returns ``log_lift``,
    ``lift_pct`` (``100 * (exp(log_lift) - 1)``) and ``incremental_sales`` (treated post-period
    sales times ``1 - 1 / (1 + lift)``).
    """
    ly = np.log(panel[outcome].to_numpy(dtype=float))
    g = pd.DataFrame({"ly": ly, "tr": panel[treated].to_numpy(), "po": panel[post].to_numpy()})
    m = g.groupby(["tr", "po"])["ly"].mean()
    log_lift = float((m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0]))
    lift = math.expm1(log_lift)
    window = panel.loc[(panel[treated] == 1) & (panel[post] == 1), outcome].sum()
    return {"log_lift": log_lift, "lift_pct": 100 * lift,
            "incremental_sales": float(window * (1 - 1 / (1 + lift)))}


def hdi(draws, prob: float = 0.94) -> tuple[float, float]:
    """Highest-density interval: the shortest interval holding ``prob`` of the draws."""
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    n = len(x)
    k = int(math.ceil(prob * n))
    if not 0 < prob < 1 or k < 1 or k > n:
        raise ValueError("need 0 < prob < 1 and enough draws")
    widths = x[k - 1:] - x[: n - k + 1]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k - 1])


MMM_CHANNELS = ["tv", "search", "social", "display"]


def fit_mmm(weekly: pd.DataFrame, *, draws: int = 500, tune: int = 500, chains: int = 2,
            l_max: int = 8, yearly_seasonality: int = 2, nuts_sampler: str = "nutpie",
            random_seed: int = 1, lift_tests: pd.DataFrame | None = None, **fit_kwargs):
    """Fit pymc-marketing's ``MMM`` to the synthetic weekly table with the generator's functional
    form: ``GeometricAdstock(l_max)``, ``LogisticSaturation()``, controls price_index, holiday and
    t, ``yearly_seasonality`` Fourier orders, default priors and default max scaling."""
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    model = MMM(date_column="date_week", channel_columns=MMM_CHANNELS, target_column="y",
                control_columns=["price_index", "holiday", "t"],
                adstock=GeometricAdstock(l_max=l_max), saturation=LogisticSaturation(),
                yearly_seasonality=yearly_seasonality)
    x = weekly.drop(columns=["y"])
    y = weekly["y"]
    if lift_tests is not None:
        model.build_model(x, y)
        model.add_lift_test_measurements(lift_tests[["channel", "x", "delta_x", "delta_y",
                                                     "sigma"]])
    model.fit(x, y, draws=draws, tune=tune, chains=chains, nuts_sampler=nuts_sampler,
              random_seed=random_seed, progressbar=False, **fit_kwargs)
    return model


def mmm_draws(model, weekly: pd.DataFrame) -> dict[str, np.ndarray]:
    """Posterior draws, flattened over chains, of per-channel ROAS (sum of contribution in sales
    units over all weeks / total spend), ``adstock_alpha``, ``saturation_lam`` and
    ``saturation_beta`` (model units). Arrays have shape (n_draws, n_channels)."""
    post = model.idata.posterior
    target_scale = float(np.asarray(model.idata.constant_data["target_scale"]).squeeze())
    contrib = post["channel_contribution"].transpose("chain", "draw", "date", "channel")
    contrib = np.asarray(contrib).reshape(-1, contrib.shape[2], contrib.shape[3]) * target_scale
    channels = [str(c) for c in post["channel_contribution"].coords["channel"].values]
    spend = weekly[channels].to_numpy(dtype=float).sum(axis=0)
    out = {"channels": np.array(channels), "roas": contrib.sum(axis=1) / spend}
    for name in ("adstock_alpha", "saturation_lam", "saturation_beta"):
        da = post[name].transpose("chain", "draw", "channel")
        out[name] = np.asarray(da).reshape(-1, da.shape[2])
    return out


def mmm_recovery_table(draws: dict[str, np.ndarray], truth: Mapping, prob: float = 0.94):
    """One row per channel and quantity: truth, posterior mean, the ``prob`` HDI, inside or not.

    ``truth`` is ``truth.json["mmm"]``. Compares roas, adstock_alpha, saturation_lam and
    saturation_beta (against ``saturation_beta_model_units``)."""
    rows = []
    keys = {"roas": "roas", "adstock_alpha": "adstock_alpha", "saturation_lam": "saturation_lam",
            "saturation_beta": "saturation_beta_model_units"}
    for j, ch in enumerate(draws["channels"]):
        for q, tkey in keys.items():
            d = draws[q][:, j]
            lo, hi = hdi(d, prob)
            tv = float(truth["channels"][str(ch)][tkey])
            rows.append({"channel": str(ch), "quantity": q, "truth": tv, "mean": float(d.mean()),
                         "hdi_low": lo, "hdi_high": hi, "interval": f"{prob:.0%} HDI",
                         "inside": lo <= tv <= hi})
    return pd.DataFrame(rows)


__all__ = [
    "WEAK_GAMMA_GAMMA_PRIORS",
    "compare",
    "did_log_lift",
    "fit_bgnbd_map",
    "fit_gamma_gamma_map",
    "fit_mmm",
    "fit_pnbd_map",
    "hdi",
    "mmm_draws",
    "mmm_recovery_table",
]
