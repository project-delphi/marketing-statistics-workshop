"""Maximum-likelihood fits of the BG/NBD and Gamma-Gamma models, without building a PyMC model.

Module 13 fits the same two models to many versions of one transaction log (one per cleaning
choice), so a fit must take a fraction of a second. These functions maximize the closed-form
log-likelihoods that pymc-marketing 1.2.0 writes, with SciPy, and return plain dicts. A
maximum-likelihood fit is what pymc-marketing's MAP gives with flat priors (``Prior("HalfFlat")``
for every parameter), which Modules 2 and 4 used to reproduce Fader, Hardie & Lee's published
CDNOW estimates; ``tests/test_mle.py`` checks both agreements. Each fit is a point estimate: no
interval.

Likelihoods, transcribed from the installed pymc-marketing 1.2.0 source (checked 2026-10-10):

* BG/NBD, ``clv/distributions.py`` ``BetaGeoNBD.logp`` (Fader, Hardie & Lee 2005; note 004)::

      LL = D1 + D2 + log(C3 + [x > 0] C4)
      D1 = lgamma(r + x) - lgamma(r) + lgamma(a + b) + lgamma(b + x) - lgamma(b) - lgamma(a + b + x)
      D2 = r log(alpha) - (r + x) log(alpha + t_x)
      C3 = ((alpha + t_x) / (alpha + T)) ** (r + x),   C4 = a / (b + x - 1)

  with x = ``frequency``, t_x = ``recency``, T = ``T``; lambda ~ Gamma(r, rate alpha), dropout
  probability ~ Beta(a, b).
* Gamma-Gamma, ``clv/models/gamma_gamma.py`` ``GammaGammaModel.build_model`` (the potential
  "Eq 1a" of Fader & Hardie, note 025), for repeat buyers (x > 0) with mean repeat spend z::

      LL = lgamma(p x + q) - lgamma(p x) - lgamma(q) + q log(v) + (p x - 1) log(z)
           + p x log(x) - (p x + q) log(x z + v)

Predictions, from the same source:

* ``BetaGeoModel.expected_purchases`` (FHL 2005 eq. 10): expected repeat purchases in the next
  ``t`` time units of a customer with history (x, t_x, T);
* ``GammaGammaModel.expected_customer_spend``: (1 - w) p v / (q - 1) + w z with
  w = p x / (p x + q - 1), the customer's mean shrunk toward the population mean p v / (q - 1).

The optimizer works on log parameters: Nelder-Mead from a fixed start, then L-BFGS-B with the
analytic gradient, then Newton steps with a finite-difference Hessian of that gradient, so the
estimate is accurate to many more digits than any check needs (and the same on every machine).
A fit that does not converge, or runs off to a parameter of 0 or infinity, raises
:class:`MLEError` rather than returning a number.

>>> from mktstats import data, mle
>>> rfm = data.cdnow_rfm(observation_period_end="1997-09-30")
>>> bg = mle.bgnbd_mle(rfm)                     # r 0.2426, alpha 4.414, a 0.793, b 2.426
>>> gg = mle.gamma_gamma_mle(rfm[rfm["frequency"] > 0])    # p 6.25, q 3.74, v 15.44
>>> mle.bgnbd_expected_purchases(bg, rfm, t=39)  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, hyp2f1

BGNBD_PARAMS = ("r", "alpha", "a", "b")
GAMMA_GAMMA_PARAMS = ("p", "q", "v")

# Start values (natural scale). Nelder-Mead moves far from them; they only need to be finite.
BGNBD_START = {"r": 0.5, "alpha": 5.0, "a": 0.5, "b": 2.0}
GAMMA_GAMMA_START = {"p": 2.0, "q": 3.0, "v": 10.0}
LOG_BOUND = 25.0  # |log parameter| beyond this (about 7e10 or 1e-11) counts as running off


class MLEError(RuntimeError):
    """The likelihood has no maximum the optimizer could find at finite parameter values."""


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def _column(rfm, name: str, what: str) -> np.ndarray:
    if name not in getattr(rfm, "columns", ()):
        raise ValueError(f"{what} needs a column {name!r}; the table has {list(rfm.columns)}")
    arr = np.asarray(rfm[name], dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{what}: column {name!r} has missing or infinite values")
    return arr


def _bgnbd_data(rfm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _column(rfm, "frequency", "bgnbd_mle")
    t_x = _column(rfm, "recency", "bgnbd_mle")
    T = _column(rfm, "T", "bgnbd_mle")
    if len(x) == 0:
        raise ValueError("bgnbd_mle: the RFM table has no rows")
    if (x < 0).any() or (t_x < 0).any() or (t_x > T + 1e-12).any():
        raise ValueError("bgnbd_mle: need frequency >= 0 and 0 <= recency <= T for every customer")
    return x, t_x, T


def _gamma_gamma_data(rfm) -> tuple[np.ndarray, np.ndarray]:
    x = _column(rfm, "frequency", "gamma_gamma_mle")
    z = _column(rfm, "monetary_value", "gamma_gamma_mle")
    if len(x) == 0:
        raise ValueError("gamma_gamma_mle: the RFM table has no rows")
    if (x <= 0).any():
        raise ValueError(
            f"gamma_gamma_mle: {int((x <= 0).sum()):,} customers have frequency 0. The spend model"
            " uses repeat buyers only: pass rfm[rfm['frequency'] > 0]."
        )
    if (z <= 0).any():
        raise ValueError(
            f"gamma_gamma_mle: {int((z <= 0).sum()):,} repeat buyers have monetary_value <= 0;"
            " the spend model needs a positive mean spend (check the cleaning: returns and"
            " cancellations must not make a purchase day's amount zero or negative)."
        )
    return x, z


# ---------------------------------------------------------------------------------------------
# log-likelihoods and their gradients (in the natural parameters)
# ---------------------------------------------------------------------------------------------
def bgnbd_loglik(params: Mapping[str, float], rfm) -> np.ndarray:
    """Per-customer BG/NBD log-likelihood at ``params`` (keys r, alpha, a, b)."""
    x, t_x, T = _bgnbd_data(rfm)
    return _bgnbd_ll(np.array([params[k] for k in BGNBD_PARAMS], float), x, t_x, T)[0]


def _bgnbd_ll(theta, x, t_x, T, grad: bool = False):
    r, alpha, a, b = theta
    pos = x > 0
    d1 = (gammaln(r + x) - gammaln(r) + gammaln(a + b) + gammaln(b + x) - gammaln(b)
          - gammaln(a + b + x))
    d2 = r * np.log(alpha) - (r + x) * np.log(alpha + t_x)
    log_c3 = (r + x) * (np.log(alpha + t_x) - np.log(alpha + T))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_c4 = np.where(pos, np.log(a) - np.log(np.where(pos, b + x - 1, 1.0)), -np.inf)
    log_sum = np.logaddexp(log_c3, log_c4)
    ll = d1 + d2 + log_sum
    if not grad:
        return ll, None
    w3 = np.exp(log_c3 - log_sum)  # C3 / (C3 + C4): 1 for x = 0
    w4 = 1.0 - w3
    g_r = (digamma(r + x) - digamma(r) + np.log(alpha) - np.log(alpha + t_x)
           + w3 * (np.log(alpha + t_x) - np.log(alpha + T)))
    g_alpha = (r / alpha - (r + x) / (alpha + t_x)
               + w3 * (r + x) * (1.0 / (alpha + t_x) - 1.0 / (alpha + T)))
    g_a = digamma(a + b) - digamma(a + b + x) + w4 / a
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(pos, 1.0 / np.where(pos, b + x - 1, 1.0), 0.0)
    g_b = digamma(a + b) + digamma(b + x) - digamma(b) - digamma(a + b + x) - w4 * inv
    return ll, np.vstack([g_r, g_alpha, g_a, g_b])


def gamma_gamma_loglik(params: Mapping[str, float], rfm) -> np.ndarray:
    """Per-customer Gamma-Gamma log-likelihood at ``params`` (keys p, q, v); repeat buyers only."""
    x, z = _gamma_gamma_data(rfm)
    return _gg_ll(np.array([params[k] for k in GAMMA_GAMMA_PARAMS], float), x, z)[0]


def _gg_ll(theta, x, z, grad: bool = False):
    p, q, v = theta
    px = p * x
    ll = (gammaln(px + q) - gammaln(px) - gammaln(q) + q * np.log(v) + (px - 1) * np.log(z)
          + px * np.log(x) - (px + q) * np.log(x * z + v))
    if not grad:
        return ll, None
    g_p = x * (digamma(px + q) - digamma(px) + np.log(z) + np.log(x) - np.log(x * z + v))
    g_q = digamma(px + q) - digamma(q) + np.log(v) - np.log(x * z + v)
    g_v = q / v - (px + q) / (x * z + v)
    return ll, np.vstack([g_p, g_q, g_v])


# ---------------------------------------------------------------------------------------------
# the optimizer
# ---------------------------------------------------------------------------------------------
def _maximize(ll_fn: Callable, names: tuple[str, ...], start: Mapping[str, float],
              model: str) -> dict[str, float]:
    """Maximize the mean log-likelihood over log parameters; see the module docstring."""

    def mean_ll(log_theta, grad=False):
        theta = np.exp(log_theta)
        ll, g = ll_fn(theta, grad)
        value = float(np.mean(ll))
        if not grad:
            return value, None
        return value, np.mean(g, axis=1) * theta  # chain rule: d/dlog(theta) = theta d/dtheta

    def objective(log_theta):
        if np.any(np.abs(log_theta) > 2 * LOG_BOUND):
            return np.inf
        value, _ = mean_ll(log_theta)
        return -value if np.isfinite(value) else np.inf

    def objective_grad(log_theta):
        value, g = mean_ll(log_theta, grad=True)
        if not np.isfinite(value) or not np.all(np.isfinite(g)):
            return np.inf, np.zeros_like(log_theta)
        return -value, -g

    x0 = np.log([start[k] for k in names])
    with np.errstate(all="ignore"):
        nm = minimize(objective, x0, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 20_000, "maxfev": 40_000})
        lb = minimize(objective_grad, nm.x, jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5_000})
        theta = lb.x if np.isfinite(lb.fun) and lb.fun <= nm.fun else nm.x
        # Newton polish with a central-difference Hessian of the analytic gradient.
        hess = None
        for _ in range(50):
            _, g = mean_ll(theta, grad=True)
            hess = _hessian(lambda th: mean_ll(th, grad=True)[1], theta)
            try:
                step = np.linalg.solve(hess, -g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            # backtrack only if the step lowers the likelihood by more than rounding: near the
            # optimum the function cannot resolve the step, the gradient can
            base = mean_ll(theta)[0]
            noise = 1e-12 * max(1.0, abs(base))
            for _ in range(30):
                if mean_ll(theta + step)[0] >= base - noise:
                    break
                step = step / 2
            else:
                break  # no step along this direction keeps the likelihood: keep theta
            theta = theta + step
            if np.max(np.abs(step)) < 1e-12:
                break
        value, g = mean_ll(theta, grad=True)
    est = {k: float(np.exp(t)) for k, t in zip(names, theta, strict=True)}
    shown = ", ".join(f"{k} = {v:.4g}" for k, v in est.items())
    if np.any(np.abs(theta) > LOG_BOUND) or not np.isfinite(value):
        raise MLEError(
            f"The {model} likelihood has no maximum at finite parameter values ({shown}). The"
            " RFM table does not look like data this model describes: check how purchases were"
            " counted (purchase days, not invoices or lines) and the units of recency and T."
        )
    eig = np.linalg.eigvalsh(0.5 * (hess + hess.T)) if hess is not None else np.array([np.nan])
    if not np.max(np.abs(g)) < 1e-6 or not np.all(eig < 0):
        raise MLEError(
            f"The {model} fit did not converge (largest gradient {np.max(np.abs(g)):.2g} at"
            f" {shown}). Check the RFM table (frequency, recency and T in the same time unit)."
        )
    return est


def _hessian(grad_fn: Callable, theta: np.ndarray, h: float = 1e-5) -> np.ndarray:
    k = len(theta)
    out = np.empty((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h
        out[:, j] = (grad_fn(theta + e) - grad_fn(theta - e)) / (2 * h)
    return 0.5 * (out + out.T)


# ---------------------------------------------------------------------------------------------
# public fits and predictions
# ---------------------------------------------------------------------------------------------
def bgnbd_mle(rfm: pd.DataFrame, start: Mapping[str, float] | None = None) -> dict[str, float]:
    """Maximum-likelihood BG/NBD fit: ``{"r", "alpha", "a", "b"}`` from ``frequency``,
    ``recency`` and ``T`` (one row per customer, recency and T in one time unit)."""
    x, t_x, T = _bgnbd_data(rfm)
    return _maximize(lambda th, grad: _bgnbd_ll(th, x, t_x, T, grad), BGNBD_PARAMS,
                     start or BGNBD_START, "BG/NBD")


def gamma_gamma_mle(rfm: pd.DataFrame, start: Mapping[str, float] | None = None
                    ) -> dict[str, float]:
    """Maximum-likelihood Gamma-Gamma fit: ``{"p", "q", "v"}`` from ``frequency`` and
    ``monetary_value`` of repeat buyers (pass ``rfm[rfm["frequency"] > 0]``)."""
    x, z = _gamma_gamma_data(rfm)
    return _maximize(lambda th, grad: _gg_ll(th, x, z, grad), GAMMA_GAMMA_PARAMS,
                     start or GAMMA_GAMMA_START, "Gamma-Gamma")


def bgnbd_expected_purchases(params: Mapping[str, float], rfm: pd.DataFrame, t: float
                             ) -> np.ndarray:
    """Expected repeat purchases of each customer in the next ``t`` time units (FHL 2005 eq. 10,
    as ``BetaGeoModel.expected_purchases(data, future_t=t)`` computes it)."""
    r, alpha, a, b = (float(params[k]) for k in BGNBD_PARAMS)
    x = np.asarray(rfm["frequency"], dtype=float)
    t_x = np.asarray(rfm["recency"], dtype=float)
    T = np.asarray(rfm["T"], dtype=float)
    numerator = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(
        r + x, b + x, a + b + x - 1, t / (alpha + T + t))
    numerator *= (a + b + x - 1) / (a - 1)
    denominator = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return numerator / denominator


def gamma_gamma_expected_spend(params: Mapping[str, float], rfm: pd.DataFrame) -> np.ndarray:
    """Expected spend per purchase of each customer (``GammaGammaModel.expected_customer_spend``):
    the population mean p v / (q - 1) for x = 0, shrunk toward the customer's own mean otherwise."""
    p, q, v = (float(params[k]) for k in GAMMA_GAMMA_PARAMS)
    x = np.asarray(rfm["frequency"], dtype=float)
    z = np.asarray(rfm["monetary_value"], dtype=float)
    w = p * x / (p * x + q - 1)
    return (1 - w) * p * v / (q - 1) + w * z


def population_mean_spend(params: Mapping[str, float]) -> float:
    """The Gamma-Gamma population mean spend per purchase, p v / (q - 1)."""
    return float(params["p"] * params["v"] / (params["q"] - 1))


__all__ = [
    "BGNBD_PARAMS",
    "GAMMA_GAMMA_PARAMS",
    "MLEError",
    "bgnbd_expected_purchases",
    "bgnbd_loglik",
    "bgnbd_mle",
    "gamma_gamma_expected_spend",
    "gamma_gamma_loglik",
    "gamma_gamma_mle",
    "population_mean_spend",
]
