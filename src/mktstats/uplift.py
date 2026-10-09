"""Uplift evaluation from scratch (numpy only): uplift and Qini curves, Qini coefficient, AUUC,
uplift at top k, and the value of a targeting policy.

Notation: customers are sorted by predicted uplift ``score``, highest first. For the first ``k``
customers, ``N_T(k)``, ``N_C(k)`` are the numbers of treated and control customers and ``Y_T(k)``,
``Y_C(k)`` the sums of their outcomes. ``n`` is the number of customers and ``phi = k / n`` the
fraction targeted.

* Qini curve (Radcliffe 2007, as written in Gutierrez & Gérardy 2017, Eq. 18):
  ``g(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k)`` — incremental outcome among the targeted treated
  customers, in outcome units (e.g. dollars).
* Uplift curve (Gutierrez & Gérardy 2017, Eq. 17):
  ``f(k) = (Y_T(k) / N_T(k) - Y_C(k) / N_C(k)) * (N_T(k) + N_C(k))``.
  A ratio with an empty group counts as 0, as in scikit-uplift 0.5 (``sklift/metrics/metrics.py``).
* Ties: customers with equal scores cannot be ordered, so the curves are evaluated only after each
  complete group of tied scores (scikit-uplift does the same). Every curve starts at (0, 0).
* Qini coefficient: area between the Qini curve and the random-targeting line from (0, 0) to
  (1, g(n)), with the x axis in fraction targeted (trapezoid rule). Units: outcome units.
  ``normalize=True`` divides by the same area for the perfect ordering of the observed data
  (treated customers by decreasing outcome first, then control customers by increasing outcome),
  giving Radcliffe's normalized coefficient (1 = perfect, 0 = random).
* AUUC: area under the uplift curve over the fraction targeted; ``subtract_random=True`` returns the
  area above the random line from (0, 0) to (1, f(n)).
* Uplift at k: ``mean(Y | T=1) - mean(Y | T=0)`` among the top ``ceil(k * n)`` customers by score
  (scikit-uplift's ``strategy="overall"``); ties at the cut are broken by the original row order.
* Policy value: a rule treats customer i when ``policy_i = 1``. With a randomized experiment and
  propensity ``e`` the inverse-probability-weighted estimate of profit per customer, relative to
  treating nobody, is
  ``mean(policy * (T * (margin * Y - cost) / e - (1 - T) * margin * Y / (1 - e)))``.
  With the true CATE (synthetic data) the same quantity is
  ``mean(policy * (margin * cate - cost))``.

Sources opened 2026-10-09: P. Gutierrez and J.-Y. Gérardy (2017), "Causal Inference and Uplift
Modeling: A review of the literature", PMLR 67:1-13, https://proceedings.mlr.press/v67/gutierrez17a.html
(Eq. 17 and 18); scikit-uplift ``sklift/metrics/metrics.py``,
https://github.com/maks-sh/scikit-uplift (tie handling, empty-group convention, normalization).
Radcliffe, N. J. (2007), "Using control groups to target on predicted lift: Building and assessing
uplift models", Direct Marketing Analytics Journal 3:14-21, is cited through those two sources (not
opened).
"""

from __future__ import annotations

import math

import numpy as np


def _inputs(y, treatment, score):
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment)
    s = np.asarray(score, dtype=float)
    if not (y.shape == t.shape == s.shape) or y.ndim != 1:
        raise ValueError("y, treatment and score must be 1-d arrays of the same length")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("treatment must be 0/1")
    if np.isnan(s).any() or np.isnan(y).any():
        raise ValueError("y and score must not contain NaN")
    return y, t.astype(bool), s


def _cumulative(y, treatment, score):
    """Cumulative N_T, N_C, Y_T, Y_C after each complete tie group, with a leading 0."""
    y, t, s = _inputs(y, treatment, score)
    order = np.argsort(-s, kind="mergesort")
    y, t, s = y[order], t[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    n_t = np.r_[0, np.cumsum(t)[ends]]
    n_c = np.r_[0, np.cumsum(~t)[ends]]
    y_t = np.r_[0.0, np.cumsum(np.where(t, y, 0.0))[ends]]
    y_c = np.r_[0.0, np.cumsum(np.where(t, 0.0, y))[ends]]
    k = np.r_[0, ends + 1]
    return k, n_t, n_c, y_t, y_c


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def qini_curve(y, treatment, score):
    """Qini curve: returns ``(phi, g)``, fraction targeted and incremental outcome (see module)."""
    k, n_t, n_c, y_t, y_c = _cumulative(y, treatment, score)
    g = y_t - y_c * _safe_div(n_t, n_c)
    return k / k[-1], g


def uplift_curve(y, treatment, score):
    """Uplift curve: returns ``(phi, f)`` (see module docstring)."""
    k, n_t, n_c, y_t, y_c = _cumulative(y, treatment, score)
    f = (_safe_div(y_t, n_t) - _safe_div(y_c, n_c)) * (n_t + n_c)
    return k / k[-1], f


def _trapezoid(curve, x) -> float:
    curve = np.asarray(curve, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.diff(x) * (curve[1:] + curve[:-1]) / 2.0))


def _area_above_random(x, curve):
    return _trapezoid(curve, x) - float(curve[-1]) / 2.0


def _perfect_score(y, treatment):
    y, t, _ = _inputs(y, treatment, np.zeros(len(np.asarray(y))))
    top = np.abs(y).max() + 1.0
    # treated by decreasing outcome first, then control by increasing outcome
    return np.where(t, top + y, -top - y)


def qini_coefficient(y, treatment, score, normalize: bool = False) -> float:
    """Area between the Qini curve and the random line (fraction-targeted x axis).

    ``normalize=True`` divides by the area of the perfect ordering of the same data.
    """
    x, g = qini_curve(y, treatment, score)
    q = _area_above_random(x, g)
    if normalize:
        xp, gp = qini_curve(y, treatment, _perfect_score(y, treatment))
        qp = _area_above_random(xp, gp)
        return q / qp if qp != 0 else float("nan")
    return q


def auuc(y, treatment, score, subtract_random: bool = False) -> float:
    """Area under the uplift curve over the fraction targeted (trapezoid rule)."""
    x, f = uplift_curve(y, treatment, score)
    area = _trapezoid(f, x)
    return area - float(f[-1]) / 2.0 if subtract_random else area


def uplift_at_k(y, treatment, score, k: float = 0.3) -> float:
    """Treated minus control mean outcome among the top ``ceil(k * n)`` customers by score."""
    y, t, s = _inputs(y, treatment, score)
    if not 0 < k <= 1:
        raise ValueError("k is a fraction in (0, 1]")
    top = np.argsort(-s, kind="mergesort")[: math.ceil(k * len(s))]
    yt, tt = y[top], t[top]
    if tt.all() or (~tt).all():
        raise ValueError("the top k contains only one group; use a larger k")
    return float(yt[tt].mean() - yt[~tt].mean())


def targeting_rule(cate, margin: float, cost: float) -> np.ndarray:
    """Treat (1) when predicted CATE x margin exceeds the cost of treating; else 0."""
    return (np.asarray(cate, dtype=float) * margin > cost).astype(np.int64)


def policy_value(y, treatment, policy, margin: float, cost: float, propensity=0.5) -> float:
    """IPW estimate of profit per customer of ``policy`` relative to treating nobody.

    ``propensity`` is P(T=1) in the experiment (scalar or per-row array).
    """
    y, t, _ = _inputs(y, treatment, np.zeros(len(np.asarray(y))))
    pi = np.asarray(policy, dtype=float)
    e = np.broadcast_to(np.asarray(propensity, dtype=float), y.shape)
    if pi.shape != y.shape:
        raise ValueError("policy must have one entry per customer")
    if ((e <= 0) | (e >= 1)).any():
        raise ValueError("propensity must be strictly between 0 and 1")
    term = np.where(t, (margin * y - cost) / e, -margin * y / (1 - e))
    return float(np.mean(pi * term))


def policy_value_true(true_cate, policy, margin: float, cost: float) -> float:
    """Exact profit per customer of ``policy`` relative to treating nobody, from the true CATE."""
    cate = np.asarray(true_cate, dtype=float)
    pi = np.asarray(policy, dtype=float)
    return float(np.mean(pi * (margin * cate - cost)))


__all__ = [
    "auuc",
    "policy_value",
    "policy_value_true",
    "qini_coefficient",
    "qini_curve",
    "targeting_rule",
    "uplift_at_k",
    "uplift_curve",
]
