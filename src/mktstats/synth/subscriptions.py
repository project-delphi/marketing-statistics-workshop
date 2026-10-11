"""Monthly subscription program with right-censored tenure and a known survival truth (Module 14).

Story. A replenishment subscription launched in January 2024 in three plans (basic, standard,
premium). Each subscriber has a 14-day free trial in which they complete onboarding or not; the
first billing date ends the trial and is **time zero**, so onboarding is known when the clock
starts (no immortal time). Trial cancellations never became subscribers and are not in the
table. Sign-ups run from ``signup_start`` to ``signup_end`` at a rate that rises linearly to
``signup_growth`` times its first value; the data stop on ``observation_end``.

Data-generating process, t in months of 30.4375 days (PyMC-Marketing's constant):

    plan        ~ Categorical(plan_probs)
    u           = hidden_confounding * N(0, 1)                       (motivation; 0 by default)
    onboarded   ~ Bernoulli(expit(logit(onboard_probs[plan]) + u))
    intro_offer ~ Bernoulli(intro_offer_prob)                        (independent of the rest)
    eta         = log_hr[plan] + log_hr["onboarded"] * onboarded - u
    H0(t)       = (t / weibull_scale) ** weibull_shape               (Weibull baseline)
    H(t)        = exp(eta) * [exp(intro_log_hr * intro_offer) * H0(min(t, L))
                              + max(H0(t) - H0(L), 0)]                (L = landmark_months)
    T           = H^-1(E),  E ~ Exponential(1)                       (closed form on each piece)

so the hazard ratios of standard, premium and onboarded are constant (proportional hazards) and
the intro offer multiplies the hazard by exp(intro_log_hr) in months 0 to L only. A subscriber
cancels on day start + ceil(30.4375 * T) (at least day 1); the cancellation is observed if that
day is on or before ``observation_end``, otherwise the subscriber is censored there.

The truth is computed in continuous time. Rounding a cancellation up to a whole day lengthens
each lifetime by less than a day: on 1,000,000 prototype rows this raised S(1 month) by under
0.002 and left S(3) to S(24) within sampling error. With ``hidden_confounding`` > 0 the
population truths integrate over u by Gauss-Hermite quadrature; at 0 they are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.special import expit, gammainc, logit
from scipy.special import gamma as gamma_fn

from mktstats.synth._common import SynthResult, make_rng, sig, sig_dict

DAYS_PER_MONTH = 30.4375
PLANS = ("basic", "standard", "premium")
PLAN_PROBS = {"basic": 0.50, "standard": 0.30, "premium": 0.20}
ONBOARD_PROBS = {"basic": 0.30, "standard": 0.55, "premium": 0.80}
LOG_HR = {"plan_standard": -0.25, "plan_premium": -0.60, "onboarded": -0.40}
SURVIVAL_TIMES = (3, 6, 12, 18, 24)
HERMITE_NODES = 40


def baseline_cumhaz(t, shape: float = 0.8, scale: float = 14.0):
    """H0(t) = (t / scale) ** shape, t in months (t >= 0)."""
    return (np.asarray(t, dtype=float) / scale) ** shape


def cumulative_hazard(t, eta, intro_offer, *, shape=0.8, scale=14.0, intro_log_hr=0.70,
                      landmark=3.0):
    """H(t | eta, offer) of the module docstring; arrays broadcast."""
    t = np.asarray(t, dtype=float)
    early = baseline_cumhaz(np.minimum(t, landmark), shape, scale)
    late = np.maximum(baseline_cumhaz(t, shape, scale) - baseline_cumhaz(landmark, shape, scale),
                      0.0)
    return np.exp(eta) * (np.exp(intro_log_hr * np.asarray(intro_offer)) * early + late)


def survival(t, eta, intro_offer, **model):
    """S(t | eta, offer) = exp(-H(t | eta, offer))."""
    return np.exp(-cumulative_hazard(t, eta, intro_offer, **model))


def _weibull_area(c, a, b, shape, scale):
    """Integral of exp(-c * (t / scale) ** shape) dt from a to b (0 <= a <= b), in closed form:
    (scale / shape) * c ** (-1 / shape) * Gamma(1 / shape) * [P(1/shape, x_b) - P(1/shape, x_a)],
    with P the regularized lower incomplete gamma function and x = c * (t / scale) ** shape."""
    c = np.asarray(c, dtype=float)
    s = 1.0 / shape
    xa = c * (a / scale) ** shape
    xb = c * (b / scale) ** shape
    return scale / shape * c ** (-s) * gamma_fn(s) * (gammainc(s, xb) - gammainc(s, xa))


def rmst(horizon, eta, intro_offer, *, shape=0.8, scale=14.0, intro_log_hr=0.70, landmark=3.0):
    """Restricted mean survival time, the integral of S(t | eta, offer) from 0 to ``horizon``,
    in closed form on each piece of the hazard (months)."""
    eta = np.asarray(eta, dtype=float)
    a = np.exp(eta + intro_log_hr * np.asarray(intro_offer, dtype=float))  # rate in months 0-L
    b = np.exp(eta)  # rate after L
    if horizon <= landmark:
        return _weibull_area(a, 0.0, horizon, shape, scale)
    early = _weibull_area(a, 0.0, landmark, shape, scale)
    # After L: S(t) = exp(-(a - b) H0(L)) * exp(-b H0(t)).
    carry = np.exp(-(a - b) * baseline_cumhaz(landmark, shape, scale))
    return early + carry * _weibull_area(b, landmark, horizon, shape, scale)


def discounted_rmst(horizon, eta, intro_offer, monthly_rate, **model):
    """The integral of S(t) * (1 + monthly_rate) ** -t from 0 to ``horizon`` (numerical)."""
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    offer = np.broadcast_to(np.asarray(intro_offer, dtype=float), eta.shape)
    landmark = model.get("landmark", 3.0)

    def f(t):
        return survival(t, eta, offer, **model) * (1.0 + monthly_rate) ** (-t)

    pieces = [(0.0, min(landmark, horizon))] + ([(landmark, horizon)] if horizon > landmark else [])
    return sum(quad_vec(f, lo, hi, epsabs=1e-11, epsrel=1e-11)[0] for lo, hi in pieces)


def lifetimes(e, eta, intro_offer, *, shape=0.8, scale=14.0, intro_log_hr=0.70, landmark=3.0):
    """Invert H at e (Exponential(1) draws): T with H(T) = e, in closed form on each piece.

    Each piece is computed only where it applies, so no power of a negative number and no
    overflow from the branch that does not apply."""
    e = np.asarray(e, dtype=float)
    eta = np.broadcast_to(np.asarray(eta, dtype=float), e.shape)
    offer = np.broadcast_to(np.asarray(intro_offer, dtype=float), e.shape)
    a = np.exp(eta + intro_log_hr * offer)
    b = np.exp(eta)
    h_l = float(baseline_cumhaz(landmark, shape, scale))
    early = e < a * h_l
    t = np.empty_like(e)
    t[early] = scale * (e[early] / a[early]) ** (1.0 / shape)
    late = ~early
    t[late] = scale * ((e[late] - a[late] * h_l) / b[late] + h_l) ** (1.0 / shape)
    return t


def _signup_fraction(v, growth: float):
    """Inverse CDF of a sign-up rate that rises linearly from 1 to ``growth`` over [0, 1]."""
    if growth == 1:
        return v
    return (np.sqrt(1.0 + (growth**2 - 1.0) * v) - 1.0) / (growth - 1.0)


class _Model:
    """The parameters, and population truths that integrate over plan, offer and u."""

    def __init__(self, plan_probs, onboard_probs, intro_offer_prob, log_hr, shape, scale,
                 intro_log_hr, landmark, hidden):
        self.plan_probs, self.onboard_probs = plan_probs, onboard_probs
        self.offer_prob, self.log_hr, self.hidden = intro_offer_prob, log_hr, hidden
        self.kw = {"shape": shape, "scale": scale, "intro_log_hr": intro_log_hr,
                   "landmark": landmark}
        if hidden == 0:
            nodes, weights = np.zeros(1), np.ones(1)
        else:
            x, w = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
            nodes, weights = hidden * x, w / w.sum()
        # One row per (plan, onboarded, offer, quadrature node), with its probability.
        rows = []
        for plan in PLANS:
            for onb in (0, 1):
                for offer in (0, 1):
                    p_onb = expit(logit(onboard_probs[plan]) + nodes)
                    p_onb = p_onb if onb else 1.0 - p_onb
                    p_offer = intro_offer_prob if offer else 1.0 - intro_offer_prob
                    for u, w, po in zip(nodes, weights, p_onb, strict=True):
                        rows.append((plan, onb, offer, u, plan_probs[plan] * p_offer * w,
                                     plan_probs[plan] * p_offer * w * po))
        self.cells = pd.DataFrame(rows, columns=["plan", "onboarded", "intro_offer", "u",
                                                 "w_set", "w_obs"])

    def eta(self, plan, onboarded, u):
        plan = np.asarray(plan)
        return (self.log_hr["plan_standard"] * (plan == "standard")
                + self.log_hr["plan_premium"] * (plan == "premium")
                + self.log_hr["onboarded"] * np.asarray(onboarded) - np.asarray(u))

    def _avg(self, values, weights):
        weights = np.asarray(weights, dtype=float)
        return np.asarray(values) @ weights / weights.sum()

    def survival(self, times, *, plan=None, onboarded=None, set_onboarded=None):
        """Population S(t): observed groups when ``onboarded`` is given, onboarding set to
        ``set_onboarded`` for everyone (standardized) when that is given."""
        c = self.cells
        if plan is not None:
            c = c[c["plan"] == plan]
        if set_onboarded is not None:
            c = c[c["onboarded"] == set_onboarded]
            weight = c["w_set"]
        else:
            if onboarded is not None:
                c = c[c["onboarded"] == onboarded]
            weight = c["w_obs"]
        eta = self.eta(c["plan"], c["onboarded"], c["u"])
        s = survival(np.asarray(times, dtype=float)[:, None], eta[None, :],
                     c["intro_offer"].to_numpy()[None, :], **self.kw)
        return self._avg(s, weight)

    def rmst(self, horizon, *, plan=None, onboarded=None, set_onboarded=None, monthly_rate=0.0):
        c = self.cells
        if plan is not None:
            c = c[c["plan"] == plan]
        if set_onboarded is not None:
            c = c[c["onboarded"] == set_onboarded]
            weight = c["w_set"]
        else:
            if onboarded is not None:
                c = c[c["onboarded"] == onboarded]
            weight = c["w_obs"]
        eta = self.eta(c["plan"], c["onboarded"], c["u"])
        offer = c["intro_offer"].to_numpy()
        if monthly_rate:
            r = discounted_rmst(horizon, eta, offer, monthly_rate, **self.kw)
        else:
            r = rmst(horizon, eta, offer, **self.kw)
        return float(self._avg(r, weight))

    def gain(self, horizon, *, plan=None, among_onboarded=None, monthly_rate=0.0):
        """Mean RMST gain from setting onboarding to 1 rather than 0: over everyone, or over
        the subscribers whose observed onboarding is ``among_onboarded`` (e.g. 0, the target
        population), within ``plan`` if given."""
        c = self.cells
        if among_onboarded is not None:
            c = c[c["onboarded"] == among_onboarded]
            weight = c["w_obs"]
        else:
            c = c[c["onboarded"] == 0]  # each (plan, offer, u) once
            weight = c["w_set"]
        if plan is not None:
            keep = (c["plan"] == plan).to_numpy()
            c, weight = c[keep], weight[keep]
        offer = c["intro_offer"].to_numpy()
        values = []
        for onb in (1, 0):
            eta = self.eta(c["plan"], np.full(len(c), onb), c["u"])
            if monthly_rate:
                values.append(discounted_rmst(horizon, eta, offer, monthly_rate, **self.kw))
            else:
                values.append(rmst(horizon, eta, offer, **self.kw))
        return float(self._avg(values[0] - values[1], weight))

    def plan_mix(self, onboarded=None):
        c = self.cells if onboarded is None else self.cells[self.cells["onboarded"] == onboarded]
        w = c.groupby("plan")["w_obs"].sum()
        return {p: float(w[p] / w.sum()) for p in PLANS}

    def share_onboarded(self, plan=None):
        c = self.cells if plan is None else self.cells[self.cells["plan"] == plan]
        return float(c.loc[c["onboarded"] == 1, "w_obs"].sum() / c["w_obs"].sum())


def true_survival(times, *, plan=None, onboarded=None, set_onboarded=None,
                  hidden_confounding: float = 0.0) -> np.ndarray:
    """The true population S(t) at ``times`` (months) under the default parameters: everyone,
    one ``plan``, an observed onboarding group (``onboarded``), or everyone with onboarding set
    to ``set_onboarded``. The same quantities as truth["survival"], at any t."""
    model = _Model(PLAN_PROBS, ONBOARD_PROBS, 0.30, LOG_HR, 0.8, 14.0, 0.70, 3.0,
                   hidden_confounding)
    return model.survival(np.atleast_1d(np.asarray(times, dtype=float)), plan=plan,
                          onboarded=onboarded, set_onboarded=set_onboarded)


def subscriptions(
    seed: int = 2034,
    *,
    n: int = 8_000,
    signup_start: str = "2024-01-01",
    signup_end: str = "2025-12-31",
    observation_end: str = "2026-06-30",
    signup_growth: float = 2.0,
    plan_probs: dict | None = None,
    onboard_probs: dict | None = None,
    intro_offer_prob: float = 0.30,
    log_hr: dict | None = None,
    weibull_shape: float = 0.8,
    weibull_scale: float = 14.0,
    intro_log_hr: float = 0.70,
    landmark_months: float = 3.0,
    hidden_confounding: float = 0.0,
    horizon_months: int = 24,
    margin: float = 12.0,
    call_cost: float = 15.0,
    q_assumed: float = 0.35,
    q_grid: tuple = (0.20, 0.35, 0.50),
    monthly_discount: float = 0.008,
) -> SynthResult:
    """Subscribers with start and (if observed) cancellation dates, plan, onboarding and offer.

    Tables
    ------
    subscribers : customer_id, start_date, cancel_date (empty if active at ``observation_end``),
        plan (basic / standard / premium), onboarded (0/1), intro_offer (0/1), and two truth
        columns that real data never has: true_tenure_months (the uncensored lifetime, whole days
        / 30.4375) and true_rmst24_gain (this subscriber's RMST over ``horizon_months`` with
        onboarded set to 1 minus with it set to 0, given plan, offer and u).

    ``margin`` (dollars a month), ``call_cost``, ``q_assumed`` and ``q_grid`` (the share of called
    subscribers a call gets to onboard) are the assumptions of the decision: call a subscriber
    who has not onboarded when q * gain * margin > call_cost.
    """
    plan_probs = dict(plan_probs or PLAN_PROBS)
    onboard_probs = dict(onboard_probs or ONBOARD_PROBS)
    log_hr = dict(log_hr or LOG_HR)
    if sorted(plan_probs) != sorted(PLANS) or abs(sum(plan_probs.values()) - 1) > 1e-12:
        raise ValueError(f"plan_probs needs the plans {PLANS} and must sum to 1")
    kw = {"shape": weibull_shape, "scale": weibull_scale, "intro_log_hr": intro_log_hr,
          "landmark": landmark_months}
    start0 = pd.Timestamp(signup_start)
    obs_end = pd.Timestamp(observation_end)
    window_days = (pd.Timestamp(signup_end) - start0).days + 1
    if obs_end <= pd.Timestamp(signup_end):
        raise ValueError("observation_end must come after signup_end")

    rng = make_rng(seed)
    plan_idx = rng.choice(len(PLANS), size=n, p=[plan_probs[p] for p in PLANS])
    plan = np.array(PLANS, dtype=object)[plan_idx]
    u = hidden_confounding * rng.standard_normal(n)  # drawn even at 0: same stream for any value
    p_onb = expit(logit(np.array([onboard_probs[p] for p in PLANS]))[plan_idx] + u)
    onboarded = (rng.random(n) < p_onb).astype(np.int64)
    intro_offer = (rng.random(n) < intro_offer_prob).astype(np.int64)
    eta = (log_hr["plan_standard"] * (plan == "standard")
           + log_hr["plan_premium"] * (plan == "premium")
           + log_hr["onboarded"] * onboarded - u)
    e = rng.exponential(1.0, n)
    t_true = lifetimes(e, eta, intro_offer, **kw)
    v = rng.random(n)
    offset = np.floor(window_days * _signup_fraction(v, signup_growth)).astype(np.int64)
    offset = np.minimum(offset, window_days - 1)

    # Whole days to cancellation, at least 1. A cap far beyond any follow-up keeps the integer
    # conversion safe for absurd lifetimes (strong hidden confounding); it never binds below
    # 1e12 days, and dates are built only for cancellations that are observed.
    true_days = np.maximum(1, np.ceil(np.minimum(DAYS_PER_MONTH * t_true, 1e12))).astype(np.int64)
    potential = (obs_end - start0).days - offset  # days from start to the observation end
    churned = true_days <= potential
    start_date = start0 + pd.to_timedelta(offset, unit="D")
    cancel_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    cancel_date[churned] = start_date[churned] + pd.to_timedelta(true_days[churned], unit="D")

    eta1 = eta - log_hr["onboarded"] * onboarded + log_hr["onboarded"]
    eta0 = eta - log_hr["onboarded"] * onboarded
    gain_row = rmst(horizon_months, eta1, intro_offer, **kw) - rmst(horizon_months, eta0,
                                                                     intro_offer, **kw)
    order = np.argsort(offset, kind="stable")
    subscribers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "start_date": start_date[order],
            "cancel_date": cancel_date.to_numpy()[order],
            "plan": plan[order],
            "onboarded": onboarded[order],
            "intro_offer": intro_offer[order],
            "true_tenure_months": np.round(true_days[order] / DAYS_PER_MONTH, 6),
            "true_rmst24_gain": np.round(gain_row[order], 4),
        }
    )

    model = _Model(plan_probs, onboard_probs, intro_offer_prob, log_hr, weibull_shape,
                   weibull_scale, intro_log_hr, landmark_months, hidden_confounding)
    truth = _truth(subscribers, model, seed=seed, n=n, signup_start=signup_start,
                   signup_end=signup_end, observation_end=observation_end,
                   signup_growth=signup_growth, horizon=horizon_months, margin=margin,
                   call_cost=call_cost, q_assumed=q_assumed, q_grid=q_grid,
                   monthly_discount=monthly_discount)
    return SynthResult("subscriptions", {"subscribers": subscribers}, truth)


def _truth(subs, model: _Model, *, seed, n, signup_start, signup_end, observation_end,
           signup_growth, horizon, margin, call_cost, q_assumed, q_grid, monthly_discount):
    kw, log_hr = model.kw, model.log_hr
    times = list(SURVIVAL_TIMES)

    def curve(**sel):
        return {str(t): sig(s, 6) for t, s in zip(times, model.survival(times, **sel),
                                                    strict=True)}

    by_plan_rmst = {}
    for plan in PLANS:
        on = model.rmst(horizon, plan=plan, set_onboarded=1)
        off = model.rmst(horizon, plan=plan, set_onboarded=0)
        by_plan_rmst[plan] = {
            "onboarded": sig(on), "not_onboarded": sig(off), "gain": sig(on - off),
            "gain_among_not_onboarded": sig(model.gain(horizon, plan=plan, among_onboarded=0)),
        }
    weights = model.plan_mix(onboarded=0)
    gain_target = model.gain(horizon, among_onboarded=0)
    gain_target_disc = model.gain(horizon, among_onboarded=0, monthly_rate=monthly_discount)
    rmst_obs = {o: model.rmst(horizon, onboarded=o) for o in (0, 1)}
    target_rows = subs["onboarded"] == 0
    hr_before = float(np.exp(kw["intro_log_hr"]))
    obs_end = pd.Timestamp(observation_end)
    follow = (obs_end - subs["start_date"]).dt.days
    tenure_days = np.where(subs["cancel_date"].notna(),
                           (subs["cancel_date"] - subs["start_date"]).dt.days, follow)
    return {
        "description": (
            "Monthly subscription program, three plans, sign-ups growing over two years, "
            "observed until observation_end. Weibull proportional-hazards churn in plan and "
            "onboarding; an intro offer that raises the hazard only before the landmark. "
            "Subscribers still active at observation_end are right-censored."
        ),
        "seed": seed,
        "n": n,
        "signup_window": [signup_start, signup_end],
        "signup_growth": signup_growth,
        "observation_end": observation_end,
        "days_per_month": DAYS_PER_MONTH,
        "time_zero": (
            "The first billing date, when the 14-day trial ends: onboarding is completed (or "
            "not) during the trial, so it is known when the clock starts and no subscriber's "
            "group depends on how long they stayed (no immortal-time bias)."
        ),
        "model": {
            "hazard": (
                "h(t | z) = h0(t) exp(eta), eta = -0.25 standard - 0.60 premium - 0.40 onboarded"
                " - u; H(t) = exp(eta) [exp(0.70 intro_offer) H0(min(t, 3)) + max(H0(t) - H0(3),"
                " 0)], H0(t) = (t / 14) ** 0.8, t in months of 30.4375 days"
            ),
            "weibull_shape": kw["shape"],
            "weibull_scale_months": kw["scale"],
            "reference_group": "basic plan, not onboarded, no intro offer",
            "log_hazard_ratios": dict(log_hr),
            "hazard_ratios": sig_dict({k: np.exp(v) for k, v in log_hr.items()}),
            "intro_offer": {"landmark_months": kw["landmark"], "log_hr_before": kw["intro_log_hr"],
                            "hr_before": sig(hr_before), "hr_after": 1.0},
            "hidden_confounding": model.hidden,
            "hidden_confounding_note": (
                "u = hidden_confounding * N(0, 1) adds to the log-odds of onboarding and "
                "subtracts from eta; the hazard ratios above are conditional on u."
            ),
        },
        "covariates": {
            "plan_probs": dict(model.plan_probs),
            "onboard_prob_by_plan": dict(model.onboard_probs),
            "intro_offer_prob": model.offer_prob,
            "share_onboarded": sig(model.share_onboarded()),
            "independence": (
                "Onboarding depends on plan (and on u when hidden_confounding > 0) and nothing "
                "else; the intro offer is independent of plan, onboarding and u."
            ),
        },
        "survival": {
            "times_months": times,
            "overall": curve(),
            "by_plan": {plan: curve(plan=plan) for plan in PLANS},
            "by_observed_onboarding": {"onboarded": curve(onboarded=1),
                                       "not_onboarded": curve(onboarded=0)},
            "standardized": {"onboarded": curve(set_onboarded=1),
                             "not_onboarded": curve(set_onboarded=0)},
            "by_intro_offer_note": "the offer is independent of plan and onboarding",
        },
        "rmst": {
            "horizon_months": horizon,
            "definition": "RMST(t*) = integral of S(t) from 0 to t*: expected months subscribed "
                          "within the first t* months",
            "by_plan": by_plan_rmst,
            "observed_groups": {"onboarded": sig(rmst_obs[1]), "not_onboarded": sig(rmst_obs[0]),
                                "unadjusted_gain": sig(rmst_obs[1] - rmst_obs[0])},
            "target_population": "subscribers who did not complete onboarding",
            "target_plan_weights": sig_dict(weights),
            "gain_target": sig(gain_target),
            "gain_target_definition": (
                "mean over the target population of RMST(24) with onboarding set to 1 minus with "
                "it set to 0; with hidden_confounding = 0 it equals the plan-weighted sum of the "
                "per-plan gains"
            ),
            "gain_target_sample_mean": sig(float(subs.loc[target_rows, "true_rmst24_gain"].mean())),
            "monthly_discount": monthly_discount,
            "gain_target_discounted": sig(gain_target_disc),
            "discounting": "each month subscribed at time t is worth (1 + monthly_discount) ** -t",
        },
        "decision": {
            "margin_per_month": margin,
            "call_cost": call_cost,
            "q_assumed": q_assumed,
            "q_grid": list(q_grid),
            "rule": "call a subscriber who did not onboard when q * gain * margin > call_cost",
            "true_value_per_call": {str(q): sig(q * gain_target * margin) for q in q_grid},
            "true_break_even_q": sig(call_cost / (gain_target * margin)),
            "n_target": int(target_rows.sum()),
        },
        "sample": {
            "n_churned": int(subs["cancel_date"].notna().sum()),
            "share_churned": sig(float(subs["cancel_date"].notna().mean())),
            "max_follow_up_months": sig(float(tenure_days.max() / DAYS_PER_MONTH)),
            "n_followed_24_months": int((tenure_days >= 24 * DAYS_PER_MONTH).sum()),
            "share_onboarded_by_plan": {p: sig(float(subs.loc[subs["plan"] == p,
                                                              "onboarded"].mean()))
                                        for p in PLANS},
        },
    }
