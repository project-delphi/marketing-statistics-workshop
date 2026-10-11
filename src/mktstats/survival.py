"""Survival analysis helpers for Module 14 (subscription churn).

Three kinds of things live here:

- **Checks on confidence intervals** (``mktstats.checks`` checks intervals of posterior draws;
  these take an interval that a library computed): :func:`ci_contains`,
  :func:`k_of_K_ci_contains`. Like every check they raise ``AssertionError`` with a message
  that names the interval (type and level) and returns quietly when the input passes.
- A **stratified bootstrap**, :func:`bootstrap_interval`: resample rows with replacement within
  groups, recompute a statistic, and return the percentile interval with the draws.
- The lab's **reference solutions** and provided helpers (:func:`kaplan_meier`,
  :func:`expected_periods`, :func:`cox_hazard_ratios`, :func:`landmark_split`,
  :func:`rmst_gain`, :func:`tenure_and_churn`, :func:`monthly_churn_rate`) and
  :func:`checkpoint_outcomes`, which applies every checkpoint rule of the lab to one dataset.
  Tests and the multi-seed coverage sweep use them; the notebook keeps its own copies in its
  solution cells, so a learner reads the code where they need it.

lifelines is imported inside the functions that use it, so ``import mktstats.survival`` loads
only numpy and pandas. Checked against lifelines 0.30.3 (``_variables.yml`` ``packages``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375
COVARIATES = ["plan_standard", "plan_premium", "onboarded", "intro_offer"]


def _fail(message: str) -> None:
    raise AssertionError(message)


def _finite(x, what: str) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        _fail(f"{what} should be a number; got {x!r}.")
    if not np.isfinite(value):
        _fail(f"{what} is {value}: it should be a finite number.")
    return value


def _interval_label(level: float, kind: str) -> str:
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, not {level}")
    return f"{level * 100:g}% {kind}"


def _check_interval(lower, upper, what: str) -> tuple[float, float]:
    lo = _finite(lower, f"The lower end of the interval for {what}")
    hi = _finite(upper, f"The upper end of the interval for {what}")
    if lo > hi:
        _fail(
            f"The interval for {what} runs from {lo:.4g} down to {hi:.4g}: the lower end is above"
            " the upper end. Check the column order (lower, then upper)."
        )
    return lo, hi


# ---- checks on confidence intervals -----------------------------------------------------------


def ci_contains(
    truth,
    lower,
    upper,
    level: float = 0.95,
    name: str | None = None,
    kind: str = "confidence interval",
    hint: str = "",
    label: str | None = None,
) -> bool:
    """The true value lies inside [lower, upper], a ``level`` interval of the named ``kind``
    ("confidence interval", "pointwise confidence interval", "bootstrap percentile interval").
    ``hint`` is added to the failure message (what to check and how to recover); ``label``
    (e.g. "Checkpoint 3") starts it."""
    what = name or "the quantity"
    interval = _interval_label(level, kind)
    t = _finite(truth, f"The true value of {what}")
    lo, hi = _check_interval(lower, upper, what)
    if not lo <= t <= hi:
        side = "below" if t < lo else "above"
        _fail(
            (f"{label}: t" if label else "T")
            + f"he true value of {what}, {t:.4g}, is {side} its {interval} [{lo:.4g}, {hi:.4g}]."
            + (f" {hint.strip()}" if hint else "")
        )
    return True


def k_of_K_ci_contains(
    truths: Mapping[str, float],
    intervals: Mapping[str, Sequence[float]],
    k: int,
    level: float = 0.95,
    name: str | None = None,
    kind: str = "confidence interval",
    hint: str = "",
    label: str | None = None,
) -> dict:
    """At least ``k`` of the ``K`` true values lie inside their intervals.

    ``truths`` maps a key (a covariate, a time) to its true value; ``intervals`` maps the same
    keys to ``(lower, upper)``. Returns ``{key: (lower, upper, inside)}``. ``hint`` and
    ``label`` as in :func:`ci_contains`."""
    if not isinstance(truths, Mapping) or not isinstance(intervals, Mapping):
        _fail("truths and intervals must both be mappings with the same keys.")
    if set(truths) != set(intervals):
        _fail(
            f"truths and intervals must have the same keys; truths has {sorted(map(str, truths))},"
            f" intervals has {sorted(map(str, intervals))}."
        )
    if not 0 < k <= len(truths):
        raise ValueError(f"k must be between 1 and {len(truths)}, not {k}")
    interval = _interval_label(level, kind)
    out = {}
    for key in truths:
        pair = intervals[key]
        if len(pair) != 2:
            _fail(f"The interval for {key} should be (lower, upper); got {pair!r}.")
        lo, hi = _check_interval(pair[0], pair[1], str(key))
        t = _finite(truths[key], f"The true value of {key}")
        out[key] = (lo, hi, bool(lo <= t <= hi))
    inside = sum(v[2] for v in out.values())
    if inside < k:
        outside = ", ".join(
            f"{key} (truth {float(truths[key]):.4g}, interval [{out[key][0]:.4g},"
            f" {out[key][1]:.4g}])"
            for key in truths
            if not out[key][2]
        )
        _fail(
            (f"{label}: o" if label else "O")
            + f"nly {inside} of {len(out)} true values of {name or 'the quantities'} are inside"
            f" their {interval}; at least {k} should be. Outside: {outside}."
            + (f" {hint.strip()}" if hint else "")
        )
    return out


# ---- the stratified bootstrap -----------------------------------------------------------------


@dataclass
class BootstrapResult:
    """What :func:`bootstrap_interval` returns: the statistic on the data, its B bootstrap
    draws, and the ``prob`` percentile interval of the draws."""

    estimate: float
    draws: np.ndarray = field(repr=False)
    lower: float
    upper: float
    prob: float
    B: int
    seed: int
    by: list

    @property
    def label(self) -> str:
        return f"{self.prob * 100:g}% bootstrap percentile interval"


def bootstrap_interval(
    fn: Callable[[pd.DataFrame], float],
    data: pd.DataFrame,
    by: str | Sequence[str] | None,
    B: int = 200,
    seed: int = 14,
    prob: float = 0.95,
) -> BootstrapResult:
    """Stratified bootstrap of ``fn(data)``.

    Each of the ``B`` resamples draws, within every group of ``by`` (all rows if ``None``), as
    many rows as the group has, with replacement, from ``numpy.random.default_rng(seed)``; so
    every resample keeps the group sizes. ``fn`` gets the resample (index reset) and returns a
    number. The interval is the (1 - prob) / 2 and (1 + prob) / 2 quantiles of the draws
    (numpy's default linear interpolation)."""
    if not 0 < prob < 1:
        raise ValueError(f"prob must be between 0 and 1, not {prob}")
    if B < 2:
        raise ValueError(f"B must be at least 2, not {B}")
    if len(data) == 0:
        raise ValueError("bootstrap_interval needs data with at least one row")
    keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if keys:
        groups = [np.asarray(ix) for _, ix in sorted(data.groupby(keys).indices.items())]
    else:
        groups = [np.arange(len(data))]
    estimate = float(fn(data))
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        pos = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        draws[b] = float(fn(data.iloc[pos].reset_index(drop=True)))
    if not np.isfinite(draws).all():
        raise ValueError(
            f"{int((~np.isfinite(draws)).sum())} of {B} bootstrap draws are not finite numbers:"
            " check that the statistic is defined on every resample"
        )
    lo, hi = np.quantile(draws, [(1 - prob) / 2, (1 + prob) / 2])
    return BootstrapResult(estimate, draws, float(lo), float(hi), prob, B, seed, keys)


# ---- the lab's provided helpers and reference solutions ----------------------------------------


def tenure_and_churn(subscribers: pd.DataFrame,
                     observation_end: str = "2026-06-30") -> pd.DataFrame:
    """Add ``tenure_months`` ((cancel date, or the observation end, minus start) in days /
    30.4375) and ``churned`` (1 if a cancellation is recorded) to a copy of the table."""
    end = pd.Timestamp(observation_end)
    start = pd.to_datetime(subscribers["start_date"])
    cancel = pd.to_datetime(subscribers["cancel_date"])
    days = (cancel.fillna(end) - start).dt.days
    return subscribers.assign(tenure_months=days / DAYS_PER_MONTH,
                              churned=cancel.notna().astype(int))


def add_plan_dummies(subs: pd.DataFrame) -> pd.DataFrame:
    """``plan_standard`` and ``plan_premium`` (0/1); basic is the reference."""
    return subs.assign(plan_standard=(subs["plan"] == "standard").astype(int),
                       plan_premium=(subs["plan"] == "premium").astype(int))


def kaplan_meier(duration, event) -> pd.Series:
    """Kaplan-Meier: S(t) just after each distinct time with at least one event, ascending."""
    df = pd.DataFrame({"t": np.asarray(duration, dtype=float),
                       "d": np.asarray(event, dtype=float)})
    by_time = df.groupby("t")["d"].agg(["sum", "size"])
    at_risk = by_time["size"][::-1].cumsum()[::-1]  # duration >= t_j, censored ones included
    s = (1 - by_time["sum"] / at_risk).cumprod()
    return s[by_time["sum"] > 0].rename("survival")


def expected_periods(survival, horizon: int) -> float:
    """S(0) + S(1) + ... + S(horizon - 1); ValueError if the curve is shorter than that."""
    s = np.asarray(survival, dtype=float)
    if len(s) < horizon:
        raise ValueError(
            f"the curve is known for periods 0 to {len(s) - 1} only; a horizon of {horizon}"
            f" needs S(0) to S({horizon - 1})"
        )
    return float(s[:horizon].sum())


def cox_hazard_ratios(subs: pd.DataFrame, covariates: list[str], alpha: float = 0.05):
    """Cox PH fit on ``covariates``; one row per covariate with ``hr``, ``lower``, ``upper``."""
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(alpha=alpha).fit(subs[list(covariates) + ["tenure_months", "churned"]],
                                       duration_col="tenure_months", event_col="churned")
    ci = np.exp(cph.confidence_intervals_)
    out = pd.DataFrame({"hr": cph.hazard_ratios_, "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]})
    return out.loc[list(covariates)]


def landmark_split(subs: pd.DataFrame, landmark: float = 3.0):
    """(before, after): everyone followed to the landmark; those still at risk after it, with
    the clock restarted at the landmark."""
    late = subs["tenure_months"] > landmark
    before = subs.assign(tenure_months=subs["tenure_months"].clip(upper=landmark),
                         churned=subs["churned"].where(~late, 0))
    after = subs[late].assign(tenure_months=subs.loc[late, "tenure_months"] - landmark)
    return before, after


def rmst_gain(subs: pd.DataFrame, horizon: float, weights) -> float:
    """Sum over plans of weights[plan] x (RMST(horizon) onboarded - RMST(horizon) not), each
    RMST from its own Kaplan-Meier curve."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    w = pd.Series(weights, dtype=float)
    if not abs(w.sum() - 1) < 1e-9:
        raise ValueError(f"weights must sum to 1, not {w.sum():.4g}")
    total = 0.0
    for plan, weight in w.items():
        area = {}
        for onb in (1, 0):
            g = subs[(subs["plan"] == plan) & (subs["onboarded"] == onb)]
            kmf = KaplanMeierFitter().fit(g["tenure_months"], g["churned"])
            area[onb] = restricted_mean_survival_time(kmf, t=horizon)
        total += weight * (area[1] - area[0])
    return float(total)


def monthly_churn_rate(subs: pd.DataFrame, months: int = 24, by: str = "intro_offer"):
    """Cancellations in month k divided by subscribers active (and observed) at the start of
    month k, k = 1..months, for each value of ``by``. A long table: ``by``, month, active,
    cancellations, rate."""
    rows = []
    for value, g in subs.groupby(by):
        t, d = g["tenure_months"].to_numpy(), g["churned"].to_numpy()
        for k in range(1, months + 1):
            active = int((t > k - 1).sum())
            cancels = int(((t > k - 1) & (t <= k) & (d == 1)).sum())
            rows.append({by: value, "month": k, "active": active, "cancellations": cancels,
                         "rate": cancels / active if active else np.nan})
    return pd.DataFrame(rows)


def dashboard_rates(subs: pd.DataFrame, months: float = 12.0) -> tuple[float, float]:
    """Two retention rates that ignore censoring: (A) the share of all subscribers who did not
    cancel within ``months``; (B) the same share among those whose status at ``months`` is
    known (cancelled by then, or observed for longer)."""
    cancelled = (subs["churned"] == 1) & (subs["tenure_months"] <= months)
    beyond = subs["tenure_months"] > months
    known = int((cancelled | beyond).sum())
    if known == 0:
        raise ValueError(
            f"no subscriber's status at {months:g} months is known (none cancelled by then or"
            " was observed for longer), so dashboard B has no denominator"
        )
    a = 1 - float(cancelled.mean())
    b = float(beyond.sum()) / known
    return a, b


def checkpoint_outcomes(subs: pd.DataFrame, truth: dict, B: int = 100, seed: int = 14) -> dict:
    """Every truth-based checkpoint rule of the Module 14 lab, applied with the reference
    solutions to one dataset (``subs`` with tenure, churn and plan dummies; ``truth`` from the
    generator). Returns the rules' outcomes and the numbers behind them."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
    from lifelines.utils import restricted_mean_survival_time

    hr = truth["model"]["hazard_ratios"]
    offer = truth["model"]["intro_offer"]
    horizon = truth["rmst"]["horizon_months"]
    out: dict = {}

    # Exercise 1's provided check: truth inside the KM's 95% pointwise interval; dashboards out.
    kmf = KaplanMeierFitter(alpha=0.05).fit(subs["tenure_months"], subs["churned"])
    inside = {}
    for t in (6, 12, 18, 24):
        lo, hi = kmf.confidence_interval_.asof(float(t)).to_numpy()
        inside[t] = bool(lo <= truth["survival"]["overall"][str(t)] <= hi)
    out["km_inside"] = sum(inside.values())
    out["km_rule"] = out["km_inside"] >= 3
    out["km_s12_inside"] = inside[12]
    lo12, hi12 = kmf.confidence_interval_.asof(12.0).to_numpy()
    a, b = dashboard_rates(subs)
    out["dashboards"] = (a, b)
    out["dashboards_rule"] = not (lo12 <= a <= hi12) and not (lo12 <= b <= hi12)

    # Exercise 3: adjusted and unadjusted Cox.
    adj = cox_hazard_ratios(subs, COVARIATES)
    truths = {"plan_standard": hr["plan_standard"], "plan_premium": hr["plan_premium"],
              "onboarded": hr["onboarded"]}
    inside3 = {k: bool(adj.loc[k, "lower"] <= v <= adj.loc[k, "upper"]) for k, v in truths.items()}
    out["cox_adjusted"] = adj
    out["cox_inside"] = sum(inside3.values())
    out["cox_rule"] = inside3["onboarded"] and out["cox_inside"] >= 2
    una = cox_hazard_ratios(subs, ["onboarded"])
    out["cox_unadjusted"] = una
    out["cox_unadjusted_rule"] = (
        not (una.loc["onboarded", "lower"] <= hr["onboarded"] <= una.loc["onboarded", "upper"])
        and una.loc["onboarded", "hr"] < adj.loc["onboarded", "hr"]
    )

    # The provided PH test on the full-period fit.
    from lifelines import CoxPHFitter

    frame = subs[COVARIATES + ["tenure_months", "churned"]]
    cph = CoxPHFitter(alpha=0.05).fit(frame, duration_col="tenure_months", event_col="churned")
    p = proportional_hazard_test(cph, frame, time_transform="rank").summary["p"]
    out["ph_p"] = p
    out["ph_offer_flagged"] = bool(p["intro_offer"] < 0.01)
    out["ph_others_flagged"] = int((p.drop("intro_offer") < 0.01).sum())

    # Exercise 4: landmark at 3 months.
    before, after = landmark_split(subs, offer["landmark_months"])
    hb, ha = cox_hazard_ratios(before, COVARIATES), cox_hazard_ratios(after, COVARIATES)
    four = {
        ("before", "intro_offer"): offer["hr_before"],
        ("after", "intro_offer"): offer["hr_after"],
        ("before", "onboarded"): hr["onboarded"],
        ("after", "onboarded"): hr["onboarded"],
    }
    tables = {"before": hb, "after": ha}
    inside4 = {k: bool(tables[k[0]].loc[k[1], "lower"] <= v <= tables[k[0]].loc[k[1], "upper"])
               for k, v in four.items()}
    out["landmark"] = {"before": hb, "after": ha}
    out["landmark_inside"] = inside4
    out["landmark_rule"] = sum(inside4.values()) >= 3

    # Exercise 5: RMST gain standardized to the target population, stratified bootstrap.
    target = subs[subs["onboarded"] == 0]
    weights = target["plan"].value_counts(normalize=True).to_dict()
    boot = bootstrap_interval(lambda d: rmst_gain(d, horizon, weights), subs,
                              by=["plan", "onboarded"], B=B, seed=seed)
    unadjusted = {}
    for onb in (1, 0):
        g = subs[subs["onboarded"] == onb]
        unadjusted[onb] = restricted_mean_survival_time(
            KaplanMeierFitter().fit(g["tenure_months"], g["churned"]), t=horizon)
    out["rmst"] = boot
    out["rmst_unadjusted"] = unadjusted[1] - unadjusted[0]
    out["rmst_rule"] = boot.lower <= truth["rmst"]["gain_target"] <= boot.upper
    out["rmst_unadjusted_rule"] = not (boot.lower <= out["rmst_unadjusted"] <= boot.upper)
    return out
