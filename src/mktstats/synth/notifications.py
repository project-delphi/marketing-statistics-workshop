"""A simulated population of app users who receive notifications (Module 18, notifications;
briefs/18-notifications.md).

The retailer's app has ``n_users`` users. Each day a **policy** decides, for every user who has
not opted out, whether to send a push notification and which of three templates: a
``reminder`` ("your basket is waiting"), an ``offer`` (10% off today) or ``social`` ("3 people
near you bought this today"). The user may open it, may visit the app on their own, may buy,
and may opt out of notifications. Everything below is an assumption of this synthetic world,
chosen so that the mechanisms a growth team argues about are all present and measurable:

1. **Engagement** theta_i ~ Normal(0, 1) (hidden) drives opens, organic visits and purchases.
   Three observed features are noisy readings of it: ``sessions_last_30`` ~ Poisson(exp(1.0 +
   0.6 theta)), ``purchases_last_90`` ~ Poisson(exp(-0.3 + 0.5 theta)) and ``tenure_days``
   (independent of theta). ``segment`` is a deterministic bucket of ``sessions_last_30``
   (low 0-1, mid 2-4, high 5+), so a policy can condition on it.
2. **Fatigue** F_i,t decays by RHO a day and rises by 1 with every send:
   F_{t+1} = RHO F_t + s_t. Fatigue lowers the chance of opening and raises the chance of
   opting out. Sensitivity kappa_i = exp(KAPPA_THETA theta_i + Normal(0, KAPPA_SIGMA)) (hidden)
   scales the opt-out slope: engaged users tolerate more notifications.
3. **A send** is opened with probability expit(OPEN_ALPHA[template] + OPEN_THETA theta -
   OPEN_FATIGUE F). An opened push is a session. After every send the user opts out with
   probability expit(OPTOUT_BASE + OPTOUT_SLOPE kappa F), F measured before the send is added.
   Opting out is absorbing: no more sends. UNINSTALL_SHARE of opt-outs are uninstalls, after
   which the user never visits again (no organic sessions).
4. **Habit.** Without a push the user visits with probability expit(ORGANIC_BASE +
   ORGANIC_THETA theta - ORGANIC_DECAY min(d, 30)), where d is the number of days since the
   last session. A session (opened push or organic visit) resets d to 0. This is the long-term
   value of a notification: a session today keeps the habit alive, so it raises organic visits
   on later days. A policy that never sends leaves that value unrealized.
5. **Purchase.** In a session the user buys with probability expit(BUY_BASE + BUY_THETA theta),
   plus BUY_OFFER on the logit scale in a session opened from an ``offer``. An order is ORDER
   dollars at GROSS_MARGIN; an order bought from an offer carries OFFER_DISCOUNT less margin.

The simulator draws the same uniforms (open, opt-out, uninstall, organic visit, purchase) for
every user on every day whatever the policy does, so two policies run with the same seed are
compared on **common random numbers**: a user who would have visited organically on day 12
under one policy visits under the other too. Potential outcomes of *today's* action
(``margin_if_send``, ``margin_if_skip``, ``session_if_send``, ``session_if_skip``, for the
reminder template) are computed from those uniforms and logged, which is what makes the true
same-day effect of a send known.

Only ``users`` is committed (``notifications_users.csv``); the daily logs are generated live
in the lab from the committed users with a visible seed, because a 28-day log of 20,000 users
is 560,000 rows. ``truth_notifications.json`` carries the parameters, the true same-day effect
of a send on the logging experiment and the measured reference results of the lab's policies
(``scripts/make_synthetic.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from mktstats.synth._common import SynthResult, make_rng, sig, sig_dict

TEMPLATES = ("reminder", "offer", "social")
SEGMENTS = ("low", "mid", "high")
SEGMENT_EDGES = (2, 5)  # sessions_last_30: low < 2 <= mid < 5 <= high
ACTIONS = ("skip", *TEMPLATES)  # action index 0 = skip, 1..3 = send that template

# Fatigue
RHO = 0.7
# Opens: logit P(open) = OPEN_ALPHA[template] + OPEN_THETA theta - OPEN_FATIGUE F
OPEN_ALPHA = {"reminder": -2.2, "offer": -1.8, "social": -2.0}
OPEN_THETA = 0.8
OPEN_FATIGUE = 0.5
# Opt-out after a send: logit P(opt out) = OPTOUT_BASE + OPTOUT_SLOPE kappa F
OPTOUT_BASE = -8.0
OPTOUT_SLOPE = 0.8
# kappa = exp(KAPPA_THETA theta + Normal(0, KAPPA_SIGMA)): engaged users tolerate more
KAPPA_THETA = -0.4
KAPPA_SIGMA = 0.5
UNINSTALL_SHARE = 0.5
# Organic visit: logit P(visit) = ORGANIC_BASE + ORGANIC_THETA theta - ORGANIC_DECAY min(d, 30)
ORGANIC_BASE = -2.2
ORGANIC_THETA = 0.9
ORGANIC_DECAY = 0.08
MAX_DAYS_SINCE = 30
# Purchase in a session: logit P(buy) = BUY_BASE + BUY_THETA theta (+ BUY_OFFER from an offer)
BUY_BASE = -1.6
BUY_THETA = 0.4
BUY_OFFER = 0.6
ORDER = 30.0
GROSS_MARGIN = 0.30
OFFER_DISCOUNT = 3.0  # 10% of the order, paid out of margin

DESCRIPTION = (
    "n_users app users. Engagement theta ~ Normal(0, 1) and fatigue sensitivity kappa = "
    "exp(KAPPA_THETA theta + Normal(0, KAPPA_SIGMA)) are hidden; sessions_last_30 ~ "
    "Poisson(exp(1 + 0.6 theta)), "
    "purchases_last_90 ~ Poisson(exp(-0.3 + 0.5 theta)) and tenure_days are observed, and "
    "segment buckets sessions_last_30 (low 0-1, mid 2-4, high 5+). Each day a policy sends one "
    "of three templates or skips. Fatigue F_{t+1} = RHO F_t + sent_t. A send is opened with "
    "probability expit(OPEN_ALPHA[template] + OPEN_THETA theta - OPEN_FATIGUE F); an open is a "
    "session. After a send the user opts out with probability expit(OPTOUT_BASE + OPTOUT_SLOPE "
    "kappa F) (F before the send), for good; UNINSTALL_SHARE of opt-outs uninstall and never "
    "visit again. Without an open the user visits with probability expit(ORGANIC_BASE + "
    "ORGANIC_THETA theta - ORGANIC_DECAY min(days_since_session, 30)); any session resets "
    "days_since_session to 0. In a session the user buys with probability expit(BUY_BASE + "
    "BUY_THETA theta + BUY_OFFER [offer]); an order is ORDER x GROSS_MARGIN of margin, "
    "OFFER_DISCOUNT less when bought from an offer. The same uniforms are drawn every day "
    "whatever the policy does (common random numbers across policies at one seed)."
)


def margin_per_order() -> float:
    return ORDER * GROSS_MARGIN


def segment_of(sessions_last_30: np.ndarray | pd.Series) -> np.ndarray:
    """The observed segment of each user: an integer 0 (low), 1 (mid), 2 (high)."""
    s = np.asarray(sessions_last_30)
    return np.digitize(s, SEGMENT_EDGES)


def notification_users(seed: int = 2038, *, n_users: int = 20_000) -> SynthResult:
    """The committed population: one row per user, in a random order so any prefix is a
    random subsample (QUICK runs use the first 5,000).

    Table ``users``: ``user_id`` (1..n_users), ``tenure_days``, ``sessions_last_30``,
    ``purchases_last_90``, ``segment`` (low/mid/high), ``days_since_session`` at the start,
    and the hidden ``true_theta`` and ``true_kappa`` (4 decimals; checkpoints only).
    """
    rng = make_rng(seed)
    theta = rng.standard_normal(n_users)
    kappa = np.exp(KAPPA_THETA * theta + rng.normal(0.0, KAPPA_SIGMA, n_users))
    tenure = rng.integers(30, 721, n_users)
    sessions = rng.poisson(np.exp(1.0 + 0.6 * theta))
    purchases = rng.poisson(np.exp(-0.3 + 0.5 * theta))
    # Days since the last session at the start: geometric in the user's own visit rate.
    p_visit = np.clip((sessions + 1) / 31.0, 0.02, 0.95)
    days_since = np.minimum(rng.geometric(p_visit) - 1, MAX_DAYS_SINCE)
    seg = segment_of(sessions)
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "tenure_days": tenure,
        "sessions_last_30": sessions,
        "purchases_last_90": purchases,
        "segment": np.array(SEGMENTS)[seg],
        "days_since_session": days_since,
        "true_theta": np.round(theta, 4),
        "true_kappa": np.round(kappa, 4),
    })
    truth: dict[str, Any] = {
        "seed": seed,
        "n_users": n_users,
        "templates": list(TEMPLATES),
        "actions": list(ACTIONS),
        "segments": list(SEGMENTS),
        "segment_edges": list(SEGMENT_EDGES),
        "dgp": {
            "description": DESCRIPTION,
            "rho": RHO,
            "open_alpha": OPEN_ALPHA,
            "open_theta": OPEN_THETA,
            "open_fatigue": OPEN_FATIGUE,
            "optout_base": OPTOUT_BASE,
            "optout_slope": OPTOUT_SLOPE,
            "kappa_theta": KAPPA_THETA,
            "kappa_sigma": KAPPA_SIGMA,
            "uninstall_share": UNINSTALL_SHARE,
            "organic_base": ORGANIC_BASE,
            "organic_theta": ORGANIC_THETA,
            "organic_decay": ORGANIC_DECAY,
            "max_days_since": MAX_DAYS_SINCE,
            "buy_base": BUY_BASE,
            "buy_theta": BUY_THETA,
            "buy_offer": BUY_OFFER,
            "order": ORDER,
            "gross_margin": GROSS_MARGIN,
            "offer_discount": OFFER_DISCOUNT,
            "margin_per_order": sig(margin_per_order()),
        },
        "segment_share": sig_dict({s: float((seg == i).mean()) for i, s in enumerate(SEGMENTS)}),
        "mean_theta_by_segment": sig_dict({s: float(theta[seg == i].mean())
                                           for i, s in enumerate(SEGMENTS)}),
        "open_rate_fresh_by_template": sig_dict({
            k: float(expit(OPEN_ALPHA[k] + OPEN_THETA * theta).mean()) for k in TEMPLATES}),
        "optout_per_send_at_fatigue": sig_dict({
            str(f): float(expit(OPTOUT_BASE + OPTOUT_SLOPE * kappa * f).mean())
            for f in (0, 1, 2, 3)}),
        "steady_state_fatigue": {
            "note": "F after a long run of sends every k-th day: 1 / (1 - RHO^k) just after "
                    "a send",
            **{str(k): sig(1 / (1 - RHO ** k)) for k in (1, 2, 7)},
        },
    }
    return SynthResult(name="notifications", data={"users": users}, truth=truth)


# ------------------------------------------------------------------------------ simulator
@dataclass
class DayResult:
    """What happened to every user on one day (arrays of length n_users)."""

    day: int
    action: np.ndarray        # 0 skip (or blocked by an opt-out), 1..3 the template sent
    sent: np.ndarray          # bool
    opened: np.ndarray        # bool
    session: np.ndarray       # bool: opened or organic visit
    purchase: np.ndarray      # bool
    margin: np.ndarray        # dollars of margin today
    opted_out_today: np.ndarray
    uninstalled_today: np.ndarray
    fatigue_before: np.ndarray    # F when the action was taken
    days_since_before: np.ndarray
    eligible: np.ndarray          # bool: not opted out before today, so a send was possible
    # Potential outcomes of today's action for the reminder template, from the same uniforms.
    margin_if_send: np.ndarray
    margin_if_skip: np.ndarray
    session_if_send: np.ndarray
    session_if_skip: np.ndarray


class Simulator:
    """The environment. ``reset()`` then ``step(actions)`` once per day, or ``run(policy)``.

    State arrays (read them from the attributes before choosing actions): ``fatigue`` F,
    ``days_since`` d, ``opted_out``, ``uninstalled``, ``sends_last_7``, ``day``. The hidden
    ``theta`` and ``kappa`` are attributes too (checkpoints only; a policy must not read them).
    """

    def __init__(self, users: pd.DataFrame, seed: int):
        self.users = users.reset_index(drop=True)
        self.n = len(self.users)
        self.seed = int(seed)
        self.theta = self.users["true_theta"].to_numpy(float)
        self.kappa = self.users["true_kappa"].to_numpy(float)
        self.segment = segment_of(self.users["sessions_last_30"])
        self.reset()

    def reset(self) -> None:
        self.rng = make_rng(self.seed)
        self.day = 0
        self.fatigue = np.zeros(self.n)
        self.days_since = self.users["days_since_session"].to_numpy(float)
        self.opted_out = np.zeros(self.n, dtype=bool)
        self.uninstalled = np.zeros(self.n, dtype=bool)
        self._send_history = np.zeros((7, self.n), dtype=bool)  # ring buffer of the last 7 days
        self.sends_last_7 = np.zeros(self.n, dtype=int)

    def _potential(self, template: int, u_open, u_organic, u_buy):
        """Session and margin today if ``template`` is sent to everyone (not opted out),
        from the given uniforms. template 0 = skip."""
        if template == 0:
            opened = np.zeros(self.n, dtype=bool)
        else:
            name = TEMPLATES[template - 1]
            p_open = expit(OPEN_ALPHA[name] + OPEN_THETA * self.theta
                           - OPEN_FATIGUE * self.fatigue)
            opened = (u_open < p_open) & ~self.opted_out
        p_organic = expit(ORGANIC_BASE + ORGANIC_THETA * self.theta
                          - ORGANIC_DECAY * np.minimum(self.days_since, MAX_DAYS_SINCE))
        organic = (u_organic < p_organic) & ~self.uninstalled
        session = opened | organic
        offer = opened & (template == 2)
        p_buy = expit(BUY_BASE + BUY_THETA * self.theta + BUY_OFFER * offer)
        purchase = session & (u_buy < p_buy)
        margin = purchase * (margin_per_order() - OFFER_DISCOUNT * offer)
        return opened, session, purchase, margin

    def step(self, actions: np.ndarray) -> DayResult:
        """Apply one day's actions (0 skip, 1 reminder, 2 offer, 3 social) and advance."""
        a = np.asarray(actions, dtype=int)
        if a.shape != (self.n,):
            raise ValueError(f"actions must have shape ({self.n},), got {a.shape}")
        if a.min() < 0 or a.max() > len(TEMPLATES):
            raise ValueError(f"actions must be in 0..{len(TEMPLATES)} ({ACTIONS})")
        # The same five uniforms every day, whatever the actions (common random numbers).
        u_open = self.rng.random(self.n)
        u_optout = self.rng.random(self.n)
        u_uninstall = self.rng.random(self.n)
        u_organic = self.rng.random(self.n)
        u_buy = self.rng.random(self.n)

        eligible = ~self.opted_out
        a = np.where(eligible, a, 0)
        sent = a > 0
        fatigue_before = self.fatigue.copy()
        days_before = self.days_since.copy()

        opened = np.zeros(self.n, dtype=bool)
        for k in range(1, len(TEMPLATES) + 1):
            mask = a == k
            if mask.any():
                o, _, _, _ = self._potential(k, u_open, u_organic, u_buy)
                opened |= o & mask
        p_organic = expit(ORGANIC_BASE + ORGANIC_THETA * self.theta
                          - ORGANIC_DECAY * np.minimum(self.days_since, MAX_DAYS_SINCE))
        organic = (u_organic < p_organic) & ~self.uninstalled
        session = opened | organic
        offer = opened & (a == 2)
        p_buy = expit(BUY_BASE + BUY_THETA * self.theta + BUY_OFFER * offer)
        purchase = session & (u_buy < p_buy)
        margin = purchase * (margin_per_order() - OFFER_DISCOUNT * offer)

        p_optout = expit(OPTOUT_BASE + OPTOUT_SLOPE * self.kappa * self.fatigue)
        optout_today = sent & (u_optout < p_optout)
        uninstall_today = optout_today & (u_uninstall < UNINSTALL_SHARE)

        # Potential outcomes of today's action (reminder vs skip), same uniforms, same state.
        _, s1, _, m1 = self._potential(1, u_open, u_organic, u_buy)
        _, s0, _, m0 = self._potential(0, u_open, u_organic, u_buy)

        # Advance the state.
        self.fatigue = RHO * self.fatigue + sent
        self.days_since = np.where(session, 0.0, self.days_since + 1)
        self.opted_out |= optout_today
        self.uninstalled |= uninstall_today
        self._send_history[self.day % 7] = sent
        self.sends_last_7 = self._send_history.sum(axis=0)
        self.day += 1
        return DayResult(
            day=self.day - 1, action=a, sent=sent, opened=opened, session=session,
            purchase=purchase, margin=margin, opted_out_today=optout_today,
            uninstalled_today=uninstall_today, fatigue_before=fatigue_before,
            days_since_before=days_before, eligible=eligible,
            margin_if_send=m1, margin_if_skip=m0,
            session_if_send=s1, session_if_skip=s0,
        )

    def run(self, policy, days: int, *, log: bool = False) -> Outcome:
        """Run ``policy(sim) -> actions`` for ``days`` days from a fresh reset."""
        self.reset()
        results = []
        for _ in range(days):
            results.append(self.step(policy(self)))
        return Outcome.from_days(self, results, log=log)


@dataclass
class Outcome:
    """A policy's run: per-user totals, per-day totals and (optionally) the user-day log."""

    days: int
    per_user: pd.DataFrame
    per_day: pd.DataFrame
    log: pd.DataFrame | None = None

    @classmethod
    def from_days(cls, sim: Simulator, results: list[DayResult], *, log: bool) -> Outcome:
        n, T = sim.n, len(results)
        stack = {f: np.stack([getattr(r, f) for r in results])
                 for f in ("sent", "opened", "session", "purchase", "margin",
                           "opted_out_today", "uninstalled_today")}
        per_user = pd.DataFrame({
            "user_id": sim.users["user_id"].to_numpy(),
            "segment": sim.users["segment"].to_numpy(),
            "sends": stack["sent"].sum(axis=0),
            "opens": stack["opened"].sum(axis=0),
            "sessions": stack["session"].sum(axis=0),
            "purchases": stack["purchase"].sum(axis=0),
            "margin": stack["margin"].sum(axis=0),
            "margin_7d": stack["margin"][: min(7, T)].sum(axis=0),
            "opted_out": stack["opted_out_today"].any(axis=0),
            "uninstalled": stack["uninstalled_today"].any(axis=0),
        })
        per_day = pd.DataFrame({
            "day": np.arange(T),
            "sends": stack["sent"].sum(axis=1),
            "opens": stack["opened"].sum(axis=1),
            "sessions": stack["session"].sum(axis=1),
            "purchases": stack["purchase"].sum(axis=1),
            "margin": stack["margin"].sum(axis=1),
            "opt_outs": stack["opted_out_today"].sum(axis=1),
            "active": n - np.cumsum(stack["opted_out_today"].sum(axis=1)),
        })
        table = None
        if log:
            rows = {
                "user_id": np.tile(sim.users["user_id"].to_numpy(), T),
                "segment": np.tile(sim.users["segment"].to_numpy(), T),
                "day": np.repeat(np.arange(T), n),
            }
            for f in ("eligible", "action", "sent", "opened", "session", "purchase", "margin",
                      "opted_out_today", "fatigue_before", "days_since_before",
                      "margin_if_send", "margin_if_skip", "session_if_send",
                      "session_if_skip"):
                rows[f] = np.concatenate([getattr(r, f) for r in results])
            table = pd.DataFrame(rows)
            for c in ("eligible", "sent", "opened", "session", "purchase", "opted_out_today",
                      "session_if_send", "session_if_skip"):
                table[c] = table[c].astype(int)
            table["fatigue_before"] = table["fatigue_before"].round(4)
        return cls(days=T, per_user=per_user, per_day=per_day, log=table)


# ------------------------------------------------------------------------------ policies
# The weekdays (day % 7) on which a cadence of k sends a week sends.
WEEKDAYS = {0: (), 1: (0,), 2: (0, 3), 3: (0, 2, 4), 5: (0, 1, 2, 3, 4), 7: tuple(range(7))}
CADENCES = tuple(sorted(WEEKDAYS))


def fixed_policy(per_week: int, template: int = 1):
    """Send ``template`` on ``per_week`` fixed weekdays (0, 1, 2, 3, 5 or 7 a week)."""
    weekdays = WEEKDAYS
    if per_week not in weekdays:
        raise ValueError(f"per_week must be one of {sorted(weekdays)}, got {per_week}")
    send_days = weekdays[per_week]

    def policy(sim: Simulator) -> np.ndarray:
        if sim.day % 7 in send_days:
            return np.full(sim.n, template, dtype=int)
        return np.zeros(sim.n, dtype=int)

    policy.__name__ = f"fixed_{per_week}_per_week"
    return policy


def volume_arms(n_users: int, seed: int) -> np.ndarray:
    """The volume experiment's assignment: each user's cadence (sends a week), drawn at random
    from CADENCES with equal probability, from its own RNG."""
    rng = make_rng(seed)
    return np.array(CADENCES)[rng.integers(0, len(CADENCES), n_users)]


def volume_policy(arms: np.ndarray, template: int = 1):
    """The volume experiment: user i is sent ``template`` on the weekdays of cadence
    ``arms[i]`` (one of CADENCES) for the whole run."""
    arms = np.asarray(arms)
    bad = set(np.unique(arms)) - set(CADENCES)
    if bad:
        raise ValueError(f"arms must be cadences in {CADENCES}, got {sorted(bad)}")
    send_table = np.array([[d in WEEKDAYS[k] for d in range(7)] for k in CADENCES])
    arm_index = np.searchsorted(CADENCES, arms)

    def policy(sim: Simulator) -> np.ndarray:
        return np.where(send_table[arm_index, sim.day % 7], template, 0)

    policy.__name__ = "volume_experiment"
    return policy


def random_policy(p_send: float, seed: int, template: int | None = 1):
    """The logging policy of the send experiment: each day, each user is sent ``template``
    (the reminder by default; ``None`` draws one of the three at random) with probability
    ``p_send``, independently of everything (its own RNG)."""
    rng = make_rng(seed)

    def policy(sim: Simulator) -> np.ndarray:
        send = rng.random(sim.n) < p_send
        which = (rng.integers(1, len(TEMPLATES) + 1, sim.n) if template is None
                 else np.full(sim.n, int(template)))
        return np.where(send, which, 0)

    policy.__name__ = f"random_{p_send}"
    return policy


def segment_policy(per_week_by_segment: dict[str, int], template: int = 1):
    """A different fixed weekly cadence per observed segment (0 = never)."""
    weekdays = WEEKDAYS
    cadence = [per_week_by_segment.get(s, 0) for s in SEGMENTS]
    for k in cadence:
        if k not in weekdays:
            raise ValueError(f"cadences must be in {sorted(weekdays)}, got {per_week_by_segment}")

    def policy(sim: Simulator) -> np.ndarray:
        today = sim.day % 7
        send_seg = np.array([today in weekdays[k] for k in cadence])
        return np.where(send_seg[sim.segment], template, 0)

    policy.__name__ = "segment_" + "_".join(f"{s}{k}" for s, k in zip(SEGMENTS, cadence,
                                                                      strict=True))
    return policy


def true_same_day_effect(log: pd.DataFrame) -> dict[str, float]:
    """The true same-day effect of a reminder on the logged user-days: the mean of the
    potential-outcome differences (sessions and margin) over the eligible user-days."""
    live = log[log["eligible"] == 1]
    return {
        "sessions_per_send": sig(float((live["session_if_send"] - live["session_if_skip"]).mean())),
        "margin_per_send": sig(float((live["margin_if_send"] - live["margin_if_skip"]).mean())),
    }
