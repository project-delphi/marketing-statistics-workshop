"""Notifications for Module 18: the two logged experiments the lab runs on the simulated
users, reference implementations of the lab's functions, the policy machinery (bandit loop,
transition model, evaluation on common random numbers) and the checkpoint checks (each
failure message says what failed, the likely cause, what to check and how to recover).

Data: ``mktstats.synth.notifications`` (the users, the simulator, the fixed and random
policies). Only the users are committed; the logs below are generated live with the seeds in
this file, so every learner and every run record sees the same experiment.

See briefs/18-notifications.md for the notation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from mktstats.synth import notifications as env
from mktstats.synth._common import make_rng

# Seeds of the live-generated experiments and runs (visible in the lab).
SEED_SEND_SIM = 2041       # the simulator during the send experiment
SEED_SEND_POLICY = 2042    # the logging policy's own draws (which user-days get a send)
SEED_VOLUME_ARMS = 2039    # the volume experiment's assignment of users to cadences
SEED_VOLUME_SIM = 2040     # the simulator during the volume experiment
SEED_EVAL = 2043           # the simulator when policies are compared (common random numbers)
SEED_BANDIT = 2044         # the bandit's own posterior draws
SEND_DAYS = 28
P_SEND = 0.3
VOLUME_DAYS = 90
HORIZON = 90
FORWARD_DAYS = 7
QUICK_USERS = 5_000
# Context buckets for the bandit and the RL state: segment x fatigue x days since a session.
FATIGUE_EDGES = (0.5, 1.0, 1.5, 2.0, 3.0)
DAYS_SINCE_EDGES = (2, 7)
N_FATIGUE = len(FATIGUE_EDGES) + 1
N_DAYS = len(DAYS_SINCE_EDGES) + 1
N_SEGMENTS = len(env.SEGMENTS)
N_CONTEXTS = N_SEGMENTS * N_FATIGUE * N_DAYS
OUT_STATE = N_CONTEXTS  # the absorbing state of a user who opted out
N_ACTIONS = len(env.ACTIONS)
Z95 = float(stats.norm.ppf(0.975))

LOG_INT8 = ("eligible", "action", "sent", "opened", "session", "purchase", "opted_out_today",
            "session_if_send", "session_if_skip")


def _compact(log: pd.DataFrame) -> pd.DataFrame:
    """Smaller dtypes for a user-day log (an int8 flag, a float32 dollar)."""
    out = log.copy()
    for c in LOG_INT8:
        out[c] = out[c].astype(np.int8)
    for c in ("margin", "margin_if_send", "margin_if_skip", "fatigue_before"):
        out[c] = out[c].astype(np.float32)
    out["days_since_before"] = out["days_since_before"].astype(np.int16)
    out["day"] = out["day"].astype(np.int16)
    out["segment"] = out["segment"].astype("category")
    return out


# ------------------------------------------------------------------------- the experiments
def send_experiment(users: pd.DataFrame, *, days: int = SEND_DAYS, p_send: float = P_SEND,
                    seed: int = SEED_SEND_SIM, policy_seed: int = SEED_SEND_POLICY) -> pd.DataFrame:
    """The send experiment: for ``days`` days each eligible user is sent a template chosen at
    random with probability ``p_send``, independently of everything. One row per user-day."""
    sim = env.Simulator(users, seed)
    out = sim.run(env.random_policy(p_send, policy_seed), days, log=True)
    return _compact(out.log)


def volume_experiment(users: pd.DataFrame, *, days: int = VOLUME_DAYS,
                      seed: int = SEED_VOLUME_SIM, arms_seed: int = SEED_VOLUME_ARMS,
                      log: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The volume experiment: each user is assigned a cadence (sends a week, one of
    ``env.CADENCES``) at random and receives the reminder on that cadence for ``days`` days.
    Returns the per-user table with its ``arm`` column and, with ``log=True``, the user-day
    log (the RL exercise's transitions)."""
    arms = env.volume_arms(len(users), arms_seed)
    sim = env.Simulator(users, seed)
    out = sim.run(env.volume_policy(arms), days, log=log)
    per_user = out.per_user.assign(arm=arms)
    return per_user, (_compact(out.log) if log else None)


def forward_margin(log: pd.DataFrame, horizon: int = FORWARD_DAYS) -> pd.Series:
    """Margin over days t .. t + horizon - 1 for each user-day row (NaN where the window runs
    past the end of the log), aligned with ``log``'s index."""
    wide = log.pivot(index="user_id", columns="day", values="margin").to_numpy(float)
    csum = np.concatenate([np.zeros((len(wide), 1)), np.cumsum(wide, axis=1)], axis=1)
    T = wide.shape[1]
    fwd = np.full((len(wide), T), np.nan)
    if T >= horizon:
        fwd[:, : T - horizon + 1] = csum[:, horizon:] - csum[:, : T - horizon + 1]
    row = np.searchsorted(np.sort(log["user_id"].unique()), log["user_id"].to_numpy())
    return pd.Series(fwd[row, log["day"].to_numpy()], index=log.index, name="margin_fwd")


# ------------------------------------------------------------------------------ references
def _diff_in_means(y1: np.ndarray, y0: np.ndarray) -> tuple[float, float]:
    est = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return est, se


def incremental_per_send(log: pd.DataFrame) -> pd.DataFrame:
    """Exercise 1. Same-day effect of a send on sessions and margin: the difference in means
    between sent and not-sent eligible user-days, with a 95% normal confidence interval.
    Index ``["sessions_per_send", "margin_per_send"]``; columns estimate, se, lo, hi."""
    live = log[log["eligible"] == 1]
    sent = live["sent"].to_numpy() == 1
    rows = {}
    for name, col in (("sessions_per_send", "session"), ("margin_per_send", "margin")):
        y = live[col].to_numpy(float)
        est, se = _diff_in_means(y[sent], y[~sent])
        rows[name] = {"estimate": est, "se": se, "lo": est - Z95 * se, "hi": est + Z95 * se}
    return pd.DataFrame(rows).T[["estimate", "se", "lo", "hi"]]


VOLUME_COLUMNS = ("users", "sends", "open_rate", "opens_week_1", "opt_out", "margin_7d",
                  "margin_90d")


def volume_curve(per_user: pd.DataFrame, opens_week_1: pd.Series | None = None) -> pd.DataFrame:
    """Exercise 2. One row per cadence (``arm``): users, sends per user, open rate (opens over
    sends), opens per user in week 1, opt-out share, margin per user over the first 7 days and
    over the run (``margin``, 90 days in the lab). ``opens_week_1`` (per user, aligned with
    ``per_user``) is provided by the lab; without it the column is NaN."""
    g = per_user.groupby("arm")
    out = pd.DataFrame({
        "users": g.size(),
        "sends": g["sends"].mean(),
        "open_rate": g["opens"].sum() / g["sends"].sum().replace(0, np.nan),
        "opens_week_1": (opens_week_1.groupby(per_user["arm"].to_numpy()).mean()
                         if opens_week_1 is not None else np.nan),
        "opt_out": g["opted_out"].mean(),
        "margin_7d": g["margin_7d"].mean(),
        "margin_90d": g["margin"].mean(),
    })
    out["open_rate"] = out["open_rate"].fillna(0.0)
    return out[list(VOLUME_COLUMNS)]


def segment_uplift(log: pd.DataFrame, margin_fwd: pd.Series, value_at_stake: pd.Series,
                   cost_per_send: float = 0.0) -> pd.DataFrame:
    """Exercise 3. Per segment: the effect of a send on margin over the next FORWARD_DAYS days
    (difference in means between sent and not-sent eligible user-days with a complete window,
    95% normal CI), the opt-out probability per send, the expected cost of a send (that
    probability times the segment's value at stake, plus the channel's cost per send) and
    the rule ``send = cate > cost``. Index ``env.SEGMENTS``."""
    live = log[(log["eligible"] == 1) & margin_fwd.notna()]
    y = margin_fwd.loc[live.index].to_numpy(float)
    sent = live["sent"].to_numpy() == 1
    rows = {}
    for s in env.SEGMENTS:
        m = (live["segment"].to_numpy() == s)
        est, se = _diff_in_means(y[m & sent], y[m & ~sent])
        p_out = float(live.loc[m & sent, "opted_out_today"].mean())
        cost = p_out * float(value_at_stake[s]) + cost_per_send
        rows[s] = {"cate": est, "se": se, "lo": est - Z95 * se, "hi": est + Z95 * se,
                   "p_optout": p_out, "cost": cost, "send": est > cost}
    out = pd.DataFrame(rows).T
    for c in ("cate", "se", "lo", "hi", "p_optout", "cost"):
        out[c] = out[c].astype(float)
    out["send"] = out["send"].astype(bool)
    return out


def thompson_choose(sums: np.ndarray, counts: np.ndarray, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Exercise 4. One Thompson draw per context: for every context and arm draw
    Normal(mean, sigma / sqrt(n + 1)) with mean = sums / (n + 1) (a prior of one pseudo-
    observation of 0 reward), and choose the arm with the largest draw. Returns an integer
    array of length ``sums.shape[0]``."""
    sums = np.asarray(sums, float)
    counts = np.asarray(counts, float)
    draws = rng.normal(sums / (counts + 1), sigma / np.sqrt(counts + 1))
    return draws.argmax(axis=1)


def q_iteration(P: np.ndarray, R: np.ndarray, gamma: float, n_iter: int = 1000) -> np.ndarray:
    """Exercise 5. Value iteration on a known model: Q <- R + gamma * P @ max_a' Q, starting
    from zeros, ``n_iter`` sweeps. ``P[s, a, s']`` are transition probabilities, ``R[s, a]``
    expected one-day rewards. Returns Q of shape ``(S, A)``."""
    Q = np.zeros(R.shape, dtype=float)
    for _ in range(n_iter):
        Q = R + gamma * (P @ Q.max(axis=1))
    return Q


# -------------------------------------------------------------------------- contexts, bandit
def context_index(segment: np.ndarray, fatigue: np.ndarray, days_since: np.ndarray) -> np.ndarray:
    """The context (bandit) or state (RL) of each user: segment x fatigue bucket x days-
    since-session bucket, as one integer in 0 .. N_CONTEXTS - 1."""
    f = np.digitize(np.asarray(fatigue, float), FATIGUE_EDGES)
    d = np.digitize(np.asarray(days_since, float), DAYS_SINCE_EDGES)
    return (np.asarray(segment, int) * N_FATIGUE + f) * N_DAYS + d


def context_labels() -> pd.DataFrame:
    """One row per context: segment, fatigue bucket, days-since bucket (for printing)."""
    f_lab = [f"<{FATIGUE_EDGES[0]}"] + [f"{a}-{b}" for a, b in zip(FATIGUE_EDGES[:-1],
                                                                   FATIGUE_EDGES[1:],
                                                                   strict=True)]
    f_lab.append(f">={FATIGUE_EDGES[-1]}")
    d_lab = [f"<{DAYS_SINCE_EDGES[0]}", f"{DAYS_SINCE_EDGES[0]}-{DAYS_SINCE_EDGES[1] - 1}",
             f">={DAYS_SINCE_EDGES[1]}"]
    rows = []
    for si, s in enumerate(env.SEGMENTS):
        for fi in range(N_FATIGUE):
            for di in range(N_DAYS):
                rows.append({"context": (si * N_FATIGUE + fi) * N_DAYS + di, "segment": s,
                             "fatigue": f_lab[fi], "days_since": d_lab[di]})
    return pd.DataFrame(rows).set_index("context")


def sim_context(sim: env.Simulator) -> np.ndarray:
    return context_index(sim.segment, sim.fatigue, sim.days_since)


def run_bandit(users: pd.DataFrame, penalty: pd.Series | float, *, days: int = HORIZON,
               seed: int = SEED_EVAL, bandit_seed: int = SEED_BANDIT, sigma: float = 3.0,
               choose: Callable = thompson_choose) -> tuple[env.Outcome, pd.DataFrame]:
    """A contextual Thompson-sampling bandit run for ``days`` days: each day one arm per
    context (skip, reminder, offer, social) chosen by ``choose(sums, counts, sigma, rng)``,
    applied to every user in that context; the reward of a user-day is its margin minus
    ``penalty`` (per segment, or one number) if the user opted out that day. Users already
    opted out take no part. Returns the run and the learned mean reward per context and arm."""
    sim = env.Simulator(users, seed)
    rng = make_rng(bandit_seed)
    pen = (np.asarray([float(penalty[s]) for s in env.SEGMENTS])
           if isinstance(penalty, pd.Series | dict) else np.full(N_SEGMENTS, float(penalty)))
    sums = np.zeros((N_CONTEXTS, N_ACTIONS))
    counts = np.zeros((N_CONTEXTS, N_ACTIONS))
    results = []
    for _ in range(days):
        ctx = sim_context(sim)
        arm = np.asarray(choose(sums, counts, sigma, rng), int)[ctx]
        r = sim.step(arm)
        reward = r.margin - pen[sim.segment] * r.opted_out_today
        live = r.eligible
        np.add.at(sums, (ctx[live], r.action[live]), reward[live])
        np.add.at(counts, (ctx[live], r.action[live]), 1)
        results.append(r)
    means = pd.DataFrame(sums / np.maximum(counts, 1), columns=env.ACTIONS)
    means["n"] = counts.sum(axis=1).astype(int)
    return env.Outcome.from_days(sim, results, log=False), means


# ------------------------------------------------------------------------------- offline RL
def transition_model(log: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From a user-day log (the volume experiment's): empirical ``P[s, a, s']``, mean reward
    ``R[s, a]`` (margin today) and visit counts ``N[s, a]`` over states 0 .. N_CONTEXTS (the
    last one is the absorbing opted-out state) and actions skip (0) and reminder (1). A pair
    never visited keeps P = 0 and R = 0; ``greedy_policy`` never chooses it."""
    seg = log["segment"].map({s: i for i, s in enumerate(env.SEGMENTS)}).to_numpy(int)
    s = np.where(log["eligible"].to_numpy() == 1,
                 context_index(seg, log["fatigue_before"].to_numpy(),
                               log["days_since_before"].to_numpy()), OUT_STATE)
    order = np.lexsort((log["day"].to_numpy(), log["user_id"].to_numpy()))
    s, a = s[order], log["action"].to_numpy(int)[order]
    user, margin = log["user_id"].to_numpy()[order], log["margin"].to_numpy(float)[order]
    has_next = np.r_[user[1:] == user[:-1], False]
    keep = has_next & (a <= 1)
    s_from, a_from, s_to, r = s[keep], a[keep], np.r_[s[1:], -1][keep], margin[keep]
    S, A = N_CONTEXTS + 1, 2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    N = np.zeros((S, A))
    np.add.at(P, (s_from, a_from, s_to), 1.0)
    np.add.at(R, (s_from, a_from), r)
    np.add.at(N, (s_from, a_from), 1.0)
    P = P / np.maximum(N, 1)[:, :, None]
    R = R / np.maximum(N, 1)
    # Opted out: a send is impossible, so the send action is the skip action.
    P[OUT_STATE, 1], R[OUT_STATE, 1] = P[OUT_STATE, 0], R[OUT_STATE, 0]
    N[OUT_STATE, 1] = N[OUT_STATE, 0]
    return P, R, N


def with_send_penalty(R: np.ndarray, penalty: float) -> np.ndarray:
    """Horizon's device: the reward of a send is reduced by ``penalty`` dollars, so the
    policy must earn it back in margin or in later sessions. Returns a new R."""
    out = np.array(R, dtype=float)
    out[:, 1] -= penalty
    return out


def greedy_policy(Q: np.ndarray, N: np.ndarray, min_visits: int = 30):
    """The policy that takes the best action by Q in each state, among the actions the data
    tried at least ``min_visits`` times in that state (an action never tried there is not
    chosen: the model knows nothing about it). Skip where no action qualifies."""
    masked = np.where(N >= min_visits, Q, -np.inf)
    best = np.where(np.isfinite(masked).any(axis=1), masked.argmax(axis=1), 0)

    def policy(sim: env.Simulator) -> np.ndarray:
        return np.where(sim.opted_out, 0, best[sim_context(sim)])

    policy.__name__ = "offline_rl"
    return policy, best


# ------------------------------------------------------------------------------ evaluation
def evaluate(users: pd.DataFrame, policies: dict[str, Callable], *, days: int = HORIZON,
             seed: int = SEED_EVAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every policy for ``days`` days from the same seed (common random numbers) and
    return a summary per policy and the per-user margins (one column per policy)."""
    margins, rows = {}, {}
    for name, policy in policies.items():
        out = env.Simulator(users, seed).run(policy, days)
        pu = out.per_user
        margins[name] = pu["margin"].to_numpy()
        rows[name] = summarize(pu, days)
    return pd.DataFrame(rows).T, pd.DataFrame(margins)


def summarize(per_user: pd.DataFrame, days: int = HORIZON) -> dict[str, float]:
    n = len(per_user)
    sends = per_user["sends"].sum()
    return {
        "sends_per_user": per_user["sends"].mean(),
        "open_rate": per_user["opens"].sum() / sends if sends else 0.0,
        "opt_out": per_user["opted_out"].mean(),
        "uninstalled": per_user["uninstalled"].mean(),
        "margin_7d": per_user["margin_7d"].mean(),
        f"margin_{days}d": per_user["margin"].mean(),
        f"margin_{days}d_per_1000": 1000 * per_user["margin"].mean(),
        "se_margin": per_user["margin"].std(ddof=1) / np.sqrt(n),
    }


def paired_difference(margins: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    """Mean per-user margin difference a - b with its 95% normal CI and the share of users
    for whom a was at least as good (the runs share random numbers, so pairs are matched)."""
    d = margins[a].to_numpy() - margins[b].to_numpy()
    est, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"difference": est, "se": se, "lo": est - Z95 * se, "hi": est + Z95 * se,
            "p_value": float(2 * stats.norm.sf(abs(est / se))) if se > 0 else 0.0}


def value_at_stake(per_user: pd.DataFrame, cadence: int = 3) -> pd.Series:
    """What an opt-out puts at risk, per segment: the segment's mean margin over the run under
    the cadence ``cadence`` arm of the volume experiment (its best uniform cadence)."""
    return per_user[per_user["arm"] == cadence].groupby("segment")["margin"].mean().reindex(
        env.SEGMENTS)


def opens_week_1(users: pd.DataFrame, *, seed: int = SEED_VOLUME_SIM,
                 arms_seed: int = SEED_VOLUME_ARMS) -> pd.Series:
    """Opens per user in the first 7 days of the volume experiment (the same run, replayed
    for 7 days: the simulator is deterministic in its seed)."""
    arms = env.volume_arms(len(users), arms_seed)
    out = env.Simulator(users, seed).run(env.volume_policy(arms), 7)
    return out.per_user["opens"]


# ------------------------------------------------------------------------------- the checks
def _recover(n: int) -> str:
    return f" To move on now, run workshop.use_reference({n}) and come back to this later."


def _frame(obj, what: str, n: int) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise AssertionError(
            f"Checkpoint {n}: {what} returned a {type(obj).__name__}, not a pandas DataFrame."
            " Check the return statement." + _recover(n))
    return obj


def _need_columns(df: pd.DataFrame, cols: tuple, what: str, n: int) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(
            f"Checkpoint {n}: {what} is missing the column(s) {missing}; it needs"
            f" {list(cols)}." + _recover(n))


def check_incremental_per_send(fn: Callable, log: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """Exercise 1: the structure, agreement with the reference, and that the 95% interval
    covers the true same-day effect (truth from the potential outcomes in the log)."""
    got = _frame(fn(log), "incremental_per_send(log)", 1)
    _need_columns(got, ("estimate", "se", "lo", "hi"), "incremental_per_send(log)", 1)
    for name in ("sessions_per_send", "margin_per_send"):
        if name not in got.index:
            raise AssertionError(
                f"Checkpoint 1: the result has no row {name!r}; it needs rows"
                " 'sessions_per_send' and 'margin_per_send'." + _recover(1))
    want = incremental_per_send(log)
    naive = log.groupby("sent")[["session", "margin"]].mean()
    for name, col in (("sessions_per_send", "session"), ("margin_per_send", "margin")):
        est, se = float(got.loc[name, "estimate"]), float(got.loc[name, "se"])
        ref = float(want.loc[name, "estimate"])
        all_rows = float(naive.loc[1, col] - naive.loc[0, col])
        if not np.isclose(est, ref, rtol=1e-5, atol=1e-6):
            hint = ""
            if abs(est - all_rows) < 1e-9:
                hint = (" This is the difference over all user-days: users who had already"
                        " opted out are in the not-sent group but could never be sent."
                        " Keep only rows with eligible == 1 before comparing.")
            raise AssertionError(
                f"Checkpoint 1: your {name} is {est:.4f}; the difference in means between"
                f" sent and not-sent eligible user-days is {ref:.4f}.{hint} Check the filter"
                " (eligible == 1), the groups (sent == 1 against sent == 0) and the column"
                f" ({col!r})." + _recover(1))
        if not np.isclose(se, float(want.loc[name, "se"]), rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"Checkpoint 1: your standard error for {name} is {se:.5f}; the reference is"
                f" {float(want.loc[name, 'se']):.5f}. Use sqrt(var1 / n1 + var0 / n0) with"
                " the sample variance (ddof=1) of each group." + _recover(1))
        lo, hi = float(got.loc[name, "lo"]), float(got.loc[name, "hi"])
        if abs(lo - (est - Z95 * se)) > 1e-6 or abs(hi - (est + Z95 * se)) > 1e-6:
            raise AssertionError(
                f"Checkpoint 1: your interval for {name} is [{lo:.4f}, {hi:.4f}]; a 95% normal"
                f" interval is estimate ± 1.96 se = [{est - Z95 * se:.4f},"
                f" {est + Z95 * se:.4f}]." + _recover(1))
        tv = float(truth[name])
        if not lo <= tv <= hi:
            raise AssertionError(
                f"Checkpoint 1: the true same-day effect {tv:.4f} ({name}) lies outside your"
                f" 95% interval [{lo:.4f}, {hi:.4f}]. The estimate matches the reference, so"
                " this is sampling chance: on the committed users and seeds it does not"
                " happen; if you changed the seed, users or days, read the notebook's note."
                + _recover(1))
    return got


def check_volume_curve(fn: Callable, per_user: pd.DataFrame, opens_w1: pd.Series) -> pd.DataFrame:
    """Exercise 2: structure, agreement with the reference, and the two claims the lab makes
    on the committed data: the daily arm has the most opens in week 1, the 3-a-week arm the
    highest 90-day margin."""
    got = _frame(fn(per_user, opens_w1), "volume_curve(per_user, opens_w1)", 2)
    _need_columns(got, VOLUME_COLUMNS, "volume_curve(...)", 2)
    want = volume_curve(per_user, opens_w1)
    missing = [a for a in want.index if a not in got.index]
    if missing:
        raise AssertionError(
            f"Checkpoint 2: the result has no row for cadence(s) {missing}; group by `arm` so"
            f" every cadence {list(want.index)} gets a row." + _recover(2))
    got = got.loc[want.index]
    for c in VOLUME_COLUMNS:
        g, w = got[c].to_numpy(float), want[c].to_numpy(float)
        if not np.allclose(g, w, atol=1e-6, rtol=1e-5, equal_nan=True):
            cause = {
                "open_rate": "open_rate is total opens divided by total sends in the arm (not"
                             " the mean of per-user rates, which would weight a user with one"
                             " send like one with sixty)",
                "sends": "sends is the mean of the per-user `sends` column",
                "opt_out": "opt_out is the share of users with opted_out True",
                "margin_7d": "margin_7d is the mean of the per-user `margin_7d` column",
                "margin_90d": "margin_90d is the mean of the per-user `margin` column",
                "opens_week_1": "opens_week_1 is the mean of `opens_w1` over the users in the"
                                " arm (group it by per_user['arm'])",
                "users": "users is the number of rows in the arm",
            }[c]
            i = int(np.argmax(~np.isclose(g, w, atol=1e-6, rtol=1e-5, equal_nan=True)))
            raise AssertionError(
                f"Checkpoint 2: column {c!r} differs from the reference at cadence"
                f" {want.index[i]}: yours {g[i]:.4f}, reference {w[i]:.4f}. {cause}."
                + _recover(2))
    if int(got["opens_week_1"].idxmax()) != 7:
        raise AssertionError(
            "Checkpoint 2: on the committed data the daily arm has the most opens per user"
            f" in week 1; your table puts cadence {int(got['opens_week_1'].idxmax())} first."
            " Your numbers match the reference, so the data changed: check the users,"
            " the seeds and the number of days." + _recover(2))
    if int(got["margin_90d"].idxmax()) != 3:
        raise AssertionError(
            "Checkpoint 2: on the committed data the 3-a-week arm has the highest 90-day"
            f" margin per user; your table puts cadence {int(got['margin_90d'].idxmax())}"
            " first. Your numbers match the reference, so the data changed: check the users,"
            " the seeds and the number of days." + _recover(2))
    return got


def check_segment_uplift(fn: Callable, log: pd.DataFrame, margin_fwd: pd.Series,
                         value: pd.Series, cost_per_send: float = 0.0) -> pd.DataFrame:
    """Exercise 3: structure, agreement with the reference, and the claim that holds on the
    committed data in both modes: every segment's 7-day effect of a send is positive."""
    got = _frame(fn(log, margin_fwd, value, cost_per_send), "segment_uplift(...)", 3)
    cols = ("cate", "se", "lo", "hi", "p_optout", "cost", "send")
    _need_columns(got, cols, "segment_uplift(...)", 3)
    missing = [s for s in env.SEGMENTS if s not in got.index]
    if missing:
        raise AssertionError(
            f"Checkpoint 3: the result has no row for segment(s) {missing}; it needs one row"
            f" per segment {list(env.SEGMENTS)}." + _recover(3))
    got = got.loc[list(env.SEGMENTS)]
    want = segment_uplift(log, margin_fwd, value, cost_per_send)
    for s in env.SEGMENTS:
        if not np.isclose(float(got.loc[s, "cate"]), float(want.loc[s, "cate"]),
                          rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"Checkpoint 3: your 7-day effect for segment {s!r} is"
                f" {float(got.loc[s, 'cate']):.4f}; the reference is"
                f" {float(want.loc[s, 'cate']):.4f}. Keep eligible user-days with a complete"
                " window (margin_fwd not NaN), then take the difference in means of margin_fwd"
                " between sent and not-sent rows within the segment." + _recover(3))
        for c in ("se", "p_optout", "cost"):
            if not np.isclose(float(got.loc[s, c]), float(want.loc[s, c]), rtol=1e-5, atol=1e-6):
                cause = {
                    "se": "se is sqrt(var1 / n1 + var0 / n0) with ddof=1",
                    "p_optout": "p_optout is the mean of opted_out_today over the segment's"
                                " sent rows (the same rows as the effect)",
                    "cost": "cost is p_optout x value_at_stake[segment] + cost_per_send",
                }[c]
                raise AssertionError(
                    f"Checkpoint 3: {c} for segment {s!r} is {float(got.loc[s, c]):.5f}; the"
                    f" reference is {float(want.loc[s, c]):.5f}. {cause}." + _recover(3))
        if bool(got.loc[s, "send"]) != bool(want.loc[s, "send"]):
            raise AssertionError(
                f"Checkpoint 3: send for segment {s!r} is {bool(got.loc[s, 'send'])}; with"
                f" cate {float(want.loc[s, 'cate']):.4f} and cost {float(want.loc[s, 'cost']):.4f}"
                " the rule send = cate > cost gives the opposite." + _recover(3))
    if (got["cate"] <= 0).any():
        raise AssertionError(
            "Checkpoint 3: on the committed data every segment's 7-day effect of a send is"
            f" positive; yours are {got['cate'].round(4).to_dict()}. Your estimates match the"
            " reference, so the data changed: check the users, the seeds and the number of"
            " days." + _recover(3))
    return got


def check_thompson(fn: Callable) -> None:
    """Exercise 4: on a fixed table the function returns one arm per context, chooses the
    best-known arm most of the time where the means are far apart and the data plentiful,
    still explores where there are no data, and uses the rng (two seeds differ)."""
    sums = np.array([[0.0, 1000.0, 0.0, 0.0],   # arm 1 far ahead (mean 1 vs 0), n = 1000 each
                     [0.0, 0.0, 0.0, 0.0],      # nothing known: every arm should be tried
                     [1000.0, 0.0, 0.0, 0.0]])  # arm 0 far ahead, n = 1000 each
    counts = np.array([[1000.0] * 4, [0.0] * 4, [1000.0] * 4])
    rng = make_rng(0)
    first = fn(sums, counts, 3.0, rng)
    first = np.asarray(first)
    if first.shape != (3,):
        raise AssertionError(
            f"Checkpoint 4: thompson_choose returned shape {first.shape} for 3 contexts; it"
            " returns one arm per context (argmax over axis=1)." + _recover(4))
    if not np.issubdtype(first.dtype, np.integer):
        raise AssertionError(
            f"Checkpoint 4: thompson_choose returned values of type {first.dtype}; it returns"
            " integer arm indexes (the argmax), not the draws." + _recover(4))
    picks = np.array([np.asarray(fn(sums, counts, 3.0, rng)) for _ in range(2000)])
    share_1 = (picks[:, 0] == 1).mean()
    share_0 = (picks[:, 2] == 0).mean()
    if share_1 < 0.99 or share_0 < 0.99:
        raise AssertionError(
            f"Checkpoint 4: with 1,000 pulls per arm and means 1 against 0, the function"
            f" chose the best arm {share_1:.0%} (context 0) and {share_0:.0%} (context 2) of"
            " the time; it should be almost always. The posterior standard deviation is"
            " sigma / sqrt(n + 1): check that counts scale the spread, and that the mean is"
            " sums / (counts + 1)." + _recover(4))
    tried = np.unique(picks[:, 1])
    if len(tried) < 4:
        raise AssertionError(
            f"Checkpoint 4: in a context with no data the function only ever chose arms"
            f" {tried.tolist()}; with equal means and equal spread every arm should be drawn"
            " some of the time. Draw from Normal(mean, sigma / sqrt(n + 1)) with rng.normal,"
            " not the mean itself." + _recover(4))
    rng_a, rng_a2, rng_b = make_rng(1), make_rng(1), make_rng(2)
    a = np.array([np.asarray(fn(sums, counts, 3.0, rng_a)) for _ in range(50)])
    a2 = np.array([np.asarray(fn(sums, counts, 3.0, rng_a2)) for _ in range(50)])
    b = np.array([np.asarray(fn(sums, counts, 3.0, rng_b)) for _ in range(50)])
    if not np.array_equal(a[:, 1], a2[:, 1]) or np.array_equal(a[:, 1], b[:, 1]):
        raise AssertionError(
            "Checkpoint 4: two generators with the same seed must give the same 50 choices"
            " and two with different seeds different ones; yours do not. The function must"
            " draw from `rng`, not from a generator of its own or numpy's global state."
            + _recover(4))


TINY_P = np.array([[[1.0, 0.0], [0.0, 1.0]],    # state 0: skip stays, send -> state 1
                   [[1.0, 0.0], [0.0, 1.0]]])   # state 1: skip -> state 0, send stays
TINY_R = np.array([[0.0, 1.0], [0.0, -1.0]])
TINY_GAMMA = 0.5
TINY_Q = np.array([[2 / 3, 4 / 3], [2 / 3, -2 / 3]])


def check_q_iteration(fn: Callable) -> None:
    """Exercise 5: the two-state example worked by hand, and convergence (one more sweep
    changes nothing)."""
    Q = np.asarray(fn(TINY_P, TINY_R, TINY_GAMMA, 200), float)
    if Q.shape != (2, 2):
        raise AssertionError(
            f"Checkpoint 5: q_iteration returned shape {Q.shape} on a 2-state, 2-action"
            " example; it returns Q with one row per state and one column per action."
            + _recover(5))
    if not np.allclose(Q, TINY_Q, atol=1e-6):
        one_sweep = TINY_R + TINY_GAMMA * (TINY_P @ np.zeros(2))
        hint = ""
        if np.allclose(Q, one_sweep, atol=1e-9):
            hint = " This is a single sweep from zeros: repeat the update n_iter times."
        elif np.allclose(Q, TINY_R, atol=1e-9):
            hint = " This is R itself: add gamma times the expected value of the next state."
        raise AssertionError(
            f"Checkpoint 5: on the two-state example your Q is {np.round(Q, 4).tolist()}; by"
            f" hand it is {np.round(TINY_Q, 4).tolist()} (V = [4/3, 2/3] at gamma = 0.5).{hint}"
            " The update is Q <- R + gamma * P @ max over actions of Q, where P @ v gives,"
            " for each (state, action), the expected v of the next state." + _recover(5))
    Q2 = np.asarray(fn(TINY_P, TINY_R, TINY_GAMMA, 201), float)
    if not np.allclose(Q, Q2, atol=1e-6):
        raise AssertionError(
            "Checkpoint 5: 200 and 201 sweeps give different Q on the two-state example;"
            " value iteration with gamma = 0.5 converges in far fewer. Check that each sweep"
            " uses the previous sweep's Q, not the initial zeros." + _recover(5))
