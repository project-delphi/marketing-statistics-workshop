"""Multi-touch attribution for Module 16: reference implementations of the lab's functions,
the checkpoint checks (each failure message says what failed, the likely cause, what to check and
how to recover), and the vectorized machinery behind the lab's bootstrap.

A journey is one row with ``path`` (touches in time order joined by ``" > "``) and ``converted``
(0/1). Every method hands out the converting journeys' conversions:

- heuristics: last-click, first-click, linear (each touch gets 1/L of its journey);
- Markov: a first-order chain over the states start, the channels, conv and null; the removal
  effect of channel c is 1 - p_{-c} / p, where p_{-c} stops every journey that reaches c
  (row c set to 0 and c -> null set to 1); credit_c = Y * RE_c / sum_k RE_k;
- Shapley: for each set S of channels that occurs on a converting journey, the game on S with
  v(T) = conversion rate of journeys whose set of channels is exactly T (v of the empty set = 0);
  each such conversion is split in proportion to phi_c(v; S).

See briefs/16-multi-touch-attribution.md for the notation.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

SEPARATOR = " > "
METHODS = ("last", "first", "linear", "markov", "shapley")


def states(channels: Iterable[str]) -> list[str]:
    """Row and column labels of the transition matrix."""
    return ["start", *channels, "conv", "null"]


# ---------------------------------------------------------------------------- references
def journey_states(path: str, converted: int) -> list[str]:
    """One journey as its chain states: ``"A > B"``, 1 -> ``["start", "A", "B", "conv"]``."""
    return ["start", *path.split(SEPARATOR), "conv" if converted else "null"]


def heuristic_credit(journeys: pd.DataFrame, rule: str, channels: list[str]) -> pd.Series:
    """Conversions credited to each channel by ``rule`` in {"last", "first", "linear"}."""
    conv = journeys.loc[journeys["converted"] == 1, "path"].str.split(SEPARATOR)
    if rule == "last":
        credit = conv.str[-1].value_counts()
    elif rule == "first":
        credit = conv.str[0].value_counts()
    elif rule == "linear":
        share = 1 / conv.str.len()
        touches = pd.DataFrame({"channel": conv, "share": share}).explode("channel")
        credit = touches.groupby("channel")["share"].sum()
    else:
        raise ValueError(f"rule must be 'last', 'first' or 'linear', not {rule!r}")
    return credit.reindex(channels, fill_value=0).astype(float)


def transition_matrix(journeys: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """First-order transition probabilities, index and columns ``states(channels)``."""
    counts = Counter()
    for path, y in zip(journeys["path"], journeys["converted"], strict=True):
        s = journey_states(path, y)
        counts.update(zip(s[:-1], s[1:], strict=True))
    labels = states(channels)
    M = pd.DataFrame(0.0, index=labels, columns=labels)
    for (a, b), n in counts.items():
        M.loc[a, b] = n
    M.loc["conv", "conv"] = 1.0
    M.loc["null", "null"] = 1.0
    return M.div(M.sum(axis=1), axis=0)


def conversion_probability(P: pd.DataFrame) -> float:
    """x_start, the probability that a journey eventually reaches conv: x = (I - Q)^-1 r."""
    T = [s for s in P.index if s not in ("conv", "null")]
    Q = P.loc[T, T].to_numpy()
    r = P.loc[T, "conv"].to_numpy()
    x = np.linalg.solve(np.eye(len(T)) - Q, r)
    return float(x[T.index("start")])


def removal_effects(P: pd.DataFrame, channels: list[str]) -> pd.Series:
    """RE_c = 1 - p_{-c} / p, with every journey that reaches c stopped there (c -> null)."""
    p = conversion_probability(P)
    out = {}
    for c in channels:
        Pc = P.copy()
        Pc.loc[c, :] = 0.0
        Pc.loc[c, "null"] = 1.0
        out[c] = 1 - conversion_probability(Pc) / p
    return pd.Series(out, dtype=float)


def markov_credit(removal: pd.Series | dict, conversions: float) -> pd.Series:
    """Y * RE_c / sum_k RE_k (the lab's normalization: removal effects add up to more than 1).
    Accepts a dict {channel: RE_c} as well, since Checkpoint 3 accepts one."""
    removal = pd.Series(removal, dtype=float)
    return conversions * removal / removal.sum()


def shapley_values(players: list[str], v: Callable[[frozenset], float]) -> dict[str, float]:
    """Exact Shapley values: phi_c = sum over T not containing c of
    |T|! (n - |T| - 1)! / n! * [v(T + c) - v(T)]."""
    n = len(players)
    phi = {}
    for c in players:
        others = [p for p in players if p != c]
        total = 0.0
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for T in itertools.combinations(others, k):
                T = frozenset(T)
                total += w * (v(T | {c}) - v(T))
        phi[c] = total
    return phi


def shapley_by_orders(players: list[str], v: Callable[[frozenset], float]) -> dict[str, float]:
    """The Shapley value as the average marginal contribution over all n! join orders."""
    phi = dict.fromkeys(players, 0.0)
    orders = list(itertools.permutations(players))
    for order in orders:
        joined = frozenset()
        for c in order:
            phi[c] += v(joined | {c}) - v(joined)
            joined = joined | {c}
    return {c: x / len(orders) for c, x in phi.items()}


def set_key(channels: Iterable[str]) -> str:
    """A readable key for a set of channels: sorted names joined by ``" + "``."""
    return " + ".join(sorted(set(channels)))


def coalition_table(journeys: pd.DataFrame) -> pd.DataFrame:
    """One row per set of channels that occurs: ``channels`` (frozenset), ``size``,
    ``journeys``, ``conversions`` and ``rate`` (= v(S)), indexed by :func:`set_key`."""
    sets = journeys["path"].str.split(SEPARATOR).map(frozenset)
    keys = sets.map(set_key)
    table = journeys.groupby(keys)["converted"].agg(journeys="size", conversions="sum")
    first = pd.Series(sets.to_numpy(), index=keys.to_numpy())
    table.insert(0, "channels", first[~first.index.duplicated()].reindex(table.index))
    table.insert(1, "size", table["channels"].map(len))
    table["rate"] = table["conversions"] / table["journeys"]
    table.index.name = "set"
    return table.sort_values(["size", "journeys"], ascending=[True, False])


def shapley_credit(coalitions: pd.DataFrame, channels: list[str],
                   shapley_fn: Callable = shapley_values) -> tuple[pd.Series, pd.DataFrame]:
    """Shapley credit per channel and the per-set shares phi_c / v(S) (rows: sets with a
    conversion). Raises if a subset needed by a converting set has no journeys."""
    rate = {row.channels: row.rate for row in coalitions.itertuples()}

    def v(S):
        if not S:
            return 0.0
        if S not in rate:
            raise KeyError(S)
        return rate[S]

    credit = pd.Series(0.0, index=list(channels))
    shares = {}
    for row in coalitions[coalitions["conversions"] > 0].itertuples():
        S = row.channels
        missing = [set_key(T) for k in range(1, len(S)) for T in itertools.combinations(S, k)
                   if frozenset(T) not in rate]
        if missing:
            raise ValueError(f"the set {row.Index} needs v of {missing}, which no journey has")
        phi = dict(shapley_fn(sorted(S), v))
        shares[row.Index] = {c: phi[c] / row.rate for c in S}
        for c in S:
            credit[c] += row.conversions * phi[c] / row.rate
    return credit, pd.DataFrame(shares).T.reindex(columns=list(channels))


def credit_vs_truth(credit: pd.DataFrame, incremental: pd.Series) -> pd.DataFrame:
    """Per method: share error (total variation distance between credit shares and true
    incremental shares) and the channel with the largest absolute gap, with its signed gap."""
    shares = credit.div(credit.sum(axis=0), axis=1)
    true = incremental.reindex(credit.index) / incremental.sum()
    gap = shares.sub(true, axis=0)
    worst = gap.abs().idxmax()
    return pd.DataFrame({
        "tvd": 0.5 * gap.abs().sum(),
        "largest_gap_channel": worst,
        "largest_gap": [gap.loc[worst[m], m] for m in gap.columns],
    })


# ----------------------------------------------------------------------- the bootstrap
class PathCounts:
    """The journeys aggregated by (path, converted), with what each distinct row adds to every
    method's counts, so that a resample (a weight per row) is a few weighted counts.

    Counting uses ``np.bincount``, not matrix products: a BLAS matrix-vector product starts
    one thread per visible core, and under a container CPU quota smaller than the core count
    (``docker run --cpus 2`` on a 10-core laptop) those threads made the bootstrap ten times
    slower (measured 2026-10-10, OpenBLAS 0.3.27)."""

    def __init__(self, journeys: pd.DataFrame, channels: list[str]):
        self.channels = list(channels)
        agg = journeys.groupby(["path", "converted"]).size().rename("n").reset_index()
        self.rows = agg
        self.weights = agg["n"].to_numpy(dtype=float)
        self.converted = agg["converted"].to_numpy(dtype=float)
        labels = states(channels)
        self.labels = labels
        idx = {s: i for i, s in enumerate(labels)}
        S = len(labels)
        cidx = {c: i for i, c in enumerate(channels)}
        t_row, t_pair, l_row, l_chan, l_share, last, first, sets = ([] for _ in range(8))
        for k, (path, y) in enumerate(zip(agg["path"], agg["converted"], strict=True)):
            p = path.split(SEPARATOR)
            s = journey_states(path, y)
            for a, b in zip(s[:-1], s[1:], strict=True):
                t_row.append(k)
                t_pair.append(idx[a] * S + idx[b])
            for c in p:
                l_row.append(k)
                l_chan.append(cidx[c])
                l_share.append(1 / len(p))
            last.append(cidx[p[-1]])
            first.append(cidx[p[0]])
            sets.append(frozenset(p))
        self.t_row, self.t_pair = np.array(t_row), np.array(t_pair)
        self.l_row, self.l_chan, self.l_share = np.array(l_row), np.array(l_chan), np.array(l_share)
        self.last, self.first = np.array(last), np.array(first)
        self.sets = sorted(set(sets), key=lambda s: (len(s), sorted(s)))
        where = {s: i for i, s in enumerate(self.sets)}
        self.set_index = np.array([where[s] for s in sets])

    def credit(self, w: np.ndarray, removal_fn: Callable = removal_effects,
               shapley_fn: Callable = shapley_values) -> pd.DataFrame:
        """Credited conversions per channel (rows) and method (columns) for row weights w."""
        S, K = len(self.labels), len(self.channels)
        conv_w = w * self.converted
        out = {
            "last": np.bincount(self.last, weights=conv_w, minlength=K),
            "first": np.bincount(self.first, weights=conv_w, minlength=K),
            "linear": np.bincount(self.l_chan, weights=conv_w[self.l_row] * self.l_share,
                                  minlength=K),
        }
        counts = np.bincount(self.t_pair, weights=w[self.t_row], minlength=S * S).reshape(S, S)
        counts[self.labels.index("conv"), self.labels.index("conv")] = 1.0
        counts[self.labels.index("null"), self.labels.index("null")] = 1.0
        P = pd.DataFrame(counts / counts.sum(axis=1, keepdims=True), index=self.labels,
                         columns=self.labels)
        re = pd.Series(removal_fn(P, self.channels), dtype=float).reindex(self.channels)
        out["markov"] = (conv_w.sum() * re / re.sum()).to_numpy()
        n_set = np.bincount(self.set_index, weights=w, minlength=len(self.sets))
        c_set = np.bincount(self.set_index, weights=conv_w, minlength=len(self.sets))
        table = pd.DataFrame({
            "channels": self.sets,
            "size": [len(s) for s in self.sets],
            "journeys": n_set,
            "conversions": c_set,
        }, index=[set_key(s) for s in self.sets])
        table = table[table["journeys"] > 0]
        table["rate"] = table["conversions"] / table["journeys"]
        out["shapley"] = shapley_credit(table, self.channels, shapley_fn)[0].to_numpy()
        return pd.DataFrame(out, index=self.channels)[list(METHODS)]


def bootstrap_credit(journeys: pd.DataFrame, channels: list[str], n_boot: int,
                     rng: np.random.Generator, removal_fn: Callable = removal_effects,
                     shapley_fn: Callable = shapley_values) -> np.ndarray:
    """Credited conversions on ``n_boot`` resamples of journeys (with replacement):
    an array (n_boot, channels, methods), methods in :data:`METHODS` order."""
    pc = PathCounts(journeys, channels)
    n = int(pc.weights.sum())
    out = np.empty((n_boot, len(channels), len(METHODS)))
    for b in range(n_boot):
        w = rng.multinomial(n, pc.weights / n).astype(float)
        out[b] = pc.credit(w, removal_fn, shapley_fn).to_numpy()
    return out


# ------------------------------------------------------------------------------ checks
TINY = pd.DataFrame({
    "path": ["A > B", "A", "B", "B > A", "A > A > B"],
    "converted": [1, 0, 1, 0, 1],
})
TINY_CHANNELS = ["A", "B"]


def _recover(n: int | str) -> str:
    if isinstance(n, int):
        return f" To move on now, run workshop.use_reference({n}) and come back to this later."
    return f" To move on now, run workshop.use_reference({n}) and rerun from its checkpoint."


def _fmt(x: dict | pd.Series, digits: int = 4) -> str:
    items = x.items() if hasattr(x, "items") else enumerate(x)
    return "{" + ", ".join(f"{k}: {float(v):.{digits}g}" for k, v in items) + "}"


def _as_series(obj, what: str, n: int) -> pd.Series:
    if isinstance(obj, dict):
        obj = pd.Series(obj, dtype=float)
    if not isinstance(obj, pd.Series):
        raise AssertionError(
            f"Checkpoint {n}: {what} returned a {type(obj).__name__}, not a pandas Series."
            " The lab's functions return one value per channel, indexed by channel name."
            " Check the return statement." + _recover(n))
    return obj


def check_heuristic_credit(fn: Callable, journeys: pd.DataFrame, channels: list[str],
                           n_converting: int) -> pd.DataFrame:
    """Exercise 1: TINY by hand, then sums on the journeys. Returns the credit table."""
    want = {"last": {"A": 0.0, "B": 3.0}, "first": {"A": 2.0, "B": 1.0},
            "linear": {"A": 7 / 6, "B": 11 / 6}}
    for rule, expected in want.items():
        got = _as_series(fn(TINY, rule, TINY_CHANNELS), f"heuristic_credit(TINY, {rule!r}, ...)", 1)
        if list(got.index) != TINY_CHANNELS or got.isna().any():
            raise AssertionError(
                f"Checkpoint 1: heuristic_credit(TINY, {rule!r}, ['A', 'B']) returned"
                f" {_fmt(got)}, with index {list(got.index)}; it must have one value for each of"
                " ['A', 'B'], in that order, with no NaN. A channel that got no credit is usually"
                " missing from the result. Return every channel in `channels`, with 0 where it got"
                " no credit (`reindex(channels, fill_value=0)`)." + _recover(1))
        if rule == "linear" and np.allclose(got.to_numpy(), [1.0, 2.0]):
            raise AssertionError(
                "Checkpoint 1: linear credit on TINY is {A: 1, B: 2}; worked by hand it is"
                " {A: 7/6, B: 11/6}. Linear here is per touch, not per distinct channel: journey 5"
                " (A > A > B) gives A two shares of 1/3. Split the path into its touches and give"
                " each touch 1/L, where L is the number of touches." + _recover(1))
        if not np.allclose(got.to_numpy(dtype=float), list(expected.values()), atol=1e-12, rtol=0):
            if np.isclose(got.sum(), 5.0):
                cause = (" Your credit adds up to 5, the number of journeys in TINY, but only 3"
                         " converted: only converting journeys give credit; keep the rows with"
                         " `converted == 1`.")
            else:
                takes = {"last": "the last touch", "first": "the first touch",
                         "linear": "every touch, 1/L each"}[rule]
                cause = (f" Check which touch the rule takes ({takes}) and that only converting"
                         " journeys give credit.")
            raise AssertionError(
                f"Checkpoint 1: heuristic_credit(TINY, {rule!r}, ['A', 'B']) is {_fmt(got)};"
                f" worked by hand it is {_fmt(expected)}." + cause + _recover(1))
    table = {}
    for rule in ("last", "first", "linear"):
        got = _as_series(fn(journeys, rule, channels),
                         f"heuristic_credit(journeys, {rule!r}, ...)", 1).astype(float)
        if list(got.index) != list(channels) or got.isna().any():
            raise AssertionError(
                f"Checkpoint 1: on the journeys, {rule} credit has index {list(got.index)} or NaN"
                f" values; it must have one value for each of {list(channels)}. Return every"
                " channel, with 0 where it got no credit (`reindex(channels, fill_value=0)`)."
                + _recover(1))
        if abs(got.sum() - len(journeys)) < 1e-6:
            raise AssertionError(
                f"Checkpoint 1: {rule} credit adds up to {got.sum():,.0f}, the number of journeys;"
                f" it must add up to the {n_converting:,} converting journeys. Only converting"
                " journeys give credit; keep the rows with `converted == 1`." + _recover(1))
        if abs(got.sum() - n_converting) > 1e-9 or (got < 0).any():
            raise AssertionError(
                f"Checkpoint 1: {rule} credit on the journeys adds up to {got.sum():,.4f} (smallest"
                f" value {got.min():.4f}); it must add up to the {n_converting:,} converting"
                " journeys, with no negative value. Each converting journey hands out exactly one"
                " conversion. Check that every converting journey is counted once." + _recover(1))
        table[rule] = got
    return pd.DataFrame(table)


def _check_labels(M: pd.DataFrame, labels: list[str], where: str, n: int) -> None:
    if not isinstance(M, pd.DataFrame):
        raise AssertionError(
            f"Checkpoint {n}: transition_matrix({where}) returned a {type(M).__name__}, not a"
            " pandas DataFrame with states as index and columns. Check the return statement."
            + _recover(n))
    if list(M.index) != labels or list(M.columns) != labels:
        raise AssertionError(
            f"Checkpoint {n}: on {where}, the matrix has rows {list(M.index)} and columns"
            f" {list(M.columns)}; both must be {labels}. Counting builds the labels in the order"
            f" they occur. Reindex rows and columns to {labels} (`.reindex(index=labels,"
            " columns=labels, fill_value=0)`)." + _recover(n))


def check_transition_matrix(fn: Callable, journeys: pd.DataFrame, channels: list[str]) -> None:
    """Exercise 2: TINY by hand, then structure on the journeys."""
    labels = states(TINY_CHANNELS)
    M = fn(TINY, TINY_CHANNELS)
    _check_labels(M, labels, "TINY", 2)
    M = M.astype(float)
    if M.loc[["conv", "null"]].isna().all(axis=1).any():
        raise AssertionError(
            "Checkpoint 2: the conv or null row of the TINY matrix is NaN. A row of zeros"
            " divided by its total of 0 gives NaN; set conv -> conv and null -> null to 1 before"
            " dividing." + _recover(2))
    start_msg = (
        " by hand it is start -> A 3/5, start -> B 2/5 (three journeys start with A, two with B)."
        " Each journey adds exactly one start -> first-touch transition; check that"
        " `journey_states` is applied to every journey and that its consecutive pairs are"
        " counted." + _recover(2))
    if M.loc["start"].isna().all() or (M.loc["start"].fillna(0) == 0).all():
        raise AssertionError(
            f"Checkpoint 2: TINY's start row is {_fmt(M.loc['start'])};" + start_msg)
    rows = M.sum(axis=1, skipna=False)
    if not np.allclose(rows.to_numpy(), 1.0, atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 2: the rows of the TINY matrix add up to {_fmt(rows)}; every row must add"
            " up to 1. Divide each row by its own total (`M.div(M.sum(axis=1), axis=0)`), not by"
            " column totals or the grand total." + _recover(2))
    want = pd.DataFrame(0.0, index=labels, columns=labels)
    want.loc["start", ["A", "B"]] = [3 / 5, 2 / 5]
    want.loc["A", ["A", "B", "null"]] = [1 / 5, 2 / 5, 2 / 5]
    want.loc["B", ["A", "conv"]] = [1 / 4, 3 / 4]
    want.loc["conv", "conv"] = want.loc["null", "null"] = 1.0
    if not np.allclose(M.loc["start"].to_numpy(), want.loc["start"].to_numpy(),
                       atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 2: TINY's start row is {_fmt(M.loc['start'])};" + start_msg)
    diff = (M - want).abs()
    if diff.to_numpy().max() > 1e-12:
        a = diff.max(axis=1).idxmax()
        raise AssertionError(
            f"Checkpoint 2: TINY's {a} row is {_fmt(M.loc[a])}; by hand it is {_fmt(want.loc[a])}."
            " Usually a transition is missing or counted twice. Count every consecutive pair of"
            " each journey's states (start, touches, then conv or null) once, then divide each"
            " row by its total." + _recover(2))
    P = fn(journeys, channels)
    labels = states(channels)
    _check_labels(P, labels, "the journeys", 2)
    bad = []
    rows = P.sum(axis=1, skipna=False)
    if not np.allclose(rows.to_numpy(dtype=float), 1.0, atol=1e-12, rtol=0):
        bad.append(f"row totals {_fmt(rows)}")
    if (P["start"] != 0).any():
        bad.append("a nonzero start column (no journey returns to start)")
    for s in ("conv", "null"):
        unit = pd.Series(0.0, index=labels)
        unit[s] = 1.0
        if not np.allclose(P.loc[s].to_numpy(dtype=float), unit.to_numpy()):
            bad.append(f"the {s} row is not {s} -> {s} = 1")
    if bad:
        raise AssertionError(
            "Checkpoint 2: on the journeys the matrix has " + "; ".join(bad) + ". Each row must"
            " add up to 1, start must have no incoming transitions, and conv and null must be"
            " absorbing. Check the row division and the conv -> conv and null -> null entries."
            + _recover(2))


def check_chain(prob_fn: Callable, P_tiny: pd.DataFrame, P: pd.DataFrame,
                observed_share: float) -> float:
    """The labeled check after Exercise 2: the chain reproduces the observed share."""
    p_tiny = prob_fn(P_tiny)
    if abs(p_tiny - 3 / 5) > 1e-12:
        raise AssertionError(
            f"The chain's conversion probability on TINY is {p_tiny:.4f}; worked by hand it is 3/5."
            " Your transition_matrix passed Checkpoint 2, so the provided conversion_probability"
            " was probably changed: rerun its cell. If that does not help, tell the instructor.")
    p = prob_fn(P)
    if abs(p - observed_share) > 1e-9:
        raise AssertionError(
            f"The chain's conversion probability is {p:.3f}, but {observed_share:.2%} of journeys"
            " converted, and a chain estimated by counting must reproduce that share exactly."
            " Your transition_matrix passed Checkpoint 2, so it counts transitions differently"
            " from the reference: check that every journey adds exactly one start -> first-touch"
            " transition and exactly one last-touch -> conv or null transition. To move on now,"
            " run `workshop.use_reference(2)` and rerun from Exercise 2's checkpoint.")
    return p


def _call_removal(fn: Callable, P: pd.DataFrame, channels: list[str], call: str):
    """Call the learner's removal_effects, turning the two errors a wrong removal raises
    into recovery messages."""
    try:
        return fn(P, channels)
    except np.linalg.LinAlgError as exc:
        raise AssertionError(
            f"Checkpoint 3: {call} raised LinAlgError: {exc}. The chain has no solution when the"
            " removed channel loops to itself (`P.loc[c, c] = 1`): a journey that reaches c never"
            " leaves it, so I - Q has a row of zeros. Send its transitions to null instead: set"
            " row c to 0, then `P.loc[c, 'null'] = 1`." + _recover(3)) from None
    except ZeroDivisionError as exc:
        raise AssertionError(
            f"Checkpoint 3: {call} raised ZeroDivisionError: {exc}. In TINY, removing B loses"
            " every conversion, so p_-B = 0, while p = 3/5. Dividing by zero usually means p was"
            " computed from the removed copy inside the loop (then p is p_-c). Compute p once,"
            " from the unchanged P, before the loop, and divide by it: RE = 1 - p_-c / p."
            + _recover(3)) from None


def check_removal_effects(fn: Callable, P: pd.DataFrame, channels: list[str]) -> pd.Series:
    """Exercise 3: TINY by hand (and P unchanged), then the journeys against the reference."""
    P_tiny = transition_matrix(TINY, TINY_CHANNELS)
    before = P_tiny.copy()
    call = "removal_effects(P_tiny, ['A', 'B'])"
    got = _as_series(_call_removal(fn, P_tiny, TINY_CHANNELS, call), call, 3)
    if not P_tiny.equals(before):
        raise AssertionError(
            "Checkpoint 3: removal_effects changed the matrix P it was given. Then each channel"
            " after the first was removed from an already changed chain. Work on `P.copy()` for"
            " each channel, never on P itself." + _recover(3))
    got = got.reindex(TINY_CHANNELS).astype(float)
    if got.isna().any():
        raise AssertionError(
            f"Checkpoint 3: removal_effects on TINY is {_fmt(got)}; it needs one value for each of"
            " ['A', 'B']. Return a Series indexed by `channels`." + _recover(3))
    if got["A"] < -1e-9:
        raise AssertionError(
            f"Checkpoint 3: on TINY your removal effect of A is {got['A']:.2f}; worked by hand it"
            " is 1/2. A negative removal effect usually means the channel was deleted and the other"
            " transitions renormalized, which reroutes journeys around it instead of stopping them."
            " Set row c to 0 and `P.loc[c, 'null'] = 1`." + _recover(3))
    if abs(got["B"]) < 1e-9:
        raise AssertionError(
            f"Checkpoint 3: on TINY your removal effects are {_fmt(got)}; worked by hand they are"
            " {A: 0.5, B: 1}. The removal effect is 1 - p_-c / p, not p_-c / p; removing B loses"
            " every conversion in TINY, so its removal effect is 1." + _recover(3))
    if not np.allclose(got.to_numpy(), [0.5, 1.0], atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 3: on TINY your removal effects are {_fmt(got)}; worked by hand they are"
            " {A: 0.5, B: 1} (p = 3/5, p_-A = 3/10, p_-B = 0). Check the removal: row c set to 0,"
            " P.loc[c, 'null'] = 1, then RE = 1 - p_-c / p with p from the unchanged P."
            + _recover(3))
    before = P.copy()
    got = _as_series(fn(P, channels), "removal_effects(P, CHANNELS)", 3)
    got = got.reindex(channels).astype(float)
    want = removal_effects(before, channels)
    if not P.equals(before):
        raise AssertionError(
            "Checkpoint 3: removal_effects changed P. Work on `P.copy()` for each channel."
            + _recover(3))
    if got.isna().any() or ((got < 0) | (got > 1)).any() or not np.allclose(
            got.to_numpy(), want.to_numpy(), atol=1e-9, rtol=0):
        raise AssertionError(
            f"Checkpoint 3: on the journeys your removal effects are {_fmt(got)}; the reference"
            f" gives {_fmt(want)}. Every removal effect lies between 0 and 1. Check the removal"
            " (row c to 0, c -> null to 1) and that p comes from the unchanged P." + _recover(3))
    return got


def check_markov_credit(credit_tiny: pd.Series) -> None:
    """The labeled check after the provided Markov credit cell."""
    got = pd.Series(credit_tiny, dtype=float).reindex(TINY_CHANNELS)
    if not np.allclose(got.to_numpy(), [1.0, 2.0], atol=1e-12, rtol=0):
        raise AssertionError(
            f"Markov credit on TINY is {_fmt(got)}; with Y = 3 conversions and removal effects"
            " {A: 1/2, B: 1} (sum 1.5) it is {A: 1, B: 2}. Your removal_effects passed Checkpoint"
            " 3, so the provided cell was probably changed: rerun it, or run"
            " workshop.use_reference(3) and rerun from Exercise 3's checkpoint.")


class _Game:
    """A strict value function: takes a frozenset, looks it up in a table. ``rejected`` keeps
    the first argument it refused, so a check can tell v's errors from the caller's own."""

    def __init__(self, table: dict[frozenset, float]):
        self.table = table
        self.rejected = None

    def __call__(self, S):
        if isinstance(S, frozenset) and S in self.table:
            return self.table[S]
        if self.rejected is None:
            self.rejected = S
        if not isinstance(S, frozenset):
            raise TypeError(f"v takes a frozenset, not a {type(S).__name__}")
        raise KeyError(S)


def random_game(players: list[str], seed: int) -> dict[frozenset, float]:
    """v(S) uniform on [0, 1) for every non-empty S, v(empty) = 0 (seeded)."""
    rng = np.random.default_rng(seed)
    table = {frozenset(): 0.0}
    for k in range(1, len(players) + 1):
        for S in itertools.combinations(players, k):
            table[frozenset(S)] = float(rng.random())
    return table


THREE_PLAYER_GAME = {
    frozenset(): 0.0, frozenset("A"): 0.10, frozenset("B"): 0.20, frozenset("C"): 0.0,
    frozenset("AB"): 0.40, frozenset("AC"): 0.10, frozenset("BC"): 0.20, frozenset("ABC"): 0.40,
}
TWO_PLAYER_GAME = {frozenset(): 0.0, frozenset("A"): 0.10, frozenset("B"): 0.05,
                   frozenset("AB"): 0.20}


def check_shapley_values(fn: Callable, seed: int) -> dict[str, float]:
    """Exercise 4: two hand examples, then efficiency, brute force and symmetry on a random
    five-player game. Returns the five-player values."""
    def call(players, table):
        game = _Game(table)
        try:
            out = fn(players, game)
        except (KeyError, TypeError) as exc:
            if game.rejected is None:  # not v's error: let the learner's own traceback show
                raise
            raise AssertionError(
                f"Checkpoint 4: calling v inside shapley_values raised {type(exc).__name__}: {exc}."
                " v takes a frozenset of players (a coalition); a list, tuple or set fails. Call"
                " `v(frozenset(T) | {c})` and `v(frozenset(T))`." + _recover(4)) from None
        if not isinstance(out, Mapping):
            raise AssertionError(
                f"Checkpoint 4: shapley_values({players}, v) returned a {type(out).__name__}, not a"
                " dict. The cells after this checkpoint use the result as a dict (`phi[c]` and"
                " `phi.values()`), so return a dict {player: value}: `values[c] = total` for each"
                " player c, then `return values`." + _recover(4))
        try:
            return {k: float(x) for k, x in out.items()}
        except (TypeError, ValueError) as exc:
            raise AssertionError(
                f"Checkpoint 4: shapley_values({players}, v) returned a dict whose values are not"
                f" all numbers ({type(exc).__name__}: {exc}). Return a dict {{player: value}} with"
                " one number per player." + _recover(4)) from None

    three = call(["A", "B", "C"], THREE_PLAYER_GAME)
    if abs(three.get("C", np.nan)) > 1e-12:
        raise AssertionError(
            f"Checkpoint 4: in the three-player game C adds nothing to any coalition, but your"
            f" value for C is {three.get('C', np.nan):.4f}; it must be 0. Sum over the coalitions"
            " T that do not contain c, and compare v(T | {c}) with v(T)." + _recover(4))
    two = call(["A", "B"], TWO_PLAYER_GAME)
    if np.allclose([two.get("A", np.nan), two.get("B", np.nan)], [0.175, 0.100], atol=1e-9):
        raise AssertionError(
            "Checkpoint 4: the two-player example gives {A: 0.175, B: 0.100}, which adds up to"
            " 0.275 instead of v(AB) - v(empty) = 0.20; by hand it is {A: 0.125, B: 0.075}. The"
            " weight is |T|! (n - |T| - 1)! / n! for a coalition T that leaves c out;"
            " (n - |T|)! in place of (n - |T| - 1)! gives these numbers." + _recover(4))
    if not np.allclose([two.get("A", np.nan), two.get("B", np.nan)], [0.125, 0.075],
                       atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 4: the two-player example (v(A) = 0.10, v(B) = 0.05, v(AB) = 0.20) gives"
            f" {_fmt(two)}; by hand it is {{A: 0.125, B: 0.075}}: A joins first or second with"
            " equal chance, adding 0.10 or 0.20 - 0.05. Check the weights and that you sum over"
            " every coalition T without c, the empty one included." + _recover(4))
    if not np.allclose([three["A"], three["B"]], [0.15, 0.25], atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 4: the three-player game gives {_fmt(three)}; by hand (averaging over the"
            " six orders) it is {A: 0.15, B: 0.25, C: 0}. Check the weights |T|! (n - |T| - 1)! /"
            " n! and the marginal contributions v(T | {c}) - v(T)." + _recover(4))
    players = ["p1", "p2", "p3", "p4", "p5"]
    table = random_game(players, seed)
    phi = call(players, table)
    total, v_all = sum(phi.values()), table[frozenset(players)]
    if abs(total - v_all) > 1e-12:
        raise AssertionError(
            f"Checkpoint 4: on the random five-player game (seed {seed}) your values add up to"
            f" {total:.6f}, but efficiency requires v(N) - v(empty) = {v_all:.6f}."
            " Check the weights and the range of coalition sizes (0 to n - 1)." + _recover(4))
    brute = shapley_by_orders(players, _Game(table))
    worst = max(players, key=lambda c: abs(phi[c] - brute[c]))
    if abs(phi[worst] - brute[worst]) > 1e-12:
        raise AssertionError(
            f"Checkpoint 4: on the random five-player game (seed {seed}) your value for {worst} is"
            f" {phi[worst]:.6f}; the average over all 120 orders is {brute[worst]:.6f}. Check the"
            " weights |T|! (n - |T| - 1)! / n!." + _recover(4))
    swap = {"p1": "p2", "p2": "p1"}
    swapped = {frozenset(swap.get(c, c) for c in S): x for S, x in table.items()}
    phi_s = call(players, swapped)
    if abs(phi_s["p1"] - phi["p2"]) > 1e-12 or abs(phi_s["p2"] - phi["p1"]) > 1e-12:
        raise AssertionError(
            "Checkpoint 4: swapping p1 and p2 in the game did not swap their values (symmetry)."
            " Check that each player's value uses only v, not the player's position in the list."
            + _recover(4))
    return phi


def check_credit_vs_truth(fn: Callable, credit: pd.DataFrame,
                          incremental: pd.Series) -> pd.DataFrame:
    """Exercise 5: the hand example, then the data against the reference."""
    hand_credit = pd.DataFrame({"m1": [5.0, 3.0, 2.0], "m2": [2.0, 2.0, 6.0]},
                               index=["A", "B", "C"])
    hand_inc = pd.Series([6.0, 2.0, 0.0], index=["A", "B", "C"])
    got = fn(hand_credit, hand_inc)
    if not isinstance(got, pd.DataFrame) or not {"tvd", "largest_gap_channel",
                                                 "largest_gap"} <= set(got.columns):
        raise AssertionError(
            "Checkpoint 5: credit_vs_truth must return a DataFrame with one row per method and"
            " columns tvd, largest_gap_channel and largest_gap." + _recover(5))
    if list(got.index) != ["m1", "m2"]:
        raise AssertionError(
            f"Checkpoint 5: on the hand example the rows are {list(got.index)}; they must be the"
            " methods, ['m1', 'm2'] (the columns of `credit`)." + _recover(5))
    tvd = got.loc["m1", "tvd"]
    if np.isclose(tvd, 2.0):
        raise AssertionError(
            "Checkpoint 5: on the hand example m1's tvd is 2.0; by hand it is 0.25. Compare shares,"
            " not counts: divide each column of `credit` by its total and `incremental` by its"
            " total." + _recover(5))
    if np.isclose(tvd, 0.5):
        raise AssertionError(
            "Checkpoint 5: on the hand example m1's tvd is 0.50; by hand it is 0.25. The share"
            " error has a factor 1/2, because each misplaced share is counted once where it is"
            " missing and once where it is extra." + _recover(5))
    if not np.allclose(got["tvd"].astype(float).to_numpy(), [0.25, 0.60], atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 5: on the hand example tvd is {_fmt(got['tvd'])}; by hand it is"
            " {m1: 0.25, m2: 0.60}. D = 1/2 x the sum over channels of |credit share - true"
            " share|." + _recover(5))
    if got.loc["m1", "largest_gap_channel"] == "C":
        raise AssertionError(
            "Checkpoint 5: on the hand example m1's largest gap is reported as C (+0.20); by hand"
            " it is A (-0.25). Pick the channel with the largest absolute gap, then report its gap"
            " with its sign." + _recover(5))
    if list(got["largest_gap_channel"]) != ["A", "C"] or not np.allclose(
            got["largest_gap"].astype(float).to_numpy(), [-0.25, 0.60], atol=1e-12, rtol=0):
        raise AssertionError(
            f"Checkpoint 5: on the hand example the largest gaps are"
            f" {list(zip(got['largest_gap_channel'], got['largest_gap'], strict=True))}; by hand"
            " they are m1: A, -0.25 and m2: C, +0.60 (credit share - true share, with its sign)."
            + _recover(5))
    got = fn(credit, incremental)
    want = credit_vs_truth(credit, incremental)
    ok = (list(got.index) == list(want.index)
          and np.allclose(got["tvd"].astype(float), want["tvd"], atol=1e-9, rtol=0)
          and list(got["largest_gap_channel"]) == list(want["largest_gap_channel"])
          and np.allclose(got["largest_gap"].astype(float), want["largest_gap"], atol=1e-9, rtol=0))
    if not ok:
        raise AssertionError(
            "Checkpoint 5: on the lab's credit table your result differs from the reference:\n"
            f"{got}\nreference:\n{want}\nCheck that shares are taken within each method (column)"
            " and that `incremental` is aligned to the channels of `credit`." + _recover(5))
    for m in ("markov", "shapley"):
        if m in want.index and want.loc[m, "largest_gap_channel"] != "retargeting":
            raise AssertionError(
                f"Checkpoint 5: the reference says {m}'s largest gap is"
                f" {want.loc[m, 'largest_gap_channel']}, not retargeting, so the data or the"
                " credit table above are not the lab's. Rerun the notebook from the top; if"
                " that does not help, tell the instructor.")
    return got
