"""Module 16's attribution helpers: the reference implementations on hand-worked examples, and
every checkpoint check passing on the reference and failing on each wrong answer the lab brief
lists (briefs/16-multi-touch-attribution.md, "Failure messages"), with that answer's message."""

import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from mktstats import attribution as at
from mktstats.synth.mta import CHANNELS, mta_journeys

CH = list(CHANNELS)


@pytest.fixture(scope="module")
def small():
    """A smaller simulated dataset (fast), with every channel set present."""
    return mta_journeys(seed=7, n_users=30_000).journeys


# ------------------------------------------------------------------- reference, by hand
def test_heuristics_on_tiny():
    assert at.heuristic_credit(at.TINY, "last", ["A", "B"]).tolist() == [0, 3]
    assert at.heuristic_credit(at.TINY, "first", ["A", "B"]).tolist() == [2, 1]
    np.testing.assert_allclose(at.heuristic_credit(at.TINY, "linear", ["A", "B"]),
                               [7 / 6, 11 / 6], atol=1e-12)
    with pytest.raises(ValueError):
        at.heuristic_credit(at.TINY, "middle", ["A", "B"])


def test_chain_on_tiny():
    P = at.transition_matrix(at.TINY, ["A", "B"])
    assert P.loc["start", "A"] == pytest.approx(3 / 5)
    assert P.loc["A", ["A", "B", "null"]].tolist() == pytest.approx([1 / 5, 2 / 5, 2 / 5])
    assert P.loc["B", ["A", "conv"]].tolist() == pytest.approx([1 / 4, 3 / 4])
    assert at.conversion_probability(P) == pytest.approx(3 / 5, abs=1e-12)
    re = at.removal_effects(P, ["A", "B"])
    assert re.tolist() == pytest.approx([0.5, 1.0], abs=1e-12)
    assert at.markov_credit(re, 3).tolist() == pytest.approx([1.0, 2.0])


def test_shapley_hand_examples():
    v2 = at._Game(at.TWO_PLAYER_GAME)
    assert at.shapley_values(["A", "B"], v2) == pytest.approx({"A": 0.125, "B": 0.075})
    v3 = at._Game(at.THREE_PLAYER_GAME)
    assert at.shapley_values(["A", "B", "C"], v3) == pytest.approx({"A": 0.15, "B": 0.25, "C": 0})
    players = ["p1", "p2", "p3", "p4", "p5"]
    g = at._Game(at.random_game(players, 1))
    assert at.shapley_values(players, g) == pytest.approx(at.shapley_by_orders(players, g),
                                                          abs=1e-12)


def test_credit_vs_truth_hand_example():
    credit = pd.DataFrame({"m1": [5.0, 3.0, 2.0], "m2": [2.0, 2.0, 6.0]}, index=list("ABC"))
    out = at.credit_vs_truth(credit, pd.Series([6.0, 2.0, 0.0], index=list("ABC")))
    assert out["tvd"].tolist() == pytest.approx([0.25, 0.60])
    assert out["largest_gap_channel"].tolist() == ["A", "C"]
    assert out["largest_gap"].tolist() == pytest.approx([-0.25, 0.60])


def test_vectorized_counts_equal_the_direct_methods(small):
    pc = at.PathCounts(small, CH)
    got = pc.credit(pc.weights)
    direct = pd.DataFrame({r: at.heuristic_credit(small, r, CH)
                           for r in ("last", "first", "linear")})
    direct["markov"] = at.markov_credit(at.removal_effects(at.transition_matrix(small, CH), CH),
                                        small["converted"].sum())
    direct["shapley"] = at.shapley_credit(at.coalition_table(small), CH)[0]
    pd.testing.assert_frame_equal(got, direct[list(at.METHODS)], atol=1e-9, check_names=False)


def test_bootstrap_shape_and_totals(small):
    arr = at.bootstrap_credit(small, CH, 5, np.random.default_rng(0))
    assert arr.shape == (5, 5, 5)
    totals = arr.sum(axis=1)  # every method hands out all conversions of the resample
    np.testing.assert_allclose(totals, totals[:, :1].repeat(5, axis=1), rtol=1e-9)


def test_shapley_credit_stops_on_a_missing_set():
    j = pd.DataFrame({"path": ["A > B", "A"], "converted": [1, 0]})
    with pytest.raises(ValueError, match="needs v of"):
        at.shapley_credit(at.coalition_table(j), ["A", "B"])


# ------------------------------------------------------------- checks pass on reference
def test_checks_pass_on_the_reference(small):
    n_conv = int(small["converted"].sum())
    table = at.check_heuristic_credit(at.heuristic_credit, small, CH, n_conv)
    assert table.sum().tolist() == pytest.approx([n_conv] * 3)
    at.check_transition_matrix(at.transition_matrix, small, CH)
    P = at.transition_matrix(small, CH)
    at.check_chain(at.conversion_probability, at.transition_matrix(at.TINY, ["A", "B"]), P,
                   small["converted"].mean())
    re = at.check_removal_effects(at.removal_effects, P, CH)
    at.check_markov_credit(at.markov_credit(at.removal_effects(
        at.transition_matrix(at.TINY, ["A", "B"]), ["A", "B"]), 3))
    at.check_shapley_values(at.shapley_values, seed=2040)
    credit = table.assign(markov=at.markov_credit(re, n_conv),
                          shapley=at.shapley_credit(at.coalition_table(small), CH)[0])
    inc = pd.Series({"display": 358, "social": 448, "search": 340, "email": 114,
                     "retargeting": 0})
    at.check_credit_vs_truth(at.credit_vs_truth, credit, inc)


# ------------------------------------------------- each listed wrong answer, its message
def _fails(match, fn, *args, **kwargs):
    with pytest.raises(AssertionError, match=match) as info:
        fn(*args, **kwargs)
    assert "workshop.use_reference(" in str(info.value)
    return str(info.value)


def credit_all_journeys(journeys, rule, channels):
    paths = journeys["path"].str.split(" > ")
    return paths.str[-1].value_counts().reindex(channels, fill_value=0).astype(float)


def linear_per_channel(journeys, rule, channels):
    if rule != "linear":
        return at.heuristic_credit(journeys, rule, channels)
    conv = journeys.loc[journeys["converted"] == 1, "path"].str.split(" > ").map(set)
    out = Counter()
    for s in conv:
        for c in s:
            out[c] += 1 / len(s)
    return pd.Series(out, dtype=float).reindex(channels, fill_value=0)


def no_reindex(journeys, rule, channels):
    conv = journeys.loc[journeys["converted"] == 1, "path"].str.split(" > ")
    return conv.str[-1].value_counts().astype(float)


@pytest.mark.parametrize("wrong, message", [
    (credit_all_journeys, "only converting journeys give credit"),
    (linear_per_channel, "per touch, not per distinct channel"),
    (no_reindex, r"reindex\(channels, fill_value=0\)"),
])
def test_exercise_1_messages(small, wrong, message):
    _fails(message, at.check_heuristic_credit, wrong, small, CH, int(small["converted"].sum()))


def credit_all_journeys_on_data(journeys, rule, channels):
    """Right on TINY's three converting journeys only if it used them: credits every journey."""
    if len(journeys) == len(at.TINY):
        return at.heuristic_credit(journeys, rule, channels)
    return at.heuristic_credit(journeys.assign(converted=1), rule, channels)


def test_exercise_1_message_on_the_data(small):
    msg = _fails("the number of journeys", at.check_heuristic_credit,
                 credit_all_journeys_on_data, small, CH, int(small["converted"].sum()))
    assert "converted == 1" in msg


def _counts(journeys, channels):
    labels = at.states(channels)
    c = Counter()
    for path, y in zip(journeys["path"], journeys["converted"], strict=True):
        s = at.journey_states(path, y)
        c.update(zip(s[:-1], s[1:], strict=True))
    M = pd.DataFrame(0.0, index=labels, columns=labels)
    for (a, b), n in c.items():
        M.loc[a, b] = n
    return M


def tm_column_totals(journeys, channels):
    M = _counts(journeys, channels)
    M.loc["conv", "conv"] = M.loc["null", "null"] = 1
    return M.div(M.sum(axis=0), axis=1)


def tm_grand_total(journeys, channels):
    M = _counts(journeys, channels)
    M.loc["conv", "conv"] = M.loc["null", "null"] = 1
    return M / M.to_numpy().sum()


def tm_no_absorbing(journeys, channels):
    M = _counts(journeys, channels)
    return M.div(M.sum(axis=1), axis=0)


def tm_start_to_every_touch(journeys, channels):
    M = _counts(journeys, channels)
    for path in journeys["path"]:
        for c in path.split(" > ")[1:]:
            M.loc["start", c] += 1
    M.loc["conv", "conv"] = M.loc["null", "null"] = 1
    return M.div(M.sum(axis=1), axis=0)


def tm_no_start(journeys, channels):
    M = _counts(journeys, channels)
    M.loc["start"] = 0.0
    M.loc["conv", "conv"] = M.loc["null", "null"] = 1
    return M.div(M.sum(axis=1), axis=0)


def tm_unordered(journeys, channels):
    P = at.transition_matrix(journeys, channels)
    order = sorted(P.index)
    return P.loc[order, order]


@pytest.mark.parametrize("wrong, message", [
    (tm_column_totals, "Divide each row by its own total"),
    (tm_grand_total, "Divide each row by its own total"),
    (tm_no_absorbing, "row of zeros divided by its total of 0 gives NaN"),
    (tm_start_to_every_touch, "exactly one start -> first-touch transition"),
    (tm_no_start, "exactly one start -> first-touch transition"),
    (tm_unordered, "Reindex rows and columns"),
])
def test_exercise_2_messages(small, wrong, message):
    _fails(message, at.check_transition_matrix, wrong, small, CH)


def test_chain_message(small):
    P = at.transition_matrix(small, CH)
    bent = P.copy()
    bent.loc["retargeting", ["conv", "null"]] = (bent.loc["retargeting", ["conv", "null"]]
                                                 .to_numpy()[::-1])
    with pytest.raises(AssertionError, match="must reproduce that share exactly") as info:
        at.check_chain(at.conversion_probability, at.transition_matrix(at.TINY, ["A", "B"]),
                       bent, small["converted"].mean())
    assert "workshop.use_reference(2)" in str(info.value)


def re_delete_renormalize(P, channels):
    p = at.conversion_probability(P)
    out = {}
    for c in channels:
        Pc = P.drop(index=c, columns=c)
        Pc = Pc.div(Pc.sum(axis=1), axis=0)
        out[c] = 1 - at.conversion_probability(Pc) / p
    return pd.Series(out)


def re_ratio(P, channels):
    return 1 - at.removal_effects(P, channels)


def re_mutates(P, channels):
    p = at.conversion_probability(P)
    out = {}
    for c in channels:
        P.loc[c, :] = 0.0
        P.loc[c, "null"] = 1.0
        out[c] = 1 - at.conversion_probability(P) / p
    return pd.Series(out)


@pytest.mark.parametrize("wrong, message", [
    (re_delete_renormalize, "deleted and the other transitions renormalized"),
    (re_ratio, r"1 - p_-c / p, not p_-c / p"),
    (re_mutates, r"Work on `P.copy\(\)`"),
])
def test_exercise_3_messages(small, wrong, message):
    P = at.transition_matrix(small, CH)
    _fails(message, at.check_removal_effects, wrong, P, CH)


def test_delete_renormalize_values_on_tiny():
    """The brief's computed wrong value: RE_A = -0.67 on TINY."""
    P = at.transition_matrix(at.TINY, ["A", "B"])
    assert re_delete_renormalize(P, ["A", "B"])["A"] == pytest.approx(-2 / 3)


def test_markov_credit_message():
    with pytest.raises(AssertionError, match=r"\{A: 1, B: 2\}"):
        at.check_markov_credit(pd.Series({"A": 1.5, "B": 1.5}))


def shapley_wrong_weight(players, v):
    n = len(players)
    phi = {}
    for c in players:
        others = [p for p in players if p != c]
        phi[c] = sum(math.factorial(k) * math.factorial(n - k) / math.factorial(n)
                     * (v(frozenset(T) | {c}) - v(frozenset(T)))
                     for k in range(n) for T in itertools.combinations(others, k))
    return phi


def shapley_no_difference(players, v):
    n = len(players)
    phi = {}
    for c in players:
        others = [p for p in players if p != c]
        phi[c] = sum(math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                     * v(frozenset(T) | {c})
                     for k in range(n) for T in itertools.combinations(others, k))
    return phi


def shapley_tuple(players, v):
    out = {}
    for c in players:
        others = [p for p in players if p != c]
        out[c] = sum(v(T + (c,)) for k in range(len(players)) for T in
                     itertools.combinations(others, k))
    return out


@pytest.mark.parametrize("wrong, message", [
    (shapley_wrong_weight, r"\(n - \|T\|\)! in place of \(n - \|T\| - 1\)!"),
    (shapley_no_difference, "do not contain c, and compare"),
    (shapley_tuple, "v takes a frozenset"),
])
def test_exercise_4_messages(wrong, message):
    _fails(message, at.check_shapley_values, wrong, seed=2040)


def test_wrong_weight_values_match_the_brief():
    got = shapley_wrong_weight(["A", "B"], at._Game(at.TWO_PLAYER_GAME))
    assert got == pytest.approx({"A": 0.175, "B": 0.100})


def cvt_counts(credit, incremental):
    gap = credit.sub(incremental.reindex(credit.index), axis=0)
    worst = gap.abs().idxmax()
    return pd.DataFrame({"tvd": 0.5 * gap.abs().sum(), "largest_gap_channel": worst,
                         "largest_gap": [gap.loc[worst[m], m] for m in gap.columns]})


def cvt_no_half(credit, incremental):
    out = at.credit_vs_truth(credit, incremental)
    return out.assign(tvd=2 * out["tvd"])


def cvt_signed_max(credit, incremental):
    shares = credit / credit.sum()
    gap = shares.sub(incremental.reindex(credit.index) / incremental.sum(), axis=0)
    worst = gap.idxmax()
    return pd.DataFrame({"tvd": 0.5 * gap.abs().sum(), "largest_gap_channel": worst,
                         "largest_gap": [gap.loc[worst[m], m] for m in gap.columns]})


@pytest.mark.parametrize("wrong, message", [
    (cvt_counts, "Compare shares, not counts"),
    (cvt_no_half, "factor 1/2"),
    (cvt_signed_max, "largest absolute gap"),
])
def test_exercise_5_messages(wrong, message):
    credit = pd.DataFrame({"m1": [1.0, 2.0], "m2": [2.0, 1.0]}, index=["x", "y"])
    _fails(message, at.check_credit_vs_truth, wrong, credit, pd.Series({"x": 1.0, "y": 1.0}))
