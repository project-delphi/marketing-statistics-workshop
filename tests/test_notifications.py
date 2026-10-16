"""Module 18's helpers (mktstats.notifications): the reference implementations on hand-worked
examples, the policy machinery, and every checkpoint check passing on the reference and
failing on each wrong answer the lab brief lists, with that answer's message."""

import numpy as np
import pandas as pd
import pytest

from mktstats import notifications as nt
from mktstats.synth import notifications as env
from mktstats.synth._common import make_rng


@pytest.fixture(scope="module")
def users():
    return env.notification_users().users.iloc[:3000].reset_index(drop=True)


@pytest.fixture(scope="module")
def send_log(users):
    return nt.send_experiment(users, days=14)


@pytest.fixture(scope="module")
def volume(users):
    return nt.volume_experiment(users, days=30, log=True)


# ------------------------------------------------------------------- reference, by hand
def test_incremental_per_send_by_hand():
    log = pd.DataFrame({
        "eligible": [1, 1, 1, 1, 0, 1],
        "sent": [1, 1, 0, 0, 0, 0],
        "session": [1, 0, 0, 1, 0, 0],
        "margin": [9.0, 0.0, 0.0, 9.0, 0.0, 0.0],
    })
    got = nt.incremental_per_send(log)
    assert got.loc["sessions_per_send", "estimate"] == pytest.approx(0.5 - 1 / 3)
    assert got.loc["margin_per_send", "estimate"] == pytest.approx(4.5 - 3.0)
    se = np.sqrt(0.5 / 2 + (1 / 3) / 3)  # var(ddof=1): [1,0] -> 0.5; [0,1,0] -> 1/3
    assert got.loc["sessions_per_send", "se"] == pytest.approx(se)
    assert got.loc["sessions_per_send", "lo"] == pytest.approx(0.5 - 1 / 3 - nt.Z95 * se)


def test_volume_curve_by_hand():
    per_user = pd.DataFrame({
        "arm": [0, 3, 3, 7], "sends": [0, 30, 36, 60], "opens": [0, 3, 6, 6],
        "opted_out": [False, False, True, True], "margin_7d": [1.0, 2.0, 4.0, 3.0],
        "margin": [10.0, 20.0, 30.0, 15.0],
    })
    w1 = pd.Series([0.0, 0.5, 1.5, 2.0])
    got = nt.volume_curve(per_user, w1)
    assert got.loc[3, "open_rate"] == pytest.approx(9 / 66)  # total opens over total sends
    assert got.loc[3, "opens_week_1"] == pytest.approx(1.0)
    assert got.loc[3, "opt_out"] == pytest.approx(0.5)
    assert got.loc[0, "open_rate"] == 0.0
    assert got.loc[7, "margin_90d"] == 15.0 and got.loc[3, "margin_7d"] == 3.0


def test_forward_margin_by_hand():
    log = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 2], "day": [0, 1, 2, 0, 1, 2],
                        "margin": [1.0, 2.0, 4.0, 0.0, 9.0, 0.0]})
    fwd = nt.forward_margin(log, horizon=2)
    assert fwd.tolist()[:2] == [3.0, 6.0] and np.isnan(fwd.iloc[2])
    assert fwd.tolist()[3:5] == [9.0, 9.0] and np.isnan(fwd.iloc[5])


def test_segment_uplift_by_hand():
    log = pd.DataFrame({
        "eligible": [1] * 6, "segment": ["low"] * 6,
        "sent": [1, 1, 0, 0, 0, 1], "opted_out_today": [0, 1, 0, 0, 0, 0],
    })
    fwd = pd.Series([9.0, 0.0, 0.0, 9.0, 0.0, np.nan])
    value = pd.Series({"low": 10.0, "mid": 20.0, "high": 30.0})
    got = nt.segment_uplift(log, fwd, value, cost_per_send=0.5)
    assert got.loc["low", "cate"] == pytest.approx(4.5 - 3.0)
    assert got.loc["low", "p_optout"] == pytest.approx(0.5)  # the NaN-window send is dropped
    assert got.loc["low", "cost"] == pytest.approx(0.5 * 10 + 0.5)
    assert not bool(got.loc["low", "send"])
    assert np.isnan(got.loc["mid", "cate"])


def test_q_iteration_by_hand():
    Q = nt.q_iteration(nt.TINY_P, nt.TINY_R, nt.TINY_GAMMA, 100)
    np.testing.assert_allclose(Q, nt.TINY_Q, atol=1e-9)


def test_thompson_choose_shapes_and_exploitation():
    rng = make_rng(0)
    sums = np.array([[0.0, 500.0], [0.0, 0.0]])
    counts = np.array([[1000.0, 1000.0], [0.0, 0.0]])
    picks = np.array([nt.thompson_choose(sums, counts, 3.0, rng) for _ in range(500)])
    assert picks.shape == (500, 2)
    assert (picks[:, 0] == 1).mean() > 0.99
    assert 0.3 < (picks[:, 1] == 1).mean() < 0.7


# ------------------------------------------------------------------------ machinery
def test_transition_model_rows_sum_to_one(volume):
    _, log = volume
    P, R, N = nt.transition_model(log)
    assert P.shape == (nt.N_CONTEXTS + 1, 2, nt.N_CONTEXTS + 1)
    visited = N > 0
    np.testing.assert_allclose(P[visited].sum(axis=1), 1.0)
    assert (P[~visited] == 0).all()
    # One transition per user-day except each user's last day; template sends (2, 3) excluded,
    # and the opted-out state's send column is a copy of its skip column.
    assert N[: nt.OUT_STATE].sum() + N[nt.OUT_STATE, 0] <= len(log) - log["user_id"].nunique()
    assert (R >= 0).all()


def test_greedy_policy_stays_within_data(volume, users):
    _, log = volume
    P, R, N = nt.transition_model(log)
    Q = nt.q_iteration(P, nt.with_send_penalty(R, 0.05), 0.9, 200)
    policy, best = nt.greedy_policy(Q, N, min_visits=30)
    for s in range(nt.N_CONTEXTS + 1):
        assert N[s, best[s]] >= 30 or best[s] == 0
    out = env.Simulator(users, 1).run(policy, 5)
    assert out.per_user["sends"].sum() > 0


def test_with_send_penalty():
    R = np.array([[1.0, 2.0], [0.0, 0.5]])
    np.testing.assert_allclose(nt.with_send_penalty(R, 0.25), [[1.0, 1.75], [0.0, 0.25]])
    assert R[0, 1] == 2.0  # not changed in place


def test_evaluate_and_paired_difference(users):
    summ, margins = nt.evaluate(users, {"a": env.fixed_policy(0), "b": env.fixed_policy(3)},
                                days=20)
    assert list(summ.index) == ["a", "b"] and "margin_20d" in summ.columns
    d = nt.paired_difference(margins, "b", "a")
    assert d["difference"] == pytest.approx((margins["b"] - margins["a"]).mean())
    assert d["lo"] < d["difference"] < d["hi"]


def test_run_bandit_runs_and_learns(users):
    out, means = nt.run_bandit(users, 0.0, days=15)
    assert out.per_user["sends"].sum() > 0
    assert list(means.columns) == [*env.ACTIONS, "n"]
    assert means["n"].sum() <= 15 * len(users)


def test_context_index_and_labels():
    idx = nt.context_index(np.array([0, 2]), np.array([0.0, 5.0]), np.array([0, 10]))
    assert idx.tolist() == [0, nt.N_CONTEXTS - 1]
    labels = nt.context_labels()
    assert len(labels) == nt.N_CONTEXTS and labels.loc[0, "segment"] == "low"


# ------------------------------------------------------------- checks fail on wrong answers
def test_check_1_wrong_answers(send_log):
    truth = env.true_same_day_effect(send_log)
    nt.check_incremental_per_send(nt.incremental_per_send, send_log, truth)

    def all_rows(log):  # forgets the eligibility filter
        return nt.incremental_per_send(log.assign(eligible=1))

    with pytest.raises(AssertionError, match="already opted out"):
        nt.check_incremental_per_send(all_rows, send_log, truth)

    def wrong_se(log):
        out = nt.incremental_per_send(log)
        out["se"] = out["se"] * 2
        return out

    with pytest.raises(AssertionError, match="standard error"):
        nt.check_incremental_per_send(wrong_se, send_log, truth)

    def wrong_interval(log):
        out = nt.incremental_per_send(log)
        out["lo"] = out["estimate"] - out["se"]
        return out

    with pytest.raises(AssertionError, match="1.96"):
        nt.check_incremental_per_send(wrong_interval, send_log, truth)
    with pytest.raises(AssertionError, match="not a pandas DataFrame"):
        nt.check_incremental_per_send(lambda log: {"a": 1}, send_log, truth)


def test_check_2_wrong_answers(users, volume):
    per_user, _ = volume
    w1 = nt.opens_week_1(users)
    curve = nt.volume_curve(per_user, w1)
    if int(curve["margin_90d"].idxmax()) == 3 and int(curve["opens_week_1"].idxmax()) == 7:
        nt.check_volume_curve(nt.volume_curve, per_user, w1)

    def mean_of_rates(pu, w):
        out = nt.volume_curve(pu, w)
        out["open_rate"] = pu.assign(r=pu["opens"] / pu["sends"].replace(0, np.nan)).groupby(
            "arm")["r"].mean().fillna(0)
        return out

    with pytest.raises(AssertionError, match="open_rate"):
        nt.check_volume_curve(mean_of_rates, per_user, w1)

    def missing_arm(pu, w):
        return nt.volume_curve(pu, w).drop(index=5)

    with pytest.raises(AssertionError, match="no row for cadence"):
        nt.check_volume_curve(missing_arm, per_user, w1)


def test_check_3_wrong_answers(send_log, volume):
    per_user, _ = volume
    fwd = nt.forward_margin(send_log)
    V = nt.value_at_stake(per_user)
    if (nt.segment_uplift(send_log, fwd, V)["cate"] > 0).all():
        nt.check_segment_uplift(nt.segment_uplift, send_log, fwd, V)

    def optout_on_all_sent_rows(log, mf, value, cost):  # p_optout over rows the effect drops
        out = nt.segment_uplift(log, mf, value, cost)
        live = log[log["eligible"] == 1]
        out["p_optout"] = live[live["sent"] == 1].groupby("segment")["opted_out_today"].mean()
        return out

    with pytest.raises(AssertionError, match="same rows as the effect"):
        nt.check_segment_uplift(optout_on_all_sent_rows, send_log, fwd, V)

    def wrong_cost(log, mf, value, cost):
        out = nt.segment_uplift(log, mf, value, cost)
        out["cost"] = out["p_optout"]
        return out

    with pytest.raises(AssertionError, match="cost is p_optout"):
        nt.check_segment_uplift(wrong_cost, send_log, fwd, V)


def test_check_4_wrong_answers():
    nt.check_thompson(nt.thompson_choose)

    def greedy(sums, counts, sigma, rng):  # no exploration
        return (sums / (counts + 1)).argmax(axis=1)

    with pytest.raises(AssertionError, match="no data"):
        nt.check_thompson(greedy)

    def no_shrink(sums, counts, sigma, rng):  # spread does not fall with data
        return rng.normal(sums / (counts + 1), sigma).argmax(axis=1)

    with pytest.raises(AssertionError, match="sigma / sqrt"):
        nt.check_thompson(no_shrink)

    own = make_rng(0)

    def own_rng(sums, counts, sigma, rng):  # ignores the generator it is given
        return nt.thompson_choose(sums, counts, sigma, own)

    with pytest.raises(AssertionError, match="draw from `rng`"):
        nt.check_thompson(own_rng)

    def draws(sums, counts, sigma, rng):
        return rng.normal(sums / (counts + 1), sigma / np.sqrt(counts + 1))

    with pytest.raises(AssertionError, match="shape"):
        nt.check_thompson(draws)


def test_check_5_wrong_answers():
    nt.check_q_iteration(nt.q_iteration)
    with pytest.raises(AssertionError, match="single sweep"):
        nt.check_q_iteration(lambda P, R, g, n: R + g * (P @ np.zeros(R.shape[0])))

    def no_max(P, R, g, n):
        Q = np.zeros(R.shape)
        for _ in range(n):
            Q = R + g * (P @ Q.mean(axis=1))
        return Q

    with pytest.raises(AssertionError, match="by\n? ?hand|by hand"):
        nt.check_q_iteration(no_max)
    with pytest.raises(AssertionError, match="shape"):
        nt.check_q_iteration(lambda P, R, g, n: np.zeros(3))
