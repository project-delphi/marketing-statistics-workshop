"""Module 18's simulated users and environment (mktstats.synth.notifications): determinism,
shapes, truth keys, the committed file, common random numbers across policies, the potential
outcomes, and the claims the lab makes, checked with the reference solutions on the committed
users in both modes (and, marked slow, on more seeds)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mktstats import notifications as nt
from mktstats.synth import notifications as env

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"


@pytest.fixture(scope="module")
def res():
    return env.notification_users()


@pytest.fixture(scope="module")
def users(res):
    return res.users


@pytest.fixture(scope="module")
def quick(users):
    return users.iloc[: nt.QUICK_USERS].reset_index(drop=True)


def test_deterministic(res):
    again = env.notification_users()
    pd.testing.assert_frame_equal(res.users, again.users)
    assert res.truth == again.truth
    assert not env.notification_users(seed=1, n_users=500).users.equals(
        env.notification_users(seed=2, n_users=500).users)


def test_users_table(res):
    u, t = res.users, res.truth
    assert list(u.columns) == ["user_id", "tenure_days", "sessions_last_30", "purchases_last_90",
                               "segment", "days_since_session", "true_theta", "true_kappa"]
    assert len(u) == t["n_users"] == 20_000
    assert u["user_id"].tolist() == list(range(1, 20_001))
    assert set(u["segment"]) == set(env.SEGMENTS)
    assert (np.array(env.SEGMENTS)[env.segment_of(u["sessions_last_30"])] == u["segment"]).all()
    assert u["days_since_session"].between(0, env.MAX_DAYS_SINCE).all()
    assert (u["true_kappa"] > 0).all()
    assert t["dgp"]["margin_per_order"] == pytest.approx(9.0)
    for key in ("send_experiment", "volume_experiment", "quick_users", "segment_share"):
        assert key in json.loads((SYNTH / "truth_notifications.json").read_text())


def test_committed_file_matches_generator(users):
    on_disk = pd.read_csv(SYNTH / "notifications_users.csv")
    pd.testing.assert_frame_equal(on_disk, users, check_dtype=False)


def test_truth_file_same_day_effect(users):
    truth = json.loads((SYNTH / "truth_notifications.json").read_text())
    log = nt.send_experiment(users)
    assert env.true_same_day_effect(log) == truth["send_experiment"]["true_same_day_effect"]


def test_common_random_numbers(quick):
    """Two policies at one seed share every uniform: a user who visits organically on a day
    under the never-send policy visits under the daily policy too (sessions are a superset),
    and a policy's run is reproducible."""
    never = env.Simulator(quick, 5).run(env.fixed_policy(0), 20, log=True).log
    daily = env.Simulator(quick, 5).run(env.fixed_policy(7), 20, log=True).log
    again = env.Simulator(quick, 5).run(env.fixed_policy(7), 20, log=True).log
    pd.testing.assert_frame_equal(daily, again)
    organic = never["session"].to_numpy() == 1
    live = daily["eligible"].to_numpy() == 1
    # Same user-day rows in both logs: an organic visit survives unless the user uninstalled.
    uninstalled = daily["eligible"].to_numpy() == 0  # only opted-out users lose visits
    assert (daily["session"].to_numpy()[organic & live] == 1).all()
    assert uninstalled.sum() > 0


def test_potential_outcomes_consistent(quick):
    """Where the reminder was sent, the realized outcome equals the send potential outcome;
    where nothing was sent, it equals the skip potential outcome."""
    log = env.Simulator(quick, 6).run(env.random_policy(0.5, 7), 10, log=True).log
    sent = log["action"] == 1
    assert (log.loc[sent, "margin"] == log.loc[sent, "margin_if_send"]).all()
    assert (log.loc[sent, "session"] == log.loc[sent, "session_if_send"]).all()
    skip = log["action"] == 0
    assert (log.loc[skip, "margin"] == log.loc[skip, "margin_if_skip"]).all()
    assert (log.loc[skip, "session"] == log.loc[skip, "session_if_skip"]).all()


def test_step_validation(quick):
    sim = env.Simulator(quick, 1)
    with pytest.raises(ValueError):
        sim.step(np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        sim.step(np.full(len(quick), 4))
    with pytest.raises(ValueError):
        env.fixed_policy(4)
    with pytest.raises(ValueError):
        env.volume_policy(np.array([4] * len(quick)))


def test_opted_out_users_get_nothing(quick):
    out = env.Simulator(quick, 2).run(env.fixed_policy(7), 60, log=True).log
    after = out[(out["eligible"] == 0)]
    assert len(after) > 0
    assert (after["sent"] == 0).all() and (after["opened"] == 0).all()


def test_fatigue_and_habit_mechanics():
    """Fatigue decays by RHO and rises by one per send; a session resets days_since."""
    u = env.notification_users(seed=3, n_users=200).users
    sim = env.Simulator(u, 4)
    sim.step(np.ones(200, dtype=int))
    assert np.allclose(sim.fatigue, 1.0)
    r = sim.step(np.zeros(200, dtype=int))
    assert np.allclose(sim.fatigue, env.RHO)
    assert (sim.days_since[r.session] == 0).all()
    assert (sim.days_since[~r.session] == r.days_since_before[~r.session] + 1).all()


@pytest.mark.parametrize("mode", ["FULL", "QUICK"])
def test_lab_claims_hold_on_committed_users(users, quick, mode):
    """The claims the checkpoints make, in both modes, with the reference solutions."""
    u = users if mode == "FULL" else quick
    log = nt.send_experiment(u)
    truth = env.true_same_day_effect(log)
    nt.check_incremental_per_send(nt.incremental_per_send, log, truth)
    per_user, _ = nt.volume_experiment(u)
    nt.check_volume_curve(nt.volume_curve, per_user, nt.opens_week_1(u))
    fwd = nt.forward_margin(log)
    V = nt.value_at_stake(per_user)
    nt.check_segment_uplift(nt.segment_uplift, log, fwd, V)
    # The openers-vs-non-openers gap is an order of magnitude above the true effect.
    sent = log[log["sent"] == 1]
    gap = sent.groupby("opened")["margin"].mean().diff().iloc[-1]
    assert gap > 5 * truth["margin_per_send"]


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(2050, 2060))
def test_claims_over_seeds(seed):
    """Over other simulator seeds (same users): the 95% interval covers the true same-day
    effect for both outcomes on at least 9 of 10 seeds, the daily arm has the most week-1
    opens, and 3 a week the highest 90-day margin."""
    users = env.notification_users().users
    log = nt.send_experiment(users, seed=seed, policy_seed=seed + 100)
    est = nt.incremental_per_send(log)
    truth = env.true_same_day_effect(log)
    test_claims_over_seeds.covered = getattr(test_claims_over_seeds, "covered", []) + [
        all(est.loc[k, "lo"] <= truth[k] <= est.loc[k, "hi"] for k in truth)]
    per_user, _ = nt.volume_experiment(users, seed=seed, arms_seed=seed + 200)
    curve = nt.volume_curve(per_user, nt.opens_week_1(users, seed=seed, arms_seed=seed + 200))
    assert int(curve["opens_week_1"].idxmax()) == 7
    assert int(curve["margin_90d"].idxmax()) == 3
    if seed == 2059:
        assert sum(test_claims_over_seeds.covered) >= 9
