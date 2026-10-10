"""Module 16's journey generator (mktstats.synth.mta): determinism, shapes, truth keys, the
committed file, the random-exposure and holdout options, and the claims the lab makes, checked
with the reference solution on the committed data (and, marked slow, on more seeds)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from mktstats import attribution as at
from mktstats.synth import mta

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
CH = list(mta.CHANNELS)
PAID = ["display", "social", "search", "retargeting"]


@pytest.fixture(scope="module")
def res():
    return mta.mta_journeys()


@pytest.fixture(scope="module")
def credit(res):
    return _credit(res.journeys)


def _credit(j):
    table = pd.DataFrame({r: at.heuristic_credit(j, r, CH) for r in ("last", "first", "linear")})
    P = at.transition_matrix(j, CH)
    table["markov"] = at.markov_credit(at.removal_effects(P, CH), j["converted"].sum())
    table["shapley"] = at.shapley_credit(at.coalition_table(j), CH)[0]
    return table


def test_deterministic(res):
    again = mta.mta_journeys()
    pd.testing.assert_frame_equal(res.journeys, again.journeys)
    assert res.truth == again.truth
    assert not mta.mta_journeys(seed=1, n_users=5_000).journeys.equals(
        mta.mta_journeys(seed=2, n_users=5_000).journeys)


def test_journeys_table(res):
    j, t = res.journeys, res.truth
    assert list(j.columns) == ["user_id", "path", "converted", "true_intent"]
    assert j["user_id"].is_monotonic_increasing and j["user_id"].is_unique
    assert j["user_id"].between(1, t["n_users"]).all()
    assert set(j["converted"].unique()) == {0, 1}
    touches = j["path"].str.split(" > ").explode()
    assert set(touches) == set(CH)
    assert len(j) == t["n_journeys"]
    assert j["converted"].sum() == t["conversions_in_journeys"]
    assert t["conversions_total"] == t["conversions_in_journeys"] + t["conversions_untracked"]
    assert t["conversions_untracked"] > 0  # users with no touch also convert


def test_truth_keys(res):
    t = res.truth
    want = {"seed", "n_users", "channels", "path_separator", "path_codes", "csv_encoding", "dgp",
            "n_journeys", "conversions_total", "conversions_in_journeys", "conversions_untracked",
            "incremental_conversions", "incremental_conversions_expected",
            "incremental_conversions_all_channels",
            "incremental_conversions_all_channels_expected", "definition", "reach_share",
            "mean_intent_reached", "conversion_rate_reached", "conversion_rate_not_reached",
            "economics", "true_cost_per_incremental_conversion", "random_exposure_reach_prob"}
    assert set(t) == want
    assert t["seed"] == 2036 and t["n_users"] == 100_000 and t["channels"] == CH
    json.dumps(t)  # JSON-serialisable
    for key in ("incremental_conversions", "reach_share", "mean_intent_reached",
                "conversion_rate_reached", "conversion_rate_not_reached",
                "random_exposure_reach_prob", "true_cost_per_incremental_conversion"):
        assert list(t[key]) == CH, key


def _make_synthetic():
    spec = importlib.util.spec_from_file_location("make_synthetic",
                                                  ROOT / "scripts" / "make_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_files_decode_to_the_generator(res):
    csv = pd.read_csv(SYNTH / "mta_journeys.csv")
    assert set(csv["path"].str.cat()) <= set(mta.PATH_CODES)
    pd.testing.assert_frame_equal(mta.decode_paths(csv), res.journeys)
    committed = json.loads((SYNTH / "truth_mta.json").read_text())
    _, truth = _make_synthetic().build_extension("mta")
    assert committed == json.loads(json.dumps(truth))
    assert committed["files"] == ["mta_journeys.csv"]
    assert (SYNTH / "mta_journeys.csv").stat().st_size < 2 * 1024 * 1024


def test_economics_match_the_core_truth(res):
    core = json.loads((SYNTH / "truth.json").read_text())
    econ = res.truth["economics"]
    assert econ["mean_order_value"] == core["retailer"]["population_mean_spend"]
    assert econ["gross_margin"] == core["channel_value"]["margin"]
    assert econ["margin_per_conversion"] == pytest.approx(9.0)
    assert econ["scalable"]["email"] is False
    inc = res.truth["incremental_conversions"]
    for c in CH:
        cost = res.truth["true_cost_per_incremental_conversion"][c]
        if inc[c] == 0:
            assert cost is None
        else:
            assert cost == pytest.approx(econ["spend"][c] / inc[c], rel=1e-5)


def test_incremental_truth(res):
    t = res.truth
    inc, exp = t["incremental_conversions"], t["incremental_conversions_expected"]
    assert inc["retargeting"] == 0 and exp["retargeting"] == 0  # theta = 0 by construction
    for c in CH:
        assert abs(inc[c] - exp[c]) < 4 * np.sqrt(max(exp[c], 1)) + 5, c
    # effects add on the log-odds scale, so single-channel effects overlap
    assert t["incremental_conversions_all_channels"] < sum(inc.values())
    # about a third of all conversions are incremental (brief, claim 1)
    assert 0.25 < t["incremental_conversions_all_channels"] / t["conversions_total"] < 0.4


def test_gauss_hermite_reach_matches_monte_carlo():
    z = np.random.default_rng(0).standard_normal(2_000_000)
    flat = mta.random_exposure_reach_prob()
    for c in CH:
        mc = expit(mta.REACH_A[c] + mta.REACH_G[c] * z).mean()
        assert flat[c] == pytest.approx(mc, abs=1e-3), c


def test_random_exposure_reuses_every_draw(res):
    flat = mta.mta_journeys(random_exposure=True)
    j0 = res.journeys.set_index("user_id")
    j1 = flat.journeys.set_index("user_id")
    both = j0.index.intersection(j1.index)
    sets0 = j0.loc[both, "path"].str.split(" > ").map(frozenset)
    sets1 = j1.loc[both, "path"].str.split(" > ").map(frozenset)
    same = both[(sets0 == sets1).to_numpy()]
    assert len(same) > 20_000
    # the same channels reached -> the same touches, times and outcome
    pd.testing.assert_frame_equal(j0.loc[same], j1.loc[same])
    t = flat.truth
    for c in CH:
        assert t["reach_share"][c] == pytest.approx(t["random_exposure_reach_prob"][c], abs=0.005)
        assert abs(t["mean_intent_reached"][c]) < 0.03, c
    assert t["dgp"]["random_exposure"] is True
    assert t["incremental_conversions"]["retargeting"] == 0


def test_holdout_has_its_own_stream(res):
    h = mta.mta_journeys(holdout={"retargeting": 0.2})
    ho = h.holdout
    assert list(ho.columns) == ["user_id", "holdout", "converted"]
    reached = res.journeys.loc[res.journeys["path"].str.contains("retargeting"), "user_id"]
    assert set(ho["user_id"]) == set(reached)  # every user the channel would reach
    assert ho["holdout"].mean() == pytest.approx(0.2, abs=0.01)
    held = set(ho.loc[ho["holdout"] == 1, "user_id"])
    j0 = res.journeys.set_index("user_id")
    j1 = h.journeys.set_index("user_id")
    others = [u for u in j0.index if u not in held]
    pd.testing.assert_frame_equal(j0.loc[others], j1.loc[others])
    held_paths = j1.loc[j1.index.isin(held), "path"]
    assert not held_paths.str.contains("retargeting").any()
    # retargeting has no effect, so withholding it changes no outcome
    conv0 = j0["converted"].reindex(ho["user_id"], fill_value=0).to_numpy()
    assert (conv0 == ho["converted"].to_numpy()).all()
    th = h.truth["holdout"]
    assert th["users_eligible"] == len(ho) and th["users_held_out"] == len(held)
    assert th["incremental_conversions_served"] == 0
    with pytest.raises(ValueError):
        mta.mta_journeys(n_users=100, holdout={"retargeting": 0.2, "email": 0.1})
    with pytest.raises(ValueError):
        mta.mta_journeys(n_users=100, holdout={"tv": 0.2})


# --------------------------------------------- the lab's claims, on the committed data
def test_chain_reproduces_the_observed_share(res):
    j = res.journeys
    P = at.transition_matrix(j, CH)
    assert at.conversion_probability(P) == pytest.approx(j["converted"].mean(), abs=1e-12)
    re = at.removal_effects(P, CH)
    assert re.between(0, 1).all() and re.sum() > 1.5  # removal effects overlap
    collapsed = j.assign(path=j["path"].str.split(" > ").map(
        lambda p: " > ".join(c for i, c in enumerate(p) if i == 0 or c != p[i - 1])))
    re_c = at.removal_effects(at.transition_matrix(collapsed, CH), CH)
    np.testing.assert_allclose(re_c, re, atol=1e-9)  # self-loops do not change Markov credit


def test_predict_answers_are_not_near_a_boundary(res, credit):
    """Exercise 1's bins (20%, 30%) and Exercise 2's options (briefs/16-...: Predict)."""
    last_rt = credit.loc["retargeting", "last"] / credit["last"].sum()
    assert last_rt > 0.33  # the answer 'over 30%' with a margin of more than 3 points
    row = at.transition_matrix(res.journeys, CH).loc["display"]
    options = {"display": row["display"], "other": row[[c for c in CH if c != "display"]].sum(),
               "conv": row["conv"], "null": row["null"]}
    ranked = sorted(options.values(), reverse=True)
    assert max(options, key=options.get) == "display" and ranked[0] - ranked[1] > 0.15


def test_credit_against_the_truth(res, credit):
    j, t = res.journeys, res.truth
    assert np.allclose(credit.sum(), j["converted"].sum())
    inc = pd.Series(t["incremental_conversions"])
    out = at.credit_vs_truth(credit, inc)
    for m in ("last", "linear", "markov", "shapley"):
        assert out.loc[m, "largest_gap_channel"] == "retargeting", m
    assert out.loc["first", "largest_gap_channel"] == "search"
    assert out.loc["markov", "tvd"] < out.loc["last", "tvd"]
    assert out.loc["shapley", "tvd"] < out.loc["last", "tvd"]
    share = credit.loc["retargeting"] / credit.sum()
    assert 0.2 < share["markov"] < 0.27 and 0.2 < share["shapley"] < 0.27


def test_every_channel_set_occurs_and_shapley_shares_are_positive(res):
    coal = at.coalition_table(res.journeys)
    assert len(coal) == 31 and coal["journeys"].min() > 100
    _, shares = at.shapley_credit(coal, CH)
    assert np.nanmin(shares.to_numpy()) > 0
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-12)


def test_count_based_shapley_splits_equally(res):
    """Stretch 3: with v(S) = conversions of journeys using only channels in S, Shapley
    values equal an equal split over each converting journey's distinct channels."""
    coal = at.coalition_table(res.journeys)
    conv = dict(zip(coal["channels"], coal["conversions"], strict=True))
    phi = at.shapley_values(CH, lambda S: sum(n for T, n in conv.items() if T <= S))
    split = dict.fromkeys(CH, 0.0)
    for T, n in conv.items():
        for c in T:
            split[c] += n / len(T)
    assert phi == pytest.approx(split, abs=1e-9)


def test_bootstrap_intervals_miss_the_truth(res):
    j, t = res.journeys, res.truth
    arr = at.bootstrap_credit(j, CH, 200, np.random.default_rng(2036))
    shares = arr / arr.sum(axis=1, keepdims=True)
    lo, hi = np.quantile(shares, [0.025, 0.975], axis=0)
    inc = pd.Series(t["incremental_conversions"]).reindex(CH)
    true = (inc / inc.sum()).to_numpy()[:, None]
    assert int(((lo <= true) & (true <= hi)).sum()) == 0


def test_decision_numbers(res, credit):
    """Retargeting is cheapest per credited conversion under both data-driven methods, and a
    20% holdout's MDE, scaled to every user it reaches, is below its break-even."""
    spend = pd.Series(res.truth["economics"]["spend"])
    for m in ("markov", "shapley"):
        cost = (spend / credit[m])[PAID]
        assert cost.idxmin() == "retargeting", m
        assert (cost < 9).all(), m  # by attribution every paid channel pays back
    j = res.journeys
    has = j["path"].str.contains("retargeting")
    n, p = int(has.sum()), j.loc[has, "converted"].mean()
    n0 = round(0.2 * n)
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(p * (1 - p) * (1 / (n - n0)
                                                                             + 1 / n0))
    assert mde * n < spend["retargeting"] / 9
    true_cost = res.truth["true_cost_per_incremental_conversion"]
    assert true_cost["display"] > 9 and true_cost["retargeting"] is None


# ------------------------------------------------------------------ more seeds (slow)
@pytest.mark.slow
@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15, 16, 17])
def test_claims_on_more_seeds(seed):
    r = mta.mta_journeys(seed)
    cr = _credit(r.journeys)
    out = at.credit_vs_truth(cr, pd.Series(r.truth["incremental_conversions"]))
    for m in ("last", "linear", "markov", "shapley"):
        assert out.loc[m, "largest_gap_channel"] == "retargeting", m
    assert out.loc["markov", "tvd"] < out.loc["last", "tvd"]
    assert out.loc["shapley", "tvd"] < out.loc["last", "tvd"]
    assert len(at.coalition_table(r.journeys)) == 31
    flat = mta.mta_journeys(seed, random_exposure=True)
    cf = _credit(flat.journeys)
    of = at.credit_vs_truth(cf, pd.Series(flat.truth["incremental_conversions"]))
    assert of.loc["shapley", "tvd"] < out.loc["shapley", "tvd"]
    share = cf.loc["retargeting"] / cf.sum()
    assert 0.05 < share["shapley"] < 0.15 and 0.1 < share["markov"] < 0.2
