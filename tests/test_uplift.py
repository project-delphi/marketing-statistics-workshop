"""Uplift metrics on examples small enough to compute by hand (see mktstats.uplift docstring)."""

import numpy as np
import pytest

from mktstats import uplift

# four customers, already in score order: treated responder, treated non-responder,
# control responder, control non-responder
Y = np.array([1.0, 0.0, 1.0, 0.0])
T = np.array([1, 1, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6])


def test_qini_curve_by_hand():
    # g(k) = Y_T - Y_C * N_T / N_C (0 when N_C = 0): k=1: 1, k=2: 1, k=3: 1 - 1*2/1, k=4: 1 - 1*2/2
    x, g = uplift.qini_curve(Y, T, S)
    np.testing.assert_allclose(x, [0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(g, [0, 1, 1, -1, 0])


def test_uplift_curve_by_hand():
    # f(k) = (Y_T/N_T - Y_C/N_C) * (N_T + N_C): 1*1, 0.5*2, (0.5 - 1)*3, 0*4
    _, f = uplift.uplift_curve(Y, T, S)
    np.testing.assert_allclose(f, [0, 1, 1, -1.5, 0])


def test_qini_coefficient_and_auuc_by_hand():
    # trapezoids of width 0.25 under g: 0.125 + 0.25 + 0 - 0.125; random line ends at g(n) = 0
    assert uplift.qini_coefficient(Y, T, S) == pytest.approx(0.25)
    # perfect order (T,1), (T,0), (C,0), (C,1): g = 0, 1, 1, 1, 0 -> area 0.75
    assert uplift.qini_coefficient(Y, T, S, normalize=True) == pytest.approx(0.25 / 0.75)
    # under f: 0.125 + 0.25 - 0.0625 - 0.1875
    assert uplift.auuc(Y, T, S) == pytest.approx(0.125)
    assert uplift.auuc(Y, T, S, subtract_random=True) == pytest.approx(0.125)


def test_ties_are_evaluated_as_groups():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    t = np.array([1, 0, 1, 0])
    s = np.array([1.0, 1.0, 0.0, 0.0])
    x, g = uplift.qini_curve(y, t, s)
    np.testing.assert_allclose(x, [0, 0.5, 1.0])
    np.testing.assert_allclose(g, [0, 1, 2])
    assert uplift.qini_coefficient(y, t, s) == pytest.approx(0.0)
    # row order inside a tie group does not matter
    perm = np.array([1, 0, 3, 2])
    assert uplift.qini_coefficient(y[perm], t[perm], s[perm]) == pytest.approx(0.0)


def test_constant_score_is_random():
    rng = np.random.default_rng(1)
    y = rng.gamma(2, 10, 200) * rng.binomial(1, 0.1, 200)
    t = rng.binomial(1, 0.5, 200)
    assert uplift.qini_coefficient(y, t, np.zeros(200)) == pytest.approx(0.0, abs=1e-9)
    assert uplift.auuc(y, t, np.zeros(200), subtract_random=True) == pytest.approx(0.0, abs=1e-9)


def test_uplift_at_k_by_hand():
    # top 3: treated outcomes 1, 0 (mean 0.5), control outcome 1 -> -0.5
    assert uplift.uplift_at_k(Y, T, S, k=0.75) == pytest.approx(-0.5)
    assert uplift.uplift_at_k(Y, T, S, k=1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="only one group"):
        uplift.uplift_at_k(Y, T, S, k=0.5)


def test_targeting_rule_and_policy_value_by_hand():
    cate = np.array([4.0, -1.0, 2.0, 0.0])
    # treat when cate * 0.5 > 1: only the first (2 * 0.5 = 1 is not > 1)
    np.testing.assert_array_equal(uplift.targeting_rule(cate, 0.5, 1.0), [1, 0, 0, 0])
    policy = np.array([1, 0, 1, 0])
    # true value: mean of [0.5*4 - 1, 0, 0.5*2 - 1, 0] = 0.25
    assert uplift.policy_value_true(cate, policy, 0.5, 1.0) == pytest.approx(0.25)
    # IPW with e = 0.5: row terms [(5-1)/0.5, (0-1)/0.5, -0/0.5, -2.5/0.5] = [8, -2, 0, -5]
    y = np.array([10.0, 0.0, 0.0, 5.0])
    t = np.array([1, 1, 0, 0])
    assert uplift.policy_value(y, t, policy, 0.5, 1.0, propensity=0.5) == pytest.approx(2.0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        uplift.qini_curve(Y, np.array([1, 2, 0, 0]), S)
    with pytest.raises(ValueError):
        uplift.qini_curve(Y[:3], T, S)
    with pytest.raises(ValueError):
        uplift.policy_value(Y, T, np.ones(4), 0.5, 1.0, propensity=1.0)
