"""Retailer (Pareto/NBD + covariates + Gamma-Gamma) and plain BG/NBD generators: determinism,
shapes, invariants, and agreement with pymc-marketing's RFM conventions."""

import numpy as np
import pandas as pd
import pytest

from mktstats import synth


@pytest.fixture(scope="module")
def retail():
    return synth.retailer()


@pytest.fixture(scope="module")
def bg():
    return synth.btyd_bgnbd()


def test_retailer_is_deterministic(retail):
    again = synth.retailer()
    for name in retail.data:
        pd.testing.assert_frame_equal(retail.data[name], again.data[name])
    assert retail.truth == again.truth
    other = synth.retailer(seed=1)
    assert not other.transactions.equals(retail.transactions)


def test_retailer_shapes_and_columns(retail):
    tx, cu = retail.transactions, retail.customers
    assert list(tx.columns) == ["customer_id", "date", "amount"]
    assert list(cu.columns[:3]) == ["customer_id", "acquisition_channel", "first_date"]
    for col in ["true_alive_at_cal_end", "true_alive_at_end", "true_lambda", "true_mu",
                "true_mean_spend", "channel_social", "channel_referral"]:
        assert col in cu.columns
    assert len(cu) == 5000 and cu["customer_id"].is_unique
    assert set(tx["customer_id"]) == set(cu["customer_id"])  # every customer bought once at least


def test_retailer_invariants(retail):
    tx, cu, tr = retail.transactions, retail.customers, retail.truth
    first = tx.groupby("customer_id")["date"].min()
    assert (first.reindex(cu["customer_id"]).to_numpy() == cu["first_date"].to_numpy()).all()
    assert cu["first_date"].max() <= pd.Timestamp(tr["acquisition_end"])
    assert tx["date"].min() >= pd.Timestamp(tr["start_date"])
    assert tx["date"].max() <= pd.Timestamp(tr["holdout_end"])
    assert (tx["amount"] > 0).all()
    assert (cu["true_alive_at_end"] <= cu["true_alive_at_cal_end"]).all()
    # nobody who had dropped out by the calibration end buys in the holdout
    dead = set(cu.loc[cu["true_alive_at_cal_end"] == 0, "customer_id"])
    hold = tx.loc[tx["date"] > pd.Timestamp(tr["calibration_end"]), "customer_id"]
    assert not dead & set(hold)
    # indicator columns agree with the channel label; search is the reference
    for c, col in tr["covariates"]["columns"].items():
        assert ((cu["acquisition_channel"] == c).astype(int) == cu[col]).all()
    assert tr["reference_channel"] == "search"


def test_retailer_channel_effects_have_the_stated_sign(retail):
    cu = retail.customers
    lam = cu.groupby("acquisition_channel")["true_lambda"].mean()
    mu = cu.groupby("acquisition_channel")["true_mu"].mean()
    assert lam["social"] < lam["search"] < lam["referral"]  # gamma_purchase -0.4, 0, +0.3
    assert mu["referral"] < mu["search"] < mu["social"]  # gamma_dropout -0.6, 0, +0.6


def test_retailer_new_customer_value_matches_closed_form(retail):
    tr = retail.truth
    r, a0, s, b0 = (tr["pnbd"][k] for k in ("r", "alpha", "s", "beta"))
    h = tr["value_by_channel"]["horizon_weeks"]
    for c, v in tr["value_by_channel"]["new_customer"].items():
        a = a0 * np.exp(-tr["covariates"]["gamma_purchase"][c])
        b = b0 * np.exp(-tr["covariates"]["gamma_dropout"][c])
        # E[X(t)] = (r / alpha) * beta / (1 - s) * (((beta + t) / beta)**(1 - s) - 1), s != 1
        closed = r / a * b / (1 - s) * (((b + h) / b) ** (1 - s) - 1)
        assert v["expected_repeat_purchases"] == pytest.approx(closed, rel=1e-5)
        assert v["discounted_clv_excluding_first_purchase"] < 30 * closed


def test_retailer_existing_value_close_to_holdout(retail):
    # the expected holdout purchases are a forecast; with ~1,000-2,200 customers per channel the
    # observed holdout mean should be within 10%
    for c, v in retail.truth["value_by_channel"]["existing_customers_at_cal_end"].items():
        assert v["observed_holdout_mean_purchases"] == pytest.approx(
            v["mean_expected_purchases"], rel=0.1), c


def test_retailer_rejects_bad_reference():
    with pytest.raises(ValueError, match="reference"):
        synth.retailer(gamma_purchase={"search": 0.1, "social": 0, "referral": 0})


def test_bgnbd_is_deterministic(bg):
    again = synth.btyd_bgnbd()
    pd.testing.assert_frame_equal(bg.rfm, again.rfm)
    pd.testing.assert_frame_equal(bg.transactions, again.transactions)
    assert bg.truth == again.truth


def test_bgnbd_rfm_invariants(bg):
    rfm = bg.rfm
    assert len(rfm) == bg.truth["n_customers"]
    assert (rfm["frequency"] % 1 == 0).all()
    assert (rfm["recency"] <= rfm["T"]).all()
    assert (rfm.loc[rfm["frequency"] == 0, "recency"] == 0).all()
    assert (rfm.loc[rfm["true_alive_at_cal_end"] == 0, "test_frequency"] == 0).all()
    # BG/NBD customers drop out only right after a repeat purchase, so every dead customer has at
    # least two purchase rows (the daily RFM can merge a same-day repeat into the first purchase)
    rows = bg.transactions.groupby("customer_id").size()
    dead = rfm.loc[rfm["true_alive_at_cal_end"] == 0, "customer_id"]
    assert (rows.reindex(dead) >= 2).all()


def test_bgnbd_rfm_matches_pymc_marketing(bg):
    clv_utils = pytest.importorskip("pymc_marketing.clv.utils")
    ref = clv_utils.rfm_summary(bg.transactions, "customer_id", "date",
                                observation_period_end=bg.truth["calibration_end"],
                                time_unit="D", time_scaler=7)
    ref = ref.sort_values("customer_id").reset_index(drop=True)
    ours = bg.rfm[["customer_id", "frequency", "recency", "T"]]
    pd.testing.assert_frame_equal(ours, ref[["customer_id", "frequency", "recency", "T"]],
                                  check_dtype=False)


def test_bgnbd_holdout_counts(bg):
    tx, rfm = bg.transactions, bg.rfm
    hold = tx.loc[tx["date"] > pd.Timestamp(bg.truth["calibration_end"])].drop_duplicates()
    counts = hold.groupby("customer_id").size()
    got = rfm.set_index("customer_id")["test_frequency"]
    assert (got.reindex(counts.index) == counts).all()
    assert got.drop(counts.index).eq(0).all()
