"""Every check passes a good input and fails a broken one with a message that says what to
look at (CONTRIBUTING.md, "Checks")."""

import numpy as np
import pandas as pd
import pytest

from mktstats import checks

RNG = np.random.default_rng(0)
NORMAL = RNG.normal(0.0, 1.0, 20_000)
EXPONENTIAL = RNG.exponential(1.0, 20_000)


def rfm(recency=(0.0, 10.5, 30.0, 3.0)):
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "frequency": [0, 2, 5, 1],
            "recency": list(recency),
            "T": [40.0, 38.0, 36.0, 20.0],
        }
    )


def swapped(df):
    return df.rename(columns={"frequency": "recency", "recency": "frequency"})


# ---- columns -------------------------------------------------------------------------------


def test_columns_passes_and_fails():
    checks.columns(rfm(), ["frequency", "T"])
    with pytest.raises(AssertionError, match=r"missing the column\(s\) \['monetary_value'\]"):
        checks.columns(rfm(), ["frequency", "monetary_value"])


# ---- rfm_table -------------------------------------------------------------------------------


def test_rfm_table_passes():
    checks.rfm_table(rfm(), customer_id="customer_id")


def test_rfm_table_fails_on_swapped_columns_with_fractional_recency():
    with pytest.raises(AssertionError, match="swap"):
        checks.rfm_table(swapped(rfm()))


def test_rfm_table_fails_on_swapped_columns_with_whole_day_recency():
    days = rfm(recency=(0, 70, 200, 21)).assign(T=[300, 300, 300, 300])
    checks.rfm_table(days)
    with pytest.raises(AssertionError, match="swap"):
        checks.rfm_table(swapped(days))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"recency": [0.0, 10.5, 50.0, 3.0]}, "exceeds T"),
        ({"recency": [2.0, 10.5, 30.0, 3.0]}, "no repeat purchase"),
        ({"frequency": [0, -1, 5, 1]}, "whole numbers"),
        ({"recency": [0.0, -1.0, 30.0, 3.0]}, "negative"),
    ],
)
def test_rfm_table_fails_on_broken_values(change, message):
    with pytest.raises(AssertionError, match=message):
        checks.rfm_table(rfm().assign(**change))


def test_rfm_table_fails_on_repeated_customers():
    with pytest.raises(AssertionError, match="one row per customer"):
        checks.rfm_table(rfm().assign(customer_id=[1, 1, 3, 4]), customer_id="customer_id")


def test_rfm_table_fails_on_missing_column():
    with pytest.raises(AssertionError, match="missing"):
        checks.rfm_table(rfm().drop(columns="T"))


# ---- intervals -------------------------------------------------------------------------------


def test_hdi_and_eti_of_a_normal_agree():
    lo, hi = checks.interval(NORMAL, 0.94, "hdi")
    elo, ehi = checks.interval(NORMAL, 0.94, "eti")
    assert lo == pytest.approx(-1.88, abs=0.05) and hi == pytest.approx(1.88, abs=0.05)
    assert elo == pytest.approx(lo, abs=0.05) and ehi == pytest.approx(hi, abs=0.05)


def test_hdi_and_eti_differ_for_a_skewed_posterior():
    lo, _ = checks.interval(EXPONENTIAL, 0.94, "hdi")
    elo, _ = checks.interval(EXPONENTIAL, 0.94, "eti")
    assert lo < 0.01 < elo  # the HDI starts at the mode, the ETI at the 3% quantile
    checks.in_interval(0.005, EXPONENTIAL, prob=0.94, kind="hdi")
    with pytest.raises(AssertionError, match="94% ETI"):
        checks.in_interval(0.005, EXPONENTIAL, prob=0.94, kind="eti", name="r")


def test_in_interval_fails_when_the_truth_is_outside():
    checks.in_interval(0.3, NORMAL, prob=0.94, kind="hdi", name="alpha")
    with pytest.raises(AssertionError, match="true value of alpha, 5, is outside the 94% HDI"):
        checks.in_interval(5.0, NORMAL, prob=0.94, kind="hdi", name="alpha")


def test_interval_needs_a_named_kind_and_a_probability():
    with pytest.raises(ValueError):
        checks.interval(NORMAL, 0.94, "central")
    with pytest.raises(ValueError):
        checks.interval(NORMAL, 94, "hdi")


def test_k_of_K_in_interval():
    truths = {"tv": 0.0, "search": 0.5, "social": 10.0}
    draws = {k: NORMAL for k in truths}
    out = checks.k_of_K_in_interval(truths, draws, k=2, prob=0.94, kind="hdi", name="ROAS")
    assert [v[2] for v in out.values()] == [True, True, False]
    with pytest.raises(AssertionError, match=r"Only 2 of 3 .*Outside: social"):
        checks.k_of_K_in_interval(truths, draws, k=3, prob=0.94, kind="hdi", name="ROAS")


def test_k_of_K_needs_matching_keys():
    with pytest.raises(AssertionError, match="same keys"):
        checks.k_of_K_in_interval({"a": 0.0}, {"b": NORMAL}, k=1)


# ---- close, probability, monotone ------------------------------------------------------------


def test_close():
    checks.close(1.0005, 1.0, rel=1e-3, name="r")
    checks.close([1.0, 2.0], [1.0, 2.01], abs=0.02)
    with pytest.raises(AssertionError, match="r is 1.1; expected 1"):
        checks.close(1.1, 1.0, rel=1e-3, name="r")
    with pytest.raises(AssertionError, match="1 of 2 values"):
        checks.close([1.0, 2.5], [1.0, 2.0], abs=0.1)
    with pytest.raises(ValueError, match="tolerance"):
        checks.close(1.0, 1.0)


def test_probability():
    checks.probability([0.0, 0.5, 1.0])
    with pytest.raises(AssertionError, match=r"in \[0, 1\]"):
        checks.probability([0.2, 1.2], name="P(alive)")
    with pytest.raises(AssertionError, match="missing"):
        checks.probability([0.2, np.nan])


def test_monotone():
    checks.monotone([1, 2, 2, 3])
    checks.monotone([3, 2, 1], increasing=False, strict=True)
    with pytest.raises(AssertionError, match="step 1 goes from 3 to 2"):
        checks.monotone([1, 3, 2])
    with pytest.raises(AssertionError, match="increase"):
        checks.monotone([1, 2, 2], strict=True)


# ---- frames_agree ----------------------------------------------------------------------------


def table():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "first": pd.to_datetime(["1997-01-01", "1997-01-02", "1997-02-01"]),
            "n": [4, 1, 2],
            "spend": [100.5, 6.79, 20.0],
        }
    )


def test_frames_agree_ignores_order_and_dtypes():
    other = table().iloc[::-1].astype({"n": float})
    other["first"] = other["first"].astype("datetime64[us]")
    checks.frames_agree(table(), other, on="customer_id")


def test_frames_agree_fails_on_a_differing_value():
    other = table().assign(n=[4, 2, 2])
    with pytest.raises(AssertionError, match="Column 'n' differs .* for 1 row"):
        checks.frames_agree(table(), other, on="customer_id", names=("SQL", "pandas"))


def test_frames_agree_fails_on_missing_or_repeated_keys():
    with pytest.raises(AssertionError, match="rows"):
        checks.frames_agree(table(), table().iloc[:2], on="customer_id")
    with pytest.raises(AssertionError, match="only one of"):
        checks.frames_agree(table(), table().assign(customer_id=[1, 2, 4]), on="customer_id")
    with pytest.raises(AssertionError, match="repeated"):
        checks.frames_agree(table(), table().assign(customer_id=[1, 1, 3]), on="customer_id")
