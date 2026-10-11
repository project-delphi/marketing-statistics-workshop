"""The Module 13 lab's own checkpoint cells (labs/src/13-messy-real-data.py) on wrong answers a
learner plausibly writes, and the message each one gets.

The fixture runs the lab's code cells on the reference solutions up to Part D (Online Retail II
is not loaded; Part D's checks get stand-ins for its counts and fits), skipping the
PyMC-Marketing cross-check, which nothing later uses. Each wrong answer is written the way a
learner would write it, f-string braces included."""

import contextlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LAB = Path(__file__).resolve().parent.parent / "labs" / "src" / "13-messy-real-data.py"
NUMBERED = re.compile(r"workshop\.checkpoint\((\d+)\)")
LABELLED = re.compile(r'workshop\.checkpoint\(label="([^"]+)"\)')
COLUMNS = ("invoice, stock_code, description, quantity, invoice_date, price, customer_id,"
           " country")


class _Workshop:
    """Enough of the harness to run the lab's cells outside a notebook: a solution cell binds
    the reference as it is defined, and a checkpoint is a plain block whose error propagates."""

    def solution(self, n):
        return lambda obj: obj

    def checkpoint(self, n=None, label=None):
        return contextlib.nullcontext()


def _code_cells():
    jupytext = pytest.importorskip("jupytext")
    return [c.source for c in jupytext.read(LAB, fmt="py:percent").cells
            if c.cell_type == "code" and c.source.strip()]


def _cell(code, start):
    return next(c for c in code if c.startswith(start))


@pytest.fixture(scope="module")
def lab():
    """(namespace after the code cells before Part D, run on the reference solutions;
    {n or label: source of that checkpoint}; every code cell's source)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mktstats import checks

    code = _code_cells()
    found = {}
    for c in code:
        if m := NUMBERED.search(c) or LABELLED.search(c):
            found[int(m[1]) if m[1].isdigit() else m[1]] = c
    ns = {"QUICK": False, "workshop": _Workshop(), "checks": checks}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        for source in code:
            if source.startswith("retail_lines = data.load_online_retail_ii()"):
                break  # Part D
            if "pymc_marketing" in source or source is found["helper = PyMC-Marketing"]:
                continue
            exec(compile(source, str(LAB), "exec"), ns)  # noqa: S102
            plt.close("all")
    return ns, found, code


def run_checkpoint(lab, key, **names):
    """Run checkpoint ``key`` on a copy of the lab's namespace with ``names`` replaced."""
    ns, found, _ = lab
    scope = {**ns, **names}
    exec(compile(found[key], f"<checkpoint {key}>", "exec"), scope)  # noqa: S102
    return scope


def checkpoint_message(lab, key, **names) -> str:
    with pytest.raises(AssertionError) as err:
        run_checkpoint(lab, key, **names)
    return str(err.value)


# ---- Exercise 1: flag_lines --------------------------------------------------------------------


def fl_without_not(con, table="lines"):
    return con.sql(f"""
        SELECT line_id,
               customer_id IS NULL AS no_customer,
               regexp_matches(stock_code, '^[0-9]{{5}}') AS non_product,
               ROW_NUMBER() OVER (PARTITION BY {COLUMNS} ORDER BY line_id) > 1 AS duplicate
        FROM {table}
        ORDER BY line_id
    """).df()


def fl_single_braces(con, table="lines"):
    # {5} inside an f-string is the number 5: DuckDB gets '^[0-9]5'
    return con.sql(f"""
        SELECT line_id,
               customer_id IS NULL AS no_customer,
               NOT regexp_matches(stock_code, '^[0-9]{5}') AS non_product,
               ROW_NUMBER() OVER (PARTITION BY {COLUMNS} ORDER BY line_id) > 1 AS duplicate
        FROM {table}
        ORDER BY line_id
    """).df()


def fl_partition_with_line_id(con, table="lines"):
    return con.sql(f"""
        SELECT line_id,
               customer_id IS NULL AS no_customer,
               NOT regexp_matches(stock_code, '^[0-9]{{5}}') AS non_product,
               ROW_NUMBER() OVER (PARTITION BY line_id, {COLUMNS} ORDER BY line_id) > 1
                   AS duplicate
        FROM {table}
        ORDER BY line_id
    """).df()


def test_checkpoint_1_passes_on_the_reference(lab):
    run_checkpoint(lab, 1)


def test_checkpoint_1_names_a_test_the_wrong_way_round(lab):
    msg = checkpoint_message(lab, 1, flag_lines=fl_without_not)
    assert "you flag stock code 15056 (toy line 0) as non-product and keep POST" in msg
    assert "wrong way round" in msg and "put NOT before regexp_matches(...)" in msg
    assert "can end in a letter" not in msg


def test_checkpoint_1_names_lost_braces(lab):
    msg = checkpoint_message(lab, 1, flag_lines=fl_single_braces)
    assert "you flag stock code 22000 (toy line 7) as non-product but keep 15056" in msg
    assert "DuckDB gets '^[0-9]5'" in msg and "Inside an f-string write {{5}}" in msg


def test_checkpoint_1_names_line_id_in_the_partition(lab):
    msg = checkpoint_message(lab, 1, flag_lines=fl_partition_with_line_id)
    assert msg.startswith("Checkpoint 1: toy line 1 is an exact copy of line 0, and you do not"
                          " flag it")
    assert "leave line_id out of the PARTITION BY" in msg
    assert "missing" not in msg  # not the message about NULLs in the partition


# ---- Exercise 2: match_cancellations -------------------------------------------------------------

CONDITIONS = {"customer": "c.customer_id = p.customer_id",
              "code": "c.stock_code = p.stock_code",
              "price": "c.price = p.price",
              "quantity": "-c.quantity = p.quantity",
              "time": "c.invoice_date >= p.invoice_date"}


def mc(drop=(), add=(), join="ASOF LEFT JOIN"):
    """A learner's match_cancellations: the reference's conditions less ``drop``, plus ``add``."""
    on = " AND ".join([v for k, v in CONDITIONS.items() if k not in drop] + list(add))

    def match_cancellations(con, table="kept"):
        return con.sql(f"""
            WITH c AS (SELECT * FROM {table} WHERE starts_with(invoice, 'C')),
                 p AS (SELECT * FROM {table} WHERE NOT starts_with(invoice, 'C'))
            SELECT c.line_id AS cancel_id, p.line_id AS original_id
            FROM c {join} p ON {on}
            ORDER BY cancel_id
        """).df()

    return match_cancellations


def test_checkpoint_2_passes_on_the_reference_and_on_mc(lab):
    run_checkpoint(lab, 2)
    run_checkpoint(lab, 2, match_cancellations=mc())


# The four joins that passed the old toy table and failed only in Part D's counts on Online
# Retail II (6,107 reference matches; 14,436, 8,195, 6,348 and 6,104, measured 2026-10-10).
@pytest.mark.parametrize("wrong, expected", [
    (mc(drop=["customer"]), "toy cancellation 11 (customer 1) gets line 10, a later purchase of"
                            " the same item by customer 2"),
    (mc(drop=["code"]), "toy cancellation 14 (stock code 10006) gets line 13"),
    (mc(drop=["price"]), "toy cancellation 17 (price 7.00) gets line 16"),
    (mc(drop=["time"], add=["c.invoice_date > p.invoice_date"]),
     "toy cancellation 19 has the same timestamp as purchase 18"),
], ids=["no customer", "no stock code", "no price", "strict >"])
def test_checkpoint_2_separates_each_missing_condition(lab, wrong, expected):
    msg = checkpoint_message(lab, 2, match_cancellations=wrong)
    assert expected in msg
    assert "Check each condition of the join against the Task text" not in msg


@pytest.mark.parametrize("wrong", [
    mc(drop=["quantity"], add=["c.quantity = p.quantity"]),
    mc(add=["c.invoice = p.invoice"]),
], ids=["quantity sign not flipped", "invoice equality"])
def test_checkpoint_2_names_a_join_that_matches_nothing(lab, wrong):
    msg = checkpoint_message(lab, 2, match_cancellations=wrong)
    assert msg.startswith("Checkpoint 2: none of the 8 toy cancellations matches a purchase")
    assert "-c.quantity = p.quantity, not c.quantity = p.quantity" in msg
    assert "do not join on invoice" in msg


@pytest.mark.parametrize("wrong, expected", [
    (mc(drop=["quantity"]), "toy cancellation 3 returns 1 of the 3 units"),
    (mc(drop=["time"], add=["c.invoice_date <= p.invoice_date"]),
     "toy cancellation 4 is dated before purchase 5"),
    (mc(join="LEFT JOIN"), "toy cancellation 8 gets [7, 6]"),
    (mc(join="ASOF JOIN"), "rows for the 8 cancellation lines of the toy table: use ASOF LEFT"),
], ids=["no quantity", "reversed inequality", "ordinary join", "inner ASOF join"])
def test_checkpoint_2_keeps_its_earlier_messages(lab, wrong, expected):
    assert expected in checkpoint_message(lab, 2, match_cancellations=wrong)


# ---- Exercise 3: variant_effects ---------------------------------------------------------------


def ve(change):
    """A learner's variant_effects that measures each customer's change with ``change``."""

    def variant_effects(scores, reference="reference", threshold=0.10):
        ref = scores[reference]
        return pd.DataFrame({
            "total_change_pct": 100 * (scores.sum() / ref.sum() - 1),
            "customers_moved": (change(scores, ref) > threshold).sum(),
        })

    return variant_effects


INVERTED = ve(lambda s, ref: (1 / s.div(ref, axis=0) - 1).abs())  # reference / variant
INVERTED_NO_ABS = ve(lambda s, ref: 1 / s.div(ref, axis=0) - 1)
OVER_VARIANT = ve(lambda s, ref: s.sub(ref, axis=0).abs() / s)  # |variant - ref| / variant
LOG_RATIO = ve(lambda s, ref: np.log(s.div(ref, axis=0)).abs())
SYMMETRIC = ve(lambda s, ref: s.sub(ref, axis=0).abs().div((s.add(ref, axis=0)) / 2))


def test_checkpoint_3_passes_on_the_reference(lab):
    run_checkpoint(lab, 3)


@pytest.mark.parametrize("wrong", [INVERTED, INVERTED_NO_ABS, OVER_VARIANT],
                         ids=["reference / variant", "the same, no abs", "|diff| / variant"])
def test_checkpoint_3_toy_catches_an_inverted_ratio(lab, wrong):
    msg = checkpoint_message(lab, 3, variant_effects=wrong)
    assert msg.startswith("Checkpoint 3: in the toy's 'down' column customer 4 goes from 20 to"
                          " 18.1, a change of 9.5%")
    assert "Divide the variant by the reference" in msg and "on the sweep" not in msg


@pytest.mark.parametrize("wrong", [LOG_RATIO, SYMMETRIC], ids=["log ratio", "symmetric"])
def test_checkpoint_3_sweep_failure_gives_a_likely_cause(lab, wrong):
    msg = checkpoint_message(lab, 3, variant_effects=wrong)
    assert msg.startswith("Checkpoint 3: customers_moved on the sweep:")
    assert "Your function passed the toy" in msg
    assert "|variant / reference - 1| > threshold, with the reference as the denominator" in msg
    assert msg.endswith("run workshop.use_reference(3).")


# ---- Exercise 4: capped_spend_fit ----------------------------------------------------------------


def cs(clip, before_end=False):
    """A learner's capped_spend_fit, clipping with ``clip(amount, cap)``."""

    def capped_spend_fit(tx, pct, cal_end):
        from mktstats import data, mle

        end = pd.Timestamp(cal_end)
        cap = tx.loc[tx["date"] < end if before_end else tx["date"] <= end,
                     "amount"].quantile(pct / 100)
        capped = data.rfm_summary(tx.assign(amount=clip(tx["amount"], cap)), "customer_id",
                                  "date", "amount", observation_period_end=cal_end,
                                  time_unit="D", time_scaler=7)
        return {"cap": float(cap), **mle.gamma_gamma_mle(capped[capped["frequency"] > 0])}

    return capped_spend_fit


def test_checkpoint_4_passes_on_the_reference_and_on_cs(lab):
    run_checkpoint(lab, 4)
    run_checkpoint(lab, 4, capped_spend_fit=cs(lambda a, cap: a.clip(upper=cap)))


def test_checkpoint_4_names_clip_as_a_floor(lab):
    msg = checkpoint_message(lab, 4, capped_spend_fit=cs(lambda a, cap: a.clip(cap)))
    assert "(mle.MLEError)" in msg and "Series.clip(cap) sets a floor" in msg
    assert "clip(upper=cap)" in msg and msg.endswith("run workshop.use_reference(4).")


def test_checkpoint_4_names_a_cap_without_the_calibration_end(lab):
    ns = lab[0]
    assert (ns["tx"]["date"] == pd.Timestamp(ns["CAL_END"])).any()  # else the check is moot
    msg = checkpoint_message(lab, 4, capped_spend_fit=cs(lambda a, cap: a.clip(upper=cap),
                                                         before_end=True))
    assert re.match(r"Checkpoint 4: your cap is [\d.]+: your cap leaves out the purchase days on"
                    r" the calibration end itself: include the calibration end \(<=\)", msg)


# ---- Part D: the counts, the fits and Checkpoint 5 -----------------------------------------------


def total(n):
    """A stand-in whose .sum() is n, for a flag column or the matched mask."""
    return pd.Series([n])


RIGHT_D1 = {
    "retail_shares": {"no_customer": total(235_287), "non_product": total(5_992),
                      "duplicate": total(11_812)},
    "retail_matches": pd.DataFrame(index=range(17_586)),
    "matched": total(6_107),
    "retail_tx": pd.DataFrame({"customer_id": np.arange(32_721) % 5_839}),
}


def test_the_counts_check_passes_on_the_reference_counts(lab):
    run_checkpoint(lab, "Online Retail II counts", **RIGHT_D1)


@pytest.mark.parametrize("names, use_reference", [
    ({"retail_shares": {**RIGHT_D1["retail_shares"], "duplicate": total(11_800)}}, 1),
    ({"matched": total(6_104)}, 2),
    ({"retail_tx": pd.DataFrame({"customer_id": np.arange(32_700) % 5_839})}, None),
], ids=["flags", "matches", "purchase days"])
def test_the_counts_check_says_to_rerun_d1_first(lab, names, use_reference):
    msg = checkpoint_message(lab, "Online Retail II counts", **{**RIGHT_D1, **names})
    assert ("then rerun D1 (the cell under \"Run · D1\", which computes the counts) and then this"
            " cell. Rerunning this cell alone checks the old counts again.") in msg
    if use_reference:
        assert f"Run workshop.use_reference({use_reference}), then rerun D1" in msg
    else:
        assert "rerun the cell under \"Run · The pipeline\" in Part A, then rerun D1" in msg


def test_the_fits_check_routes_to_the_counts_check(lab):
    fits = {"r": 0.6720, "alpha": 9.4444, "a": 0.15034, "b": 3.0511}
    spend = {"p": 2.4017, "q": 3.5534, "v": 418.79}
    run_checkpoint(lab, "Online Retail II fits", retail_bg=fits, retail_gg=spend)
    msg = checkpoint_message(lab, "Online Retail II fits", retail_bg={**fits, "r": 0.70},
                             retail_gg=spend)
    assert msg.startswith("r on Online Retail II (reference 0.672) is 0.7;")
    assert "Check the formula and the units" not in msg
    assert "the counts check after D1 (\"Online Retail II counts\") tests those: rerun D1" in msg


def test_checkpoint_5_points_to_the_counts_when_the_data_differ(lab):
    ns, _, code = lab
    scope = dict(ns)
    exec(compile(_cell(code, "# @title Solution 5"), str(LAB), "exec"), scope)  # noqa: S102
    index = pd.Index(range(5_008), name="customer_id")
    msg = checkpoint_message((scope, lab[1], code), 5,
                             exp_purchases=pd.Series(1.0, index=index),
                             exp_spend=pd.Series(np.linspace(0, 700, 5_008), index=index))
    assert "clear the bar at £10, 30% and 10%; the reference is 2,212" in msg
    assert ("if the counts check after D1 (\"Online Retail II counts\") failed, fix that"
            " first") in msg


# ---- displays ----------------------------------------------------------------------------------


def test_the_sweep_prints_the_forecast_as_a_whole_number(lab):
    shown = lab[0]["purchase_model"].loc["reference", "52-week forecast"]
    assert re.fullmatch(r"\d{1,3}(,\d{3})*", shown), shown


def test_the_effects_table_names_customers_and_dollar_changes(lab):
    ns, _, code = lab
    scope = dict(ns)
    exec(compile(_cell(code, "def score_change("), str(LAB), "exec"), scope)  # noqa: S102
    effects = scope["effects"]
    row = effects.loc["keep non-product lines"]
    assert re.fullmatch(r"customer 1311: \$0\.00057 → \$213\.\d\d", row["largest_pct_customer"])
    assert re.fullmatch(r"customer 1377: \$29\.65 → \$664\.\d\d", row["largest_change_usd"])
    assert pd.isna(effects.loc["reference", "largest_change_usd"])
