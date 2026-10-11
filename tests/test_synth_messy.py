"""mktstats.synth.messy (Module 13's messy log): determinism, the Online Retail II schema, the
truth file, that each cleaning rule flags exactly the lines its labels added, that the reference
pipeline recovers the clean log exactly, and (slow) the tolerances the checkpoints cite."""

import json

import numpy as np
import pandas as pd
import pytest

from mktstats import data, mle, synth
from mktstats.synth import messy

TRUTH_KEYS = {"about", "files", "description", "seed", "retailer_seed", "rates", "measured",
              "start_date", "calibration_end", "holdout_end", "holdout_weeks", "rfm_settings",
              "n_lines", "n_customers", "rules", "labels", "clean", "wholesale", "retail",
              "tolerances"}
# Error rates measured on Online Retail II (load_online_retail_ii(), 2026-10-10) and the share
# of the messy log's measured rate that may differ from them.
REAL_RATES = {"lines_without_customer": 0.2252, "postage_invoices": 0.0488,
              "fee_only_invoices": 0.0102, "duplicate_identified_lines": 0.0144,
              "cancellation_identified_lines": 0.0228, "extra_same_day_invoices": 0.1005}


@pytest.fixture(scope="module")
def res():
    return messy.messy_retailer()


@pytest.fixture(scope="module")
def parsed(res):
    lines = res.lines.assign(invoice_date=pd.to_datetime(res.lines["invoice_date"],
                                                         format="%Y-%m-%d %H:%M"))
    return lines, messy.reference_flags(lines)


@pytest.fixture(scope="module")
def committed():
    return messy.read_lines(data.repo_file("synthetic/messy_lines.csv"))


def test_deterministic(res):
    again = messy.messy_retailer()
    pd.testing.assert_frame_equal(res.lines, again.lines)
    assert json.dumps(res.truth, sort_keys=True) == json.dumps(again.truth, sort_keys=True)
    other = messy.messy_retailer(seed=2034)
    assert not other.lines["invoice_date"].equals(res.lines["invoice_date"])


def test_committed_file_is_the_default_seed(res, parsed, committed):
    pd.testing.assert_frame_equal(committed, parsed[0][messy.COLUMNS])
    on_disk = json.loads(data.repo_file("synthetic/truth_messy.json").read_text())
    assert TRUTH_KEYS <= set(on_disk)
    assert on_disk["rules"] == res.truth["rules"] and on_disk["seed"] == 2033


def test_schema_matches_online_retail_ii(committed):
    assert list(committed.columns) == list(data.ONLINE_RETAIL_COLUMNS.values())
    expected = {"invoice": "object", "stock_code": "object", "description": "string",
                "quantity": "int64", "invoice_date": "datetime64[ns]", "price": "float64",
                "customer_id": "Int64", "country": "object"}
    assert {c: str(t) for c, t in committed.dtypes.items()} == expected
    assert committed["invoice_date"].is_monotonic_increasing
    assert (committed["invoice_date"].dt.hour.between(8, 23)).all()
    assert committed["invoice_date"].dt.minute.nunique() > 50  # the time of day survives the CSV
    number = committed["invoice"].str.removeprefix("C").astype(int)
    assert number.min() == 500000 and number.is_monotonic_increasing
    is_c = committed["invoice"].str.startswith("C")
    assert (committed.loc[is_c, "quantity"] < 0).all()
    assert (committed.loc[~is_c, "quantity"] > 0).all()
    assert committed.loc[~committed["stock_code"].isin(["POST", "C2", "M", "BANK CHARGES"]),
                         "stock_code"].str.fullmatch(r"[0-9]{5}[A-Z]?").all()


def test_the_csv_stays_small():
    assert data.repo_file("synthetic/messy_lines.csv").stat().st_size <= 2.5e6


def test_truth_keys_and_counts(res):
    t = res.truth
    assert TRUTH_KEYS - {"about", "files"} <= set(t)
    assert set(t["rules"]) == set(messy.RULES) and set(t["labels"]) == set(messy.LABELS)
    for block in (t["rules"], t["labels"]):
        for entry in block.values():
            assert entry["count"] > 0 and len(entry["sha256"]) == 64
    assert t["rules"]["cancellation"]["count"] == 2 * t["rules"]["cancellation"]["cancel_lines"]
    assert t["n_customers"] == t["rates"]["n_retail"] + t["rates"]["n_wholesale"]
    assert len(t["wholesale"]["customer_ids"]) == t["rates"]["n_wholesale"]
    for block in ("holdout_forecast", "capped_gamma_gamma"):
        assert block in t["tolerances"]


def test_error_rates_are_near_online_retail_ii(res):
    for name, real in REAL_RATES.items():
        assert abs(res.truth["measured"][name] / real - 1) < 0.15, (name, res.truth["measured"])


def test_each_rule_flags_exactly_its_labels(res, parsed):
    _, flags = parsed
    label = res.labels["label"].to_numpy()
    copy_of = res.labels["copy_of"].to_numpy()
    origin = np.where(copy_of >= 0, label[np.clip(copy_of, 0, None)], label)
    np.testing.assert_array_equal(flags["no_customer"], origin == "guest")
    np.testing.assert_array_equal(flags["non_product"],
                                  np.isin(origin, ["postage", "carriage", "fee_invoice"]))
    np.testing.assert_array_equal(flags["duplicate"], label == "duplicate")
    np.testing.assert_array_equal(flags["cancellation"],
                                  np.isin(label, ["cancelled_order", "returned_line"]))
    for name in messy.RULES:
        pos = np.flatnonzero(flags[name].to_numpy())
        assert messy.positions_sha256(pos) == res.truth["rules"][name]["sha256"]


def test_every_cancellation_has_one_original_and_no_ties(parsed, res):
    lines, flags = parsed
    kept = lines.loc[~flags[["no_customer", "non_product", "duplicate"]].any(axis=1)]
    m = messy.reference_matches(kept)
    assert m["original_id"].notna().all() and not m["original_id"].duplicated().any()
    # No two candidate purchase lines share a key and a timestamp, so an ASOF join (which may
    # break such ties either way) matches the same lines as merge_asof.
    p = kept.loc[~kept["invoice"].str.startswith("C")]
    assert not p.duplicated(["customer_id", "stock_code", "price", "quantity",
                             "invoice_date"]).any()
    # Clean the whole log, then split: some pairs straddle the calibration end, none the end.
    when = lines["invoice_date"]
    cal = pd.Timestamp(res.truth["calibration_end"]) + pd.Timedelta(days=1)
    orig, canc = when[m["original_id"].astype(int)].to_numpy(), when[m["cancel_id"]].to_numpy()
    assert ((orig < cal) & (canc >= cal)).any()
    assert canc.max() < pd.Timestamp(res.truth["holdout_end"]) + pd.Timedelta(days=1)


def test_the_reference_pipeline_recovers_the_clean_log_exactly(res, parsed):
    from mktstats import checks

    lines, flags = parsed
    tx = messy.reference_purchases(lines, flags)
    clean = messy.clean_transactions(res.truth, synth.retailer(seed=2026).transactions)
    cal = res.truth["calibration_end"]
    ours = data.rfm_summary(tx, "customer_id", "date", "amount", observation_period_end=cal,
                            time_unit="D", time_scaler=7)
    theirs = data.rfm_summary(clean, "customer_id", "date", "amount", observation_period_end=cal,
                              time_unit="D", time_scaler=7)
    checks.frames_agree(ours, theirs, on="customer_id")
    assert len(ours) == res.truth["n_customers"]
    hold = clean.loc[clean["date"] > cal, ["customer_id", "date"]].drop_duplicates()
    assert len(hold) == res.truth["clean"]["holdout_purchase_days"]
    # Module 1's rules alone do not recover it
    m1 = lines.loc[lines["customer_id"].notna() & ~lines["invoice"].str.startswith("C")]
    m1 = m1.assign(amount=m1["quantity"] * m1["price"], date=m1["invoice_date"].dt.normalize())
    m1 = m1.groupby(["customer_id", "date"], as_index=False)["amount"].sum()
    with pytest.raises(AssertionError):
        checks.frames_agree(data.rfm_summary(m1, "customer_id", "date", "amount",
                                             observation_period_end=cal, time_unit="D",
                                             time_scaler=7), theirs, on="customer_id")


def _reference_estimates(result):
    """What the lab's checkpoints compare with the tolerances: the reference forecast error
    and the P99-capped Gamma-Gamma fit."""
    t = result.truth
    lines = result.lines.assign(invoice_date=pd.to_datetime(result.lines["invoice_date"]))
    tx = messy.reference_purchases(lines, messy.reference_flags(lines))
    cal = t["calibration_end"]

    def rfm_of(log):
        return data.rfm_summary(log, "customer_id", "date", "amount", observation_period_end=cal,
                                time_unit="D", time_scaler=7)

    rfm = rfm_of(tx)
    forecast = mle.bgnbd_expected_purchases(mle.bgnbd_mle(rfm), rfm, t["holdout_weeks"]).sum()
    cap = tx.loc[tx["date"] <= cal, "amount"].quantile(0.99)
    capped = rfm_of(tx.assign(amount=tx["amount"].clip(upper=cap)))
    gg = mle.gamma_gamma_mle(capped[capped["frequency"] > 0])
    return {"holdout_forecast": forecast / t["clean"]["holdout_purchase_days"] - 1,
            "p": gg["p"] / 6 - 1, "q": gg["q"] / 4 - 1, "v": gg["v"] / 15 - 1,
            "population_mean_spend": mle.population_mean_spend(gg) / 30 - 1}


def _within(errors, tol):
    assert abs(errors["holdout_forecast"]) <= tol["holdout_forecast"]["rel"], errors
    for k in ("p", "q", "v", "population_mean_spend"):
        assert abs(errors[k]) <= tol["capped_gamma_gamma"][k]["rel"], (k, errors)


def test_the_committed_seed_is_within_its_tolerances(res):
    _within(_reference_estimates(res), res.truth["tolerances"])


def test_the_tolerances_can_fail(res):
    errors = _reference_estimates(res)
    with pytest.raises(AssertionError):
        _within({**errors, "holdout_forecast": 0.25}, res.truth["tolerances"])
    with pytest.raises(AssertionError):
        _within({**errors, "population_mean_spend": 0.081}, res.truth["tolerances"])


@pytest.mark.slow
def test_tolerances_hold_over_seeds():
    """Re-measures TOLERANCES over the 30 independent seeds they were set on (about a minute)."""
    tol = messy.TOLERANCES
    seeds = messy.TOLERANCE_SEEDS
    errors = pd.DataFrame([_reference_estimates(messy.messy_retailer(seed=s, retailer_seed=r))
                           for s, r in zip(seeds["seed"], seeds["retailer_seed"], strict=True)])
    worst = errors.abs().max()
    assert len(errors) == 30
    assert worst["holdout_forecast"] < tol["holdout_forecast"]["rel"]
    assert worst["holdout_forecast"] == pytest.approx(tol["holdout_forecast"]["largest_measured"],
                                                      abs=1e-3)
    assert errors["holdout_forecast"].mean() == pytest.approx(
        tol["holdout_forecast"]["mean_measured"], abs=1e-3)
    for k in ("p", "q", "v", "population_mean_spend"):
        block = tol["capped_gamma_gamma"][k]
        assert worst[k] < block["rel"]
        assert worst[k] == pytest.approx(block["largest_measured"], abs=1e-3), k
