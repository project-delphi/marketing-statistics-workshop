"""The messy log (Module 13): the synthetic retailer's purchases written as a line log shaped like
Online Retail II, with labeled errors added at rates measured on Online Retail II.

Not exported from ``mktstats.synth`` (so the core modules' dependencies are unchanged): import it as
``from mktstats.synth import messy``. The committed copy is ``data/synthetic/messy_lines.csv`` with
its truth in ``data/synthetic/truth_messy.json`` (``scripts/make_synthetic.py``, ``EXTENSIONS``).

Data-generating process (``briefs/13-messy-real-data.md``):

1. **Clean purchases.** The retailer's transactions (``synth.retailer(seed=retailer_seed)``, rates
   per week) for its first ``n_retail`` customers, plus ``n_wholesale`` wholesale customers (IDs
   5001, 5002, ...) acquired on a uniform day of the first 26 weeks, with the retailer's
   reference-channel Pareto/NBD (r 0.8, alpha 5, s 0.5, beta 10 per week) and Gamma-Gamma spend
   with p 6, q 4 and v = 15 x ``wholesale_multiplier`` (mean spend per purchase 40 times the
   retail population mean). The **clean log** is the union.
2. **Invoices and lines.** Each clean purchase becomes one invoice of 1 + Poisson(``extra_lines``)
   product lines with distinct stock codes from a catalog matching ``^[0-9]{5}[A-Z]?$``; quantity x
   price adds up to the purchase amount exactly in cents. Timestamp: the purchase day plus a
   uniform minute between 08:00 and 19:59.
3. **Errors**, each a label (the labels live only in the truth): ``guest`` invoices without a
   customer ID; ``postage`` (POST) and ``carriage`` (C2) lines on identified invoices;
   ``fee_invoice`` (a separate invoice with one ``M`` or ``BANK CHARGES`` line); ``duplicate``
   (an exact copy of a line, all eight columns equal, right after it); ``cancelled_order`` (an
   extra order and a ``C`` invoice negating every line); ``returned_line`` (an extra line on a
   real invoice and a later ``C`` invoice returning it in full); ``same_day_split`` (a purchase's
   lines split into two invoices on the same day); ``wholesale`` (the wholesale customers' lines:
   real customers, not errors). Every cancellation has its original, so the cleaning rules can
   recover the clean log exactly, and no cancellation crosses the last day.
4. **Output.** Table ``lines`` with the columns of ``mktstats.data.load_online_retail_ii()``:
   invoice, stock_code, description, quantity, invoice_date, price, customer_id, country, in
   timestamp order. ``invoice_date`` is a ``"%Y-%m-%d %H:%M"`` string (parse it with
   ``pd.to_datetime``), so the CSV keeps the time of day. Table ``labels`` (one row per line:
   ``label``, ``copy_of``) is for tests only and is not written to the CSV.

Truth: the rates and their measured values, the calibration and holdout ends, each cleaning
rule's count and the sha256 of its flagged positions (:func:`reference_flags`), each label's
count and hash, the wholesale customers (true parameters, clean transactions, true expected
revenue), the same totals for the retail customers, the clean log's realized holdout, and the
``tolerances`` block (``TOLERANCES``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from mktstats.synth._common import SynthResult, make_rng, sig
from mktstats.synth.btyd import _poisson_purchases, retailer

COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
           "customer_id", "country"]
RULES = ("no_customer", "non_product", "duplicate", "cancellation", "nonpositive")
LABELS = ("guest", "postage", "carriage", "fee_invoice", "duplicate", "cancelled_order",
          "returned_line", "same_day_split", "wholesale")

# Rates: share of what each error is added to (see the generator spec in the brief), and the
# real rate it was matched to (measured 2026-10-10 on the table load_online_retail_ii() returns).
DEFAULT_RATES = {
    "n_retail": 2500,                # the retailer's customers 1..n_retail (keeps the CSV small)
    "n_wholesale": 13,               # 0.5% of customers
    "wholesale_multiplier": 40.0,    # v = 15 x 40: mean spend per purchase $1,200
    "extra_lines": 0.25,             # product lines per invoice = 1 + Poisson(extra_lines)
    "guest_invoices": 0.285,         # guest invoices per identified invoice (real: 22.5% of lines)
    "postage": 0.051,                # of identified orders (real: POST on 4.88% of invoices)
    "carriage": 0.007,               # of identified orders (real: C2 on 0.69% of invoices)
    "fee_manual": 0.065,             # of customers: one standalone M invoice
    "fee_bank": 0.02,                # of customers: one standalone BANK CHARGES invoice
    "duplicate": 0.0152,             # of lines outside cancellation pairs (real: 1.44% of lines)
    "cancelled_order": 0.0105,       # of purchases (C lines: real 2.28% of identified lines)
    "returned_line": 0.0175,         # of purchases
    "same_day_cancel": 0.15,         # of cancellations and returns (real: 11.6% same day)
    "same_day_split": 0.08,          # of purchases (real: 10.2% extra same-day invoices)
    "catalog_size": 600,
}

START = pd.Timestamp("2024-01-01")
DAYS = 728                     # 2024-01-01 .. 2025-12-28, the retailer's calibration + holdout
CAL_END_DAY = 363              # 2024-12-29
RETAIL_QTY = (1, 2, 3, 4, 6, 12)
WHOLESALE_QTY = (1, 2, 3, 4, 6, 12, 24, 48, 96)
OPEN_MINUTE, CLOSE_MINUTE = 8 * 60, 20 * 60          # timestamps 08:00 .. 19:59
OTHER_COUNTRIES = ("Germany", "France", "EIRE", "Netherlands", "Spain", "Belgium")

# Tolerances for the Module 13 checkpoints, measured on 2026-10-10 over 30 independent generator
# seeds (messy seeds 3000-3029 with retailer seeds 4000-4029) with the reference pipeline
# (tests/test_synth_messy.py::test_tolerances_hold_over_seeds re-measures them; slow). Each bound
# sits above the largest relative error measured, |estimate / truth - 1|.
TOLERANCE_SEEDS = {"seed": list(range(3000, 3030)), "retailer_seed": list(range(4000, 4030))}
TOLERANCES: dict = {
    "basis": "set above the largest relative error of the reference pipeline over 30 "
             "independent generator seeds",
    "seeds": "messy seeds 3000-3029 with retailer seeds 4000-4029, default rates",
    "holdout_forecast": {
        "what": "BG/NBD maximum-likelihood fit on the reference calibration RFM; expected "
                "purchase days in the 52-week holdout, summed, against the clean log's realized "
                "holdout purchase days",
        "rel": 0.20,
        "largest_measured": 0.178,
        "mean_measured": 0.080,
        "basis": "the tolerance, set above the largest error over 30 independent generator "
                 "seeds (17.8%; mean +8.0%: a BG/NBD over-predicts this Pareto/NBD holdout)",
    },
    "capped_gamma_gamma": {
        "what": "Gamma-Gamma maximum-likelihood fit on the calibration RFM with purchase-day "
                "amounts capped at their 99th percentile (calibration days, all customers), "
                "against the retail truth p 6, q 4, v 15 and mean spend 30",
        "p": {"rel": 0.90, "largest_measured": 0.862},
        "q": {"rel": 0.15, "largest_measured": 0.140},
        "v": {"rel": 0.55, "largest_measured": 0.514},
        "population_mean_spend": {"rel": 0.08, "largest_measured": 0.073},
        "basis": "the largest relative error over 30 independent generator seeds, rounded up "
                 "(p, q and v trade off against each other; their mean p v / (q - 1) is tight)",
    },
}


def positions_sha256(positions: Iterable[int]) -> str:
    """sha256 of the sorted 0-based row positions, written as comma-separated decimals."""
    text = ",".join(str(int(p)) for p in sorted(int(p) for p in positions))
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# cleaning rules (pandas reference; the lab writes them in DuckDB SQL)
# ---------------------------------------------------------------------------------------------
def read_lines(path) -> pd.DataFrame:
    """Read a messy-log CSV with the dtypes of ``load_online_retail_ii()``."""
    df = pd.read_csv(path, dtype={"invoice": str, "stock_code": str, "description": "string",
                                  "quantity": "int64", "price": "float64",
                                  "customer_id": "Int64", "country": str},
                     keep_default_na=False, na_values={"customer_id": [""]})
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="%Y-%m-%d %H:%M")
    return df[COLUMNS]


def reference_flags(lines: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule (True = drop the line), one row per line, in row order.

    ``no_customer``: no customer ID. ``non_product``: the stock code does not start with five
    digits. ``duplicate``: an exact copy (all eight columns, missing values equal) of an earlier
    line. ``cancellation``, on the lines the first three keep: every line of a ``C`` invoice and
    the purchase line each one cancels, the latest line at or before it of the same customer with
    the same stock code, the same price and the opposite quantity. ``nonpositive`` (Module 1's
    last rule, on all lines): quantity <= 0 or price <= 0.
    """
    lines = lines.reset_index(drop=True)
    flags = pd.DataFrame(index=lines.index)
    flags["no_customer"] = lines["customer_id"].isna()
    flags["non_product"] = ~lines["stock_code"].astype(str).str.match(r"^[0-9]{5}")
    flags["duplicate"] = lines.duplicated(subset=COLUMNS, keep="first")
    kept = lines.loc[~(flags["no_customer"] | flags["non_product"] | flags["duplicate"])]
    matches = reference_matches(kept)
    cancel = pd.Series(False, index=lines.index)
    cancel.loc[matches["cancel_id"].to_numpy()] = True
    cancel.loc[matches["original_id"].dropna().astype("int64").to_numpy()] = True
    flags["cancellation"] = cancel
    flags["nonpositive"] = (lines["quantity"] <= 0) | (lines["price"] <= 0)
    return flags[list(RULES)]


def reference_matches(kept: pd.DataFrame) -> pd.DataFrame:
    """``cancel_id``, ``original_id`` (row positions; ``original_id`` missing when unmatched):
    each cancellation line against the latest earlier purchase line with the same customer,
    stock code and price and the opposite quantity (``merge_asof``, backward, ties allowed)."""
    is_c = kept["invoice"].str.startswith("C")
    key = ["customer_id", "stock_code", "price", "quantity"]
    canc = kept.loc[is_c, key + ["invoice_date"]].copy()
    canc["quantity"] = -canc["quantity"]
    canc["cancel_id"] = canc.index.to_numpy()
    buys = kept.loc[~is_c, key + ["invoice_date"]].copy()
    buys["original_id"] = buys.index.to_numpy()
    for df in (canc, buys):
        df["customer_id"] = df["customer_id"].astype("int64")
    m = pd.merge_asof(canc.sort_values(["invoice_date", "cancel_id"]),
                      buys.sort_values(["invoice_date", "original_id"]),
                      on="invoice_date", by=key, direction="backward", allow_exact_matches=True)
    out = m[["cancel_id", "original_id"]].sort_values("cancel_id").reset_index(drop=True)
    out["original_id"] = out["original_id"].astype("Int64")
    return out


def reference_purchases(lines: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Purchase days (customer_id, date, amount) after every rule: the reference pipeline that
    ``TOLERANCES`` and the tests use (the lab's ``purchases`` with nothing kept)."""
    lines = lines.reset_index(drop=True)
    drop = flags[list(RULES)].to_numpy().any(axis=1) | lines["invoice"].str.startswith("C")
    out = lines.loc[~drop, ["customer_id", "invoice_date"]].copy()
    out["amount"] = lines.loc[~drop, "quantity"] * lines.loc[~drop, "price"]
    out["customer_id"] = out["customer_id"].astype("int64")
    out["date"] = out["invoice_date"].dt.normalize()
    return (out.groupby(["customer_id", "date"], as_index=False)["amount"].sum()
            .sort_values(["customer_id", "date"], ignore_index=True))


def clean_transactions(truth: dict, retailer_transactions: pd.DataFrame) -> pd.DataFrame:
    """The clean log the cleaning rules must recover: the retailer's transactions of customers
    1..n_retail plus the wholesale customers' transactions stored in the truth."""
    w = pd.DataFrame(truth["wholesale"]["transactions"],
                     columns=truth["wholesale"]["transactions_columns"])
    w["date"] = pd.to_datetime(w["date"])
    r = retailer_transactions.loc[retailer_transactions["customer_id"]
                                  <= int(truth["rates"]["n_retail"]),
                                  ["customer_id", "date", "amount"]]
    return pd.concat([r, w], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------------------------
def _catalog(rng, n: int) -> np.ndarray:
    nums = rng.choice(np.arange(10000, 99999), size=n, replace=False)
    suffix = rng.choice(np.array(["", "", "", "", "", "A", "B", "C", "L", "S"]), size=n)
    return np.array([f"{a}{b}" for a, b in zip(nums, suffix, strict=True)])


def _split(rng, cents: int, k: int, qty: tuple[int, ...]) -> list[tuple[int, int]]:
    """Split ``cents`` into ``k`` lines (quantity, unit price in cents) that add up exactly."""
    k = int(max(1, min(k, cents)))
    if k == 1:
        parts = [cents]
    else:
        cuts = np.sort(rng.choice(np.arange(1, cents), size=k - 1, replace=False))
        parts = np.diff(np.r_[0, cuts, cents]).tolist()
    out = []
    for part in parts:
        ok = [q for q in qty if part % q == 0]
        w = 1.0 / np.arange(1, len(ok) + 1)
        q = int(ok[rng.choice(len(ok), p=w / w.sum())])
        out.append((q, int(part // q)))
    return out


def _wholesale(rng, n: int, multiplier: float, first_id: int = 5001):
    """Wholesale customers: the retailer's reference-channel Pareto/NBD, spend 40x."""
    r, alpha, s, beta, p, q, v = 0.8, 5.0, 0.5, 10.0, 6.0, 4.0, 15.0 * multiplier
    d0 = rng.integers(0, 7 * 26, size=n)
    lam = rng.gamma(r, 1.0 / alpha, size=n)
    mu = rng.gamma(s, 1.0 / beta, size=n)
    tau = rng.exponential(1.0 / mu)
    active = np.minimum(tau, (DAYS - d0) / 7.0)
    idx, t, _ = _poisson_purchases(rng, lam, active)
    nu = rng.gamma(q, 1.0 / v, size=n)
    cust = np.r_[np.arange(n), idx]
    day = np.r_[d0, d0[idx] + np.floor(7.0 * t).astype(np.int64)]
    amount = np.maximum(np.round(rng.gamma(p, 1.0 / nu[cust]), 2), 0.01)
    order = np.lexsort((day, cust))
    cust, day, amount = cust[order], day[order], amount[order]
    ids = first_id + np.arange(n)
    tx = pd.DataFrame({"customer_id": ids[cust], "date": START + pd.to_timedelta(day, unit="D"),
                       "amount": amount})
    info = pd.DataFrame({"customer_id": ids, "true_lambda": lam, "true_mu": mu,
                         "true_mean_spend": p / nu,
                         "true_alive_at_cal_end": (d0 + 7.0 * tau > CAL_END_DAY + 1).astype(int)})
    params = {"r": r, "alpha": alpha, "s": s, "beta": beta, "p": p, "q": q, "v": v}
    return tx, info, params


def true_expected_revenue(lam, mu, alive, mean_spend, weeks: float) -> np.ndarray:
    """alive x m x lambda x (1 - exp(-mu h)) / mu (lambda h when mu = 0): the true expected spend
    over the next ``weeks`` weeks of a Pareto/NBD customer, undiscounted."""
    lam, mu, alive, m = (np.asarray(a, float) for a in (lam, mu, alive, mean_spend))
    frac = np.where(mu > 0, -np.expm1(-mu * weeks) / np.where(mu > 0, mu, 1.0), weeks)
    return alive * m * lam * frac


def messy_retailer(seed: int = 2033, *, retailer_seed: int = 2026, **rates) -> SynthResult:
    """The messy log: see the module docstring. ``rates`` override ``DEFAULT_RATES``."""
    unknown = set(rates) - set(DEFAULT_RATES)
    if unknown:
        raise TypeError(f"unknown rate(s) {sorted(unknown)}; known: {sorted(DEFAULT_RATES)}")
    R = {**DEFAULT_RATES, **rates}
    rng = make_rng(seed)

    ret = retailer(seed=retailer_seed)
    n_retail = int(R["n_retail"])
    retail_tx = ret.transactions.loc[ret.transactions["customer_id"] <= n_retail]
    retail_cust = ret.customers.loc[ret.customers["customer_id"] <= n_retail]
    w_tx, w_info, w_params = _wholesale(rng, int(R["n_wholesale"]),
                                        float(R["wholesale_multiplier"]))
    clean = (pd.concat([retail_tx, w_tx], ignore_index=True)
             .sort_values(["customer_id", "date"], kind="stable").reset_index(drop=True))
    ids = np.sort(clean["customer_id"].unique())
    wholesale_ids = set(w_info["customer_id"].tolist())
    country = dict(zip(ids.tolist(), np.where(
        rng.random(len(ids)) < 0.85, "United Kingdom",
        rng.choice(np.array(OTHER_COUNTRIES), size=len(ids))).tolist(), strict=True))
    catalog = _catalog(rng, int(R["catalog_size"]))

    # Each line: [ts (minutes from START), invoice seq, line seq, customer, code, desc, qty,
    # price cents, is cancellation invoice, label]
    rows: list[list] = []
    inv_count = [0]

    def new_invoice() -> int:
        inv_count[0] += 1
        return inv_count[0]

    def stamp(day: int) -> int:
        return int(day) * 1440 + int(rng.integers(OPEN_MINUTE, CLOSE_MINUTE))

    def order_lines(cust, ts, cents, inv, label, qty, exclude=()):
        k = 1 + int(rng.poisson(R["extra_lines"]))
        parts = _split(rng, cents, k, qty)
        pool = catalog if not exclude else np.array([c for c in catalog if c not in exclude])
        codes = rng.choice(pool, size=len(parts), replace=False)
        return [[ts, inv, j, cust, str(c), f"ITEM {c}", q, pc, False, label]
                for j, ((q, pc), c) in enumerate(zip(parts, codes, strict=True))]

    first_day = clean.groupby("customer_id")["date"].min()
    clean_days = ((clean["date"] - START).dt.days).to_numpy()
    clean_cents = np.round(clean["amount"].to_numpy() * 100).astype(np.int64)
    clean_ids = clean["customer_id"].to_numpy()
    for cid, day, cents in zip(clean_ids.tolist(), clean_days.tolist(), clean_cents.tolist(),
                               strict=True):
        qty = WHOLESALE_QTY if cid in wholesale_ids else RETAIL_QTY
        base = "wholesale" if cid in wholesale_ids else "clean"
        ts = stamp(day)
        inv = new_invoice()
        lines = order_lines(cid, ts, cents, inv, base, qty)
        if len(lines) >= 2 and rng.random() < R["same_day_split"] / _p_two_or_more(R):
            inv2 = new_invoice()
            ts2 = ts + int(rng.integers(5, 181))
            cut = int(rng.integers(1, len(lines)))
            for line in lines[cut:]:
                line[0], line[1] = ts2, inv2
                line[9] = "same_day_split" if base == "clean" else base
        extra = len(lines)
        if rng.random() < R["postage"]:
            lines.append([ts, inv, extra, cid, "POST", "POSTAGE", 1, 495, False, "postage"])
            extra += 1
        if rng.random() < R["carriage"]:
            lines.append([ts, inv, extra, cid, "C2", "CARRIAGE", 1, 2500, False, "carriage"])
            extra += 1
        if rng.random() < R["returned_line"]:
            used = {line[4] for line in lines}
            code = str(rng.choice(np.array([c for c in catalog if c not in used])))
            q = int(rng.choice(np.array(qty[:4])))
            pc = int(rng.integers(95, 1596))
            lines.append([ts, inv, extra, cid, code, f"ITEM {code}", q, pc, False,
                          "returned_line"])
            rday = day if rng.random() < R["same_day_cancel"] else day + int(rng.integers(1, 31))
            rday = min(rday, DAYS - 1)
            rts = ts + int(rng.integers(5, 121)) if rday == day else stamp(rday)
            rows.append([rts, new_invoice(), 0, cid, code, f"ITEM {code}", -q, pc, True,
                         "returned_line"])
        rows.extend(lines)
        if rng.random() < R["cancelled_order"]:
            oday = min(day + int(rng.integers(0, 15)), DAYS - 1)
            ots = stamp(oday)
            ocents = int(round(rng.gamma(6.0, cents / 6.0))) + 1
            olines = order_lines(cid, ots, ocents, new_invoice(), "cancelled_order", qty)
            if rng.random() < R["same_day_cancel"]:
                cday = oday
            else:
                cday = min(oday + int(rng.integers(1, 15)), DAYS - 1)
            cts = ots + int(rng.integers(5, 121)) if cday == oday else stamp(cday)
            cinv = new_invoice()
            rows.extend(olines)
            rows.extend([[cts, cinv, j, cid, c, d, -q, pc, True, "cancelled_order"]
                         for j, (_, _, _, _, c, d, q, pc, _, _) in enumerate(olines)])

    # standalone fee invoices, on a random day after the customer's first purchase
    for share, code, desc in ((R["fee_manual"], "M", "Manual"),
                              (R["fee_bank"], "BANK CHARGES", "Bank Charges")):
        chosen = ids[rng.random(len(ids)) < share]
        for cid in chosen.tolist():
            f = int((first_day[cid] - START).days)
            day = int(rng.integers(f + 1, DAYS)) if f + 1 < DAYS else f
            pc = int(rng.integers(100, 6001)) if code == "M" else 1500
            rows.append([stamp(day), new_invoice(), 0, cid, code, desc, 1, pc, False,
                         "fee_invoice"])

    # guest invoices (no customer ID)
    n_identified = len({r[1] for r in rows})
    n_guest = int(round(R["guest_invoices"] * n_identified))
    gdays = rng.integers(0, DAYS, size=n_guest)
    gcents = np.maximum(np.round(rng.gamma(6.0, 5.0, size=n_guest) * 100), 50).astype(np.int64)
    for day, cents in zip(gdays.tolist(), gcents.tolist(), strict=True):
        rows.extend(order_lines(None, stamp(day), cents, new_invoice(), "guest", RETAIL_QTY))

    L = pd.DataFrame(rows, columns=["ts", "inv", "seq", "customer_id", "stock_code",
                                    "description", "quantity", "price_c", "is_c", "label"])
    L = L.sort_values(["ts", "inv", "seq"], kind="stable").reset_index(drop=True)
    guest_inv = L.loc[L["customer_id"].isna(), "inv"].unique()
    g_country = dict(zip(guest_inv.tolist(), np.where(
        rng.random(len(guest_inv)) < 0.85, "United Kingdom",
        rng.choice(np.array(OTHER_COUNTRIES), size=len(guest_inv))).tolist(), strict=True))
    L["country"] = [country[c] if c == c and c is not None else g_country[i]
                    for c, i in zip(L["customer_id"].tolist(), L["inv"].tolist(), strict=True)]
    # invoice numbers in timestamp order; cancellations get a C prefix
    first = L.drop_duplicates("inv")[["inv", "is_c"]].reset_index(drop=True)
    number = dict(zip(first["inv"].tolist(),
                      [("C" if c else "") + str(500000 + i)
                       for i, c in enumerate(first["is_c"].tolist())], strict=True))
    L["invoice"] = L["inv"].map(number)

    # exact duplicate lines, right after the copied line; never a line of a cancellation pair
    # (the C line or the purchase it cancels), so a copy never adds a purchase day
    pair = L["label"].isin(["cancelled_order", "returned_line"]).to_numpy()
    copy = ~pair & (rng.random(len(L)) < R["duplicate"])
    pos = np.arange(len(L))
    order = np.argsort(np.r_[pos, pos[copy] + 0.5], kind="stable")
    labels = np.r_[L["label"].to_numpy(), np.full(int(copy.sum()), "duplicate", dtype=object)]
    copy_of = np.r_[np.full(len(L), -1), pos[copy]]
    L = pd.concat([L, L.loc[copy]], ignore_index=True).iloc[order].reset_index(drop=True)
    labels, copy_of = labels[order], copy_of[order]
    new_pos = np.empty(len(order), dtype=np.int64)
    new_pos[order] = np.arange(len(order))
    copy_of = np.where(copy_of >= 0, new_pos[np.clip(copy_of, 0, None)], -1)

    stamp_ts = START + pd.to_timedelta(L["ts"].to_numpy(), unit="min")
    lines = pd.DataFrame({
        "invoice": L["invoice"].astype(str),
        "stock_code": L["stock_code"].astype(str),
        "description": L["description"].astype("string"),
        "quantity": L["quantity"].astype("int64"),
        "invoice_date": pd.Series(stamp_ts).dt.strftime("%Y-%m-%d %H:%M"),
        "price": L["price_c"].to_numpy() / 100.0,
        "customer_id": pd.array(L["customer_id"].tolist(), dtype="Int64"),
        "country": L["country"].astype(str),
    })
    label_frame = pd.DataFrame({"label": labels.astype(str), "copy_of": copy_of})
    truth = _truth(seed, retailer_seed, R, lines, label_frame, clean, ret, retail_cust,
                   w_info, w_tx, w_params)
    return SynthResult("messy", {"lines": lines, "labels": label_frame}, truth)


def _p_two_or_more(R) -> float:
    """P(an order has at least two product lines): split probability is conditioned on it."""
    return float(1.0 - np.exp(-R["extra_lines"]))


def _truth(seed, retailer_seed, R, lines, label_frame, clean, ret, retail_cust, w_info, w_tx,
           w_params) -> dict:
    parsed = lines.assign(invoice_date=pd.to_datetime(lines["invoice_date"],
                                                      format="%Y-%m-%d %H:%M"))
    flags = reference_flags(parsed)
    is_c = parsed["invoice"].str.startswith("C")
    rules = {}
    for name in RULES:
        pos = np.flatnonzero(flags[name].to_numpy())
        rules[name] = {"count": int(len(pos)), "sha256": positions_sha256(pos)}
    rules["cancellation"]["cancel_lines"] = int((flags["cancellation"] & is_c).sum())
    rules["cancellation"]["applies_to"] = "lines that no_customer, non_product and duplicate keep"
    rules["nonpositive"]["applies_to"] = "all lines (Module 1's last rule)"
    labels = {}
    for name in LABELS:
        if name == "wholesale":
            pos = np.flatnonzero(parsed["customer_id"].isin(w_info["customer_id"]).fillna(False))
        else:
            pos = np.flatnonzero(label_frame["label"].to_numpy() == name)
        labels[name] = {"count": int(len(pos)), "sha256": positions_sha256(pos)}

    cal_end = START + pd.Timedelta(days=CAL_END_DAY)
    end = START + pd.Timedelta(days=DAYS - 1)
    hold = clean.loc[clean["date"] > cal_end, ["customer_id", "date"]].drop_duplicates()
    measured = _measured_rates(parsed, flags, label_frame)

    def revenue(info, weeks, alive_col):
        return true_expected_revenue(info["true_lambda"], info["true_mu"], info[alive_col],
                                     info["true_mean_spend"], weeks)

    wholesale = {
        "customer_ids": [int(c) for c in w_info["customer_id"]],
        "spend_multiplier": float(R["wholesale_multiplier"]),
        "params": w_params,
        "true_mean_spend_population": sig(w_params["p"] * w_params["v"] / (w_params["q"] - 1)),
        "customers": [
            {"customer_id": int(row.customer_id), "true_lambda": sig(row.true_lambda),
             "true_mu": sig(row.true_mu), "true_alive_at_cal_end": int(row.true_alive_at_cal_end),
             "true_mean_spend": sig(row.true_mean_spend),
             "true_expected_revenue_26w": sig(r26), "true_expected_revenue_52w": sig(r52)}
            for row, r26, r52 in zip(w_info.itertuples(index=False),
                                     revenue(w_info, 26, "true_alive_at_cal_end"),
                                     revenue(w_info, 52, "true_alive_at_cal_end"), strict=True)],
        "transactions": [[int(c), str(d.date()), float(a)] for c, d, a in
                         zip(w_tx["customer_id"], w_tx["date"], w_tx["amount"], strict=True)],
        "transactions_columns": ["customer_id", "date", "amount"],
        "true_expected_revenue_26w": sig(revenue(w_info, 26, "true_alive_at_cal_end").sum()),
        "true_expected_revenue_52w": sig(revenue(w_info, 52, "true_alive_at_cal_end").sum()),
    }
    retail = {
        "customer_ids": f"1..{int(R['n_retail'])} of retailer(seed={retailer_seed})",
        "n_customers": int(len(retail_cust)),
        "gamma_gamma": ret.truth["gamma_gamma"],
        "population_mean_spend": ret.truth["population_mean_spend"],
        "true_expected_revenue_26w": sig(revenue(retail_cust, 26, "true_alive_at_cal_end").sum()),
        "true_expected_revenue_52w": sig(revenue(retail_cust, 52, "true_alive_at_cal_end").sum()),
    }
    return {
        "description": (
            "The synthetic retailer's purchases (first n_retail customers) plus wholesale "
            "customers, written as an Online-Retail-II-style line log with labeled errors. "
            "Positions are 0-based rows of messy_lines.csv; sha256 is of the sorted positions "
            "joined by commas (mktstats.synth.messy.positions_sha256)."
        ),
        "seed": seed,
        "retailer_seed": retailer_seed,
        "rates": {k: (int(v) if float(v).is_integer() and k.startswith("n_") else v)
                  for k, v in R.items()},
        "measured": measured,
        "start_date": str(START.date()),
        "calibration_end": str(cal_end.date()),
        "holdout_end": str(end.date()),
        "holdout_weeks": 52,
        "rfm_settings": {"time_unit": "D", "time_scaler": 7},
        "n_lines": int(len(lines)),
        "n_customers": int(clean["customer_id"].nunique()),
        "rules": rules,
        "labels": labels,
        "clean": {
            "n_purchases": int(len(clean)),
            "n_purchase_days": int(len(clean[["customer_id", "date"]].drop_duplicates())),
            "holdout_purchase_days": int(len(hold)),
            "rule": "clean log = retailer_transactions for customer_id <= n_retail, plus "
                    "wholesale.transactions",
        },
        "wholesale": wholesale,
        "retail": retail,
        "tolerances": TOLERANCES,
    }


def _measured_rates(lines, flags, label_frame) -> dict:
    """The error rates of the generated log, measured the way the real rates were."""
    ident = lines["customer_id"].notna()
    is_c = lines["invoice"].str.startswith("C")
    real_inv = lines.loc[ident & ~is_c, "invoice"]
    n_inv = real_inv.nunique()
    post_inv = lines.loc[ident & (lines["stock_code"] == "POST"), "invoice"].nunique()
    prod = ident & ~is_c & ~flags["non_product"] & ~flags["duplicate"]
    days = lines.loc[prod].assign(day=lines.loc[prod, "invoice_date"].dt.normalize())
    per_day = days.groupby(["customer_id", "day"])["invoice"].nunique()
    fee_only = (lines.loc[ident & ~is_c].assign(np_=flags["non_product"])
                .groupby("invoice")["np_"].all().sum())
    return {
        "lines_without_customer": sig(float((~ident).mean()), 4),
        "postage_invoices": sig(post_inv / n_inv, 4),
        "fee_only_invoices": sig(float(fee_only) / n_inv, 4),
        "duplicate_identified_lines": sig(float((flags["duplicate"] & ident).sum() / ident.sum()),
                                          4),
        "cancellation_identified_lines": sig(float((is_c & ident).sum() / ident.sum()), 4),
        "extra_same_day_invoices": sig(float((per_day - 1).sum()) / n_inv, 4),
        "lines_per_invoice": sig(len(lines) / lines["invoice"].nunique(), 4),
    }


__all__ = [
    "COLUMNS",
    "DEFAULT_RATES",
    "LABELS",
    "RULES",
    "TOLERANCES",
    "TOLERANCE_SEEDS",
    "clean_transactions",
    "messy_retailer",
    "positions_sha256",
    "read_lines",
    "reference_flags",
    "reference_matches",
    "reference_purchases",
    "true_expected_revenue",
]
