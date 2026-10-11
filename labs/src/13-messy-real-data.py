# ---
# jupyter:
#   mktstats:
#     verified: >-
#       2026-10-10, Technical Expert, against installed source and by running. DuckDB 1.3.2:
#       regexp_matches, starts_with, ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) with
#       missing values equal inside a partition, ASOF LEFT JOIN with equality conditions
#       (including -c.quantity = p.quantity) and one inequality; flags and matches equal a
#       pandas implementation (duplicated, merge_asof backward) on both logs, except that
#       ASOF picks either of two identical purchase lines on the same invoice and minute
#       (Online Retail II has a few; purchase days and invoices are the same either way).
#       pymc-marketing 1.2.0: BetaGeoModel and GammaGammaModel with Prior("HalfFlat") for
#       every parameter, fit(method="map", progressbar=False), fit_summary(); mktstats.mle
#       transcribes BetaGeoNBD.logp, GammaGammaModel.build_model, expected_purchases and
#       expected_customer_spend (tests/test_mle.py). pandas 2.2.3: Series.quantile(0.99) of
#       1..100 is 99.01; quantile(99) raises ValueError ("percentiles should all be in the
#       interval [0, 1]"). Reference counts below were measured on the files with the sha256
#       in mktstats.data.DATASETS and in data/synthetic.
# ---

# %% [markdown]
# # Part A · A dirty log with a known clean answer
#
# Module 1 cleaned UCI Online Retail II with three rules (drop lines without a customer ID, drop
# invoices that start with `C`, drop `quantity <= 0` or `price <= 0`) and checked them only by row
# counts. This lab builds a cleaning pipeline you can prove right, measures what each cleaning
# choice does to the models of Modules 2 and 4 (the **BG/NBD**: how often each customer buys and
# the chance they have stopped; the **Gamma-Gamma**: how much they spend per purchase), and then
# runs the same pipeline on the real log to choose whom to contact.
#
# Four terms used throughout. A **line** is one row of the log: one product on one invoice. An
# **invoice** is one order, usually several lines. A **purchase day** is all of a customer's
# invoices on one calendar day, merged into one purchase, as in Module 1. A **cancellation line**
# is a line on an invoice whose number starts with `C`; its quantity is negative.
#
# The **messy log** loaded below has exactly the columns of Online Retail II, with the same kinds
# of mess at the same rates (measured on Online Retail II). It was made from the synthetic
# retailer of Modules 1 to 5 (its customers 1 to 2,500, plus 13 wholesale buyers), so the clean
# answer is known. Its amounts are in dollars; Online Retail II's are in pounds sterling.
#
# **QUICK changes nothing in this lab:** no model is sampled, and every fit takes a fraction of a
# second, so the whole lab is a few minutes of compute (an estimate until the run records on the
# readiness page say otherwise). The Decision cell still prints the mode.

# %%
import hashlib
import json
import time
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mktstats import data, mle

con = duckdb.connect()  # one in-memory database for the whole lab
truth = json.loads(data.repo_file("synthetic/truth_messy.json").read_text())  # the messy truth
retail_truth = data.load_truth()["retailer"]  # the synthetic retailer's truth (Modules 1 to 5)
CAL_END = truth["calibration_end"]  # 2024-12-29: the messy log's calibration end

lines = pd.read_csv(data.repo_file("synthetic/messy_lines.csv"),
                    dtype={"invoice": str, "stock_code": str, "description": "string",
                           "customer_id": "Int64", "country": str})
lines["invoice_date"] = pd.to_datetime(lines["invoice_date"], format="%Y-%m-%d %H:%M")
lines.insert(0, "line_id", np.arange(len(lines)))  # the row position: the truth refers to it
con.register("lines", lines)


def positions_sha256(positions):
    """The truth file's fingerprint of a set of lines: sha256 of the sorted line_ids."""
    text = ",".join(str(int(p)) for p in sorted(int(p) for p in positions))
    return hashlib.sha256(text.encode()).hexdigest()


is_c = lines["invoice"].str.startswith("C")
module1_keeps = (lines["customer_id"].notna() & ~is_c & (lines["quantity"] > 0)
                 & (lines["price"] > 0))
print(f"Messy log: {len(lines):,} lines, {lines['invoice'].nunique():,} invoices,"
      f" {lines['customer_id'].nunique():,} customers")
print(f"Lines without a customer ID: {lines['customer_id'].isna().mean():.1%};"
      f" lines on C (cancellation) invoices: {is_c.mean():.1%}")
print(f"Module 1's three rules keep {module1_keeps.sum():,} lines ({module1_keeps.mean():.0%})")
print("Three lines found by pattern: a POST line, a C invoice line, a line without a customer ID")
pd.concat([lines.loc[lines["stock_code"] == "POST"].head(1), lines.loc[is_c].head(1),
           lines.loc[lines["customer_id"].isna()].head(1)])

# %% [markdown]
# ## Exercise 1 · Flag non-product and duplicate lines (8 minutes)
#
# A **flag** is a true/false column that marks a line for removal without removing it, so each
# rule can be switched off later. Two kinds of line survive Module 1's rules:
#
# - a **non-product line**: its stock code does not start with five digits (postage `POST`,
#   carriage `C2`, manual adjustments `M`, `BANK CHARGES`, test products `TEST001`);
# - an **exact duplicate**: a line equal to an earlier line in all eight columns. The first copy
#   is kept; later copies are flagged.
#
# **Predict.** Module 1's three rules keep about three lines in four of this log (printed above).
# Of the lines they keep, what share will your two new rules flag as non-product or duplicate:
# under 1%, 1 to 10%, or over 10%? Write your guess.
#
# **Task.** Complete `flag_lines(con, table="lines")`: one row per line, ordered by `line_id`,
# with the columns `line_id`, `no_customer` (written for you, as a pattern), `non_product` and
# `duplicate`. A line can carry more than one flag. Two DuckDB tools:
#
# - `regexp_matches(text, pattern)` is true when the pattern matches part of the text; the
#   pattern `'^[0-9]{5}'` means five digits at the start. `non_product` is true when the code
#   does **not** match;
# - `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY line_id)` numbers the rows of each group, as in
#   Module 1.
#
# **Watch the braces.** The query is a Python f-string (that is how `{table}` gets filled in), so
# a brace meant for SQL must be doubled: type the pattern as `'^[0-9]{{5}}'`. Typed with single
# braces, Python turns `{5}` into `5` and the pattern silently changes.
#
# The eight original columns are `invoice, stock_code, description, quantity, invoice_date,
# price, customer_id, country`. Delete the `raise` line when your query is complete. The
# checkpoint first runs your function on a nine-line toy table, then on the messy log.

# %% tags=["exercise"]
def flag_lines(con, table="lines"):
    query = f"""
        SELECT line_id,
               customer_id IS NULL AS no_customer
               -- TODO 1: , <expression> AS non_product
               -- TODO 1: , <expression> AS duplicate
        FROM {table}
        ORDER BY line_id
    """
    raise NotImplementedError("TODO 1")  # delete this line when your query is complete
    return con.sql(query).df()


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def flag_lines(con, table="lines"):
    columns = ("invoice, stock_code, description, quantity, invoice_date, price, customer_id,"
               " country")
    query = f"""
        SELECT line_id,
               customer_id IS NULL AS no_customer,
               NOT regexp_matches(stock_code, '^[0-9]{{5}}') AS non_product,
               ROW_NUMBER() OVER (PARTITION BY {columns} ORDER BY line_id) > 1 AS duplicate
        FROM {table}
        ORDER BY line_id
    """
    return con.sql(query).df()


# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    toy = pd.DataFrame({
        "line_id": range(9),
        "invoice": ["500001"] * 7 + ["500002"] * 2,
        "stock_code": ["15056", "15056", "15056", "POST", "M", "TEST001", "15056BL", "22000",
                       "22000"],
        "description": pd.array(["ITEM 15056"] * 3 + ["POSTAGE", "Manual",
                                 "This is a test product.", "ITEM 15056BL", "ITEM 22000",
                                 "ITEM 22000"], dtype="string"),
        "quantity": [1, 1, 2, 1, 1, 1, 1, 3, 3],
        "invoice_date": pd.to_datetime(["2024-03-01 10:00"] * 7 + ["2024-03-02 11:00"] * 2),
        "price": [2.0, 2.0, 2.0, 4.95, 10.0, 1.0, 3.0, 1.5, 1.5],
        "customer_id": pd.array([7] * 7 + [None, None], dtype="Int64"),
        "country": ["United Kingdom"] * 9,
    })
    con.register("toy_lines", toy)
    got = flag_lines(con, "toy_lines")
    checks.columns(got, ["line_id", "no_customer", "non_product", "duplicate"],
                   name="flag_lines(con, 'toy_lines')")
    assert got["line_id"].tolist() == list(range(9)), (
        "Checkpoint 1: return one row per line of the table, ORDER BY line_id. To move on now,"
        " run workshop.use_reference(1)."
    )
    f = {c: got[c].fillna(False).astype(bool).tolist()
         for c in ("no_customer", "non_product", "duplicate")}
    move_on = " To move on now, run workshop.use_reference(1)."
    assert not (f["duplicate"][0] and f["duplicate"][1]), (
        "Checkpoint 1: you flag both copies of toy line 0 as duplicate. The first copy of a line"
        " is the real one: flag only later copies, ROW_NUMBER() OVER (PARTITION BY <all eight"
        " columns> ORDER BY line_id) > 1." + move_on
    )
    assert not f["duplicate"][0], (
        "Checkpoint 1: you flag toy line 0 as the duplicate and keep its later copy, line 1."
        " Number the copies in line order, ORDER BY line_id (ascending) inside the window, so"
        " the earliest copy is kept." + move_on
    )
    assert not f["duplicate"][2], (
        "Checkpoint 1: toy line 2 equals line 0 except for the quantity, and you flag it as a"
        " duplicate. A duplicate must match in all eight columns: partition by every original"
        " column, not by invoice and stock code." + move_on
    )
    assert not (f["non_product"][0] and not f["non_product"][3]), (
        "Checkpoint 1: you flag stock code 15056 (toy line 0) as non-product and keep POST"
        " (line 3): your test is the wrong way round. non_product is true when the code does"
        " not start with five digits: put NOT before regexp_matches(...)." + move_on
    )
    assert not (f["non_product"][7] and not f["non_product"][0] and not f["non_product"][6]), (
        "Checkpoint 1: you flag stock code 22000 (toy line 7) as non-product but keep 15056 and"
        " 15056BL: the pattern lost its braces. Python fills {5} in an f-string with the number"
        " 5, so DuckDB gets '^[0-9]5' (a digit, then a 5). Inside an f-string write {{5}}:"
        " '^[0-9]{{5}}'." + move_on
    )
    assert not f["non_product"][6], (
        "Checkpoint 1: you flag stock code 15056BL as non-product. A product code can end in a"
        " letter: test only that the code starts with five digits, '^[0-9]{5}'." + move_on
    )
    assert f["duplicate"][1], (
        "Checkpoint 1: toy line 1 is an exact copy of line 0, and you do not flag it as a"
        " duplicate. Flag every copy after the first, ROW_NUMBER() OVER (PARTITION BY <the eight"
        " original columns> ORDER BY line_id) > 1, and leave line_id out of the PARTITION BY:"
        " it differs on every line, so with it no line has a copy." + move_on
    )
    assert f["duplicate"][8], (
        "Checkpoint 1: toy line 8 is an exact copy of line 7, including its missing customer"
        " ID, and you do not flag it as a duplicate. PARTITION BY treats missing values as"
        " equal; a join on = does not." + move_on
    )
    expected = {"no_customer": [False] * 7 + [True, True],
                "non_product": [False, False, False, True, True, True, False, False, False],
                "duplicate": [False, True, False, False, False, False, False, False, True]}
    for col, want in expected.items():
        assert f[col] == want, (
            f"Checkpoint 1: on the toy table your {col} column is {f[col]}; it should be {want}"
            " (lines 3 to 5 are POST, M and TEST001; line 1 copies line 0; lines 7 and 8 have no"
            " customer and are copies of each other)." + move_on
        )
    flags = flag_lines(con)
    assert len(flags) == len(lines), (
        f"Checkpoint 1: {len(flags):,} rows for the {len(lines):,} lines of the messy log: one"
        " row per line." + move_on
    )
    for name in ("no_customer", "non_product", "duplicate"):
        mine = flags.loc[flags[name].fillna(False).astype(bool), "line_id"]
        ref = truth["rules"][name]
        assert len(mine) == ref["count"], (
            f"Checkpoint 1: your {name} column flags {len(mine):,} lines of the messy log; the"
            f" truth is {ref['count']:,}. The toy cases passed, so compare your expression with"
            " the rule in the Task text." + move_on
        )
        assert positions_sha256(mine) == ref["sha256"], (
            f"Checkpoint 1: you flag the right number of lines for {name} ({len(mine):,}) but"
            " not the same lines. For duplicate, check ORDER BY line_id inside the window, so"
            " the earliest copy is kept." + move_on
        )

# %% [markdown]
# The line that starts "Your two new rules flag" is your Predict answer. Then look at the
# customer printed last: find their non-product lines (codes that do not start with five digits)
# and note which invoice each one sits on.

# %%
flags = flag_lines(con)
shares = flags[["no_customer", "non_product", "duplicate"]].astype(bool)
for name in shares:
    print(f"{name:12s}: {shares[name].sum():6,} lines ({shares[name].mean():.1%} of all lines)")
new_rules = (shares["non_product"] | shares["duplicate"]).to_numpy() & module1_keeps.to_numpy()
print(f"Your two new rules flag {new_rules.sum():,} of the {module1_keeps.sum():,} lines Module 1"
      f" keeps: {new_rules.sum() / module1_keeps.sum():.1%}")
# A customer with a standalone M invoice in the calibration period (on a day with no other
# invoice of theirs) and a POST line
day = lines["invoice_date"].dt.normalize()
invoices_that_day = lines.assign(day=day).groupby(["customer_id", "day"])["invoice"].nunique()
m_lines = lines.loc[(lines["stock_code"] == "M") & lines["customer_id"].notna()
                    & (day <= pd.Timestamp(CAL_END))]
standalone = [c for c, d in zip(m_lines["customer_id"], day[m_lines.index], strict=True)
              if invoices_that_day[(c, d)] == 1]
with_post = set(lines.loc[lines["stock_code"] == "POST", "customer_id"].dropna())
sizes = lines.groupby("customer_id").size()
candidates = [c for c in standalone if c in with_post and sizes[c] <= 10] or standalone
example_customer = int(min(candidates))
print(f"Customer {example_customer}: a standalone M invoice and a POST line")
lines.loc[lines["customer_id"] == example_customer].drop(columns=["country", "line_id"])

# %% [markdown]
# **Explain.** Compare with your guess. Two kinds of line Module 1 keeps: a `POST` line on an
# order, and a standalone `M` invoice on a day the customer bought nothing else. Which one adds a
# purchase day to the customer's frequency, and which one only inflates an amount? Point to the
# customer printed above.
#
# <details><summary>Why this solution works</summary>
#
# `regexp_matches` with `'^[0-9]{5}'` tests only the start of the code, so `15056BL` stays a
# product and `POST`, `M`, `C2`, `BANK CHARGES` and `TEST001` are flagged. `ROW_NUMBER()` over a
# partition of all eight columns numbers identical lines in `line_id` order, so `> 1` flags every
# copy after the first; `PARTITION BY` treats two missing customer IDs as equal, which an ordinary
# join on `=` would not. The `POST` line sits on an invoice with products, so it inflates that
# purchase day's amount; the standalone `M` invoice adds a purchase day (frequency + 1) with a
# made-up amount; a duplicate line inflates the amount. On real data the five-digit rule also drops
# a few product-like codes (`PADS`, 19 lines with a customer; `SP1002`, 2): a known approximation.
# Treating exact copies as errors is an assumption too; Part B measures what it does.
# </details>

# %% [markdown]
# The lines no rule flags are the table `kept`, which Exercise 2 reads.

# %%
kept = lines.loc[~shares.any(axis=1).to_numpy()]
con.register("kept", kept)
print(f"kept: {len(kept):,} lines, of which {kept['invoice'].str.startswith('C').sum():,} are"
      " cancellation lines")

# %% [markdown]
# ### Run · An ASOF join on four orders (worked example)
#
# Exercise 2 needs a join you may not have used. An **ASOF join** matches each row on the left to
# the nearest earlier row on the right with the same keys, at most one. Here it is on a case you
# know, last-click attribution: each order gets the customer's latest click at or before it.
# The equality conditions choose the group (here the customer); the one inequality chooses the
# nearest earlier row.
#
# Look at orders 101 and 103: each order gets the customer's latest click at or before it, at
# most one, and `LEFT` keeps orders with none.

# %%
clicks = pd.DataFrame({
    "customer_id": [1, 1, 2],
    "clicked_at": pd.to_datetime(["2024-03-01 09:00", "2024-03-04 18:00", "2024-03-05 12:00"]),
    "campaign": ["spring_email", "search_ad", "social_ad"],
})
orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 1, 2, 3],
    "ordered_at": pd.to_datetime(["2024-03-02 10:00", "2024-03-06 08:00", "2024-03-05 11:00",
                                  "2024-03-06 09:00"]),
})
con.register("clicks", clicks)
con.register("orders", orders)
con.sql("""
    SELECT o.order_id, o.customer_id, o.ordered_at, c.campaign, c.clicked_at
    FROM orders AS o ASOF LEFT JOIN clicks AS c
      ON o.customer_id = c.customer_id AND o.ordered_at >= c.clicked_at
    ORDER BY o.order_id
""").df()

# %% [markdown]
# ## Exercise 2 · Match cancellations to the purchases they cancel (10 minutes)
#
# Module 1 dropped the cancellation lines but kept the purchases they cancel.
#
# **Predict.** Does keeping the cancelled purchases make the average customer look like a bigger
# spender, a more frequent buyer, both, or neither? Write your guess.
#
# **Task.** Complete `match_cancellations(con, table="kept")`, returning one row per cancellation
# line with the columns `cancel_id` and `original_id`: the `line_id` of the cancellation line and
# of the purchase line it cancels (`original_id` missing when there is none). The rule, in words:
# a cancellation line cancels the **latest** purchase line, at or before its own timestamp, of the
# same customer, with the same stock code, the same price and the opposite quantity. Write it as
# an ASOF join, as in the worked example: the equalities choose the group, the inequality on
# `invoice_date` picks the latest earlier purchase. The stub holds the two inputs: `c`, the
# cancellation lines, and `p`, the purchase lines. Delete the `raise` line when your query is
# complete.
#
# Write every part of the rule: four equalities and one inequality. On the messy log a few
# missing conditions change nothing, so the checkpoint first tests each one on a toy table built
# to tell them apart. Stuck after about ten minutes? Run `workshop.use_reference(2)`, go on, and
# come back to this after the Decision.

# %% tags=["exercise"]
def match_cancellations(con, table="kept"):
    query = f"""
        WITH c AS (SELECT * FROM {table} WHERE starts_with(invoice, 'C')),
             p AS (SELECT * FROM {table} WHERE NOT starts_with(invoice, 'C'))
        SELECT c.line_id AS cancel_id, p.line_id AS original_id
        FROM c
        -- TODO 2: join p so that each cancellation line gets the latest purchase line at or
        --         before it of the same customer, stock code and price, opposite quantity
        ORDER BY cancel_id
    """
    raise NotImplementedError("TODO 2")  # delete this line when your query is complete
    return con.sql(query).df()


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def match_cancellations(con, table="kept"):
    query = f"""
        WITH c AS (SELECT * FROM {table} WHERE starts_with(invoice, 'C')),
             p AS (SELECT * FROM {table} WHERE NOT starts_with(invoice, 'C'))
        SELECT c.line_id AS cancel_id, p.line_id AS original_id
        FROM c ASOF LEFT JOIN p
          ON c.customer_id = p.customer_id
         AND c.stock_code = p.stock_code
         AND c.price = p.price
         AND -c.quantity = p.quantity
         AND c.invoice_date >= p.invoice_date
        ORDER BY cancel_id
    """
    return con.sql(query).df()


# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    toy = pd.DataFrame({
        "line_id": range(20),
        "invoice": ["500000", "C500001", "500002", "C500003", "C500004", "500005", "500006",
                    "500007", "C500008", "500009", "500010", "C500011", "500012", "500013",
                    "C500014", "500015", "500016", "C500017", "500018", "C500019"],
        "stock_code": ["10001", "10001", "10002", "10002", "10003", "10003", "10004", "10004",
                       "10004", "10005", "10005", "10005", "10006", "10007", "10006", "10008",
                       "10008", "10008", "10009", "10009"],
        "quantity": [2, -2, 3, -1, -4, 4, 1, 1, -1, 1, 1, -1, 2, 2, -2, 3, 3, -3, 1, -1],
        "invoice_date": pd.to_datetime([
            "2024-01-05 09:00", "2024-01-05 11:30",  # same-day full cancellation: matched
            "2024-01-06 10:00", "2024-01-08 10:00",  # returns 1 of the 3 units: not matched
            "2024-01-09 10:00", "2024-01-10 10:00",  # cancellation before the purchase: no
            "2024-01-11 10:00", "2024-01-12 10:00", "2024-01-13 10:00",  # the later of two
            "2024-01-14 09:00", "2024-01-14 10:00",  # then the same item, by customer 2
            "2024-01-14 12:00",
            "2024-01-15 09:00", "2024-01-15 10:00",  # then another code, same price and units
            "2024-01-15 12:00",
            "2024-01-16 09:00", "2024-01-16 10:00",  # then the same code at another price
            "2024-01-16 12:00",
            "2024-01-17 09:00", "2024-01-17 09:00",  # cancelled in the same minute
        ]),
        "price": [2.5, 2.5, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0,
                  7.0, 6.5, 7.0, 8.0, 8.0],
        "customer_id": pd.array([1] * 10 + [2] + [1] * 9, dtype="Int64"),
    })
    con.register("toy_kept", toy)
    got = match_cancellations(con, "toy_kept")
    checks.columns(got, ["cancel_id", "original_id"], name="match_cancellations(con, 'toy_kept')")
    move_on = " To move on now, run workshop.use_reference(2)."
    pairs = [(int(c), None if pd.isna(o) else int(o))
             for c, o in zip(got["cancel_id"], got["original_id"], strict=True)]
    assert len(pairs) >= 8, (
        f"Checkpoint 2: you return {len(pairs)} rows for the 8 cancellation lines of the toy"
        " table: use ASOF LEFT JOIN, so a cancellation with no match stays, with original_id"
        " missing." + move_on
    )
    assert len(pairs) == 8 and len({c for c, _ in pairs}) == 8 and (8, 6) not in pairs, (
        f"Checkpoint 2: toy cancellation 8 gets {[o for c, o in pairs if c == 8]}; it should get"
        " only line 7, the later of two identical purchases. An ordinary join returns every"
        " earlier match (and MIN the earliest); ASOF returns only the latest earlier one." + move_on
    )
    pairs = dict(pairs)
    assert any(o is not None for o in pairs.values()), (
        "Checkpoint 2: none of the 8 toy cancellations matches a purchase, not even"
        " cancellation 1, which cancels all of purchase 0 two hours later (same customer, stock"
        " code and price, 2 units). Two likely causes: the sign of the quantity (a cancellation's"
        " quantity is negative: -c.quantity = p.quantity, not c.quantity = p.quantity), or a"
        " condition on the invoice (a cancellation is on its own C invoice, never on the"
        " purchase's invoice: do not join on invoice)." + move_on
    )
    assert pairs[3] is None, (
        "Checkpoint 2: toy cancellation 3 returns 1 of the 3 units bought on line 2, and you"
        " match it. A cancellation matches only a line with the opposite quantity:"
        " -c.quantity = p.quantity." + move_on
    )
    assert pairs[4] is None, (
        "Checkpoint 2: toy cancellation 4 is dated before purchase 5, and you match them. A"
        " cancellation can only cancel an earlier purchase: c.invoice_date >= p.invoice_date."
        + move_on
    )
    assert pairs[11] != 10, (
        "Checkpoint 2: toy cancellation 11 (customer 1) gets line 10, a later purchase of the"
        " same item by customer 2. A customer can only cancel their own purchase:"
        " c.customer_id = p.customer_id." + move_on
    )
    assert pairs[14] != 13, (
        "Checkpoint 2: toy cancellation 14 (stock code 10006) gets line 13, a later purchase of"
        " stock code 10007 at the same price and quantity. A cancellation cancels the same"
        " product: c.stock_code = p.stock_code." + move_on
    )
    assert pairs[17] != 16, (
        "Checkpoint 2: toy cancellation 17 (price 7.00) gets line 16, a later purchase of the"
        " same stock code at 6.50. The price must match too: c.price = p.price." + move_on
    )
    assert pairs[19] is not None, (
        "Checkpoint 2: toy cancellation 19 has the same timestamp as purchase 18 (the log"
        " records minutes), and you do not match them. \"At or before\" includes the same"
        " minute: c.invoice_date >= p.invoice_date, not >." + move_on
    )
    expected = {1: 0, 3: None, 4: None, 8: 7, 11: 9, 14: 12, 17: 15, 19: 18}
    assert pairs == expected, (
        f"Checkpoint 2: on the toy table you return {pairs}; it should be {expected}. Check"
        " each condition of the join against the Task text." + move_on
    )
    matches = match_cancellations(con)
    ref = truth["rules"]["cancellation"]
    assert len(matches) == ref["cancel_lines"], (
        f"Checkpoint 2: you return {len(matches):,} rows for the {ref['cancel_lines']:,}"
        " cancellation lines of the messy log: one row per cancellation line." + move_on
    )
    unmatched = int(matches["original_id"].isna().sum())
    assert unmatched == 0, (
        f"Checkpoint 2: {unmatched:,} cancellation lines found no purchase; in the messy log"
        " every one has its original. Run your function on kept (Exercise 1's unflagged"
        " lines), and keep C invoices in c only." + move_on
    )
    twice = int(matches["original_id"].duplicated().sum())
    assert twice == 0, (
        f"Checkpoint 2: {twice:,} purchase lines are matched by two cancellations; in the messy"
        " log each purchase is cancelled at most once. Check the equality conditions." + move_on
    )
    dropped = set(matches["cancel_id"].astype(int)) | set(matches["original_id"].astype(int))
    assert positions_sha256(dropped) == ref["sha256"], (
        f"Checkpoint 2: you match every cancellation once, but not to the purchases the truth"
        f" says ({len(dropped):,} lines against {ref['count']:,}). Check that the join picks"
        " the latest earlier purchase." + move_on
    )

# %% [markdown]
# ### Run · The pipeline
#
# The **pipeline** is the ordered list of cleaning rules; the **reference pipeline** applies all
# of them (yours from Exercises 1 and 2, then Module 1's); a **variant** is the pipeline with one
# rule changed. The next cell defines three functions that Parts B, C and D reuse; read their
# docstrings (the code is there if you want it). `purchases` drops what your flags and matches
# say, then applies Module 1's rules, and adds up `quantity * price` per customer and purchase
# day.

# %%
def purchases(lines, flags, matches, keep_rules=(), unit="day"):
    """Purchases (customer_id, date, amount) from a line log.

    Drops lines without a customer, every cancellation line (Module 1's rule) and lines with
    quantity <= 0 or price <= 0; then the non-product lines, the duplicate lines and the matched
    originals of cancellations, except for the rules named in `keep_rules` ("non_product",
    "duplicate", "cancellation"). keep_rules=("cancellation",) is Module 1's cancellation rule:
    drop the C lines, keep the purchases they cancel. `unit="day"` sums per customer and purchase
    day; "invoice" per invoice (dated by its timestamp); "line" keeps each line as a purchase.
    """
    flag = flags.set_index("line_id").reindex(lines["line_id"]).fillna(False).astype(bool)
    drop = flag["no_customer"].to_numpy().copy()
    for rule in ("non_product", "duplicate"):
        if rule not in keep_rules:
            drop |= flag[rule].to_numpy()
    if "cancellation" not in keep_rules:
        drop |= lines["line_id"].isin(matches["original_id"].dropna().astype("int64")).to_numpy()
    drop |= lines["invoice"].str.startswith("C").to_numpy()
    drop |= ((lines["quantity"] <= 0) | (lines["price"] <= 0)).to_numpy()
    out = lines.loc[~drop, ["customer_id", "invoice", "invoice_date"]].copy()
    out["amount"] = lines.loc[~drop, "quantity"] * lines.loc[~drop, "price"]
    out["customer_id"] = out["customer_id"].astype("int64")
    if unit == "day":
        out["date"] = out["invoice_date"].dt.normalize()
        out = out.groupby(["customer_id", "date"], as_index=False)["amount"].sum()
    elif unit == "invoice":
        out = out.groupby(["customer_id", "invoice"], as_index=False).agg(
            date=("invoice_date", "min"), amount=("amount", "sum"))
    elif unit == "line":
        out = out.rename(columns={"invoice_date": "date"})
    else:
        raise ValueError("unit is 'day', 'invoice' or 'line'")
    return out[["customer_id", "date", "amount"]].sort_values(["customer_id", "date"],
                                                              ignore_index=True)


def calibration_rfm(tx, cal_end):
    """The RFM table of purchase days on or before cal_end, in weeks (as in Modules 1, 2, 4)."""
    return data.rfm_summary(tx, "customer_id", "date", "amount", observation_period_end=cal_end,
                            time_unit="D", time_scaler=7)


def rfm_by_invoice(tx, cal_end):
    """The RFM table when each row of tx (an invoice, or a line) counts as a purchase.

    frequency = rows - 1; recency and T in days / 7 from the first row's day; monetary value =
    mean amount over the rows after the first. Two rows on one day give recency 0 with
    frequency 1, so checks.rfm_table does not apply here.
    """
    end = pd.Timestamp(cal_end)
    tx = tx.loc[tx["date"].dt.normalize() <= end].sort_values(["customer_id", "date"])
    g = tx.groupby("customer_id")
    first, last = g["date"].min().dt.normalize(), g["date"].max().dt.normalize()
    repeat = tx.loc[(g.cumcount() > 0).to_numpy()]
    out = pd.DataFrame({"frequency": (g.size() - 1).astype(float),
                        "recency": (last - first).dt.days / 7,
                        "T": (end - first).dt.days / 7})
    out["monetary_value"] = repeat.groupby("customer_id")["amount"].mean()
    return out.fillna({"monetary_value": 0.0}).reset_index()


matches = match_cancellations(con)
print(f"{len(matches):,} cancellation lines matched to {matches['original_id'].nunique():,}"
      " purchase lines")

# %% [markdown]
# ### Run · Part A: the clean answer
#
# The models are fitted on the **calibration period**, the purchases up to the calibration end
# (2024-12-29 for the messy log); the year after it is the **holdout**, kept back to check the
# forecasts. The **RFM table** has one row per customer, as in Modules 1, 2 and 4: `frequency`
# (repeat purchase days), `recency` (weeks from the first to the last purchase day), `T` (weeks
# from the first purchase day to the calibration end) and `monetary_value` (mean amount over the
# repeat purchase days).
#
# Your pipeline turns the kept lines into purchase days. If the rules are right, the calibration
# RFM table equals that of the clean log (the retailer's transactions for customers 1 to 2,500,
# plus the wholesale buyers' transactions stored in the truth) for every customer, to 1e-9.

# %%
tx = purchases(lines, flags, matches)
rfm = calibration_rfm(tx, CAL_END)
retail_tx = data.load_synthetic("retailer_transactions")
wholesale_tx = pd.DataFrame(truth["wholesale"]["transactions"],
                            columns=truth["wholesale"]["transactions_columns"])
wholesale_tx["date"] = pd.to_datetime(wholesale_tx["date"])
clean_tx = pd.concat([retail_tx.loc[retail_tx["customer_id"] <= truth["rates"]["n_retail"]],
                      wholesale_tx], ignore_index=True)
clean_rfm = calibration_rfm(clean_tx, CAL_END)

# %% tags=["checkpoint"]
with workshop.checkpoint(label="Part A: the clean answer"):
    try:
        checks.frames_agree(rfm, clean_rfm, on="customer_id",
                            names=("your pipeline's RFM table", "the clean log's RFM table"))
    except AssertionError as err:
        both = rfm.merge(clean_rfm, on="customer_id", how="outer", suffixes=("", "_clean"))
        differ = both.loc[~np.isclose(both["frequency"], both["frequency_clean"])
                          | ~np.isclose(both["monetary_value"], both["monetary_value_clean"])
                          | both.isna().any(axis=1), "customer_id"].head(3).tolist()
        flagged = (flags.set_index("line_id")[["no_customer", "non_product", "duplicate"]]
                   .astype(bool).any(axis=1).reindex(lines["line_id"]).fillna(False).to_numpy())
        used = ~(flagged | is_c.to_numpy()
                 | lines["line_id"].isin(matches["original_id"].dropna()).to_numpy())
        copies = lines.drop(columns="line_id").duplicated().to_numpy()
        key = ["customer_id", "stock_code", "price", "quantity"]
        cancelled = lines[key].merge(lines.loc[is_c, key].assign(quantity=lambda d: -d["quantity"])
                                     .drop_duplicates(), how="left", indicator=True)
        cancelled = (cancelled["_merge"] == "both").to_numpy()
        notes = []
        for c in differ:
            own = (lines["customer_id"] == c).fillna(False).to_numpy() & used
            kinds = []
            if (~lines.loc[own, "stock_code"].str.match(r"^[0-9]{5}")).any():
                kinds.append("a non-product line kept (Exercise 1, workshop.use_reference(1))")
            if copies[own].any():
                kinds.append("a duplicate line kept (Exercise 1, workshop.use_reference(1))")
            if cancelled[own].any():
                kinds.append("a cancelled purchase kept (Exercise 2, workshop.use_reference(2))")
            notes.append(f"customer {c}: {', '.join(kinds) or 'a line dropped that should stay'}")
        raise AssertionError(
            f"{err}\nThe first customers that differ, and what your pipeline kept for them: "
            + "; ".join(notes) + ". Fix the rule named, or run the use_reference(...) shown,"
            " then rerun the cells from Exercise 1 down."
        ) from None
print(f"Exact: your pipeline reproduces the clean RFM table of all {len(rfm):,} customers.")
m1_cancel = calibration_rfm(purchases(lines, flags, matches, keep_rules=("cancellation",)),
                            CAL_END)
for label, table in (("Module 1's cancellation rule", m1_cancel), ("your pipeline", rfm)):
    print(f"{label:28s}: mean frequency {table['frequency'].mean():.3f}, mean repeat spend"
          f" ${table.loc[table['frequency'] > 0, 'monetary_value'].mean():.2f}")
pairs = matches.dropna().astype("int64").merge(lines[["line_id", "invoice_date"]],
                                      left_on="original_id", right_on="line_id")
pairs = pairs.merge(lines[["line_id", "invoice_date"]], left_on="cancel_id", right_on="line_id",
                    suffixes=("_purchase", "_cancel"))
next_day = pd.Timestamp(CAL_END) + pd.Timedelta(days=1)
straddle = (pairs["invoice_date_purchase"] < next_day) & (pairs["invoice_date_cancel"] >= next_day)
print(f"{straddle.sum():,} matched purchases fall in the calibration period and their"
      f" cancellations after it (after {CAL_END}).")

# %% [markdown]
# **Explain** (Exercise 2). Compare with your guess: the two lines that start "Module 1's
# cancellation rule" and "your pipeline" are the answer. Then look at the last line: matched
# pairs that straddle the calibration end. If you had split the log at the calibration end first
# and matched cancellations within each part, what would have happened to those purchases?
#
# <details><summary>Why this solution works</summary>
#
# `ASOF LEFT JOIN` keeps every cancellation line and gives it at most one purchase: among the
# purchase lines with the same customer, stock code, price and opposite quantity, the one with
# the latest timestamp at or before the cancellation. Kept cancelled purchases add both purchase
# days (an order on a day the customer bought nothing else) and amount, so the average customer
# looks slightly more frequent and slightly bigger. Had you split first, the cancellations of the
# straddling pairs would sit in the holdout part, so their purchases would stay in the calibration
# data as if they were real: clean the whole log, then split. The matching also runs after the
# duplicate rule, because a duplicated purchase would otherwise compete for the match. On Online
# Retail II only about a third of the cancellation lines find a purchase (Part D); the rest
# return part of an order, adjust a price or return goods bought before December 2009, and are
# dropped without netting, so those purchases keep their gross amount: an approximation.
# </details>

# %% [markdown]
# # Part B · Which errors matter, and for what
#
# ### Run · What gets fitted, and why not PyMC-Marketing
#
# From here on, the lab fits the BG/NBD and the Gamma-Gamma many times: once for each way of
# cleaning the log. Each fit is a **maximum-likelihood fit**: the parameter values that make the
# observed RFM table most likely. You have met it: in Modules 2 and 4, PyMC-Marketing's MAP (the
# single most probable parameter values under the posterior) with flat priors gave exactly these
# values and reproduced Fader, Hardie & Lee's published CDNOW estimates.
# `mktstats.mle` maximizes the same two likelihoods (written with NumPy from the
# formulas in PyMC-Marketing's source) with SciPy's optimizer. It builds no PyMC model, so there is
# nothing to compile; the next cell times both. Like MAP, it gives one number per parameter and no
# interval: every comparison in Parts B to D is between point estimates. As in Module 4, the
# purchase model is the BG/NBD; the retailer was generated by a Pareto/NBD, so the BG/NBD has no
# true values to hit. Part B compares each variant's purchase model with the reference pipeline
# and the realized holdout, and its spend model with the true p, q, v.
#
# The lab fits each variant once; it fits many variants because the comparison across them is
# the lesson. The next cell takes about a minute on Colab (an estimate), most of it PyMC-Marketing
# compiling its model. While it runs, read Exercise 3's Predict below and write your guess.

# %%
warnings.filterwarnings("ignore", message=".*merge_dataset.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="IProgress not found")  # tqdm, outside Colab
t0 = time.perf_counter()
bg_ref = mle.bgnbd_mle(rfm)
gg_ref = mle.gamma_gamma_mle(rfm[rfm["frequency"] > 0])
helper_seconds = time.perf_counter() - t0

t0 = time.perf_counter()
from pymc_extras.prior import Prior
from pymc_marketing import clv

import_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
bg_pmm = clv.BetaGeoModel(model_config={k: Prior("HalfFlat") for k in mle.BGNBD_PARAMS})
bg_pmm.fit(data=rfm[["customer_id", "frequency", "recency", "T"]], method="map",
           progressbar=False)
gg_pmm = clv.GammaGammaModel(model_config={k: Prior("HalfFlat") for k in mle.GAMMA_GAMMA_PARAMS})
gg_pmm.fit(data=rfm.loc[rfm["frequency"] > 0, ["customer_id", "frequency", "monetary_value"]],
           method="map", progressbar=False)
pmm_seconds = time.perf_counter() - t0
# the MAP values, unrounded (fit_summary rounds them for display)
pmm_values = {**{k: float(bg_pmm.fit_result[k].mean()) for k in mle.BGNBD_PARAMS},
              **{k: float(gg_pmm.fit_result[k].mean()) for k in mle.GAMMA_GAMMA_PARAMS}}
cross_check = pd.DataFrame({"mktstats.mle": {**bg_ref, **gg_ref},
                            "PyMC-Marketing MAP, flat priors": pmm_values})
cross_check["relative difference"] = (cross_check.iloc[:, 0] / cross_check.iloc[:, 1] - 1).abs()
print(f"Seconds in this run: mktstats.mle {helper_seconds:.2f} (both fits);"
      f" PyMC-Marketing import {import_seconds:.1f}, fits {pmm_seconds:.1f}")
cross_check.map(lambda x: f"{x:.4g}")

# %% tags=["checkpoint"]
with workshop.checkpoint(label="helper = PyMC-Marketing"):
    for param, row in cross_check.iterrows():
        assert row["relative difference"] <= 0.005, (
            f"The helper and PyMC-Marketing disagree on {param} ({row.iloc[0]:.4g} against"
            f" {row.iloc[1]:.4g}). Your Part A code is not involved: both fits use the same"
            " table. Rerun this cell; if it still fails, tell the instructor (a version"
            " mismatch). To move on now, run the next cell: Part B uses the helper's values."
        )

# %% [markdown]
# ## Exercise 3 · Aggregate and individual effects (7 minutes)
#
# The next cell fits the reference pipeline and five variants: keep the non-product lines; keep
# the duplicate lines; Module 1's cancellation rule; all three of Module 1's rules (the variant
# Module 1 used); and counting invoices instead of purchase days. For each it scores every
# customer: **revenue score** = expected purchase days in the next 52 weeks × expected spend per
# purchase day.
#
# **Predict.** Under Module 1's rules, does the total of the 52-week revenue scores change by
# under 2%, 2 to 10%, or over 10%? And how many of the 2,513 customers see their own score move by
# more than 10%: under 100, 100 to 1,000, or over 1,000? Write both guesses before you run the
# next cell.

# %% [markdown]
# ### Run · One rule at a time
#
# Look at which table moves more: the purchase model or the spend model. The reference row
# shows values; the other rows show the change from it in percent. The columns, as in Modules 2
# and 4: r and α (`alpha`) describe how purchase rates vary across customers, a and b the chance
# of dropping out after a purchase; p is the shape of each customer's spend per purchase, q and v
# describe how customers' average spend varies, and the **population mean spend** is
# p v / (q − 1), the average spend per purchase day across customers.

# %%
VARIANTS = {
    "reference": {},
    "keep non-product lines": {"keep_rules": ("non_product",)},
    "keep duplicate lines": {"keep_rules": ("duplicate",)},
    "Module 1's cancellation rule": {"keep_rules": ("cancellation",)},
    "Module 1's rules": {"keep_rules": ("non_product", "duplicate", "cancellation")},
    "count invoices, not days": {"unit": "invoice"},
}
HOLDOUT_WEEKS = truth["holdout_weeks"]  # 52
fits, scores = {}, {}
for name, how in VARIANTS.items():
    v_tx = purchases(lines, flags, matches, **how)
    v_rfm = (rfm_by_invoice(v_tx, CAL_END) if how.get("unit") == "invoice"
             else calibration_rfm(v_tx, CAL_END))
    v_rfm = v_rfm.set_index("customer_id").reindex(rfm["customer_id"]).reset_index()
    bg = mle.bgnbd_mle(v_rfm)
    gg = mle.gamma_gamma_mle(v_rfm[v_rfm["frequency"] > 0])
    exp_n = mle.bgnbd_expected_purchases(bg, v_rfm, HOLDOUT_WEEKS)
    fits[name] = {"bg": bg, "gg": gg, "forecast": exp_n.sum()}
    scores[name] = pd.Series(exp_n * mle.gamma_gamma_expected_spend(gg, v_rfm),
                             index=v_rfm["customer_id"])
scores = pd.DataFrame(scores)
holdout_days = len(clean_tx.loc[clean_tx["date"] > CAL_END, ["customer_id", "date"]]
                   .drop_duplicates())


def change_table(rows, whole=()):
    """The reference row's values (columns in `whole` as whole numbers); other rows in %."""
    table = pd.DataFrame(rows).T
    shown = table.map(lambda x: f"{x:,.3f}").astype(object)
    for col in whole:
        shown.loc["reference", col] = f"{table.loc['reference', col]:,.0f}"
    for name in table.index[1:]:
        shown.loc[name] = [f"{100 * (x / r - 1):+.1f}%"
                           for x, r in zip(table.loc[name], table.loc["reference"], strict=True)]
    return shown


purchase_model = change_table({n: {**f["bg"], "52-week forecast": f["forecast"]}
                               for n, f in fits.items()}, whole=["52-week forecast"])
spend_model = change_table({n: {**f["gg"], "population mean spend": mle.population_mean_spend(
    f["gg"])} for n, f in fits.items()})
line_rfm = rfm_by_invoice(purchases(lines, flags, matches, unit="line"), CAL_END)
print(f"Purchase model (BG/NBD). Holdout purchase days the clean log realized: {holdout_days:,}")
print(purchase_model.to_string())
print(f"\nSpend model (Gamma-Gamma). Truth for the retail customers: p {retail_truth['gamma_gamma']['p']},"
      f" q {retail_truth['gamma_gamma']['q']}, v {retail_truth['gamma_gamma']['v']},"
      f" mean ${retail_truth['population_mean_spend']:.0f}")
print(spend_model.to_string())
print(f"\nCounting lines as purchases (not fitted): mean frequency {line_rfm['frequency'].mean():.2f}"
      f" against {rfm['frequency'].mean():.2f} with purchase days")

# %% [markdown]
# **Task.** Write `variant_effects(scores, reference="reference", threshold=0.10)`. `scores` has
# one row per customer (index `customer_id`) and one column per variant. Return one row per column
# of `scores` (the variant names as the index) with two columns:
#
# - `total_change_pct` = 100 × (sum of the variant ÷ sum of the reference − 1);
# - `customers_moved` = the number of customers whose score changed by more than `threshold`
#   in either direction: |variant ÷ reference − 1| > `threshold`.
#
# Tip: `scores.div(scores[reference], axis=0)` divides every column by the reference column,
# customer by customer, and `.sum()` on a table of True/False counts the Trues in each column.

# %% tags=["exercise"]
def variant_effects(scores, reference="reference", threshold=0.10):
    # TODO 3: compare every column of scores with the reference column
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def variant_effects(scores, reference="reference", threshold=0.10):
    ref = scores[reference]
    ratio = scores.div(ref, axis=0)
    return pd.DataFrame({
        "total_change_pct": 100 * (scores.sum() / ref.sum() - 1),
        "customers_moved": ((ratio - 1).abs() > threshold).sum(),
    })


# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    toy = pd.DataFrame({"reference": [10.0, 20.0, 50.0, 20.0], "up": [11.5, 19.0, 50.0, 20.0],
                        "down": [10.0, 16.0, 50.0, 18.1]},
                       index=pd.Index([1, 2, 3, 4], name="customer_id"))
    got = variant_effects(toy)
    checks.columns(got, ["total_change_pct", "customers_moved"], name="variant_effects(toy)")
    move_on = " To move on now, run workshop.use_reference(3)."
    assert list(got.index) == ["reference", "up", "down"], (
        f"Checkpoint 3: return one row per column of scores, with the column names as the index;"
        f" you return the index {list(got.index)}." + move_on
    )
    total, moved = got["total_change_pct"], got["customers_moved"]
    assert abs(total["reference"]) < 1e-9 and moved["reference"] == 0, (
        "Checkpoint 3: compare every column with the reference column, including the reference"
        " itself, which must give 0 and 0." + move_on
    )
    assert abs(total["up"] - 0.005) > 1e-6, (
        "Checkpoint 3: total_change_pct is in percent: multiply by 100 (the toy's 'up' column"
        " changes the total from 100 to 100.5, which is +0.5%)." + move_on
    )
    assert abs(total["up"] - 0.5) < 1e-9 and abs(total["down"] + 5.9) < 1e-9, (
        f"Checkpoint 3: on the toy, total_change_pct is {total['up']:.4g} and"
        f" {total['down']:.4g}; it should be 0.5 and -5.9. total_change_pct compares totals,"
        " 100 x (sum of the variant / sum of the reference - 1), not the average of each"
        " customer's change." + move_on
    )
    assert not (moved["down"] == 2 and moved["up"] <= 1), (
        "Checkpoint 3: in the toy's 'down' column customer 4 goes from 20 to 18.1, a change of"
        " 9.5% (18.1 / 20 = 0.905), and you count them as moved. Divide the variant by the"
        " reference, not the reference by the variant (20 / 18.1 = 1.105 would count)." + move_on
    )
    assert moved["up"] == 1, (
        f"Checkpoint 3: in the toy's 'up' column only customer 1 moves by more than 10% (+15%);"
        f" you count {moved['up']}. Compare each customer's ratio with 1 + threshold." + move_on
    )
    assert moved["down"] == 1, (
        "Checkpoint 3: in the toy's 'down' column customer 2 falls by 20% and you do not count"
        " them. Count customers whose score moved by more than 10% in either direction: use the"
        " absolute change." + move_on
    )
    effects = variant_effects(scores)
    ref = scores["reference"]
    independent_total = np.array([100 * (scores[c].sum() / ref.sum() - 1) for c in scores])
    independent_moved = np.array([int((np.abs(scores[c] / ref - 1) > 0.10).sum())
                                  for c in scores])
    checks.close(effects.loc[list(scores), "total_change_pct"].to_numpy(dtype=float),
                 independent_total, abs=1e-9, name="total_change_pct on the sweep")
    try:
        checks.close(effects.loc[list(scores), "customers_moved"].to_numpy(dtype=float),
                     independent_moved, abs=1e-9, name="customers_moved on the sweep")
    except AssertionError as err:
        raise AssertionError(
            f"Checkpoint 3: {err} Your function passed the toy, so it agrees with the rule on"
            " clear cases but not near 10%, where the sweep has many customers. The likely cause"
            " is a different measure of change: use exactly |variant / reference - 1| >"
            " threshold, with the reference as the denominator (not the variant, and not the"
            " average of the two), no logarithm and no rounding." + move_on
        ) from None

# %% [markdown]
# A check of the provided sweep: its reference row is the fit on the clean log (the same table),
# and its 52-week forecast is within the tolerance measured over independent generator seeds.

# %% tags=["checkpoint"]
with workshop.checkpoint(label="sweep: the reference row"):
    clean_bg = mle.bgnbd_mle(clean_rfm)
    for param, value in clean_bg.items():
        checks.close(fits["reference"]["bg"][param], value, rel=1e-6,
                     name=f"the reference row's {param} against the clean log's fit")
    tol = truth["tolerances"]["holdout_forecast"]
    error = fits["reference"]["forecast"] / holdout_days - 1
    assert abs(error) <= tol["rel"], (
        f"The reference forecast is {fits['reference']['forecast']:,.0f} purchase days against"
        f" {holdout_days:,} realized ({error:+.1%}); {tol['basis']} is {tol['rel']:.0%}. This"
        " is provided code: rerun Part A, and tell the instructor if it fails again."
    )
    print(f"52-week forecast {fits['reference']['forecast']:,.0f} against {holdout_days:,}"
          f" realized ({error:+.1%}). Allowed: {tol['rel']:.0%},"
          f" {tol['basis'].removeprefix('the tolerance, ')}.")

# %% [markdown]
# Look at the `customers_moved` column next to `total_change_pct`: compare them with your two
# guesses. `largest_change_pct` is the largest single customer's change, in percent;
# `largest_pct_customer` names that customer and gives their score under the reference, then
# under the variant. In the two rows that keep non-product lines, `largest_change_pct` runs to
# tens of millions of percent. That is not a bug: it is one customer who bought often early in
# the calibration period and then went quiet for months, so the reference gives them a score of a
# fraction of a cent, and whom one manual-adjustment (`M`) invoice near the calibration end makes
# look active again. A percent change from almost nothing is enormous; read that column as "one
# customer moved a lot", never as a size. For a size, `largest_change_usd` shows, in the same
# form, the customer whose score moved the most in dollars.

# %%
def score_change(variant, customer):
    """'customer N: $reference score → $variant score' (cents, or two digits under one cent)."""
    shown = [f"${x:,.2f}" if x >= 0.01 else f"${x:.2g}"
             for x in scores.loc[customer, ["reference", variant]]]
    return f"customer {customer}: {shown[0]} → {shown[1]}"


effects = variant_effects(scores)
effects["largest_change_pct"] = [100 * (scores[c] / scores["reference"] - 1).abs().max()
                                 for c in effects.index]
# who is behind the largest percent change, and the largest change in dollars
variants = [c for c in effects.index if c != "reference"]
effects["largest_pct_customer"] = pd.Series(
    {c: score_change(c, (scores[c] / scores["reference"] - 1).abs().idxmax()) for c in variants})
effects["largest_change_usd"] = pd.Series(
    {c: score_change(c, (scores[c] - scores["reference"]).abs().idxmax()) for c in variants})
effects.assign(total_change_pct=effects["total_change_pct"].map("{:+.1f}".format),
               largest_change_pct=effects["largest_change_pct"].map("{:,.0f}".format)).fillna("")

# %% [markdown]
# **Explain.** Compare with your prediction. Why can the total move a few percent while hundreds
# of customers move by more than 10%? Point to the p, q, v columns of the spend-model table.
#
# <details><summary>Why this solution works</summary>
#
# The total compares sums; `customers_moved` compares each customer with themself, in both
# directions. Module 1's leftovers change the data of a minority of customers, but the spend
# parameters are shared by everyone, and p, q and v trade off against each other: they can move by
# tens of percent while the population mean p v / (q − 1) moves a few percent. Each customer's
# expected spend is their own mean shrunk toward the population mean with weight
# p x / (p x + q − 1), so when p moves, every repeat buyer's shrinkage changes. Counting invoices
# instead of purchase days does more damage than any row-level rule: it splits one purchase into
# several, which the BG/NBD reads as faster buying and the Gamma-Gamma as smaller, more variable
# spend. Duplicates leave the purchase model untouched (a copy never adds a day) and move only
# the spend model. A customer whose score changes by a huge percentage was nearly zero under the
# reference (a long-silent customer) and looks alive again when a fee invoice adds a recent
# "purchase". Report the quantity your decision uses, not the parameters.
# </details>

# %% [markdown]
# # Part C · Extreme orders
#
# ### Run · The wholesale customers
#
# A **wholesale customer** is a business that buys in bulk: few purchase days, very large amounts.
# Thirteen of the 2,513 customers (IDs 5001 to 5013) are wholesale buyers; their true mean spend
# per purchase day is about $1,200, against $30 for the retail customers. They are real
# customers, not errors. Look at who owns the largest calibration purchase days.

# %%
wholesale_ids = set(truth["wholesale"]["customer_ids"])
cal_days = tx.loc[tx["date"] <= CAL_END]
largest = cal_days.nlargest(10, "amount").assign(
    wholesale=lambda d: d["customer_id"].isin(wholesale_ids))
print(f"{len(cal_days):,} calibration purchase days; median amount ${cal_days['amount'].median():.2f}")
largest

# %% [markdown]
# ## Exercise 4 · Cap order values for the spend fit (7 minutes)
#
# To **cap at a percentile** is to replace every purchase-day amount above, say, the 99th
# percentile with that percentile's value (also called winsorizing).
#
# **Predict.** Cap every purchase day's amount at the 99th percentile and refit the Gamma-Gamma.
# Will the fitted population mean spend move toward the true $30 or away from it? And if the
# capped amounts are also used to value customers, will the 13 wholesale customers' predicted
# 52-week revenue, compared with their true value, be lower by under 25%, 25 to 50%, or over 50%?
# Write both guesses.
#
# **Task.** Write `capped_spend_fit(tx, pct, cal_end)` returning a dict with `cap` (the `pct`
# percentile of the calibration purchase-day amounts: all customers, first purchases included,
# days on or before `cal_end`) and `p`, `q`, `v` from `mle.gamma_gamma_mle` fitted on the
# calibration RFM of `tx` with every amount clipped at `cap` (`calibration_rfm`, then repeat
# buyers only). Two hints: `pct` is in percent, and pandas' `Series.quantile` takes a fraction:
# `quantile(0.99)` of 1, 2, ..., 100 is 99.01 (linear interpolation, the default). Clip with
# `Series.clip(upper=cap)` and name the argument: `clip(cap)` sets a floor, not a cap.

# %% tags=["exercise"]
def capped_spend_fit(tx, pct, cal_end):
    # TODO 4: the cap, the clipped amounts, the calibration RFM, then the spend fit
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def capped_spend_fit(tx, pct, cal_end):
    cap = tx.loc[tx["date"] <= pd.Timestamp(cal_end), "amount"].quantile(pct / 100)
    capped = calibration_rfm(tx.assign(amount=tx["amount"].clip(upper=cap)), cal_end)
    return {"cap": float(cap), **mle.gamma_gamma_mle(capped[capped["frequency"] > 0])}


# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    move_on = " To move on now, run workshop.use_reference(4)."
    cal_amounts = tx.loc[tx["date"] <= pd.Timestamp(CAL_END), "amount"]
    try:
        got = capped_spend_fit(tx, 99, CAL_END)
    except ValueError as err:
        if "percentiles should all be in the interval" in str(err):
            raise AssertionError(
                "Checkpoint 4: pandas raised 'percentiles should all be in the interval [0, 1]':"
                " pct is in percent, so use quantile(pct / 100)." + move_on
            ) from None
        raise
    except mle.MLEError:
        raise AssertionError(
            "Checkpoint 4: the spend fit in your function found no maximum (mle.MLEError): the"
            " amounts it fitted are not spread like spend. The usual cause is the clip:"
            " Series.clip(cap) sets a floor, so every amount becomes at least the cap. Cap with"
            " clip(upper=cap)." + move_on
        ) from None
    checks.columns(pd.DataFrame([got]), ["cap", "p", "q", "v"], name="capped_spend_fit(...)")
    cap99 = cal_amounts.quantile(0.99)
    first = tx.groupby("customer_id")["date"].transform("min")
    wrong_caps = {
        "your cap uses purchase days after the calibration end: compute it from"
        " date <= cal_end only": tx["amount"].quantile(0.99),
        "your cap leaves out the purchase days on the calibration end itself: include the"
        " calibration end (<=), date <= cal_end": tx.loc[
            tx["date"] < pd.Timestamp(CAL_END), "amount"].quantile(0.99),
        "cap each purchase day's amount, first purchases included, before building the RFM"
        " table; your cap leaves out first purchases": cal_amounts[
            (tx["date"] > first)[cal_amounts.index]].quantile(0.99),
        "cap each purchase day's amount before building the RFM table; capping customers' means"
        " (monetary_value) is a different, weaker cap": rfm["monetary_value"].quantile(0.99),
        "cap each purchase day's amount before building the RFM table; capping repeat buyers'"
        " means (monetary_value) is a different, weaker cap": rfm.loc[
            rfm["frequency"] > 0, "monetary_value"].quantile(0.99),
    }
    for message, value in wrong_caps.items():
        if np.isclose(value, cap99, rtol=1e-9):
            continue  # this mistake would give the right cap on these data
        assert not np.isclose(got["cap"], value, rtol=1e-9), (
            f"Checkpoint 4: your cap is {got['cap']:.2f}: {message}." + move_on)
    checks.close(got["cap"], cap99, rel=1e-9,
                 name="the cap (99th percentile of the calibration purchase-day amounts)")
    capped = calibration_rfm(tx.assign(amount=tx["amount"].clip(upper=cap99)), CAL_END)
    expected = mle.gamma_gamma_mle(capped[capped["frequency"] > 0])
    for param in ("p", "q", "v"):
        assert np.isclose(got[param], expected[param], rtol=1e-6), (
            f"Checkpoint 4: the cap is right, but {param} = {got[param]:.4g} where a fit on the"
            f" clipped amounts gives {expected[param]:.4g}. Fit on the clipped amounts, repeat"
            " buyers only (rfm[rfm['frequency'] > 0])." + move_on
        )
    tol = truth["tolerances"]["capped_gamma_gamma"]
    for param in ("p", "q", "v"):
        checks.close(got[param], retail_truth["gamma_gamma"][param], rel=tol[param]["rel"],
                     name=f"capped {param} against the truth ({tol['basis']})")
    checks.close(mle.population_mean_spend(got), retail_truth["population_mean_spend"],
                 rel=tol["population_mean_spend"]["rel"],
                 name=f"capped population mean spend against the truth ({tol['basis']})")
    full = capped_spend_fit(tx, 100, CAL_END)
    checks.close(full["cap"], cal_amounts.max(), rel=1e-12,
                 name="the cap at pct=100 (the largest calibration amount)")
    for param in ("p", "q", "v"):
        checks.close(full[param], fits["reference"]["gg"][param], rel=1e-6,
                     name=f"{param} at pct=100 against the reference spend fit (nothing capped)")

# %% [markdown]
# Look at where each row lands against the truth (p 6, q 4, v 15, mean $30). The **oracle** row
# drops the 13 wholesale customers using the truth's labels, which a real analyst would not have:
# it shows what a perfect fix would give. Then look at the wholesale customers' value three ways:
# no cap; parameters fitted on capped amounts but each customer valued on their own uncapped
# mean ("raw means"); and capped amounts used for the valuation too.

# %%
true_ww = truth["wholesale"]["true_expected_revenue_52w"]
rows = {}
for label, pct in (("no cap", None), ("P99.9", 99.9), ("P99.5", 99.5), ("P99", 99), ("P95", 95)):
    fit = capped_spend_fit(tx, pct, CAL_END) if pct else {"cap": np.nan, **fits["reference"]["gg"]}
    rows[label] = {**fit, "population mean": mle.population_mean_spend(fit)}
oracle_rfm = rfm[~rfm["customer_id"].isin(wholesale_ids)]
oracle = mle.gamma_gamma_mle(oracle_rfm[oracle_rfm["frequency"] > 0])
rows["drop wholesale (oracle)"] = {"cap": np.nan, **oracle,
                                   "population mean": mle.population_mean_spend(oracle)}
print(pd.DataFrame(rows).T.round(3).to_string())

is_w = rfm["customer_id"].isin(wholesale_ids).to_numpy()
exp_n52 = mle.bgnbd_expected_purchases(fits["reference"]["bg"], rfm, HOLDOUT_WEEKS)
p99 = rows["P99"]
rfm_capped = calibration_rfm(tx.assign(amount=tx["amount"].clip(upper=p99["cap"])), CAL_END)
ways = {
    "no cap": mle.gamma_gamma_expected_spend(fits["reference"]["gg"], rfm),
    "P99 for fitting only (raw means)": mle.gamma_gamma_expected_spend(p99, rfm),
    "P99 amounts for valuation too": mle.gamma_gamma_expected_spend(p99, rfm_capped),
}
print(f"\nThe 13 wholesale customers' 52-week revenue; truth ${true_ww:,.0f}")
for label, spend in ways.items():
    value = (exp_n52 * spend)[is_w].sum()
    print(f"  {label:34s}: ${value:9,.0f} ({100 * (value / true_ww - 1):+.0f}%)")

# %% [markdown]
# **Explain.** Compare with your guesses. Which of the three would you use to value customers, and
# why does the cap belong in the fit but not in the valuation? Point to the P99 row of the first
# table (against the oracle row, on p, q and v, not only the mean) and to the three wholesale
# lines.
#
# <details><summary>Why this solution works</summary>
#
# The cap comes from the calibration days only, first purchases included, so nothing from the
# holdout leaks in; clipping the purchase-day amounts before the RFM table caps each purchase, not
# each customer's mean. A cap stops a few huge purchases from dictating the shape of the spend
# distribution, which is why P99 lands near the oracle that drops the wholesale buyers using the
# truth; capping deeper (95th percentile) also trims ordinary large retail orders and biases p, q
# and v the other way (its population mean lands near $30 only because those shifts offset). The
# Gamma-Gamma shrinks each customer's mean toward the population mean, and most wholesale
# customers have few repeat purchases, so even uncapped they are undervalued. **Cap for fitting,
# not for valuation:** with P99-fitted parameters and their raw means the wholesale customers are
# valued about as well as with no cap, while everyone else gets parameters close to the oracle's;
# with capped amounts the wholesale customers lose most of their value. A segment of their own is
# the real fix (stretch).
# </details>

# %% [markdown]
# # Part D · Online Retail II
#
# Part D opens with three provided cells, D1 to D3. Reading them takes about four minutes (an
# estimate), counted in Exercise 5's 8 minutes.
#
# ### Run · D1 · Your pipeline on the real log
#
# The next cell loads Online Retail II. The first time, it downloads a 7 MB copy and checks its
# SHA-256 hash (seconds); if that copy cannot be reached, it downloads the 45 MB original from UCI
# and converts it, which takes minutes. While it loads, read D2's text below. Then compare the
# two logs' first rows: the same eight columns. The cell runs your `flag_lines` and
# `match_cancellations` and the pipeline unchanged.

# %%
retail_lines = data.load_online_retail_ii()
retail_lines.insert(0, "line_id", np.arange(len(retail_lines)))
print("Online Retail II (pounds sterling):")
print(retail_lines.head(3).drop(columns="line_id").to_string())
print("\nThe messy log (dollars):")
print(lines.head(3).drop(columns="line_id").to_string())
con.register("retail_lines", retail_lines)
retail_flags = flag_lines(con, "retail_lines")
retail_shares = retail_flags[["no_customer", "non_product", "duplicate"]].astype(bool)
retail_kept = retail_lines.loc[~retail_shares.any(axis=1).to_numpy()]
con.register("retail_kept", retail_kept)
retail_matches = match_cancellations(con, "retail_kept")
retail_tx = purchases(retail_lines, retail_flags, retail_matches)
matched = retail_matches["original_id"].notna()
again = int(retail_matches.loc[matched, "original_id"].duplicated().sum())
print(f"\n{len(retail_lines):,} lines. Flagged: no customer {retail_shares['no_customer'].sum():,},"
      f" non-product {retail_shares['non_product'].sum():,},"
      f" duplicate {retail_shares['duplicate'].sum():,}")
print(f"Cancellation lines matched: {matched.sum():,} of {len(retail_matches):,}"
      f" ({matched.mean():.0%}); {again} of them picked a purchase another cancellation had"
      " already picked")
print(f"{len(retail_tx):,} purchase days of {retail_tx['customer_id'].nunique():,} customers")

# %% [markdown]
# An ASOF join gives each cancellation at most one purchase, but two cancellations can pick the
# same purchase (the number printed above); one-to-one pairing is left as an approximation. When
# the lab was built, pairing them one to one matched 39 more purchase lines (£4,955) and changed
# one customer's place on the contact list of the Decision (measured 2026-10-10).
#
# The next check compares your counts with the reference. If it fails, run the
# `workshop.use_reference(...)` its message names, then rerun D1 (the cell above, which computes
# the counts) before rerunning the check.

# %% tags=["checkpoint"]
with workshop.checkpoint(label="Online Retail II counts"):
    # Measured 2026-10-10 on the table with the sha256 in mktstats.data.DATASETS.
    reference = {"no customer": 235_287, "non-product": 5_992, "duplicate": 11_812}
    mine = {"no customer": retail_shares["no_customer"].sum(),
            "non-product": retail_shares["non_product"].sum(),
            "duplicate": retail_shares["duplicate"].sum()}
    rerun = (" then rerun D1 (the cell under \"Run · D1\", which computes the counts) and then"
             " this cell. Rerunning this cell alone checks the old counts again.")
    for name, value in reference.items():
        assert mine[name] == value, (
            f"Your flag_lines flags {mine[name]:,} {name} lines of Online Retail II; the"
            f" reference is {value:,}. Run workshop.use_reference(1),{rerun}"
        )
    assert len(retail_matches) == 17_586 and matched.sum() == 6_107, (
        f"Your match_cancellations returns {len(retail_matches):,} cancellation lines with"
        f" {matched.sum():,} matched; the reference is 17,586 with 6,107 matched. Run"
        f" workshop.use_reference(2),{rerun}"
    )
    assert len(retail_tx) == 32_721 and retail_tx["customer_id"].nunique() == 5_839, (
        f"{len(retail_tx):,} purchase days of {retail_tx['customer_id'].nunique():,} customers;"
        " the reference is 32,721 of 5,839. This is provided code: rerun the cell under"
        f" \"Run · The pipeline\" in Part A,{rerun}"
    )

# %% [markdown]
# ### Run · D2 · Fits and the holdout
#
# Calibration ends on 2011-06-30; the holdout runs to 2011-12-09, the last day in the data
# (23.14 weeks). The holdout includes the pre-Christmas season, which the BG/NBD does not model.
# **Left censoring:** customers who bought before December 2009 enter the data as if they were
# new (Module 1's cohort plot), so their age T is too short (stretch 2). The fits use the `mle`
# helper only; it was checked against PyMC-Marketing at the start of Part B.

# %%
RETAIL_CAL_END, RETAIL_END = "2011-06-30", "2011-12-09"
RETAIL_HOLDOUT_WEEKS = (pd.Timestamp(RETAIL_END) - pd.Timestamp(RETAIL_CAL_END)).days / 7
retail_rfm = calibration_rfm(retail_tx, RETAIL_CAL_END)
retail_bg = mle.bgnbd_mle(retail_rfm)
retail_gg = mle.gamma_gamma_mle(retail_rfm[retail_rfm["frequency"] > 0])
holdout = retail_tx.loc[(retail_tx["date"] > RETAIL_CAL_END) & (retail_tx["date"] <= RETAIL_END)]
retail_rfm["holdout"] = retail_rfm["customer_id"].map(holdout.groupby("customer_id").size()).fillna(0)
retail_rfm["predicted"] = mle.bgnbd_expected_purchases(retail_bg, retail_rfm, RETAIL_HOLDOUT_WEEKS)
print(f"{len(retail_rfm):,} customers at {RETAIL_CAL_END}, {(retail_rfm['frequency'] > 0).sum():,}"
      f" repeat buyers; holdout {RETAIL_HOLDOUT_WEEKS:.2f} weeks")
print("BG/NBD:", {k: round(v, 4) for k, v in retail_bg.items()})
print("Gamma-Gamma:", {k: round(v, 4) for k, v in retail_gg.items()},
      f"population mean spend £{mle.population_mean_spend(retail_gg):,.1f}")

# %% tags=["checkpoint"]
with workshop.checkpoint(label="Online Retail II fits"):
    # Measured 2026-10-10 on the sha256-checked table; the cleaned data are the same for every
    # correct pipeline, so a correct run reproduces them within 0.5%.
    reference = {"r": 0.6720, "alpha": 9.4444, "a": 0.15034, "b": 3.0511,
                 "p": 2.4017, "q": 3.5534, "v": 418.79}
    estimate = {**retail_bg, **retail_gg}
    try:
        for param, value in reference.items():
            checks.close(estimate[param], value, rel=0.005,
                         name=f"{param} on Online Retail II (reference {value})")
    except AssertionError as err:
        raise AssertionError(
            str(err).removesuffix(" Check the formula and the units.") + " The fits are"
            " provided code, so the cleaned data differ from the reference, and the counts check"
            " after D1 (\"Online Retail II counts\") tests those: rerun D1 and that check, do"
            " what its message says if it fails, then rerun D2 and this cell."
        ) from None

# %% [markdown]
# Look at whether the predicted and actual bars agree for customers with few and with many
# calibration purchases.

# %%
group = retail_rfm["frequency"].clip(upper=7).astype(int).astype(str).replace("7", "7+")
tracking = retail_rfm.groupby(group).agg(customers=("customer_id", "size"),
                                         actual=("holdout", "mean"), predicted=("predicted", "mean"))
tracking = tracking.reindex(["0", "1", "2", "3", "4", "5", "6", "7+"])
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(tracking))
ax.bar(x - 0.2, tracking["actual"], width=0.4, color="0.3", label="actual")
ax.bar(x + 0.2, tracking["predicted"], width=0.4, color="0.75", hatch="//", edgecolor="0.3",
       label="predicted (BG/NBD)")
ax.set_xticks(x, tracking.index)
ax.set_xlabel("Repeat purchase days in calibration (to 2011-06-30)")
ax.set_ylabel("Mean holdout purchase days")
ax.legend()
plt.tight_layout()
total_pred, total_actual = retail_rfm["predicted"].sum(), retail_rfm["holdout"].sum()
print(f"Holdout purchase days: predicted {total_pred:,.0f}, actual {total_actual:,.0f}"
      f" ({total_pred / total_actual - 1:+.1%})")
tracking.round(2)

# %% tags=["checkpoint"]
with workshop.checkpoint(label="Online Retail II holdout"):
    assert abs(total_pred / total_actual - 1) <= 0.10, (
        f"The BG/NBD predicts {total_pred:,.0f} holdout purchase days against {total_actual:,.0f}"
        " actual: more than 10% apart. The reference run is 5% below the actual. This is"
        " provided code: rerun D1 and D2."
    )

# %% [markdown]
# The Gamma-Gamma assumes that spend per purchase is independent of how often a customer buys
# (Fader & Hardie, note 025; Module 4 accepted a correlation of 0.11 on CDNOW). A **rank
# correlation** (Spearman's) is Pearson's correlation of the ranks, so a few huge values cannot
# dominate it. A positive correlation means frequent buyers tend to spend more per purchase, so a
# customer with few purchases is likely shrunk toward a population mean that is too high for them.
# Noted, not fixed here.

# %%
repeat = retail_rfm[retail_rfm["frequency"] > 0]
capped_mv = repeat["monetary_value"].clip(upper=repeat["monetary_value"].quantile(0.99))
print(f"Repeat buyers ({len(repeat):,}): frequency and monetary value, Pearson"
      f" {repeat['frequency'].corr(repeat['monetary_value']):.3f}, Spearman (rank)"
      f" {repeat['frequency'].corr(repeat['monetary_value'], method='spearman'):.3f};"
      f" Pearson with monetary value capped at its 99th percentile"
      f" {repeat['frequency'].corr(capped_mv):.3f}")

# %% [markdown]
# ### Run · D3 · One customer, two pipelines
#
# Look at how much of this customer's value comes from lines your pipeline removes. The
# "dropped by" column names the rule that removes each line; a blank means the line is kept.

# %%
HORIZON = 26  # weeks: the contact decision looks half a year ahead
CUSTOMER = 12346
own = retail_lines.loc[retail_lines["customer_id"] == CUSTOMER].copy()
own_flags = retail_flags.set_index("line_id").loc[own["line_id"]]
own["dropped by"] = ""
for rule in ("duplicate", "non_product"):
    own.loc[own_flags[rule].astype(bool).to_numpy(), "dropped by"] = rule
own.loc[own["line_id"].isin(retail_matches["original_id"].dropna()), "dropped by"] = "cancelled"
own.loc[own["invoice"].str.startswith("C"), "dropped by"] = "C invoice"
print(f"Customer {CUSTOMER}: {len(own)} lines; the largest ten by value:")
print(own.assign(value=own["quantity"] * own["price"]).nlargest(10, "value")[
    ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "dropped by"]
].to_string(index=False))

m1_tx = purchases(retail_lines, retail_flags, retail_matches,
                  keep_rules=("non_product", "duplicate", "cancellation"))
m1_rfm = calibration_rfm(m1_tx, RETAIL_CAL_END)
m1_bg = mle.bgnbd_mle(m1_rfm)
m1_gg = mle.gamma_gamma_mle(m1_rfm[m1_rfm["frequency"] > 0])
retail_rfm["exp_purchases"] = mle.bgnbd_expected_purchases(retail_bg, retail_rfm, HORIZON)
retail_rfm["exp_spend"] = mle.gamma_gamma_expected_spend(retail_gg, retail_rfm)
compare = {}
for label, table, bg, gg in (("Module 1's rules", m1_rfm, m1_bg, m1_gg),
                             ("your pipeline", retail_rfm, retail_bg, retail_gg)):
    row = table.loc[table["customer_id"] == CUSTOMER]
    revenue = (mle.bgnbd_expected_purchases(bg, row, HORIZON)
               * mle.gamma_gamma_expected_spend(gg, row))[0]
    compare[label] = {"frequency": row["frequency"].iloc[0],
                      "mean repeat spend (£)": row["monetary_value"].iloc[0],
                      "expected 26-week revenue (£)": revenue}
print()
print(pd.DataFrame(compare).round(2).to_string())

exp_purchases = retail_rfm.set_index("customer_id")["exp_purchases"]
exp_spend = retail_rfm.set_index("customer_id")["exp_spend"]
revenue_26 = exp_purchases * exp_spend
print(f"\nExpected 26-week revenue per customer (your pipeline): mean £{revenue_26.mean():,.0f},"
      f" median £{revenue_26.median():,.0f}")

# %% [markdown]
# ## Exercise 5 · The retention-contact list (8 minutes)
#
# A **contact lift** is the assumed share by which a retention contact raises a customer's
# purchases; it is an assumption until an experiment measures it. A contact gains lift × margin ×
# expected revenue and costs the contact cost. This is Module 5's break-even rule turned around:
# contact when the break-even lift, cost ÷ (margin × expected revenue), is below the assumed lift.
#
# **Predict.** A contact costs £10, the margin is 30% and the contact is assumed to lift a
# customer's purchases by 10%. (a) How much expected 26-week revenue must a customer have before a
# contact pays: about £33, about £333, or about £3,333? (b) With the mean and the median printed
# above, what share of the 5,008 customers clears that bar: under 25%, 25 to 50%, or over 50%?
# Write both guesses.
#
# **Task.** Write `contact_list(exp_purchases, exp_spend, cost, margin, lift)` returning a Series
# of True/False on the same index as `exp_purchases`: True when
# lift × margin × exp_purchases × exp_spend > cost.

# %% tags=["exercise"]
def contact_list(exp_purchases, exp_spend, cost, margin, lift):
    # TODO 5: the expected gain of a contact, compared with its cost
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def contact_list(exp_purchases, exp_spend, cost, margin, lift):
    return lift * margin * exp_purchases * exp_spend > cost


# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    move_on = " To move on now, run workshop.use_reference(5)."
    hand_n = pd.Series([2.0, 2.0, 0.5], index=pd.Index([1, 2, 3], name="customer_id"))
    hand_s = pd.Series([16.0, 16.0, 16.0], index=hand_n.index)
    # lift 0.5, margin 0.25: customers 1 and 2 gain 0.5 x 0.25 x 2 x 16 = 4.00 exactly
    at_bar = contact_list(hand_n, hand_s, cost=4.0, margin=0.25, lift=0.5)
    assert isinstance(at_bar, pd.Series) and at_bar.index.equals(hand_n.index), (
        "Checkpoint 5: return one True/False per customer, a Series on the same index as"
        " exp_purchases, not a count or a list of IDs." + move_on
    )
    above = contact_list(hand_n, hand_s, cost=10.0, margin=0.25, lift=0.5).astype(bool)
    assert not above.any(), (
        f"Checkpoint 5: at a cost of 10.00 the hand customers gain at most 4.00, yet you contact"
        f" {int(above.sum())} of them. The gain from a contact is lift x margin x expected"
        " purchases x expected spend, not the revenue itself, and not with the lift or the"
        " margin left out." + move_on
    )
    assert not at_bar.astype(bool).any(), (
        "Checkpoint 5: a customer whose expected gain (4.00) equals the cost (4.00) gains"
        " nothing net, and you contact them: contact only when the gain is greater than the"
        " cost (>, not >=)." + move_on
    )
    below = contact_list(hand_n, hand_s, cost=3.5, margin=0.25, lift=0.5).astype(bool)
    assert below.tolist() == [True, True, False], (
        f"Checkpoint 5: at cost 3.50 the hand cases should give [True, True, False] (gains 4.00,"
        f" 4.00 and 1.00); you give {below.tolist()}. The gain from a contact is lift x margin x"
        " expected purchases x expected spend." + move_on
    )
    counts = [int(contact_list(exp_purchases, exp_spend, c, 0.30, 0.10).sum())
              for c in (1, 5, 10, 20, 50, 100)]
    checks.monotone(counts, increasing=False, name="contacts as the cost rises (£1 to £100)")
    n_list = counts[2]
    assert n_list < 0.9 * len(exp_purchases), (
        f"Checkpoint 5: you contact {n_list:,} of {len(exp_purchases):,} customers at £10. The"
        " gain from a contact is lift x margin x expected revenue, not the revenue itself." + move_on
    )
    assert n_list == 2_212, (  # measured 2026-10-10 on the sha256-checked table
        f"Checkpoint 5: {n_list:,} customers clear the bar at £10, 30% and 10%; the reference is"
        " 2,212. Your function passed the hand cases, so check the data first: if the counts"
        " check after D1 (\"Online Retail II counts\") failed, fix that first (its message says"
        " how), then rerun the cells from D1 down to this one. If it passed, check the rule"
        " against the Task text." + move_on
    )

# %% [markdown]
# Look at the share contacted (your Predict answer), then at who is on the list: the customers with
# the largest expected gain, and those just above the bar.

# %%
COST, MARGIN, LIFT = 10.0, 0.30, 0.10
retail_rfm["gain"] = LIFT * MARGIN * retail_rfm["exp_purchases"] * retail_rfm["exp_spend"]
on_list = contact_list(exp_purchases, exp_spend, COST, MARGIN, LIFT)
print(f"Bar: expected 26-week revenue above £{COST / (LIFT * MARGIN):,.2f}."
      f" Contacted: {on_list.sum():,} of {len(on_list):,} ({on_list.mean():.0%})")
show = ["customer_id", "frequency", "recency", "T", "monetary_value", "exp_purchases",
        "exp_spend", "gain"]
listed = retail_rfm[retail_rfm["customer_id"].isin(on_list[on_list].index)]
print("Largest expected gain:")
print(listed.nlargest(10, "gain")[show].round(2).to_string(index=False))
print("Closest above the bar:")
print(listed.nsmallest(10, "gain")[show].round(2).to_string(index=False))

# %% [markdown]
# **Explain.** Compare with your guesses. Which customers clear the bar: frequent, recent buyers or
# big one-time spenders? Point to the rows above, and find the customers with frequency 0 in the
# second table: their `exp_spend` is a number D2 printed. Which one, and why does every customer
# with no repeat purchase get it?
#
# <details><summary>Why this solution works</summary>
#
# The rule compares the expected gain with the cost, strictly, so a customer exactly at the bar
# is not contacted. The bar is £10 ÷ (10% × 30%) = £333.33 of expected 26-week revenue. A customer
# with no repeat purchase gets the population mean as expected spend (Module 4's weight w is 0
# when x = 0), so one big first order does not make a big expected spend; frequent, recent buyers
# have both more expected purchases and a spend estimate that leans on their own mean. Whether the
# contact causes the lift is an experiment's question (Modules 6 and 11), not this model's.
# </details>

# %% [markdown]
# ## Decision · Whom do we contact, and which cleaning choice moves the list?
#
# **The rule, stated before any number:** contact a customer when lift × margin × expected 26-week
# revenue > contact cost; at £10, 30% and 10%, that is expected 26-week revenue above £333. The
# margin is Module 5's 30%. The cost and the lift are assumptions, not measurements: the cell
# repeats the rule at a 5% and a 20% lift, and for each cleaning choice it counts the customers it
# adds to the list and drops from it.

# %%
def scores_26(table, bg, gg):
    return pd.Series(mle.bgnbd_expected_purchases(bg, table, HORIZON)
                     * mle.gamma_gamma_expected_spend(gg, table), index=table["customer_id"])


base = contact_list(exp_purchases, exp_spend, COST, MARGIN, LIFT)
gain = (LIFT * MARGIN * exp_purchases * exp_spend)[base]
choices = {}
for name, how in (("count invoices, not purchase days", {"unit": "invoice"}),
                  ("Module 1's rules", {"keep_rules": ("non_product", "duplicate",
                                                       "cancellation")}),
                  ("Module 1's cancellation rule", {"keep_rules": ("cancellation",)}),
                  ("keep non-product lines", {"keep_rules": ("non_product",)}),
                  ("keep duplicate lines", {"keep_rules": ("duplicate",)})):
    v_tx = purchases(retail_lines, retail_flags, retail_matches, **how)
    v_rfm = (rfm_by_invoice(v_tx, RETAIL_CAL_END) if how.get("unit") == "invoice"
             else calibration_rfm(v_tx, RETAIL_CAL_END))
    choices[name] = scores_26(v_rfm, mle.bgnbd_mle(v_rfm),
                              mle.gamma_gamma_mle(v_rfm[v_rfm["frequency"] > 0]))
cal_amount = retail_tx.loc[retail_tx["date"] <= RETAIL_CAL_END, "amount"]
for pct in (99.0, 99.9):
    cap = cal_amount.quantile(pct / 100)
    c_rfm = calibration_rfm(retail_tx.assign(amount=retail_tx["amount"].clip(upper=cap)),
                            RETAIL_CAL_END)
    c_gg = mle.gamma_gamma_mle(c_rfm[c_rfm["frequency"] > 0])
    choices[f"cap at P{pct:g} (£{cap:,.0f})"] = scores_26(c_rfm, retail_bg, c_gg)
    if pct == 99.0:
        choices[f"cap at P{pct:g}, for fitting only"] = scores_26(retail_rfm, retail_bg, c_gg)
moved = []
for name, s in choices.items():
    listed_v = LIFT * MARGIN * s.reindex(base.index).fillna(0.0) > COST
    added, dropped = int((listed_v & ~base).sum()), int((~listed_v & base).sum())
    moved.append({"cleaning choice": name, "added": added, "dropped": dropped,
                  "moved": added + dropped})
moved = pd.DataFrame(moved).sort_values("moved", ascending=False, ignore_index=True)
at_lift = {lift: int(contact_list(exp_purchases, exp_spend, COST, MARGIN, lift).sum())
           for lift in (0.05, 0.20)}
mode = "QUICK run (QUICK changes nothing in this lab)" if QUICK else "FULL run"
print(f"Number          : {base.sum():,} of {len(base):,} customers clear the bar at a 10% lift"
      f" ({at_lift[0.05]:,} at 5%, {at_lift[0.20]:,} at 20%).")
print(f"                  Contact cost £{base.sum() * COST:,.0f}; expected extra margin at a 10%"
      f" lift £{gain.sum():,.0f} (net £{gain.sum() - base.sum() * COST:,.0f}).")
print(f"                  Point estimates: no interval. {mode}.")
print("Rule            : contact when 10% lift x 30% margin x expected 26-week revenue > £10"
      " (revenue > £333).")
top = moved.iloc[0]
print(f"Cleaning        : the choice that moves the list most is {top['cleaning choice']},"
      f" {top['moved']:,} customers ({top['added']:,} added, {top['dropped']:,} dropped).")
print("No interval: the fits are point estimates; the spread shown is across assumed lifts and"
      " cleaning choices.")
moved

# %% [markdown]
# **Recommendation.** Write one sentence a manager could act on: how many customers to contact and
# what it costs; how sure you are, which here means how much the list depends on the assumed lift
# compared with the cleaning choices (no cleaning choice above moves as many customers as halving
# or doubling the lift); and what would change it: a measured lift, for example from contacting a
# random part of the list and holding the rest back as a control group (Module 6; not the
# time-period holdout of Parts B and D). Cleaning still decides *who* is on the list: look again
# at customer 12346 under Module 1's rules.
#
# ```text
# Number          : (printed above)
# Rule            : contact when 10% lift x 30% margin x expected 26-week revenue > £10.
# Cleaning        : (printed above)
# What would change it: the lift. A randomized control group held back from the list
#                   (Module 6) would measure it.
# Your sentence   : ________________________________________________
# ```

# %% [markdown]
# ## Stretch (optional) · What cleaning cannot fix, and better answers
#
# 1. **Missing IDs that cleaning cannot fix.** In the messy log, remove the customer ID from a
#    seeded 10% of identified invoices (`np.random.default_rng(13)`; guest checkout by known
#    customers), rerun the pipeline and the sweep, and compare frequency, r/α and the holdout
#    forecast with the reference. Unlike the errors above, this one cannot be cleaned away;
#    Online Retail II's 22.5% of lines without an ID may hide some of it.
# 2. **Left censoring.** On Online Retail II, keep only customers whose first observed purchase is
#    on or after 2010-04-01 and compare the BG/NBD parameters and the contact list.
# 3. **A wholesale segment.** Flag customers whose mean purchase day exceeds ten times the median
#    customer's and value them with their own mean spend instead of the shrunk Gamma-Gamma estimate;
#    compare with `truth["wholesale"]` on the messy log.
# 4. **A posterior for the borderline customers.** Fit `clv.BetaGeoModel()` and
#    `clv.GammaGammaModel()` to `retail_rfm` with `method="mcmc"` (2 chains; this adds minutes of
#    compute), apply the contact rule to each posterior draw, and report, for the customers near the
#    bar, the probability that contacting pays.
