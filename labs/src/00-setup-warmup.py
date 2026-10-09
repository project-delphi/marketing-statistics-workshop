# %% [markdown]
# # Part A · Your runtime
#
# A *runtime* is the machine your notebook runs on. On Colab it is a virtual machine that
# Google starts for you and deletes when you disconnect, so anything you install or download
# lasts only for the session. The cell above installed what this lab needs (on Colab) and
# printed where you are running.
#
# **Predict.** How many CPUs does this runtime have: 1, 2, 4 or 8? And how much memory: under
# 8 GB, 8 to 16 GB, or more? Write both guesses down before you run the next cell.

# %%
import duckdb
import pandas as pd

from mktstats import runtime

for key, value in runtime.info().items():
    print(f"{key:>14}: {value}")
print(f"{'pandas':>14}: {pd.__version__}")
print(f"{'duckdb':>14}: {duckdb.__version__}")
print(f"{'QUICK':>14}: {QUICK} (it has no effect in this lab)")

# %% [markdown]
# **Explain.** Compare with your guesses. The CPU count and memory depend on the runtime type
# you choose (on Colab: Runtime > Change runtime type), so keep a note of them: later labs
# report run times, and a time means little without the machine it ran on. Google publishes
# the software each Colab runtime ships with (operating system, Python and R versions,
# preinstalled packages) in the
# [googlecolab/backend-info](https://github.com/googlecolab/backend-info) repository. The
# workshop pins its Python packages to that list, so the versions tested in CI and in the
# workshop's Docker image are the versions you run here.

# %% [markdown]
# # Part B · From a transaction log to a customer table
#
# Most customer analytics starts from a *transaction log*: one row per purchase. Here it is
# the CDNOW log that ships with PyMC-Marketing's example data. CDNOW was an online CD
# retailer. The columns:
#
# | column | meaning |
# |---|---|
# | `_id`, `id` | two identifiers of the customer, one-to-one; we use `id` |
# | `date` | the purchase date as an integer, `YYYYMMDD` (for example `19970118`) |
# | `cds_bought` | number of CDs in the purchase |
# | `spent` | amount paid, in dollars |
#
# The next cell downloads the file once, checks it against a known SHA-256 hash (so you know
# you have the same bytes as everyone else), and caches it.

# %%
# TODO(mktstats.data): switch to the package's CDNOW loader once mktstats.data has it.
import hashlib
import os
import time
import urllib.request
from pathlib import Path

CDNOW_URL = (
    "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/cdnow_transactions.csv"
)
CDNOW_SHA256 = "7311ae0fb676e9f454bba310b10bc769b1a6838aeec09bd00175eb49481dddac"


def load_cdnow(url=CDNOW_URL, sha256=CDNOW_SHA256):
    """The raw CDNOW transaction log, downloaded once and checked against its hash."""
    cache = Path(os.environ.get("MKTSTATS_CACHE", Path.home() / ".cache" / "mktstats"))
    path = cache / "cdnow_transactions.csv"
    if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256):
        cache.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                break
            except OSError as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Could not download {url} ({exc}). Check the network and rerun."
                    ) from exc
                time.sleep(2 * (attempt + 1))
        got = hashlib.sha256(data).hexdigest()
        if got != sha256:
            raise RuntimeError(
                f"The downloaded file has SHA-256 {got}, not {sha256}: it changed upstream."
                " Tell the instructor."
            )
        path.write_bytes(data)
    return pd.read_csv(path)


tx_raw = load_cdnow()
con = duckdb.connect()
con.register("transactions", tx_raw)
print(f"{len(tx_raw):,} transactions, {tx_raw['id'].nunique():,} customers")
tx_raw.head()

# %% [markdown]
# DuckDB is an analytical database that runs inside this notebook: no server, and it can query
# a pandas DataFrame directly. `con.register("transactions", tx_raw)` made the DataFrame
# visible to SQL as a table named `transactions`. Try it:

# %%
con.sql("SELECT COUNT(*) AS n_transactions, SUM(spent) AS total_spend FROM transactions").df()

# %% [markdown]
# ## Exercise 1 · The customer table in SQL (12 minutes)
#
# **Predict.** The log has one row per transaction, and the cell above printed how many
# customers it covers. What share of those customers bought only once: under 25%, 25 to 50%,
# or over 50%? Write your guess down; the cell after Exercise 3 prints the answer.
#
# **Task.** Write `customer_table_sql(con)`: a DuckDB query on `transactions` that returns one
# row per customer with these columns:
#
# - `customer_id` (the `id` column)
# - `first_purchase` and `last_purchase`: dates, not `YYYYMMDD` integers. In DuckDB,
#   `strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE` turns `19970118` into a date.
# - `n_purchases`: the number of transactions
# - `total_spend`: the sum of `spent`
#
# Return a pandas DataFrame: `con.sql(query).df()`.

# %% tags=["exercise"]
def customer_table_sql(con):
    # TODO 1: write the query, run it with con.sql(query).df()
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def customer_table_sql(con):
    query = """
        SELECT
            id AS customer_id,
            MIN(strptime(CAST(date AS VARCHAR), '%Y%m%d'))::DATE AS first_purchase,
            MAX(strptime(CAST(date AS VARCHAR), '%Y%m%d'))::DATE AS last_purchase,
            COUNT(*) AS n_purchases,
            SUM(spent) AS total_spend
        FROM transactions
        GROUP BY id
        ORDER BY customer_id
    """
    return con.sql(query).df()


# %% [markdown]
# **Explain.** The table has exactly one row per customer, the count printed above, because
# `GROUP BY id` collapses each customer's rows into one; every other column must then be an
# aggregate (`MIN`, `MAX`, `COUNT`, `SUM`). Converting the date before taking `MIN` and `MAX`
# matters little here (`YYYYMMDD` integers sort in date order) but matters as soon as you
# subtract dates. Your guess about one-time buyers is checked after Exercise 3.
#
# <details><summary>Why convert the dates at all?</summary>
#
# Recency and customer age, the inputs of the models in Modules 1 to 4, are differences between
# dates. `19970201 - 19970131` is 70, not 1: integer dates give nonsense the moment you do
# arithmetic with them.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    customers_sql = customer_table_sql(con)
    checks.columns(
        customers_sql,
        ["customer_id", "first_purchase", "last_purchase", "n_purchases", "total_spend"],
        name="customer_table_sql(con)",
    )
    n_customers = tx_raw["id"].nunique()
    assert len(customers_sql) == n_customers and customers_sql["customer_id"].is_unique, (
        f"Expected one row per customer ({n_customers:,}), got {len(customers_sql):,} rows."
        " GROUP BY the customer id."
    )
    assert pd.api.types.is_datetime64_any_dtype(customers_sql["first_purchase"]), (
        "first_purchase should be a date, not an integer like 19970118: convert it with"
        " strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE."
    )
    assert (customers_sql["first_purchase"] <= customers_sql["last_purchase"]).all(), (
        "Some customers' first purchase comes after their last: check MIN and MAX."
    )
    checks.close(customers_sql["n_purchases"].sum(), len(tx_raw), abs=0,
                 name="the number of purchases summed over customers")
    checks.close(customers_sql["total_spend"].sum(), tx_raw["spent"].sum(), rel=1e-9,
                 name="total spend summed over customers")

# %% [markdown]
# ## Exercise 2 · The same table in pandas (12 minutes)
#
# Two independent implementations that agree are the cheapest test you will ever write.
#
# **Predict.** Which step is most likely to make a pandas version disagree with the SQL one:
# the date conversion, the purchase count, or the spend total? Pick one and write it down.
#
# **Task.** Write `customer_table_pandas(tx)` returning the same columns as Exercise 1, from
# the DataFrame `tx` (the raw log). `pd.to_datetime(tx["date"].astype(str), format="%Y%m%d")`
# converts the dates; then group by `id` and aggregate.

# %% tags=["exercise"]
def customer_table_pandas(tx):
    # TODO 2: convert the dates, group by customer, aggregate
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def customer_table_pandas(tx):
    dates = pd.to_datetime(tx["date"].astype(str), format="%Y%m%d")
    return (
        tx.assign(date=dates)
        .groupby("id")
        .agg(
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            n_purchases=("date", "size"),
            total_spend=("spent", "sum"),
        )
        .rename_axis("customer_id")
        .reset_index()
        .sort_values("customer_id", ignore_index=True)
    )


# %% [markdown]
# **Explain.** Compare with your pick. If the checkpoint failed on your first try, which column
# did its message name, and what made that step differ from SQL? Named aggregation
# (`first_purchase=("date", "min")`) mirrors the SQL `SELECT` list one to one, which makes the
# two easy to compare by eye. `size` counts rows, as `COUNT(*)` does.
#
# <details><summary>What the checkpoint compares</summary>
#
# It matches the two tables on `customer_id` and compares every shared column: dates exactly,
# counts and spend within a tiny tolerance (floating-point sums can differ in the last digit
# depending on the order of addition).
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    customers_pd = customer_table_pandas(tx_raw)
    checks.frames_agree(
        customers_sql,
        customers_pd,
        on="customer_id",
        names=("the SQL table", "the pandas table"),
    )

# %% [markdown]
# ## Exercise 3 · Repeat purchases count days, not rows (10 minutes)
#
# The models from Module 2 on describe *repeat purchases*: purchases after the first. They
# count purchase occasions, not rows, so two transactions by the same customer on the same day
# are one occasion. Then `repeat_purchases = (number of distinct purchase days) - 1`.
#
# **Predict.** Do any CDNOW customers have two transactions on the same day? For how many
# customers will counting rows instead of days change the number of repeat purchases: none,
# under 100, or 100 or more? Write your guess down.
#
# **Task.** Write `repeat_purchases(tx)` in pandas, returning one row per customer with
# `customer_id`, `purchase_days` (distinct dates) and `repeat_purchases`.

# %% tags=["exercise"]
def repeat_purchases(tx):
    # TODO 3: count distinct purchase days per customer, then subtract one
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def repeat_purchases(tx):
    days = (
        tx.groupby("id")["date"]
        .nunique()
        .rename("purchase_days")
        .rename_axis("customer_id")
        .reset_index()
    )
    return days.assign(repeat_purchases=days["purchase_days"] - 1)


# %% [markdown]
# **Explain.** `nunique` counts distinct values, so same-day transactions collapse into one
# occasion. A customer who bought once has zero repeat purchases.
#
# <details><summary>Why the checkpoint uses a tiny made-up log</summary>
#
# Customer 1 below buys twice on 1 January and once on 1 March: one repeat purchase, although
# there are three rows. A log built to contain the tricky case tests the definition directly;
# the full CDNOW log then checks your function against an independent SQL count.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    toy = pd.DataFrame(
        {
            "_id": [10, 10, 10, 20],
            "id": [1, 1, 1, 2],
            "date": [19970101, 19970101, 19970301, 19970105],
            "cds_bought": [1, 2, 1, 1],
            "spent": [10.0, 20.0, 5.0, 7.0],
        }
    )
    got = repeat_purchases(toy).set_index("customer_id")["repeat_purchases"]
    assert got.get(1) == 1, (
        f"Customer 1 bought twice on 1997-01-01 and once on 1997-03-01: one repeat purchase"
        f" (same-day purchases count once). Your function gives {got.get(1)}."
    )
    assert got.get(2) == 0, f"Customer 2 bought once: zero repeat purchases, not {got.get(2)}."
    expected = con.sql(
        "SELECT id AS customer_id, COUNT(DISTINCT date) - 1 AS repeat_purchases"
        " FROM transactions GROUP BY id"
    ).df()
    checks.frames_agree(
        repeat_purchases(tx_raw)[["customer_id", "repeat_purchases"]],
        expected,
        on="customer_id",
        names=("repeat_purchases(tx_raw)", "an independent SQL count"),
    )

# %% [markdown]
# Compare with both predictions: the share of customers who bought only once (Exercise 1) and
# the number whose count changes (Exercise 3). Counting rows would give those customers extra
# "repeat purchases" on days they already bought: which input of the Module 2 models
# (frequency, recency or age) would that inflate, and would it make them look more or less
# loyal than they are?

# %%
repeats = repeat_purchases(tx_raw)
by_rows = customers_sql.set_index("customer_id")["n_purchases"] - 1
by_days = repeats.set_index("customer_id")["repeat_purchases"]
print(f"Customers who bought only once: {(by_days == 0).mean():.1%}")
print(f"Customers whose count changes if you count rows instead of days: {(by_rows != by_days).sum()}")

# %% [markdown]
# # Part C · Using a coding agent to scaffold and check an analysis
#
# A *coding agent* is an AI assistant that can write code, and often run it, in your project.
# It can draft queries like the ones above from a description of the tables. Treat what it
# writes as a colleague's first draft: useful, and unverified until checked.
#
# 1. **Give it the definitions, not just the column names.** "Repeat purchases" was ambiguous
#    until Exercise 3 pinned it down (days, not rows). Ambiguity is where a plausible-looking
#    query goes wrong.
# 2. **Ask for an independent check, not only the answer.** A second implementation in another
#    tool (Exercise 2) and invariants that must hold (one row per customer, totals that match
#    the raw log) catch errors that reading the code misses.
# 3. **Run the checks yourself and read the code.** An agent saying "this works" is not
#    evidence; a passing checkpoint is.
# 4. **Mind the data.** Do not paste customer records into a tool your organization has not
#    approved for them. Describing the schema, or sharing a small made-up sample like the one
#    in checkpoint 3, is usually enough.
#
# A prompt in this spirit:
#
# > Table `transactions(_id, id, date, cds_bought, spent)`: `id` is the customer, `date` is an
# > integer `YYYYMMDD`. Write a DuckDB query with one row per customer: first and last purchase
# > dates (as dates), number of transactions, total spend. Then write a pandas version and a
# > test that the two agree on every customer.

# %% [markdown]
# ## Decision · Sign off the customer table
#
# The pre-work ends in two decisions, both taken from evidence this notebook produced: is the
# customer table fit to hand to Module 1, and where will you run the labs?
#
# **Number.** The runtime report in Part A (CPUs, memory, versions) and the printout below:
# whether the table's purchases and spend match the raw log, and which checkpoints passed.
#
# **Rule.** Hand the table on only if every checkpoint passed and the totals match the log.
# Run the labs on Colab if this notebook ran here with every checkpoint passing; use the
# workshop's Docker image (see the Setup page of the workshop site; the site is linked at the
# end of this notebook) if the Colab install failed twice or your organization blocks Colab. Do the R
# notebooks of Modules 3, 7 and 11 if the R version of this notebook
# (`labs/r/00-setup-warmup.ipynb`) ran too; if it failed and you have no Docker, follow those
# modules in Python and from the module pages.

# %%
signoff = customers_sql.merge(repeats, on="customer_id")
print(f"Customers: {len(signoff):,}")
print(f"Purchases: {signoff['n_purchases'].sum():,} (log: {len(tx_raw):,})")
print(f"Spend: ${signoff['total_spend'].sum():,.2f} (log: ${tx_raw['spent'].sum():,.2f})")
print(f"Window: {signoff['first_purchase'].min():%Y-%m-%d} to {signoff['last_purchase'].max():%Y-%m-%d}")
print("Checkpoints passed:", ", ".join(k for k, (ok, _) in workshop.results.items() if ok) or "none")

# %% [markdown]
# **Recommendation.** Write two sentences. First: would you hand this table to the modeling
# team, yes or no, and which checks your answer rests on? Second: where will you run the labs
# (Colab, Docker or AWS), will you do the R notebooks, and which result above does that rest
# on? The site's FAQ covers the usual setup problems.
#
# Your sentences: ________________________________________________

# %% [markdown]
# ## Stretch (optional) · Customers by first-purchase month
#
# Write a DuckDB query that counts customers by the month of their first purchase and gives
# each cohort's total spend over the whole window. Which month brought the most valuable
# customers per head?
#
# <details><summary>One way to write it</summary>
#
# ```sql
# WITH firsts AS (
#     SELECT id, MIN(strptime(CAST(date AS VARCHAR), '%Y%m%d'))::DATE AS first_purchase,
#            SUM(spent) AS spend
#     FROM transactions GROUP BY id
# )
# SELECT date_trunc('month', first_purchase) AS cohort, COUNT(*) AS customers,
#        SUM(spend) AS spend, SUM(spend) / COUNT(*) AS spend_per_customer
# FROM firsts GROUP BY cohort ORDER BY cohort
# ```
# </details>
