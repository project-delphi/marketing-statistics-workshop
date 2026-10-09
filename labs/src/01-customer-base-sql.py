# ---
# jupyter:
#   mktstats:
#     verified: >-
#       2026-10-09, Technical Expert. DuckDB 1.3.2 (installed): window functions MIN() OVER,
#       ROW_NUMBER, LAG, aggregate FILTER, date_trunc, DATE_DIFF, starts_with; `/` on integers
#       returns a DOUBLE (7 / 2 = 3.5). mktstats.data: load_cdnow, load_online_retail_ii
#       (snake_case columns, Int64 customer_id), rfm_summary (pandas mirror of pymc-marketing
#       1.2.0 clv.utils.rfm_summary; equal on CDNOW at 1997-09-30 with time_unit="D",
#       time_scaler=7: 1,411 customers without a repeat purchase; time_unit="W" gives 1,422).
#       mktstats.synth.retailer (seed 2026): transactions, customers.true_alive_at_cal_end,
#       truth["calibration_end"]. Reference counts below were measured on the cached files
#       with the sha256 in mktstats.data.DATASETS.
# ---

# %% [markdown]
# # Part A · From log to purchases
#
# A *transaction log* has one row per purchase line: who bought, when, what, for how much. A
# *customer-base analysis* turns it into one row per customer and asks who is still buying.
# This lab uses three logs, all queried with SQL in DuckDB (an analytical database that runs
# inside the notebook):
#
# | table | what it is |
# |---|---|
# | `retail_raw` | UCI Online Retail II: about a million invoice lines of a UK online retailer, Dec 2009 to Dec 2011 (real, CC BY 4.0) |
# | `cdnow_tx` | CDNOW: purchases of 2,357 customers of an online CD store, 1997 to mid-1998 (real) |
# | `synth_tx` | the workshop's synthetic retailer: we generated it, so we know which customers are still alive |
#
# The next cell loads CDNOW and the synthetic retailer (small, seconds) and registers them
# with DuckDB under the names above. Every table has the columns `customer_id`, `date`,
# `amount`.

# %%
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mktstats import data, synth

con = duckdb.connect()  # one in-memory database for the whole lab

cdnow = data.load_cdnow().rename(columns={"id": "customer_id", "spent": "amount"})
con.register("cdnow_tx", cdnow[["customer_id", "date", "amount"]])

retailer = synth.retailer()  # seed 2026, the same data as data/synthetic/retailer_*.csv
tx = retailer.transactions  # customer_id, date, amount: every purchase
customers = retailer.customers  # one row per customer, with the true alive status
truth = retailer.truth
con.register("synth_tx", tx)

print(f"CDNOW: {len(cdnow):,} rows, {cdnow['customer_id'].nunique():,} customers")
print(f"Synthetic retailer: {len(tx):,} purchases, {len(customers):,} customers,"
      f" calibration period ends {truth['calibration_end']}")

# %% [markdown]
# ## Exercise 1 · Clean Online Retail II in SQL (7 minutes)
#
# A purchase in a raw log is not one row. Online Retail II has one row per invoice *line*
# (a product on an invoice), lines without a customer, cancelled invoices and returns.
#
# **Predict.** What share of the invoice lines have no Customer ID: under 10%, 10 to 30%, or
# over 30%? Write your guess down.
#
# **Task.** Write `clean_online_retail(con, table="retail_raw")`: a DuckDB query that returns
# one row per invoice with the columns `customer_id`, `invoice`, `invoice_date` (a DATE, no
# time of day) and `amount` (the sum of `quantity * price` over the invoice's lines). Apply
# these rules to the lines before you add them up:
#
# 1. drop lines with a missing `customer_id`;
# 2. drop cancellations: invoices that start with `C` (`starts_with(invoice, 'C')`);
# 3. drop lines with `quantity <= 0` or `price <= 0` (returns, corrections, free items).
#
# The raw columns are `invoice`, `stock_code`, `description`, `quantity`, `invoice_date`
# (date and time), `price`, `customer_id`, `country`. Return `con.sql(query).df()`.
# The checkpoint first runs your function on a tiny table built to contain each case, so it
# needs no download.

# %% tags=["exercise"]
def clean_online_retail(con, table="retail_raw"):
    # TODO 1: filter the lines, then GROUP BY customer and invoice
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def clean_online_retail(con, table="retail_raw"):
    query = f"""
        SELECT customer_id,
               invoice,
               CAST(MIN(invoice_date) AS DATE) AS invoice_date,
               SUM(quantity * price) AS amount
        FROM {table}
        WHERE customer_id IS NOT NULL
          AND NOT starts_with(invoice, 'C')
          AND quantity > 0
          AND price > 0
        GROUP BY customer_id, invoice
        ORDER BY customer_id, invoice
    """
    return con.sql(query).df()


# %% [markdown]
# **Explain.** Compare with your prediction once the full data is loaded below. Then say why
# each rule matters for a model of *customers*: which rule removes rows that would make spend
# negative, and which removes rows that cannot belong to anyone?
#
# <details><summary>Why this solution works</summary>
#
# The `WHERE` clause applies the three rules line by line, before `GROUP BY` adds the lines of
# an invoice together. Grouping by invoice (not by timestamp) matters: a few invoices carry two
# timestamps a minute apart, and grouping by the timestamp would split them in two. Lines
# without a customer cannot be attributed to anyone, so they cannot enter a customer table.
# Cancellations and returns have negative quantities; left in, they make spend negative and
# break the spend model of Day 2. Many of this retailer's customers are wholesalers (UCI
# dataset page), so a single invoice can be large: that is real, not an error.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    toy = pd.DataFrame({
        "invoice": ["100", "100", "101", "C102", "103", "103", "104", "105"],
        "customer_id": pd.array([1, 1, None, 1, 2, 2, 2, 3], dtype="Int64"),
        "quantity": [2, 1, 5, -1, 0, 3, 1, -2],
        "price": [3.0, 4.0, 1.0, 3.0, 2.0, 2.0, 0.0, 5.0],
        "invoice_date": pd.to_datetime([
            "2010-12-01 09:00", "2010-12-01 09:01", "2010-12-01 10:00", "2010-12-02 11:00",
            "2010-12-03 12:00", "2010-12-03 12:00", "2010-12-04 13:00", "2010-12-05 14:00",
        ]),
    })
    con.register("retail_toy", toy)
    got = clean_online_retail(con, "retail_toy")
    checks.columns(got, ["customer_id", "invoice", "invoice_date", "amount"],
                   name="clean_online_retail(con)")
    kept = sorted(got["invoice"].astype(str))
    reasons = {"101": "rule 1 (missing customer_id)", "C102": "rule 2 (cancellation)",
               "104": "rule 3 (price 0)", "105": "rule 3 (quantity below 0)"}
    wrong = [f"{inv}: {why}" for inv, why in reasons.items() if inv in kept]
    assert not wrong, (
        f"These toy invoices should have been dropped: {wrong}. Check the WHERE clause."
        " To move on now, run workshop.use_reference(1)."
    )
    assert kept == ["100", "103"], (
        f"The toy table should give one row each for invoices 100 and 103; you return {kept}."
        " One row per invoice: GROUP BY customer_id and invoice, not by the timestamp"
        " (invoice 100 has lines at 09:00 and 09:01)."
    )
    amounts = got.set_index(got["invoice"].astype(str))["amount"]
    checks.close(amounts["100"], 2 * 3.0 + 1 * 4.0, abs=1e-9,
                 name="amount of toy invoice 100 (2 x 3.00 + 1 x 4.00)")
    checks.close(amounts["103"], 3 * 2.0, abs=1e-9,
                 name="amount of toy invoice 103 (its line with quantity 0 is dropped)")
    dates = pd.to_datetime(got["invoice_date"])
    assert (dates == dates.dt.normalize()).all(), (
        "invoice_date still has a time of day: CAST(... AS DATE) so that later queries count"
        " purchase days."
    )

# %% [markdown]
# ### Run · Load Online Retail II
#
# The first time, the next cell downloads a 45 MB file from the UCI repository, checks its
# SHA-256 hash and converts it (an estimate: one to three minutes, depending on the network);
# after that it reads a cached copy in seconds. While it runs, read Exercise 2 below. With QUICK on, the lab keeps the second year
# only (December 2010 to December 2011), so counts are about half.

# %%
retail = data.load_online_retail_ii()
if QUICK:
    retail = retail.loc[retail["invoice_date"] >= "2010-12-01"].reset_index(drop=True)
con.register("retail_raw", retail)
print(f"{len(retail):,} invoice lines, {retail['customer_id'].nunique():,} customers"
      f" ({'QUICK: second year only' if QUICK else 'both years'})")
print(f"Lines with no Customer ID: {retail['customer_id'].isna().mean():.1%}")
print(f"Lines on cancelled invoices: {retail['invoice'].str.startswith('C').mean():.1%}")

# %% [markdown]
# Now the same function on the real table. The reference counts below were measured on the
# file with the SHA-256 hash the loader checks, so a different count means a rule differs.

# %% tags=["checkpoint"]
with workshop.checkpoint(1, label="1b"):
    retail_clean = clean_online_retail(con)
    expected_rows = 18_532 if QUICK else 36_969  # invoices after cleaning, measured 2026-10-09
    assert retail_clean["customer_id"].notna().all(), "Some rows have no customer_id: rule 1."
    assert not retail_clean["invoice"].astype(str).str.startswith("C").any(), (
        "Some invoices start with C: rule 2 (cancellations)."
    )
    assert (retail_clean["amount"] > 0).all(), (
        "Some invoices have amount <= 0: rule 3 (quantity and price must both be above 0)."
    )
    assert retail_clean["invoice"].is_unique, (
        "Some invoices appear twice: GROUP BY customer_id and invoice only."
    )
    assert len(retail_clean) == expected_rows, (
        f"{len(retail_clean):,} invoices after cleaning; the reference gives {expected_rows:,}."
        " Check each rule of the WHERE clause. To move on now, run workshop.use_reference(1)."
    )

# %% [markdown]
# The cleaned invoices become the table `retail_tx`, with the same columns as the others.

# %%
con.register("retail_clean", retail_clean)
con.sql("CREATE OR REPLACE VIEW retail_tx AS"
        " SELECT customer_id, invoice_date AS date, amount FROM retail_clean")
print(f"{len(retail_clean):,} invoices from {retail_clean['customer_id'].nunique():,} customers")

# %% [markdown]
# ### Run · Purchase days and gaps with `LAG` (worked example)
#
# A *window function* computes a value for each row from other rows of the same group,
# without collapsing the group the way `GROUP BY` does. `PARTITION BY customer_id ORDER BY
# purchase_date` makes each customer's purchases a group in date order; `ROW_NUMBER()` numbers
# them and `LAG(purchase_date)` reads the previous row's date. The query first collapses the
# log to one row per customer per purchase **day**: the models of Module 2 count purchase
# days, so two orders on one day are one purchase occasion.

# %%
gaps = con.sql("""
    WITH days AS (
        SELECT customer_id, CAST(date AS DATE) AS purchase_date, SUM(amount) AS day_amount
        FROM cdnow_tx
        GROUP BY customer_id, purchase_date
    )
    SELECT customer_id, purchase_date, day_amount,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY purchase_date) AS purchase_index,
           purchase_date - LAG(purchase_date) OVER (PARTITION BY customer_id ORDER BY purchase_date)
               AS days_since_prev
    FROM days
    ORDER BY customer_id, purchase_date
""").df()
gaps.head(8)

# %% [markdown]
# Look at how many gaps are under 7 days (left of the dashed line): those are the purchases a
# weekly count would merge with the one before.

# %%
gap_days = gaps["days_since_prev"].dropna().astype(float)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(gap_days, bins=np.arange(0, 550, 7), color="0.45")
ax.axvline(7, color="black", linestyle="--", linewidth=1, label="7 days")
ax.set_xlabel("Days since the customer's previous purchase day (CDNOW)")
ax.set_ylabel("Repeat purchase days")
ax.legend()
print(f"{(gap_days < 7).mean():.1%} of {len(gap_days):,} gaps are under 7 days")

# %% [markdown]
# # Part B · The RFM table
#
# The Module 2 models need three numbers per customer, measured at a *cutoff* date (the end
# of the calibration period) and in weeks:
#
# - **frequency** $x$: the number of purchase days after the first (repeat purchases);
# - **recency** $t_x$: the time from the first purchase day to the last one;
# - **age** $T$: the time from the first purchase day to the cutoff.
#
# With mean spend per repeat purchase day (*monetary value*), this is the *RFM table*: recency,
# frequency, monetary value.
#
# ## Exercise 2 · Frequency, recency, age and spend with window functions (12 minutes)
#
# **Predict.** For one customer, which is larger, recency or $T$? Can they be equal? Write one
# sentence for each.
#
# **Task.** Write `rfm_sql(con, table, cutoff, unit_days=7)`: a DuckDB query on `table`
# (columns `customer_id`, `date`, `amount`) that returns one row per customer with
# `customer_id, frequency, recency, T, monetary_value`, using only purchases on or before
# `cutoff` (a string such as `"1997-09-30"`):
#
# - start from purchase days, as in the worked example (`GROUP BY customer_id, purchase_date`,
#   summing `amount` into `day_amount`);
# - find each customer's first day with `MIN(purchase_date) OVER (PARTITION BY customer_id)`;
# - `frequency` = number of purchase days minus 1;
# - `recency` = days from the first to the last purchase day, divided by `unit_days`;
# - `T` = days from the first purchase day to `cutoff`, divided by `unit_days`;
# - `monetary_value` = mean `day_amount` over the repeat purchase days (0 when there are none).
#
# In DuckDB, `DATE_DIFF('day', a, b)` counts the days from `a` to `b`, and
# `AVG(x) FILTER (WHERE condition)` averages only the rows that meet the condition.

# %% tags=["exercise"]
def rfm_sql(con, table, cutoff, unit_days=7):
    # TODO 2: purchase days -> first day per customer -> one row per customer
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def rfm_sql(con, table, cutoff, unit_days=7):
    query = f"""
        WITH days AS (
            SELECT customer_id, CAST(date AS DATE) AS purchase_date, SUM(amount) AS day_amount
            FROM {table}
            WHERE CAST(date AS DATE) <= DATE '{cutoff}'
            GROUP BY customer_id, purchase_date
        ),
        marked AS (
            SELECT *, MIN(purchase_date) OVER (PARTITION BY customer_id) AS first_day
            FROM days
        )
        SELECT customer_id,
               COUNT(*) - 1 AS frequency,
               DATE_DIFF('day', MIN(first_day), MAX(purchase_date)) / {unit_days} AS recency,
               DATE_DIFF('day', MIN(first_day), DATE '{cutoff}') / {unit_days} AS T,
               COALESCE(AVG(day_amount) FILTER (WHERE purchase_date > first_day), 0)
                   AS monetary_value
        FROM marked
        GROUP BY customer_id
        ORDER BY customer_id
    """
    return con.sql(query).df()


# %% [markdown]
# **Explain.** Compare with your prediction: recency can equal $T$ only for a customer whose
# last purchase falls on the cutoff day. Then say why the cutoff filter has to come *before*
# the purchase days are counted, not after.
#
# <details><summary>Why this solution works</summary>
#
# The `days` step merges same-day orders into one purchase day and drops everything after the
# cutoff, so nothing from the holdout period leaks into the table. The window `MIN(...) OVER`
# attaches each customer's first day to every one of their rows, which lets the final
# `GROUP BY` measure recency and $T$ from it and leave the first purchase out of the spend
# average. $x$, $t_x$ and $T$ are the *sufficient statistics* of the Module 2 models: they
# summarize everything a BG/NBD or Pareto/NBD model needs to know about a customer's purchase
# history. Weeks are days divided by 7 here. pymc-marketing's `rfm_summary(time_unit="W")`
# is different: it merges purchases made in the same calendar week and counts whole weeks, so
# it gives more customers without a repeat purchase and different model estimates.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    rfm = rfm_sql(con, "cdnow_tx", "1997-09-30")
    # checks.rfm_table compares frequency with recency in the unit purchases are counted in
    # (days), so give it the table in days.
    checks.rfm_table(rfm.assign(recency=rfm["recency"] * 7, T=rfm["T"] * 7),
                     customer_id="customer_id", name="rfm_sql(con, 'cdnow_tx', '1997-09-30')")
    reference = data.rfm_summary(cdnow, "customer_id", "date", "amount",
                                 observation_period_end="1997-09-30", time_unit="D",
                                 time_scaler=7)
    try:
        checks.frames_agree(rfm, reference, on="customer_id",
                            names=("rfm_sql", "the reference rfm_summary"))
    except AssertionError as err:
        raise AssertionError(
            f"{err}\nCheck that same-day purchases are merged into one purchase day and that"
            " recency and T are in weeks as days / 7, not whole calendar weeks. To move on"
            " now, run workshop.use_reference(2)."
        ) from None
    no_repeat = int((rfm["frequency"] == 0).sum())
    assert no_repeat == 1411, (
        f"{no_repeat:,} customers have frequency 0; Fader, Hardie & Lee report 1,411 of 2,357"
        " for this calibration period. Count purchase days, and only up to the cutoff."
    )

# %% [markdown]
# `data.rfm_summary` is the workshop's pandas copy of pymc-marketing's `rfm_summary`, with
# the same conventions; the workshop's tests check that the two agree. Compare the share
# without a repeat purchase with the Module 0 pre-work, which used the whole log:

# %%
print(f"CDNOW at 1997-09-30: {len(rfm):,} customers, {(rfm['frequency'] == 0).mean():.1%}"
      " made no repeat purchase")
rfm.describe().round(2)

# %% [markdown]
# # Part C · Cohorts and unobserved churn
#
# A *cohort* is the group of customers whose first purchase fell in the same month. A
# *retention curve* follows one cohort: the share of its customers who buy in month
# $k = 0, 1, 2, \dots$ after their first month.
#
# ## Exercise 3 · Cohort retention curves (8 minutes)
#
# **Predict.** Twelve months after their first purchase, what share of a cohort will buy in
# that month: under 10%, 10 to 30%, or over 30%? Will the curve flatten or fall to zero?
#
# **Task.** Write `cohort_retention_sql(con, table)` returning one row per cohort and month
# with `cohort_month` (the first day of the customer's first purchase month),
# `months_since_first` ($k$), `cohort_size` (customers in the cohort) and `active_share`
# (the share of the cohort with at least one purchase in month $k$). Use
# `date_trunc('month', date)` for the month and `DATE_DIFF('month', a, b)` for $k$.

# %% tags=["exercise"]
def cohort_retention_sql(con, table):
    # TODO 3: customer-months -> first month per customer -> share active per cohort and k
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def cohort_retention_sql(con, table):
    query = f"""
        WITH months AS (
            SELECT DISTINCT customer_id, date_trunc('month', CAST(date AS DATE)) AS month
            FROM {table}
        ),
        cohorts AS (
            SELECT *, MIN(month) OVER (PARTITION BY customer_id) AS cohort_month
            FROM months
        )
        SELECT cohort_month,
               DATE_DIFF('month', cohort_month, month) AS months_since_first,
               MAX(COUNT(*)) OVER (PARTITION BY cohort_month) AS cohort_size,
               COUNT(*) / MAX(COUNT(*)) OVER (PARTITION BY cohort_month) AS active_share
        FROM cohorts
        GROUP BY cohort_month, months_since_first
        ORDER BY cohort_month, months_since_first
    """
    return con.sql(query).df()


# %% [markdown]
# **Explain.** Compare with your prediction. "Active in month $k$" is not "alive": which kind
# of customer is alive but inactive in a given month? Point to the curves below.
#
# <details><summary>Why this solution works</summary>
#
# `SELECT DISTINCT customer_id, month` keeps one row per customer per month with a purchase,
# so `COUNT(*)` in the final step counts customers, not orders. Every customer is active in
# their first month, so the count at $k = 0$ is the cohort size; the window
# `MAX(COUNT(*)) OVER (PARTITION BY cohort_month)` picks it out. The curves fall fast and then
# flatten: many customers buy once, and the rest buy every few months, so a customer can skip
# a month and still be a customer. Online Retail II starts in December 2009, so its first
# cohort holds everyone who had bought before as well (*left censoring*: the data cannot say
# when they first bought). CDNOW contains only customers whose first purchase was in the first
# quarter of 1997, so it has three cohorts.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    for table in ("cdnow_tx", "retail_tx"):
        coh = cohort_retention_sql(con, table)
        checks.columns(coh, ["cohort_month", "months_since_first", "cohort_size",
                             "active_share"], name=f"cohort_retention_sql(con, {table!r})")
        checks.probability(coh["active_share"], name=f"active_share ({table})")
        first = coh.loc[coh["months_since_first"] == 0]
        assert np.allclose(first["active_share"], 1.0), (
            f"{table}: active_share at months_since_first = 0 should be 1 for every cohort"
            " (every customer buys in their first month). Check the cohort size."
        )
        n_customers = con.sql(f"SELECT COUNT(DISTINCT customer_id) FROM {table}").fetchone()[0]
        assert first["cohort_size"].sum() == n_customers, (
            f"{table}: cohort sizes add up to {first['cohort_size'].sum():,}, but there are"
            f" {n_customers:,} customers. Count each customer once, in their first month."
        )
    # An independent pandas count for CDNOW: months numbered from year 0, one row per
    # customer-month, then customers active in month k over customers in the cohort.
    pm = pd.DataFrame({"customer_id": cdnow["customer_id"],
                       "m": cdnow["date"].dt.year * 12 + cdnow["date"].dt.month - 1})
    pm = pm.drop_duplicates()
    pm["first"] = pm.groupby("customer_id")["m"].transform("min")
    pm["months_since_first"] = pm["m"] - pm["first"]
    expected = pm.groupby(["first", "months_since_first"]).size().rename("n").reset_index()
    expected["active_share"] = expected["n"] / expected.groupby("first")["n"].transform("max")
    expected["cohort_month"] = pd.to_datetime(pd.DataFrame(
        {"year": expected["first"] // 12, "month": expected["first"] % 12 + 1, "day": 1}))
    ours = cohort_retention_sql(con, "cdnow_tx")
    ours["cohort_month"] = pd.to_datetime(ours["cohort_month"])
    keys = ["cohort_month", "months_since_first"]
    checks.frames_agree(ours[keys + ["active_share"]], expected[keys + ["active_share"]],
                        on=keys, names=("cohort_retention_sql on CDNOW",
                                        "an independent pandas count"))

# %% [markdown]
# Look at where each curve flattens and at what level: that level is the share of a cohort
# that keeps buying in any given month, not the share that is still a customer. (The last
# point of each Online Retail II curve is December 2011, of which the data hold only nine
# days, so it drops.)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
styles = ["-o", "-s", "-^", "-v", "-D", "-x"]
for ax, (table, label, n) in zip(axes, [("retail_tx", "Online Retail II", 6),
                                        ("cdnow_tx", "CDNOW", 3)], strict=True):
    coh = cohort_retention_sql(con, table)
    for style, (cohort, g) in zip(styles, list(coh.groupby("cohort_month"))[:n], strict=False):
        ax.plot(g["months_since_first"], g["active_share"], style, markersize=3,
                label=f"{pd.Timestamp(cohort):%b %Y} (n={g['cohort_size'].iloc[0]:,})")
    ax.set_title(f"{label}: first {n} monthly cohorts")
    ax.set_xlabel("Months since first purchase (k)")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Share of cohort buying in month k")
axes[0].set_ylim(0, 0.6)  # month 0 (share 1) is off the top of the plot
plt.tight_layout()
coh_retail = cohort_retention_sql(con, "retail_tx")
print("Month-12 share, Online Retail II cohorts:",
      coh_retail.loc[coh_retail["months_since_first"] == 12, "active_share"].round(3).tolist()[:6])

# %% [markdown]
# ## Exercise 4 · How wrong is the recency rule? (8 minutes)
#
# A common churn rule: "a customer with no purchase in the last $N$ days has left". In a
# *non-contractual* business (no subscription to cancel) nobody tells you they left, so churn
# is never observed and the rule cannot be checked on real data. On the synthetic retailer we
# know each customer's true status at the end of the calibration period
# (`customers["true_alive_at_cal_end"]`, 1 = still alive).
#
# **Predict.** Of the synthetic customers with no purchase in the last 90 days before the
# calibration end, what share are still alive: under 10%, 10 to 40%, or over 40%?
#
# **Task.** Write two short functions:
#
# - `recency_rule(rfm, days, unit_days=7)` returning a boolean Series, True ("churned") when
#   the time since the last purchase, `(T - recency) * unit_days` days, is more than `days`;
# - `confusion(flag, true_alive)` returning a dict with the counts `tp, fp, fn, tn`, where
#   *positive* means flagged as churned and the truth is *not alive*. So `fp` counts customers
#   who are alive but flagged.
#
# The next cell builds the synthetic retailer's RFM table at the calibration end with your
# `rfm_sql` and attaches the true status.

# %%
synth_rfm = rfm_sql(con, "synth_tx", truth["calibration_end"]).merge(
    customers[["customer_id", "true_alive_at_cal_end"]], on="customer_id")
synth_rfm.head()

# %% tags=["exercise"]
def recency_rule(rfm, days, unit_days=7):
    # TODO 4: days since the last purchase > days
    raise NotImplementedError("TODO 4")


def confusion(flag, true_alive):
    # TODO 4: count the four combinations of flagged / not flagged and dead / alive
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def recency_rule(rfm, days, unit_days=7):
    return (rfm["T"] - rfm["recency"]) * unit_days > days


@workshop.solution(4)
def confusion(flag, true_alive):
    flag = np.asarray(flag, dtype=bool)
    dead = ~np.asarray(true_alive, dtype=bool)
    return {"tp": int((flag & dead).sum()), "fp": int((flag & ~dead).sum()),
            "fn": int((~flag & dead).sum()), "tn": int((~flag & ~dead).sum())}


# %% [markdown]
# **Explain.** Compare with your prediction. Why are so many customers who have not bought for
# 90 days still alive? Think of a customer who buys about four times a year.
#
# <details><summary>Why this solution works</summary>
#
# $T - t_x$ is the time since the last purchase, so the rule is one comparison. A customer
# with a low purchase rate has long gaps between purchases even while alive, so any fixed
# cutoff calls slow buyers "churned" (false positives) and keeps recently lapsed fast buyers
# (false negatives). A longer $N$ trades false positives for false negatives; no $N$ removes
# both, because the rule ignores how often each customer used to buy. The true status exists
# only because we generated the data.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    toy_rfm = pd.DataFrame({"T": [20.0, 20.0, 20.0], "recency": [0.0, 10.0, 20.0]})
    flags = pd.Series(recency_rule(toy_rfm, days=70)).astype(bool).tolist()
    assert flags == [True, False, False], (
        f"Toy customers last bought 140, 70 and 0 days before the cutoff; with days=70 only the"
        f" first is churned (70 is not more than 70). You flag {flags}. Use"
        " (T - recency) * unit_days > days."
    )
    toy_counts = confusion([True, True, False, False], [0, 1, 0, 1])
    assert toy_counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}, (
        f"confusion([True, True, False, False], [0, 1, 0, 1]) should give one of each, not"
        f" {toy_counts}. Positive = flagged churned; the truth is positive when NOT alive."
    )
    counts = confusion(recency_rule(synth_rfm, 90), synth_rfm["true_alive_at_cal_end"])
    assert sum(counts.values()) == len(synth_rfm), (
        f"The four counts add up to {sum(counts.values()):,}, not the {len(synth_rfm):,}"
        " customers: each customer belongs in exactly one of tp, fp, fn, tn."
    )
    reference_90 = {"tp": 2566, "fp": 832, "fn": 153, "tn": 1449}  # measured 2026-10-09
    assert counts == reference_90, (
        f"With days=90 you get {counts}; the reference gives {reference_90}. Check the"
        " direction of the comparison and which way round positive and alive are. To move on"
        " now, run workshop.use_reference(4)."
    )
    flagged = counts["tp"] + counts["fp"]
    print(f"90-day rule: {flagged:,} customers flagged as churned, of whom {counts['fp']:,}"
          f" ({counts['fp'] / flagged:.1%}) are still alive.")

# %% [markdown]
# ## Decision · Should we stop retargeting by an inactivity rule?
#
# The retailer pays to retarget every customer with ads. A proposal: stop retargeting anyone
# inactive for $N$ days. Dropping a customer who has left saves money; dropping one who is
# still alive may lose their next purchases.
#
# **The rule, stated before the numbers.** Use an inactivity rule only if the margin from the
# alive customers it drops is less than the retargeting cost it saves. Assumptions (the
# retailer's, not measured here): retargeting costs \$1.50 per customer per quarter; each
# purchase earns \$6 of margin; and, pessimistically, a dropped customer who is still alive
# makes none of their next purchases. The comparison covers the 52-week holdout period after
# the calibration end (four quarters).

# %%
COST_PER_QUARTER = 1.50  # retargeting cost per customer per quarter (assumption)
MARGIN_PER_PURCHASE = 6.00  # margin per purchase day (assumption)
QUARTERS = truth["holdout_weeks"] / 13
holdout = con.sql(f"""
    SELECT customer_id, COUNT(DISTINCT CAST(date AS DATE)) AS holdout_purchases
    FROM synth_tx WHERE CAST(date AS DATE) > DATE '{truth["calibration_end"]}'
    GROUP BY customer_id
""").df()
decision = synth_rfm.merge(holdout, on="customer_id", how="left").fillna({"holdout_purchases": 0})
rows = []
for n_days in (30, 60, 90, 180):
    drop = recency_rule(decision, n_days).to_numpy(dtype=bool)
    alive = decision["true_alive_at_cal_end"].to_numpy() == 1
    lost = decision.loc[drop & alive, "holdout_purchases"].sum() * MARGIN_PER_PURCHASE
    saved = drop.sum() * COST_PER_QUARTER * QUARTERS
    rows.append({"N days": n_days, "customers dropped": int(drop.sum()),
                 "of which alive": int((drop & alive).sum()),
                 "their holdout purchases": int(decision.loc[drop & alive, "holdout_purchases"].sum()),
                 "margin lost ($)": round(lost), "retargeting saved ($)": round(saved),
                 "net ($)": round(saved - lost)})
table = pd.DataFrame(rows)
print(f"Synthetic retailer, {len(decision):,} customers, holdout {truth['holdout_weeks']} weeks."
      " No posterior here, so no probability: the numbers depend on the assumptions above.")
print("Mode:", "QUICK (Online Retail II was cut; this table uses the full synthetic data)"
      if QUICK else "FULL")
table

# %% [markdown]
# **Recommendation.** Read the table row by row: `net ($)` is the retargeting cost saved minus
# the margin lost, and the rule says use $N$ only where it is positive. Write one sentence:
# which $N$ (if any) you would use, how many customers and dollars it concerns, and what would
# change your answer. There is no interval here, so say instead how sure you are: how far the
# assumed margin or cost would have to move to flip the sign of `net ($)`. A model that tells
# alive customers from lapsed ones better than recency alone (Module 2) would change it too.
#
# Your sentence: ________________________________________________

# %% [markdown]
# ## Stretch (optional) · RFM segments in one pass
#
# 1. Rewrite `rfm_sql` without the `marked` step: compute the first day with a window inside
#    the final `SELECT`, and use `QUALIFY` (DuckDB's filter on window results) if you need one.
# 2. Score the synthetic customers into quintiles with `NTILE(5) OVER (ORDER BY recency)` and
#    `NTILE(5) OVER (ORDER BY frequency)`, and compare the true alive share across the 25
#    segments with Exercise 4. Does a recency-and-frequency segment separate alive from lapsed
#    customers better than recency alone?
