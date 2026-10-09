# Lab brief · Module 1 · Transaction logs to customer-base analysis

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebook | `labs/src/01-customer-base-sql.py` → `labs/python/01-customer-base-sql.ipynb` |
| Lab slot | 50 min (`modules.m01.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 35 (limit 35) + decision 5 + slack 5 = 50. Minutes are estimates until a pilot. |
| Day | Day 1, first module after the warm-up |

## Question and decision

**Question.** Who are our customers, and can we tell who has left?

**Decision.** Is an "inactive for N days" rule good enough to stop retargeting customers? Measure how many
still-active customers it would drop on the synthetic retailer, where we know who is alive.

## What this lab fixes

Pre-work counted purchases per customer but never defined a purchase (same-day orders, cancellations,
returns) or who is still a customer.

## Data

| Source | Loader (contract) | Use |
|---|---|---|
| UCI Online Retail II (real; CC BY 4.0; 1,067,371 rows, 2009-12-01 to 2011-12-09) | `mktstats.data.load_online_retail_ii()` | cleaning, cohorts |
| CDNOW 1/10th sample (real; 2,357 customers, 1997-01 to 1998-06) | `mktstats.data.load_cdnow()` | RFM checkpoint, cohorts |
| Synthetic retailer | `mktstats.synth.retailer(seed=<default>)` → `.data["transactions"]`, `.data["customers"]` (with `true_alive_at_cal_end`), `.truth["calibration_end"]` | unobserved churn |

Register all three in one DuckDB connection (`con.register(...)` or `CREATE TABLE ... AS SELECT * FROM df`).
QUICK: Online Retail II restricted to the 2010-12-01 to 2011-12-09 year (about half the rows); CDNOW and the
synthetic data in full (small).

## Model fits

None. This module is SQL and counting. The first checkpoint is Exercise 1.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · From log to purchases | 1 · Clean Online Retail II in SQL | 7 |
| | *Run: purchase days and gaps with `LAG` (worked example, provided)* | — |
| B · The RFM table | 2 · Frequency, recency, age and spend with window functions | 12 |
| C · Cohorts and unobserved churn | 3 · Cohort retention curves | 8 |
| | 4 · How wrong is the recency rule? | 8 |
| **Exercises** | | **35** |
| Decision | Should we stop retargeting by an inactivity rule? | 5 |

Expected compute: seconds per query in both modes (DuckDB on about 1 million rows); no model fits. QUICK
uses one year of Online Retail II.

### Exercise 1 · Clean Online Retail II in SQL (7 minutes)

- **Predict.** What share of rows have no Customer ID: under 10%, 10 to 30%, or over 30%?
- **Function.** `clean_online_retail(con, table: str = "retail_raw") -> pd.DataFrame` returning one row
  per invoice: `customer_id, invoice, invoice_date (DATE), amount` where amount = Σ Quantity × Price.
  Rules: drop rows with missing Customer ID; drop cancellations (Invoice starting with `C`); drop rows with
  Quantity ≤ 0 or Price ≤ 0; aggregate to invoice.
- **Checkpoint.** No null `customer_id`; no invoice starts with `C`; `amount > 0`; row count equals the
  reference count for the committed data snapshot (the Technical Expert embeds it with the loader's
  sha256). Recovery message lists which rule failed.
- **Explain.** Many of this retailer's customers are wholesalers (UCI page). Cancellations and returns
  would make spend negative and break the spend model on Day 2.

### Run · Purchase days and gaps with `LAG` (provided worked example)

A provided query builds one row per customer per purchase **day** (`GROUP BY customer_id,
purchase_date`, summing the amount into `day_amount`) and adds `purchase_index` (`ROW_NUMBER() OVER
(PARTITION BY customer_id ORDER BY purchase_date)`) and `days_since_prev` (`purchase_date -
LAG(purchase_date) OVER (...)`), then plots the distribution of gaps for CDNOW. One sentence before the
plot: "Look at how many gaps are under 7 days; BTYD models count purchase days, so two orders on one day
are one purchase occasion." This is the pattern Exercise 2 reuses.

### Exercise 2 · Frequency, recency, age and spend with window functions (12 minutes)

- **Predict.** For a customer, which is larger, recency or age (T)? Can they be equal? Write one sentence
  for each.
- **Function.** `rfm_sql(con, table: str, cutoff: str, unit_days: int = 7) -> pd.DataFrame` built on the
  purchase-day rows (use `MIN(...) OVER (PARTITION BY customer_id)` or `ROW_NUMBER` to find each
  customer's first day), with
  `customer_id, frequency, recency, T, monetary_value`, using only purchases on or before `cutoff`:
  - `frequency` x = number of purchase days after the first;
  - `recency` t_x = (last purchase day − first purchase day) / `unit_days`;
  - `T` = (cutoff − first purchase day) / `unit_days`;
  - `monetary_value` = mean `day_amount` over repeat purchase days (0 when x = 0).
- **Checkpoint (two).**
  1. `checks.rfm_table(rfm)`: 0 ≤ recency ≤ T, frequency ≥ 0, recency = 0 exactly when frequency = 0.
  2. On CDNOW with `cutoff = "1997-09-30"` (the 39-week calibration period of Fader, Hardie & Lee's
     BG/NBD note 004): equal to `pymc_marketing.clv.rfm_summary(tx, "customer_id", "date",
     monetary_value_col="amount", observation_period_end="1997-09-30", time_unit="D", time_scaler=7)`
     to 1e-9, and exactly **1,411** customers have frequency 0 (the note reports 1,411 of 2,357 made no
     repeat purchase). Recovery message: "check that same-day purchases are merged and that times are
     in weeks as days/7, not whole weeks."
- **Explain.** x, t_x and T are the sufficient statistics of the Day 1 models: they summarize everything
  a BG/NBD or Pareto/NBD needs about a customer. `rfm_summary(time_unit="W")` is **not** the same: it
  merges purchases within the same week and floors times, giving 1,428 non-repeaters and different
  model estimates (measured 2026-10-09). Use days scaled by 7.

### Exercise 3 · Cohort retention curves (8 minutes)

- **Predict.** Twelve months after their first purchase, what share of a cohort will still be buying in that month: under 10%, 10 to 30%, or over 30%? Will the curve flatten or fall to zero?
- **Function.** `cohort_retention_sql(con, table: str) -> pd.DataFrame` with `cohort_month,
  months_since_first, active_share`: the share of each monthly acquisition cohort with at least one
  purchase in month k after their first purchase (k = 0, 1, …).
- **Checkpoint.** `active_share == 1` at k = 0 for every cohort; all values in [0, 1]
  (`checks.probability`); cohort sizes sum to the number of customers.
- **Explain.** "Active in month k" is not "alive": customers skip months. Plot Online Retail II and
  CDNOW (provided plotting cell). CDNOW is a single acquisition cohort (first purchase in Q1 1997), so its
  curve is one line.

### Exercise 4 · How wrong is the recency rule? (8 minutes)

- **Predict.** Of synthetic customers with no purchase in the last 90 days before the calibration end,
  what share are still alive: under 10%, 10 to 40%, or over 40%?
- **Function.** `recency_rule(rfm: pd.DataFrame, days: int, unit_days: int = 7) -> pd.Series[bool]`
  flagging "churned" when `(T - recency) * unit_days > days`; and `confusion(flag, true_alive) -> dict`
  with keys `tp, fp, fn, tn` where positive = flagged churned and truth = not alive.
- **Checkpoint.** Counts sum to the number of customers; for `days=90` the counts equal the reference
  counts on the committed synthetic data; `fp` (alive but flagged) is reported.
- **Explain.** In a non-contractual business nobody tells you they left, so "churn" is never observed.
  The alive flag exists only because we generated the data.

### Decision · Should we stop retargeting by an inactivity rule? (5 minutes)

Provided cell, in the standard order:
- **Number:** for N in {30, 60, 90, 180} days, customers dropped, of which truly alive, and the purchases
  those alive customers make in the holdout window (synthetic transactions after the calibration end),
  valued at an assumed $6 margin per purchase; retargeting cost saved at an assumed $1.50 per customer
  per quarter.
- **Rule:** "Use an inactivity rule only if the margin from the alive customers it drops is less than the
  retargeting cost it saves." (No posterior here, so no probability; Module 2 adds one.)
- **Recommendation:** the learner's sentence choosing N or rejecting the rule, and what would change it
  (a model that tells alive from dead customers better than recency alone: Module 2).

## Provided scaffolding

Loaders and DuckDB registration; plotting cells for gaps, cohorts and the decision table; the reference
`rfm_summary` call for the checkpoint.

## Stretch (optional)

Rewrite Exercise 2 with `QUALIFY` and a single pass; or compute RFM quintile segments (`NTILE(5)`) and
compare them with Exercise 4's truth.

## Known pitfalls

- Online Retail II comes in two sheets/years in the original Excel file; the loader should return one
  table. Invoice dates have times; cast to DATE before counting purchase days.
- `rfm_summary(time_unit="W")` merges same-week purchases (see Exercise 2).
- Do not use the holdout period in Exercise 2 (cutoff filter first).
- The synthetic customers acquired after the calibration end must be excluded from Exercise 4.

## APIs verified (and how)

- `pymc_marketing.clv.rfm_summary(transactions, customer_id_col, datetime_col, monetary_value_col=None,
  datetime_format=None, observation_period_end=None, time_unit="D", time_scaler=1, ...)`: signature from
  the installed source (pymc-marketing 1.2.0). Behaviour checked by running it on a hand-made log and on
  CDNOW (2026-10-09, laptop): same-day purchases merged, `monetary_value` = mean of repeat purchases,
  1,411 non-repeaters with `time_unit="D", time_scaler=7`, 1,428 with `time_unit="W"`.
- DuckDB 1.3.2 window functions (`ROW_NUMBER`, `LAG`) are standard; no version-specific syntax used.

## Open items for the Technical Expert

- `load_online_retail_ii()`: one table, original column names or snake_case (state which), parquet mirror
  allowed by CC BY 4.0; `load_cdnow()`: columns `customer_id, date, n_cds, amount`.
- Embed reference counts (Exercise 1 rows, Exercise 4 confusion) from the committed data snapshot.
