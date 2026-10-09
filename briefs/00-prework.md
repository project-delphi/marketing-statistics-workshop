# Lab brief · Module 0 · Pre-work: environment, warm-up and coding agents

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebooks | `labs/src/00-setup-warmup.py` → `labs/python/00-setup-warmup.ipynb`; `labs/src/00-setup-warmup.R` → `labs/r/00-setup-warmup.ipynb` |
| Lab slot | 60 min, optional, done before Day 1 (`modules.m00.minutes.lab`). Planned: 58 min. |
| Content keys | `modules.m00` in `_variables.yml` (summary, objectives, decision) |

## Question and decision

**Question.** Is my environment ready, and can I count purchases per customer in pandas, SQL or dplyr?

**Decision.** Where will I run the labs (Colab, local Docker or AWS), and will I do the R notebooks? The
decision is based on an environment check that actually ran, not on what we hope works.

## Data

- Synthetic retailer, small slice: `mktstats.synth.retailer(seed=<default>)`, `.data["transactions"]`
  (customer_id, date, amount). Use the first 90 days of transactions so every cell is instant.
- R: the committed `data/synthetic/retailer_transactions.csv` (file name per the Technical Expert's
  generator) read through the R helpers.
- No downloads besides the install. The first checkpoint runs in seconds.

## Model fits

None.

## Parts and exercises (Python; the R notebook mirrors them)

| Part | Exercise | Minutes |
|---|---|---|
| A · Environment | 1 · Read your environment report | 5 |
| B · Transaction-log warm-up | 2 · Purchases per customer in pandas | 8 |
| | 3 · The same in DuckDB SQL | 8 |
| | 4 · Acquisition cohorts | 9 |
| C · One simulated customer | 5 · Simulate purchases until the customer leaves | 12 |
| D · Working with a coding agent | 6 · Catch the agent's bug with a test | 13 |
| Decision | Where will you run the labs? | 3 |
| **Total** | | **58** |

### Exercise 1 · Read your environment report (5 minutes)

- **Predict.** How many CPU cores does your runtime have, and which Python version is it?
- **Function.** `env_summary(report: dict) -> dict` returning `{"python": str, "cpus": int,
  "on_colab": bool, "pymc_marketing": str}` from the report the generated install cell produces
  (the Technical Expert names the report object; propose `mktstats.runtime.report()`).
- **Checkpoint.** Python starts with `"3.13"`; `pymc_marketing` equals the pin embedded from
  `_variables.yml packages`; `cpus >= 1`; `on_colab` is a bool. Recovery message: "Restart the session
  and Run all; if the version is still wrong, see Setup → Colab install failed."
- **Explain.** Why the workshop pins versions, and why the readiness page records CPU count and runtime.
  Colab's free runtime has 2 vCPUs; laptop times are not Colab times.

### Exercise 2 · Purchases per customer in pandas (8 minutes)

- **Predict.** What share of customers bought exactly once in the first 90 days?
- **Function.** `purchases_per_customer(tx: pd.DataFrame) -> pd.DataFrame` with one row per customer:
  `customer_id, n_purchases, first_date, last_date`.
- **Checkpoint.** Row count equals `tx.customer_id.nunique()`; `n_purchases.sum() == len(tx)`;
  `first_date <= last_date` everywhere; matches the reference on 5 named customers.
- **Explain.** Two purchases on the same day count twice here. Is that one purchase or two? (Module 1
  answers: the models count purchase days.)

### Exercise 3 · The same in DuckDB SQL (8 minutes)

- **Predict.** Will SQL and pandas agree exactly? What could make them differ?
- **Function.** `purchases_per_customer_sql(con, table: str = "tx") -> pd.DataFrame` using `GROUP BY`.
  Provided: `con = duckdb.connect(); con.register("tx", tx)`.
- **Checkpoint.** Equal to the pandas result after sorting by `customer_id` (`pd.testing.assert_frame_equal`
  with dtype tolerance), with a recovery message naming the first mismatching customer.
- **Explain.** SQL is what you will write against a warehouse; DuckDB runs the same SQL in the notebook.

### Exercise 4 · Acquisition cohorts (9 minutes)

- **Predict.** Which week acquired the most customers?
- **Function.** `cohort_sizes(tx, freq: str = "W") -> pd.Series` indexed by first-purchase period.
- **Checkpoint.** Sum equals the number of customers; index is a `PeriodIndex` with the requested
  frequency; the largest cohort matches the reference.
- **Explain.** A cohort is defined by first purchase, not by calendar date of activity. Module 1 draws
  retention curves per cohort.

### Exercise 5 · Simulate purchases until the customer leaves (12 minutes)

- **Predict.** A customer buys on average every 2 weeks and stays 20 weeks on average. How many
  purchases do you expect in a year?
- **Function.** `simulate_customer(lam: float, mu: float, T: float, rng) -> np.ndarray` of purchase
  times in weeks: purchases arrive as a Poisson process with rate `lam` per week until an exponential
  lifetime with rate `mu` or the end of observation `T`, whichever comes first.
- **Checkpoint.** Over 20,000 customers with `lam=0.5, mu=0.05, T=52`, the mean number of purchases is
  within 3% of `lam * (1 - exp(-mu*T)) / mu` (`checks.close(..., rel=0.03)`); all times lie in `[0, T]`
  and are sorted (`checks.monotone`).
- **Explain.** This is the story of one Pareto/NBD customer. Tomorrow every customer gets their own
  `lam` and `mu`, and we never see when anyone left.

### Exercise 6 · Catch the agent's bug with a test (13 minutes)

- **Setup (provided).** A function `agent_rfm(tx, cutoff)`, presented as "written by a coding agent",
  that returns frequency, recency and T, with one planted bug: it measures recency from the cutoff
  backwards ("weeks since last purchase") instead of "time of last purchase since the first". The
  reference `reference_rfm` is correct.
- **Predict.** Read `agent_rfm`. Would you use it? What would you check first?
- **Function.** `test_rfm(fn) -> None`: build a three-customer transaction log by hand with known
  answers (one customer who never returned, one who bought on the cutoff day, one in between) and assert
  each value with a message.
- **Checkpoint.** `test_rfm(reference_rfm)` passes and `test_rfm(agent_rfm)` raises `AssertionError`.
  Both must hold; a test that never fails is not a test.
- **Explain.** How to use a coding agent in this workshop: let it draft, then verify with a small case
  whose answer you know, and never trust a library call it wrote from memory without checking the
  installed version. "Weeks since last purchase" is a fine feature, but it is not the `recency` the
  models expect.

### Decision · Where will you run the labs? (3 minutes)

Provided cell prints `workshop.summary()` and the environment summary. Markdown asks the learner to note:
runtime chosen, whether the R notebook ran, install minutes as measured. Link to Setup and FAQ.

## R notebook (`00-setup-warmup.R`)

Same six exercises with dplyr and the duckdb R package (`DBI::dbGetQuery`). Exercise 1 in R also reports
whether CLVTools, grf and CausalImpact load, with versions, because Modules 3, 7 and 11 need them; it
installs them from the dated P3M snapshot (`environment.p3m_url`) and records install seconds. GeoLift
is not installed here (Module 7 installs it). Exercise 6 in R uses `stopifnot()` with messages.

## Provided scaffolding

Data loading and the slice; DuckDB connection; `reference_rfm` and `agent_rfm`; the decision summary.

## Stretch (optional)

Vectorize `simulate_customer` for 100,000 customers and time both versions.

## Known pitfalls

- Colab can preload an older PyMC; the install cell's restart guard must catch it ("Restart session,
  then Run all").
- Time zones: CSV dates must be parsed as dates, not datetimes with time zones, or SQL and pandas
  disagree at midnight.
- The R runtime on Colab is a separate runtime type; the notebook header says how to switch.

## APIs verified (and how)

None beyond pandas/DuckDB basics. DuckDB 1.3.2 is the pinned Python version (`environment/colab-constraints.txt`).

## Open items for the Technical Expert

- Name the runtime report object used in Exercise 1.
- R: record per-package install seconds in the run record so `setup.qmd` can quote measured times.
