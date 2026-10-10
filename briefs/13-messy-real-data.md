# Lab brief · Module 13 · Messy real data

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. The lab is **not written and has not run**. Numbers marked *prototype* come from throwaway scripts run on a laptop (Apple Silicon, the repository's `.venv` with the pinned versions: pandas 2.2.3, numpy 2.1.3, scipy 1.16.3, duckdb 1.3.2, pymc-marketing 1.2.0). They are not lab results, not Colab times, and must be re-measured on the generator the Technical Expert builds. Reviewed by the Pedagogy Expert on 2026-10-10 against `briefs/_lab-standard.md`; the rulings, the changes and their reasons are under "Pedagogy review" at the end. |
| Notebook | `labs/src/13-messy-real-data.py` → `labs/python/13-messy-real-data.ipynb` |
| Lab slot | 55 min (`modules.m13.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (8 + 10 + 7 + 7 + 8; limit 40) + decision 5 + slack 5 = 55. Each exercise's minutes include reading the provided cells immediately before it, so provided cells print short summaries, not long tables. Minutes are estimates until a pilot. |
| Track | Extensions (day `d6`), optional. Assumes Modules 1, 2 and 4; Module 5's break-even lift helps. |

## Question and decision

**Question.** Can we trust BTYD and spend models fitted to a real, messy transaction log, and which
cleaning choices actually matter?

**Decision.** Which UCI Online Retail II customers get a retention contact at the end of June 2011, and
which cleaning choice moves that list most.

## What this lab fixes

Module 1 cleaned Online Retail II with three rules (drop lines without a customer ID, drop invoices that
start with `C`, drop `quantity <= 0` or `price <= 0`) and checked them only by row counts. Those rules keep
postage, fee and test-product lines, exact duplicate lines, and every purchase that was later cancelled
(Module 1 drops the cancellation, not the order it cancels). Modules 2 and 4 then fitted curated or
synthetic logs. Nothing so far showed whether a cleaning pipeline is right, or what a cleaning error does
to a model or to a decision.

## Design in one paragraph

Ground truth first, then real data, as in every core module. A seeded generator turns the synthetic
retailer's clean purchases into a log with **exactly the schema of `load_online_retail_ii()`** and adds
labeled errors at **rates measured on Online Retail II** (the labels live only in the truth file). The
learner writes the cleaning rules in DuckDB SQL and proves them by reproducing the clean RFM table
exactly. A one-rule-at-a-time sweep then measures what each error does. Because dozens of fits are needed,
the sweep uses a closed-form maximum-likelihood helper that maximizes the same likelihoods as
PyMC-Marketing (prototype agreement: largest relative difference 1.2 × 10⁻⁴), confirmed once in the lab
with PyMC-Marketing itself. The same pipeline is then applied unchanged to Online Retail II.

What the prototype data show, and therefore what the lab promises: at Online Retail II's error rates,
row-level errors move **aggregate** quantities only a few percent (Module 1's rules: +2.7% on the 52-week
purchase forecast, +4.2% on the population mean spend), but they move **individual customers** a lot
(2,610 of 5,025 customers' expected 52-week revenue moves by more than 10%) and they move the individual
Gamma-Gamma parameters a lot (p −19%, v +38%, because p, q and v trade off against each other). Counting
invoices instead of purchase days does more damage than any row-level error. A few wholesale buyers distort
the spend parameters; capping at the 99th percentile restores them, at the price of the wholesale
customers' value if the capped amounts are also used for valuation. On the real data, the assumed lift of
the contact moves the decision far more than any cleaning choice, but cleaning errors concentrate on
specific customers (under Module 1's rules a test account is worth £18,027 over 26 weeks instead of
£113).

## Notation (as in Modules 1, 2 and 4)

- *Line*: one row of the log (one product on an invoice). *Invoice*: one order, several lines.
  *Purchase day* (purchase occasion): all of a customer's invoices on one calendar day, merged, as in
  Module 1. A *cancellation line* belongs to an invoice whose number starts with `C`; it has a negative
  quantity. A *non-product line* has a stock code that does not start with five digits (postage `POST`,
  carriage `C2`, manual adjustment `M`, `BANK CHARGES`, test products `TEST001`, …).
- x = repeat purchase days (frequency), t_x = recency, T = age, in weeks as days / 7; z̄ = mean amount
  over repeat purchase days (`monetary_value`).
- BG/NBD: λ ~ Gamma(r, α), α a rate; dropout probability ~ Beta(a, b). Gamma-Gamma: p, q, v;
  population mean spend p v / (q − 1); expected spend of a customer
  E[Z | x, z̄] = (1 − w) · p v / (q − 1) + w · z̄ with w = p x / (p x + q − 1) (pymc-marketing 1.2.0
  `expected_customer_spend`).
- *Revenue score* R_i(h) = E[purchase days in the next h weeks] × E[spend per purchase day].
- *Pipeline*: the ordered cleaning rules. *Reference pipeline*: all rules. *Variant*: the reference with
  one rule changed.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic messy log (new; see the generator spec) | `pd.read_csv(data.repo_file("synthetic/messy_lines.csv"), dtype=..., parse_dates=["invoice_date"])` in a provided cell, cast to the dtypes of `load_online_retail_ii()` | Parts A–C |
| Its truth | `json.loads(data.repo_file("synthetic/truth_messy.json").read_text())` | checkpoints |
| Synthetic retailer (the clean answer) | `data.load_synthetic("retailer_transactions")`, `data.retailer_rfm()`, `data.load_truth()["retailer"]` | clean target, spend truth |
| UCI Online Retail II (real; CC BY 4.0) | `data.load_online_retail_ii()` (both sheets, overlap removed, snake_case; the parquet mirror is sha256-checked) | Part D, decision |

Online Retail II prices are in pounds sterling (UCI page); the synthetic retailer is in dollars. Say so in
the notebook. Calibration period for Online Retail II: to **2011-06-30**; holdout 2011-07-01 to
**2011-12-09** (the last day in the data; 23.14 weeks). The holdout includes the pre-Christmas season,
which the BG/NBD does not model; the notebook says so.

### Other datasets considered

- **CDNOW** is already the benchmark of Modules 1–4, and its rows carry only customer, date, number of CDs
  and dollars: no invoices, stock codes or cancellation markers to clean. Not used here.
- **Olist (Brazilian E-Commerce Public Dataset, Kaggle)** is **not used**. Checked 2026-10-10: Kaggle's API
  metadata gives the license as CC BY-NC-SA 4.0 (non-commercial, share-alike), so it cannot be mirrored
  next to this repository's CC BY 4.0 content, and Kaggle's API documentation says a Kaggle account is
  needed (an anonymous request to the API's download URL did return a signed download link on 2026-10-10,
  which is not documented and may stop). More decisively, it barely repeats: in the files downloaded that
  day, 99,441 orders belong to 96,096 customers (`customer_unique_id`), and only 2,015 of the 93,358
  customers with a delivered order bought on more than one day (2.2%). A BTYD model has almost nothing to
  learn from that. It does carry one lesson the briefing can mention: `customer_id` is unique per order
  (99,441 values), so keying on it makes every customer a one-time buyer; repeat buyers appear only through
  `customer_unique_id`.

## Synthetic generator spec (new file `src/mktstats/synth/messy.py`)

Not exported from `synth/__init__.py`. Function `messy_retailer(seed: int = 2033, *, retailer_seed: int = 2026,
**rates) -> SynthResult` with one table `lines`. Registered in `scripts/make_synthetic.py` `EXTENSIONS`
as `"messy"`; files `messy_lines.csv` and `truth_messy.json`. Seed 2033 is the next unused seed after the
capstone's 2032 (check against Modules 14–17).

**Data-generating process.**

1. **Clean purchases.** Start from `synth.retailer(seed=2026).transactions` (38,637 purchases of 5,000
   customers, identical to `retailer_transactions.csv`). Add **25 wholesale customers**, IDs 5001–5025,
   acquired on a uniform day in the first 26 weeks, with the retailer's reference-channel Pareto/NBD
   (r 0.8, α 5, s 0.5, β 10 per week) and Gamma-Gamma spend with p 6, q 4 and v = 15 × 40 = 600 (mean
   spend per purchase $1,200, 40 times the retail population mean of $30). The **clean log** is the union.
2. **Invoices and lines.** Each clean purchase becomes one invoice of 1 + Poisson(1) product lines with
   distinct stock codes from a 600-code catalog matching `^[0-9]{5}[A-Z]?$`. Quantities × prices add up to
   the purchase amount **exactly in cents** (quantities from {1, 2, 3, 4, 6, 12}; wholesale also 24, 48,
   96). Timestamp: the purchase date plus a uniform minute between 08:00 and 19:59. Description
   `"ITEM <code>"`. Country: one per customer, 85% `"United Kingdom"` (it carries no signal).
3. **Errors**, at rates matched to Online Retail II (measured 2026-10-10 on the table
   `load_online_retail_ii()` returns; the real rate is in brackets):

| Label | What is added | Rate (real) | Rule that removes it |
|---|---|---|---|
| `guest` | Extra invoices with no customer ID, amounts Gamma(6, scale 5) | 29% as many invoices as identified ones, ≈ 23% of lines (22.5% of lines have no ID) | `no_customer` |
| `postage` | A `POST` line ("POSTAGE", quantity 1, $4.95) on an identified invoice | 4.9% of invoices (4.88%) | `non_product` |
| `carriage` | A `C2` line ("CARRIAGE", $25) | 0.7% of invoices (254 of 36,975) | `non_product` |
| `fee_invoice` | A separate invoice with one `M` ("Manual", $1–$60) line for 7% of customers, or one `BANK CHARGES` ($15) line for 2%, on a random day after the first purchase | ≈ 1% of invoices (378 of 36,975 identified invoices hold only non-product lines) | `non_product` |
| `duplicate` | An exact copy of a line, all eight columns equal, placed right after it | 1.44% of lines (1.44% of identified lines) | `duplicate` |
| `cancelled_order` | An extra order (same customer, on a purchase day or up to 14 days later) and a `C` invoice negating every line | 1.2% of purchases; same day for 15% (11.6% of matched cancellations are same-day) | `cancellation` |
| `returned_line` | An extra line on a real invoice and a later `C` invoice returning it in full (same code, price, quantity), 1–30 days later (same day 15%) | 2% of purchases | `cancellation` |
| `same_day_split` | A purchase with ≥ 2 lines split into two invoices 5–180 minutes apart on the same day | 9% of purchases (10.2% of identified invoices are a customer's second or later invoice that day) | counting purchase days |
| `wholesale` | The 25 wholesale customers (step 1) | 0.5% of customers | none: they are real (Part C) |

   Combined, cancellation lines are ≈ 2.1% of identified lines (real 2.28%). Every synthetic cancellation
   has an original, so exact recovery is possible; in the real data only some do (Part D). Invoice
   numbers: six digits in timestamp order from 500000; cancellation invoices get a `C` prefix. No synthetic
   cancellation or return crosses `2025-12-28`; some cross the calibration end (2024-12-29), so the
   pipeline must clean before it splits.
4. **Output.** Table `lines` with exactly the columns and dtypes of `load_online_retail_ii()`: `invoice`
   (str), `stock_code` (str), `description` (string), `quantity` (int64), `invoice_date` (minute
   resolution), `price` (float64), `customer_id` (Int64, missing for guests), `country` (str). Rows in
   timestamp order. **No label column.** Write `invoice_date` as a preformatted `"%Y-%m-%d %H:%M"` string:
   `make_synthetic._write_frames` passes `date_format="%Y-%m-%d"`, which would drop the time.

**Truth file `truth_messy.json`.** `seed`, `retailer_seed`, the rates, `calibration_end` 2024-12-29,
`holdout_end` 2025-12-28; for each **rule** (`no_customer`, `non_product`, `duplicate`, then
`cancellation` applied to the lines the first three keep, then Module 1's `quantity > 0 and price > 0`):
the number of lines it flags and a sha256 of the sorted 0-based row positions (for `cancellation`, also
`cancel_lines`, the number of cancellation lines, prototype 1,751 of 3,502 flagged); for each **label** in the
table above, the line count and the sha256 of its positions; `wholesale`: IDs, the multiplier, each
customer's true λ, μ, alive at the calibration end, true mean spend, their **clean transactions**
(about 200 rows, so a checkpoint can rebuild the full clean log as `retailer_transactions` plus these), and
their true expected 26- and 52-week revenue (closed form, undiscounted:
alive · m · λ · (1 − e^(−μh)) / μ, with the limit λ h when μ = 0); the same totals for the 5,000 retail
customers from `retailer_customers.csv`; the realized holdout purchase days of the clean log;
`tolerances` measured over independent generator seeds (Open items).

**Size.** Prototype: 110,480 lines, 7.2 MB as CSV (1.5 MB gzipped), five times the largest committed
data file (Open items).

## Model fits

No MCMC in the core lab. Every fit is a maximum-likelihood (flat-prior MAP) fit:

| Fit | How | Count | Prototype time (laptop) |
|---|---|---|---|
| BG/NBD and Gamma-Gamma for each variant (Parts B–D) | new helper `mktstats.mle` (closed-form log-likelihoods, scipy) | about 20 fits in all | 0.08 s (BG/NBD, CDNOW) to 0.2 s (retailer) each; whole synthetic sweep 2.1 s, real sweep 3.2 s |
| BG/NBD and Gamma-Gamma on the messy log's reference pipeline (start of Part B, before the first sweep) | PyMC-Marketing `BetaGeoModel` / `GammaGammaModel`, `Prior("HalfFlat")` for every parameter, `method="map"` | 2 (once, as a cross-check) | 2.6 s and 0.2 s after earlier compiles (prototype on cleaned Online Retail II; agreement on the retailer 4 × 10⁻⁵ and < 10⁻⁵) |

Why the helper: on Colab a single PyMC-Marketing MAP fit costs tens of seconds, mostly compilation (spike
S2: a first BG/NBD MAP on CDNOW took 30.3 s), so twenty of them would break the 4-minute cell target.
The helper maximizes the likelihoods written in the installed pymc-marketing 1.2.0 source
(`clv/distributions.py` `BetaGeoNBD.logp`; `clv/models/gamma_gamma.py` `GammaGammaModel.build_model`,
the potential "Eq 1a" of Fader & Hardie's note 025). Prototype: it reproduces note 004 (r 0.2426,
α 4.4136, a 0.7929, b 2.4259) and note 025 (p 6.2496, q 3.7442, v 15.4435) on CDNOW, and agrees with the
PyMC-Marketing flat-prior MAP to a largest relative difference of 1.2 × 10⁻⁴ (BG/NBD) and 1 × 10⁻⁵
(Gamma-Gamma) on cleaned Online Retail II, and 4 × 10⁻⁵ and < 10⁻⁵ on the retailer. Each variant is
fitted once, but the lab fits many variants: a deliberate exception to "fit each model once per lab"
(CONTRIBUTING.md), because the comparison across variants is the lesson and each fit is sub-second.

**Pedagogy Expert ruling (2026-10-10): exception approved, on three conditions.** From the learner's side
the helper is not a new model: Module 2 fitted the BG/NBD with flat priors by MAP and said that this *is*
the maximum-likelihood estimate Fader, Hardie & Lee published (r 0.243, α 4.414, a 0.793, b 2.426 on
CDNOW), and Module 4 did the same for the Gamma-Gamma (p 6.25, q 3.74, v 15.44). The helper maximizes the
same likelihood without building a PyMC model. What a learner cannot do is trust twenty fits from a
function they have never seen, so:

1. **Introduce the helper before its first use**, at the start of Part B, not in Part D: one Markdown
   paragraph (below, "What gets fitted, and why not PyMC-Marketing") that says what is maximized, that the
   answer is a point estimate with no interval (as MAP was in Module 2), and why the sweep cannot use
   PyMC-Marketing (each PyMC-Marketing fit builds and compiles a model; twenty of them would take minutes).
2. **The one PyMC-Marketing cross-check runs there**, on the reference pipeline's calibration RFM of the
   messy log (the table the sweep starts from), and the cell prints the seconds each method took, measured
   in that run. Confirming before relying is the order a skeptical analyst expects; the cross-check was in
   Part D, after twenty fits had already been used. Moving it also takes a 1–2 minute cell out of Part D,
   which is the most crowded part of the lab. The same model is still fitted with PyMC-Marketing once.
3. **Say plainly what the exception is**: each variant is fitted once; the lab fits many variants because
   the comparison is the lesson; every fit is a point estimate, so the comparisons in Parts B–D have no
   intervals, and the Decision's uncertainty comes from the assumed lift and the cleaning choices (the
   stretch adds a posterior).

**QUICK** changes nothing in this lab: nothing samples, and the whole lab is estimated at under 3 minutes
of compute. The header says so, and the Decision cell still prints the mode.

**Expected run time on Colab FULL (estimates, no run record yet).** Install about 40 s (the Colab records of
Modules 2 and 4, which install PyMC-Marketing, measured 37.4–46.0 s over their three runs on 2026-10-10;
for this lab it is an estimate). Reading the messy CSV and the Online Retail II parquet: seconds each
(Module 1's whole notebook, which loads Online Retail II, took 9.1–10.8 s on Colab). DuckDB rules on 1
million rows: under 1 s on the laptop. Both sweeps: under 30 s each. The PyMC-Marketing cross-check cell at
the start of Part B (import plus two MAP fits): 1 to 2 minutes, the largest cell (from S2: import 16.8 s,
first BG/NBD MAP 30.3 s); the Markdown before it tells the learner to read Exercise 3's Predict while it
runs. Total compute: under 3 minutes. Targets: install ≤ 3 min, any cell ≤ 4 min, total ≤ 15 min.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · A dirty log with a known clean answer | *Run: load the messy log (provided)* | — |
| | 1 · Flag non-product and duplicate lines | 8 |
| | *Run: ASOF JOIN on four orders (worked example, provided)* | — |
| | 2 · Match cancellations to the purchases they cancel | 10 |
| | *Run: the pipeline and the exact-recovery check (provided)* | — |
| B · Which errors matter, and for what | *Run: what gets fitted, and the PyMC-Marketing cross-check (provided)* | — |
| | 3 · Aggregate and individual effects (the provided sweep runs inside it, after the Predict) | 7 |
| C · Extreme orders | *Run: the wholesale customers (provided)* | — |
| | 4 · Cap order values for the spend fit | 7 |
| D · Online Retail II | *Run: three provided cells (the pipeline on the real log; one customer, two pipelines; fits and holdout)* | — |
| | 5 · The retention-contact list | 8 |
| **Exercises** | | **40** |
| Decision | Whom do we contact, and which cleaning choice moves the list? | 5 |

**Running clock** (estimates until a pilot; `teach.qmd` repeats it): open and install 0–5; Exercise 1 to
minute 13; Exercise 2 to 23; Exercise 3 to 30; Exercise 4 to 37; Exercise 5 to 45; Decision to 50; 5
minutes of slack. Exercise 2 (new SQL) is the likeliest to overrun; anyone still on it at minute 25 runs
`workshop.use_reference(2)` and moves on.

**Prototype answers stay out of the notebook's prompts.** The parenthesized prototype values after each
Predict below are for the builder. The notebook prints the measured answer after the checkpoint, never
next to the prompt. If a measured answer lands near a bin boundary, move the boundaries so that it sits
well inside a bin, and tell the reviewer.

Names used throughout: `lines` (the raw log), `flags`, `kept`, `matches`, `tx` (purchase days:
`customer_id, date, amount`), `rfm`, `truth` (the messy truth), `retail_truth` (`truth.json["retailer"]`),
`scores`. Every random step (none in the core lab; the stretch has one) uses a visible seed.

### Terms the notebook defines, where they are first used

One plain sentence each, in the cell that first needs it, not in an opening glossary. Suggested wording:

| Term (first used) | One sentence for the notebook |
|---|---|
| line, invoice (Part A opening) | A line is one row of the log: one product on one invoice. An invoice is one order, usually several lines. |
| purchase day (Part A opening) | All of a customer's invoices on one calendar day, merged into one purchase, as in Module 1. |
| cancellation line (Part A opening) | A line on an invoice whose number starts with `C`; its quantity is negative. |
| non-product line (Exercise 1) | A line whose stock code does not start with five digits: postage, carriage, fees, bank charges, test products. |
| exact duplicate (Exercise 1) | A line equal to an earlier line in all eight columns; the first copy is kept, later copies are flagged. |
| flag (Exercise 1) | A true/false column that marks a line for removal without removing it, so each rule can be switched off later. |
| ASOF join (worked example before Exercise 2) | A join that matches each row on the left to the nearest earlier row on the right with the same keys, at most one. |
| pipeline, variant (Part A pipeline cell) | The pipeline is the ordered list of cleaning rules; a variant is the pipeline with one rule changed. |
| maximum-likelihood fit (Part B opening) | The parameter values that make the observed RFM table most likely; Module 2's MAP with flat priors gives the same values. |
| cap at a percentile (Part C) | Replace every purchase-day amount above, say, the 99th percentile with that percentile's value (also called winsorizing). |
| wholesale customer (Part C) | A business that buys in bulk: few purchase days, very large amounts. Real customers, not errors. |
| left censoring (Part D) | Customers who bought before the data start enter the data as if they were new, so their age T is too short. |
| rank correlation (Part D) | Spearman's correlation: Pearson's correlation of the ranks, so a few huge values cannot dominate it. |
| contact lift (Exercise 5) | The assumed share by which a retention contact raises a customer's purchases; an assumption until an experiment measures it. |

### Provided before Exercise 1

Load the messy log with the real loader's dtypes, add `line_id = 0, 1, …` (row position), register it in
DuckDB as `lines`. One sentence: the messy log has exactly the columns of Online Retail II, which Module 1
cleaned, with the same kinds of mess at the same rates; it was made from the synthetic retailer, so the
clean answer is known, and Part D runs the pipeline you build here on the real log. Print: lines, invoices,
lines per invoice, the share of lines without a customer ID and on `C` invoices, the number of lines Module
1's three rules keep (Exercise 1's Predict uses it), and three example rows found by pattern, not by label:
a `POST` line, a `C` invoice line and a line without a customer ID. Online Retail II is **not** loaded here:
it loads in Part D, so nothing before Exercise 1's first checkpoint downloads more than the messy log
(prototype 7.2 MB as CSV, 1.5 MB gzipped; Open item 2).

### Exercise 1 · Flag non-product and duplicate lines (8 minutes)

The minutes include reading the opening cell; this is the first exercise of the lab.

- **Predict.** Module 1's three rules keep about three lines in four of this log (the opening cell prints the
  count). Of the lines they keep, what share will your two new rules flag as non-product or duplicate:
  under 1%, 1 to 10%, or over 10%? Write your guess. (Not measured directly in the prototype; its shares of
  all lines, 2.4% non-product and 1.5% duplicate, put it near 5%. The checkpoint cell prints the measured
  share.)
- **Function.** `flag_lines(con, table: str = "lines") -> pd.DataFrame` with one row per line, ordered by
  `line_id`: `line_id`, `no_customer` (`customer_id IS NULL`), `non_product` (stock code does not start
  with five digits: `NOT regexp_matches(stock_code, '^[0-9]{5}')`), `duplicate` (an exact copy of an earlier
  line: `ROW_NUMBER() OVER (PARTITION BY` all eight original columns `ORDER BY line_id) > 1`). A line can
  carry more than one flag. **The stub already contains the query skeleton with the `no_customer` column
  written**, as a pattern; the learner writes the other two expressions. The Task text names
  `regexp_matches(text, pattern)` and recalls `ROW_NUMBER()` from Module 1.
- **Checkpoint (two, before any fit).**
  1. A toy table built in the cell: a product line kept; its exact copy flagged `duplicate` (the first copy
     not); a line equal except for the quantity kept; `POST`, `M` and `TEST001` flagged `non_product`; a
     product code with a letter suffix (`15056BL`) kept; a line without a customer flagged `no_customer`;
     an exact copy of a no-customer line flagged as both. Recovery messages name the case, e.g. "you flag
     both copies of line 0: keep the first (ROW_NUMBER() ... > 1)".
  2. On the messy log: each column's count equals `truth["rules"][name]["count"]` and the sha256 of its
     flagged positions equals the truth's; the message gives both counts. The cell then prints each flag's
     count and share, the share of Module 1's kept lines the two new rules flag (the Predict's answer), and
     the lines of one customer who has a standalone `M` invoice (Explain uses it).
- **Failure messages** (the four parts of `briefs/_lab-standard.md` §5, each ending with "To move on now,
  run `workshop.use_reference(1)`"). Checked on a toy of this shape with DuckDB 1.3.2 on 2026-10-10:
  - Both copies of a duplicate flagged (`COUNT(*) OVER (PARTITION BY …) > 1` instead of `ROW_NUMBER()`):
    "the first copy of a line is the real one: flag only later copies, `ROW_NUMBER() OVER (PARTITION BY
    <all eight columns> ORDER BY line_id) > 1`".
  - The line that differs only in quantity flagged: "a duplicate must match in all eight columns: partition
    by every original column, not by invoice and stock code".
  - `15056BL` flagged (pattern anchored at both ends, `'^[0-9]{5}$'`): "a product code can end in a letter:
    test only that the code *starts* with five digits, `'^[0-9]{5}'`".
  - The copy of the no-customer line not flagged `duplicate` (a self-join on `=`, where a missing value never
    equals another): "the two lines are equal including the missing customer ID; `PARTITION BY` treats
    missing values as equal, a join on `=` does not".
  - On the messy log, the right count but a different hash: "you flag the right number of lines but not the
    same lines: for `duplicate`, check `ORDER BY line_id` inside the window, so the earliest copy is kept".
- **Explain.** Compare with your guess: a few percent of the lines. Two lines Module 1 keeps: a `POST` line
  on an order, and a standalone `M` invoice on a day the customer bought nothing else. Which one adds a
  purchase day to the customer's frequency, and which only inflates an amount? Point to the customer printed
  above. Details: the `POST` line inflates that purchase's amount; the standalone `M` invoice adds a
  purchase day (frequency + 1) with a made-up amount; a duplicate line inflates the amount. On real data the
  five-digit rule also drops a few product-like codes (`PADS`, 19 lines with a customer; `DCGS…` and gift
  vouchers, none with a customer; `SP1002`, 2): state it as a known approximation. Exact copies may be
  genuine repeated lines; treating them as errors is an assumption, measured in Part B.

### Run · ASOF JOIN on four orders (worked example, provided before Exercise 2)

Exercise 2 needs a join most analysts have not used, so the notebook shows it first on a case they know,
last-click attribution, as Module 1 showed `LAG` before Exercise 2 used it. Two tiny tables, `clicks`
(customer_id, clicked_at, campaign) and `orders` (order_id, customer_id, ordered_at), and:

```sql
SELECT o.order_id, o.customer_id, o.ordered_at, c.campaign, c.clicked_at
FROM orders AS o ASOF LEFT JOIN clicks AS c
  ON o.customer_id = c.customer_id AND o.ordered_at >= c.clicked_at
ORDER BY o.order_id
```

Data (ran with DuckDB 1.3.2, 2026-10-10): clicks (1, 03-01 09:00, spring_email), (1, 03-04 18:00,
search_ad), (2, 03-05 12:00, social_ad); orders 101 (customer 1, 03-02 10:00), 102 (1, 03-06 08:00), 103
(2, 03-05 11:00), 104 (3, 03-06 09:00). Result: 101 → spring_email (the latest click before it; the search
click came later), 102 → search_ad (the later of two earlier clicks), 103 → none (its only click came after
the order), 104 → none (no clicks). One sentence before the output: "Look at orders 101 and 103: each
order gets the customer's latest click at or before it, at most one, and `LEFT` keeps orders with none." The
equality conditions choose the group (here the customer); the one inequality chooses the nearest earlier
row. Reading time comes out of Exercise 2's minutes.

### Exercise 2 · Match cancellations to the purchases they cancel (10 minutes)

The ASOF join is new SQL to most of this audience; the minutes include reading the worked example above.

- **Predict.** Module 1 drops cancellation lines but keeps the purchases they cancel. Does that make the
  average customer look like a bigger spender, a more frequent buyer, both, or neither? Write your guess.
  (Prototype: both, slightly: mean frequency +1.4%, mean repeat spend +1.2%. The exact-recovery cell after
  the pipeline prints both means under Module 1's cancellation rule and under yours.)
- **Function.** `match_cancellations(con, table: str = "kept") -> pd.DataFrame` with columns `cancel_id`,
  `original_id` (the `line_id`s of the cancellation line and of the purchase line it cancels; `original_id`
  missing when there is no match), one row per cancellation line, where `kept` is the lines Exercise 1 does
  not flag (registered by a provided cell). A cancellation line cancels the **latest earlier** purchase line
  of the same customer with the same stock code, the same price and the opposite quantity, at or before its
  timestamp. In DuckDB:
  `FROM c ASOF LEFT JOIN p ON c.customer_id = p.customer_id AND c.stock_code = p.stock_code AND c.price =
  p.price AND -c.quantity = p.quantity AND c.invoice_date >= p.invoice_date`, with `c` the lines whose
  invoice starts with `C` and `p` the others. The pipeline drops every cancellation line and every
  matched original. **The Task text states the matching rule in words** (same customer, same stock code,
  same price, opposite quantity, at or before the cancellation) and points back to the worked example;
  **the stub contains the two CTEs `c` and `p`** and leaves the join and its conditions as the TODO, so the
  one new thing is writing the ASOF conditions. The checkpoint tests results, not syntax: any query that
  picks the latest earlier matching purchase passes.
- **Checkpoint (two).**
  1. Toy: a same-day full cancellation is matched; a return of fewer units than bought is not; a
     cancellation dated before the purchase is not; with two identical earlier purchases, the later one is
     matched.
  2. On the messy log: one row per cancellation line (`truth["rules"]["cancellation"]["cancel_lines"]`;
     prototype 1,751), every one matched, no `original_id` twice.
- **Failure messages** (four parts, each ending with "To move on now, run `workshop.use_reference(2)`").
  The toy's four cases, checked with DuckDB 1.3.2 on 2026-10-10, separate the likely mistakes:
  - Partial return matched (no quantity condition): "this cancellation returns 1 of the 3 units bought;
    a cancellation matches only a line with the opposite quantity, `-c.quantity = p.quantity`".
  - Rows missing for unmatched cancellations (`ASOF JOIN` without `LEFT`): "you return N rows for M
    cancellation lines: use `ASOF LEFT JOIN`, so a cancellation with no match stays, with `original_id`
    missing".
  - Two rows for one cancellation, or the earlier of two identical purchases (an ordinary join, or a
    `MIN`): "an ordinary join returns every earlier match; `ASOF` returns only the latest earlier one".
  - The cancellation dated before its purchase matched (inequality reversed): "a cancellation can only cancel
    an earlier purchase: `c.invoice_date >= p.invoice_date`".
  - On the messy log, unmatched cancellations: "N cancellation lines found no purchase. Run your function on
    `kept` (Exercise 1's unflagged lines), and keep `C` invoices in `c` only".

**Provided pipeline** (a Run cell after Exercise 2's checkpoint, reused in Parts B, C and D; the learner
reads its docstrings, not its code):
`purchases(lines, flags, matches, keep_rules=(), unit="day") -> tx` drops flagged and matched lines except
the rules in `keep_rules`, applies Module 1's last rule (`quantity > 0 and price > 0`), and sums
`quantity * price` per customer and purchase day (`unit="day"`) or per invoice (`unit="invoice"`, dated
by the invoice's first timestamp). `calibration_rfm(tx, cal_end)` is
`data.rfm_summary(tx, "customer_id", "date", "amount", observation_period_end=cal_end, time_unit="D",
time_scaler=7)`, named once so that Part B's sweep and Exercise 4 use the same call. For `unit="invoice"`,
a provided `rfm_by_invoice(tx, cal_end)` counts
each invoice as a purchase (frequency = invoices − 1, recency and T in days / 7, monetary value = mean
over invoices after the first); `checks.rfm_table` is not applied there, because two invoices on one day
give recency 0 with frequency 1.

**Run · Exact recovery** (provided, right after the pipeline; a label checkpoint, "Part A: the clean
answer"). The pipeline turns the kept lines into purchase days, and `calibration_rfm(tx, "2024-12-29")`
must equal the RFM of the clean log (`retailer_transactions` plus the truth's wholesale transactions) for
all 5,025 customers: `checks.frames_agree(..., on="customer_id")` at its default 1e-9 tolerances.
Prototype: exact. On a failure the cell prints the first customers that differ and which label touches
them, and the message routes the learner: a `duplicate`, `postage`, `carriage` or `fee_invoice` label →
Exercise 1, `workshop.use_reference(1)`; a `cancelled_order` or `returned_line` label → Exercise 2,
`workshop.use_reference(2)`. On a pass it prints the Predict's answer (mean frequency and mean repeat
spend under Module 1's cancellation rule and under yours) and the number of matched pairs whose purchase
falls on or before the calibration end and whose cancellation falls after it (the Explain's evidence).

- **Explain** (Exercise 2's, after the exact-recovery cell). Compare with your guess. Then look at the
  matched pairs that straddle the calibration end: if you split the log at the calibration end first and
  matched cancellations within each part, what would happen to those purchases? Details: their
  cancellations would be in the holdout part, so the purchases would stay in the calibration data as if
  they were real; clean the whole log, then split. The matching also runs after the duplicate rule: a
  duplicated original would otherwise be the match. On real data, the prototype matched 6,107 of 17,586
  cancellation lines (35%); the rest return part of an order, adjust a price or return goods bought before
  December 2009. They are dropped without netting, so those purchases
  keep their gross amount: an approximation the notebook states.

### Run · What gets fitted, and why not PyMC-Marketing (provided, start of Part B)

The learner meets the helper here, before any sweep uses it (the ruling under "Model fits"). One Markdown
paragraph, in this spirit:

> From here on, the lab fits the BG/NBD and the Gamma-Gamma many times: once for each way of cleaning
> the log. Each fit is a *maximum-likelihood fit*: the parameter values that make the observed RFM table
> most likely. You have met it: in Modules 2 and 4, PyMC-Marketing's MAP with flat priors gave exactly
> these values and reproduced Fader, Hardie & Lee's published CDNOW estimates. `mktstats.mle` maximizes
> the same two likelihoods (written with numpy from the formulas in PyMC-Marketing's source) with SciPy's
> optimizer. It builds no PyMC model, so there is nothing to compile; the next cell times both. Like MAP,
> it gives one number per parameter and no interval. As in Module 4, the purchase model is the BG/NBD;
> the retailer was generated by a Pareto/NBD, so the BG/NBD has no true values to hit: Part B compares
> each variant's purchase model with the reference pipeline and the realized holdout, and its spend model
> with the true p, q, v.

Then one Run cell: the reference pipeline's calibration RFM of the messy log (from Part A), fitted with the
helper and with PyMC-Marketing (`Prior("HalfFlat")` for every parameter, `method="map"`), printed side by
side with each method's seconds **measured in this run**, and a label checkpoint ("helper = PyMC-Marketing")
that the two agree within 0.5% per parameter (prototype on the retailer: 4 × 10⁻⁵ and < 10⁻⁵). The
Markdown before the cell says: "This cell takes about a minute on Colab (an estimate), most of it
PyMC-Marketing compiling its model. While it runs, read Exercise 3's Predict below and write your guess."
Failure message: "The helper and PyMC-Marketing disagree on <param> (<a> against <b>). Your Part A code is
not involved: both fits use the same table. Rerun this cell; if it still fails, tell the instructor (a
version mismatch). To move on now, run the next cell: Part B uses the helper's values."

### Exercise 3 · Aggregate and individual effects (7 minutes)

The minutes include reading the sweep's two tables, which are printed after the Predict.

- **Predict.** Under Module 1's rules, does total expected 52-week revenue change by under 2%, 2 to 10%, or
  over 10%? And how many of the 5,025 customers see their own score move by more than 10%: under 100, 100
  to 1,000, or over 1,000? Write both guesses. (Prototype: +6.7%; 2,610.)

**Run · One rule at a time** (provided sweep, inside Exercise 3, after the Predict). Variants on the
messy log: the reference pipeline; keep non-product lines; keep duplicate lines; Module 1's cancellation
rule (drop `C` lines, keep their originals); Module 1's rules (all three kept); count invoices, not
purchase days. For each: calibration RFM, `mle.bgnbd_mle`, `mle.gamma_gamma_mle`, and per-customer
expected 52-week revenue `scores[variant] = R_i(52)`. Counting **lines** as purchases is shown as a
mean-frequency line only (prototype: 8.34 against 3.55): the BG/NBD fit degenerates on it in the prototype
(r and α → 0), so it is not fitted. The output is **two narrow tables**, not one ten-column table that
scrolls sideways: the purchase model (r, α, a, b, the 52-week forecast of holdout purchase days against
the clean log's realized 14,882, prototype) and the spend model (p, q, v and the population mean spend
against the retail truth, $30; the wholesale customers are in every variant). The reference row shows
values; the other rows show the change from it in percent, as in the prototype table under "Prototype
evidence".

One sentence before the tables: "Look at which table moves more: the purchase model or the spend model." A
label checkpoint: the reference row equals the fit on the clean log (it is the same table); its holdout
forecast lies within the tolerance the Technical Expert measures (prototype: 16,178 predicted against
14,882 realized, +8.7%, because a BG/NBD is fitted to Pareto/NBD data and the holdout is a single
realization; no looser than 15%, as Module 2).

- **Function.** `variant_effects(scores: pd.DataFrame, reference: str = "reference", threshold: float =
  0.10) -> pd.DataFrame`, one row per column of `scores` (index `customer_id`, one column per variant):
  `total_change_pct` = 100 × (Σ variant / Σ reference − 1); `customers_moved` = number of customers with
  |variant / reference − 1| > `threshold`. The provided display cell after the checkpoint adds
  `largest_change_pct` (100 × the largest absolute change) for the reader.
- **Checkpoint.** Toy frame with hand-computed answers (no value exactly at the threshold; one customer
  whose score falls, one whose score rises, and a total change that differs from the mean of the
  per-customer changes); the reference row is all zeros; on the sweep's `scores`, equal to an independent
  computation in the checkpoint cell to 1e-9 (as Module 1's cohort check). The display cell after it
  prints the table with `largest_change_pct` added.
- **Failure messages** (four parts, each ending with `workshop.use_reference(3)`):
  - Mean of per-customer changes instead of the change in the total: "`total_change_pct` compares totals,
    100 × (sum of the variant ÷ sum of the reference − 1), not the average of each customer's change".
  - Only rises counted: "count customers whose score moved by more than 10% in either direction: use the
    absolute change".
  - A fraction instead of a percent (0.067 for 6.7): "`total_change_pct` is in percent: multiply by 100".
  - A non-zero reference row: "compare every column with the `reference` column, including the reference
    itself, which must give 0 and 0".
- **Explain.** Compare with your prediction. Why can the total move a few percent while thousands of
  customers move by more than 10%? Point to the p, q, v columns of the spend-model table. Details:
  Module 1's leftovers shift a minority of customers' data, but the spend parameters are shared by
  everyone, and p, q, v trade off against each other (prototype: p −19%, v +38%, population mean
  +4.2%), so every customer's shrinkage toward the population mean changes. Counting invoices does more damage than any row-level rule
  (prototype: r/α +30%, p −57%, v +138%; 1,504 customers moved). Duplicates leave the purchase model
  untouched (they never add a day). Report the quantity your decision uses, not the parameters.

### Run · The wholesale customers (provided, start of Part C)

Exercise 4's Predict needs a scale, so Part C opens with one sentence and one small table: 25 of the 5,025
customers are wholesale buyers (IDs 5001–5025); their true mean spend per purchase day is $1,200, against
$30 for the retail population, and they are real customers, not errors. The table lists the ten largest
calibration purchase days with customer ID and amount. Reading time comes out of Exercise 4's minutes.
(Prototype, for the builder: 90 of the 229 calibration purchase days above the 99th percentile are
wholesale.)

### Exercise 4 · Cap order values for the spend fit (7 minutes)

The minutes include reading the wholesale cell above. `calibration_rfm` (provided with the pipeline)
keeps the RFM step out of the learner's function.

- **Predict.** Cap every purchase day's amount at the 99th percentile. Will the fitted population mean
  spend move toward the true $30 or away from it? And if the capped amounts are also used to value
  customers, will the 25 wholesale customers' predicted 52-week revenue, compared with their true value,
  be lower by under 25%, 25 to 75%, or over 75%? Write both guesses. (Prototype: toward, 31.66 → 30.69;
  over 75%, −90%.)
- **Function.** `capped_spend_fit(tx, pct: float, cal_end: str) -> dict` returning `cap` (the `pct`
  percentile of the calibration purchase-day amounts, all customers, first purchases included, pandas'
  default linear interpolation) and `p`, `q`, `v` from `mle.gamma_gamma_mle` on the calibration RFM of
  `tx` with amounts clipped at `cap` (`calibration_rfm`, repeat buyers only). The Task text gives the
  percentile convention as a hint: `pct` is in percent, and `Series.quantile(0.99)` of 1, 2, …, 100 is
  99.01 with the default linear interpolation (checked with pandas 2.2.3 on 2026-10-10).
- **Checkpoint.** On the messy log at `pct=99`, the cap equals the reference (prototype $149.48) and p, q, v
  lie within the truth tolerance measured over generator seeds
  (`truth["tolerances"]["capped_gamma_gamma"]`; prototype on the committed seed: 6.134, 4.058, 15.300
  against 6, 4, 15); the population mean spend within the same block's tolerance. At `pct=100` the cap is
  the largest calibration amount and p, q, v equal the reference row of Part B's spend table to 1e-6 (so
  the cap is applied to the amounts and nothing else changed).
- **Failure messages** (four parts, each ending with `workshop.use_reference(4)`). The checkpoint computes the
  caps that the likely mistakes give and names the one that matches the learner's:
  - Cap from all purchase days, holdout included: "your cap uses purchase days after the calibration end:
    compute it from `date <= cal_end` only".
  - Cap from repeat purchase days only, or from each customer's mean (`monetary_value`): "cap each purchase
    day's amount, first purchases included, before building the RFM table; capping customers' means is a
    different, weaker cap".
  - `quantile(pct)` raises "percentiles should all be in the interval [0, 1]" (pandas 2.2.3, checked
    2026-10-10); the checkpoint catches that `ValueError` and says: "`pct` is in percent: use
    `quantile(pct / 100)`".
  - The cap right but p, q, v outside the tolerance: "fit on the clipped amounts, repeat buyers only
    (`rfm[rfm['frequency'] > 0]`)".
- **Explain.** Use the provided table that follows (caps at none, 99.9, 99.5, 99 and 95, plus an "oracle"
  row that drops the wholesale customers using the truth labels, and the wholesale customers' 52-week value
  against their truth three ways: no cap, P99 for fitting only, P99 amounts for valuation too). Compare
  with your guesses. Which of the three would you use to value customers, and why does the cap belong in
  the fit but not in the valuation? Details: a cap stops a few huge purchases from dictating the shape of
  the spend distribution, which is why P99 lands near the oracle; capping deeper (95th percentile) also
  trims ordinary large retail orders and biases the parameters the other way. Gamma-Gamma shrinks each
  customer's mean toward the population mean, and most wholesale customers have few repeat purchases, so
  even uncapped they are undervalued (prototype −29%). **Cap for fitting, not for valuation:** with
  P99-fitted parameters and their raw means, the wholesale value error is −33%, against −90% with capped
  amounts. A segment of its own is the real fix (stretch).

### Run · Online Retail II through the tested pipeline (provided, Part D)

Three provided cells, each with a sentence before it saying what to look at and a short printed summary
(their reading time, about three to four minutes, comes out of Exercise 5's minutes). The PyMC-Marketing
cross-check is not here: it runs at the start of Part B ("Model fits").

**D1 · Your pipeline on the real log.** The Markdown before it: "The next cell loads Online Retail II. The
first time, it downloads a 7 MB copy and checks its SHA-256 hash (seconds); if that copy cannot be reached,
it downloads the 45 MB original from UCI and converts it, which takes minutes. While it loads, read D2's
text below." The cell:

1. Loads Online Retail II and prints its first rows above the messy log's (same columns, same kinds of mess;
   this comparison moved here from the opening so that nothing before Exercise 1 downloads it).
2. Runs the learner's `flag_lines` and `match_cancellations` and the provided pipeline unchanged. Label
   checkpoint "Online Retail II counts": each rule's count equals the reference measured on the
   sha256-checked table (prototype: no customer 235,287; non-product 5,992; duplicate 11,812; cancellation
   lines among kept lines 17,586, of which 6,107 matched and 123 originals matched twice; 23,570 lines
   dropped by the cancellation rule; 32,721 purchase days of 5,839 customers). A failure routes the learner
   as the exact-recovery check does: a flag count → `workshop.use_reference(1)`; a cancellation count →
   `workshop.use_reference(2)`.
3. Prints a five-line summary: lines; lines flagged by each rule; cancellation lines matched (count and
   share); purchase days; customers. One sentence says what the 123 mean: an ASOF join matches each
   cancellation to at most one purchase, but two cancellations can pick the same purchase; one-to-one
   pairing is left as an approximation.

**D2 · Fits and the holdout.** One sentence before the plot: "Look at whether the predicted and actual
bars agree for customers with few and with many calibration purchases."

1. Calibration RFM at 2011-06-30 (prototype: 5,008 customers, 3,390 repeat buyers) and holdout purchase
   days to 2011-12-09. The cell names **left censoring**: customers who bought before December 2009 enter
   the data as if new (Module 1's cohort plot), so their T is too short (stretch).
2. Fits with the `mle` helper only (it was checked against PyMC-Marketing at the start of Part B). Label
   checkpoint: each parameter within 0.5% of the reference value measured on the sha256-checked table
   (prototype: r 0.672, α 9.444, a 0.150, b 3.051; p 2.402, q 3.553, v 418.8; population mean £393.9; the
   prototype's PyMC-Marketing flat-prior MAP agreed with these to a largest relative difference of
   1.2 × 10⁻⁴; Open item 1 asks for a test that keeps checking it).
3. Holdout tracking by calibration frequency (0, 1, …, 6, 7+), table and plot (the Module 2 pattern). Label
   checkpoint: total predicted within 10% of actual (prototype: 6,738 against 7,107, −5.2%). The holdout
   includes the pre-Christmas season, which the BG/NBD does not model; the cell says so.
4. Independence check among repeat buyers: Pearson and Spearman (rank) correlation of frequency and
   monetary value, printed on one line (prototype: 0.134 and 0.255; Pearson 0.184 with monetary value capped
   at its own 99th percentile), and two sentences, no prompt: Gamma-Gamma assumes spend per purchase is
   independent of how often a customer buys (Fader & Hardie, note 025; Module 4 accepted 0.11 on CDNOW); a
   positive rank correlation means frequent buyers tend to spend more per purchase, so a customer with few
   purchases is likely shrunk toward a population mean that is too high for them. Noted, not fixed here.
   The question "The rank correlation here is about twice the Pearson one: what does a rank correlation see
   that Pearson misses when a few customers spend thousands?" is asked in the debrief (`teach.qmd`), not
   in the lab.

**D3 · One customer, two pipelines.** One sentence before it: "Look at how much of this customer's value
comes from lines your pipeline removes."

1. Customer 12346, at most ten lines shown, then one small before/after table (Module 1's rules against the
   pipeline: frequency, mean repeat spend, expected 26-week revenue; the "before" column uses the Module 1
   rules variant, fitted with the helper in this cell). Under Module 1's rules this customer has frequency 7
   and mean repeat spend £11,066.64, because most of their "purchases" are `TEST001` lines ("This is a test
   product.") in December 2009 and January 2010, plus an order of 74,215 `MEDIUM CERAMIC TOP STORAGE JAR`
   at £1.04 (£77,183.60) on 2011-01-18 that was cancelled the same day (`C541433`). After the pipeline:
   frequency 1, mean repeat spend £142.31; expected 26-week revenue falls from £18,026.6 to £113.1
   (prototype). The largest line in the data, 80,995 `PAPER CRAFT , LITTLE BIRDIE` at £2.08 (£168,469.60,
   customer 16446, 2011-12-09), was cancelled the same day too (`C581484`); it falls in the holdout.
2. The scale for Exercise 5's Predict: the mean and the median customer's expected 26-week revenue under
   the reference pipeline, printed on one line, and `exp_purchases` and `exp_spend` (26 weeks, from the
   reference fits) left in variables for Exercise 5.

### Exercise 5 · The retention-contact list (8 minutes)

The minutes include reading D1–D3. The function itself is short; the one new idea is the business rule.

- **Predict.** A contact costs £10, the margin is 30% and the contact is assumed to lift a customer's
  purchases by 10%. (a) How much expected 26-week revenue must a customer have before a contact pays: about
  £33, about £333, or about £3,333? (b) With the mean and median printed above, what share of the 5,008
  customers clears that bar: under 25%, 25 to 50%, or over 50%? Write both guesses. (Answers: £333.33 =
  10 / (0.10 × 0.30); prototype 44%. The display cell after the checkpoint prints the share.)
- **Function.** `contact_list(exp_purchases: pd.Series, exp_spend: pd.Series, cost: float, margin: float,
  lift: float) -> pd.Series` of booleans on the same index: True when lift × margin × exp_purchases ×
  exp_spend > cost. This is Module 5's break-even rule turned around: contact when the break-even lift,
  cost / (margin × expected revenue), is below the assumed lift.
- **Checkpoint.** Hand cases, including one exactly at the bar (not contacted), built from numbers that are
  exact in binary floating point (for example lift 0.5, margin 0.25, two purchases at £16: gain £4.00,
  so cost £4 is not contacted and cost £3.50 is; 0.1 × 0.3 is not exact, so a hand case built from it
  could land on either side); the count never rises when the cost rises (`checks.monotone` on counts
  over a cost grid); on Online Retail II with 26-week scores from the reference fits at (£10, 0.30, 0.10),
  the count equals the reference (prototype 2,212 of 5,008).
- **Failure messages** (four parts, each ending with `workshop.use_reference(5)`):
  - The customer exactly at the bar contacted (`>=`): "a customer whose expected gain equals the cost gains
    nothing net: contact only when the gain is greater than the cost".
  - Far too many customers (revenue compared with cost, lift or margin left out): "the gain from a contact is
    lift × margin × expected revenue, not the revenue itself".
  - A count or a list of IDs returned: "return one True/False per customer, on the same index as
    `exp_purchases`".
- **Run · the list** (provided display after the checkpoint): the share contacted (the Predict's answer), the
  ten customers with the largest expected gain and the ten closest above the bar, with frequency, recency, T,
  monetary value, expected purchases and expected spend.
- **Explain.** Compare with your guesses. Which customers clear the bar: frequent, recent buyers or big
  one-time spenders? Point to the rows above. Details: a customer with no repeat purchase gets the
  population mean as expected spend (Module 4's weight w is 0 when x = 0), so one big first order does not
  make a big expected spend; frequent, recent buyers have both more expected purchases and a spend estimate
  that leans on their own mean. Whether the contact causes the lift is an experiment's question (Modules 6
  and 11), not this model's.

## Decision · Whom do we contact, and which cleaning choice moves the list? (5 minutes)

In the standard order. **The rule comes first, in Markdown, before any number is printed:** "Contact a
customer when lift × margin × expected 26-week revenue > contact cost; at £10, 30% and 10%, that is expected
26-week revenue above £333." The cost and the lift are assumptions (stated in one cell, with a sentence
saying so); the margin is Module 5's 30%.

Then one provided cell prints:

- **Number:** customers to contact under the reference pipeline, at £10 cost, 30% margin, 10% lift and a
  26-week horizon (prototype: 2,212 of 5,008), with the money it concerns: the total contact cost
  (customers × £10) and the expected extra margin at the assumed lift (Σ lift × margin × expected 26-week
  revenue over the list); then the count at a 5% and a 20% lift (prototype: 1,114 and 3,446); then a table,
  one row per cleaning choice, of customers added and dropped against the reference list, sorted by the
  total moved. Prototype, at a 10% lift: count invoices instead of days 164 (14 added, 150 dropped); Module
  1's rules 51 (48, 3); Module 1's cancellation rule 27; keep non-product lines 20; cap at the 99th
  percentile (£3,663) 20, or 19 if capped for fitting only; keep duplicates 11; cap at the 99.9th
  percentile 2. The cell prints the mode and the sentence "No interval: the fits are point estimates; the
  spread shown is across assumed lifts and cleaning choices."
- **Recommendation:** the learner's sentence. The numbers above suggest its shape: how many customers to
  contact and what it costs; that no cleaning choice moves the list as much as halving or doubling the
  assumed lift (1,098 and 1,234 customers against at most 164, prototype); and that cleaning still decides
  *who* is on the list (a test account and cancelled bulk orders under Module 1's rules). What would change
  it: a measured lift. The uncertainty here comes from the assumed lift and the cleaning choices, not from
  a posterior (the fits are MAP; see the stretch).

The template under the cell (the code fills every number; the values below are the prototype's, to show the
format, except the margin, which the prototype did not report):

```text
Number          : 2,212 of 5,008 customers clear the bar at a 10% lift (1,114 at 5%, 3,446 at 20%).
                  Contact cost £22,120; expected extra margin at a 10% lift £____.
                  Point estimates: no interval. FULL run.
Rule            : contact when 10% lift × 30% margin × expected 26-week revenue > £10 (revenue > £333).
Cleaning        : the choice that moves the list most is counting invoices instead of purchase days,
                  164 customers (14 added, 150 dropped); Module 1's rules move 51.
What would change it: the lift. At 5% the list halves, at 20% it grows by about half; a randomized
                  holdout from the list (Module 6) would measure it.
Your sentence   : ________________________________________________
```

## Stretch (optional)

1. **Missing IDs that cleaning cannot fix.** In the messy log, remove the customer ID from a seeded 10% of
   identified invoices (guest checkout by known customers), rerun the pipeline and the sweep, and compare
   frequency, r/α and the holdout forecast with the truth. Unlike the errors above, this one cannot be
   cleaned away; Online Retail II's 22.5% of lines without an ID may hide some of it.
2. **Left censoring.** On Online Retail II, keep only customers whose first observed purchase is on or after
   2010-04-01 and compare the BG/NBD parameters and the contact list.
3. **A wholesale segment.** Flag customers whose mean purchase day exceeds ten times the median customer's
   and value them with their own mean spend instead of the shrunk Gamma-Gamma estimate; compare with the
   wholesale truth on the messy log.
4. **A posterior for the borderline customers** (adds compute: two MCMC fits with nutpie, estimated at one
   to two minutes on Colab): apply the contact rule to each draw and report, for customers near the bar,
   the probability that contacting pays.

## Known pitfalls

- `invoice_date` has a time: cast to DATE before counting purchase days (Module 1).
- DuckDB `GROUP BY customer_id` keeps the NULL customers as one group, which becomes a "customer" with
  hundreds of purchases; pandas `groupby` drops them silently. Drop them explicitly (`no_customer`).
- Duplicates: PARTITION BY treats NULLs as equal, which is what we want here (prototype: the SQL and pandas
  `duplicated()` flags agree on both logs).
- Matching must use the same price and the opposite quantity; prices are floats read from CSV, so they
  compare exactly only because both lines carry the same string. Do not round one side.
- Clean the whole log, then split at the calibration end.
- The invoice-level RFM breaks `checks.rfm_table`'s "recency = 0 exactly when frequency = 0" rule; do not
  apply it there.
- `mle.gamma_gamma_mle` needs `monetary_value > 0`, repeat buyers only, as PyMC-Marketing. Given a table
  with zero-repeat customers it should raise a message a learner can act on ("pass repeat buyers only:
  `rfm[rfm['frequency'] > 0]`"), not an optimizer error.
- The synthetic retailer is a Pareto/NBD, so the BG/NBD has no true parameters to recover; compare the
  purchase model with the reference pipeline and the realized holdout, the spend model with its truth.
- Hand cases at a threshold (Exercises 3 and 5) need numbers that are exact in binary floating point;
  0.1 × 0.3 × 333.33… is not, so a case "exactly at the bar" built from the lab's own numbers can land on
  either side.
- The truth stores only counts and hashes of flagged positions, so a checkpoint cannot print *which* lines
  differ on the messy log. The toy tables carry the diagnosis: every likely mistake listed under each
  exercise fails a named toy case first.

## APIs verified (and how)

- DuckDB 1.3.2 (installed, equals the pin): `ASOF LEFT JOIN ... ON <equalities> AND c.ts >= p.ts`,
  `regexp_matches`, `starts_with`, `ROW_NUMBER() OVER (PARTITION BY ...)` ran in the prototype on both logs;
  the flags and matches equal a pandas implementation (`duplicated`, `merge_asof(..., by=[customer_id,
  stock_code, price, quantity], direction="backward")`) row for row. DuckDB's guide states that the join
  takes the nearest preceding row and at most one match per left row.
- pymc-marketing 1.2.0 installed source: `clv/distributions.py` `BetaGeoNBD.logp(value, a, b, r, alpha,
  T)`; `clv/models/gamma_gamma.py` `GammaGammaModel.build_model` (the marginal likelihood of z̄ as a
  `pm.Potential`) and `expected_customer_spend` (the weight p x / (p x + q − 1));
  `clv/models/beta_geo.py` `expected_purchases` (FHL 2005 eq. 10, numpy and xarray, no compilation) and
  `expected_probability_alive`. `mktstats.recovery.fit_bgnbd_map(rfm, model_config=None)` and
  `fit_gamma_gamma_map(rfm, model_config="weak")` take a `model_config`; with `Prior("HalfFlat")` for every
  parameter they gave the cross-check numbers above.
- `mktstats.data.load_online_retail_ii(sample=None, seed=0)`: columns and dtypes as in the generator spec
  (read in `src/mktstats/data.py`; the prototype loaded 1,044,848 lines). `data.rfm_summary(...,
  time_unit="D", time_scaler=7)` and `checks.frames_agree(left, right, on, rel=1e-9, abs=1e-9)` as used in
  Module 1.

## References (verified)

- Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
  <https://archive.ics.uci.edu/dataset/502/online+retail+ii>, <https://doi.org/10.24432/C5CG6D>.
  *Verified 2026-10-10.* "A real online retail transaction data set of two years"; 1,067,371 instances;
  CC BY 4.0; donated 2019-09-20; InvoiceNo: "If this code starts with the letter 'c', it indicates a
  cancellation"; StockCode: "A 5-digit integral number uniquely assigned to each distinct product"; unit
  price "in sterling". (The file's columns are named `Invoice` and `Price`; the page says `InvoiceNo` and
  `UnitPrice`.)
- Fader, P. S., & Hardie, B. G. S. (2013, February). *The Gamma-Gamma model of monetary value.*
  <https://www.brucehardie.com/notes/025/> (PDF `gamma_gamma.pdf`). *Verified 2026-10-10.* States the
  assumption that "the distribution of average transaction values across customers is independent of the
  transaction process" (Part D's check).
- Fader, P. S., Hardie, B. G. S., & Lee, K. L. *Implementing the BG/NBD model for customer base analysis in
  Excel.* <https://www.brucehardie.com/notes/004/>. *Verified 2026-10-10* (page fetched); the CDNOW estimates
  the helper's test reproduces are those already cited in `references.qmd` (verified 2026-10-09).
- DuckDB. *AsOf join* (guide). <https://duckdb.org/docs/current/guides/sql_features/asof_join.html>.
  *Verified 2026-10-10.* ASOF LEFT JOIN keeps unmatched left rows with NULLs; each left row matches the
  nearest preceding right row, at most one.
- Olist. *Brazilian E-Commerce Public Dataset by Olist* (Kaggle).
  <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>; metadata read from
  <https://www.kaggle.com/api/v1/datasets/view/olistbr/brazilian-ecommerce>. *Verified 2026-10-10.* License
  CC BY-NC-SA 4.0; "100k orders from 2016 to 2018"; last updated 2021-10-01. Repeat-purchase counts
  above were measured on the files downloaded that day (not used in the lab).
- Kaggle. *Kaggle API documentation.* <https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/README.md>.
  *Verified 2026-10-10.* "First, you will need a Kaggle account"; credentials are set up with a token.

## Prototype evidence (throwaway scripts, laptop, 2026-10-10; not lab results)

Messy log (prototype generator with the rates above): 110,480 lines, 23.0% without an ID; POST on 4.46% of
identified invoices; cancellation lines 2.06% of identified lines; duplicates 1.45%; non-product revenue
2.1% (real 1.76%); extra same-day invoices 7.9%. The reference rules flag exactly the labeled lines, and
the cleaned RFM equals the clean log's for all 5,025 customers and `data.retailer_rfm()` for the 5,000
retail ones. Module 1's rules alone leave 455 customers with a different frequency and 1,481 with a
different monetary value.

Sweep, change against the reference (BG/NBD r, α, a, b; 52-week forecast; Gamma-Gamma p, q, v; population
mean spend):

| Variant | r | α | a | b | forecast | p | q | v | pop. mean |
|---|---|---|---|---|---|---|---|---|---|
| reference (values) | 0.390 | 2.732 | 0.395 | 2.783 | 16,178 | 7.615 | 3.583 | 10.735 | $31.66 |
| keep non-product | +5.7% | +5.4% | −1.7% | −2.3% | +1.7% | −14.9% | +5.5% | +27.6% | +0.9% |
| keep duplicates | 0 | 0 | 0 | 0 | 0 | −0.8% | −1.0% | +1.3% | +1.9% |
| Module 1 cancellation rule | +0.9% | −2.0% | −1.4% | −5.5% | +1.0% | −5.1% | +0.9% | +8.3% | +1.5% |
| Module 1 rules | +6.7% | +3.3% | −3.2% | −8.0% | +2.7% | −19.1% | +5.2% | +38.1% | +4.2% |
| count invoices | −1.9% | −24.6% | +6.1% | −10.6% | +4.8% | −57.4% | +6.9% | +137.8% | −7.6% |

Per-customer 52-week scores against the reference: keep non-product +2.7% total, 363 customers moved by
more than 10%; duplicates +1.4%, 148; Module 1's cancellation rule +2.6%, 292; Module 1's rules +6.7%,
2,610; count invoices −4.0%, 1,504. The reference's total is 4.7% above the true expected total
($495,822), the BG/NBD-on-Pareto/NBD gap.

Capping (Gamma-Gamma MLE; truth p 6, q 4, v 15, mean $30): no cap 7.615, 3.583, 10.735, $31.66; P99.9
($1,395.47) 7.521, 3.604, 10.945, $31.61; P99.5 ($231.71) 6.515, 3.897, 13.783, $30.99; P99 ($149.48)
6.134, 4.058, 15.300, $30.69; P95 ($79.07) 5.138, 4.795, 21.770, $29.47; wholesale dropped (oracle) 6.105,
4.155, 15.687, $30.36. Wholesale 52-week revenue against its truth ($28,413): no cap −29%, P99 values
−90%, P99 for fitting only −33%. 229 calibration purchase days exceed P99, 90 of them wholesale.

## Open items for the Technical Expert

1. **New helper module** `src/mktstats/mle.py` (a new file; nothing existing changes): `bgnbd_mle(rfm) ->
   {r, alpha, a, b}`, `gamma_gamma_mle(rfm) -> {p, q, v}` (repeat buyers only),
   `bgnbd_expected_purchases(params, rfm, t) -> np.ndarray`, `gamma_gamma_expected_spend(params, rfm) ->
   np.ndarray`. Optimize on log parameters (the prototype ran Nelder–Mead, then L-BFGS-B) and raise on
   non-convergence. Tests: note 004 and note 025 within 0.1%; agreement with PyMC-Marketing flat-prior MAP on
   the retailer within 1e-3; the two prediction functions equal `BetaGeoModel.expected_purchases` and
   `GammaGammaModel.expected_customer_spend` at the same parameters to 1e-9; a stub that returns the start
   values fails. Add a checks-can-fail test if any new check helper is written (new file, not `checks.py`).
   Since the notebook's one PyMC-Marketing cross-check now runs on the messy log (Part B), add, if the
   download is acceptable in CI, a slow test (`pytest -m slow`) that the helper agrees with the
   PyMC-Marketing flat-prior MAP on cleaned Online Retail II (prototype: 1.2 × 10⁻⁴ and 1 × 10⁻⁵), so the
   real-data agreement is still checked somewhere.
2. **Size of `messy_lines.csv`** (prototype 7.2 MB; 1.5 MB gzipped). Options: (a) commit the CSV as is;
   (b) commit `messy_lines.csv.gz` written deterministically (gzip `mtime=0`), which needs
   `_write_frames` to support a compression suffix without changing how existing CSVs are written; (c) fewer
   lines per invoice. `pd.read_csv` and `repo_file` read either. Your call; tell the lead.
3. **Tolerances** for `truth_messy.json`, measured over independent seeds of the messy generator (the
   retailer seed varied too): the capped (P99) Gamma-Gamma MLE against p, q, v and the population mean;
   the reference holdout forecast against the realized holdout (no looser than 15%). The exact-recovery
   checkpoint has no tolerance beyond `frames_agree`'s 1e-9. Name each tolerance by how it was measured
   (for example "the largest relative error over 20 generator seeds") and quote that phrase in the
   checkpoint's failure message, so a learner who misses it can tell a near miss from a wrong pipeline.
4. **Real-data reference counts** for Part D and the Decision, re-measured on the sha256-checked parquet:
   the prototype values are listed above; embed what you measure, with the date, as Module 1 does.
5. **Install list**: `nutpie` is needed only for stretch 4; PyMC-Marketing only for the cross-check cell
   (start of Part B).
   `deps` must add `src/mktstats/mle.py`, `src/mktstats/synth/messy.py`, the messy files and
   `truth_messy.json` once they exist (`tests/test_variables.py` fails on deps that match no file).
6. **Release ref**: the extension notebooks install `mktstats` at `repo.ref` (`v2026.10.1`), which has
   neither `mle.py` nor `synth/messy.py`. They need a later ref; raise with the lead before Colab runs.
7. The ASOF join can match one purchase to two cancellations (123 on real data, prototype). If you want
   one-to-one pairing on real data, rank cancellations and purchases per key; keep the learner's SQL
   simple and do it in a provided cell, or leave the approximation stated.

## Pedagogy review (2026-10-10)

Checked against `briefs/_lab-standard.md` §11 before any code exists, with Module 1's brief and lab and
Module 4's brief for comparison. Nothing here has run; the minutes are estimates until a pilot.

**Ruling 1 · Many fits with a maximum-likelihood helper: approved, with conditions** (details under "Model
fits"). Learners already know the estimator: Modules 2 and 4 showed that PyMC-Marketing's flat-prior MAP is
the maximum-likelihood estimate and reproduced the published CDNOW values with it. The comparison across
variants is the lesson, so the many fits are the point, not a lapse. The conditions are about order and
honesty: the helper is introduced before its first use, the one PyMC-Marketing cross-check runs there
(start of Part B, on the table the sweep starts from) with both methods' measured seconds, and the notebook
says every fit is a point estimate.

**Ruling 2 · Five exercises in 40 minutes: not as drafted; yes after the changes below, as an estimate.**
As drafted, Exercise 5's 9 minutes had to hold six provided steps of Part D (a 1–2 minute PyMC-Marketing
cell, a customer walk-through, holdout tracking and an independence check with its own Explain prompt)
before a learner reached the function, and Exercise 2 asked for an ASOF join, new SQL to most analysts,
with no worked example. Minutes are now 8 + 10 + 7 + 7 + 8 = 40, with a running clock and a switch minute
for Exercise 2. If a pilot runs over, cut in this order: D3's line-by-line view of customer 12346 becomes a
folded block; then the room runs `workshop.use_reference(3)` and reads Exercise 3's tables instead of
writing the function.

What changed, and why:

1. **No download before the first checkpoint beyond the messy log.** The opening cell printed Online Retail II
   next to the messy log, which loaded the real log before Exercise 1. The comparison moved to Part D's first
   cell; the opening prints counts and three example rows of the messy log instead.
2. **Exercise 1 (7 → 8 minutes).** One prediction instead of three (the share without a customer ID was
   Module 1's own Predict, and two of the draft's answers lay within 0.6 points of its 2% boundary); the stub
   writes `no_customer` as a pattern, so the learner writes two expressions; failure messages for the four
   likely mistakes, each checked on a toy table with DuckDB 1.3.2; the Explain asks one mechanism question
   (frequency or amount?) with a printed customer as evidence.
3. **A worked ASOF example before Exercise 2 (9 → 10 minutes).** Last-click attribution on four orders, run
   with DuckDB 1.3.2 and its output recorded above. The stub holds the two CTEs, so the one new thing is the
   join and its conditions. The exact-recovery check moved to a labeled cell after the pipeline, because it
   needs the pipeline, which the draft defined after Exercise 2's checkpoint; its failure message routes the
   learner to Exercise 1 or 2 by the label that touches the differing customers. The Explain now has
   evidence on screen (the pairs that straddle the calibration end), and the Predict's answer is printed.
4. **Exercise 3 (7 minutes).** The Predict comes before the sweep's tables, so the guess is made before the
   purchase-model and spend-model changes are visible; the wide table is split into two narrow ones; the
   function returns two columns, not three (`largest_change_pct` moved to the provided display).
5. **Part C opens with the wholesale customers**, so Exercise 4's Predict has a scale ($1,200 against $30 per
   purchase day).
6. **Exercise 4 (8 → 7 minutes).** `calibration_rfm` is provided with the pipeline; the toy (1 to 100 →
   99.01) became a hint, because a hundred amounts cannot also feed the Gamma-Gamma fit inside the same
   function; a `pct=100` check proves the cap is applied to the amounts; failure messages name the cap each
   likely mistake gives. The Explain asks the decision-relevant question (which valuation would you use?)
   and keeps the 95th-percentile point in its details.
7. **Part D in three cells with short summaries.** The PyMC-Marketing cross-check moved to Part B (Ruling
   1); the independence numbers stay, printed with two sentences, and the question becomes a debrief
   question; customer 12346 moved after the fits, because the draft quoted their 26-week revenue before the
   fits that produce it.
8. **Exercise 5 (9 → 8 minutes).** The Predict gives the bar to compute (£333 = £10 / (10% × 30%)) and an
   anchor (the mean and median expected revenue printed in D3); the hand case at the bar uses numbers exact
   in floating point; a display of the top of the list gives the Explain its evidence.
9. **Decision.** The rule is stated in Markdown before any number; the Number adds the money (contact cost
   and expected extra margin at the assumed lift); the cell says why there is no interval; a fill-in template
   follows, with the prototype's values marked as such and the one value the prototype did not report left
   blank.
10. **Terms** the notebook must define at first use are listed with suggested wording; prototype answers
    are kept out of the prompts.

Not changed: the content, the claims, the datasets, the generator, the stretch, the references and the
decision. `modules.m13` in `_variables.yml` still matches the brief (its objectives and outcomes name the
same work; PyMC-Marketing is still fitted once), so it was not edited.
