# Lab brief · Module 5 · From CLV to decisions

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebook | `labs/src/05-clv-decisions.py` → `labs/python/05-clv-decisions.ipynb` |
| Lab slot | 55 min (`modules.m05.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Day | Day 2: briefing before lunch, lab after lunch (lead's schedule); the CFO-memo clinic follows the lab and starts by writing the memo from the provided memo cell |

## Question and decision

**Question.** Given what customers are worth, what should we spend to acquire and to keep them?

**Decision.** Set the maximum customer acquisition cost (CAC) we can afford per acquisition channel, and
decide whether to run the proposed retention campaign.

## What this lab fixes

Module 4 produced CLV with an interval, but a number on its own does not say what to spend.

## Assumptions stated in one cell

Gross margin rate 30% of spend (an assumption, shown as a parameter); monthly discount rate 1%; horizon 36
months for acquisition decisions; retention campaign cost $2 per targeted customer. Learners may change
them; checkpoints use the defaults.

## Data

Synthetic retailer: `mktstats.synth.retailer(seed=<default>)` (transactions, customers with
`acquisition_channel`, truth incl. `pnbd`, `covariates`, `gamma_gamma`). No real dataset in this module:
the decisions need a known answer to score against, and CDNOW has no acquisition channel.

## Model fits (once, provided cells)

1. Transaction model: `BetaGeoModel` with the channel dummies as `purchase_covariate_cols` and
   `dropout_covariate_cols` in `model_config`, `method="mcmc", nuts_sampler="nutpie", progressbar=False`.
   Not the Pareto/NBD that generated the data: its MCMC took 231 s for 2×300 draws on CDNOW on Colab (lead,
   2026-10-09), and the CAC caps need a posterior. Say in the notebook that BG/NBD approximates the
   generator, which is why Exercise 1's tolerance is wider than a recovery check. QUICK: chains=2,
   draws=300, tune=300, 2,000-customer subsample. FULL: chains=2, draws=1000, tune=1000, all customers.
   Estimated time: under a minute FULL plus 35–45 s numba compilation on a fresh runtime (estimate; the
   lead measured BG/NBD 2×500 without covariates at 13.7 s on Colab).
2. `GammaGammaModel` on repeat buyers, MCMC nutpie (seconds; 1.9 s for 2×300 on CDNOW on a laptop).

Then provided: per-customer P(alive) and 36-month CLV draws (Module 4's `discounted_clv`, imported from
`mktstats` or redefined in a provided cell).

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · What can we pay to acquire a customer? | 1 · New-customer CLV by channel | 10 |
| | 2 · CAC caps under a stated risk rule | 6 |
| B · Segments and retention | 3 · Segment value and real differences | 8 |
| | 4 · Retention-campaign break-even | 8 |
| C · Production | 5 · Write scores to the warehouse | 8 |
| | *Run: the numbers for the CFO memo (provided)* | — |
| **Exercises** | | **40** |
| Decision | CAC caps and the retention campaign | 5 |

### Exercise 1 · New-customer CLV by channel (10 minutes)

- **Predict.** Rank the channels by the 36-month value of a new customer, highest first. Is the top channel
  worth less than 1.5 times, 1.5 to 3 times, or more than 3 times the bottom one?
- **Function.** `new_customer_clv(model, gg_idata, channel_rows, months, monthly_rate, margin) ->
  xr.DataArray` with dims `(chain, draw, channel)`: the margin on the acquisition purchase (month 0, not
  discounted) plus Σ_k margin × spend × (expected repeat purchases in month k) / (1 + d)^k, where repeat
  purchases come from differences of `model.expected_purchases_new_customer(data=channel_rows,
  t=k × 30.4375 / 7)` and spend is the population mean spend per draw, p·v / (q − 1). `channel_rows` has one
  row per channel with `customer_id` = channel name and the covariate dummies.
- **Checkpoint.** For each channel, the posterior mean is within a tolerance of
  `mktstats.synth.true_new_customer_clv(truth, channel, months=36, monthly_rate=0.01, margin=0.30)`
  (proposed helper, same convention), the tolerance set from the reference run in QUICK and FULL and
  stated (no looser than 20%, since BG/NBD approximates the Pareto/NBD generator); and the channel with the
  highest true value has the highest posterior mean.
- **Explain.** BTYD models count **repeat** purchases; the acquisition purchase must be added separately.
  The covariates from Module 3 are what make channels differ.

### Exercise 2 · CAC caps under a stated risk rule (6 minutes)

- **Predict.** Will a cap set at the 20th percentile of new-customer CLV be less than 5%, 5 to 15%, or more
  than 15% below a cap set at the mean?
- **Function.** `cac_cap(clv_draws: xr.DataArray, q: float | None = 0.2) -> pd.Series` per channel: the q
  quantile of the posterior (risk-averse), or the posterior mean when `q is None`.
- **Checkpoint.** For q < 0.5 the cap is below the mean for every channel; caps increase with q
  (`checks.monotone` over q = 0.1, 0.2, 0.5); values match the reference.
- **Explain.** The interval covers uncertainty about the *expected* value of a new customer, not the spread
  between customers. A CAC cap only makes sense if the channel's acquisitions are incremental: Day 3 tests
  that.

### Exercise 3 · Segment value and real differences (8 minutes)

- **Predict.** Two segments have overlapping 94% intervals for mean CLV. Can the posterior probability that
  one is more valuable still be above 0.9: yes or no?
- **Function.** `segment_value(clv_draws, segments: pd.Series) -> pd.DataFrame` with, per segment, size,
  total and per-customer mean CLV with 94% HDIs; and `prob_greater(clv_draws, segments, a, b) -> float`, the
  posterior probability that segment a's mean CLV exceeds segment b's. Segments are provided: acquisition
  channel × tenure band (< 6 months, 6–18, > 18).
- **Checkpoint.** Segment totals add up to the base total in every draw (to 1e-6); probabilities lie in
  [0, 1] (`checks.probability`); `prob_greater(a, b) + prob_greater(b, a) == 1` (ties aside).
- **Explain.** Compare segments with the posterior of their difference, not by eyeballing whether
  intervals overlap.

### Exercise 4 · Retention-campaign break-even (8 minutes)

- **Predict.** Will the break-even lift be smaller for customers with high P(alive) or low P(alive)?
- **Function.** `breakeven_lift(cost, margin_per_purchase, expected_purchases) -> np.ndarray`: the
  relative increase in expected purchases over the campaign horizon that a customer needs for the campaign
  to pay back, `cost / (margin_per_purchase × expected_purchases)`; and `campaign_table(...) ->
  pd.DataFrame` by segment with the median break-even lift and the number of customers whose break-even
  lift is below 5%, 10% and 20%.
- **Checkpoint.** Arithmetic on three hand cases; break-even lift decreases as expected purchases increase
  (`checks.monotone`); table counts match the reference.
- **Explain.** The break-even lift is smallest for high-value, likely-alive customers, but those are often
  the customers who would buy anyway. Whether the offer *causes* extra purchases is an uplift question
  (Module 11); Day 3 shows how to measure it.

### Exercise 5 · Write scores to the warehouse (8 minutes)

- **Predict.** If you run the scoring job twice for the same model version, how many rows should the table
  have: one per customer, or two?
- **Function.** `write_scores(con, scores: pd.DataFrame, model_version: str, scored_at: str) -> int`
  writing table `customer_scores` in a DuckDB file `warehouse.duckdb` with exactly:
  `customer_id VARCHAR, p_alive DOUBLE, clv_mean DOUBLE, clv_hdi_low DOUBLE, clv_hdi_high DOUBLE,
  segment VARCHAR, model_version VARCHAR, scored_at DATE`, primary key (customer_id, model_version);
  re-running for the same version replaces rows instead of duplicating them. Returns the row count.
- **Checkpoint.** `DESCRIBE customer_scores` matches the schema; row count equals the number of customers;
  no nulls; re-running leaves the count unchanged; `0 ≤ p_alive ≤ 1`; `clv_hdi_low ≤ clv_mean ≤
  clv_hdi_high`.
- **Explain.** Storing the interval, the model version and the date with every score is what makes a
  scoring job reproducible.

### Run · The numbers for the CFO memo (provided)

A provided cell collects `memo_numbers = {"base_clv": (mean, low, high), "cac_caps": {...},
"retention": {...}, "assumptions": {...}}` from the exercises above and renders a one-page memo template
(markdown) with the numbers filled in by code. The learner does **not** write the memo text in the lab: the
Day 2 clinic starts with 15 minutes of writing it from this cell, then peer review against the Pedagogy
Expert's memo checklist. (Note for the Pedagogy Expert: this moves memo writing from the lab into the
clinic so the lab fits its budget.)

### Decision · CAC caps and the retention campaign (5 minutes)

Provided cell, in the standard order:
- **Number:** per channel, the CAC cap under the mean rule and the 20th-percentile rule (Exercise 2) next
  to current CAC (an assumption table in the notebook), and the posterior probability that new-customer
  margin CLV exceeds current CAC, computed draw by draw; the retention table from Exercise 4.
- **Rule:** "Keep acquiring through a channel while its current CAC is below the cap under the risk rule
  you chose (state it)"; "Run the retention campaign for a segment only if its break-even lift is below
  what an experiment has shown is plausible."
- **Recommendation:** the learner's sentence per decision, with the probability from the Number line and
  what would change it. The retention decision is provisional: no experiment has measured the lift yet
  (Day 3). The CAC caps assume each channel's acquisitions are incremental (also Day 3). The cell prints
  the mode.

## Stretch (optional)

Payback period per channel (months until cumulative discounted margin exceeds CAC), with its interval.

## Known pitfalls

- Forgetting the acquisition purchase (Exercise 1).
- Treating overlapping intervals as "no difference" (Exercise 3).
- CAC caps assume incrementality; say so in the memo.
- DuckDB file paths on Colab are ephemeral; write to the working directory and say it disappears with
  the runtime.

## APIs verified (and how)

Installed source, pymc-marketing 1.2.0, 2026-10-09: `BetaGeoModel.expected_purchases_new_customer(data,
t=)` applies covariates through `_extract_predictive_variables` (FHL 2005 eq. 9); `ParetoNBDModel` and
`BetaGeoModel` accept `purchase_covariate_cols` and `dropout_covariate_cols` in `model_config`;
`GammaGammaModel` posterior has `p`, `q`, `v`. DuckDB `DESCRIBE` and `INSERT OR REPLACE` with a primary key
are standard in DuckDB 1.3.

## Open items for the Technical Expert

- `mktstats.synth.true_new_customer_clv(truth, channel, months, monthly_rate, margin)` with the convention
  above (acquisition purchase at month 0 plus discounted repeat purchases).
- Confirm that `BetaGeoModel` with channel covariates samples within the 4-minute cell target on Colab
  (FULL); if not, reduce draws before reducing customers, and record the setting used.
