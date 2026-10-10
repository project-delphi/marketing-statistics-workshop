# Lab brief · Module 14 · Survival analysis for subscription churn

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab is built: the generator, the notebook and every run are still to come. Numbers marked *prototype* come from a throwaway generator and fits written to test this design (scratchpad, not the repo) on a 10-core Apple Silicon laptop, native environment; they are not run records and not Colab times. Review: Pedagogy Expert. |
| Notebook | `labs/src/14-survival-churn.py` → `labs/python/14-survival-churn.ipynb` |
| Lab slot | 55 min (`modules.m14.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Day | Extensions (`d6`), optional. Builds on Module 4; does not need Module 13. |

## Question and decision

**Question.** How long do subscribers stay, what shortens or lengthens their stay, and what is a change in
churn worth in months and dollars?

**Decision.** Is an onboarding call worth its cost? Call a subscriber who has not completed onboarding
when q × ΔRMST × m > c: q is the share of called subscribers whom the call gets to complete onboarding,
ΔRMST the extra months of subscription within 24 months that onboarding brings (standardized by plan,
with a 95% bootstrap interval), m the monthly margin and c the cost of a call. Say plainly that
onboarding was not randomized, so the estimate is an association until an experiment confirms it.

## What this lab fixes

Module 4's sBG model describes one cohort that everyone joined on the same day and that was observed for
the same number of years. A real subscription program signs people up every day, so on any date most
subscribers are still active and their eventual churn time is unknown (right-censored); and the sBG has
no covariates, so it cannot say which subscribers churn faster or by how much.

## Notation (consistent with Modules 2-4)

- T = a subscriber's tenure in months from the first billing date until cancellation; t = time in months.
  A month is 30.4375 days, PyMC-Marketing's constant (Module 4).
- δ = 1 if the cancellation was observed (`churned`), 0 if the subscriber was still active at the
  observation end (censored). Observed tenure = min(T, time to the observation end).
- S(t) = P(T > t), the survival (retention) curve. h(t) = the hazard: the churn rate per month among
  subscribers still active at t. H(t) = ∫₀ᵗ h(u) du, the cumulative hazard; S(t) = exp(−H(t)).
- Kaplan-Meier: Ŝ(t) = ∏_{t_j ≤ t} (1 − d_j / n_j), over distinct cancellation times t_j, with d_j
  cancellations at t_j and n_j subscribers at risk just before t_j (those with observed tenure ≥ t_j,
  censored ones included).
- Cox model: h(t | z) = h₀(t) exp(γ′z), z the subscriber's covariates. Covariates are z and their
  coefficients γ, as in Module 3; α and β stay the sBG's Beta parameters (Module 4) and θ its churn
  probability.
  Hazard ratio HR_k = exp(γ_k): the factor by which covariate k multiplies the hazard, at every t if
  hazards are proportional. HR 0.67 means "churns at 67% of the rate", not "67% as likely to churn".
- RMST(t\*) = ∫₀^{t\*} S(t) dt, the restricted mean survival time: the expected months subscribed within
  the first t\* months. Discrete counterpart (Exercise 2): expected paid periods in the first H periods =
  Σ_{t=0}^{H−1} S(t).
- sBG (Module 4): S(t) = B(α, β + t) / B(α, β), B the beta function.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic subscription program | committed `data/synthetic/subscriptions_subscribers.csv`, read with `mktstats.data.repo_file("synthetic/subscriptions_subscribers.csv")`; truth in `data/synthetic/truth_subscriptions.json` (same loader); generator `mktstats.synth.subscriptions.subscriptions(seed=2034)` (new file, not exported from `synth/__init__.py`) | Exercises 1, 3, 4, 5, decision, stretch |
| Fader-Hardie retention counts (real) | `mktstats.data.load_sbg_retention()` (Module 4's loader): survivors of 1,000 for `regular` and `highend`, years 0-12 | Exercise 2 |

### The synthetic subscription program (generator spec)

**Story.** In January 2024 the retailer launched a monthly replenishment subscription in three plans
(basic, standard, premium). Each subscriber starts with a 14-day free trial, during which they complete
onboarding or not; the first billing date ends the trial and is **time zero**. Anyone who cancelled during
the trial never became a subscriber and is not in the table. Some subscribers signed up through an
introductory offer (the first three paid months discounted). Sign-ups ran from 2024-01-01 to 2025-12-31
and grew over that time; the data stop on **2026-06-30** (the observation end).

Why time zero is the first billing date: onboarding status must be known when the clock starts. If the
clock started at sign-up and onboarding could be completed later, every subscriber who cancelled before
completing it would count as "not onboarded", and onboarding would look protective for that reason alone
(immortal-time bias; Catalogue of Bias, Giobbie-Hurder et al. 2013). Say this in one sentence in the
notebook and in the briefing.

**Data-generating process** (`subscriptions(seed=2034, ...)`; all keyword arguments, defaults below):

1. `plan` ~ Categorical(basic 0.50, standard 0.30, premium 0.20).
2. Hidden motivation u_i = `hidden_confounding` × N(0, 1). Default `hidden_confounding = 0.0`: there is
   no hidden variable, and the Cox model below is correctly specified for plan and onboarding.
3. `onboarded` ~ Bernoulli(expit(logit(p_plan) + u_i)) with p_plan = 0.30 basic, 0.55 standard,
   0.80 premium. Onboarding depends on plan (and on u_i when the stretch turns it on), nothing else.
   This is the confounding by an observed covariate: premium subscribers onboard more and churn less.
4. `intro_offer` ~ Bernoulli(0.30), independent of everything else.
5. Linear predictor η_i = −0.25 · standard − 0.60 · premium − 0.40 · onboarded − u_i
   (true HRs 0.779, 0.549 and 0.670; basic, not onboarded, no offer is the reference).
6. Cumulative hazard, t in months, with a Weibull baseline H₀(t) = (t / 14)^0.8 (shape 0.8: the hazard
   falls with tenure; scale 14 months) and an intro-offer effect that stops at month 3:
   H_i(t) = exp(η_i) · [exp(0.70 · intro_offer) · H₀(min(t, 3)) + max(H₀(t) − H₀(3), 0)].
   So intro-offer subscribers churn at exp(0.70) = 2.014 times the rate in months 0-3 and at the same
   rate afterwards. This is the proportional-hazards violation the lab finds.
7. Lifetime T_i = H_i⁻¹(E_i), E_i ~ Exponential(1), inverted in closed form on each piece
   (H₀⁻¹(v) = 14 · v^(1/0.8)). Compute each piece only where it applies (the prototype's `np.where`
   evaluated both and warned about powers of negative numbers).
8. `start_date`: day offset in [0, 730] from 2024-01-01 with a sign-up rate that rises linearly to twice
   its starting value (`signup_growth = 2.0`): offset = ⌊731 · (√(1 + 3v) − 1)⌋, v ~ Uniform(0, 1).
9. Cancellation day = start + ⌈30.4375 · T_i⌉ days (at least 1). `cancel_date` is that date if it is on or
   before 2026-06-30, else empty (censored). Build dates only for observed cancellations: with
   `hidden_confounding = 1.0` the prototype overflowed pandas' date range on a 700-year lifetime.
10. Sort by `start_date`; `customer_id` = 1..n in that order. n = 8,000.

**Table `subscribers`** → `subscriptions_subscribers.csv`: `customer_id`, `start_date`, `cancel_date`
(empty if active at the observation end), `plan` (basic / standard / premium), `onboarded` (0/1),
`intro_offer` (0/1), and two truth columns that real data never has: `true_tenure_months` (the
uncensored lifetime, whole days / 30.4375) and `true_rmst24_gain` (this subscriber's RMST(24) with
`onboarded` set to 1 minus with it set to 0, given plan, offer and u_i; like Module 11's per-row
`true_cate`). The lab derives `tenure_months` and `churned` from the dates in a provided cell.

**`truth_subscriptions.json`** (written by `scripts/make_synthetic.py` through `EXTENSIONS`):
`seed`, `n`, the sign-up window, `signup_growth`, `observation_end`, `days_per_month`, `time_zero`
(one sentence); `model` (hazard formula as text, Weibull shape and scale, reference group, log hazard
ratios and hazard ratios, `intro_offer` {landmark_months 3, hr_before 2.01375, hr_after 1.0},
`hidden_confounding`); `covariates` (plan, onboarding-by-plan and offer probabilities, the independence
statement); `survival` at t = 3, 6, 12, 18, 24 months: overall, by plan, by observed onboarding group and
standardized (onboarding set to 1 or 0 with the population plan mix); `rmst` (horizon 24; RMST by plan and
onboarding; gain by plan; the target population "subscribers who did not complete onboarding", its plan
weights, the gain for it computed from the model and the mean of `true_rmst24_gain` over those rows; the
same gain discounted at 0.8% a month); `decision` (margin, call cost, assumed and grid values of q, the
rule as text, true value per call and true break-even q); `tolerances` (coverage measured over seeds by
the Technical Expert, as `TOLERANCES` does for truth.json).

**Truth values** (analytic, from the parameters above; independent of the seed):

| Quantity | Value |
|---|---|
| S(t) overall at 3, 6, 12, 18, 24 months | 0.7665, 0.6595, 0.5107, 0.4084, 0.3333 |
| S(t) basic / standard / premium at 6, 12, 24 | 0.591, 0.424, 0.240 / 0.688, 0.544, 0.363 / 0.788, 0.678, 0.522 |
| S(12) onboarded vs not, as observed (plan mix differs) | 0.611 vs 0.420 |
| Hazard ratios: standard, premium, onboarded | 0.779, 0.549, 0.670 |
| Intro-offer HR, months 0-3 / after 3 | 2.014 / 1.000 |
| RMST(24), not onboarded vs onboarded: basic, standard, premium | 10.41 vs 13.40, 12.30 vs 15.12, 14.79 vs 17.23 months |
| Plan mix of subscribers who did not onboard | basic 0.667, standard 0.257, premium 0.076 |
| **ΔRMST(24) for them (the decision's number)** | **2.907 months** (2.614 discounted at 0.8% a month) |
| Unadjusted ΔRMST(24), onboarded vs not as observed | 4.061 months (what a naive comparison estimates) |
| True value per call at q = 0.20, 0.35, 0.50 (m = $12, c = $15) | $6.98, $12.21, $17.44; break-even q = 0.430 |

The truth is computed in continuous time; rounding cancellations up to whole days lengthens each lifetime
by less than a day. On 1,000,000 prototype rows this raised S(1 month) by under 0.002 and left
S(3)-S(24) within sampling error. State it in the generator's docstring.

**Prototype on seed 2034** (scratch generator, so the Technical Expert's numbers will differ slightly):
4,427 of 8,000 cancellations observed (55.3%); longest follow-up 29.9 months; about 495 subscribers
observed for at least 24 months (mean over 40 seeds, 456 to 526). The plan × onboarding table: 31% of
basic, 55% of standard and 81% of premium subscribers onboarded. "Over 40 seeds" below means seeds
1000-1039 of the prototype at these settings (n = 8,000, sign-ups doubling); the bootstrap used B = 100.

## Model fits

Nothing is sampled; every fit takes well under a second.

| Fit | Where | Settings | Prototype time (laptop) |
|---|---|---|---|
| Kaplan-Meier (lifelines `KaplanMeierFitter`) | Exercise 1 checkpoint, provided plots, Exercise 5 | `alpha=0.05` (95% pointwise intervals, exponential Greenwood) | 0.004 s for 8,000 rows |
| Cox, full period, all covariates (the lab's one Cox fit) | Exercise 3; reused by the PH test | `CoxPHFitter(alpha=0.05)`, Efron ties (default) | 0.18 s |
| Cox, onboarding only | Exercise 3 checkpoint (unadjusted) | same | < 0.1 s |
| Two landmark Cox fits | Exercise 4 | same | < 0.5 s together |
| Stratified bootstrap of Exercise 5's function | Exercise 5 checkpoint and decision, computed once | B = 200 (FULL), 100 (QUICK); resample within plan × onboarding; `numpy.random.default_rng(14)` | 4.6-5.4 s (B = 200), 2.3 s (B = 100) |

QUICK changes only B. The data are the same in both modes, so every checkpoint except Exercise 5's sees
identical numbers; the header says so, and that QUICK's interval can differ in the second decimal.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Censoring and the Kaplan-Meier curve | *Run: tenure and churn from dates; two dashboard retention rates (provided)* | — |
| | 1 · Write the Kaplan-Meier estimator | 10 |
| | 2 · Where Kaplan-Meier stops: expected years on the Fader-Hardie data | 6 |
| B · Who churns faster: the Cox model | *Run: curves by plan and onboarding, log-rank tests (provided)* | — |
| | 3 · Hazard ratios adjusted for plan | 8 |
| | *Run: proportional-hazards test and monthly churn by offer (provided)* | — |
| | 4 · An effect that fades: hazard ratios before and after month 3 | 8 |
| C · From hazard ratios to months and dollars | 5 · Months gained from onboarding | 8 |
| **Exercises** | | **40** |
| Decision | Is an onboarding call worth its cost? | 5 |

Shared names: `subs` (subscribers, one row each, with `tenure_months` and `churned`), `truth` (the
truth file), `retention` (Fader-Hardie, as in Module 4), `kmf`, `cph`. Constants in one visible cell,
each with its reason: `OBS_END = "2026-06-30"`, `DAYS_PER_MONTH = 30.4375`, `ALPHA = 0.05`,
`LANDMARK = 3.0` (the offer's discount lasts three months), `HORIZON = 24` (Exercise 2 explains why not
longer), `MARGIN = 12.0` (dollars a month), `CALL_COST = 15.0`, `Q_GRID = (0.20, 0.35, 0.50)`,
`BOOT_SEED = 14`, `B = 100 if QUICK else 200`. The margin, cost and q are scenario assumptions; say so.

### Run · Tenure, churn and two dashboard numbers (provided)

Reads the CSV, computes `tenure_months` = (cancel date, or 2026-06-30 if none, minus start date) in days
/ 30.4375 and `churned` = 1 if `cancel_date` is present. Defines right-censoring in one sentence ("we know
the subscriber stayed at least this long, not when they will leave"). Prints two 12-month retention rates
a dashboard might show: **A**, the share of all subscribers who did not cancel within 12 months of
starting (prototype 52.8%), and **B**, the same share among subscribers whose 12-month status is known,
that is, who cancelled or were followed for 12 months (prototype 42.5%).

### Exercise 1 · Write the Kaplan-Meier estimator (10 minutes)

- **Predict.** The true 12-month retention is in neither dashboard number. Is it above A, between A and
  B, or below B? If between, nearer which?
- **Function.** `kaplan_meier(duration, event) -> pd.Series`: index = the distinct times with at least
  one cancellation, ascending; values Ŝ(t) just after each; name `"survival"`. A subscriber censored at
  t counts as at risk at t. Durations are in whole days, so there are ties (prototype: 899 distinct
  tenures and 664 distinct cancellation times among 8,000 rows); group them.
- **Checkpoint (first in the lab; no fit before it).**
  1. Toy: `kaplan_meier([2, 3, 3, 5, 6, 6, 8, 9], [1, 1, 0, 1, 0, 1, 1, 0])` equals {2: 0.875, 3: 0.75,
     5: 0.6, 6: 0.45, 8: 0.225} within 1e-12, and 9 is not in the index (checked against lifelines,
     prototype). Failure hints: a censored subscriber leaves the risk set *after* time t; ties at one time
     are one factor.
  2. On `subs`: equals lifelines' `KaplanMeierFitter().fit(...).survival_function_` at every
     cancellation time within 1e-9 (prototype largest gap 5.8e-15).
  3. Truth: the true S(t) at 6, 12, 18 and 24 months (`truth.survival.overall`) lies inside lifelines'
     95% pointwise confidence interval (exponential Greenwood, R's "log-log"; read with `.asof(t)`) for at
     least 3 of 4 times. Prototype seed 2034: 4 of 4 (S(12) truth 0.511, KM 0.505, interval [0.494,
     0.516]). Over 40 seeds: 4 of 4 in 34, 3 in 5, 1 in 1; S(12) alone inside in 40.
  4. Both dashboard numbers lie outside the 95% interval at 12 months (prototype: 0.528 and 0.425;
     40 of 40 seeds).
- **Explain.** Compare with your guess. Which subscribers does A count as "retained" who may still
  cancel before month 12? Which does B drop, and why does that make B too low? Point to the start dates of
  the subscribers who are censored before month 12.
- `<details>`: A treats recent starters as retained although they have not had 12 months to cancel; B
  keeps the recent starters who already cancelled but drops those who did not, so it over-counts churn.
  Kaplan-Meier uses each subscriber for exactly as long as they were observed: each month's churn rate
  comes only from subscribers still at risk then. It assumes censoring is unrelated to churn risk; here
  censoring comes only from the observation end, and the plan mix does not change with start date. It would
  fail if, say, the newest cohorts came from a promotion with different churn.

### Exercise 2 · Where Kaplan-Meier stops: expected years on the Fader-Hardie data (6 minutes)

*Provided before it:* Module 4's encoding of the counts (lost in year t → tenure t, event 1; still active
after year 7 → tenure 7, event 0); `kmf` fitted on years 0-7; `sbg_survival(alpha, beta, t)` =
exp(betaln(α, β + t) − betaln(α, β)) with the published Regular estimates α = 0.704, β = 1.182 (Module 4
reproduced them; so did our own maximum-likelihood fit, 0.7041 and 1.1820, on 2026-10-10). Define
"expected paid years in the first H years" for an annual subscription paid at each renewal:
Σ_{t=0}^{H−1} S(t). It is the discrete RMST and, times the annual margin, the undiscounted contractual CLV.

- **Predict.** Kaplan-Meier on years 0-7 says 241 of 1,000 Regular customers are active at year 7. If
  you carry that flat to year 12, will expected paid years in the first 12 be above or below the actual?
  By under 1%, 1-5% or over 5%?
- **Function.** `expected_periods(survival, horizon) -> float`: the sum of `survival[0]` to
  `survival[horizon − 1]`; raise `ValueError` with a sentence saying how far the curve is known when it
  has fewer than `horizon` values.
- **Checkpoint (structural, on real data).**
  1. `expected_periods([1, 0.5, 0.25], 3) == 1.75`; with horizon 4 it raises `ValueError`.
  2. Kaplan-Meier's own curve (`kmf.survival_function_`, years 0-7, 8 values) with horizon 12 raises.
  3. Actual Regular, years 0-12 from Table 1: 4.406 (exact).
  4. sBG with the published α, β: within 1% of the actual (prototype 4.410, +0.1%).
  5. Kaplan-Meier carried flat (`kmf.predict(range(13))`, which lifelines returns without a warning):
     overstates the actual by more than 3% (prototype 4.563, +3.6%).
- **Explain.** Kaplan-Meier matches years 0-7 exactly (prototype largest gap 3e-16). Why can it say
  nothing about year 8? What does carrying it flat assume about churn after year 7? Point to year 12:
  241 flat, 173 actual, 170 from the sBG.
- `<details>`: Kaplan-Meier is a description of the observed churn rates and has no rate for a year it
  has not seen; carrying it flat assumes nobody churns after the last observation. The sBG extrapolates
  because it assumes a form (a Beta mix of constant churn probabilities), which fits these data (Fader &
  Hardie 2007); in High End the same comparison gives actual 7.166, sBG 7.156 and flat 7.381. That is why
  the decision below uses a 24-month horizon: the subscription data follow subscribers for at most 30
  months, and beyond that we would need a parametric model and its assumption.

### Run · Curves by plan and onboarding, log-rank tests (provided)

Defines the hazard and the log-rank test (a test of whether two or more groups have the same churn rate
at every tenure; most powerful when one group's hazard is a constant multiple of the other's). Plots
Kaplan-Meier curves by plan with 95% bands and by onboarding, with one sentence before each plot saying
what to look at; prints `multivariate_logrank_test` across plans and `logrank_test` for onboarding,
and the plan × onboarding table (share onboarded per plan). Prototype: onboarded 60.3% vs 41.4%
retained at 12 months, log-rank χ² = 345 (p ≈ 5e-77); plans χ² = 415.

### Exercise 3 · Hazard ratios adjusted for plan (8 minutes)

- **Predict.** Premium subscribers onboard far more often than basic ones (table above) and also churn
  less. Once plan is in the model, will onboarding's hazard ratio be closer to 1 than without plan,
  further from 1, or the same?
- **Function.** `cox_hazard_ratios(subs, covariates, alpha=0.05) -> pd.DataFrame`: fit
  `CoxPHFitter(alpha=alpha)` on `subs[covariates + ["tenure_months", "churned"]]` with
  `duration_col="tenure_months"`, `event_col="churned"`; return one row per covariate with `hr`, `lower`,
  `upper` (the hazard ratio and its (1 − alpha) confidence interval). Dummy columns `plan_standard` and
  `plan_premium` (basic is the reference) are made in a provided cell.
- **Checkpoint.**
  1. With `["plan_standard", "plan_premium", "onboarded", "intro_offer"]`: the true HR of onboarding
     (0.670) inside its 95% confidence interval, and at least 2 of the 3 true HRs (0.779, 0.549, 0.670)
     inside theirs. Prototype seed 2034: 0.740 [0.691, 0.793], 0.508 [0.462, 0.559], 0.680 [0.637,
     0.725], all three inside. Over 40 seeds: all three inside in 34; the rule above held in 39.
     The estimates are unbiased even though `intro_offer` breaks proportional hazards (40-seed means
     0.782, 0.549, 0.669).
  2. With `["onboarded"]` alone: the truth 0.670 is outside the 95% interval and the HR is lower than the
     adjusted one (prototype 0.568 [0.535, 0.603]; outside in 40 of 40 seeds, and in all 70 seeds of two
     other settings tried). This also catches a function that ignores `covariates`.
  3. Failure hints: returning `coef` instead of `exp(coef)`; plan as text instead of dummies; another
     plan as reference.
- **Explain.** Compare with your guess. Which way did plan bias the unadjusted ratio, and why? Point to the
  share onboarded per plan and the plan hazard ratios.
- `<details>`: Plan affects both onboarding and churn, so the unadjusted comparison credits onboarding
  with part of the premium plan's lower churn. Holding plan fixed removes that part. This works here
  because plan is the only thing that drives onboarding in this simulation; real onboarding also depends on
  motivation, which is not in the data (stretch). The `intro_offer` ratio of about 1.36 is an average over
  tenure of an effect that is not constant (Exercise 4); hazard ratios are not probabilities of churning.

### Run · Proportional-hazards test and monthly churn by offer (provided)

`proportional_hazard_test(cph, data, time_transform="rank").summary`: a test, from scaled Schoenfeld
residuals (Grambsch & Therneau 1994), of whether each covariate's effect changes with tenure. Prototype:
`intro_offer` χ² = 99 (p < 1e-20); onboarding p = 0.44, plans 0.54 and 0.82. Over 40 seeds the offer was
flagged at p < 0.01 every time, and another covariate once (three tests at 0.01 give about a 3% chance of
a false flag per seed). Then a plot of the monthly churn rate
(cancellations in month k ÷ subscribers active at the start of month k), months 1-24, for `intro_offer`
1 and 0: look at where the two lines meet. Use the test, not `check_assumptions(show_plots=True)`, which
bootstraps lowess curves with an unseeded generator and prints advice.

### Exercise 4 · An effect that fades: hazard ratios before and after month 3 (8 minutes)

- **Predict.** The full-period model gave `intro_offer` a hazard ratio of about 1.36. In months 0-3,
  will it be higher, lower or about the same? After month 3?
- **Function.** `landmark_hazard_ratios(subs, covariates, landmark=3.0) -> pd.DataFrame` with columns
  `period` ("before", "after"), `covariate`, `hr`, `lower`, `upper`, calling Exercise 3's function twice:
  **before** = every subscriber, follow-up stopped at the landmark (tenure = min(tenure, 3), churned only
  if it happened by month 3); **after** = subscribers still active after month 3, with the clock restarted
  (tenure − 3). This is a landmark analysis: fix a time, keep those still at risk, start again
  (Giobbie-Hurder et al. 2013).
- **Checkpoint.** On the full covariate list: the true offer HR 2.014 inside the "before" 95% interval
  and 1.0 inside the "after" one; the true onboarding HR 0.670 inside both. Prototype seed 2034: offer
  2.061 [1.878, 2.261] and 0.955 [0.873, 1.045]; onboarding 0.680 [0.614, 0.753] and 0.681 [0.626, 0.741];
  the PH test then flags nothing at 0.01 in either period. Over 40 seeds the four held in 95%, 100%, 93%
  and 95% of seeds, all four together in 33 and at least 3 of 4 in 40, so the checkpoint requires at
  least 3 of the 4. Structural checks: "before" has all rows; "after" has only rows with tenure > 3 and
  no tenure ≤ 0.
- **Explain.** Compare with your guess. Why was 1.36 a misleading summary of the offer? What should a
  report say instead? Point to the two periods' intervals and the churn-rate plot.
- `<details>`: The full-period ratio averages a doubled churn rate in the first three months with no
  difference afterwards, weighted by when cancellations happen. Splitting at the landmark recovers both.
  Deal-seekers leave when the discount ends; those who stay are like everyone else. Alternatives:
  `strata=["intro_offer"]` gives the offer its own baseline and keeps the other ratios (prototype
  0.742, 0.511, 0.680) but reports no offer effect; a time-varying coefficient
  (`CoxTimeVaryingFitter` on episode data) is the general tool. Caution: here subscribers differ only in
  observed ways, so the survivors at month 3 are comparable; in real data the survivors are a selected
  group and an after-landmark ratio carries that selection (Hernán 2010).

### Exercise 5 · Months gained from onboarding (8 minutes)

*Provided before it:* the unadjusted RMST(24) difference between onboarded and not, from two lifelines
Kaplan-Meier fits and `restricted_mean_survival_time(kmf, t=24)` (prototype 4.06 months); the
definition of RMST with a sentence on why it suits a decision (it is in months, needs no proportional
hazards and multiplies by a monthly margin; Royston & Parmar 2013); and the target population: the
subscribers who did not onboard, whose plan mix (`weights`) is computed in the cell.

- **Predict.** Unadjusted, onboarded subscribers stay about 4.1 more months within the first 24. Standardized
  by plan, will the gain be larger, about the same or smaller? By how much: under 0.5 month, 0.5-2, over 2?
- **Function.** `rmst_gain(subs, horizon, weights) -> float`: for each plan p, RMST(horizon) of the
  onboarded minus that of the not onboarded, each from its own `KaplanMeierFitter` and
  `restricted_mean_survival_time`; return Σ_p weights[p] × difference. `weights` is a dict or Series over
  plans summing to 1 (assert it).
- **Checkpoint.**
  1. With weights {"basic": 1.0}: equals the difference of the two lifelines RMSTs for basic computed
     directly (exact).
  2. Truth: the target-population gain 2.907 months (`truth.rmst.gain_target`) lies inside the 95%
     bootstrap percentile interval of `rmst_gain` (stratified by plan × onboarding, B and seed as above).
     Prototype seed 2034: estimate 2.672; intervals [2.022, 3.218] (B = 200) and [2.283, 3.122]
     (B = 100); over 40 seeds (B = 100) the truth was inside 38 times, and the 40-seed mean estimate was
     2.897 (truth 2.907).
  3. The unadjusted 4.06 lies outside that interval (40 of 40 seeds).
- **Explain.** Compare with your guess. Which plan's subscribers dominate the target group, and why does
  standardizing to their plan mix change the answer? Point to the weights and the per-plan RMSTs.
- `<details>`: The unadjusted difference compares onboarded subscribers (one third premium) with the
  rest (two thirds basic); its own estimand is 4.06 months, of which only 2.91 is onboarding. Comparing
  within plan and weighting by the plan mix of the people a call would reach answers the decision's
  question. Stratifying fully adjusts here because onboarding depends on plan alone. Lifelines'
  `restricted_mean_survival_time(..., return_variance=True)` returns the variance of min(T, 24) across
  subscribers (prototype SD 9.45 months), not the uncertainty of the estimate; that is why the interval
  comes from the bootstrap.

### Decision · Is an onboarding call worth its cost? (5 minutes)

Provided cell, in the standard order, printing the mode:

- **Number:** ΔRMST(24) for subscribers who did not onboard, mean and 95% bootstrap percentile interval
  (Exercise 5), beside the unadjusted 4.06; value per call q × ΔRMST × m with its interval for q in
  `Q_GRID`; break-even q\* = c / (ΔRMST × m) with its interval; the share of bootstrap resamples in which
  the call pays at each q (label it as that, not as a probability). Prototype: value at q = 0.35 is $11.22
  [$8.49, $13.51] against $15 (unadjusted: $17.03); q\* = 0.468 [0.388, 0.618] (true 0.430; unadjusted
  0.308); the call pays in 0%, 0% and 70-80% of resamples at q = 0.20, 0.35, 0.50.
- **Rule** (stated before the numbers): call when q × ΔRMST × m > c, with m = $12 a month and c = $15 a
  call; q is not known.
- **Recommendation** (the learner's sentence, with what would change it): for example, do not roll out
  calls on this evidence: at q = 35% they return about $11 for each $15; they pay only if a call gets more
  than about 40-60% of reached subscribers to onboard. The unadjusted comparison would have funded them.
  Before spending, run a randomized pilot (Module 6) that measures q and 6-month retention, because
  onboarding was not randomized and motivation may drive both (Module 15 estimates such effects from
  observational data). A cheaper nudge changes the answer: at an illustrative $3 a nudge, the
  break-even q is about 9%.
  ΔRMST is undiscounted and in months subscribed; discounting at Module 4's 0.8% a month lowers the true
  gain by 10% (2.907 → 2.614).

## Stretch (optional) · A confounder that is not in the data

Regenerate live with `subscriptions(seed=2034, hidden_confounding=0.5)` (under a second): motivation
now raises the odds of onboarding and lowers churn. Rerun `cox_hazard_ratios` and `rmst_gain` and compare
with the truth, the mean of `true_rmst24_gain` among subscribers who did not onboard. Prototype: Cox HR
0.584 [0.547, 0.623] against the conditional truth 0.670; adjusted gain 3.80 months against a true 2.74.
Adjusting for plan no longer removes the bias, and nothing in the data says so. Two shorter options:
`rmst_gain` per plan (one weight of 1.0) with bootstrap intervals, to see that the data cannot separate
the plans (truth 2.99, 2.82, 2.44 months; prototype estimates 2.69, 2.54, 2.99); and the decision with
Module 4's 0.8% monthly discount.

## Expected run time (estimates; no run record yet)

| | Estimate | Basis |
|---|---|---|
| Install on Colab | under 1 minute | Colab already has autograd 1.9.1, formulaic 1.2.2 and interface_meta 2.0.1 (`environment/colab-constraints.txt`); only `lifelines` and `autograd-gamma`, both pure-Python wheels, are added. Target ≤ 3 min. |
| Longest cell | the bootstrap, under 30 s on Colab | 4.6-5.4 s for B = 200 on the laptop prototype; Colab's 2 vCPUs assumed 2-5 times slower. Target ≤ 4 min. |
| Total compute | about 1 minute FULL, less QUICK | Every other step took under 0.5 s on the laptop. Target ≤ 15 min. |

These are estimates from prototype code on a laptop. The Technical Expert replaces them with measured
Docker and Colab times from run records.

## Known pitfalls

- **Time zero and immortal time.** Groups must be defined by what is known at time zero (here the first
  billing date). A status reached later (a second purchase, a support ticket) needs a landmark or a
  time-varying covariate.
- `KaplanMeierFitter.predict(t)` beyond the last observed time returns the last value, silently
  (`estimate.asof(times)` in `fitters/__init__.py`): that is "carried flat" (Exercise 2).
- Lifelines' Kaplan-Meier intervals are pointwise (one time at a time), not a band for the whole curve.
- `restricted_mean_survival_time(..., return_variance=True)` is the spread of min(T, t\*), not the
  estimate's standard error (Exercise 5).
- `CoxPHFitter` centers covariates (h(t | z) = h₀(t) exp((z − z̄)′γ); its docstring writes x and β), so
  `baseline_survival_` is the curve at the mean covariates, not the reference group; use
  `predict_survival_function` for a profile. Hazard ratios are unaffected.
- Summary column names include the level (`exp(coef) lower 95%`), so with another `alpha` they change;
  `confidence_intervals_` is on the log scale.
- With 8,000 rows the PH test can flag small deviations (lifelines docs); judge with the churn-rate
  plot and the size of the change, not the p-value alone.
- Ties: whole-day tenures create many ties; Kaplan-Meier handles them exactly, Cox uses Efron's
  approximation (default).
- `check_assumptions` compares `round(p, 2)` with the threshold and prints; use `proportional_hazard_test`
  in a checkpoint.

## APIs verified (and how)

Installed source, lifelines 0.30.3 (`.venv/lib/python3.13/site-packages/lifelines/`), read 2026-10-10,
and run in the prototype:

- `fitters/kaplan_meier_fitter.py`: `KaplanMeierFitter(alpha=0.05, label=None)`;
  `fit(durations, event_observed=None, timeline=None, entry=None, label=None, alpha=None,
  ci_labels=None, weights=None, fit_options=None)`; `survival_function_` (index starts at 0 and includes
  censoring times); `confidence_interval_` columns `KM_estimate_lower_0.95`, `KM_estimate_upper_0.95`,
  from `_bounds`, the exponential Greenwood formula; `median_survival_time_`; `survival_function_at_times`.
- `fitters/__init__.py`: `UnivariateFitter.predict(times, interpolate=False)` = `estimate.asof(times)`.
- `fitters/coxph_fitter.py`: `CoxPHFitter(baseline_estimation_method="breslow", penalizer=0.0,
  strata=None, ..., alpha=0.05)` (alpha through `BaseFitter`); `fit(df, duration_col, event_col,
  show_progress, initial_point, strata, weights_col, cluster_col, robust, batch_mode, timeline, formula,
  entry_col, fit_options)`; `summary` columns `coef, exp(coef), se(coef), coef lower 95%, coef upper 95%,
  exp(coef) lower 95%, exp(coef) upper 95%, cmp to, z, p, -log2(p)`; `hazard_ratios_`;
  `confidence_intervals_` (`95% lower-bound`, `95% upper-bound`, coefficient scale).
- `fitters/mixins.py`: `check_assumptions(training_df, advice=True, show_plots=False,
  p_value_threshold=0.01, plot_n_bootstraps=15, columns=None, raise_on_fail=False)`; scaled Schoenfeld
  residuals; tests with time transforms "rank" and "km".
- `statistics.py`: `logrank_test(durations_A, durations_B, event_observed_A, event_observed_B, t_0=-1,
  ...)`, `multivariate_logrank_test(event_durations, groups, event_observed, ...)`,
  `proportional_hazard_test(fitted_cox_model, training_df, time_transform="rank", ...)`, each returning a
  `StatisticalResult` with `.summary` (`test_statistic`, `p`, `-log2(p)`), `.p_value`, `.test_statistic`.
- `utils/__init__.py`: `restricted_mean_survival_time(model_or_survival_function, t=np.inf,
  return_variance=False)`; exact step-function integral for a fitted `KaplanMeierFitter` (toy check:
  6.1 by hand and by lifelines); `return_variance` = E[min(T, t)²] − RMST², the spread of the restricted
  lifetime.
- PyPI JSON for 0.30.3 (read 2026-10-10): MIT, `requires_python >=3.11`, one `py3-none-any` wheel,
  uploaded 2026-03-05. Dependencies (installed METADATA): numpy, scipy, pandas >= 2.1, < 3.0, matplotlib,
  autograd, autograd-gamma, formulaic.

## Open items for the Technical Expert

1. Write `src/mktstats/synth/subscriptions.py` to the spec above (seeded `make_rng`, `SynthResult` with
   table `subscribers`, the truth block, `sig()` rounding) and register `"subscriptions"` in
   `EXTENSIONS` in `scripts/make_synthetic.py`. Then rerun the multi-seed coverage of every checkpoint
   (30 or more seeds) and record it under `tolerances`; replace the prototype numbers in this brief and in
   the notebook with yours, or tell me where they differ.
2. Add the generator, `_common.py`, the CSV, `truth_subscriptions.json` and any new helper file to
   `modules.m14.deps` (the stretch imports the generator; importing `mktstats.synth.subscriptions` also
   runs `synth/__init__.py`).
3. Checks on confidence intervals do not exist in `mktstats.checks` (its interval checks take draws). Put
   `ci_contains(truth, lower, upper, level, name)` and a k-of-K version, and the stratified
   `bootstrap_interval(fn, data, by, B, seed, prob)`, in a new file (for example
   `src/mktstats/survival.py`), each with a test that feeds it a broken input.
4. A small helper or provided cell for the monthly churn-rate plot by `intro_offer`.
5. `scripts/gen_tables.py` lists "Module 13 and its lab" under "Before you start" on Module 14's page,
   because it takes the previous module number. For optional modules that is wrong: Module 14 needs
   Module 4, not 13. Consider using `modules.mNN.prerequisites` for modules with `optional: true`.
6. Confirm the checkpoint tolerances pass on the committed seed in both QUICK and FULL and that every
   checkpoint fails on its stub.

## References (verified)

- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of
  the American Statistical Association*, 53(282), 457-481. <https://doi.org/10.1080/01621459.1958.10501452>\
  Crossref record read 2026-10-10; the publisher page refused automated access (403), so the text was
  not read. Cited as the origin of the product-limit estimator; the formula the lab uses is the one in the
  lifelines documentation below.
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society,
  Series B*, 34(2), 187-220 (Crossref lists 187-202, the paper without the discussion).
  <https://doi.org/10.1111/j.2517-6161.1972.tb00899.x>\
  *Verified (2026-10-10)*: Crossref record, and the abstract on the EPA HERO record
  <https://hero.epa.gov/reference/8785>: the hazard is a function of the explanatory variables and
  regression coefficients multiplied by an arbitrary, unknown function of time, with inference from a
  conditional likelihood.
- Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted
  residuals. *Biometrika*, 81(3), 515-526. <https://doi.org/10.1093/biomet/81.3.515>\
  *Verified (2026-10-10)*: Crossref record and the abstract at
  <https://academic.oup.com/biomet/article-abstract/81/3/515/257037>: a time-varying coefficient (their
  example: a treatment effect that weakens over time) shows up when a residual plot is smoothed, and
  several proportional-hazards tests are weighted least-squares lines on that plot. Basis of
  `proportional_hazard_test`.
- Royston, P., & Parmar, M. K. B. (2013). Restricted mean survival time: an alternative to the hazard
  ratio for the design and analysis of randomized trials with a time-to-event outcome. *BMC Medical
  Research Methodology*, 13, 152. <https://doi.org/10.1186/1471-2288-13-152>\
  *Verified (2026-10-10)* at <https://pmc.ncbi.nlm.nih.gov/articles/PMC3922847/>: the restricted mean is
  "a measure of average survival from time 0 to a specified time point" (the area under the survival
  curve), valid whatever the distribution of the event times, so it needs no proportional hazards.
  (Exercise 5, decision.)
- Hernán, M. A. (2010). The hazards of hazard ratios. *Epidemiology*, 21(1), 13-15.
  <https://doi.org/10.1097/EDE.0b013e3181c1ea43>\
  *Verified (2026-10-10)* at <https://pmc.ncbi.nlm.nih.gov/articles/PMC3653612/>: a hazard ratio can
  change during follow-up, "period-specific HRs have a built-in selection bias", and adjusted survival
  curves are a clearer summary. (Exercises 3-5.)
- Giobbie-Hurder, A., Gelber, R. D., & Regan, M. M. (2013). Challenges of guarantee-time bias. *Journal of
  Clinical Oncology*, 31(23), 2963-2969. <https://doi.org/10.1200/JCO.2013.49.5283>\
  *Verified (2026-10-10)* at <https://pmc.ncbi.nlm.nih.gov/articles/PMC3732313/>: the landmark analysis
  "selects a fixed time during follow-up as the landmark"; guarantee-time bias includes immortal-time
  bias. (Exercise 4, time zero.)
- Lee, H., & Nunan, D. (2020). Immortal time bias. In *Catalogue of Bias*.
  <https://catalogofbias.org/biases/immortal-time-bias/>\
  *Verified (2026-10-10)*: assigning groups with information from after time zero distorts the
  comparison; "assignment and time-zero must be aligned". (Data design.)
- Fader, P. S., & Hardie, B. G. S. (2007). How to project customer retention. *Journal of Interactive
  Marketing*, 21(1), 76-90. <https://doi.org/10.1002/dir.20074>\
  Already in `references.qmd` (*Verified 2026-10-09*). Table 1 is the data of Exercise 2. Our
  maximum-likelihood fit on years 0-7 (2026-10-10, scipy) gives α = 0.7041, β = 1.1820 (Regular) and
  0.6681, 3.8061 (High End), the published values.
- Davidson-Pilon, C. (2019). lifelines: survival analysis in Python. *Journal of Open Source Software*,
  4(40), 1317. <https://doi.org/10.21105/joss.01317>\
  *Verified (2026-10-10)* at <https://joss.theoj.org/papers/10.21105/joss.01317>.
- lifelines documentation, version 0.30.3 checked against the installed source:
  "Introduction to survival analysis" <https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html>
  (ignoring right-censored subjects is "a common mistake"; S(t) and the hazard defined);
  "Survival analysis with lifelines" <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html>
  (the product-limit formula; Kaplan-Meier intervals use the exponential Greenwood formula; the log-rank
  test); "Testing the proportional hazard assumptions"
  <https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html>
  (scaled Schoenfeld residuals; stratification, binning, functional form and time-varying covariates as
  remedies; with a large sample "even very small violations" are flagged).\
  *Verified (2026-10-10).*
