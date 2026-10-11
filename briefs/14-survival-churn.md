# Lab brief · Module 14 · Survival analysis for subscription churn

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Reviewed by the Pedagogy Expert on 2026-10-10 against `briefs/_lab-standard.md` (changes and reasons under "Pedagogy review" at the end). **Built and run** by the Technical Expert on 2026-10-10: worked FULL runs on macOS and in Docker are recorded (`runs/2026-10-10-macos-arm64-python-14-survival-churn.json`, `runs/2026-10-10-docker-arm64-python-14-survival-churn.json`); not yet run on Colab, which needs a release tag that contains the lab. Numbers marked *build* are the notebook's printed output on the committed data (seed 2034), re-read by the Academic Director from a worked run on 2026-10-10 (FULL unless QUICK is named) at the content and deps hashes of the run records; seed-coverage figures come from `tolerances` in `data/synthetic/truth_subscriptions.json`. Numbers still marked *prototype* come from the throwaway generator written to test the design (scratchpad, laptop) and were not re-measured by the build. "Build reconciliation" at the end lists what changed. |
| Notebook | `labs/src/14-survival-churn.py` → `labs/python/14-survival-churn.ipynb` |
| Lab slot | 55 min (`modules.m14.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (10 + 7 + 7 + 8 + 8; limit 40) + decision 5 + slack 5 = 55. Each exercise's minutes include reading the provided cells immediately before it and after its checkpoint. Minutes are estimates until a pilot. |
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

**Build on seed 2034** (the committed data; the build's generator reproduces the prototype's draws):
4,427 of 8,000 cancellations observed (55.3%); longest follow-up 29.9 months; 504 subscribers observed
for at least 24 months (`sample.n_followed_24_months`; the prototype's 40-seed mean was about 495). The
plan × onboarding table: 30.7% of 3,977 basic, 55.1% of 2,417 standard and 81.4% of 1,606 premium
subscribers onboarded. "Over seeds" below means the build's `tolerances` block: seeds 1000-1039 (the
prototype's seeds) and 3000-3039 (an independent check), 80 in all, n = 8,000, sign-ups doubling,
reference solutions, B = 100 and 200 for the bootstrap.

## Model fits

Nothing is sampled; every fit takes well under a second.

| Fit | Where | Settings | Prototype time (laptop) |
|---|---|---|---|
| Kaplan-Meier (lifelines `KaplanMeierFitter`) | Exercise 1 checkpoint (`kmf_subs`), provided plots, Exercise 2 (`kmf_regular`), Exercise 5 | `alpha=0.05` (95% pointwise intervals, exponential Greenwood) | 0.004 s for 8,000 rows |
| Cox, full period, all covariates | Exercise 3 checkpoint, inside the learner's `cox_hazard_ratios`; fitted once more as `cph` in the provided PH-test cell, because the function returns a table, not the fitted model | `CoxPHFitter(alpha=0.05)`, Efron ties (default) | 0.18 s |
| Cox, onboarding only | Exercise 3 checkpoint (unadjusted) | same | < 0.1 s |
| Two landmark Cox fits | Exercise 4 checkpoint: `cox_hazard_ratios` on each part returned by the learner's `landmark_split` | same | < 0.5 s together |
| Stratified bootstrap of Exercise 5's function | Exercise 5 checkpoint and decision, computed once | B = 200 (FULL), 100 (QUICK); resample within plan × onboarding; `mktstats.survival.bootstrap_interval(..., seed=14)` | 4.6-5.4 s (B = 200), 2.3 s (B = 100). *Build*, run records: the Checkpoint 5 cell was the slowest cell, 4.81 s on macOS and 5.94 s in Docker (`--cpus 2`), FULL |

QUICK changes only B. The data are the same in both modes, so every checkpoint except Exercise 5's sees
identical numbers; the header says so, and that QUICK's interval can differ in the second decimal.

## Terms the notebook defines, where they are first used

This lab uses about twenty terms, most of them new to this audience, more than any core lab. The notebook
spreads them out: each is defined in one plain sentence in the provided cell or exercise that first uses it,
never in an opening glossary, and the briefing has defined them first (Pedagogy review, "What the briefing
carries"). The Notation above is for the Technical Expert. Suggested wording:

| Where | Term | One sentence for the notebook |
|---|---|---|
| Part A, first cell | right-censored | We know the subscriber stayed at least this long, not when they will leave. |
| Part A, first cell | survival (retention) curve S(t) | The share of subscribers still subscribed t months after their first billing date. |
| Exercise 1 | at risk at t | Still subscribed and still observed just before t; a subscriber censored at t counts as at risk at t. |
| Exercise 1 | Kaplan-Meier estimator | Ŝ(t) is the product, over the times up to t with at least one cancellation, of the share of those at risk who did not cancel. |
| After Exercise 1 | 95% pointwise interval | An interval for S(t) at one t at a time, not a band that covers the whole curve with 95% probability. |
| Before Exercise 2 | expected paid periods in the first H | S(0) + S(1) + … + S(H − 1): how many of the first H periods an average customer pays for, the area under the survival curve. |
| Exercise 2 | carried flat | Using the last known S(t) for every later t, which assumes nobody leaves after the data end. |
| Part B, first cell | time zero | When each subscriber's clock starts, here the first billing date. A group must be defined by what is known then: a status reached later makes its group look better only because its members stayed long enough to reach it (immortal-time bias). |
| Part B, first cell | hazard h(t) | Among subscribers still active at t, the rate at which they cancel, per month. |
| Exercise 3 | Cox proportional-hazards model | A regression for the hazard, h(t \| z) = h₀(t) exp(γ′z), that leaves the baseline hazard h₀(t) unspecified. |
| Exercise 3 | hazard ratio | The factor by which a covariate multiplies the hazard: 0.67 means "cancels at 67% of the rate", not "67% as likely to cancel". |
| Exercise 3 | confounder; adjusted | A variable that affects both who onboards and who churns (here plan); adjusting for it compares subscribers on the same plan. |
| Exercise 3 | proportional hazards | The assumption, in the Cox model's name, that each hazard ratio is the same at every tenure. |
| Before Exercise 4 | the PH test | A test that asks, covariate by covariate, whether its hazard ratio changes with tenure; a small p-value says it does. |
| Exercise 4 | landmark analysis | Pick a time (the landmark), keep those still at risk then, and start the clock again. |
| Before Exercise 5 | RMST(t\*) | The expected months subscribed within the first t\* months: the area under S(t) up to t\*. |
| Before Exercise 5 | target population | The subscribers a call would reach: those who did not complete onboarding. |
| Exercise 5 | standardized by plan | Compare within each plan, then add up the differences weighted by the target population's plan mix. |
| Exercise 5 | 95% bootstrap percentile interval | Resample subscribers with replacement within each plan × onboarding group, recompute, and take the 2.5th and 97.5th percentiles. |
| Decision | break-even q\* | The share of called subscribers who must complete onboarding for a call to pay: c ÷ (ΔRMST × m). |

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Censoring and the Kaplan-Meier curve | *Run: tenure and churn from dates; two dashboard retention rates (provided)* | — |
| | 1 · Write the Kaplan-Meier estimator | 10 |
| | *Run: your curve against the truth and the dashboards (provided, labeled check)* | — |
| | *Run: the Fader-Hardie curves, years 0-7 (provided)* | — |
| | 2 · Where Kaplan-Meier stops: expected years on the Fader-Hardie data | 7 |
| | *Run: years 8-12, Kaplan-Meier carried flat against the sBG and the actual (provided)* | — |
| B · Who churns faster: the Cox model | *Run: time zero; curves by plan and onboarding; who onboards, by plan (provided)* | — |
| | 3 · Hazard ratios adjusted for plan | 7 |
| | *Run: proportional-hazards test (provided)* | — |
| | 4 · An effect that fades: hazard ratios before and after month 3 | 8 |
| | *Run: monthly churn rate by offer (provided)* | — |
| C · From hazard ratios to months and dollars | *Run: RMST, the unadjusted gain and the target population (provided)* | — |
| | 5 · Months gained from onboarding | 8 |
| | *Run: the gain plan by plan (provided)* | — |
| **Exercises** | | **40** |
| Decision | Is an onboarding call worth its cost? | 5 |

Minute marks for the room (after 5 minutes to open and install): Exercise 1 ends at minute 15, Exercise 2
at 22, Exercise 3 at 29, Exercise 4 at 37, Exercise 5 at 45; Decision 45-50; slack 50-55.

Shared names: `subs` (subscribers, one row each, with `tenure_months` and `churned`), `truth` (the
truth file), `retention` (Fader-Hardie, as in Module 4), `kmf_subs` (lifelines on `subs`, Exercise 1's
check), `kmf_regular` (lifelines on the Fader-Hardie Regular encoding, Exercise 2), `cph` (the provided
PH-test fit). Two different curves both called `kmf` would let a cell run out of order use the wrong one.
Constants in one visible cell, each with its reason: `OBS_END = "2026-06-30"`, `DAYS_PER_MONTH = 30.4375`,
`ALPHA = 0.05`, `LANDMARK = 3.0` (the offer's discount lasts three months), `HORIZON = 24` (Exercise 2
explains why not longer), `MARGIN = 12.0` (dollars a month), `CALL_COST = 15.0`,
`Q_GRID = (0.20, 0.35, 0.50)`, `BOOT_SEED = 14`, `B = 100 if QUICK else 200`,
`COVARIATES = ["plan_standard", "plan_premium", "onboarded", "intro_offer"]` (Exercises 3 and 4, so
nobody retypes it). The margin, cost and q are scenario assumptions; say so.

Every Predict ends with "Write your guess in the cell below." Every plot with two groups tells them apart
by line style or marker as well as color.

### Run · Tenure, churn and two dashboard numbers (provided)

Reads the CSV, computes `tenure_months` = (cancel date, or 2026-06-30 if none, minus start date) in days
/ 30.4375 and `churned` = 1 if `cancel_date` is present. Defines right-censoring in one sentence ("we know
the subscriber stayed at least this long, not when they will leave"). Prints two 12-month retention rates
a dashboard might show: **A**, the share of all subscribers who did not cancel within 12 months of
starting (build 52.8%), and **B**, the same share among subscribers whose 12-month status is known,
that is, who cancelled within 12 months or were observed for more than 12 months (build 42.5%, 6,570
of 8,000 subscribers).
Nothing in this cell shows the truth: Exercise 1's Predict asks about it.

### Exercise 1 · Write the Kaplan-Meier estimator (10 minutes)

- **Predict.** Neither dashboard number is the true 12-month retention. Where is the truth: above A,
  between A and B but nearer A, between them but nearer B, or below B? Write your guess in the cell
  below. (Build: A 52.8%, B 42.5%, truth 51.1%: between, nearer A, about 3.5 points above the
  midpoint.)
- **Function.** `kaplan_meier(duration, event) -> pd.Series`: index = the distinct times with at least
  one cancellation, ascending; values Ŝ(t) just after each; name `"survival"`. A subscriber censored at
  t counts as at risk at t. Durations are in whole days, so there are ties (build: 899 distinct
  tenures and 664 distinct cancellation times among 8,000 rows); group them. `duration` and `event` are
  array-likes of the same length (lists, NumPy arrays or Series); `event` is 1 for a cancellation, 0 for
  censored.
- **Scaffolding in the task cell.** The toy data of the checkpoint, worked by hand for its first two
  times: t = 2: 8 at risk, 1 cancels, Ŝ = 7/8 = 0.875; t = 3: 7 at risk (the subscriber censored at 3
  still counts), 1 cancels, Ŝ = 0.875 × 6/7 = 0.75. Hints in the stub: put the two inputs in one
  DataFrame; group by duration to get, at each time, the cancellations d_j (the sum of `event`) and the
  number of rows; the number at risk n_j is the number of rows at that time or later (a reversed
  cumulative sum); take the cumulative product of 1 − d_j / n_j; keep the times with d_j > 0. The
  reference solution is about six lines.
- **Checkpoint (first in the lab; no fit before it).** It checks the learner's function only.
  1. Toy: `kaplan_meier([2, 3, 3, 5, 6, 6, 8, 9], [1, 1, 0, 1, 0, 1, 1, 0])` equals {2: 0.875, 3: 0.75,
     5: 0.6, 6: 0.45, 8: 0.225} within 1e-12, and 9 is not in the index (checked against lifelines,
     prototype).
  2. Ties: `kaplan_meier([1, 1, 2], [1, 1, 0])` equals {1: 1/3} within 1e-12, and 2 is not in the index
     (by hand: 3 at risk at t = 1, 2 cancel, 1 − 2/3). The first toy never has two cancellations at one
     time, so it cannot catch a tie mistake; this one does, and `subs` has many ties.
  3. On `subs`: equals lifelines' `KaplanMeierFitter().fit(...).survival_function_` (`kmf_subs`) at every
     cancellation time within 1e-9 (build largest gap 6e-15).
- **Failure messages** (the four parts of `_lab-standard.md` §5, each ending with
  `workshop.use_reference(1)`). The wrong answers on the toys were computed by the Pedagogy Expert on
  2026-10-10.
  - Toy {2: 0.875, 3: 0.729, 5: 0.583, 6: 0.389, 8: 0.194}: "a subscriber censored at t is still at risk
    at t. n_j counts everyone whose duration is t_j or longer, censored or not; they leave the risk set
    after t_j."
  - Toy 0.857 at t = 2 (and 0 at t = 8; −1 on the tie toy): "subscribers who cancel at t_j are at risk at
    t_j: count duration ≥ t_j, not > t_j."
  - Tie toy 0.667: "two cancellations at t = 1 are d_j = 2 in one factor, 1 − 2/3; you counted one per
    distinct time. Sum `event` at each time." Tie toy 0.444: "you applied 1 − 1/3 twice with the same
    n_j; use one factor per time, 1 − d_j / n_j."
  - A censoring-only time (9 in the toy, 2 in the tie toy) or time 0 in the index: "keep only times with at
    least one cancellation; a censoring-only time lowers n_j for later times but adds no factor."
  - Values that rise: "you returned the share cancelled, 1 − Ŝ(t); return Ŝ(t)."
  - Toys pass but `subs` differs from lifelines (the message gives the largest gap and its time): "pass
    `subs["tenure_months"]` and `subs["churned"]` unrounded, and sort the index ascending."

*Provided after it · Your curve against the truth and the dashboards* (`with
workshop.checkpoint(label="km-vs-truth")`; read inside Exercise 1's minutes, because its Explain uses it).
The first draft's checkpoint parts 3 and 4 move here unchanged, on `kmf_subs`, which the learner's curve
now equals to 1e-9:

  3. Truth: the true S(t) at 6, 12, 18 and 24 months (`truth.survival.overall`) lies inside lifelines'
     95% pointwise confidence interval (exponential Greenwood, R's "log-log"; read with `.asof(t)`) for at
     least 3 of 4 times. Build, seed 2034: 4 of 4 (S(12) truth 0.511, KM 0.505, interval [0.494,
     0.516]). Over seeds: at least 3 of 4 in 78 of 80 (39 in each set), all 4 in 68 (34 and 34); S(12)
     alone inside in 79 (40 and 39).
  4. Both dashboard numbers lie outside the 95% interval at 12 months (build: 0.528 and 0.425; 80 of
     80 seeds).

These compare the data with the truth, so once Checkpoint 1 passes they cannot fail because of the
learner's code; their message says so ("this compares the data with the truth, not your code; if it fails,
tell the instructor"). The cell prints a short table (6, 12, 18 and 24 months: Kaplan-Meier, 95% pointwise
interval, truth) with A and B beside the 12-month row; the number of subscribers censored before month 12
and the earliest start date among them (by design they all started within the 12 months before the
observation end, because censoring comes only from the observation end); and a plot of Kaplan-Meier with
its 95% pointwise band, the true S(t) dashed, and A and B as labeled markers at 12 months. Sentence before
the plot: "Look at where A and B sit against the band at 12 months."

- **Explain.** Compare with your guess. Which subscribers does A count as "retained" who may still
  cancel before month 12? Which does B drop, and why does that make B too low? Point to the earliest
  start date of the subscribers censored before month 12, printed above.
- `<details>`: A treats recent starters as retained although they have not had 12 months to cancel; B
  keeps the recent starters who already cancelled but drops those who did not, so it over-counts churn.
  Kaplan-Meier uses each subscriber for exactly as long as they were observed: each month's churn rate
  comes only from subscribers still at risk then. It assumes censoring is unrelated to churn risk; here
  censoring comes only from the observation end, and the plan mix does not change with start date. It would
  fail if, say, the newest cohorts came from a promotion with different churn.

### Exercise 2 · Where Kaplan-Meier stops: expected years on the Fader-Hardie data (7 minutes)

*Provided before it* (read inside the 7 minutes, so it is short): two sentences recalling Module 4's data
(survivors of 1,000 customers in each of two segments, years 0-12; Fader and Hardie fit years 0-7);
Module 4's encoding of the counts (lost in year t → tenure t, event 1; still active after year 7 →
tenure 7, event 0); `kmf_regular` fitted on years 0-7; `sbg_survival(alpha, beta, t)` =
exp(betaln(α, β + t) − betaln(α, β)) with the published Regular estimates α = 0.704, β = 1.182 (Module 4
reproduced them; so did our own maximum-likelihood fit, 0.7041 and 1.1820, on 2026-10-10). Define
"expected paid years in the first H years" for an annual subscription paid at each renewal:
Σ_{t=0}^{H−1} S(t). It is the discrete RMST and, times the annual margin, the undiscounted contractual CLV.
The cell prints Regular's years 0-7 only (counts and Ŝ(t), which agree): the Predict asks about the years
after 7, so they are shown after the checkpoint. It builds four NumPy arrays for the checkpoint:
`km_years` (`kmf_regular.survival_function_`, 8 values, years 0-7), `km_flat`
(`kmf_regular.predict(range(13))`, 13 values), `sbg_years` (13 values) and `actual_years` (Table 1
÷ 1,000, 13 values). Plain arrays keep pandas index lookups out of the learner's function.

- **Predict.** From the actual counts, a Regular customer pays for 4.41 of the first 12 years on
  average. Kaplan-Meier knows years 0-7 and ends at 241 of 1,000 at year 7. Carry that value flat to
  year 12: will its expected paid years in the first 12 be above or below 4.41? By under 1%, 1 to 5%, or
  over 5%? Write your guess in the cell below. (Answer: above, by 3.6%.)
- **Function.** `expected_periods(survival, horizon) -> float`: `survival` is a 1-D list or array
  S(0), S(1), … with S(0) = 1; return the sum of `survival[0]` to `survival[horizon − 1]`; raise
  `ValueError` with a sentence saying how far the curve is known when it has fewer than `horizon` values.
  The reference is four lines.
- **Checkpoint (structural, on real data).**
  1. `expected_periods([1, 0.5, 0.25], 3) == 1.75` and `expected_periods([1, 0.5, 0.25], 2) == 1.5`;
     with horizon 4 it raises `ValueError`.
  2. Kaplan-Meier's own curve (`km_years`, years 0-7, 8 values) with horizon 12 raises.
  3. Actual Regular (`actual_years`, Table 1), horizon 12: 4.406 (exact).
  4. sBG with the published α, β: within 1% of the actual (build 4.410, +0.1%).
  5. Kaplan-Meier carried flat (`km_flat`; lifelines' `predict` returns it without a warning):
     overstates the actual by more than 3% (build 4.563, +3.6%).

  Rechecked by the Pedagogy Expert on 2026-10-10 with lifelines 0.30.3 on `data/real/sbg_retention.csv`:
  Regular 4.406 actual, 4.410 sBG, 4.563 flat; High End 7.166, 7.156, 7.381; Regular at year 12: 241 flat,
  173 actual, 170.1 sBG.
- **Failure messages** (each ending with `workshop.use_reference(2)`):
  - 0.75 on the toy: "you started at survival[1]. Everyone pays the first period, so S(0) = 1 counts: sum
    survival[0] to survival[horizon − 1]."
  - 1.75 with horizon 2: "the function ignores `horizon`; sum only the first `horizon` values."
  - No error with horizon 4 (1.75 returned): "slicing past the end of a list or array returns what is
    there without complaint (`s[:4]` of three values is three values); that is how a curve gets used past
    its data. Check `len(survival) < horizon` and raise `ValueError`."
  - `IndexError` or `KeyError` instead of `ValueError`: "check the length and raise `ValueError` yourself,
    before summing."
  - Not 4.406 on the actual curve: "the sum of S(0) to S(11) of the actual curve is 4.406; sum 12 values
    starting at year 0."

*Provided after it · Years 8-12* (read inside Exercise 2's minutes): a table of survivors of 1,000 for
years 8-12 (Kaplan-Meier flat, sBG, actual) and the three totals for H = 12 with their difference from the
actual in percent; a plot of the three curves over years 0-12, Kaplan-Meier solid to year 7 and dashed
after it, with a vertical line between years 7 and 8. Sentence before the plot: "Look at what each curve
does after year 7, the last year either method saw."

- **Explain.** Compare with your guess. Kaplan-Meier matches years 0-7 exactly (prototype largest gap
  3e-16). What does carrying it flat assume about churn after year 7, and what lets the sBG go further?
  Point to year 12 in the table (241 flat, 173 actual, 170 from the sBG).
- `<details>`: Kaplan-Meier is a description of the observed churn rates and has no rate for a year it
  has not seen; carrying it flat assumes nobody churns after the last observation. The sBG extrapolates
  because it assumes a form (a Beta mix of constant churn probabilities), which fits these data (Fader &
  Hardie 2007); in High End the same comparison gives actual 7.166, sBG 7.156 and flat 7.381. That is why
  the decision below uses a 24-month horizon: the subscription data follow subscribers for at most 30
  months, and beyond that we would need a parametric model and its assumption.

### Run · Time zero, curves by plan and onboarding, who onboards (provided)

Opens Part B with the one sentence on time zero and immortal-time bias (Data, "Why time zero is the first
billing date"). Part B is where the lab first compares groups, so the sentence goes here rather than in
Part A, where the learner's attention is on censoring. Defines the hazard. Plots Kaplan-Meier curves by
plan with 95% bands and by onboarding, with one sentence before each plot saying
what to look at; prints the plan × onboarding table (share onboarded per plan), which Exercise 3's Predict
and Explain use. Build: onboarded 60.3% vs 41.4% retained at 12 months. Also makes the dummy columns
`plan_standard` and `plan_premium` (basic is the reference).

*Moved to the briefing (Pedagogy review, item 4):* the log-rank tests, kept here for the module page: the
log-rank test (a test of whether two or more groups have the same churn rate at every tenure; most
powerful when one group's hazard is a constant multiple of the other's), `multivariate_logrank_test`
across plans and `logrank_test` for onboarding. On the committed data (computed by the Academic
Director with lifelines 0.30.3 on 2026-10-10 for the module page; the lab does not run it): log-rank
χ² = 345.2 (p ≈ 5e-77) for onboarding; plans χ² = 415.4 on 2 degrees of freedom (p ≈ 6e-91). The same
as the prototype's.

### Exercise 3 · Hazard ratios adjusted for plan (7 minutes)

- **Predict.** Premium subscribers onboard far more often than basic ones (table above) and also churn
  less. Once plan is in the model, will onboarding's hazard ratio be closer to 1 than without plan,
  further from 1, or the same? Write your guess in the cell below.
- **Function.** `cox_hazard_ratios(subs, covariates, alpha=0.05) -> pd.DataFrame`: fit
  `CoxPHFitter(alpha=alpha)` on `subs[covariates + ["tenure_months", "churned"]]` with
  `duration_col="tenure_months"`, `event_col="churned"`; return one row per covariate with `hr`, `lower`,
  `upper` (the hazard ratio and its (1 − alpha) confidence interval). Dummy columns `plan_standard` and
  `plan_premium` (basic is the reference) are made in a provided cell.
- **Hints in the stub** (the likely time sink is the API, not the idea). `CoxPHFitter` treats every column
  of the frame it is given as a covariate, so pass only the covariates and the two outcome columns
  (all of `subs` fails on the date and plan text columns). `cph.hazard_ratios_` holds the hazard ratios;
  `np.exp(cph.confidence_intervals_)` gives the interval as two columns, lower then upper, whose names
  change with `alpha` (`"95% lower-bound"`), so take them by position. Index the result by covariate, in
  the order given. The reference is about five lines. (Attribute names checked by the Pedagogy Expert in
  lifelines 0.30.3 on 2026-10-10: with `alpha=0.1` the columns are `"90% lower-bound"` and
  `"90% upper-bound"`.)
- **Checkpoint.** First the structure: index equal to `covariates` in order; columns `hr`, `lower`,
  `upper`; 0 < lower < hr < upper in every row. Then:
  1. With `COVARIATES` (`["plan_standard", "plan_premium", "onboarded", "intro_offer"]`): the true HR
     of onboarding (0.670) inside its 95% confidence interval, and at least 2 of the 3 true HRs (0.779,
     0.549, 0.670) inside theirs. Build, seed 2034: 0.740 [0.691, 0.793], 0.508 [0.462, 0.559],
     0.680 [0.637, 0.725], all three inside. Over seeds: all three inside in 66 of 80 (34 and 32); the
     rule above held in 73 of 80, about 91%: 39 on the prototype's seeds 1000-1039 (as the prototype
     found) but 34 on the independent set 3000-3039, so "39 of 40" overstated it. It rests on one 95%
     interval (onboarding's), so about 1 seed in 10 fails it: roughly the nominal 5% and some chance.
     In the set 3000-3039 its five failures (3000, 3028, 3034, 3035, 3037) were the seeds on which
     Checkpoint 5 also failed, where the onboarding effect came out stronger than the truth. The
     estimates are unbiased even though `intro_offer` breaks proportional hazards (means over seeds
     1000-1039: 0.782, 0.549, 0.669; 3000-3039: 0.779, 0.549, 0.662).
  2. With `["onboarded"]` alone: the truth 0.670 is outside the 95% interval and the HR is lower than the
     adjusted one (build 0.568 [0.535, 0.603]; outside in 80 of 80 seeds, and in all 70 prototype seeds of
     two other settings tried). This also catches a function that ignores `covariates`.
- **Failure messages** (each ending with `workshop.use_reference(3)`):
  - A hazard ratio at or below 0, or about −0.39 for onboarding (ln 0.68): "you returned the coefficient
    γ, which is on the log scale; the hazard ratio is exp(γ), `cph.hazard_ratios_`."
  - Bounds negative or not around `hr`: "`confidence_intervals_` is on the log scale; exponentiate it."
  - With `["onboarded"]` alone the ratio equals the adjusted one: "your function ignores `covariates`; fit
    on `subs[covariates + ["tenure_months", "churned"]]`."
  - Structure right but a true ratio outside its interval: "use the dummy columns `plan_standard` and
    `plan_premium` (basic is the reference), not a text `plan` column or dummies with another plan as
    the reference."
  - Passing all of `subs` stops inside lifelines, before the checkpoint's messages. In the build it
    raises `TypeError: Cannot cast DatetimeArray to dtype float64` (the date columns come first;
    reproduced by the Academic Director with lifelines 0.30.3 on the committed data, 2026-10-10). The
    Pedagogy Expert's earlier `ValueError` ("could not convert string to float: 'basic'") is what a
    frame without the date columns gives. The stub's hint and `teach.qmd` cover it; `teach.qmd` should
    quote the `TypeError`.
- **Explain.** Compare with your guess. Which way did plan bias the unadjusted ratio, and why? Point to the
  share onboarded per plan and the plan hazard ratios.
- `<details>`: Plan affects both onboarding and churn, so the unadjusted comparison credits onboarding
  with part of the premium plan's lower churn. Holding plan fixed removes that part. This works here
  because plan is the only thing that drives onboarding in this simulation; real onboarding also depends on
  motivation, which is not in the data (stretch). The `intro_offer` ratio of about 1.36 is an average over
  tenure of an effect that is not constant (Exercise 4); hazard ratios are not probabilities of churning.

### Run · Proportional-hazards test (provided)

Fits `cph = CoxPHFitter(alpha=ALPHA)` on `COVARIATES` (the same fit as Exercise 3's; 0.18 s in the
prototype) and prints `proportional_hazard_test(cph, data, time_transform="rank").summary`: a test,
from scaled Schoenfeld residuals (Grambsch & Therneau 1994), of whether each covariate's effect changes
with tenure. In the notebook this is one table and one sentence ("the test asks, for each covariate, whether its hazard ratio
changes with tenure; a small p-value says it does"); how the test works is in the briefing. Build:
`intro_offer` χ² = 99.25 (p = 2.2e-23); onboarding p = 0.44, plans 0.54 (premium) and 0.82 (standard).
Over 80 seeds the offer was flagged at p < 0.01 every time, and another covariate once (three tests at
0.01 give about a 3% chance of a false flag per seed). Use the test, not `check_assumptions(show_plots=True)`, which
bootstraps lowess curves with an unseeded generator and prints advice. The monthly churn-rate plot by
offer that the first draft put here comes after Exercise 4's checkpoint: it shows where the two rates
meet, which is the answer to Exercise 4's Predict.

### Exercise 4 · An effect that fades: hazard ratios before and after month 3 (8 minutes)

- **Predict.** The full-period model gave `intro_offer` a hazard ratio of about 1.36, and the test above
  says the offer's effect changes with tenure. The offer discounts the first three paid months. In months
  0-3, will its hazard ratio be higher than 1.36, lower or about the same? After month 3? Write both
  guesses in the cell below.
- **Function.** `landmark_split(subs, landmark=3.0) -> tuple[pd.DataFrame, pd.DataFrame]` returning
  `(before, after)`, each with the columns of `subs`:
  **before** = every subscriber, follow-up stopped at the landmark (tenure = min(tenure, 3), churned only
  if it happened by month 3); **after** = subscribers still active after month 3, with the clock restarted
  (tenure − 3). Return new frames and leave `subs` unchanged. This is a landmark analysis: fix a time,
  keep those still at risk, start again (Giobbie-Hurder et al. 2013). The reference is about five lines.
  The checkpoint, not the learner, then calls Exercise 3's `cox_hazard_ratios` on each part with
  `COVARIATES` and builds and prints the table the first draft asked the learner for (columns `period`
  ("before", "after"), `covariate`, `hr`, `lower`, `upper`): the split is the new idea, the table is
  bookkeeping.
- **Checkpoint.** First the structure: "before" has as many rows as `subs`, no tenure above the landmark,
  and no `churned` = 1 where the original tenure is above it; "after" has exactly the rows of `subs` with
  tenure > 3 and no tenure ≤ 0; `subs` is unchanged (equal to a copy taken before the call). Then
  on `COVARIATES`: the true offer HR 2.014 inside the "before" 95% interval
  and 1.0 inside the "after" one; the true onboarding HR 0.670 inside both. Build, seed 2034: offer
  2.061 [1.878, 2.261] and 0.955 [0.873, 1.045]; onboarding 0.680 [0.614, 0.753] and 0.681 [0.626, 0.741]
  (the prototype's PH test then flagged nothing at 0.01 in either period; the build does not print it).
  Over seeds 1000-1039 the four held in 38, 40, 37 and 38 seeds, all four together in 33; over
  3000-3039 in 40, 37, 37 and 38, all four in 32; at least 3 of 4 in 80 of 80, so the checkpoint
  requires at least 3 of the 4.
- **Failure messages** (each ending with `workshop.use_reference(4)`):
  - "before" has fewer rows than `subs`: "before keeps every subscriber and stops their clock at the
    landmark (`clip(upper=landmark)`); do not filter."
  - "before" counts cancellations after the landmark (the message gives how many): "these subscribers
    were still active at month 3, so in 'before' they are censored there: churned = 0."
  - "after" has a tenure ≤ 0 (rows with tenure ≤ landmark were kept), or its shortest tenure is above the
    landmark (the clock was not restarted): "after keeps those with tenure > landmark and restarts their
    clock: tenure − landmark."
  - `subs` changed: "your function changed `subs` in place, which would make Exercise 5 and the Decision
    wrong without an error. Build new frames (`subs.assign(...)` or `subs.copy()`), rerun Part A's first
    provided cell and the dummy-columns cell before Exercise 3, then this checkpoint."
  - Structure right but fewer than 3 of the 4 truths inside: on the committed data this does not happen
    with the reference; the message prints the four estimates, their intervals and the truths, and says
    to compare with `workshop.use_reference(4)`.

*Provided after it · Monthly churn rate by offer:* a plot of the monthly churn rate
(cancellations in month k ÷ subscribers active at the start of month k), months 1-24, for `intro_offer`
1 and 0. Sentence before it: "Look at the month where the two lines meet." Exercise 4's Explain points at
it.

- **Explain.** Compare with your guess. Why was 1.36 a misleading summary of the offer? What should a
  report say instead? Point to the two periods' intervals and the churn-rate plot.
- `<details>`: The full-period ratio averages a doubled churn rate in the first three months with no
  difference afterwards, weighted by when cancellations happen. Splitting at the landmark recovers both.
  Deal-seekers leave when the discount ends; those who stay are like everyone else. Alternatives:
  `strata=["intro_offer"]` gives the offer its own baseline and keeps the other ratios (0.742, 0.511,
  0.680 on the committed data, checked by the Academic Director on 2026-10-10; the lab does not run it)
  but reports no offer effect; a time-varying coefficient
  (`CoxTimeVaryingFitter` on episode data) is the general tool. Caution: here subscribers differ only in
  observed ways, so the survivors at month 3 are comparable; in real data the survivors are a selected
  group and an after-landmark ratio carries that selection (Hernán 2010).

### Exercise 5 · Months gained from onboarding (8 minutes)

*Provided before it* (read inside the 8 minutes): the unadjusted RMST(24) difference between onboarded
and not, from two lifelines
Kaplan-Meier fits and `restricted_mean_survival_time(kmf, t=24)` (build: 15.24 against 11.18 months, a
gain of 4.06; target population 4,139 subscribers, plan mix basic 0.666, standard 0.262, premium 0.072),
which shows the
learner the call their function repeats; the
definition of RMST with a sentence on why it suits a decision (it is in months, needs no proportional
hazards and multiplies by a monthly margin; Royston & Parmar 2013); and the target population: the
subscribers who did not onboard, whose plan mix (`weights`) is computed in the cell. It does not print
RMST by plan and onboarding: the Predict asks what standardizing does.

- **Predict.** Unadjusted, onboarded subscribers stay about 4.1 more months within the first 24. Standardized
  by plan, will the gain be larger, about the same or smaller? By how much: under 0.5 month, 0.5-2, over 2?
  Write your guess in the cell below. (Build: smaller by 1.4 months, 4.06 to 2.67; truth 1.15, 4.061 to
  2.907.)
- **Function.** `rmst_gain(subs, horizon, weights) -> float`: for each plan p, RMST(horizon) of the
  onboarded minus that of the not onboarded, each from its own `KaplanMeierFitter` and
  `restricted_mean_survival_time`; return Σ_p weights[p] × difference. `weights` is a dict or Series over
  plans summing to 1 (assert it). The reference is about eight lines.
- **Checkpoint.** The fast checks run first, then the bootstrap (the Markdown above the cell says
  "while it runs, read the Explain prompt below"; build: 4.81 s on macOS and 5.94 s in Docker, FULL, from
  the run records; not yet timed on Colab).
  1. With weights {"basic": 1.0}: equals the difference of the two lifelines RMSTs for basic computed
     directly (exact).
  2. With weights {"basic": 0.75, "premium": 0.25}: 0.75 × basic's direct difference + 0.25 × premium's
     (exact). A function that averages the plans without the weights passes check 1 and fails this one.
     The first draft used 0.5 and 0.5, which a plain average of the two plans also passes; the build
     changed it.
  3. Weights that sum to 0.5 ({"basic": 0.25, "premium": 0.25}) raise an error (the assertion the task
     asks for).
  4. Truth: the target-population gain 2.907 months (`truth.rmst.gain_target`) lies inside the 95%
     bootstrap percentile interval of `rmst_gain` (stratified by plan × onboarding, B and seed as above).
     Build, seed 2034: estimate 2.672; intervals [2.22, 3.10] (B = 200, FULL) and [2.28, 3.12] (B = 100,
     QUICK). The prototype's B = 200 interval was [2.022, 3.218]: the build's bootstrap
     (`mktstats.survival.bootstrap_interval`) draws its resamples differently, so its intervals differ
     at the same seed. Over seeds the truth was inside in 72 of 80 with B = 100 (38 and 34) and 73 of 80
     with B = 200 (38 and 35), about 91%: the prototype's seeds give its 38 of 40 again, the independent
     set fewer. The check rests on one 95% percentile interval, which misses at about the nominal rate
     plus some undercoverage. Mean estimates
     2.897 (seeds 1000-1039) and 2.971 (3000-3039) against the truth 2.907.
  5. The unadjusted 4.06 lies outside that interval (80 of 80 seeds, both B).
- **Failure messages** (each ending with `workshop.use_reference(5)`):
  - Check 1 negative: "you subtracted the other way: onboarded minus not onboarded."
  - Check 1 otherwise wrong: "for each plan, fit one `KaplanMeierFitter` to its onboarded subscribers and
    one to its not-onboarded subscribers, and take `restricted_mean_survival_time(kmf, t=horizon)` of
    each, as the provided cell above does for everyone."
  - Check 2 wrong: "weight each plan's difference by `weights[plan]` and add them up; a plain average
    ignores the weights."
  - Check 3, no error: "assert that the weights sum to 1 (within 1e-9), so that counts passed by mistake
    are caught."
  - Check 5, the unadjusted 4.06 inside the interval: "you compared onboarded with not onboarded across
    all plans; compare within each plan, then weight by the target population's plan mix."
  - Check 4 alone fails (build wording): "a 95% interval misses now and then (on other seeds of this
    generator the reference missed in about 1 in 10), but on the committed data the reference passes;
    compare your estimate with `workshop.use_reference(5)`."

*Provided after it · The gain plan by plan:* one row per plan with RMST(24) onboarded and not onboarded,
the difference, the weight, and weight × difference, and their sum (the estimate). The Explain points at
it.

- **Explain.** Compare with your guess. Which plan's subscribers dominate the target group, and why does
  standardizing to their plan mix change the answer? Point to the weights and the per-plan RMSTs.
- `<details>`: The unadjusted difference compares onboarded subscribers (one third premium) with the
  rest (two thirds basic); its own estimand is 4.06 months, of which only 2.91 is onboarding. Comparing
  within plan and weighting by the plan mix of the people a call would reach answers the decision's
  question. Stratifying fully adjusts here because onboarding depends on plan alone. Lifelines'
  `restricted_mean_survival_time(..., return_variance=True)` returns the variance of min(T, 24) across
  subscribers (SD 9.45 months over all subscribers on the committed data, checked by the Academic
  Director on 2026-10-10; the lab does not print it), not the uncertainty of the estimate; that is why
  the interval comes from the bootstrap.

### Decision · Is an onboarding call worth its cost? (5 minutes)

Provided cell, in the standard order, printing the mode:

- **Number:** ΔRMST(24) for subscribers who did not onboard, mean and 95% bootstrap percentile interval
  (Exercise 5), beside the unadjusted 4.06; value per call q × ΔRMST × m with its interval for q in
  `Q_GRID`; break-even q\* = c / (ΔRMST × m) with its interval; the share of bootstrap resamples in which
  the call pays at each q (label it as that, not as a probability). Build (FULL): value at q = 0.35 is
  $11.22 [$9.31, $13.03] against $15 (from the unadjusted gain it would be about $17); q\* = 0.468
  [0.403, 0.564] (true 0.430; unadjusted 0.308); the call pays in 0%, 0% and 74% of resamples at
  q = 0.20, 0.35, 0.50 (80% at q = 0.50 with QUICK's B = 100).
- **Rule** (stated before the numbers): call when q × ΔRMST × m > c, with m = $12 a month and c = $15 a
  call; q is not known.
- **Recommendation** (the learner's sentence, with what would change it): for example, do not roll out
  calls on this evidence: at q = 35% they return about $11 for each $15; they pay only if a call gets more
  than about 40-56% of reached subscribers to onboard (the build's break-even interval). The unadjusted comparison would have funded them.
  Before spending, run a randomized pilot (Module 6) that measures q and 6-month retention, because
  onboarding was not randomized and motivation may drive both (Module 15 estimates such effects from
  observational data). A cheaper nudge changes the answer: at an illustrative $3 a nudge, the
  break-even q is about 9%.
  ΔRMST is undiscounted and in months subscribed; discounting at Module 4's 0.8% a month lowers the true
  gain by 10% (2.907 → 2.614).
- **Scale** (one line in the cell, so the recommendation can say how many subscribers and dollars it
  concerns): the number of subscribers who did not onboard (the target population), what calling all of
  them would cost at c, and what it would return at each q.
- **The assumption the rule adds** (one line in the cell, beside the association warning): the rule
  values a call as if a subscriber whom the call gets to onboard, at or after the first billing date,
  gains as much as one who onboarded on their own during the trial. The pilot tests that as well.
- **Why the share of resamples is not a probability** (one sentence in the Markdown above the cell): the
  bootstrap measures sampling noise only; confounding, if any, would shift every resample the same way.

The Markdown above the cell states the rule and its economics first, then says "run the cell, then write
your sentence". The printed layout, in the order of `_lab-standard.md` §7, as the build prints it (FULL,
committed data):

```text
Number          : onboarding adds 2.67 months within 24 (95% bootstrap interval 2.22 to 3.10)
                  for subscribers who did not onboard, standardized by plan. Unadjusted: 4.06.
Rule            : call when q × ΔRMST × m > c, with m = $12 a month and c = $15 a call; q is not known.
Value per call  : q = 0.20: $6.41 (95% interval $5.32 to $7.45); calls pay in 0% of bootstrap resamples
                  q = 0.35: $11.22 (95% interval $9.31 to $13.03); calls pay in 0% of bootstrap resamples
                  q = 0.50: $16.03 (95% interval $13.29 to $18.61); calls pay in 74% of bootstrap resamples
Break-even q    : 0.468 (0.403 to 0.564); from the unadjusted gain it would be 0.308.
Scale           : 4,139 subscribers did not onboard; calling all of them costs $62,085 and returns $46,444 at q = 0.35.
Assumes         : onboarding prompted by a call works like onboarding chosen during the trial.
                  Onboarding was not randomized, so the gain is an association, not a measured effect.
Recommendation  : (yours) what to do, for how many subscribers, how sure, and what would change it.
What would change it: a pilot's measured q and retention; a cheaper nudge (at $3 the break-even q is 9%); discounting at 0.8% a month (10% less in the simulation's truth).
Your sentence   : ________________________________________________
Mode            : FULL run.
```

The truth cell after it prints the true gain 2.907 (the interval covers it), the unadjusted truth 4.061,
a call's true worth at q = 0.20, 0.35, 0.50 ($6.98, $12.21, $17.44 against $15) and the true break-even
q 0.430.

## Stretch (optional) · A confounder that is not in the data

Regenerate live with `subscriptions(seed=2034, hidden_confounding=0.5)` (under a second): motivation
now raises the odds of onboarding and lowers churn. Rerun `cox_hazard_ratios` and `rmst_gain` and compare
with the truth, the mean of `true_rmst24_gain` among subscribers who did not onboard. Build: Cox HR
0.584 [0.547, 0.623] against the conditional truth 0.670; adjusted gain 3.80 months against a true 2.74.
Adjusting for plan no longer removes the bias, and nothing in the data says so. Two shorter options:
`rmst_gain` per plan (one weight of 1.0) with bootstrap intervals, to see that the data cannot separate
the plans (truth 2.99, 2.82, 2.44 months; build estimates on the default data 2.69, 2.54, 2.99, from the
"gain plan by plan" table); and the decision with Module 4's 0.8% monthly discount.

## Run time (measured where a run record exists)

| | macOS laptop (run record) | Docker, `--cpus 2` (run record) | Colab |
|---|---|---|---|
| Worked, FULL, whole notebook | 9.0 s | 11.6 s | not run |
| Learner mode (Run all stops at the first unwritten exercise, as designed) | 3.0 s | 3.7 s | not run |
| Slowest cell: Checkpoint 5's bootstrap, B = 200 | 4.81 s | 5.94 s | not run |
| Install | 0 s (preinstalled) | 0 s (preinstalled) | not run; estimated under 1 minute |

Records: `runs/2026-10-10-macos-arm64-python-14-survival-churn.json` and
`runs/2026-10-10-docker-arm64-python-14-survival-churn.json`, both at `ref` v2026.10.1, content
`7264963f11767143`, deps `d36add16277e80bd`; the laptop was shared with other build agents during the
runs. The Colab install estimate is unchanged: Colab already has autograd 1.9.1, formulaic 1.2.2 and
interface_meta 2.0.1 (`environment/colab-constraints.txt`), so only `lifelines` and `autograd-gamma`,
both pure-Python wheels, are added (target ≤ 3 min). The lab cannot run on Colab until a release tag
contains it: the Colab badge, the mktstats install and the data all read `repo.ref`, and the tag
v2026.10.1 has no `labs/python/14-survival-churn.ipynb`. Targets on Colab (cell ≤ 4 min, total ≤ 15 min)
are unmeasured until then.

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
- Changing `subs` in place inside a learner's function (most likely in Exercise 4) makes Exercise 5 and
  the Decision wrong without any error; Checkpoint 4 compares `subs` with a copy taken before the call.
- Slicing past the end of a list or array returns a shorter result without complaint, which is how a
  curve gets used beyond its data (Exercise 2's `ValueError`).

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

All six were done in the build (checked by the Academic Director on 2026-10-10 against the files named):
the generator and its `EXTENSIONS` entry with an 80-seed `tolerances` block (1); `modules.m14.deps` (2);
`src/mktstats/survival.py` with `ci_contains`, `k_of_K_ci_contains` and `bootstrap_interval`, and tests
that feed them broken input in `tests/test_survival.py` (3); `survival.monthly_churn_rate` (4); the
module header lists `modules.m14.prerequisites` (5); and the worked runs pass in QUICK and FULL with
every checkpoint failing on its stub (6; re-run by the Academic Director on 2026-10-10). The list as
written:

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

## Pedagogy review (2026-10-10)

Checked against `briefs/_lab-standard.md` §11 before any code exists. What changed, and why:

1. **Budget: 10 + 7 + 7 + 8 + 8 = 40 (limit 40), with reading counted.** Exercise 2 had 6 minutes, which
   the Academic Director flagged as tight, and it was: its function is four lines, but in those minutes the
   learner switches dataset (Fader-Hardie, years instead of months, Module 4's encoding), meets a new idea
   (expected paid periods, the area under the curve) and reads a comparison. Counting the provided cell
   before it, a realistic estimate is 6.5 to 7 minutes. It now has 7, its provided cell prints only years
   0-7, and it hands over plain arrays. Exercise 3 goes from 8 to 7: its idea (adjust for a confounder, read
   a coefficient as a multiplicative effect) is the one this audience already knows from logistic
   regression, and the likely time sink, finding lifelines' attribute names, is now a hint in the stub.
2. **Exercise 4 asks for the split, not the table.** `landmark_hazard_ratios` built two data sets, called
   Exercise 3's function twice and reshaped the results into a long table: one new idea (the landmark)
   wrapped in about fifteen lines of bookkeeping. The learner now writes `landmark_split(subs, landmark)`
   (about five lines); the checkpoint fits and builds the same table. Structural checks give a precise
   message for each likely mistake, including changing `subs` in place, which would have made Exercise 5
   and the Decision wrong without an error.
3. **Predicts ask about what has not been shown yet.** The monthly churn-rate plot by offer showed where
   the two rates meet before Exercise 4's Predict asked exactly that; it now comes after the checkpoint, as
   the Explain's evidence. Exercise 2's provided cell no longer prints years 8-12, and its Predict has a
   reference value (4.41 expected paid years) so the guess has a scale. Exercise 1's Predict names four
   options. Every Predict asks for the guess in writing.
4. **Cognitive load.** The lab uses about twenty terms (the table under "Terms the notebook defines"),
   most of them new to analysts who know regression, more than any core lab. That is at the limit for
   one 55-minute lab, not over it, as long as each exercise carries one idea and each provided cell adds
   at most one term. To get there: the log-rank tests move to the briefing (a p-value of 5e-77 that no
   later step uses, while Exercise 3's Cox model answers the same question with an effect size); the
   PH-test cell is one table and one sentence, with Schoenfeld residuals left to the briefing; Greenwood's
   formula stays out of the notebook ("pointwise" is the only word the learner needs); the time-zero
   sentence moves to Part B, where groups are first compared; every term is defined where it is first
   used. Nothing moved to the stretch: each of the five exercises serves an objective in `modules.m14`.
5. **Checkpoint 1 checks only the learner's code.** Its parts 3 and 4 (the truth inside lifelines'
   interval, the dashboards outside) depend only on the data once part 2 passes, so they move to a labeled
   check in the provided cell right after, which also prints the table, the start dates and the plot the
   Explain points at. A tie toy is added: the first toy never has two cancellations at one time, so it
   could not catch either tie mistake (0.667 and 0.444 on the tie toy instead of 1/3).
6. **Failure messages for every checkpoint**, with the likely wrong answers computed on the toys where there
   is one (by the Pedagogy Expert, 2026-10-10). Exercise 2 adds horizon 2, which catches a function that
   ignores `horizon`; Exercise 5 adds a 50/50 weights case (a function that averages the plans without
   weights passes the {"basic": 1.0} case) and the check that its assertion fires.
7. **Scaffolding where the minutes would go to the API, not the idea:** Exercise 1's toy worked by hand
   for two times and the reversed-cumulative-sum hint; Exercise 3's column selection and the two lifelines
   attributes; Exercise 2's plain arrays instead of lifelines frames; separate names `kmf_subs` and
   `kmf_regular`; `COVARIATES` in the constants cell.
8. **Decision.** The rule comes first in Markdown; the printed layout is compact, with "What would change
   it" and "Your sentence" lines as in the standard's template and a Scale line (how many subscribers and
   dollars the recommendation concerns); an "Assumes" line names the assumption the
   rule adds on top of confounding (onboarding prompted by a call works like onboarding chosen during the
   trial; the pilot tests it); one sentence says why the share of bootstrap resamples is not a probability.

**If a pilot runs long.** Exercise 2 is the first to become a provided cell: keep its Predict, the
years 8-12 table and its Explain, and provide `expected_periods`; that frees about 6 minutes. Keep
Exercise 4 hands-on: the landmark is the idea learners would otherwise only hear about.

**What the briefing carries, so that the lab can stay short** (for the Academic Director's module page):
censoring, with a picture of staggered start dates and the observation end; the Kaplan-Meier product on a
small worked example; the log-rank test (the text and prototype numbers kept under Part B above); the
hazard, the Cox model and the hazard ratio as a ratio of rates, not of probabilities (Hernán 2010);
immortal-time bias, with a picture; what the PH test checks (scaled Schoenfeld residuals); the landmark
idea and its selection caveat; RMST and why it suits a decision (Royston & Parmar 2013); and the intro
offer's story (deal-seekers leave when the discount ends), so that Exercise 4's Predict is reasoning, not a
guess about the simulation.

**For the Academic Director to reconsider.** The "Assumes" line in the Decision is a new statement of an
assumption, not a new fact; check its wording against the module page. The log-rank test is out of the lab;
if you want it back, the stretch is the place, not a core cell.

## Build reconciliation (2026-10-10, Academic Director)

The brief now describes the lab as built. I re-ran the notebook in worked mode (FULL, then QUICK) at
the content and deps hashes of the two run records and read its printed output; every number marked
*build* above comes from that output, from `truth_subscriptions.json` (truth, `sample`, `tolerances`)
or from the run records. What changed against the prototype brief:

1. **Checkpoint 5, check 2 uses weights {basic 0.75, premium 0.25}**, not 0.5 and 0.5: with equal
   weights a plain average of the two plans passed, so the check could not catch the mistake it names.
2. **Coverage is about 91% for Checkpoints 3 and 5, not 38-40 of 40.** Over 80 seeds (1000-1039 and an
   independent 3000-3039) Checkpoint 3 held in 73 and Checkpoint 5's truth-inside check in 72 (B = 100)
   and 73 (B = 200). On the prototype's own 40 seeds the build gives the prototype's 39 and 38; the
   independent 40 give 34 and 34-35. Each rests on one 95% interval, so about 1 seed in 10 misses: the nominal 5% plus
   some undercoverage of the percentile bootstrap, and on 3000-3039 the two failed on the same five
   seeds. Checkpoint 4 (3 of 4) held in 80 of 80, the Kaplan-Meier check (3 of 4) in 78, and every
   "unadjusted is outside" check in 80. All rules hold on the committed seed in QUICK and FULL. The
   failure message of Checkpoint 5 says "about 1 in 10", not "2 of 40".
3. **The bootstrap intervals differ from the prototype's** at the same seed, because the build's
   `bootstrap_interval` draws its resamples differently: FULL (B = 200) gives an RMST gain of 2.672
   [2.22, 3.10] against the true 2.907 (prototype [2.022, 3.218]); QUICK (B = 100) [2.28, 3.12]. The
   Decision's intervals follow: $11.22 [$9.31, $13.03] per call at q = 0.35 and break-even q 0.468
   [0.403, 0.564], so the sample recommendation now says "about 40-56%".
4. **Passing all of `subs` to `CoxPHFitter` raises `TypeError: Cannot cast DatetimeArray to dtype
   float64`**, not the `ValueError` the brief quoted (the date columns are reached first).
5. **Run times are measured**, from the two run records (9.0 s worked FULL on macOS, 11.6 s in Docker
   with `--cpus 2`; slowest cell the bootstrap, 4.81 s and 5.94 s). Colab: not run; it needs a release
   tag that contains the lab.
6. **Unchanged**: every point estimate on seed 2034 (dashboards, Kaplan-Meier, Fader-Hardie totals, Cox
   and landmark hazard ratios with their intervals, PH test, RMST by plan, stretch). The generator
   reproduces the prototype's draws, so these equal the prototype's to the digits quoted; the build
   prints the PH test's statistic as 99.25 (p = 2.2e-23), where the prototype said 99 (p < 1e-20).
7. **Numbers the lab does not print** and that the module page quotes (log-rank χ² 345.2 and 415.4, the
   stratified Cox ratios, the SD of min(T, 24)) were computed by the Academic Director on the committed
   data with lifelines 0.30.3 on 2026-10-10 and are labeled as such where they appear.
8. **On the two points left for me to reconsider:** the Decision's "Assumes" line is kept as written; the
   module page states the same assumption in its decision section. The log-rank test stays out of the
   lab; the module page carries it.
