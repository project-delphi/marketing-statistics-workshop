# Lab brief · Module 11 · Uplift and heterogeneous treatment effects

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebooks | `labs/src/11-grf-causal-forest.R` → `labs/r/11-grf-causal-forest.ipynb` (Part A, Hillstrom); `labs/src/11-uplift-econml.py` → `labs/python/11-uplift-econml.ipynb` (Part B, synthetic with truth; decision) |
| Lab slot | 55 min for both notebooks (`modules.m11.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (R 20 + Python 20; limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Order | Open the **Python** notebook first and run its install cell (≈ 31 s + 17 s import on Colab, lead's measurement). Then open the **R** notebook (grf installs in ≈ 5 s) and work Part A. Return to the Python notebook for Part B and the decision. |
| Day | Day 5, first module (before the capstone) |

## Question and decision

**Question.** Which customers buy *because of* the email offer, and whom should we send it to?

**Decision.** Send the offer only to customers whose expected incremental margin exceeds the offer cost;
report how many that is and the incremental profit compared with mailing everyone.

## What this lab fixes

Module 10 decides how much to spend per channel but treats every customer the same; Module 6 measured the
average effect of the email, which hides who responds.

## Notation

Treatment T ∈ {0, 1} (offer sent), outcome Y (two-week spend, or conversion), features X. Conditional average
treatment effect (CATE, also "uplift") τ(x) = E[Y(1) − Y(0) | X = x]. Target a customer if margin × τ(x) >
cost per offer.

## Data

| Source | Loader | Use |
|---|---|---|
| Hillstrom (real, randomized) | R: committed or downloaded via `R/mktstats.R` (the Technical Expert decides; MineThatData does not state a licence, so do not mirror it); Python not needed | Part A: Womens e-mail versus No e-mail, outcome `spend`, features `recency, history, mens, womens, zip_code, newbie, channel` |
| Synthetic email experiment (the retailer's story; randomized) | `mktstats.synth.email_experiment(seed=<default>)` with per-row `true_cate` (spend scale), truth `ate`, `offer_cost`, `margin` | Part B, decision |

Splits: 50/50 train/evaluation split with a visible seed in both notebooks; every evaluation (RATE, Qini,
profit) uses the held-out half only.

## Model fits

- R: two causal forests on Hillstrom (training forest for priorities; evaluation forest for RATE), both with
  `W.hat = 0.5` because assignment was randomized with probability one half within the two arms used; QUICK
  `num.trees = 500`, FULL 2,000 (grf's default). Not yet timed on Colab.
- Python (provided cell): `SLearner(overall_model=HistGradientBoostingRegressor())`,
  `TLearner(models=HistGradientBoostingRegressor())`, `XLearner(models=HistGradientBoostingRegressor(),
  propensity_model=LogisticRegression())`, and `CausalForestDML(model_y=HistGradientBoostingRegressor(),
  model_t=HistGradientBoostingClassifier(), discrete_treatment=True, n_estimators=N, random_state=SEED)`
  with N = 200 QUICK, 1,000 FULL (must be divisible by `subforest_size`, default 4). Each `.fit(Y, T, X=X)` on
  the training half. A 20,000-row `CausalForestDML` with 200 trees took 17.4 s on a laptop in spike S2 (not a
  Colab time).

## Parts and exercises

| Notebook · Part | Exercise | Minutes |
|---|---|---|
| R · A · Causal forest on Hillstrom | 1 · Average effect, doubly robust | 7 |
| | 2 · Does the forest rank customers well? RATE on held-out data | 7 |
| | 3 · Who clears the bar? | 6 |
| Python · B · Uplift models against the truth | *Run: four uplift models (provided)* | — |
| | 4 · How close are the effect estimates? | 8 |
| | 5 · Qini curves on held-out data | 6 |
| | 6 · Profit of a targeting rule | 6 |
| **Exercises** | | **40** |
| Decision (Python notebook) | Whom to send the offer? | 5 |

### Exercise 1 · Average effect, doubly robust (7 minutes, R)

- **Predict.** In Module 6 you saw Hillstrom's arms. Will the Womens e-mail's average effect on two-week
  spend be below $0.50, between $0.50 and $1.50, or above $1.50 per customer?
- **Function.** `fit_forest(X, Y, W, num_trees, seed)` returning `causal_forest(X, Y, W, W.hat = 0.5,
  num.trees = num_trees, seed = seed)`; a provided line then calls `average_treatment_effect(forest,
  target.sample = "all")` (AIPW by default) and prints the 95% interval as estimate ± 1.96 × std.err.
- **Checkpoint.** `forest$W.hat` is 0.5 for every row; the 95% interval contains the simple difference in
  means of the training half (randomized data: the two should agree within sampling error) and has positive
  width. First R checkpoint; the fit is the R notebook's only long step.
- **Explain.** Knowing the assignment probability (randomized) removes one source of error; the AIPW
  estimate adjusts for chance imbalances in X (Athey & Wager 2019 for an application).

### Exercise 2 · Does the forest rank customers well? RATE on held-out data (7 minutes, R)

- **Predict.** Will the RATE (Qini target) on held-out customers have a 95% interval that excludes 0: yes
  or no?
- **Function.** `rate_holdout(train_forest, X_eval, Y_eval, W_eval, num_trees, seed)`: priorities =
  `predict(train_forest, X_eval)$predictions`; fit an evaluation forest on the held-out half (`W.hat = 0.5`);
  return `rank_average_treatment_effect(eval_forest, priorities, target = "QINI")`.
- **Checkpoint.** The result has `estimate`, `std.err` and a `TOC` data frame with `q` from 0.1 to 1;
  with a provided random priority vector instead, the 95% interval contains 0.
- **Explain.** RATE asks whether customers ranked higher really gain more (Yadlowsky et al. 2025). Using a
  separate evaluation forest on held-out data keeps the ranking from grading itself. Plot the TOC (provided).

### Exercise 3 · Who clears the bar? (6 minutes, R)

- **Predict.** At a $0.10 cost per email and a 30% margin, will more or less than half of customers be worth
  emailing?
- **Function.** `policy_value(tau_hat, scores, margin, cost)` returning the share targeted
  (margin × tau_hat > cost) and the estimated incremental profit per 1,000 customers, Σ over targeted
  (margin × Γ_i − cost) scaled to 1,000, where Γ_i are the doubly robust scores from
  `get_scores(eval_forest)`.
- **Checkpoint.** Arithmetic on a toy input; "target everyone" equals n × (margin × mean(Γ) − cost) scaled to
  1,000; "target nobody" is 0.
- **Explain.** Doubly robust scores give an unbiased value for any targeting rule on held-out data, so you
  can compare rules without an experiment per rule.

### Exercise 4 · How close are the effect estimates? (8 minutes, Python)

- **Predict.** Which model will rank customers best by their true effect: S-learner, T-learner, X-learner or
  causal forest? Pick one.
- **Function.** `cate_accuracy(cate_hat: pd.DataFrame, true_cate: np.ndarray) -> pd.DataFrame` with, per
  model column, RMSE against the true CATE and Spearman rank correlation.
- **Checkpoint.** Exact on a toy input; on the evaluation half, the causal forest's 95% ATE interval
  (`cf.ate_interval(X_eval, alpha=0.05)`) contains `truth.ate`, and its Spearman correlation is at least the
  reference value minus 0.05.
- **Explain.** The S-learner can shrink the treatment effect toward zero because the treatment is one feature
  among many; the T-learner's two separate models add their errors; the X-learner and DML forest target the
  difference directly (Künzel et al. 2019; Chernozhukov et al. 2018).

### Exercise 5 · Qini curves on held-out data (6 minutes, Python)

- **Predict.** Will the model with the best rank correlation in Exercise 4 also have the largest Qini
  coefficient: yes or no?
- **Function.** `qini_table(y, t, scores: dict[str, np.ndarray]) -> pd.DataFrame` using
  `mktstats.uplift.qini_coefficient(y, t, score)` per model, plus rows for a random score and for the true
  CATE (the best possible ranking).
- **Checkpoint.** The true-CATE row is the largest; the random row is within a tolerance of 0 set from 200
  random permutations (the Technical Expert states it); provided plot uses `mktstats.uplift.qini_curve`.
- **Explain.** A Qini curve plots cumulative incremental gain as you target more of the ranked list
  (Radcliffe & Surry 2011); its area summarizes ranking quality, not the size of the effect.

### Exercise 6 · Profit of a targeting rule (6 minutes, Python)

- **Predict.** Will targeting by the causal forest make more true profit than mailing everyone: yes or no?
- **Function.** `targeting_profit(cate_hat, true_cate, margin, cost) -> dict` with the share targeted
  (margin × cate_hat > cost), the expected profit from the model's view (Σ margin × cate_hat − cost over
  targeted) and the true profit (Σ margin × true_cate − cost over targeted).
- **Checkpoint.** The true profit per customer equals `mktstats.uplift.policy_value_true(true_cate, policy,
  margin, cost)` for the same policy; targeting by the true CATE reproduces `truth.policy_value_per_customer.oracle`
  (0.740 on main, with `treat_all` 0.498 and `treat_none` 0) and is at least every model's true profit;
  arithmetic on a toy input. Margin 0.35 and offer cost $0.50 come from the truth.
- **Explain.** A model can rank well and still misjudge the effect's size, which moves the cutoff; the gap
  between the model's expected profit and the true profit is that error.

### Decision · Whom to send the offer? (5 minutes, Python notebook)

Provided cell, in the standard order:
- **Number:** for the causal forest on the synthetic experiment: customers targeted, expected incremental
  profit and its 95% interval from the held-out doubly robust estimate, against "mail all" and "mail none";
  the share of targeted customers whose 90% CATE interval (`effect_interval(alpha=0.1)`) lies above
  cost ÷ margin. Optional: the Hillstrom numbers from Exercise 3 typed into a provided variable.
- **Rule:** "Send the offer when margin × expected uplift > cost per offer (truth's `margin` and
  `offer_cost`)."
- **Recommendation:** the learner's sentence with how many customers, the incremental profit against mailing
  everyone with its range, and what would change it (a different offer cost; a drift in who receives
  emails). Then the cell shows the true profit. The cell prints the mode.

## Stretch (optional)

R: `best_linear_projection(eval_forest, X_eval[, c("recency", "history")])` to describe who benefits;
Python: the same Part B on Hillstrom (no truth; compare Qini only).

## Known pitfalls

- Hillstrom has three arms; drop the Mens e-mail arm (or analyze it separately) so W is binary with
  probability one half.
- Spend is mostly zeros; effects per customer are small and noisy; use enough trees in FULL.
- Never evaluate a ranking on the data that trained it.
- `CausalForestDML` with `discrete_treatment=True` needs a classifier for `model_t`.
- `n_estimators` must be divisible by `subforest_size` (default 4).

## APIs verified (and how)

- R, image `mktstats-env:spike` (grf 2.6.1), 2026-10-09: `args(causal_forest)` (`W.hat`, `num.trees = 2000`,
  `seed`), `args(average_treatment_effect)` (`target.sample`, `method = c("AIPW", "TMLE")`),
  `args(rank_average_treatment_effect)` (`target = c("AUTOC", "QINI")`, `q`, `R = 200`); ran a 4,000-row
  simulation: the result has `estimate, std.err, target, TOC` and `TOC` has columns `estimate, std.err, q,
  priority`. `get_scores` is exported (`get_scores.causal_forest(forest, subset = NULL, ...)`), checked
  with `args()`; not run here.
- Python, econml 0.17.0 installed: `CausalForestDML.__init__` (incl. `discrete_treatment`, `n_estimators`,
  `subforest_size=4`, `random_state`), `.fit(Y, T, *, X, W)`, `.effect`, `.effect_interval(X, alpha)`,
  `.ate_interval(X, alpha)`; `SLearner(overall_model=)`, `TLearner(models=)`, `XLearner(models=,
  propensity_model=)`.
- `mktstats.uplift` on main has `qini_curve(y, treatment, score)`, `qini_coefficient(y, treatment, score,
  normalize=False)`, `auuc`, `uplift_curve`, `uplift_at_k`, `targeting_rule(cate, margin, cost)`,
  `policy_value(...)` and `policy_value_true(true_cate, policy, margin, cost)` (function names read on main;
  not run here).

## Open items for the Technical Expert

- Resolved on main: the uplift API above and an `email_experiment` with Hillstrom-like features and
  `true_cate` on the spend scale (truth `tolerances.email_uplift` checks the true CATE's Qini beats a random
  score). Confirm `qini_coefficient`'s sign and scale for Exercise 5's random-score tolerance.
