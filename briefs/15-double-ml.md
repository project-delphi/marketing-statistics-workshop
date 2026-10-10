# Lab brief · Module 15 · Causal effects without an experiment: double machine learning

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab was built: no notebook exists and nothing here has run as a lab. Numbers marked **prototype** come from a throwaway script on the Academic Director's laptop (Apple Silicon, scikit-learn limited to 2 threads with `OMP_NUM_THREADS=2`), not from the lab, Docker or Colab. They show that the data support the lesson; the lab prints its own numbers. Review: Pedagogy Expert. |
| Notebook | `labs/src/15-double-ml.py` → `labs/python/15-double-ml.ipynb` |
| Lab slot | 55 min (`modules.m15.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Day | Extensions (`d6`), optional. Comes after Modules 6 and 11, which it builds on (not after Module 14, whatever the generated "Before you start" list says; see open items). |

## Question and decision

**Question.** Did customers who chose to install the retailer's new app spend more *because of* the app,
given that the customers who install are the ones who already buy most?

**Decision.** Decide whether a credit paid to customers for installing the app pays back within 26 weeks,
from the incremental margin per persuaded install and its 95% confidence interval, and name the unobserved
confounder that could overturn the decision (`modules.m15.decision`). The lab prices a $2.50 credit at the
retailer's 30% gross margin.

## What this lab fixes

Modules 6, 7 and 11 estimated the effects of things the company randomized or switched on. Installing an
app, joining a loyalty program or finishing onboarding (Module 14's hazard ratio) are chosen by the
customers themselves, and the customers who choose them differ from those who do not. This lab estimates a
causal effect from such observational data under a stated assumption (all common causes are observed),
checks the method against a known truth and against a randomized benchmark, and shows what happens when the
assumption fails.

Not a repeat of Module 11: there, `CausalForestDML` ran on randomized data with a known propensity of 0.5,
so the treatment model had nothing to learn. Here the propensity is unknown and nonlinear, and learning it is
the point.

## Notation (match the module page and Module 11)

| Symbol | Meaning | Column / code |
|---|---|---|
| $T_i \in \{0, 1\}$ | customer $i$ installed the app in the 4 weeks after launch | `app_installed`, `t` |
| $Y_i$ | dollars spent in the 26 weeks after that install window | `spend_next_26w`, `y` |
| $X_i$ | pre-launch covariates (12 months before launch) | `W` in code: EconML calls controls `W` and effect modifiers `X`; this lab passes all covariates as `W` and `X=None` |
| $e(x) = P(T = 1 \mid X = x)$ | propensity score (probability of installing) | `t_hat` = $\hat e$ |
| $q(x) = E[Y \mid X = x]$ | expected spend given covariates (EconML's user guide also calls it $q$) | `y_hat` = $\hat q$ |
| $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ | conditional average treatment effect, as in Module 11 | truth column `true_effect` |
| $\tilde Y = Y - \hat q(X)$, $\tilde T = T - \hat e(X)$ | residuals | `y - y_hat`, `t - t_hat` |
| $\theta$ | the single number double machine learning (DML) estimates here | `theta` |
| $m$ | gross margin rate on spend, 0.30 (Module 5's assumption) | `MARGIN` |

**Assumptions.** (1) *No unobserved confounding* (unconfoundedness, conditional exogeneity): given $X$,
installing is as good as random, $Y(0), Y(1) \perp T \mid X$. (2) *Overlap*: $0 < e(x) < 1$ for every $x$.
DML needs both; it cannot test the first.

**The estimator** (partialling out; Robinson 1988 with nonparametric regressions; Chernozhukov et al. 2018
with machine learning and cross-fitting):

$$\hat\theta = \frac{\sum_i \tilde T_i \tilde Y_i}{\sum_i \tilde T_i^2}, \qquad
\widehat{\mathrm{se}}(\hat\theta) = \frac{\sqrt{\sum_i \tilde T_i^2 (\tilde Y_i - \hat\theta \tilde T_i)^2}}{\sum_i \tilde T_i^2},
\qquad \text{95\% CI: } \hat\theta \pm 1.96\,\widehat{\mathrm{se}},$$

with $\hat q$ and $\hat e$ *cross-fitted*: each customer's predictions come from models fitted on the other
folds. The standard error is the heteroskedasticity-robust (HC0) one; the interval is a 95%
normal-approximation confidence interval.

**What $\theta$ is when effects vary.** With $Y = \mu_0(X) + T\,\tau(X) + \varepsilon$ we have
$Y - q(X) = (T - e(X))\,\tau(X) + \varepsilon$, and $E[(T - e(X))^2 \mid X] = e(X)(1 - e(X))$, so the
residual-on-residual regression converges to

$$\theta = \frac{E[\,e(X)(1 - e(X))\,\tau(X)\,]}{E[\,e(X)(1 - e(X))\,]},$$

an average of the effects weighted by $e(1 - e)$: largest (0.25) for customers whose install was a toss-up,
near 0 for customers almost sure to install or not. These are the *overlap weights* of Li, Morgan &
Zaslavsky (2018), who weight each unit by its probability of being in the other group. If $\tau$ is
constant, $\theta = \tau$. In this lab the effect varies, so every checkpoint names which average it compares
with. The derivation goes on the module page; the lab states the result in Exercise 5.

## The synthetic world: `app_install` (generator spec)

**Story.** The retailer launched a shopping app. It announced the app by e-mail and on the checkout page
during the 4 launch weeks, so customers who open e-mails, shop on their phones, or had bought recently (and
so were likely to be shopping again during the launch weeks) were more likely to install. Loyal, frequent,
recent customers are also the ones who spend most in the next 26 weeks. That is the confounding.

**Files and code** (rules for extension datasets, `scripts/make_synthetic.py`):

- Generator: new file `src/mktstats/synth/app_install.py`, function `app_install(seed=2035, *, n=20_000,
  **params) -> SynthResult` (name `"app_install"`, one table `customers`). Seed 2035 is Module 15's
  assigned seed. Not exported from `synth/__init__.py`.
- Registered in `EXTENSIONS` as `"app_install"`; writes `data/synthetic/app_install_customers.csv` and
  `data/synthetic/truth_app_install.json`. `truth.json` is untouched.
- The lab reads only the committed files:
  `pd.read_csv(repo_file("synthetic/app_install_customers.csv"))` and
  `json.loads(repo_file("synthetic/truth_app_install.json").read_text())` (`mktstats.data.repo_file`; no change
  to `data.py`).
- Size: 20,000 rows, 14 columns, about 1.4 MB as CSV (prototype).

**Columns** (`customers`):

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | int | 1..n |
| `acquisition_channel` | str | `search`, `social`, `referral` (the shared vocabulary, `mktstats.synth.channels`) |
| `tenure_months` | int | months since first purchase, 3..96 |
| `orders_12m` | int | orders in the 12 months before launch, ≥ 1 |
| `days_since_last_order` | int | at launch, 1..365 |
| `avg_order_value` | float | dollars per order, 12 months before launch |
| `email_open_rate` | float | share of e-mails opened, 0..1 |
| `mobile_share` | float | share of web sessions on a phone, 0..1 (from web analytics; many CRM extracts lack it) |
| `loyalty_member` | int | 0/1 |
| `app_installed` | int | $T$ |
| `spend_next_26w` | float | $Y$, dollars, 0 when no purchase |
| `true_propensity` | float | truth: $e(x_i)$ |
| `true_effect` | float | truth: $\tau(x_i)$, dollars over 26 weeks |
| `true_baseline` | float | truth: $\mu_0(x_i) = E[Y(0) \mid x_i]$ |

The three `true_*` columns are for checkpoints only and must never enter a model (the lab's `features()`
helper selects columns by name).

**Data-generating process** (numpy `Generator(PCG64(seed))`; `expit` is the logistic function; the
order of draws is the Technical Expert's choice, so committed values will differ from the prototype's):

1. Latent activity $a_i \sim N(0, 1)$. It only makes the covariates correlated; it never enters $T$ or $Y$.
   It is not saved.
2. Covariates:
   - `acquisition_channel` ~ Categorical(search 0.45, social 0.35, referral 0.20);
   - `tenure_months` = clip(round(3 + Gamma(shape 2, scale 12)), 3, 96);
   - `orders_12m` = 1 + Poisson(exp(0.55 + 0.6 a + 0.15·[referral]));
   - `days_since_last_order` = clip(round(365 · Beta(1, 1 + 0.6 · orders_12m)), 1, 365);
   - `avg_order_value` = round(exp(N(log 55 + 0.15·[referral] + 0.1 a, sd 0.45)), 2);
   - `email_open_rate` = round(Beta(6 μ, 6(1 − μ)), 3) with μ = expit(−0.8 + 0.5 a);
   - `mobile_share` = round(Beta(4 μ, 4(1 − μ)), 3) with μ = expit(−0.3 + 0.9·[social] + 0.2·[search] − 0.01 (tenure − 24));
   - `loyalty_member` ~ Bernoulli(expit(−1.3 + 0.7 a + 0.015 · tenure)).
3. Install. With $r_{30} = \exp(-\text{days} / 30)$ (recent buyer, sharp in the first weeks):
   $\eta(x) = -4.2 + 3.0\,r_{30} + 0.5\log(1 + \text{orders}) + 2.4\,\text{mobile} + 1.2\,\text{email} + 0.5\,\text{loyalty} + 0.4\,[\text{social}] - 0.2\,[\text{referral}]$,
   $e(x) = \mathrm{expit}(\eta(x))$, $T \sim \mathrm{Bernoulli}(e(x))$.
4. Spend without the app. Expected orders in 26 weeks
   $r_{26}(x) = 0.3 \cdot \text{orders} \cdot (0.3 + 1.6\,r_{30}) \cdot (1 + 0.2\,\text{loyalty})$;
   expected spend $\mu_0(x) = r_{26}(x) \cdot \text{aov} \cdot (0.5 + \text{mobile})$.
5. Effect (selection on gains: heavier spenders gain more): $\tau(x) = 8 + 0.12\,\mu_0(x)$ dollars over 26 weeks.
6. Outcome. A customer buys at least once with probability $p(x) = 1 - \exp(-r_{26}(x))$, which the app does
   not change; given a purchase, spend ~ Gamma(shape 3, mean $(\mu_0(x) + T\,\tau(x)) / p(x)$), rounded to
   cents; otherwise 0. So $E[Y \mid X, T] = \mu_0(X) + T\,\tau(X)$ exactly.

Every arrow into $T$ and $Y$ starts at a column in the file, so assumption (1) holds given all nine
covariates (channel as two indicators, `search` the reference). By construction: `mobile_share` is a
confounder (enters $\eta$ and $\mu_0$); `email_open_rate` moves install only; `tenure_months` enters neither
directly; recency enters both through the sharp curve $r_{30}$, which a linear term cannot follow.

**The incentive** (for the decision; an assumption of the synthetic world). Each customer has
$U_i \sim \mathrm{Uniform}(0, 1)$ and installs in the launch window if $U_i < e(x_i)$. A credit that
multiplies the odds of installing by $k$ would make them install if $U_i < \mathrm{expit}(\eta(x_i) + \log k)$.
The *persuaded* customers are those in between; their average effect is
$\sum_i w_i \tau_i / \sum_i w_i$ with $w_i = \mathrm{expit}(\eta_i + \log k) - e(x_i)$. We also assume that
nobody who failed to install in the window would install later without the credit, so every credit paid
buys a persuaded install (cost per persuaded install = credit). As $k \to 1$, $w_i \propto e_i(1 - e_i)$: the
customers a small nudge persuades are weighted exactly as DML weights them. This is a property of the
logistic install model, not a general law; the module page should say so.

**`truth_app_install.json`** (written by `build_extension`, which adds `about` and `files`):

- `seed`, `n`, `description` (one paragraph: the story and the assumption that all confounders are observed);
- `parameters`: every constant above, by name; `formulas`: the equations above as strings;
- `estimands` (all over the n customers, with the true propensity as weights where needed, each with a
  `definition` string): `ate` (mean τ), `att` (weights e), `atc` (weights 1 − e), `effect_overlap`
  (weights e(1 − e): what $\hat\theta$ targets), `naive_expected` (the population difference in means);
- `incentive`: `odds_multiplier` 2, `effect_persuaded`, `share_persuaded`, and a table for multipliers
  1.25, 1.5, 2, 3, 5, 10;
- `decision`: `margin` 0.30, `credit` 2.50, `horizon_weeks` 26, `break_even_effect` = credit / margin
  (8.3333), the rule as text, and the persuasion assumption as text;
- `confounders`: for each covariate, whether it enters install, spend, both or neither;
- `install_rate` (realized), `tolerances` (coverage measured by the Technical Expert over at least 30
  seeds, as for `truth.json`).

**Prototype values** (seed 2035, n = 20,000; the committed values will differ with the draw order):

| Quantity | Prototype |
|---|---|
| install rate | 29.3% |
| mean spend, installers / non-installers | $97.54 / $31.90 (61% of customers spend $0) |
| `ate` / `att` / `atc` | $13.58 / $17.64 / $11.87 |
| `effect_overlap` | $14.28 |
| `effect_persuaded`, odds × 1.25, 2, 10 | $14.07, $13.66, $12.58 (shares persuaded 3.7%, 12.2%, 42.5%) |
| `naive_expected` | $65.73 |
| true propensity: 1st / 50th / 99th percentile | 0.036 / 0.232 / 0.895 (3.3% below 0.05; 0.09% above 0.95) |

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic app-install customers (the retailer's story; observational, known truth) | `repo_file("synthetic/app_install_customers.csv")`, `repo_file("synthetic/truth_app_install.json")` | Exercises 1–5, decision |
| Hillstrom e-mail experiment (real, randomized: Mens e-mail, Womens e-mail, No e-mail; two-week outcomes) | `mktstats.data.load_hillstrom()` (downloaded, sha256-checked, cached; not mirrored) | Exercise 6: Womens e-mail versus No e-mail, outcome `visit` |

Why `visit` and not `spend` for Hillstrom: in the prototype the confounded sample's naive estimate of the
effect on spend ($0.56, SE $0.19) could not be told apart from the benchmark ($0.42, SE $0.13), so the bias
would not show. On visits it shows clearly (below).

## Model fits

- **Nuisance models:** scikit-learn `HistGradientBoostingRegressor` (spend) and
  `HistGradientBoostingClassifier` (install), both `max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
  min_samples_leaf=50, early_stopping=False, random_state=SEED`. `early_stopping=False` must be explicit:
  the default `'auto'` switches early stopping on above 10,000 rows (scikit-learn 1.6.1 docstring), which
  would hold out a random validation set inside every fold.
- **Why not LightGBM** (the original plan): it gave the same lesson in the prototype (DML $13.92, 95% CI
  [$10.94, $16.91]) but (a) the macOS wheel of LightGBM 4.6.0 does not import without Homebrew's OpenMP
  (`brew install libomp`, per its README; it failed on this laptop), which breaks the local uv path, and
  (b) inside EconML it raised 20 `UserWarning`s per `LinearDML` fit ("X does not have valid feature
  names"). Histogram gradient boosting ships with scikit-learn, raised no warnings, and is what Module 11's
  learners already used. LightGBM is still installed (econml 0.17.0 requires `lightgbm>=4.5.0`), but
  `import econml.dml` and `econml.dr` did not import it on this laptop, so the lab never touches it.
  LightGBM can be named on the module page as a drop-in alternative.
- **Folds:** `folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(W, t))`,
  stratified on install, built once in a provided cell and passed to both the hand-written estimator and
  EconML (`cv=folds`), so their answers can be compared exactly.
- **Fits, once each:** the cross-fitted $\hat q$ and $\hat e$ (10 boosted fits, provided cell after
  Checkpoint 2), reused by Exercises 3 and 5; `LinearDML` in Exercise 4 refits the same 10 on purpose (the
  check is that the library reproduces your numbers); the overfit demonstration (2 fits on all rows); the
  hidden-confounder demonstration (10 fits); one `LinearDML` on the Hillstrom sample.
- **QUICK:** a seeded random 10,000 of the 20,000 customers and a seeded half of Hillstrom; same models.
  Truth targets are recomputed on the rows in use from the `true_*` columns, so checkpoints stay exact.
  In the prototype QUICK did not save time (boosting time depends on iterations more than rows), so its
  value here is a second, noisier sample, not speed. The header says what QUICK changes.
- `SEED = 2035`, visible in the first code cell; `warnings.filterwarnings` only for messages the brief
  names (see pitfalls), each with a comment.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Why the obvious answers are wrong | 1 · Naive and regression estimates against the truth | 7 |
| B · Double machine learning by hand | 2 · Cross-fitted predictions | 8 |
| | *Run: the two prediction models (provided)* | — |
| | 3 · Residual on residual, with a standard error | 7 |
| | *Run: what skipping cross-fitting does (provided)* | — |
| C · The library, and whose effect it is | 4 · LinearDML reproduces your estimate | 5 |
| | 5 · Overlap: whose effect is it? | 6 |
| D · A real benchmark | 6 · Recover an experiment from a confounded sample | 7 |
| **Exercises** | | **40** |
| Decision | Pay customers to install? (opens with a provided hidden-confounder run) | 5 |

Names used throughout: `customers`, `truth`, `y`, `t`, `W`, `folds`, `y_hat`, `t_hat`, `mine` (the
learner's DML result), `est` (the fitted `LinearDML`), `hill`, `MARGIN`, `CREDIT`, `SEED`.

### Provided before Exercise 1

Load the CSV and truth; `features(df)` returns the 9-column numpy matrix (`tenure_months, orders_12m,
days_since_last_order, avg_order_value, email_open_rate, mobile_share, loyalty_member, channel_social,
channel_referral`). Print the install rate, mean spend by group, and a truth table with three averages of
the true effect: all customers (ATE), installers (ATT), non-installers (ATC). Define a helper
`overlap_truth(rows)` = $\sum e_i(1 - e_i)\tau_i / \sum e_i(1 - e_i)$ over the given rows, from the
`true_*` columns (used by checkpoints; its meaning is explained in Exercise 5).

### Exercise 1 · Naive and regression estimates against the truth (7 minutes)

- **Predict.** Installers spent far more than non-installers in the next 26 weeks (printed above); the true
  average effect for all customers is in the truth table. A regression of spend on install *and* all nine
  covariates: will its estimate be close to the naive difference, about halfway, or close to the truth?
  Write one.
- **Function.** `baseline_estimates(y, t, W) -> pd.DataFrame` with index `["naive", "regression"]` and
  columns `estimate, se, ci_low, ci_high`. Naive: $\bar Y_1 - \bar Y_0$ with
  $\mathrm{se} = \sqrt{s_1^2/n_1 + s_0^2/n_0}$. Regression: the coefficient on `t` from
  `sm.OLS(y, sm.add_constant(np.column_stack([t, W]))).fit(cov_type="HC1")` and its robust standard error.
  Both intervals: estimate ± 1.96 se (95% normal-approximation confidence interval).
- **Checkpoint 1** (first checkpoint; no model fit before it, OLS takes milliseconds). Exact on a toy input
  without noise (for example $y = 2 + 3t + 5w$: regression estimate 3, naive known in closed form). On the
  app data: both 95% intervals lie entirely above the true ATT, the largest of the three true averages
  ("too high under every definition of the effect"). Prototype FULL: naive $65.64 [$61.15, $70.14],
  regression $32.11 [$28.76, $35.47], ATT $17.64; QUICK: $67.32 [$60.60, $74.04] and $32.49 [$27.62,
  $37.36]. Across 40 seeds of the same data-generating process (LightGBM prototype), both intervals
  excluded the overlap-weighted truth (about $14.3, below the ATT) every time, at n = 20,000 and
  n = 10,000.
- **Run (provided, after the checkpoint).** A plot of install rate and mean spend against
  `days_since_last_order` (binned, two panels): both fall steeply over the first few weeks, then flatten.
  The sentence before it: "Look at how both curves bend in the first 60 days: a straight line in days
  cannot follow that."
- **Explain.** Compare with your prediction. The regression adjusts for all nine covariates and still
  lands about twice the truth. Which feature of the data does a linear control miss? Point to the plot.
  <details> Customers who bought in the last few weeks are both far more likely to install (they were
  shopping during the launch weeks) and far heavier spenders. A linear term in days fits the average slope
  and leaves the sharp early drop in the residual, where it still moves with install, so part of the
  confounding is credited to the app. The right transformation, $\exp(-\text{days}/30)$, and the product of
  orders and order value would fix this regression, but you would have to know the shape. DML learns the
  shape from the data instead.</details>

### Exercise 2 · Cross-fitted predictions (8 minutes)

- **Intro (markdown).** DML needs two predictions per customer from the covariates alone: $\hat q(x)$,
  expected spend, and $\hat e(x)$, the probability of installing (the propensity score). Each customer's
  predictions must come from a model that never saw that customer: split the customers into $K = 5$ folds;
  for each fold, fit on the other four and predict the held-out one. This is **cross-fitting**
  (Chernozhukov et al. 2018).
- **Predict.** A model that memorizes its training data (one nearest neighbor) predicts spend. For the
  customers it was trained on, what share of predictions will equal the actual spend exactly: about 0%,
  about 50% or 100%? And for customers in the held-out fold?
- **Function.** `cross_fit(model, X, target, folds) -> np.ndarray` of length `len(target)`: for each
  `(train, test)` in `folds`, `fitted = clone(model).fit(X[train], target[train])`; store
  `fitted.predict_proba(X[test])[:, 1]` if the model has `predict_proba`, else `fitted.predict(X[test])`.
- **Checkpoint 2** (toy inputs only, before the big fit). (a) `KNeighborsRegressor(n_neighbors=1)` on 40
  toy rows with distinct targets: no out-of-fold prediction equals its own row's target (in-sample, every one
  would; the failure message says so). (b) With `LinearRegression()`, the result equals
  `cross_val_predict(model, X, target, cv=folds)` to 1e-10. (c) With `KNeighborsClassifier(n_neighbors=5)`,
  it equals `cross_val_predict(..., method="predict_proba")[:, 1]` to 1e-10 and `checks.probability` passes.
  No NaN; the right length. (Use k-nearest neighbors, not `LogisticRegression`, in the toy check: with
  SciPy 1.16.3 and scikit-learn 1.6.1 it printed a `DeprecationWarning` about L-BFGS-B options in the
  prototype.)
- **Run (provided): the two prediction models.** `y_hat = cross_fit(model_y, W, y, folds)`,
  `t_hat = cross_fit(model_t, W, t, folds)` with the boosted models above; print the out-of-fold R² for
  spend and the AUC for install. Prototype FULL: R² 0.47, AUC 0.78; 3.0 s on the laptop.
- **Explain.** Compare with your prediction. Why would in-sample predictions be a problem for the next
  step, which works with what the predictions leave over? <details> A flexible model fitted and evaluated
  on the same customers reproduces part of their noise, so the leftover (residual) is too small and no
  longer behaves like the leftover for a new customer. Out-of-fold predictions are honest: every customer
  is predicted by a model that did not see them, and with five folds each model still learns from 80% of the
  data. Exercise 3's provided run shows what goes wrong without it.</details>

### Exercise 3 · Residual on residual, with a standard error (7 minutes)

- **Intro (markdown).** The two equations for $\hat\theta$ and its standard error (Notation). The idea is
  the one behind controls in linear regression (the coefficient on $T$ equals the slope of the $Y$ residual
  on the $T$ residual after both are regressed on the controls), with the linear regressions replaced by
  cross-fitted machine learning. $\tilde T$ is the part of installing the covariates cannot predict: the
  "as good as random" variation the effect is learned from.
- **Predict.** Will the 95% confidence interval contain the true effect this estimator targets (printed
  by the checkpoint)? Yes or no. And how wide will it be: about ±$1, ±$3 or ±$10?
- **Function.** `dml_estimate(y, t, y_hat, t_hat) -> dict` with `theta, se, ci_low, ci_high` from the
  formulas above (HC0 standard error, ±1.96).
- **Checkpoint 3.** Exact on a toy input (residuals chosen so $\hat\theta$ and its standard error are
  known in closed form). On the app data: `overlap_truth(all rows in use)` lies inside the 95% CI. The
  message names it "the true overlap-weighted average effect (Exercise 5 explains which customers it
  describes)". Prototype FULL: $13.51 [$10.51, $16.50] against $14.28; QUICK: $14.22 [$9.79, $18.65]
  against $14.34. Coverage measured on the prototype data: 29 of 30 seeds with histogram boosting at
  n = 20,000; with LightGBM 38 of 40 at n = 20,000 and 39 of 40 at n = 10,000 (mean error +$0.51 and +$0.23,
  error SD close to the mean standard error, $1.59 vs $1.55 and $2.22 vs $2.21). The Technical Expert must
  confirm that the committed seed passes in both modes.
- **Run (provided): what skipping cross-fitting does.** Fit an overfitting booster
  (`max_leaf_nodes=255, min_samples_leaf=2, max_iter=600, learning_rate=0.1`) on *all* customers, predict
  the same customers, and call `dml_estimate`. Print the estimate, its standard error next to the
  cross-fitted one, and the mean of $|T - \hat e|$. Prototype FULL (histogram boosting): $12.13 ± $6.95
  (standard error 4.5 times the cross-fitted one), mean $|T - \hat e|$ = 0.003; QUICK: $291 ± $96. With
  LightGBM over 8 seeds at FULL the in-sample estimates ranged from $14.60 to $45.10 with standard errors
  3.7 to 5.7 times the cross-fitted ones. The point estimate swings by seed, so no checkpoint on it; if the
  Technical Expert wants one, assert only that the in-sample standard error is at least twice the
  cross-fitted one, after measuring it over seeds. Slowest cell of the lab: 9.9 s on the laptop (FULL).
- **Explain.** Compare with your prediction and with Exercise 1: name the two things DML did that the
  regression did not. Then look at the overfit run: why did its standard error grow? <details> DML learned
  the curved relationships between the covariates and both install and spend, and it removed them from both
  before relating the leftovers. A model that memorizes the data predicts every install almost exactly
  (mean $|T - \hat e|$ about 0.003), so almost no "as good as random" variation is left and the estimate
  divides noise by almost nothing. With cross-fitting, small errors in $\hat q$ and $\hat e$ do not shift
  $\hat\theta$ to first order (its estimating equation is Neyman-orthogonal), which is why imperfect machine
  learning predictions still give a valid interval (Chernozhukov et al. 2018; EconML's DML user
  guide).</details>

### Exercise 4 · LinearDML reproduces your estimate (5 minutes)

- **Intro (markdown).** EconML's `LinearDML` does Exercises 2 and 3 in one call. Our covariates are its
  controls `W`; `X=None` asks for one constant effect; `discrete_treatment=True` makes it treat install as a
  category, so `model_t` must be a classifier; `cv=folds` gives it our folds (its default is `cv=2`).
- **Predict.** With the same folds, the same two models and the same seed, will EconML's estimate equal
  yours to the cent, differ by a few cents, or differ by dollars?
- **Function.** `fit_linear_dml(y, t, W, folds, model_y, model_t, seed)` returning
  `LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=folds,
  random_state=seed).fit(y, t, X=None, W=W)`.
- **Checkpoint 4.** `float(est.ate())` equals `mine["theta"]` within rel 1e-6 (prototype: 2e-15 in both
  modes); the bounds of `est.ate_interval(alpha=0.05)` equal yours within rel 1e-4 (EconML's default
  inference for `LinearDML` is `StatsModelsInference(cov_type="HC1")`, whose variance is HC0 times
  $n/(n-1)$: prototype bounds differed by $0.00007); the overlap truth lies inside EconML's interval. Then
  print `est.intercept__inference().summary_frame(alpha=0.05)` and the per-fold `est.nuisance_scores_y`
  (out-of-fold R²) and `est.nuisance_scores_t` (out-of-fold accuracy). Do not print `est.summary()`: with
  `X=None` it begins "Coefficient Results: X is None, please call intercept_inference to learn the
  constant!" (prototype), which confuses more than it helps. Prototype: 3.1 s.
- **Explain.** Did EconML match to the cent? Name one setting that would make the library's answer differ
  from yours on the same data. <details> The default `cv=2` (different folds), different nuisance models,
  or `mc_iters`, which repeats cross-fitting with new folds and averages the predictions. The interval is
  slightly wider only because of the $n/(n-1)$ correction. In practice use the library (tested, with more
  inference options and heterogeneous effects through `X`), and keep the two-step version in your head to
  check it.</details>

### Exercise 5 · Overlap: whose effect is it? (6 minutes)

- **Intro (markdown).** Overlap means every kind of customer had some chance both to install and not to:
  $0 < e(x) < 1$. Provided plot: histograms of $\hat e$ for installers and non-installers on one axis
  (shape or hatching, not color alone), with the sentence "Look at the left edge: customers almost sure not
  to install have no installers like them to compare with." Then the result from Notation in one line: the
  residual-on-residual estimate is a weighted average of the customers' effects with weights
  $e(1 - e)$ (overlap weights, Li, Morgan & Zaslavsky 2018). **Trimming** drops customers with $\hat e$
  outside $[\ell, h]$; Crump et al. (2009) give $[0.1, 0.9]$ as a rule of thumb for the average effect.
- **Predict.** The cell above printed the share of customers with $\hat e$ below 0.05 (prototype 5.5%).
  What share of the total weight $\hat e(1 - \hat e)$ do they carry: about the same, much less (around 1%),
  or much more?
- **Function.** `trimmed_estimate(y, t, y_hat, t_hat, low, high) -> dict`: keep rows with
  `low <= t_hat <= high`, call `dml_estimate` on them, and add `share_kept` and `weight_share_dropped`
  ($\sum_{\text{dropped}} \hat e(1 - \hat e) / \sum_{\text{all}} \hat e(1 - \hat e)$). The predictions come
  from Exercise 2; nothing is refitted.
- **Checkpoint 5** (with `low, high = 0.05, 0.95`). `share_kept` equals the mask's mean; `theta` equals
  `dml_estimate` on the masked rows (rel 1e-12); `weight_share_dropped < 1 - share_kept`; and
  `overlap_truth(kept rows)` lies inside the 95% CI. Prototype FULL: kept 94.5%, weight dropped 1.2%,
  $13.85 [$10.85, $16.86] against $14.36; QUICK: 92.8%, 1.4%, $14.41 [$9.87, $18.95] against $14.46.
- **Run (provided).** A profile table: means of `orders_12m, days_since_last_order, loyalty_member,
  mobile_share, avg_order_value` for all customers, installers, non-installers, and weighted by
  $\hat e(1 - \hat e)$ ("whose effect the estimate describes"), plus trimmed estimates at $[0.1, 0.9]$ and
  $[0.2, 0.8]$ next to their truths. Prototype FULL: $[0.1, 0.9]$ kept 79%, $14.15 vs truth $14.61;
  $[0.2, 0.8]$ kept 50%, $15.84 vs truth $15.23. Weighted profile: 87.5 days since last order vs 102.0 for
  all customers, mobile share 0.57 vs 0.52, loyalty 0.35 vs 0.31.
- **Explain.** Compare with your prediction. Trimming dropped about one customer in eighteen but moved the
  estimate by well under a dollar: why so little? Then read the profile: which customers does the estimate
  describe, and which does it say least about? <details> Residual-on-residual regression already gives
  customers who were almost sure to install, or almost sure not to, almost no weight, because their install
  leaves little unexplained variation; trimming them removes little. Estimators that divide by
  $\hat e$, such as the doubly robust average effect in the stretch, react much more to extreme propensity
  scores. The estimate describes customers whose install was in doubt: a little more recent, more mobile and
  more loyal than average. Trimming changes the population, and so the target: at $[0.2, 0.8]$ half the
  customers remain and the true average among them is higher. Which average matters depends on the decision:
  the true averages here are $13.58 (all customers), $17.64 (installers), $11.87 (non-installers) and
  $14.28 (overlap-weighted), prototype values.</details>

### Exercise 6 · Recover an experiment from a confounded sample (7 minutes)

- **Intro (markdown).** A way to test an observational method on real data: take a randomized experiment,
  whose simple comparison is the truth, spoil the randomization with a known rule, and see whether the
  method recovers the experiment (the design of LaLonde 1986). Data: Hillstrom's Womens e-mail versus No
  e-mail customers (Module 6), outcome `visit` (visited the website within two weeks). Benchmark: the
  difference in visit rates on all of them. The rule: keep every No E-Mail customer; keep a Womens E-Mail
  customer with probability 0.9 if `recency <= 3` (months) or `history >= 250` (dollars last year), and
  with probability 0.2 otherwise. E-mailed customers in the sample are then mostly recent or big spenders,
  who visit more anyway.
- **Predict.** On the spoiled sample, will the naive difference in visit rates be above, below or about
  the benchmark (printed above, about 4.5 percentage points)? Will DML's 95% interval contain the benchmark?
- **Function.** `confound(df, seed) -> pd.DataFrame` applying the rule with
  `np.random.default_rng(seed).uniform(size=len(df))`, returning the kept rows with a fresh index.
- **Checkpoint 6** (before the fit). Every control row is kept; among treated customers who are recent or
  big spenders the kept share is within 0.90 ± 0.02, among the others within 0.20 ± 0.02 (binomial standard
  errors 0.003 and 0.004 at FULL, with 11,357 and 10,030 treated customers in the two groups; about 0.004
  and 0.006 at QUICK). Then a provided cell runs `baseline_estimates` and `fit_linear_dml` (same boosted
  models and 5 stratified folds) on the sample, and a second labelled checkpoint asserts: the naive 95% CI
  excludes the benchmark, and the `LinearDML` 95% CI contains it. Prototype FULL: benchmark 4.52 pp (se 0.32), sample of
  33,526 customers (36.4% e-mailed), naive 6.18 pp [5.39, 6.96], regression 4.76 pp, `LinearDML` 4.65 pp
  [3.83, 5.46]; QUICK: benchmark 4.60, naive 6.02 [4.92, 7.12], `LinearDML` 4.69 [3.54, 5.84]. Over 20
  spoiling seeds the `LinearDML` interval contained the benchmark 20 times and the naive interval excluded it
  20 times, on all of Hillstrom and on a random half. 4.3 s on the laptop (FULL).
- **Explain.** Compare with your predictions. Linear regression also came close here (4.76 pp). Why did it
  work on Hillstrom but not on the app data? And what did we know here that you never know at work?
  <details> Visiting rises smoothly with past spend and recency, and the rule selects on those two observed
  columns, so linear controls capture most of the selection; in the app data the confounding ran through a
  sharp curve. More important: we built the selection, so we *know* it depends only on columns the model
  sees. In real observational data that is an assumption. LaLonde (1986) found that many econometric
  methods failed to reproduce an experiment's results, and in 15 advertising experiments at Facebook,
  matching and regression methods overestimated the experimental effects (Gordon et al. 2019). One more
  reason this benchmark works: the e-mail's effect on visits is about the same for the customers the rule
  favors and for the rest (4.55 vs 4.50 pp, prototype), so reweighting does not change the target. A rule
  that selected on `womens` (effect 7.3 vs 1.1 pp in the prototype) would make the sample's overlap-weighted
  effect differ from the benchmark even with perfect adjustment.</details>

### Decision · Pay customers to install? (5 minutes)

Provided cells, in the standard order:

- **Run: what a hidden confounder would do.** Refit $\hat q$ and $\hat e$ without `mobile_share` (as if the
  CRM extract had no web analytics) and, for contrast, without `email_open_rate`; print each estimate and
  its shift from yours. Prototype FULL: without `mobile_share` $19.90 [$16.80, $23.00], +$6.39, now
  excluding the truth; without `email_open_rate` +$0.13. One sentence before the output: "`mobile_share`
  drives both install and spend; `email_open_rate` drives only install." Across seeds the shift without
  `mobile_share` was at least +$5.06 at FULL and +$4.62 at QUICK (40 LightGBM seeds each) and +$5.34 with
  histogram boosting (30 seeds); a labelled check may assert a shift above +$3. 5.8 s on the laptop.
- **Number:** incremental margin per persuaded install over 26 weeks, $m\,\hat\theta$, with its 95% CI.
  Prototype FULL: 0.30 × $13.51 = $4.05 ($3.15 to $4.95); QUICK $4.27 ($2.94 to $5.59). Also the break-even
  effect, credit ÷ margin = $8.33 of spend, and how far the estimate is above it.
- **Rule** (stated before the result is read): offer the $2.50 credit only if the lower end of the 95% CI of
  incremental margin per persuaded install is above $2.50. Assumption, printed with the rule: every customer
  who installs after the offer is persuaded (customers who did not install in the launch window would not
  install in the next 26 weeks without the credit), so the credit is the cost per persuaded install. If
  some would install anyway, the cost per persuaded install is the credit divided by the persuaded share,
  which only a randomized test of the credit can measure (Module 6).
- **Recommendation:** the learner's sentence: offer or not, the margin per persuaded install and its range,
  and what would change it. The cell computes the hidden-confounder comparison (your estimate minus the
  `mobile_share` shift, against the break-even) and prints whether it would flip the decision. Prototype
  FULL: "a confounder as strong as `mobile_share`, pushing in the same direction and still unobserved,
  would lower the estimate by about $6.4 to about $7.1, below the $8.33 break-even: the decision would
  flip." Prototype QUICK: $14.22 − $5.50 = $8.72, just above the break-even, so it would not; the cell must
  not hard-code either sentence (a decision near its threshold, `briefs/_lab-standard.md`). The learner names the
  confounder they fear most (for example: purchase intent at launch, when customers about to buy install the
  app to do it; or phone use, if the extract lacks it). The cell prints the mode ("QUICK run: treat this as a
  rough answer").
- **Then the truth** (separate cell): the true effect on the customers a credit that doubles the odds of
  installing would persuade ($13.66 prototype, so a true margin of $4.10 per persuaded install), next to the
  true non-installer average ($11.87) and the effect on persuaded customers for larger incentives ($12.58 at
  ten times the odds). The credit really pays in this synthetic world; the lesson is how close one hidden
  variable comes to overturning it, and that bigger incentives reach customers who gain less.

## Stretch (optional)

Nothing later depends on these. Prototype times with histogram boosting, FULL, on the laptop.

1. **Who gains most?** `CausalForestDML(model_y=..., model_t=..., discrete_treatment=True, cv=folds,
   n_estimators=200, min_samples_leaf=20, random_state=SEED)` with `X` = `orders_12m,
   days_since_last_order, avg_order_value, loyalty_member, mobile_share` and `W` = the rest. Compare
   `effect(X)` with `true_effect`: Spearman correlation and the mean by quartile of `true_baseline`.
   Prototype: 5.9 s; Spearman 0.52; quartile means $9.1, $10.8, $13.3, $19.7 against true $8.7, $9.9,
   $12.1, $23.6. Builds on Module 11, now with a learned propensity.
2. **A different average.** `LinearDRLearner(model_propensity=..., model_regression=..., cv=folds,
   min_propensity=0.01, random_state=SEED).fit(y, t, X=None, W=W)` estimates the average effect for *all*
   customers (doubly robust). Prototype: $11.45 [$7.88, $15.01] against the true ATE $13.58; 4.4 s. Try
   other `min_propensity` values and watch the interval: this estimator divides by $\hat e$.
3. **How strong would a hidden confounder have to be?** EconML 0.17's `LinearDML.robustness_value(
   null_hypothesis=8.33)` and `sensitivity_interval(alpha=0.05, c_y=..., c_t=...)`, which its docs say are
   based on Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis. Prototype: robustness value 0.011 against the
   break-even $8.33 and 0.050 against 0. The docs describe `c_y` and `c_t` only as "the level of
   confounding" from 0 to 1; read the paper's definitions before quoting a number, and compare with the
   `mobile_share` experiment above.

## Expected run time (estimates)

Laptop prototype, FULL, scikit-learn limited to 2 threads (Colab has 2 CPUs). **No Colab or Docker time
exists.** Colab may well be slower per thread; the Technical Expert measures it.

| Step | Prototype laptop (s) |
|---|---|
| Load data, Exercise 1, toy checks | < 0.1 |
| Cross-fitted predictions (10 boosted fits) | 3.0 |
| Overfit demonstration (slowest cell) | 9.9 |
| Exercise 4 `LinearDML` | 3.1 |
| Hidden-confounder demonstration (10 fits) | 5.8 |
| Exercise 6 (Hillstrom from cache, `LinearDML`) | 4.3 |
| **Core total** | **about 26** (QUICK about 27) |
| Stretch 1–3 | about 15 |

Against the targets (install ≤ 3 min, any cell ≤ 4 min, total compute ≤ 15 min): compute has a wide
margin even if Colab is several times slower. Install: Colab lacks `econml` 0.17.0 (and its dependency
`sparse`); scikit-learn 1.6.1, statsmodels 0.15.0, matplotlib 3.10.0, shap and lightgbm 4.6.0 are
preinstalled (`environment/colab-constraints.txt`). Spike S2 installed pymc-marketing, econml and nutpie
together in 31 s, so econml alone should take less (estimate). The first run also downloads Hillstrom
(about 4 MB) unless it is cached.

## Known pitfalls

- `LinearDML(cv=2)` is the default; pass `cv=folds` or the comparison with the hand version fails.
- `discrete_treatment=True` needs a classifier for `model_t`; a classifier with `discrete_treatment=False`
  raises "Cannot use a classifier as a first stage model when the target is continuous!" (EconML source).
- Never put post-install variables (app sessions, spend in the launch weeks) in `W`: the app changes them.
  Never let the `true_*` columns into `W`.
- Compare $\hat\theta$ with the overlap-weighted truth, not the ATE: on the prototype data the 95% CI
  contained the ATE in only 34 of 40 seeds at n = 20,000 (37 of 40 at n = 10,000).
- `HistGradientBoosting*(early_stopping='auto')` turns on above 10,000 rows; set it to `False`.
- Spend is heavy-tailed (61% zeros, maximum above $3,000 in the prototype); intervals are wide; use
  `visit` for Hillstrom.
- Hillstrom spells the zip code "Surburban"; reuse Module 11's `features()` for its columns.
- Keep the profile and the decision in dollars over 26 weeks; the effect is not per year.

## APIs verified (and how)

All against the installed packages in `.venv/lib/python3.13/site-packages` on 2026-10-10, then by running
every call in the prototype (econml 0.17.0, scikit-learn 1.6.1, statsmodels 0.15.0, scipy 1.16.3, numpy
2.1.3, pandas 2.2.3, lightgbm 4.6.0).

- `econml/dml/dml.py`: `LinearDML.__init__(*, model_y='auto', model_t='auto', featurizer=None, ...,
  discrete_treatment=False, categories='auto', cv=2, mc_iters=None, mc_agg='mean', random_state=None, ...)`;
  `fit(Y, T, *, X=None, W=None, ..., cache_values=False, inference='auto')`. With `X=None` the final stage
  regresses the outcome residual on the treatment residual with no other term (`_FinalWrapper._combine`).
  `_FirstStageWrapper.predict` returns `predict_proba(...)[:, 1:]` for a discrete treatment.
  `sensitivity_summary`, `sensitivity_interval`, `robustness_value` are defined on `DML` (lines 633–739).
- `econml/_ortho_learner.py`: an integer `cv` becomes `StratifiedKFold` for a discrete treatment with
  `shuffle=True` and the estimator's `random_state`; an iterable of `(train, test)` splits is used as given.
- `econml/_cate_estimator.py`: `ate(X=None, *, T0=0, T1=1)` and `ate_interval(X=None, *, T0=0, T1=1,
  alpha=0.05)`; `StatsModelsCateEstimatorMixin` maps `inference='auto'` to `StatsModelsInference`, and
  `econml/inference/_inference.py` gives it `cov_type='HC1'`; `econml/sklearn_extensions/linear_model.py`
  applies $n/(n - k)$ for HC1. Measured: EconML's standard error = hand HC0 × 1.0000250 = $\sqrt{20000/19999}$.
- `intercept__inference().summary_frame(alpha=0.05)` works; `ate_inference()` returns a
  `PopulationSummaryResults` without `summary_frame` (prototype).
- `econml.dr.LinearDRLearner(*, model_propensity='auto', model_regression='auto', ..., min_propensity=1e-06,
  cv=2, random_state=None, ...)`; `CausalForestDML` as in Module 11's brief.
- scikit-learn: `HistGradientBoostingRegressor/Classifier(learning_rate=0.1, max_iter=100,
  max_leaf_nodes=31, min_samples_leaf=20, early_stopping='auto', random_state=None, ...)` (defaults);
  `cross_val_predict(estimator, X, y=None, *, ..., cv=None, ..., method='predict')`;
  `StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)`; `KNeighborsClassifier` has
  `predict_proba`.
- statsmodels: `OLS.fit(method='pinv', cov_type='nonrobust', ...)` accepts `cov_type='HC1'`.
- LightGBM 4.6.0 (checked, not used): `LGBMModel.__init__` defaults (`n_estimators=100,
  learning_rate=0.1, num_leaves=31, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
  random_state=None, n_jobs=None`); importing it on this macOS laptop failed with "Library not loaded:
  @rpath/libomp.dylib".

## Open items for the Technical Expert

- Build `src/mktstats/synth/app_install.py` to the spec, register it in `EXTENSIONS`, add a test that the
  truth's estimands equal the weighted means of the `true_*` columns, measure coverage of Checkpoints 3 and 5
  and of the Hillstrom check over at least 30 seeds, and write the measured tolerances into
  `truth_app_install.json`. Replace every prototype number in the notebook's markdown with recorded ones.
- `modules.m15.notebooks[0].install`: `[econml, scikit-learn, statsmodels, matplotlib]` (drop `lightgbm`
  from the list, see Model fits; econml still pulls it in as a dependency, at the pinned 4.6.0). `deps`: add `data/synthetic/app_install_customers.csv` and
  `data/synthetic/truth_app_install.json` (not `truth.json`).
- `scripts/gen_tables.py` lists the previous module ("Module 14 … and its lab") under "Before you start"
  for every module. For the optional extension modules this is wrong; consider rendering
  `modules.mNN.prerequisites` for modules with `optional: true`. (Also, every module page renders
  "Python. Python." because `language_why` starts with the language; the core modules do the same.)
- Decide whether the overfit and hidden-confounder demonstrations get labelled checkpoints (only on
  quantities that were stable across seeds: the standard-error ratio, and a shift above +$3).
- Confirm that histogram boosting with `random_state` gives identical results with the folds passed to
  both the hand version and EconML on Colab and in Docker (it did on the laptop, to 2e-15).

## References (verified)

Each was opened on the date shown; the note says what the page supports here. Proposed additions to
`references.qmd` are in the Academic Director's report to the lead.

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*,
  21(1), C1–C68. <https://doi.org/10.1111/ectj.12097>; abstract at <https://arxiv.org/abs/1608.00060>\
  *Verified (2026-10-10)*: the Crossref record (journal, volume, issue, pages) and the arXiv abstract: an
  orthogonal score built from auxiliary and main machine-learning predictions, K-fold sample splitting
  called cross-fitting, estimates that converge at the $1/\sqrt{n}$ rate and are approximately unbiased and
  normal, and valid confidence intervals. Already in `references.qmd` (Module 11).
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931–954.
  <https://ideas.repec.org/a/ecm/emetrp/v56y1988i4p931-54.html> (DOI 10.2307/1912705)\
  *Verified (2026-10-10)*: the abstract on RePEc: estimates based on a wrong parametric form of the
  nonparametric part are generally inconsistent; inserting nonparametric regression estimates gives a
  root-N-consistent, asymptotically normal estimator of the slope. The ancestor of partialling out.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via propensity score weighting.
  *Journal of the American Statistical Association*, 113(521), 390–400.
  <https://doi.org/10.1080/01621459.2016.1260466>; abstract at <https://arxiv.org/abs/1609.07494>\
  *Verified (2026-10-10)*: the Crossref record and the arXiv abstract (the arXiv listing says the preprint
  was withdrawn by the author; the journal article stands): overlap weights, proportional to each unit's
  probability of being assigned to the other group, are bounded and minimize the asymptotic variance of the
  weighted average treatment effect. The $e(1 - e)$ form of DML's implicit weights is derived in this
  brief, not quoted from the paper.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in
  estimation of average treatment effects. *Biometrika*, 96(1), 187–199.
  <https://academic.oup.com/biomet/article-abstract/96/1/187/235329> (DOI 10.1093/biomet/asn055)\
  *Verified (2026-10-10)*: the abstract: a good approximation to their optimal selection rule, for a wide
  range of distributions, is to discard units with estimated propensity scores outside [0.1, 0.9].
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental
  data. *American Economic Review*, 76(4), 604–620.
  <https://ideas.repec.org/a/aea/aecrev/v76y1986i4p604-20.html>\
  *Verified (2026-10-10)*: the abstract: compares nonexperimental estimates with the results of a
  randomized employment and training experiment; many econometric procedures fail to replicate the
  experimental results. The design of Exercise 6.
- Gordon, B. R., Zettelmeyer, F., Bhargava, N., & Chapsky, D. (2019). A comparison of approaches to
  advertising measurement: Evidence from big field experiments at Facebook. *Marketing Science*, 38(2),
  193–225. <https://doi.org/10.1287/mksc.2018.1135>\
  *Verified (2026-10-09)* in `references.qmd`; the publisher's page returned 403 on 2026-10-10. The
  Marketing Science Institute summary
  <https://msi.org/working-papers/a-comparison-of-approaches-to-advertising-measurement-evidence-from-big-field-experiments-at-facebook-2>,
  *verified (2026-10-10)*: 15 US ad experiments at Facebook; matching and regression-based methods
  overestimated effectiveness relative to the randomized experiments.
- Chernozhukov, V., Cinelli, C., Newey, W., Sharma, A., & Syrgkanis, V. Long story short: Omitted variable
  bias in causal machine learning. <https://arxiv.org/abs/2112.13398> (the arXiv page lists the journal
  reference *The Review of Economics and Statistics*, 2026, DOI 10.1162/REST.a.1705)\
  *Verified (2026-10-10)*, abstract only: bounds on omitted-variable bias for causal parameters, including
  under partial linearity, from plausibility judgments on the maximum explanatory power of omitted
  variables, with inference that can use machine-learning estimators. Stretch 3 only; the definitions of
  EconML's `c_y` and `c_t` were not checked against the paper.
- **EconML** (PyWhy): `LinearDML` reference
  <https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html> and the double machine learning
  user guide <https://www.pywhy.org/EconML/spec/estimation/dml.html>\
  *Verified (2026-10-10)*: both pages are labelled econml 0.16.0; the lab pins 0.17.0, whose source was
  checked (APIs verified, above). The guide writes the model as $Y = \theta(X) T + g(X, W) + \varepsilon$,
  residualizes $Y$ and $T$ on $(X, W)$ and regresses residual on residual, requires cross-fitted nuisance
  estimates, and attributes the robustness to Neyman orthogonality; `LinearDML`'s final stage is
  "implemented as a statsmodel regression"; `cv` accepts an iterable of (train, test) splits;
  `sensitivity_interval`, `sensitivity_summary` and `robustness_value` are "Based on [Chernozhukov2022]".
- **LightGBM** 4.6.0: Python package README
  <https://raw.githubusercontent.com/microsoft/LightGBM/v4.6.0/python-package/README.rst> and the
  `LGBMRegressor` reference <https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html>\
  *Verified (2026-10-10)*: macOS wheels need the OpenMP library, installed with `brew install libomp`
  (README); constructor defaults and the meaning of `random_state` and `n_jobs` (reference). Supports the
  choice of scikit-learn's boosting instead.
- **Hillstrom e-mail experiment**. Hillstrom, K. (2008, March 20). *The MineThatData e-mail analytics and
  data mining challenge.* <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>\
  *Verified (2026-10-10)*: 64,000 customers who last purchased within twelve months, randomly split into
  Mens e-mail, Womens e-mail and no e-mail; visit, conversion and spend over the following two weeks; the
  covariates used here.
