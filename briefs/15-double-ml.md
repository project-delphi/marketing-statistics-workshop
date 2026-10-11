# Lab brief · Module 15 · Causal effects without an experiment: double machine learning

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab was built. Numbers marked **prototype** come from a throwaway script on the Academic Director's laptop (Apple Silicon, scikit-learn limited to 2 threads with `OMP_NUM_THREADS=2`), not from the lab, Docker or Colab. Reviewed by the Pedagogy Expert on 2026-10-10 against `briefs/_lab-standard.md`; the changes and their reasons are under "Pedagogy review" at the end. Numbers marked **review re-implementation** come from the Pedagogy Expert's own re-implementation of the spec below (same laptop type, 2 threads; scratchpad, not the repo), used only to compute likely wrong answers for the failure messages. **Built and run** (2026-10-10): worked, FULL runs of the whole notebook are recorded on the laptop and in Docker (`runs/2026-10-10-*-python-15-double-ml.json`); not yet on Colab. Numbers marked **lab, FULL** are the notebook's printed output at full settings on the committed data; **30 seeds** are the Technical Expert's sweeps in `data/synthetic/truth_app_install.json` `tolerances`. Where the build re-measured a prototype number, the lab's number replaces it; see "Build reconciliation" at the end. |
| Notebook | `labs/src/15-double-ml.py` → `labs/python/15-double-ml.ipynb` |
| Lab slot | 55 min (`modules.m15.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (8 + 8 + 9 + 7 + 8; limit 40) + decision 5 + slack 5 = 55. Each exercise's minutes include reading the provided cells inside it. Minutes are estimates until a pilot. |
| Day | Extensions (`d6`), optional. Comes after Modules 6 and 11, which it builds on. The generated "Before you start" list now renders `modules.m15.prerequisites` (Modules 6 and 11), not Module 14. |

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
with. The derivation goes in the briefing and on the module page; the notebook states the result in one
sentence immediately before Checkpoint 3 and restates the weight at the start of Part C, where Exercise 4
uses it (Pedagogy review, item 2).

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
- Size: 20,000 rows, 14 columns, 1.44 MB as CSV (committed file, 1,439,191 bytes).

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
order of draws was the Technical Expert's choice; the build kept the prototype's, so the committed values
equal the prototype's, see Build reconciliation):

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

**Prototype values** (seed 2035, n = 20,000). The build kept the prototype's order of draws, so the
committed data and `truth_app_install.json` reproduce every value below (lab, FULL; truth file):

| Quantity | Prototype |
|---|---|
| install rate | 29.3% |
| mean spend, installers / non-installers | $97.54 / $31.90 (61% of customers spend $0) |
| `ate` / `att` / `atc` | $13.58 / $17.64 / $11.87 |
| `effect_overlap` | $14.28 |
| `effect_persuaded`, odds × 1.25, 2, 10 | $14.07, $13.66, $12.58 (shares persuaded 3.7%, 12.2%, 42.5%); the truth file stores 13.655 for odds × 2, which the lab prints as $13.65 |
| `naive_expected` | $65.73 |
| true propensity: 1st / 50th / 99th percentile | 0.036 / 0.232 / 0.895 (3.3% below 0.05; 0.09% above 0.95) |

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic app-install customers (the retailer's story; observational, known truth) | `repo_file("synthetic/app_install_customers.csv")`, `repo_file("synthetic/truth_app_install.json")` | Exercises 1–5 and the decision |
| Hillstrom e-mail experiment (real, randomized: Mens e-mail, Womens e-mail, No e-mail; two-week outcomes) | `mktstats.data.load_hillstrom()` (downloaded, sha256-checked, cached; not mirrored) | Exercise 5, second half (provided cells that call the learner's functions): Womens e-mail versus No e-mail, outcome `visit` |

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
  Exercise 2's Explain), reused by Exercises 3 and 4 and the Decision; the overfit demonstration (2 fits on
  all rows, after Exercise 3); `LinearDML` in Exercise 5 refits the same 10 on purpose (the check is that
  the library reproduces your numbers); one `LinearDML` on the Hillstrom sample (Exercise 5, provided); the
  hidden-confounder demonstration (10 fits, Decision).
- **QUICK:** a seeded random 10,000 of the 20,000 customers and a seeded half of Hillstrom; same models.
  Truth targets are recomputed on the rows in use from the `true_*` columns, so checkpoints stay exact.
  In the prototype QUICK did not save time (boosting time depends on iterations more than rows), so its
  value here is a second, noisier sample, not speed. The header says what QUICK changes, and the provided
  fits after Exercise 2 say it again (`_lab-standard.md` §6): "With QUICK on, you have half the customers:
  intervals are wider and will not match the module page, and a decision near its threshold can flip. That
  is a lesson about sample size, not a bug." Draw the subsample with
  `np.random.default_rng(SEED).choice(len(customers), 10_000, replace=False)`, not `DataFrame.sample`
  (Open items: the structure test).
- `SEED = 2035`, visible in the first code cell; `warnings.filterwarnings` only for messages the brief
  names (see pitfalls), each with a comment.

## Terms the notebook defines, where they are first used

The Notation above is for the Technical Expert. The notebook defines each term in one plain sentence in the
cell that first uses it, never in an opening glossary, and the briefing has defined them first (Pedagogy
review, "What the briefing carries"). Most names are familiar to this audience; the new ideas are
cross-fitting, partialling out, overlap weights and trimming, one per exercise. Suggested wording:

| Where | Term | One sentence for the notebook |
|---|---|---|
| Part A, first cell | observational data; self-selection | Nobody assigned the app: customers chose to install it, and the customers who chose it differ from those who did not. |
| Part A, first cell | confounder (common cause) | A variable that moves both install and spend, such as having bought in the last few weeks; it makes installers spend more whether or not the app does anything. |
| Provided before Exercise 1 | true effect | How much more a customer spends over 26 weeks with the app than without it. Known here only because we simulated the data. |
| Provided before Exercise 1 | ATE, ATT, ATC | The average true effect over all customers, over installers, and over non-installers. |
| Exercise 1 | naive difference | Mean spend of installers minus mean spend of non-installers. |
| Exercise 1 | linear controls | The covariates entered as straight-line terms in a regression of spend on install. |
| Exercise 1 | 95% normal-approximation confidence interval | The estimate ± 1.96 standard errors. |
| Exercise 1 | robust (HC1) standard error | A standard error that does not assume every customer's spend is equally noisy (heavy spenders are noisier). |
| Part B, first cell | double machine learning (DML) | Estimate the effect after removing, from both spend and install, what machine-learning models predict from the covariates. |
| Part B, first cell | no unobserved confounding | DML's assumption: every common cause of install and spend is among the covariates. The data cannot test it. |
| Exercise 2 | propensity score | ê(x), the predicted probability that a customer like x installs. |
| Exercise 2 | fold; cross-fitting; out-of-fold prediction | Split the customers into 5 groups (folds) and predict each fold with a model fitted on the other four, so no customer is predicted by a model that saw them. |
| After Exercise 2 | R²; AUC | R²: the share of the variation in spend that the out-of-fold predictions explain. AUC: the chance that a random installer gets a higher predicted probability than a random non-installer (0.5 is guessing, 1 is perfect). |
| Exercise 3 | residual | What the prediction leaves over: Ỹ = spend − q̂(x), T̃ = install − ê(x). |
| Exercise 3 | partialling out (residual on residual) | Regress the spend residual on the install residual; the slope is the effect. A linear regression with controls does the same with straight lines. |
| Before Checkpoint 3 | overlap-weighted average effect | When the effect differs between customers, this estimate averages their effects with weight e(1 − e): most for customers whose install was a toss-up, almost none for those nearly sure either way. |
| After Exercise 3 | in-sample prediction | A prediction for a customer the model was fitted on. |
| Part C, first cell | overlap | Every kind of customer had some chance both to install and not to: 0 < e(x) < 1. |
| Part C, first cell | overlap weight | e(1 − e): 0.25 at e = 0.5, near 0 when e is near 0 or 1. |
| Exercise 4 | trimming | Dropping customers whose ê lies outside [low, high] before estimating. It changes which customers the estimate describes. |
| Exercise 5 | LinearDML; controls `W`; effect modifiers `X` | EconML's version of Exercises 2 and 3. `W`: covariates to adjust for. `X`: covariates the effect may vary with; none here, so one effect. |
| Exercise 5 | nuisance model | EconML's name for the two prediction models: needed for the estimate, not of interest themselves. |
| Exercise 5, Hillstrom | benchmark | The difference in visit rates in the randomized experiment: the right answer for this test. |
| Exercise 5, Hillstrom | percentage point (pp) | A difference between two percentages: 10% to 14.5% is 4.5 pp. |
| Decision | persuaded install | An install that happens because of the credit and would not have happened without it. |
| Decision | break-even effect | The extra spend per persuaded install at which the margin pays back the credit: credit ÷ margin. |

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Why the obvious answers are wrong | *Run: data, truth table and a labeled data check (provided)* | — |
| | 1 · Naive and regression estimates against the truth | 8 |
| | *Run: install rate and spend against recency (provided plot)* | — |
| B · Double machine learning by hand | 2 · Cross-fitted predictions | 8 |
| | *Run: the two prediction models (provided)* | — |
| | 3 · Residual on residual, with a standard error | 9 |
| | *Run: what skipping cross-fitting does (provided, labeled check)* | — |
| C · Whose effect is it? | *Run: the overlap plot (provided)* | — |
| | 4 · Overlap and trimming | 7 |
| | *Run: whose effect the estimate describes (provided profile)* | — |
| D · The library, checked on a real experiment | 5 · LinearDML: your estimate, then a confounded experiment | 8 |
| | *Run: spoil the Hillstrom experiment and estimate (provided, with the learner's functions)* | — |
| **Exercises** | | **40** |
| Decision | Pay customers to install? (opens with the provided hidden-confounder run) | 5 |

Each exercise's minutes include reading the provided cells listed inside it (Part A's opening cells count in
Exercise 1, Part C's overlap plot in Exercise 4); their compute is seconds. Minute marks for the room, after
5 minutes to open and install: Exercise 1 ends at minute 13, Exercise 2 at 21, Exercise 3 at 30, Exercise 4
at 37, Exercise 5 at 45; Decision 45–50; slack 50–55. Minutes are estimates until a pilot.

Every exercise runs Predict → Task → stub → folded solution → checkpoint → its provided cells → Explain with
its `<details>`, except where an exercise below says the Explain comes first. Every Predict ends with "Write
your guess in the cell below." Every checkpoint failure message has the four parts of `_lab-standard.md` §5
(what failed with the numbers, the likely cause, what to check, and `workshop.use_reference(N)`); the lists
below give the likely wrong answers, computed on the toys by the Pedagogy Expert (numpy 2.1.3, statsmodels
0.15.0, econml 0.17.0, scikit-learn 1.6.1), and what each message says about them. A labeled check of
provided code says that it checks the lab, not the learner's code, and ends "tell the instructor"; checks
re-raise, so Run all stops there.

Names used throughout: `customers`, `truth`, `y`, `t`, `W`, `folds`, `model_y`, `model_t`, `y_hat`,
`t_hat`, `mine` (the learner's DML result), `est` (the fitted `LinearDML` on the app data), `hill` (Hillstrom,
Womens e-mail and No e-mail rows), `spoiled` (the confounded Hillstrom sample), `MARGIN`, `CREDIT`, `SEED`.
Learner functions: `baseline_estimates`, `cross_fit`, `dml_estimate`, `trimmed_estimate`, `fit_linear_dml`.
Provided helpers: `features`, `overlap_truth`, `confound`.

### Provided before Exercise 1

Part A's first cell defines observational data and a confounder (Terms) and says what the lab does: try the
obvious estimates, then double machine learning, against a truth we know because we simulated the
customers. Load the CSV and truth; `features(df)` returns the 9-column numpy matrix (`tenure_months,
orders_12m, days_since_last_order, avg_order_value, email_open_rate, mobile_share, loyalty_member,
channel_social, channel_referral`). Print the install rate, mean spend by group, and a truth table with three
averages of the true effect: all customers (ATE), installers (ATT), non-installers (ATC), each with its
one-line definition. Define a helper `overlap_truth(rows)` = $\sum e_i(1 - e_i)\tau_i / \sum e_i(1 - e_i)$
over the given rows, from the `true_*` columns (used by checkpoints; Exercise 3 says what it is, Exercise 4
explores it).

Then the lab's first checkpoint, a labeled check of the data, `with workshop.checkpoint(label="data"):`
the number of rows (20,000 FULL, 10,000 QUICK), the 14 columns, `features(customers)` has 9 columns and
the column names it selects include no `true_*` column, and `truth["estimands"]` has `ate`, `att`, `atc`
and `effect_overlap`. It comes before any model fit, so a broken install or a stale data file shows up in the
first minutes. Message: "The app data did not load as expected: {what differs}. Rerun the install cell and
then this cell; if it still fails, tell the instructor."

### Exercise 1 · Naive and regression estimates against the truth (8 minutes)

- **Predict.** Installers spent far more than non-installers in the next 26 weeks (printed above), and the
  true averages in the truth table lie between about $12 and $18. A regression of spend on install *and*
  all nine covariates: will its estimate for install be above $50, between $25 and $50, or below $25?
  (Lab, FULL: naive $65.64; regression $32.11, as in the prototype; prototype QUICK $32.49. The three options are three beliefs:
  controls fix nothing, half, or nearly all of the bias. If the committed data put the regression within $3
  of a boundary, move the boundaries.)
- **Function.** `baseline_estimates(y, t, W) -> pd.DataFrame` with index `["naive", "regression"]` and
  columns `estimate, se, ci_low, ci_high`. Naive: $\bar Y_1 - \bar Y_0$ with
  $\mathrm{se} = \sqrt{s_1^2/n_1 + s_0^2/n_0}$. Regression: the coefficient on `t` from
  `sm.OLS(y, sm.add_constant(np.column_stack([t, W]))).fit(cov_type="HC1")` and its robust standard error.
  Both intervals: estimate ± 1.96 se (95% normal-approximation confidence interval). The task cell gives
  the `sm.OLS` call: the lesson is the comparison, not statsmodels.
- **Checkpoint 1** (the first numbered checkpoint; OLS takes milliseconds).
  1. Toy, exact (1e-9): `t = [0, 0, 0, 1, 1, 1]`, `w = [0, 1, 2, 1, 2, 3]`, `y = 2 + 3t + 5w =
     [2, 7, 12, 10, 15, 20]`. Naive 8 (the effect 3 plus 5 times the gap of 1 in mean `w`), its standard
     error √(50/3) ≈ 4.082; regression 3, with standard error 0 (the toy has no noise).
  2. App data: both rows equal a reference computed inside the checkpoint (the naive formulas in numpy, the
     regression with the same statsmodels call) to rel 1e-9. This is where the standard errors are checked.
  3. App data: both 95% intervals lie entirely above the true ATT, the largest of the three true averages
     ("too high under every definition of the effect"). Lab, FULL: naive $65.64 [$61.14, $70.14],
     regression $32.11 [$28.76, $35.47], ATT $17.64; prototype QUICK: $67.32 [$60.60, $74.04] and $32.49
     [$27.62, $37.36]. 30 seeds (histogram boosting): both intervals lay above the true ATT in 30 of 30 at
     FULL and at QUICK; the regression estimate ranged from $28.18 to $37.03 at FULL. (The draft quoted 40
     LightGBM prototype seeds here; see Build reconciliation.)
- **Failure messages** (each ends with `workshop.use_reference(1)`):
  - Regression 2 on the toy: "you took the constant; `sm.add_constant` puts it first, so the coefficient on
    `t` is `params[1]`." Regression 5: "you took the coefficient on `w`; `t` is the first column after the
    constant, `params[1]`." Regression 3.29: "the regression has no constant; wrap the columns in
    `sm.add_constant(...)`." Regression 8, the same as the naive difference: "the regression left out the
    covariates; pass `np.column_stack([t, W])`."
  - Naive standard error 3.333 on the toy: "use the sample variance, `np.var(x, ddof=1)` (pandas `.var()`
    does this by default)."
  - App data, naive standard error differs from the reference: "use the unpooled formula
    √(s₁²/n₁ + s₀²/n₀), not a pooled variance." Regression standard error differs: "fit with
    `cov_type="HC1"`; the default assumes every customer's spend is equally noisy." An interval differs: "a
    95% interval is estimate ± 1.96 × se."
  - Wrong index or columns: "return index `['naive', 'regression']` and columns `estimate, se, ci_low,
    ci_high`, in that order."
  - Part 3 fails after parts 1 and 2 passed: "your function matches the reference, so this compares the
    data with the truth, not your code: tell the instructor."
- **Run (provided, after the checkpoint).** A plot of install rate and mean spend against
  `days_since_last_order` (binned, two panels sharing the x-axis; axes in days, % and dollars): both fall
  steeply over the first few weeks, then flatten. The sentence before it: "Look at how both curves bend in
  the first 60 days: a straight line in days cannot follow that."
- **Explain.** Compare with your guess. The regression adjusts for all nine covariates and still lands far
  above every true average. Which feature of the data does a straight-line control miss? Point to the plot.
  <details> Customers who bought in the last few weeks are both far more likely to install (they were
  shopping during the launch weeks) and far heavier spenders. A linear term in days fits the average slope
  and leaves the sharp early drop in the residual, where it still moves with install, so part of the
  confounding is credited to the app. The right transformation, $\exp(-\text{days}/30)$, and the product of
  orders and order value would fix this regression, but you would have to know the shape. DML learns the
  shape from the data instead.</details>

### Exercise 2 · Cross-fitted predictions (8 minutes)

- **Part B's first cell (markdown).** Defines double machine learning and its assumption, no unobserved
  confounding, in one sentence each (Terms), and says the assumption cannot be tested from the data; the
  Decision returns to it.
- **Intro (markdown).** DML needs two predictions per customer from the covariates alone: $\hat q(x)$,
  expected spend, and $\hat e(x)$, the probability of installing (the propensity score). Each customer's
  predictions must come from a model that never saw that customer: split the customers into $K = 5$ folds;
  for each fold, fit on the other four and predict the held-out one. This is **cross-fitting**
  (Chernozhukov et al. 2018).
- **Predict.** The checkpoint below fits a model that memorizes its training data (one nearest neighbor) to
  40 made-up customers whose spends are all different. For the customers it was trained on, what share of
  its predictions will equal the customer's own spend exactly: about 0%, about 50% or 100%? And for each
  customer predicted out of fold, by a model fitted on the other folds?
  (The question is about the toy on purpose: 61% of the app customers spend $0, so on the app data an
  out-of-fold prediction of $0 would often be exactly right, and the answer would turn on that detail.)
- **Function.** `cross_fit(model, X, target, folds) -> np.ndarray` of length `len(target)`: for each
  `(train, test)` in `folds`, `fitted = clone(model).fit(X[train], target[train])`; store
  `fitted.predict_proba(X[test])[:, 1]` if the model has `predict_proba`, else `fitted.predict(X[test])`.
- **Checkpoint 2** (toy inputs only, before the big fit). Toy folds: `KFold(5, shuffle=True,
  random_state=SEED)` for (a) and (b), `StratifiedKFold` on the toy labels for (c). (a)
  `KNeighborsRegressor(n_neighbors=1)` on 40 toy rows with distinct targets: no out-of-fold prediction equals
  its own row's target. The checkpoint prints the answer to the Predict: "In sample: 40 of 40 predictions
  equal the customer's own spend. Out of fold: 0 of 40." (b) With `LinearRegression()`, the result equals
  `cross_val_predict(model, X, target, cv=folds)` to 1e-10. (c) With `KNeighborsClassifier(n_neighbors=5)`,
  it equals `cross_val_predict(..., method="predict_proba")[:, 1]` to 1e-10 and `checks.probability` passes.
  No NaN; the right length. (Use k-nearest neighbors, not `LogisticRegression`, in the toy check: with
  SciPy 1.16.3 and scikit-learn 1.6.1 it printed a `DeprecationWarning` about L-BFGS-B options in the
  prototype.)
- **Failure messages** (each ends with `workshop.use_reference(2)`):
  - (a) fails, "k of 40 out-of-fold predictions equal the customer's own spend": "those customers were in
    the model's training data. For each fold fit on `X[train]` and predict `X[test]`; a model fitted on all
    rows and scored on them gets 40 of 40."
  - (b) differs from `cross_val_predict`: "put each fold's predictions at its test positions,
    `out[test] = ...`; appending fold after fold puts them in the wrong order. Fit a fresh
    `clone(model)` on each fold's training rows."
  - (c) returns only 0s and 1s: "for a classifier, store `predict_proba(X[test])[:, 1]`, the probability
    of class 1, not `predict`'s 0/1 labels." (c) returns 1 minus the expected values: "column 1 is the
    probability of installing; column 0 is the probability of not installing." Checkpoint 2 must catch
    both: later they would show only as a wrong $\hat\theta$ at Checkpoint 3 ($9.02 and $4.88 against a
    correct $13.10 in the review re-implementation), whose message could not point back here.
  - NaN or the wrong length: "return one prediction per row of `X`, filling every fold's test rows."
- **Explain** (before the provided fits; it uses the toy). Compare with your guess. Exercise 3 works with
  what the predictions leave over: spend minus predicted spend. What would be left over from in-sample
  predictions like the toy's, and what could an estimate learn from it? <details> Nothing would be left:
  every leftover would be 0, so there would be no variation to learn the effect from. A flexible model
  fitted and evaluated on the same customers reproduces part of their noise, so its leftovers are too small
  and no longer behave like the leftovers for a new customer. Out-of-fold predictions are honest: every
  customer is predicted by a model that did not see them, and with five folds each model still learns from
  80% of the data. The provided run after Exercise 3 shows the same thing on the app data.</details>
- **Run (provided): the two prediction models.** `y_hat = cross_fit(model_y, W, y, folds)`,
  `t_hat = cross_fit(model_t, W, t, folds)` with the boosted models above; print the out-of-fold R² for
  spend and the AUC for install, and the QUICK sentence (Model fits). The sentence before the cell: "This
  cell fits 10 models, two per fold. While it runs, read Exercise 3's Predict." Prototype FULL: R² 0.47,
  AUC 0.78; 3.0 s on the laptop. Lab, FULL: R² 0.47, AUC 0.78.

### Exercise 3 · Residual on residual, with a standard error (9 minutes)

- **Intro (markdown).** The two equations for $\hat\theta$ and its standard error (Notation). The idea is
  the one behind controls in linear regression (the coefficient on $T$ equals the slope of the $Y$ residual
  on the $T$ residual after both are regressed on the controls), with the linear regressions replaced by
  cross-fitted machine learning. $\tilde T$ is the part of installing the covariates cannot predict: the
  "as good as random" variation the effect is learned from.
- **Predict.** Exercise 1's regression gave about $32 (printed above), and the true averages lie between
  about $12 and $18. Where will the DML estimate land: near the regression, about halfway, or among the
  true averages? And the half-width of its 95% interval: about $1, $3 or $10? (Lab, FULL
  $13.51 ± $3.00; prototype QUICK $14.22 ± $4.43.)
- **Function.** `dml_estimate(y, t, y_hat, t_hat) -> dict` with `theta, se, ci_low, ci_high` from the
  formulas above (HC0 standard error, ±1.96). The reference is about five lines.
- **Before the checkpoint (markdown; the only place before Part C that names the estimand).** "The
  checkpoint compares your interval with the true *overlap-weighted* average effect, not the ATE. When the
  effect differs between customers, residual-on-residual regression averages their effects with weight
  e(1 − e): largest (0.25) for customers whose install was a toss-up, near 0 for those almost sure to
  install or not. Exercise 4 shows who those customers are." No derivation: it is in the briefing and on the
  module page.
- **Checkpoint 3.**
  1. Toy, exact (1e-12): `t = [1, 0, 1, 0]`, `t_hat = [0.5, 0.5, 0.25, 0.25]`, `y = [0, 0, 3, 2]`,
     `y_hat = [1, 2, 1, 3]`, so $\tilde T$ = (0.5, −0.5, 0.75, −0.25) and $\tilde Y$ = (−1, −2, 2, −1):
     $\hat\theta = 2.25 / 1.125 = 2$, standard error √10 / 3 ≈ 1.054, 95% CI [−0.066, 4.066]. $\tilde T$
     has a nonzero mean on purpose, so a regression with an intercept gives a different answer.
  2. App data: `overlap_truth(all rows in use)` lies inside the 95% CI. Lab, FULL: $13.51 [$10.51,
     $16.50] against $14.28; prototype QUICK: $14.22 [$9.79, $18.65] against $14.34. 30 seeds (histogram
     boosting, `tolerances.checkpoint_3`): covered 27 of 30 at FULL and 29 of 30 at QUICK (mean error
     −$0.20 and −$0.02; error SD close to the mean standard error, $1.66 vs $1.56 and $2.28 vs $2.22). The
     committed seed passes in both modes (run records). The draft's LightGBM figures (38 of 40 and 39 of
     40) are prototype results, not the build's.
- **Failure messages** (each ends with `workshop.use_reference(3)`; toy values):
  - θ̂ 0.5: "you used `t`, not the install residual `t - t_hat`." θ̂ 1.556: "you used `y`, not the spend
    residual `y - y_hat`." θ̂ 2.353: "the formula has no intercept: θ̂ = Σ T̃Ỹ / Σ T̃². `np.polyfit` or a
    regression with a constant adds one." θ̂ −2: "only one residual has its sign flipped; both are actual
    minus predicted."
  - Standard error 1.277: "that is the classical standard error, which assumes equal noise for every
    customer; use √(Σ T̃² (Ỹ − θ̂T̃)²) / Σ T̃²." 1.217: "that is HC1 (statsmodels' `cov_type="HC1"`), which
    multiplies by √(n / (n − 1)); the formula here is HC0. On 20,000 customers the two differ by 0.003%,
    on 4 toy rows by 15%." 1.111: "that is the squared standard error; take the square root of
    Σ T̃² (Ỹ − θ̂T̃)², then divide by Σ T̃²." 1.118: "divide by Σ T̃², not by its square root."
    1.678: "inside the standard error the residual is Ỹ − θ̂T̃, not Ỹ."
  - Interval [0.266, 3.734]: "a 95% interval uses 1.96, not 1.645."
  - Part 2 fails after the toy passed: "your formula is right, so this compares the provided predictions
    with the truth: tell the instructor." The message names the target as in the sentence above and prints
    the ATE beside it, so nobody concludes that the estimate missed "the" effect.
- **Explain** (right after the checkpoint, while the result is fresh; the provided run follows it). Compare
  with your guess and with Exercise 1's regression, which used the same nine covariates. What did DML do
  with them that the regression did not? Point to the recency plot from Exercise 1. <details> DML learned
  the curved relationships between the covariates and both install and spend, such as the sharp drop over
  the first weeks since the last order, and removed them from both before relating the leftovers. The
  regression could remove only straight-line relationships, so part of the confounding stayed in its
  estimate. What would change the result: a common cause that is not among the covariates; the Decision
  shows one.</details>
- **Run (provided): what skipping cross-fitting does.** Fit an overfitting booster
  (`max_leaf_nodes=255, min_samples_leaf=2, max_iter=600, learning_rate=0.1`) on *all* customers, predict
  the same customers, and call `dml_estimate`. Print the estimate, its standard error next to the
  cross-fitted one and their ratio, and the mean of $|T - \hat e|$ in sample and out of fold, with a
  computed sentence ("the standard error is 4.5 times the cross-fitted one; the point estimate changes from
  seed to seed"), never a number fixed in Markdown. The sentence before the cell: "Look at the standard
  error, and at how close the in-sample predictions of install come to the actual 0s and 1s." Lab,
  FULL (as the prototype): $12.13 ± $6.95 (standard error 4.5 times the cross-fitted one), mean
  $|T - \hat e|$ = 0.003 in sample against 0.319 out of fold; prototype QUICK: $291 ± $96. 30 seeds
  (`tolerances.no_cross_fitting`): the ratio was 4.0 to 6.5 at FULL and 34 to 50 at QUICK, never below
  2. (The draft's LightGBM range over 8 seeds, ratios 3.7 to 5.7, is a prototype result.) Review
  re-implementation, FULL: $29.83 ± $7.03, ratio 4.7, mean $|T - \hat e|$ 0.003 in sample against 0.327
  out of fold. No check on the point estimate. A labeled check, `with workshop.checkpoint(label="no
  cross-fitting"):`, asserts that the in-sample standard error is at least twice the cross-fitted one,
  only after the Technical Expert has measured the ratio over at least 30 seeds in both modes (Pedagogy
  review, item 3); its message: "This compares two provided fits, not your code: without cross-fitting the
  standard error should be several times larger, and it was only {ratio}. Tell the instructor." Then a
  folded `<details><summary>Why the standard error grows</summary>`: a model that memorizes the data
  predicts every install almost exactly (mean $|T - \hat e|$ about 0.003), so almost no "as good as random"
  variation is left and the estimate divides noise by almost nothing. With cross-fitting, small errors in
  $\hat q$ and $\hat e$ do not shift $\hat\theta$ to first order (the briefing's name for this is
  Neyman orthogonality), which is why imperfect machine-learning predictions still give a valid interval
  (Chernozhukov et al. 2018; EconML's DML user guide). Slowest cell of the lab: 9.9 s on the laptop (FULL).

### Exercise 4 · Overlap and trimming (7 minutes)

- **Part C's first cell (markdown).** Overlap means every kind of customer had some chance both to install
  and not to: $0 < e(x) < 1$. Then the reminder, in one line, of what Checkpoint 3 said: the
  residual-on-residual estimate is a weighted average of the customers' effects with weights $e(1 - e)$
  (overlap weights, Li, Morgan & Zaslavsky 2018): 0.25 at $e = 0.5$, near 0 at either end. **Trimming**
  drops customers with $\hat e$ outside $[\ell, h]$; Crump et al. (2009) give $[0.1, 0.9]$ as a rule of
  thumb for the average effect.
- **Run (provided): the overlap plot.** Histograms of $\hat e$ for installers and non-installers on one
  axis (shape or hatching, not color alone; x-axis "predicted probability of installing"), with the
  sentence before it: "Look at the left edge: customers almost sure not to install have no installers like
  them to compare with." The cell prints the share of customers with $\hat e$ outside $[0.05, 0.95]$
  (prototype and lab, FULL: 5.5%).
- **Predict.** The cell above printed the share of customers with $\hat e$ outside $[0.05, 0.95]$. What
  share of the total weight $\hat e(1 - \hat e)$ do they carry: about the same, much less (around 1%), or
  much more?
- **Function.** `trimmed_estimate(y, t, y_hat, t_hat, low, high) -> dict`: keep rows with
  `low <= t_hat <= high`, call `dml_estimate` on them, and add `share_kept` and `weight_share_dropped`
  ($\sum_{\text{dropped}} \hat e(1 - \hat e) / \sum_{\text{all}} \hat e(1 - \hat e)$). The predictions come
  from Exercise 2; nothing is refitted. The constants cell: `LOW, HIGH = 0.05, 0.95  # drops only
  customers nearly sure either way: their weight is under 0.0475, a fifth of the largest`.
- **Checkpoint 4.**
  1. Toy, exact (1e-12): `t_hat = [0.02, 0.10, 0.30, 0.50, 0.70, 0.97]`, `t = [0, 0, 1, 0, 1, 1]`,
     `y = [0, 1, 4, 2, 5, 9]`, `y_hat = [0.5, 1, 2, 2, 3, 6]`, `low, high = 0.05, 0.95`: rows 2 to 5 kept,
     `share_kept` 4/6, `weight_share_dropped` = 0.0487 / 0.8087 ≈ 0.0602, `theta` ≈ 2.381 (equal to
     `dml_estimate` on the kept rows).
  2. App data (with `LOW, HIGH`): `share_kept` equals the mask's mean; `theta` equals `dml_estimate` on
     the masked rows (rel 1e-12); `weight_share_dropped < 1 - share_kept`; and `overlap_truth(kept rows)`
     lies inside the 95% CI. Lab, FULL (as the prototype): kept 94.5%, weight dropped 1.2%, $13.85 [$10.85,
     $16.86] against $14.36; prototype QUICK: 92.8%, 1.4%, $14.41 [$9.87, $18.95] against $14.46. Review
     re-implementation, FULL: kept 96.3%, weight dropped 0.8%. 30 seeds (`tolerances.checkpoint_4`): the
     kept customers' truth inside the interval in 27 at FULL and 29 at QUICK; less weight than customers
     dropped in 30 and 30.
- **Failure messages** (each ends with `workshop.use_reference(4)`; toy values):
  - `theta` 2.496: "that is the estimate on all six rows; subset `y`, `t`, `y_hat` and `t_hat` with the
    same mask before calling `dml_estimate`."
  - `weight_share_dropped` 0.333: "that is the share of customers dropped; weight each customer by
    `t_hat * (1 - t_hat)`." 0.940: "that is the share of weight kept; return the share dropped." NaN or 0:
    "the weights come from `t_hat`, not `t`: `t * (1 - t)` is 0 for every customer."
  - `share_kept` 4: "return a share, `mask.mean()`, not a count."
  - Part 2's truth check fails after part 1 passed: "your function is right, so this compares the provided
    predictions with the truth: tell the instructor."
- **Run (provided): whose effect the estimate describes.** A profile table: means of `orders_12m,
  days_since_last_order, loyalty_member, mobile_share, avg_order_value` for all customers, installers,
  non-installers, and weighted by $\hat e(1 - \hat e)$ ("whose effect the estimate describes"), plus
  trimmed estimates at $[0.1, 0.9]$ and $[0.2, 0.8]$ next to their truths. Sentence before it: "Look at the
  last column: how does the customer the estimate describes differ from the average customer?" Prototype
  FULL: $[0.1, 0.9]$ kept 79%, $14.15 vs truth $14.61; $[0.2, 0.8]$ kept 50%, $15.84 vs truth $15.23.
  Weighted profile: 87.5 days since last order vs 102.0 for all customers, mobile share 0.57 vs 0.52,
  loyalty 0.35 vs 0.31.
- **Explain.** Compare with your guess. Trimming dropped about one customer in twenty (the share printed
  above; prototype 5.5%) but moved the estimate by well under a dollar: why so little? Then read the
  profile: which customers does the estimate describe, and which does it say least about? <details> Residual-on-residual regression already gives
  customers who were almost sure to install, or almost sure not to, almost no weight, because their install
  leaves little unexplained variation; trimming them removes little. Estimators that divide by
  $\hat e$, such as the doubly robust average effect in the stretch, react much more to extreme propensity
  scores. The estimate describes customers whose install was in doubt: a little more recent, more mobile and
  more loyal than average. Trimming changes the population, and so the target: at $[0.2, 0.8]$ half the
  customers remain and the true average among them is higher. Which average matters depends on the decision:
  the true averages here are $13.58 (all customers), $17.64 (installers), $11.87 (non-installers) and
  $14.28 (overlap-weighted), prototype values.</details>

### Exercise 5 · LinearDML: your estimate, then a confounded experiment (8 minutes)

- **Intro (markdown).** EconML's `LinearDML` does Exercises 2 and 3 in one call. Our covariates are its
  controls `W`; `X=None` asks for one constant effect; `discrete_treatment=True` makes it treat install as a
  category, so `model_t` must be a classifier; `cv=folds` gives it our folds (its default is `cv=2`). You
  use it twice: on the app data, where it must reproduce your Exercise 3 numbers, and on a real experiment
  that we spoil on purpose.
- **Predict (1 of 2).** With the same folds, the same two models and the same seed, will EconML's estimate
  equal yours to the cent, differ by a few cents, or differ by dollars?
- **Function.** `fit_linear_dml(y, t, W, folds, model_y, model_t, seed)` returning
  `LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=folds,
  random_state=seed).fit(y, t, X=None, W=W)`. The task lists the four settings and what each means; the
  learner writes the call.
- **Checkpoint 5.** `float(est.ate())` equals `mine["theta"]` within rel 1e-6 (prototype: 2e-15 in both
  modes; lab data, FULL, recomputed by the Academic Director: 2.0e-15; 30 seeds: at most 2.3e-15); the bounds of `est.ate_interval(alpha=0.05)` equal yours within rel 1e-4 (EconML's default
  inference for `LinearDML` is `StatsModelsInference(cov_type="HC1")`, whose variance is HC0 times
  $n/(n-1)$: prototype bounds differed by $0.00007); the overlap truth lies inside EconML's interval. Then
  print `est.intercept__inference().summary_frame(alpha=0.05)` and the per-fold `est.nuisance_scores_y`
  (out-of-fold R²) and `est.nuisance_scores_t` (out-of-fold accuracy), one line each. Do not print
  `est.summary()`: with `X=None` it begins "Coefficient Results: X is None, please call
  intercept_inference to learn the constant!" (prototype), which confuses more than it helps. Prototype:
  3.1 s.
- **Failure messages** (each ends with `workshop.use_reference(5)`). The checkpoint catches EconML's own
  exceptions and turns them into these messages, so that no learner reads a bare traceback (econml 0.17.0,
  review re-implementation):
  - `AttributeError: Cannot use a classifier as a first stage model when the target is continuous!`: "set
    `discrete_treatment=True`: install is 0 or 1, and `model_t` is a classifier."
  - `AssertionError: X was not None when fitting, so can't be none for score or effect`, from
    `est.ate()`: "pass the covariates as controls, `W=W`, with `X=None`. `X` is for covariates the effect
    varies with, and then EconML wants one effect per customer."
  - The estimate differs from yours (review re-implementation: $13.56 with the default `cv=2` against
    $13.10 with `cv=folds`): "pass `cv=folds`, the folds your `cross_fit` used; EconML's default is
    `cv=2`." (`cv=5` with `random_state=SEED` also passes: EconML then builds the same stratified,
    shuffled folds, to 4e-14 in the review re-implementation.)
  - Bounds differ: "use `est.ate_interval(alpha=0.05)` for a 95% interval."
  - The truth check fails after the equality passed: as Checkpoint 3, "tell the instructor."
- **Run (provided): a real benchmark.** Markdown first: a way to test an observational method on real
  data is to take a randomized experiment, whose simple comparison is the truth, spoil the randomization
  with a known rule, and see whether the method recovers the experiment (the design of LaLonde 1986). Data:
  Hillstrom's Womens e-mail versus No e-mail customers (Module 6), outcome `visit` (visited the website
  within two weeks). Benchmark: the difference in visit rates on all of them. The rule: keep every No E-Mail
  customer; keep a Womens E-Mail customer with probability 0.9 if `recency <= 3` (months) or
  `history >= 250` (dollars last year), and with probability 0.2 otherwise. E-mailed customers in the sample
  are then mostly recent or big spenders, who visit more anyway.
  Then a provided cell loads Hillstrom (a download on first use, cached), prints the benchmark, and applies
  the provided `confound(df, seed)` (the rule with `np.random.default_rng(seed).uniform(size=len(df))`,
  kept rows with a fresh index); it prints the kept share in each group (control 1.00; treated recent or
  big spenders about 0.90; other treated about 0.20), the sample size and the share e-mailed.
- **Predict (2 of 2).** On the spoiled sample, will the naive difference in visit rates be above, below or
  about the benchmark printed above (about 4.5 percentage points)?
- **Run (provided), then a labeled checkpoint.** `baseline_estimates` and `fit_linear_dml` (the learner's
  functions; same boosted models; 5 stratified folds built on the sample) on `spoiled`; print one table in
  percentage points: benchmark, naive, regression and `LinearDML`, each with its 95% CI. Then
  `with workshop.checkpoint(5, label="5 on Hillstrom"):` asserts the kept shares (every control row kept;
  treated recent or big spenders within 0.90 ± 0.02, the others within 0.20 ± 0.02: binomial standard errors
  0.003 and 0.004 at FULL, with 11,357 and 10,030 treated customers in the two groups, checked on the
  cached file in review; about 0.004 and 0.006 at QUICK), that the naive 95% CI excludes the benchmark, and
  that the `LinearDML` 95% CI contains it. Prototype FULL: benchmark 4.52 pp (se 0.32), sample of 33,526
  customers (36.4% e-mailed), naive 6.18 pp [5.39, 6.96], regression 4.76 pp, `LinearDML` 4.65 pp
  [3.83, 5.46] (benchmark, sample size, share e-mailed and naive estimate reproduced in review with seed
  2035); QUICK: benchmark 4.60, naive 6.02 [4.92, 7.12], `LinearDML` 4.69 [3.54, 5.84]. Lab, FULL: the
  same values (regression 95% CI [3.96, 5.56]). 30 spoiling seeds (`tolerances.hillstrom`): on all of
  Hillstrom the naive interval excluded the benchmark 30 times and `LinearDML`'s contained it 30 times;
  on half of Hillstrom `LinearDML` still contained it 30 times, while the naive interval excluded it 29
  times with the lab's half and 27 times over random halves. (The draft's "20 of 20 on all of Hillstrom
  and on a random half" was the prototype's.) 4.3 s on the laptop (FULL). Message, since both
  functions already passed their checkpoints: "Your functions passed Checkpoints 1 and 5, so this compares
  the provided spoiled sample with the experiment: {which part failed, with the numbers}. Tell the
  instructor."
- **Explain.** Compare with your two guesses. On the spoiled sample the naive comparison missed the
  experiment and `LinearDML` recovered it. What did we know about this sample that you never know at work?
  Point to the rule above. <details> We built the selection, so we *know* it depends only on columns the
  model sees (`recency` and `history`). In real observational data that is an assumption, and nothing in
  the data can confirm it. LaLonde (1986) found that many econometric methods failed to reproduce an
  experiment's results, and in 15 advertising experiments at Facebook, matching and regression methods
  overestimated the experimental effects (Gordon et al. 2019). Linear regression also came close here
  (4.76 pp): visiting rises smoothly with past spend and recency, so linear controls capture most of this
  selection; in the app data the confounding ran through a sharp curve. One more reason this benchmark
  works: the e-mail's effect on visits is about the same for the customers the rule favors and for the
  rest (4.55 vs 4.50 pp, prototype, reproduced in review), so reweighting does not change the target. A
  rule that selected on `womens` (effect 7.3 vs 1.1 pp in the prototype) would make the sample's
  overlap-weighted effect differ from the benchmark even with perfect adjustment. On the library: it matched
  your Exercise 3 numbers because it used your folds and models; with its default `cv=2`, or with
  `mc_iters` (repeated cross-fitting with new folds), it would differ. Its interval is slightly wider only
  because of the $n/(n-1)$ correction. In practice use the library (tested, with more inference options and
  heterogeneous effects through `X`), and keep the two-step version in your head to check it.</details>

### Decision · Pay customers to install? (5 minutes)

The Markdown above the cells states, before any number is printed: the decision; the rule and its economics
(the $2.50 credit, the 30% margin, so the break-even effect is $2.50 ÷ 0.30 = $8.33 of extra spend per
persuaded install); the three assumptions (below); and "Run the cells, write your sentence, then look at
the truth." It defines a persuaded install and the break-even effect (Terms).

Provided cells, in the standard order:

- **Run: what a hidden confounder would do.** Refit $\hat q$ and $\hat e$ without `mobile_share` (as if the
  CRM extract had no web analytics) and, for contrast, without `email_open_rate`; print each estimate and
  its shift from yours. Prototype FULL: without `mobile_share` $19.90 [$16.80, $23.00], +$6.39, now
  excluding the truth; without `email_open_rate` +$0.13 (review re-implementation: +$5.88 and +$0.25). One
  sentence before the output: "`mobile_share` drives both install and spend; `email_open_rate` drives only
  install." Lab, FULL: the same values ($13.64 [$10.64, $16.65] without
  `email_open_rate`). 30 seeds (`tolerances.hidden_confounder`): the shift without `mobile_share` ranged
  from +$4.33 to +$7.20 at FULL and +$4.67 to +$8.20 at QUICK. (The draft's minima, +$5.06 and +$4.62 over
  40 LightGBM seeds and +$5.34 over 30 histogram-boosting seeds, were prototype results.) A labeled check,
  `with workshop.checkpoint(label="hidden confounder"):`, asserts a shift above +$3 (Pedagogy review, item
  3), with the message: "This compares two provided fits, not your code: leaving out `mobile_share`, which
  drives both install and spend, should raise the estimate by several dollars, and it moved it by {shift}.
  Tell the instructor." 5.8 s on the laptop.
- **The decision cell.** It computes every number and chooses every verb from the results (offer or do
  not offer; flips or does not flip), so FULL and QUICK can print different sentences; it prints the mode.
  - **Number:** incremental margin per persuaded install over 26 weeks, $m\,\hat\theta$, with its 95% CI;
    the effect behind it; the break-even effect, credit ÷ margin = $8.33 of spend, and how far the estimate
    is above it; and the scale, per 1,000 persuaded installs. Prototype FULL: 0.30 × $13.51 = $4.05
    ($3.15 to $4.95), and the lab's FULL output is the same; prototype QUICK $4.27 ($2.94 to $5.59).
  - **Rule** (stated before the result is read): offer the $2.50 credit only if the lower end of the 95% CI
    of incremental margin per persuaded install is above $2.50 (the same as: the effect's CI lies above
    $8.33). The largest credit the rule allows is that lower end.
  - **Assumes** (printed with the rule): (1) every common cause of install and spend is in the data;
    (2) every customer who installs after the offer is persuaded (customers who did not install in the
    launch window would not install in the next 26 weeks without the credit), so the credit is the cost per
    persuaded install; if some would install anyway, the cost per persuaded install is the credit divided by
    the persuaded share, which only a randomized test of the credit can measure (Module 6); (3) the
    customers the credit persuades gain about as much as the customers the estimate describes (Exercise 4's
    profile). For a small credit, (3) holds approximately in this synthetic world, because of the form of
    its install model (The incentive, above); in general it is an assumption.
  - **Recommendation and what would change it.** The cell applies the rule, then applies the same rule to
    your interval shifted by the `mobile_share` shift ("a common cause that strong, unobserved and pushing
    the same way"), and also reports where the shifted point estimate falls against the break-even.
    Prototype FULL: the interval $10.51 to $16.50 shifted by $6.39 is $4.12 to $10.11 of spend ($1.24 to
    $3.03 of margin), so the rule would say do not offer, and the point estimate ($7.12) also falls below
    the $8.33 break-even. Prototype QUICK: the shifted interval $4.29 to $13.15 flips the rule as well,
    while the shifted point estimate ($14.22 − $5.50 = $8.72) stays above the break-even. The cell must
    not hard-code either sentence (a decision near its threshold, `briefs/_lab-standard.md`).
  - The printed layout, in the order of `_lab-standard.md` §7 (lab FULL values, which differ from the
    prototype's only in rounding: $5.17 above the break-even, not $5.18, and $4,052 per 1,000; the cell
    fills every number):

    ```text
    Number            : each persuaded install adds $4.05 of margin over 26 weeks (95% CI $3.15 to $4.95):
                        an effect of $13.51 of spend (95% CI $10.51 to $16.50) at a 30% margin.
                        Break-even: $2.50 credit / 30% = $8.33 of extra spend; the estimate is $5.17 above it.
                        Per 1,000 persuaded installs: about $4,052 of margin ($3,153 to $4,951)
                        for $2,500 of credits.
    Rule              : offer the $2.50 credit only if the lower end of the 95% CI of margin per persuaded
                        install is above $2.50 (the same as: the effect's CI lies above $8.33).
    Assumes           : (1) every common cause of install and spend is in the data; (2) everyone who
                        installs after the offer is persuaded by it; (3) they gain about as much as the
                        customers the estimate describes (Exercise 4).
    Recommendation    : offer the credit: the lower end, $3.15, is above $2.50. The largest credit the
                        rule allows is $3.15.
    Hidden confounder : leaving out mobile_share moved the estimate by +$6.39. A common cause that strong,
                        unobserved and pushing the same way, would shift your interval to $4.12 to $10.11
                        of spend ($1.24 to $3.03 of margin): by the rule, do not offer. The point estimate,
                        $7.12, would also fall below the $8.33 break-even.
    What would change it: a common cause missing from the data (above); customers who would install
                        anyway taking the credit (a randomized test of the credit measures that, Module 6);
                        a larger credit, which reaches customers who gain less.
    Mode              : FULL run.
    Your sentence     : ________________________________________________
    ```

  - **The learner's sentence**, from a template in the Markdown under the cell: "(Offer / Do not offer) the
    $2.50 install credit: each persuaded install adds about $____ of margin over 26 weeks (95% CI $____ to
    $____). This assumes every common cause of install and spend is in our data; the one I fear most is
    ____, and if it were as strong as `mobile_share` the answer would be ____." The learner names the
    confounder they fear most (for example: purchase intent at launch, when customers about to buy install
    the app to do it; or phone use, if the extract lacks it).
- **Then the truth** (separate cell, after the sentence): the true effect on the customers a credit that
  doubles the odds of installing would persuade (in words: a customer with a 20% chance would have a 33%
  chance), $13.66 prototype ($13.65 as the lab prints the truth file's 13.655), so a true margin of $4.10 per persuaded install, next to the true non-installer
  average ($11.87) and the effect on persuaded customers for larger incentives ($12.58 at ten times the
  odds). The credit really pays in this synthetic world; the lesson is how close one hidden variable comes
  to overturning it, and that bigger incentives reach customers who gain less.

## Stretch (optional)

Nothing later depends on these. Prototype times with histogram boosting, FULL, on the laptop. The lab's
FULL output reproduces every number below.

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
   "Based on [Chernozhukov2022]" (see Build reconciliation: that key is ambiguous in the installed source;
   cite "Long story short" by title, not by EconML's key). Prototype: robustness value 0.011 against the
   break-even $8.33 and 0.050 against 0. The docs describe `c_y` and `c_t` only as "the level of
   confounding" from 0 to 1; read the paper's definitions before quoting a number, and compare with the
   `mobile_share` experiment above.

## Expected run time (estimates)

Laptop prototype, FULL, scikit-learn limited to 2 threads (Colab has 2 CPUs). **No Colab time exists.**
Colab may well be slower per thread. Measured since (run records, worked mode, FULL, whole notebook,
2026-10-10, both on an Apple Silicon laptop shared with other build jobs): 48.3 s on macOS with two
threads, slowest cell 12.7 s; 52.8 s in the Docker image with `--cpus 2` and two threads, slowest cell
14.7 s (`runs/2026-10-10-macos-arm64-python-15-double-ml.json`,
`runs/2026-10-10-docker-arm64-python-15-double-ml.json`). Neither is a Colab time.

| Step | Prototype laptop (s) |
|---|---|
| Load data, Exercise 1, toy checks | < 0.1 |
| Cross-fitted predictions (10 boosted fits) | 3.0 |
| Overfit demonstration (slowest cell) | 9.9 |
| Exercise 5 `LinearDML` on the app data | 3.1 |
| Exercise 5, Hillstrom (from cache, `LinearDML`) | 4.3 |
| Hidden-confounder demonstration (10 fits, Decision) | 5.8 |
| **Core total** | **about 26** (QUICK about 27) |
| Stretch 1–3 | about 15 |

Against the targets (install ≤ 3 min, any cell ≤ 4 min, total compute ≤ 15 min): compute has a wide
margin even if Colab is several times slower. Install: Colab lacks `econml` 0.17.0 (and its dependency
`sparse`); scikit-learn 1.6.1, statsmodels 0.15.0, matplotlib 3.10.0, shap and lightgbm 4.6.0 are
preinstalled (`environment/colab-constraints.txt`). Spike S2 installed pymc-marketing, econml and nutpie
together in 31 s, so econml alone should take less (estimate). The first run also downloads Hillstrom
(about 4 MB) unless it is cached.

The review re-implementation (same settings, 2 threads, another laptop run, not a run record) took 3.4 s
for the cross-fitted predictions, 11.3 s for the overfit demonstration and 3.0 to 3.5 s per `LinearDML`
fit, in line with the prototype.

## Known pitfalls

- `LinearDML(cv=2)` is the default; pass `cv=folds` or the comparison with the hand version fails.
- `discrete_treatment=True` needs a classifier for `model_t`; a classifier with `discrete_treatment=False`
  raises "Cannot use a classifier as a first stage model when the target is continuous!" (EconML source; an
  `AttributeError` at `fit` with econml 0.17.0, review re-implementation). A regressor as `model_t` with
  `discrete_treatment=True` does not fail: it warns "First stage model has discrete target but model is not
  a classifier!" and runs.
- Covariates passed as `X` instead of `W` fit without complaint, and `est.ate()` then raises
  `AssertionError: X was not None when fitting, so can't be none for score or effect` (review
  re-implementation). Checkpoint 5 turns both errors into messages.
- Never put post-install variables (app sessions, spend in the launch weeks) in `W`: the app changes them.
  Never let the `true_*` columns into `W`.
- Compare $\hat\theta$ with the overlap-weighted truth, not the ATE, because the overlap-weighted
  average is what residual on residual estimates when effects vary (the derivation under "What $\theta$
  is when effects vary"). Coverage does not make this case on these data: over 30 seeds with histogram
  boosting the 95% CI contained the overlap-weighted truth in 27 at FULL and 29 at QUICK, and the ATE in
  27 and 28, because the two averages here ($14.28 and $13.58) differ by less than half a standard error.
  (The draft's "the ATE in only 34 of 40 seeds" came from the LightGBM prototype and is not a result of
  this build; see Build reconciliation.)
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
- Checked again in the Pedagogy review (2026-10-10, same installed versions, by running the calls on a
  re-implementation of the spec): `LinearDML` with `cv=folds` equals the hand estimate (difference below
  1e-13); the default `cv=2` gives a different estimate ($13.56 against $13.10); `cv=5` with
  `random_state=SEED` gives the same folds as `StratifiedKFold(5, shuffle=True, random_state=SEED)`
  (difference 4e-14); the two errors in Known pitfalls, with their exception types. The toy values under
  each checkpoint were computed with numpy and statsmodels 0.15.0.
- LightGBM 4.6.0 (checked, not used): `LGBMModel.__init__` defaults (`n_estimators=100,
  learning_rate=0.1, num_leaves=31, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
  random_state=None, n_jobs=None`); importing it on this macOS laptop failed with "Library not loaded:
  @rpath/libomp.dylib".

## Open items for the Technical Expert

- Build `src/mktstats/synth/app_install.py` to the spec, register it in `EXTENSIONS`, add a test that the
  truth's estimands equal the weighted means of the `true_*` columns, measure coverage of Checkpoints 3 and 4
  and of the Hillstrom check (`5 on Hillstrom`) over at least 30 seeds, and write the measured tolerances into
  `truth_app_install.json`. Replace every prototype number in the notebook's markdown with recorded ones.
- Measure the two labeled demonstration checks over the same seeds, in both modes: the in-sample standard
  error at least twice the cross-fitted one (`no cross-fitting`), and the shift without `mobile_share` above
  +$3 (`hidden confounder`). Write the minima into `truth_app_install.json` `tolerances`. If either fails on
  any seed, drop that check and keep the cell's printed sentence computed from its numbers (Pedagogy review,
  item 3).
- `tests/test_notebooks_structure.py` matches `.fit(` and `sample(` (its `FIT` pattern) in every code cell
  before the first checkpoint, solution cells included. Solution 1 contains `sm.OLS(...).fit(...)`, so the
  labeled `data` check must come before Exercise 1's cells (it does in this brief), and the QUICK subsample
  in the load cell must not call `DataFrame.sample` (Model fits). Every cell that contains `.fit(` also needs
  the comment Module 11 uses ("EconML and scikit-learn draw no progress bar, so there is no
  progressbar=False to pass"), for `test_the_lab_never_downloads_files_through_colab`; that includes the
  solution cells of Exercises 1, 2 and 5.
- Every checkpoint's listed wrong answers get a test (a `mktstats.checks`-style test, or a verify-mode run
  with each wrong function) that expects the listed message, as for Modules 14 and 16.
- `modules.m15.notebooks[0].install`: `[econml, scikit-learn, statsmodels, matplotlib]` (drop `lightgbm`
  from the list, see Model fits; econml still pulls it in as a dependency, at the pinned 4.6.0).
  `deps`: add `data/synthetic/app_install_customers.csv` and `data/synthetic/truth_app_install.json`
  (not `truth.json`).
- `scripts/gen_tables.py` lists the previous module ("Module 14 … and its lab") under "Before you start"
  for every module. For the optional extension modules this is wrong; consider rendering
  `modules.mNN.prerequisites` for modules with `optional: true`. (Also, every module page renders
  "Python. Python." because `language_why` starts with the language; the core modules do the same.)
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
  experimental results. The design of the Hillstrom benchmark in Exercise 5.
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
  *Verified (2026-10-10)*: both pages are labeled econml 0.16.0; the lab pins 0.17.0, whose source was
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

## Pedagogy review (2026-10-10)

Checked against `briefs/_lab-standard.md` §11 before any code exists. The rulings on the two questions the
lead asked come first (items 1 to 3), then the other changes and why.

1. **Six exercises become five.** Six is inside the standard's range, but these six did not fit their
   minutes once reading is counted. Three of them carried ideas new to this audience (cross-fitting,
   partialling out, overlap weights), and five provided demonstrations sat between them (the recency plot,
   the overfit run, the overlap plot and profile, the Hillstrom fits, the hidden-confounder run), whose
   reading the stated minutes left out. Estimated with reading (no pilot): 8 + 8 + 9 + 5 + 8 + 8 = 46
   minutes against 40. The cut keeps every check, number and dataset:
   - The Hillstrom benchmark moves into the `LinearDML` exercise (now Exercise 5) as provided cells that
     call the learner's own `baseline_estimates` and `fit_linear_dml`. `confound(df, seed)` becomes a
     provided helper: its code is a sampling rule in pandas, and the idea it carries (spoil an experiment
     with a known rule) is in the design, which the learner reads and makes a prediction about. The learner
     still writes the EconML call, the skill they will reuse at work, and sees it pay off on real data.
   - Overlap and trimming (now Exercise 4) moves before the library, right after Exercise 3, whose
     checkpoint is the first to name the overlap-weighted target, so "why that average?" is answered at
     once. The library exercise, the shortest, falls at minutes 37 to 45, when the room is most tired, and
     its Hillstrom Explain ("what did we know here that you never know at work?") leads into the
     Decision's hidden-confounder run.
   - Minutes 8 + 8 + 9 + 7 + 8 = 40. Exercise 1 goes from 7 to 8 (the first exercise: a truth table with
     three averages and a plot to read); Exercise 3 from 7 to 9 (two formulas, the estimand sentence and
     the overfit run); Exercise 4 from 6 to 7 (a plot before it, a profile after it).
   - `modules.m15` `summary` and `objectives` now list the steps in the lab's order; includes regenerated.
2. **The estimand goes to the briefing; the notebook states it in one line, twice.** As drafted,
   Checkpoint 3 compared the learner's interval with a target named only by a forward reference ("Exercise 5
   explains"), and Exercise 3's Predict asked whether the interval would contain "the true effect this
   estimator targets", which the learner had not been told. An analyst who has just seen the ATE, ATT and
   ATC in a table will compare with the ATE, and on the prototype data the 95% CI missed the ATE in 6 of 40
   seeds (LightGBM prototype; the build's 30 seeds missed the ATE and the overlap-weighted truth equally
   often, so the case for naming the target rests on the derivation, see Build reconciliation). The derivation and a plot of e(1 − e) against e go in the briefing and on the module page. The
   notebook states the result in plain words in one sentence immediately before Checkpoint 3, and repeats
   the weight at the start of Part C, where Exercise 4's Predict and function use it. Exercise 3's Predict
   now asks where the estimate lands relative to Exercise 1's regression and the printed true averages.
3. **Demonstrations get labeled checks on stable quantities only, and no numbered checkpoint.** Numbered
   checkpoints are for the learner's code, and no demonstration adds any. Two demonstrations get a labeled
   check, on the quantities the Academic Director found stable across seeds, because the text around them
   makes a claim that a change in data or library could silently falsify, and a check is how CI and the
   Colab sweep notice. A check re-raises, so a failure stops Run all; with committed data and pinned
   versions it can fail in class only where CI has already failed. Each message says it checks the lab,
   not the learner, and to tell the instructor.
   - Overfit run: in-sample standard error at least twice the cross-fitted one (prototype 4.5; LightGBM 3.7
     to 5.7 over 8 seeds; about 42 from the prototype's QUICK numbers; review re-implementation 4.7). Never
     the point estimate, which changes from seed to seed. The printed sentence is computed.
   - Hidden-confounder run: a shift above +$3 (minimum +$4.62 QUICK and +$5.06 FULL over 40 LightGBM
     seeds; +$5.34 over 30 histogram-boosting seeds; prototype figures. The build's 30 seeds: minimum
     +$4.33 FULL and +$4.67 QUICK, so the check held on every seed). The Decision's sensitivity line is built on this
     shift, so its sign matters.
   - Overlap plot: no check. Checkpoint 4's `weight_share_dropped < 1 - share_kept` checks the claim it
     illustrates.
   - Both checks go in only after the Technical Expert has measured them over at least 30 seeds in both
     modes; one that fails on any seed is dropped and its sentence stays computed (Open items).
   - The kept-share tolerances of the old Checkpoint 6 now sit in the Hillstrom checkpoint
     (`5 on Hillstrom`), since `confound` is provided.
4. **Predicts with an answer the learner can check.** Exercise 1's options were "close to the naive
   difference, about halfway, or close to the truth", and the prototype's $32 sits between the last two;
   they are now dollar ranges ($25 and $50 as boundaries), to move if the committed value lands near one.
   Exercise 2's Predict asked about spend predicted out of fold: with 61% of customers at $0, an
   out-of-fold prediction of $0 is often exactly right, so the expected answer ("about 0%") would have been
   wrong on the app data. It now asks about the 40-row toy, and the checkpoint prints the answer. Exercise
   5 keeps two Predicts (to the cent; naive against the benchmark) and drops a third (whether the DML
   interval contains the benchmark).
5. **Explains answerable from what the learner has just seen.** Exercise 2's Explain needed the overfit
   run, which came after Exercise 3; it now uses the toy and comes before the provided fits. Exercise 3's
   Explain asked two questions (DML against the regression, and the overfit standard error); it keeps the
   first, right after the checkpoint, and the overfit run gets a "Look at" sentence and its own folded
   explanation, where Neyman orthogonality appears as the briefing's name for the idea, not as a term to
   learn. Exercise 5's Explain asks one question; the point about library settings moves into its
   `<details>`.
6. **A first check before anything else.** A labeled `data` check after loading catches a broken install
   or a stale file in the first minutes. It is also the first checkpoint cell, which the structure test
   needs because Solution 1 calls `.fit(` (Open items).
7. **Failure messages for every checkpoint**, with the likely wrong answers computed on toys built to tell
   them apart: Exercise 1's toy separates the constant, the `w` coefficient, a missing constant and missing
   covariates; Exercise 3's toy has an install residual with a nonzero mean, so an intercept changes the
   answer, and it separates five wrong standard errors; Exercise 4 gains a toy (the draft checked only
   internal consistency on the app data). Checkpoint 5 turns EconML's two likely exceptions into messages
   (checked with econml 0.17.0).
8. **Decision.** The rule, its economics and the assumptions come first, in Markdown. "Assumes" names
   three assumptions; the third (the customers a credit persuades gain about as much as those the estimate
   describes) is new wording for what the incentive spec implies. The hidden-confounder line applies the
   decision's own rule to the shifted interval and still reports the shifted point estimate (see below). A
   scale line (per 1,000 persuaded installs) and the largest credit the rule allows make the
   recommendation something a manager can act on, and match `modules.m15.outcomes` ("a break-even credit
   per install"). The learner fills in a one-sentence template, then sees the truth; "doubles the odds" is
   said in plain numbers.
9. **Terms**: a new section lists where each term is first defined, so that each exercise brings one new
   idea.

**If a pilot runs long.** Exercise 1 is the first to become a provided cell: keep its Predict, the plot
and its Explain, and provide `baseline_estimates` (Module 6's difference in means and a regression this
audience already writes); that frees about 4 minutes. Next, the overfit run moves to the stretch (about 1.5
minutes, and the slowest cell). Keep Exercises 2, 3 and 4 hands-on: they carry the module's three new ideas.

**What the briefing carries, so that the lab can stay short** (for the Academic Director's module page
and slides): the app story as a diagram of common causes (recency, phone use, loyalty → install and
spend); why straight-line controls fail on a sharp curve (the recency plot); partialling out in ordinary
least squares as the bridge to DML; cross-fitting as a picture of five folds; the overlap-weight result with
its derivation and a plot of e(1 − e); overlap and trimming; that no unobserved confounding cannot be
tested, and what a sensitivity analysis does instead; Neyman orthogonality in one sentence (why imperfect
predictions still give a valid interval); LaLonde's design; and why the customers a small credit persuades
are weighted like DML's estimate in this synthetic world (a property of the logistic install model, not a
general law).

**For the Academic Director to reconsider.**

- The sensitivity line. The draft compared the shifted *point estimate* with the break-even, which flips
  the decision at FULL ($7.12) but not at QUICK ($8.72). The rule is about the lower end of the interval,
  so the brief now applies the same rule to the shifted interval: it flips in both modes ($4.12 and $4.29
  against $8.33), and the cell still reports the point estimate. If you prefer the point-estimate
  comparison, the Markdown should say why, because it is a weaker bar than the rule the learner has just
  applied.
- Assumption (3) in the Decision is a new statement of an assumption, not a new fact; check its wording
  against the module page.
- The new order (overlap before the library) and the provided `confound` change the lab's shape, not its
  content. Objective 5 still holds, because the learner's own functions recover the benchmark.

## Build reconciliation (2026-10-10, Academic Director)

Checked against the build: `labs/src/15-double-ml.py`, `src/mktstats/dml.py`,
`src/mktstats/synth/app_install.py`, `data/synthetic/truth_app_install.json` (its 30-seed `tolerances`)
and the two run records. Numbers come from a worked, FULL run of the whole notebook by the Academic
Director on 2026-10-10 (`scripts/test_notebooks.py --full --mode worked`, laptop, two threads; it passed,
with the 5 checkpoints verified on their stubs; not a run record, and its 255 s wall time on a heavily
loaded laptop is not a timing).

- **The data reproduce the prototype.** The generator kept the prototype's order of draws, so the
  committed CSV and truth file give the prototype's install rate, spends, true averages, incentive table
  and propensity quantiles exactly, and the lab's FULL output reproduces the prototype's FULL numbers:
  naive $65.64 [$61.14, $70.14] (the draft had $61.15), regression $32.11, R² 0.47, AUC 0.78, DML $13.51
  [$10.51, $16.50], the overfit run, the trims, the profile, Hillstrom, the hidden-confounder shifts, the
  decision and the stretch. Rounding differences only: the estimate is $5.17 above the break-even (draft
  $5.18), per 1,000 persuaded installs $4,052 ($3,153 to $4,951) (draft $4,050, $3,150 to $4,950), the
  half-width ± $3.00 (draft ± $2.99), and the persuaded effect prints as $13.65 (truth file 13.655; draft
  $13.66). QUICK numbers above remain the prototype's; this reconciliation did not re-run QUICK.
- **Seed sweeps replaced.** Every coverage and range claim above now quotes the build's 30 seeds with
  histogram boosting (`tolerances`): Checkpoint 1 30/30 in both modes; Checkpoint 3 27/30 FULL, 29/30
  QUICK; Checkpoint 4 27/30 and 29/30; Checkpoint 5 equality at most 2.3e-15; the overfit ratio 4.0 to 6.5
  FULL and 34 to 50 QUICK; the hidden-confounder shift +$4.33 to +$7.20 FULL and +$4.67 to +$8.20 QUICK;
  Hillstrom naive excluded 30/30 at FULL (29/30 with the lab's half, 27/30 over random halves at QUICK)
  and `LinearDML` covered 30/30 in every case. The draft's LightGBM seed counts (40 seeds, 8 seeds) are
  prototype results and are labeled as such where they remain.
- **The ATE-coverage claim is withdrawn.** "The 95% CI contained the ATE in only 34 of 40 seeds" was a
  LightGBM prototype figure. With histogram boosting the interval covered the ATE as often as the
  overlap-weighted truth (27/30 FULL, 28/30 QUICK against 27 and 29), because the two averages ($13.58
  and $14.28) differ by less than half a standard error. The case for comparing with the overlap-weighted
  average is the derivation in "What $\theta$ is when effects vary", which the module page carries; no
  coverage figure should be quoted for it.
- **Nuisance models.** Built as specified: histogram gradient boosting, `early_stopping=False`, no
  LightGBM (it needs Homebrew's libomp to import on macOS, and warned inside EconML in the prototype).
- **EconML's interval.** `LinearDML`'s bounds equal the hand HC0 interval widened by √(n/(n − 1)) (HC1)
  and by 1.959964/1.96 (EconML's exact normal quantile against the lab's 1.96): a factor of 1.0000066 at
  n = 20,000, measured on the lab's data. The checkpoint compares with this widened interval at rel 1e-4.
  On the lab's data the estimates differ by 2.0e-15, relative.
- **Measured run times** (run records, worked, FULL): 48.3 s on macOS with two threads, 52.8 s in Docker
  with `--cpus 2`; the slowest cell is the overfit demonstration (12.7 s and 14.7 s). No Colab run: the lab
  needs a release tag that contains it.
- **Sensitivity references.** EconML's docstrings attribute `sensitivity_interval` and `robustness_value`
  to "[Chernozhukov2022]". In the installed econml 0.17.0 source that key is defined only in
  `econml/validate/drtester.py`, for a different paper (lead's note during the references consolidation,
  2026-10-10; not re-checked here). Cite "Long story short" by its title and link, as in the References
  above, never by EconML's key. The lab's stretch text (`labs/src`) still says the documentation bases them
  on Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis; a change there is the Technical Expert's, at the
  next lab edit.
- **Open items for the Technical Expert, as found in the build:** generator, `EXTENSIONS` entry, truth and
  tolerances done; both labeled demonstration checks kept (each held on all 30 seeds in both modes);
  `tests/test_dml.py` and `tests/test_synth_app_install.py` exist; install list and deps as specified;
  "Before you start" renders the declared prerequisites. Equality of the hand and EconML estimates holds
  on the laptop and in Docker (Checkpoint 5 passed in both records); Colab is not yet checked.
- **For the Academic Director to reconsider (Pedagogy review):** the sensitivity line applies the rule to
  the shifted interval and also reports the shifted point estimate; the lab does both and the module page
  reports both (FULL: the interval flips to "do not offer", the point estimate $7.12 falls below $8.33).
  Assumption (3) is worded on the module page as in the lab, with the derivation that makes it hold
  approximately for a small credit in this simulation and the statement that it is not a general law.
