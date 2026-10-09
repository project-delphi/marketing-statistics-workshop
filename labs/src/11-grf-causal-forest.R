# ---
# jupyter:
#   api_checked: |
#     2026-10-09, with args() and the printed source in the workshop image (mktstats-env:spike,
#     R 4.6.1, grf 2.6.1), then by running every call below on Hillstrom in that image:
#     causal_forest(X, Y, W, Y.hat, W.hat, num.trees = 2000, ..., seed); the forest keeps
#       X.orig, Y.orig, W.hat and "_num_trees"
#     average_treatment_effect(forest, target.sample = c("all", ...), method = c("AIPW", "TMLE"))
#     rank_average_treatment_effect(forest, priorities, target = c("AUTOC", "QINI"), q, R = 200)
#       returns estimate, std.err, target, TOC (columns estimate, std.err, q, priority)
#     get_scores (exported; causal_forest method: forest, subset, ...)
#     best_linear_projection(forest, A, ..., vcov.type = "HC3")
#     predict (causal_forest method: object, newdata, ...)$predictions
#     plot (rank_average_treatment_effect method: main, xlab, ylab go to matplot)
#     tools::sha256sum(files)
#   note: This header is notebook metadata for maintainers; scripts/gen_notebooks.py drops it.
# ---

# %% [markdown]
# # Part A · Causal forest on the Hillstrom e-mail test
#
# **Order of work.** This lab has two notebooks. If you have not done so yet, open the Python
# notebook `11-uplift-econml` and run its install cell first (it takes about a minute on
# Colab), then come back here. Work Part A here (Exercises 1 to 3, 20 minutes), then return to
# the Python notebook for Part B and the decision.
#
# **The experiment.** In March 2008 Kevin Hillstrom's MineThatData challenge published an
# e-mail test on 64,000 customers who had bought in the last twelve months. Each customer was
# assigned at random, with probability one third each, to a Mens e-mail, a Womens e-mail or no
# e-mail, and their visits, purchases and spend over the next two weeks were recorded. We use
# two arms, **Womens e-mail** versus **No e-mail**: about 42,700 customers, each with chance
# one half of getting the e-mail.
#
# **Terms.**
#
# - Treatment $W_i \in \{0, 1\}$: customer $i$ got the e-mail. Outcome $Y_i$: dollars spent in
#   the two weeks after. Features $X_i$: what we knew before the e-mail (recency in months,
#   last year's spend `history`, bought men's or women's merchandise, zip code type, new
#   customer, purchase channel).
# - The **conditional average treatment effect** (CATE, also called *uplift*) is
#   $\tau(x) = E[Y(1) - Y(0) \mid X = x]$: how much more a customer like $x$ spends *because of*
#   the e-mail. Its average over all customers is the average treatment effect (ATE).
# - A **causal forest** (grf: Athey, Tibshirani & Wager) is a random forest that splits
#   customers where the *effect* differs, not where spend differs. It predicts $\hat\tau(x)$.
#
# **QUICK.** With QUICK on, the notebook keeps a random half of the customers and grows 200
# trees per forest instead of 1,000. Intervals are wider and the numbers will not match the
# module page; a decision near its threshold can flip. That is a lesson about sample size, not
# a bug.

# %%
suppressPackageStartupMessages(library(grf))

SEED <- 2026        # one visible seed: the subsample, the 50/50 split, both forests, the random check
NUM_TREES <- if (QUICK) 200 else 1000  # trees per forest (grf's default is 2,000; 1,000 halves the wait)
QUICK_SHARE <- 0.5  # QUICK keeps this share of the customers
MARGIN <- 0.30      # Exercise 3: assumed gross margin on spend (Hillstrom publishes none)
COST <- 0.10        # Exercise 3: assumed cost of sending one e-mail

cat("grf", as.character(packageVersion("grf")), "· QUICK =", QUICK, "·", NUM_TREES, "trees per forest\n")

# %% [markdown]
# ## Exercise 1 · Average effect, doubly robust (7 minutes)
#
# Start with the estimate you know from Module 6. In a randomized experiment the **difference
# in means** $\bar Y_1 - \bar Y_0$ (mean spend of e-mailed customers minus mean spend of the
# others) estimates the ATE without bias. Its standard error is
# $\sqrt{s_1^2 / n_1 + s_0^2 / n_0}$, where $s_1^2, s_0^2$ are the sample variances of spend
# (`var()`) and $n_1, n_0$ the numbers of customers in each arm.
#
# **Predict.** In Module 6 you saw Hillstrom's arms. Will the Womens e-mail's average effect on
# two-week spend be below \$0.50, between \$0.50 and \$1.50, or above \$1.50 per customer?
# Write your answer down.
#
# **Task.** Write `diff_in_means(Y, W)` returning a list with `estimate` (the difference in
# means) and `std.err` (its standard error, the formula above). Then a provided cell fits the
# causal forest and compares its doubly robust estimate with yours.

# %% tags=["exercise"]
diff_in_means <- function(Y, W) {
  # TODO 1: mean of Y where W == 1 minus mean where W == 0, and its standard error
  stop("TODO 1")
}

# %% [markdown]
# **Solution 1 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 1 — try it yourself first { display-mode: "form" }
solution(1, "diff_in_means", function(Y, W) {
  y1 <- Y[W == 1]
  y0 <- Y[W == 0]
  list(estimate = mean(y1) - mean(y0), std.err = sqrt(var(y1) / length(y1) + var(y0) / length(y0)))
})

# %% tags=["checkpoint"]
checkpoint(1, {
  toy <- diff_in_means(Y = c(1, 3, 2, 6), W = c(0, 0, 1, 1))
  if (!all(c("estimate", "std.err") %in% names(toy))) {
    stop("diff_in_means() should return a list with estimate and std.err; it has ",
         paste(names(toy), collapse = ", "), ".", call. = FALSE)
  }
  check_close(toy$estimate, 2, abs = 1e-9,
              name = "The toy difference in means, mean(2, 6) - mean(1, 3),")
  check_close(toy$std.err, sqrt(5), abs = 1e-9,
              name = paste("The toy standard error, sqrt(var(c(2, 6)) / 2 + var(c(1, 3)) / 2) = sqrt(8 / 2 + 2 / 2),",
                           "(var() divides by n - 1; divide each variance by its own arm's size)"))
})

# %% [markdown]
# Now the real data. The next cell downloads the Hillstrom file once (4 MB; a gzip copy on
# Amazon S3 is the fallback), checks it against a known SHA-256 hash, and caches it. The data
# are not stored in the workshop's repository: MineThatData states no licence for them.

# %%
# TODO(R/mktstats.R): use a shared Hillstrom loader once the helpers have one.
HILLSTROM_SOURCES <- list(
  list(url = paste0("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_",
                    "E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"),
       file = "hillstrom.csv",
       sha256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"),
  list(url = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
       file = "hillstrom.csv.gz",
       sha256 = "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f")
)

load_hillstrom <- function(sources = HILLSTROM_SOURCES) {
  cache <- Sys.getenv("MKTSTATS_CACHE", file.path(path.expand("~"), ".cache", "mktstats"))
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  verified <- function(path, sha) file.exists(path) && identical(unname(tools::sha256sum(path)), sha)
  read <- function(path) {
    con <- if (grepl("\\.gz$", path)) gzfile(path) else path
    utils::read.csv(con, stringsAsFactors = FALSE)
  }
  for (s in sources) {  # a verified copy in the cache: no download
    path <- file.path(cache, s$file)
    if (verified(path, s$sha256)) return(read(path))
  }
  for (s in sources) {
    path <- file.path(cache, s$file)
    for (attempt in 1:3) {
      done <- tryCatch({
        utils::download.file(s$url, path, quiet = TRUE, mode = "wb")
        TRUE
      }, error = function(e) FALSE, warning = function(w) FALSE)
      if (done && verified(path, s$sha256)) return(read(path))
      if (attempt < 3) Sys.sleep(2 * attempt)
    }
    unlink(path)
  }
  stop("Could not download the Hillstrom data with the expected SHA-256 hash from either source. ",
       "Check the network and rerun this cell; if it fails again, tell the instructor.", call. = FALSE)
}

hillstrom <- load_hillstrom()
cat(format(nrow(hillstrom), big.mark = ","), "customers\n")
print(table(hillstrom$segment))

# %% [markdown]
# Keep the two arms, build the feature matrix (categories as 0/1 columns; the reference levels
# are Urban and Phone), and split the customers 50/50 at random into a **training half**,
# which fits the forest, and an **evaluation half**, which only grades it (Exercise 2).

# %%
two_arms <- hillstrom[hillstrom$segment %in% c("Womens E-Mail", "No E-Mail"), ]
set.seed(SEED)
if (QUICK) two_arms <- two_arms[sort(sample.int(nrow(two_arms), round(QUICK_SHARE * nrow(two_arms)))), ]

X <- cbind(
  recency = two_arms$recency,
  history = two_arms$history,
  mens = two_arms$mens,
  womens = two_arms$womens,
  newbie = two_arms$newbie,
  zip_rural = as.numeric(two_arms$zip_code == "Rural"),
  zip_suburban = as.numeric(two_arms$zip_code == "Surburban"),
  channel_web = as.numeric(two_arms$channel == "Web"),
  channel_multichannel = as.numeric(two_arms$channel == "Multichannel")
)
Y <- two_arms$spend
W <- as.integer(two_arms$segment == "Womens E-Mail")

in_train <- sample.int(nrow(X)) <= nrow(X) %/% 2
X_train <- X[in_train, ]; Y_train <- Y[in_train]; W_train <- W[in_train]
X_eval <- X[!in_train, ]; Y_eval <- Y[!in_train]; W_eval <- W[!in_train]

cat(sprintf("Training half: %s customers (%.1f%% e-mailed); evaluation half: %s customers.\n",
            format(length(Y_train), big.mark = ","), 100 * mean(W_train), format(length(Y_eval), big.mark = ",")))
cat(sprintf("Share who spent anything: %.2f%%. Mean spend: $%.2f e-mailed, $%.2f not e-mailed.\n",
            100 * mean(Y_train > 0), mean(Y_train[W_train == 1]), mean(Y_train[W_train == 0])))

# %% [markdown]
# **Run.** The next cell fits a causal forest on the training half. Read the call: `W.hat` is
# each customer's probability of being e-mailed, the *propensity score*. grf estimates it with
# another forest unless you give it; here the e-mail was assigned at random with probability
# one half, so we pass `W.hat = 0.5`. This is the notebook's longest step: about a minute on
# Colab with 1,000 trees (FULL). While it runs, look at your prediction.
#
# `average_treatment_effect()` then estimates the ATE with the *augmented inverse probability
# weighting* (AIPW, "doubly robust") estimator: it averages, over customers, the forest's
# $\hat\tau(X_i)$ plus a correction from each customer's own outcome. It is unbiased if either
# the outcome model or the propensity score is right, and here the propensity is known.

# %%
started <- Sys.time()
forest <- causal_forest(X_train, Y_train, W_train, W.hat = 0.5, num.trees = NUM_TREES, seed = SEED)
ate <- average_treatment_effect(forest, target.sample = "all")
ate_ci <- ate[["estimate"]] + c(-1, 1) * 1.96 * ate[["std.err"]]
dm <- diff_in_means(Y_train, W_train)
dm_ci <- dm$estimate + c(-1, 1) * 1.96 * dm$std.err
cat(sprintf("Fitted %d trees on %s customers in %.0f s.\n", NUM_TREES,
            format(nrow(X_train), big.mark = ","), as.numeric(difftime(Sys.time(), started, units = "secs"))))
cat(sprintf("AIPW (causal forest):  $%.3f per customer, 95%% interval [$%.3f, $%.3f], width $%.3f\n",
            ate[["estimate"]], ate_ci[1], ate_ci[2], diff(ate_ci)))
cat(sprintf("Difference in means:   $%.3f per customer, 95%% interval [$%.3f, $%.3f], width $%.3f\n",
            dm$estimate, dm_ci[1], dm_ci[2], diff(dm_ci)))
cat("Both intervals: estimate ± 1.96 standard errors (normal approximation).\n")

# %% tags=["checkpoint"]
checkpoint(1, label = "1 on Hillstrom", {
  if (!isTRUE(all(forest$W.hat == 0.5))) {
    stop("The forest's W.hat is not 0.5 for every customer; the provided cell passes W.hat = 0.5.",
         call. = FALSE)
  }
  if (!(dm$std.err > 0 && ate[["std.err"]] > 0)) {
    stop(sprintf("A standard error is not positive (difference in means %.3g, AIPW %.3g): check diff_in_means().",
                 dm$std.err, ate[["std.err"]]), call. = FALSE)
  }
  if (dm$estimate < ate_ci[1] || dm$estimate > ate_ci[2]) {
    stop(sprintf(paste0("Your difference in means, %.3f, is outside the AIPW 95%% interval [%.3f, %.3f].",
                        " In a randomized experiment both estimate the same effect, so they should agree within",
                        " sampling error. Check that diff_in_means() subtracts the W == 0 mean from the W == 1 mean."),
                 dm$estimate, ate_ci[1], ate_ci[2]), call. = FALSE)
  }
})

# %% [markdown]
# **Explain.** Compare with your prediction. The two estimates are close, as they should be when
# the e-mail was assigned at random. What does the AIPW estimate use that the difference in
# means ignores, and why does that barely change the width here? Point to the widths printed
# above and to the share of customers who spent anything.
#
# <details><summary>Why this solution works</summary>
#
# The difference in means uses only who got the e-mail and what they spent. The AIPW estimate
# also uses the features: the forest predicts each customer's spend, so only the part of spend
# the features do not explain adds noise, and it corrects for chance imbalances in the features
# between the arms. That narrows the interval only when the features predict the outcome. Here
# fewer than 1% of customers buy in two weeks and a handful of large orders moves the mean, so
# recency and history explain little of who spends, and the two intervals are about as wide.
# Knowing the propensity (`W.hat = 0.5`) keeps the correction exact. A larger test, or an outcome
# with less variance (visits instead of spend), would narrow both. Athey & Wager (2019) apply the
# same estimator to a real study.
# </details>

# %% [markdown]
# ## Exercise 2 · Does the forest rank customers well? RATE on held-out data (7 minutes)
#
# A targeting list ranks customers by $\hat\tau(x)$, their *priority*. Is the ranking any good?
# The **TOC** (targeting operator characteristic) answers it: for the top share $q$ of the
# ranked list, $\mathrm{TOC}(q)$ = the average effect among those customers minus the average
# effect among all customers. A good ranking has a TOC well above 0 for small $q$. The **RATE**
# (rank-weighted average treatment effect) summarizes the curve in one number; with
# `target = "QINI"` it weights each $q$ by $q$, like the area under a Qini curve. A RATE of 0
# means the ranking does no better than random (Yadlowsky et al. 2025).
#
# The forest must not grade itself. So the priorities come from the **training** forest,
# predicted for the **evaluation** customers, and the TOC is estimated by a second forest
# fitted only on the evaluation half, whose doubly robust scores measure each customer's effect.
#
# **Predict.** Will the RATE (Qini target) on held-out customers have a 95% interval that
# excludes 0: yes or no? Write it down.
#
# **Task.** Write `rate_holdout(train_forest, X_eval, Y_eval, W_eval, num_trees, seed)`:
#
# 1. `priorities <- predict(train_forest, X_eval)$predictions`;
# 2. fit the evaluation forest on the held-out half, as the provided cell above did on the
#    training half (`W.hat = 0.5`, `num.trees = num_trees`, `seed = seed`);
# 3. return `list(rate = rank_average_treatment_effect(eval_forest, priorities, target = "QINI"),
#    eval_forest = eval_forest)` (Exercise 3 reuses the evaluation forest).
#
# The checkpoint runs your function, so it takes about a minute on Colab (FULL).

# %% tags=["exercise"]
rate_holdout <- function(train_forest, X_eval, Y_eval, W_eval, num_trees, seed) {
  # TODO 2: priorities from the training forest, an evaluation forest, then the RATE
  stop("TODO 2")
}

# %% [markdown]
# **Solution 2 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 2 — try it yourself first { display-mode: "form" }
solution(2, "rate_holdout", function(train_forest, X_eval, Y_eval, W_eval, num_trees, seed) {
  priorities <- predict(train_forest, X_eval)$predictions
  eval_forest <- causal_forest(X_eval, Y_eval, W_eval, W.hat = 0.5, num.trees = num_trees, seed = seed)
  list(rate = rank_average_treatment_effect(eval_forest, priorities, target = "QINI"),
       eval_forest = eval_forest)
})

# %% [markdown]
# The checkpoint compares your RATE with one computed from the training forest's predictions
# for the evaluation customers (`tau_eval`, computed here once; Exercise 3 uses it too).

# %%
tau_eval <- predict(forest, X_eval)$predictions
cat("The training forest's predicted effects for held-out customers (dollars):\n")
print(summary(tau_eval))

# %% tags=["checkpoint"]
checkpoint(2, {
  set.seed(SEED)  # the RATE's standard error comes from a bootstrap
  holdout <- rate_holdout(forest, X_eval, Y_eval, W_eval, num_trees = NUM_TREES, seed = SEED)
  if (!is.list(holdout) || !all(c("rate", "eval_forest") %in% names(holdout))) {
    stop("rate_holdout() should return list(rate = ..., eval_forest = ...).", call. = FALSE)
  }
  eval_forest <- holdout$eval_forest
  rate <- holdout$rate
  if (!inherits(eval_forest, "causal_forest") || nrow(eval_forest$X.orig) != nrow(X_eval) ||
      !isTRUE(all.equal(as.numeric(eval_forest$Y.orig), as.numeric(Y_eval)))) {
    stop("eval_forest should be a causal forest fitted on X_eval, Y_eval and W_eval (the held-out half), ",
         "never on the training half: a ranking graded on its own training data looks better than it is.",
         call. = FALSE)
  }
  if (!isTRUE(all(eval_forest$W.hat == 0.5))) {
    stop(sprintf(paste0("The evaluation forest's W.hat ranges from %.3f to %.3f; it should be 0.5 for every",
                        " customer. grf estimated it because W.hat was not given: pass W.hat = 0.5."),
                 min(eval_forest$W.hat), max(eval_forest$W.hat)), call. = FALSE)
  }
  if (!all(c("estimate", "std.err", "TOC") %in% names(rate))) {
    stop("rate should be the result of rank_average_treatment_effect(), with estimate, std.err and TOC; ",
         "it has ", paste(names(rate), collapse = ", "), ".", call. = FALSE)
  }
  if (!grepl("QINI", rate[["target"]])) {
    stop("The RATE target is ", rate[["target"]], ", not QINI: pass target = \"QINI\".", call. = FALSE)
  }
  if (!isTRUE(all.equal(sort(unique(rate$TOC$q)), seq(0.1, 1, by = 0.1)))) {
    stop("The TOC should be evaluated at q = 0.1, 0.2, ..., 1 (grf's default); leave q at its default.",
         call. = FALSE)
  }
  expected <- rank_average_treatment_effect(eval_forest, tau_eval, target = "QINI")
  if (!isTRUE(all.equal(rate[["estimate"]], expected[["estimate"]], tolerance = 1e-8))) {
    stop(sprintf(paste0("Your RATE is %.4f; with the training forest's predictions for the evaluation customers",
                        " as priorities it is %.4f. Check that the priorities come from",
                        " predict(train_forest, X_eval)."), rate[["estimate"]], expected[["estimate"]]),
         call. = FALSE)
  }
  set.seed(SEED)
  random_rate <- rank_average_treatment_effect(eval_forest, runif(nrow(X_eval)), target = "QINI")
  random_ci <- random_rate[["estimate"]] + c(-1, 1) * 1.96 * random_rate[["std.err"]]
  if (random_ci[1] > 0 || random_ci[2] < 0) {
    stop(sprintf(paste0("A random ranking should have a RATE near 0, but its 95%% interval is [%.4f, %.4f].",
                        " This is a problem with the data or the evaluation forest, not with your code:",
                        " tell the instructor."), random_ci[1], random_ci[2]), call. = FALSE)
  }
})

# %% [markdown]
# Look at whether the solid TOC line stays above 0 (the dotted line) for the top 10% to 30% of
# the ranked customers, and whether the dashed 95% bars include 0 there.

# %%
rate_ci <- rate[["estimate"]] + c(-1, 1) * 1.96 * rate[["std.err"]]
cat(sprintf("RATE (Qini target), held-out customers: %.4f dollars per customer, 95%% interval [%.4f, %.4f].\n",
            rate[["estimate"]], rate_ci[1], rate_ci[2]))
cat(sprintf("Random ranking, for comparison:          %.4f, 95%% interval [%.4f, %.4f].\n",
            random_rate[["estimate"]], random_ci[1], random_ci[2]))
plot(rate, main = "TOC on the evaluation half (95% intervals dashed)",
     ylab = "Extra effect over the average ($ per customer)",
     xlab = "Share of customers targeted, highest predicted effect first (q)")

# %% [markdown]
# **Explain.** Compare with your prediction. What does the interval of the RATE tell you about
# whether the forest found customers who respond more than average, and how does the TOC at
# $q = 0.1$ compare with its interval? Name one reason heterogeneity is hard to detect here.
#
# <details><summary>Why this solution works</summary>
#
# RATE asks whether customers ranked higher really gain more, measured with doubly robust
# scores from a forest that never saw the ranking's training data; grading the ranking on the
# customers that trained it would reward overfitting. On Hillstrom the TOC's intervals are wide:
# few customers buy, spend is skewed, and each half has about 21,000 customers, so differences
# in effect of a few tens of cents per customer are hard to tell from noise. A RATE whose
# interval includes 0 does not prove the effect is the same for everyone; it says this ranking
# has not shown it is better than random. More customers, a less noisy outcome (visits) or
# features that matter more would change that.
# </details>

# %% [markdown]
# ## Exercise 3 · Who clears the bar? (6 minutes)
#
# Sending an e-mail costs `COST` = \$0.10 and each dollar of spend earns a margin `MARGIN` =
# 30% (both assumed here; Hillstrom publishes neither). The rule: e-mail a customer when
# $0.30 \times \hat\tau(x) > \$0.10$, that is when the predicted effect is above \$0.33.
#
# To value a rule without a new experiment, use the evaluation forest's **doubly robust
# scores** $\Gamma_i$ (`get_scores(eval_forest)`): one number per held-out customer whose
# average over any group of customers estimates that group's average effect without bias.
# The incremental profit of a rule, against e-mailing nobody, per 1,000 customers is
# $1000 \times \frac{1}{n}\sum_{i \text{ targeted}} (0.30\,\Gamma_i - 0.10)$.
#
# **Predict.** At a \$0.10 cost per e-mail and a 30% margin, will more or less than half of
# the customers be worth e-mailing?
#
# **Task.** Write `policy_value(tau_hat, scores, margin, cost)` returning a list with
# `share_targeted` (the share with `margin * tau_hat > cost`) and `profit_per_1000` (the
# formula above, dividing by all $n$ customers, not only the targeted ones).

# %%
scores_eval <- get_scores(eval_forest)
cat(sprintf("Mean doubly robust score on the evaluation half (its ATE estimate): $%.3f per customer.\n",
            mean(scores_eval)))

# %% tags=["exercise"]
policy_value <- function(tau_hat, scores, margin, cost) {
  # TODO 3: who is targeted, and the profit per 1,000 customers
  stop("TODO 3")
}

# %% [markdown]
# **Solution 3 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 3 — try it yourself first { display-mode: "form" }
solution(3, "policy_value", function(tau_hat, scores, margin, cost) {
  send <- margin * tau_hat > cost
  list(share_targeted = mean(send), profit_per_1000 = 1000 * mean(send * (margin * scores - cost)))
})

# %% tags=["checkpoint"]
checkpoint(3, {
  toy <- policy_value(c(1, 0.2, 0.5, -1), c(2, 1, -1, 0.5), margin = 0.3, cost = 0.1)
  if (!all(c("share_targeted", "profit_per_1000") %in% names(toy))) {
    stop("policy_value() should return a list with share_targeted and profit_per_1000; it has ",
         paste(names(toy), collapse = ", "), ".", call. = FALSE)
  }
  check_close(toy$share_targeted, 0.5, abs = 1e-9,
              name = "share_targeted on the toy input (customers 1 and 3 have 0.3 x tau_hat > 0.1)")
  check_close(toy$profit_per_1000, 25, abs = 1e-9,
              name = paste("profit_per_1000 on the toy input, 1000 x (0.3 x 2 - 0.1 + 0.3 x (-1) - 0.1) / 4",
                           "(divide by all 4 customers),"))
  everyone <- policy_value(rep(Inf, length(scores_eval)), scores_eval, MARGIN, COST)
  check_close(everyone$profit_per_1000, 1000 * (MARGIN * mean(scores_eval) - COST), rel = 1e-9,
              name = "The profit per 1,000 of e-mailing everyone, 1000 x (margin x mean(scores) - cost),")
  nobody <- policy_value(rep(-Inf, length(scores_eval)), scores_eval, MARGIN, COST)
  check_close(c(nobody$share_targeted, nobody$profit_per_1000), c(0, 0), abs = 1e-12,
              name = "The share and profit of e-mailing nobody")
})

# %% [markdown]
# The table compares three rules on the held-out customers, with 95% intervals (estimate ± 1.96
# standard errors of the mean of the per-customer terms). Look at whether the forest's rule
# beats e-mailing everyone by more than the half-width of the difference's interval.

# %%
rule_terms <- function(send) 1000 * as.numeric(send) * (MARGIN * scores_eval - COST)
mean_ci <- function(x) mean(x) + c(0, -1.96, 1.96) * sd(x) / sqrt(length(x))
send_forest <- MARGIN * tau_eval > COST
send_all <- rep(TRUE, length(tau_eval))
rules <- rbind(
  "forest rule" = mean_ci(rule_terms(send_forest)),
  "e-mail everyone" = mean_ci(rule_terms(send_all)),
  "e-mail nobody" = c(0, 0, 0),
  "forest rule minus everyone" = mean_ci(rule_terms(send_forest) - rule_terms(send_all))
)
colnames(rules) <- c("profit_per_1000", "lower_95", "upper_95")
policy <- policy_value(tau_eval, scores_eval, MARGIN, COST)
cat(sprintf("The forest's rule e-mails %.1f%% of held-out customers.\n", 100 * policy$share_targeted))
print(round(rules, 1))

# %% [markdown]
# **Explain.** Compare with your prediction. Does the forest's rule earn more or less than
# e-mailing everyone, and is the difference larger than the half-width of its interval? Point to
# the last row. What does a targeting rule save, and what can it lose?
#
# <details><summary>Why this solution works</summary>
#
# The doubly robust scores give an unbiased estimate of the value of *any* targeting rule on
# held-out data, so you can compare rules without running an experiment for each. E-mailing
# everyone pays the cost for every customer, including those the e-mail does not move; the
# forest's rule skips customers whose predicted effect is below \$0.33. The gain from targeting
# is the cost saved on those customers, which is certain, minus the effect lost where the forest
# is wrong about them, which is not; that is why the difference between two rules can have a
# narrower interval than either rule's profit. Notice too that the evaluation half's average
# effect (the mean score above) differs from the training half's in Exercise 1: with spend this
# skewed, two random halves of 21,000 customers can disagree by half a dollar. A higher cost per
# e-mail, or a lower average effect, makes targeting matter more.
# </details>

# %% [markdown]
# ## Decision · What the Hillstrom test says about whom to e-mail
#
# The full decision, with a true answer to check against, is in the Python notebook. Here is
# the Hillstrom version: number, rule, recommendation.

# %%
usd <- function(x) sprintf("%s$%.1f", ifelse(x < 0, "-", ""), abs(x))  # -$7.3, $8.7
forest_row <- rules["forest rule", ]
everyone_row <- rules["e-mail everyone", ]
gap <- rules["forest rule minus everyone", ]
share <- sprintf("%.0f%%", 100 * policy$share_targeted)
advice <- if (gap[2] > 0) {
  sprintf("e-mail only the %s of customers the forest selects: %s more per 1,000 customers than e-mailing everyone (95%% interval %s to %s).",
          share, usd(gap[1]), usd(gap[2]), usd(gap[3]))
} else if (gap[3] < 0) {
  sprintf("e-mail everyone rather than the forest's list, which earns %s less per 1,000 customers (95%% interval %s to %s less).",
          usd(-gap[1]), usd(-gap[3]), usd(-gap[2]))
} else {
  sprintf(paste0("this test cannot tell the forest's list (%s of customers) from e-mailing everyone: %s per 1,000",
                 " customers, 95%% interval %s to %s. E-mailing everyone earns %s per 1,000 (%s to %s)."),
          share, usd(gap[1]), usd(gap[2]), usd(gap[3]), usd(everyone_row[1]), usd(everyone_row[2]), usd(everyone_row[3]))
}
cat(sprintf("Number         : the forest's rule e-mails %s of %s held-out customers and earns %s per 1,000\n",
            share, format(length(tau_eval), big.mark = ","), usd(forest_row[1])))
cat(sprintf("                 (95%% interval %s to %s); e-mailing everyone %s (%s to %s); e-mailing nobody $0.\n",
            usd(forest_row[2]), usd(forest_row[3]), usd(everyone_row[1]), usd(everyone_row[2]), usd(everyone_row[3])))
cat(sprintf("                 RATE (Qini) of the forest's ranking: 95%% interval [%.4f, %.4f].\n", rate_ci[1], rate_ci[2]))
cat(sprintf("Rule           : e-mail when %.2f x predicted uplift > $%.2f cost per e-mail.\n", MARGIN, COST))
cat("Recommendation :", advice, "\n")
cat("What would change it: a higher cost per e-mail, a larger test, or a ranking whose RATE interval excludes 0.\n")
cat("Mode           :", if (QUICK) "QUICK run: treat this as a rough answer." else "FULL run.", "\n")
cat("Your sentence  : ________________________________________________\n")

# %% [markdown]
# Write your sentence, and note the forest rule's share and profit per 1,000: the Python
# notebook's Decision cell has a place for them. **Now return to the Python notebook for
# Part B.**

# %% [markdown]
# ## Stretch (optional) · Who benefits? A best linear projection
#
# `best_linear_projection(eval_forest, A)` fits the best linear approximation of $\tau(x)$ on
# the columns of `A`, with heteroskedasticity-robust standard errors. Does the e-mail's effect
# fall with recency (months since the last purchase) or rise with last year's spend?

# %%
best_linear_projection(eval_forest, X_eval[, c("recency", "history")])
