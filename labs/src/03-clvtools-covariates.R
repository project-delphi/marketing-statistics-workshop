# ---
# jupyter:
#   mktstats:
#     verified: >-
#       2026-10-09, Technical Expert, in the workshop image (R 4.6.1, CLVTools 0.12.1):
#       args(clvdata) (data.transactions, date.format, time.unit, estimation.split, data.end,
#       name.id, name.date, name.price; no verbose argument), args(SetStaticCovariates),
#       args(pnbd) (optimx.args, verbose), methods for clv.fitted (coef, confint, vcov, summary,
#       predict; confint is a Wald interval, coef +- z * sqrt(diag(vcov))).
#       summary(clv)$descriptives.transactions is a table with columns Name,
#       Estimation, Holdout, Total (character). Ran: data(cdnow) with estimation.split =
#       "1997-09-30" gives 1,411 zero repeaters and 1,882 holdout transactions; pnbd() gives
#       r 0.55327, alpha 10.5778, s 0.60602, beta 11.66391 (within 0.04% of the Module 2 MAP);
#       coefficients of 0/1 covariates are named trans.channel_social, life.channel_social, ...
#       On the committed retailer the four channel coefficients lie inside confint(level = 0.9)
#       (3 or 4 of 4 on each of 20 generator seeds).
# ---

# %% [markdown]
# # Part A · CLVTools on the Day 1 data
#
# **Before you start:** open the Python notebook of this lab (`03-btyd-vs-ml`) and run its
# install cell, so it is ready when you get there. Then work through this R notebook.
#
# CLVTools is an R package for buy-till-you-die models. It fits the same Pareto/NBD as
# Module 2 by maximum likelihood, and it can add customer characteristics (*covariates*) to
# the purchase and dropout processes. The flow is always: a `clvdata()` object (the
# transaction log plus the calibration/holdout split), then a model such as `pnbd()`.
#
# The next cell loads the CDNOW log that ships with CLVTools (2,357 customers; transactions
# on the same day are already combined) and the workshop's synthetic retailer, whose
# customers came through three acquisition channels: search, social and referral.

# %%
suppressPackageStartupMessages(library(CLVTools))
data("cdnow", package = "CLVTools")  # columns Id, Date, CDs, Price
tx_retail <- mkt_data("synthetic/retailer_transactions.csv")  # customer_id, date, amount
names(tx_retail) <- c("Id", "Date", "Price")  # the column names CLVTools expects by default
customers <- mkt_data("synthetic/retailer_customers.csv")
truth <- mkt_data("synthetic/truth.json")$retailer
channels <- c("search", "social", "referral")  # search is the reference channel

summary_value <- function(clv, name, column = "Estimation") {
  # one entry of the table that summary(clv) prints, e.g. "Total # zero repeaters"
  d <- summary(clv)$descriptives.transactions
  d[[column]][d$Name == name]
}
cat("CDNOW:", nrow(cdnow), "transactions;", "synthetic retailer:", nrow(tx_retail),
    "transactions,", nrow(customers), "customers; calibration ends", truth$calibration_end, "\n")
cat("QUICK has no effect in this notebook: every fit takes seconds.\n")

# %% [markdown]
# ## Exercise 1 · Build the clv.data object and compare with Day 1 (7 minutes)
#
# **Predict.** Will CLVTools' Pareto/NBD estimate of $r$ on CDNOW be within 1% of
# PyMC-Marketing's Module 2 value (0.553): yes or no? Which of $r$, $\alpha$, $s$, $\beta$ is
# most likely to differ?
#
# **Task.** Write two functions:
#
# - `make_clvdata(tx, split)` returning
#   `clvdata(tx, date.format = "ymd", time.unit = "week", estimation.split = split,
#   name.id = "Id", name.date = "Date", name.price = "Price")`. Pass `split` as a date string
#   (`"1997-09-30"`): a number would count weeks from the first transaction instead.
# - `compare_with_day1(fit, day1)` returning a data frame with the columns `parameter`,
#   `CLVTools` (from `coef(fit)`), `PyMC_Marketing` (from the named vector `day1`) and
#   `rel_difference` (CLVTools / PyMC-Marketing − 1).
#
# The checkpoint tests `make_clvdata` first, before any model is fitted.

# %% tags=["exercise"]
make_clvdata <- function(tx, split) {
  # TODO 1: clvdata(...) with a date split
  stop("TODO 1")
}

compare_with_day1 <- function(fit, day1) {
  # TODO 1: one row per parameter of day1
  stop("TODO 1")
}

# %% [markdown]
# **Solution 1 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 1 — try it yourself first { display-mode: "form" }
solution(1, "make_clvdata", function(tx, split) {
  clvdata(tx, date.format = "ymd", time.unit = "week", estimation.split = split,
          name.id = "Id", name.date = "Date", name.price = "Price")
})

solution(1, "compare_with_day1", function(fit, day1) {
  estimate <- coef(fit)[names(day1)]
  data.frame(parameter = names(day1), CLVTools = unname(estimate),
             PyMC_Marketing = unname(day1), rel_difference = unname(estimate / day1 - 1))
})

# %% [markdown]
# **Explain.** After the fit below, compare with your prediction. Two independent
# implementations, maximum likelihood in R and a flat-prior MAP in PyMC, see the same data:
# what should you expect, and what would a large difference tell you?
#
# <details><summary>Why this solution works</summary>
#
# `clvdata()` combines same-day transactions, splits the log at the date you give and
# measures time in weeks from each customer's first purchase, the same $x$, $t_x$ and $T$ as
# Module 2. With flat priors a MAP estimate is the maximum-likelihood estimate, so the two
# packages should agree up to the optimizer's tolerance; a large gap would mean the data or
# the time unit differ. The dropout parameters $s$ and $\beta$ move the most when the data
# change slightly (try `split = 39`, which ends the calibration on 1997-10-01): dropout is
# never observed, so the likelihood is flat in that direction.
# </details>

# %% tags=["checkpoint"]
checkpoint(1, {
  clv_cdnow <- make_clvdata(cdnow, "1997-09-30")
  if (!inherits(clv_cdnow, "clv.data")) {
    stop("make_clvdata() should return the object clvdata() makes; it returned a ",
         class(clv_cdnow)[1], ". To move on now, run use_reference(1).", call. = FALSE)
  }
  zero <- as.numeric(summary_value(clv_cdnow, "Total # zero repeaters"))
  if (!isTRUE(zero == 1411)) {
    stop(sprintf(paste0("The calibration period has %s customers without a repeat purchase; Module 2",
                        " (and the BG/NBD note) has 1,411. Check that estimation.split is the date",
                        " \"1997-09-30\", not a number of weeks, and time.unit = \"week\". To move",
                        " on now, run use_reference(1)."), format(zero)), call. = FALSE)
  }
  end <- summary_value(clv_cdnow, "Period End")
  if (!identical(end, "1997-09-30")) {
    stop("The calibration period ends on ", end, ", not 1997-09-30: pass the date string as",
         " estimation.split. To move on now, run use_reference(1).", call. = FALSE)
  }
  holdout <- as.numeric(summary_value(clv_cdnow, "Total # Transactions", "Holdout"))
  check_close(holdout, 1882, abs = 0, name = "holdout transactions (Module 2 counted 1,882 purchase days)")
})

# %% [markdown]
# Now the Pareto/NBD fit on CDNOW, by maximum likelihood (a second or two).

# %%
fit_cdnow <- pnbd(clv_cdnow, verbose = FALSE)
coef(fit_cdnow)

# %% [markdown]
# The Module 2 values below come from its recorded run (Pareto/NBD, flat priors, MAP,
# PyMC-Marketing 1.2.0).

# %% tags=["checkpoint"]
checkpoint(1, {
  day1 <- c(r = 0.553279, alpha = 10.577699, s = 0.606230, beta = 11.668429)
  comparison <- compare_with_day1(fit_cdnow, day1)
  check_columns(comparison, c("parameter", "CLVTools", "PyMC_Marketing", "rel_difference"),
                name = "compare_with_day1(fit_cdnow, day1)")
  if (!identical(as.character(comparison$parameter), names(day1))) {
    stop("One row per parameter, in the order r, alpha, s, beta. To move on now, run",
         " use_reference(1).", call. = FALSE)
  }
  check_close(comparison$CLVTools, unname(coef(fit_cdnow)[names(day1)]), rel = 1e-12,
              name = "the CLVTools column (coef(fit))")
  check_close(comparison$rel_difference, comparison$CLVTools / comparison$PyMC_Marketing - 1,
              abs = 1e-12, name = "rel_difference (CLVTools / PyMC-Marketing - 1)")
  worst <- max(abs(comparison$rel_difference))
  if (worst > 0.01) {
    stop(sprintf(paste0("CLVTools and PyMC-Marketing differ by up to %.1f%%. With the same data they",
                        " agree within 0.1%%: check the split date and the time unit in make_clvdata()."),
                 100 * worst), call. = FALSE)
  }
  print(comparison, digits = 5)
}, label = "1b")

# %% [markdown]
# # Part B · Channel as a covariate
#
# CLVTools adds a covariate $z_i$ (here two 0/1 indicators, `channel_social` and
# `channel_referral`; search is the reference with both 0) to the Pareto/NBD as
# $\alpha_i = \alpha\, e^{-\gamma_{\text{trans}} z_i}$ and $\beta_i = \beta\, e^{-\gamma_{\text{life}} z_i}$
# (Fader & Hardie's note 019, cited by CLVTools' `pnbd` help page). A positive
# $\gamma_{\text{trans}}$ raises the purchase rate; a positive $\gamma_{\text{life}}$ raises the
# dropout rate, so customers leave sooner. Pass the indicators, not the channel names:
# CLVTools would pick the reference level of a text column alphabetically (`referral`).
#
# Look at each channel's share of customers who never bought again in the calibration period:

# %%
clv_retail <- make_clvdata(tx_retail, truth$calibration_end)
cal_days <- unique(tx_retail[as.Date(tx_retail$Date) <= as.Date(truth$calibration_end), c("Id", "Date")])
purchase_days <- table(cal_days$Id)
customers$no_repeat <- as.vector(purchase_days[as.character(customers$customer_id)]) == 1
channel_table <- do.call(rbind, lapply(channels, function(ch) {
  in_channel <- customers$acquisition_channel == ch
  data.frame(channel = ch, customers = sum(in_channel),
             share_no_repeat = round(mean(customers$no_repeat[in_channel]), 3))
}))
channel_table

# %% [markdown]
# ## Exercise 2 · Fit the Pareto/NBD with the channel covariate (8 minutes)
#
# **Predict.** For the channel with the highest share of customers who never bought again,
# will its dropout coefficient $\gamma_{\text{life}}$ be positive (customers leave sooner) or
# negative?
#
# **Task.** Write `fit_channel_pnbd(clv, customers)` that builds a data frame `cov` with the
# columns `Id` (from `customers$customer_id`), `channel_social` and `channel_referral`, adds it
# to `clv` with `SetStaticCovariates(clv, data.cov.life = cov, data.cov.trans = cov,
# names.cov.life = ..., names.cov.trans = ..., name.id = "Id")`, and returns
# `pnbd(..., verbose = FALSE)` of the result. The fit takes 5 to 20 seconds.

# %% tags=["exercise"]
fit_channel_pnbd <- function(clv, customers) {
  # TODO 2: covariate table -> SetStaticCovariates -> pnbd
  stop("TODO 2")
}

# %% [markdown]
# **Solution 2 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 2 — try it yourself first { display-mode: "form" }
solution(2, "fit_channel_pnbd", function(clv, customers) {
  cov <- data.frame(Id = customers$customer_id, channel_social = customers$channel_social,
                    channel_referral = customers$channel_referral)
  indicators <- c("channel_social", "channel_referral")
  clv_cov <- SetStaticCovariates(clv, data.cov.life = cov, data.cov.trans = cov,
                                 names.cov.life = indicators, names.cov.trans = indicators,
                                 name.id = "Id")
  pnbd(clv_cov, verbose = FALSE)
})

# %% [markdown]
# **Explain.** Compare with your prediction, using the table under the checkpoint. What part
# of the customer-to-customer differences does the channel explain, and what do $r$, $\alpha$,
# $s$, $\beta$ describe now?
#
# <details><summary>Why this solution works</summary>
#
# The same two indicators enter both processes, so the model estimates four channel
# coefficients: purchase and dropout, for social and for referral, each relative to search.
# Customers still differ within a channel (the gamma distributions absorb that); the
# covariates explain the part of the differences that is shared by a channel. $r$, $\alpha$,
# $s$, $\beta$ now describe the reference channel, search. A 90% confidence interval from
# `confint()` contains the true coefficient in about 90% of repeated samples; with four of
# them, all four cover only about 66% of the time, so the check allows one miss.
# </details>

# %% tags=["checkpoint"]
checkpoint(2, {
  fit_cov <- fit_channel_pnbd(clv_retail, customers)
  est <- coef(fit_cov)
  wanted <- c("trans.channel_social", "trans.channel_referral", "life.channel_social",
              "life.channel_referral")
  if (!all(wanted %in% names(est))) {
    stop("The fit has no coefficients named ", paste(setdiff(wanted, names(est)), collapse = ", "),
         ". Pass the two 0/1 indicator columns as covariates for both processes. To move on now,",
         " run use_reference(2).", call. = FALSE)
  }
  if (anyNA(est)) {
    stop("Some coefficients are NA: the optimizer failed. Refit with",
         " pnbd(..., optimx.args = list(method = \"Nelder-Mead\")), or run use_reference(2).",
         call. = FALSE)
  }
  ci <- confint(fit_cov, level = 0.9)
  true_value <- c(trans.channel_social = truth$covariates$gamma_purchase$social,
                  trans.channel_referral = truth$covariates$gamma_purchase$referral,
                  life.channel_social = truth$covariates$gamma_dropout$social,
                  life.channel_referral = truth$covariates$gamma_dropout$referral)
  recovery <- data.frame(coefficient = wanted, truth = unname(true_value[wanted]),
                         estimate = unname(est[wanted]), ci90_low = unname(ci[wanted, 1]),
                         ci90_high = unname(ci[wanted, 2]))
  recovery$inside <- recovery$truth >= recovery$ci90_low & recovery$truth <= recovery$ci90_high
  if (sum(recovery$inside) < 3) {
    print(recovery, digits = 3)
    stop(sprintf(paste0("Only %d of the 4 true channel coefficients lie inside their 90%% confidence",
                        " intervals (at least 3 should). Check that cov holds channel_social and",
                        " channel_referral as 0/1 numbers for both processes. To move on now, run",
                        " use_reference(2)."), sum(recovery$inside)), call. = FALSE)
  }
  print(recovery, digits = 3)
})

# %% [markdown]
# ## Exercise 3 · Translate coefficients into rates (7 minutes)
#
# **Predict.** A purchase coefficient of 0.3: does it raise the average purchase rate by about
# 3%, 30% or 35%?
#
# **Task.** Write `channel_rates(fit, channels)` returning one row per channel with
# `channel`, `purchase_rate` ($r / \alpha_c$, purchases per week),
# `dropout_rate` ($s / \beta_c$, per week) and `mean_lifetime_weeks` ($1 / $ dropout rate).
# Here $\alpha_c = \alpha\, e^{-\gamma_{\text{trans},c}}$ and $\beta_c = \beta\, e^{-\gamma_{\text{life},c}}$,
# with $\gamma = 0$ for search. The coefficients are in `coef(fit)`, e.g.
# `coef(fit)[["trans.channel_social"]]`. The lifetime is the one implied by the mean dropout
# rate, not the mean of individual lifetimes.

# %% tags=["exercise"]
channel_rates <- function(fit, channels) {
  # TODO 3: r / alpha_c, s / beta_c and 1 / (s / beta_c) for each channel
  stop("TODO 3")
}

# %% [markdown]
# **Solution 3 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 3 — try it yourself first { display-mode: "form" }
solution(3, "channel_rates", function(fit, channels) {
  b <- coef(fit)
  gamma <- function(process, ch) {
    key <- paste0(process, ".channel_", ch)
    if (key %in% names(b)) b[[key]] else 0  # 0 for the reference channel
  }
  purchase <- sapply(channels, function(ch) b[["r"]] / (b[["alpha"]] * exp(-gamma("trans", ch))))
  dropout <- sapply(channels, function(ch) b[["s"]] / (b[["beta"]] * exp(-gamma("life", ch))))
  data.frame(channel = channels, purchase_rate = unname(purchase), dropout_rate = unname(dropout),
             mean_lifetime_weeks = unname(1 / dropout))
})

# %% [markdown]
# **Explain.** Compare with your prediction: $e^{0.3} = 1.35$. Which channel brings customers
# who buy most often, and which brings customers who stay longest? Point to the table.
#
# <details><summary>Why this solution works</summary>
#
# A Gamma distribution with shape $r$ and rate $\alpha_c$ has mean $r/\alpha_c$, so a
# covariate multiplies the mean purchase rate by $e^{\gamma}$, exactly like a coefficient in a
# log-linear regression: 0.3 means +35%, not +30%. The same holds for dropout. The checkpoint
# compares each channel's purchase rate with the average true rate of that channel's
# customers in this dataset. The estimates sit a few percent below it for every channel,
# partly because the data record purchases by calendar day, so two purchases on one day count
# once. This table feeds the acquisition-cost caps of Module 5.
# </details>

# %% tags=["checkpoint"]
checkpoint(3, {
  rates <- channel_rates(fit_cov, channels)
  check_columns(rates, c("channel", "purchase_rate", "dropout_rate", "mean_lifetime_weeks"),
                name = "channel_rates(fit_cov, channels)")
  b <- coef(fit_cov)
  g_trans <- c(search = 0, social = b[["trans.channel_social"]], referral = b[["trans.channel_referral"]])
  g_life <- c(search = 0, social = b[["life.channel_social"]], referral = b[["life.channel_referral"]])
  expected_purchase <- unname(b[["r"]] / b[["alpha"]] * exp(g_trans[channels]))
  expected_dropout <- unname(b[["s"]] / b[["beta"]] * exp(g_life[channels]))
  if (!identical(as.character(rates$channel), channels)) {
    stop("One row per channel in the order search, social, referral.", call. = FALSE)
  }
  check_close(rates$purchase_rate, expected_purchase, rel = 1e-9,
              name = "purchase_rate = r / (alpha * exp(-gamma_trans))")
  check_close(rates$dropout_rate, expected_dropout, rel = 1e-9,
              name = "dropout_rate = s / (beta * exp(-gamma_life))")
  check_close(rates$mean_lifetime_weeks, 1 / expected_dropout, rel = 1e-9,
              name = "mean_lifetime_weeks = 1 / dropout_rate")
  # Truth: the mean true purchase rate of each channel's customers. Over 20 generator seeds the
  # estimate was at most 8.7% from it (measured 2026-10-09), so 10% is the tolerance.
  rates$true_purchase_rate <- sapply(channels, function(ch)
    mean(customers$true_lambda[customers$acquisition_channel == ch]))
  rates$true_dropout_rate <- sapply(channels, function(ch)
    mean(customers$true_mu[customers$acquisition_channel == ch]))
  for (i in seq_along(channels)) {
    gap <- rates$purchase_rate[i] / rates$true_purchase_rate[i] - 1
    if (abs(gap) > 0.10) {
      stop(sprintf(paste0("%s: estimated purchase rate %.4f per week, true mean %.4f (%+.0f%%; allowed",
                          " 10%%). Check the sign: alpha_c = alpha * exp(-gamma), so the rate is",
                          " r / alpha * exp(+gamma). To move on now, run use_reference(3)."),
                   channels[i], rates$purchase_rate[i], rates$true_purchase_rate[i], 100 * gap),
           call. = FALSE)
    }
  }
  print(rates, digits = 3)
})

# %% [markdown]
# ## Decision · Treat channels differently?
#
# **The rule, stated before the numbers.** Treat a channel differently in acquisition
# spending if its purchase or dropout rate differs from search by more than 20% and the 90%
# confidence interval of that ratio excludes 1 (no difference). The 20% is a business
# threshold the retailer chose: smaller differences are not worth separate budgets. The ratio
# of a channel's rate to search's is $e^{\gamma}$, and its interval is $e^{\text{confint}}$.

# %%
ci90 <- confint(fit_cov, level = 0.9)
rows <- list()
for (process in c("trans", "life")) {
  for (ch in channels[-1]) {
    key <- paste0(process, ".channel_", ch)
    ratio <- exp(coef(fit_cov)[[key]])
    low <- exp(ci90[key, 1]); high <- exp(ci90[key, 2])
    rows[[key]] <- data.frame(
      channel = ch, rate = if (process == "trans") "purchase" else "dropout",
      ratio_to_search = round(ratio, 3), ci90_low = round(low, 3), ci90_high = round(high, 3),
      treat_differently = abs(ratio - 1) > 0.20 && (low > 1 || high < 1))
  }
}
decision <- do.call(rbind, rows)
rownames(decision) <- NULL
cat("Synthetic retailer, Pareto/NBD with channel covariates, maximum likelihood; 90% Wald",
    "intervals from confint().\n")
decision

# %% [markdown]
# **Recommendation.** Write one sentence: which channels to treat differently and why, how
# sure you are (the intervals), and what would change it (more customers per channel, or a
# different threshold than 20%). Module 5 turns these rates into the most you can pay to
# acquire a customer in each channel. Carry the sentence to the Python notebook, where the
# lab's second decision (which forecaster) is made.
#
# Your sentence: ________________________________________________

# %% [markdown]
# ## Stretch (optional) · Intervals on each customer's forecast
#
# `predict(fit_cov)` gives, per customer, `PAlive`, `CET` (expected transactions in the
# holdout) and `predicted.CLV`. `predict(fit_cov, uncertainty = "boots", num.boots = 20)` adds
# bootstrap intervals by refitting the model on resampled customers: expect several minutes
# on two CPUs. Which customers have the widest interval on `CET`, and why?
