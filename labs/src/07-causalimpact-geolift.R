# %% [markdown]
# <!--
# APIs checked for this lab (2026-10-09) inside the workshop image (R 4.6.1; CausalImpact 1.4.1,
# bsts 0.9.11, GeoLift 2.7.5 and augsynth 0.2.0 at the pinned commits) with `args()` and the
# package source: `CausalImpact(data, pre.period, post.period, model.args, bsts.model,
# post.period.response, alpha = 0.05)`, whose model fixes the bsts sampler's seed internally
# (`seed = 1` in `ConstructModel`); `impact$summary` has rows `Average` and `Cumulative` and columns
# `Actual`, `Pred`, `AbsEffect`, `AbsEffect.lower`, `AbsEffect.upper`, `RelEffect`, `alpha`, `p` and
# others. `GeoDataRead(data, date_id, location_id, Y_id, format, ...)`,
# `GeoLiftMarketSelection(data, treatment_periods, N, effect_size, lookback_window, cpic, budget,
# alpha, Correlations, fixed_effects, side_of_test, parallel, print, ...)`, `GeoLiftPower(data,
# locations, effect_size, treatment_periods, lookback_window, cpic, side_of_test, parallel, ...)`,
# `augsynth(form, unit, time, data, t_int, ...)` and `summary(fit, inf_type, alpha)`.
# -->
#
# **Order of work.** This R notebook is the second of the lab's two. Run its install cell
# (above) first: CausalImpact, GeoLift and their dependencies are the slowest installs of the
# lab (about 85 seconds of installs in the Colab timing spike of 2026-10-09). While it runs,
# work Parts A and B of the Python notebook, `labs/python/07-synthetic-control-did.ipynb`
# (Exercises 1 to 4, about 28 minutes). About 33 minutes into the lab, come back here for
# Part C: Exercises 1 and 2 (about 12 minutes) and the short decision cell. Then return to the
# Python notebook for the lab's decision. Part D is the demo for the afternoon geo-test design
# clinic, not part of the lab.
#
# **How this lab runs.** CausalImpact fits by Markov chain Monte Carlo (MCMC): 5,000 draws in a full
# run, 1,000 with `QUICK <- TRUE` (intervals then wobble a little more). Part D runs GeoLift's market
# selection: two and three treated markets in a full run, two with QUICK.

# %%
suppressPackageStartupMessages(library(CausalImpact))
panel <- mkt_data("synthetic/geo_panel.csv")
panel$date <- as.Date(panel$date)
truth <- mkt_data("synthetic/truth.json")$geo_panel
TREATED <- truth$treated_geos
TEST_START <- as.Date(truth$test_start)
TEST_END <- as.Date(truth$test_end)
SEED <- 2029            # set before every CausalImpact call below
NITER <- if (QUICK) 1000 else 5000  # MCMC draws per fit
GROSS_MARGIN <- truth$campaign$margin   # 0.30 (truth.json), as in the Python notebook
CAMPAIGN_COST <- truth$campaign$cost    # $25,000 (truth.json), as in the Python notebook
options(repr.plot.width = 9, repr.plot.height = 5)

if (length(unique(panel$geo)) != 40 || length(unique(panel$date)) != 104 || length(TREATED) != 8) {
  stop("The geo panel does not have 40 geos x 104 weeks and 8 treated geos: rerun the install cell,",
       " then this one. If it persists, tell the instructor.", call. = FALSE)
}
cat(length(unique(panel$geo)), "geos x", length(unique(panel$date)), "weeks; treated:",
    paste(TREATED, collapse = ", "), "\n")
cat("Test window:", format(TEST_START), "to", format(TEST_END), "· MCMC draws per fit:", NITER, "\n")

# %% [markdown]
# Two helpers: `wide_panel()` turns the long panel into one column per geo (rows are weeks, in
# date order), and `cumulative()` reads the four numbers this lab uses from a CausalImpact result.

# %%
wide_panel <- function(panel) {
  w <- reshape(panel[c("date", "geo", "sales")], idvar = "date", timevar = "geo", direction = "wide")
  w <- w[order(w$date), ]
  names(w) <- sub("^sales\\.", "", names(w))
  rownames(w) <- NULL
  w
}

cumulative <- function(impact) {
  unlist(impact$summary["Cumulative", c("AbsEffect", "AbsEffect.lower", "AbsEffect.upper", "RelEffect")])
}

head(wide_panel(panel)[, 1:6], 3)

# %% [markdown]
# # Part C · A Bayesian structural time-series view
#
# **CausalImpact** (Brodersen et al. 2015) is a Bayesian synthetic control over time. It models the
# treated series as a **local level** (a baseline that drifts slowly from week to week) plus a
# regression on the control series, fits that model on the pre-period, and forecasts it through the
# test window. The forecast is the counterfactual; actual minus forecast is the effect. The
# regression uses a **spike-and-slab** prior: each control geo is either in the model or out of it,
# and the posterior says how often each one is in. Because the fit is Bayesian, the interval is a
# **95% posterior interval**: given the model and the data, the effect lies inside it with
# probability 0.95.
#
# `CausalImpact(data, pre.period, post.period, model.args = list(niter = ...), alpha = 0.05)` takes a
# matrix whose **first column is the response** (the treated geos' total) and whose other columns
# are the controls, and the first and last row numbers of the pre-period and the test window.

# %% [markdown]
# ## Exercise 1 · CausalImpact on the panel (7 minutes)
#
# **Predict.** Will CausalImpact's 95% interval for cumulative incremental sales contain the true
# value: yes or no? Will it be wider or narrower than ±10% of the estimate?
#
# **Task.** Write `run_causal_impact(panel, treated_geos, test_start, test_end, niter)`:
#
# 1. `w <- wide_panel(panel[panel$date <= test_end, ])`;
# 2. a matrix with the treated geos' weekly total first, `rowSums(w[treated_geos])`, then the
#    control geos (every other geo column) as `as.matrix(...)`;
# 3. `pre.period = c(1, n_pre)` and `post.period = c(n_pre + 1, nrow(w))`, where `n_pre` counts
#    the weeks before `test_start`;
# 4. return the result of `CausalImpact(..., model.args = list(niter = niter), alpha = 0.05)`.

# %% tags=["exercise"]
run_causal_impact <- function(panel, treated_geos, test_start, test_end, niter) {
  # TODO 1: wide table, response first then controls, pre and post rows, CausalImpact()
  stop("TODO 1")
}

# %% [markdown]
# **Solution 1 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 1 — try it yourself first { display-mode: "form" }
solution(1, "run_causal_impact", function(panel, treated_geos, test_start, test_end, niter) {
  w <- wide_panel(panel[panel$date <= test_end, ])
  controls <- setdiff(names(w)[-1], treated_geos)
  data <- cbind(treated = rowSums(w[treated_geos]), as.matrix(w[controls]))
  n_pre <- sum(w$date < test_start)
  CausalImpact(data, pre.period = c(1, n_pre), post.period = c(n_pre + 1, nrow(w)),
               model.args = list(niter = niter), alpha = 0.05)
})

# %% tags=["checkpoint"]
checkpoint(1, {
  set.seed(SEED)
  impact <- run_causal_impact(panel, TREATED, TEST_START, TEST_END, NITER)
  if (!inherits(impact, "CausalImpact")) {
    stop("run_causal_impact() should return the object CausalImpact() returns, not ",
         class(impact)[1], ". Fix TODO 1, or run use_reference(1).", call. = FALSE)
  }
  in_window <- panel$geo %in% TREATED & panel$date >= TEST_START & panel$date <= TEST_END
  check_close(impact$summary["Cumulative", "Actual"], sum(panel$sales[in_window]), rel = 1e-9,
              name = paste("CausalImpact's actual sales in the test window (the treated total must be",
                           "the FIRST column, and the test window the post-period)"))
  check_close(impact$summary["Cumulative", "alpha"], 0.05, abs = 1e-12, name = "alpha (pass alpha = 0.05)")
  ci <- cumulative(impact)
  if (truth$incremental_sales < ci[["AbsEffect.lower"]] || truth$incremental_sales > ci[["AbsEffect.upper"]]) {
    stop(sprintf(paste0("The true incremental sales, %s, are outside your 95%% posterior interval [%s, %s]. ",
                        "Check that the controls exclude every treated geo and that the pre-period ends the ",
                        "week before test_start. Fix TODO 1, or run use_reference(1)."),
                 format(round(truth$incremental_sales), big.mark = ","),
                 format(round(ci[["AbsEffect.lower"]]), big.mark = ","),
                 format(round(ci[["AbsEffect.upper"]]), big.mark = ",")), call. = FALSE)
  }
})
ci <- cumulative(impact)
cat(sprintf("Cumulative incremental sales: %s (95%% posterior interval %s to %s); truth %s\n",
            format(round(ci[["AbsEffect"]]), big.mark = ","), format(round(ci[["AbsEffect.lower"]]), big.mark = ","),
            format(round(ci[["AbsEffect.upper"]]), big.mark = ","), format(round(truth$incremental_sales), big.mark = ",")))
cat(sprintf("Relative effect: %.1f%% (truth %.1f%%); interval half-width: %.0f%% of the estimate\n",
            100 * ci[["RelEffect"]], truth$lift_pct,
            100 * (ci[["AbsEffect.upper"]] - ci[["AbsEffect.lower"]]) / 2 / ci[["AbsEffect"]]))

# %% [markdown]
# Look at the bottom panel: the cumulative effect stays near zero through the pre-period and
# climbs in the test window, with its 95% band widening as the forecast runs further from the data.

# %%
plot(impact)

# %% [markdown]
# **Explain.** Compare with both of your predictions. Why is the interval this wide when the point
# estimate is close to the truth? Point to the width of the band in the plot's first panel.
#
# <details><summary>Why this solution works</summary>
#
# The counterfactual is a forecast, and its uncertainty comes from the regional shocks the controls
# do not share, from which controls belong in the regression, and from the local level, which can
# drift during the ten weeks. Summed over ten weeks those uncertainties add up, so the interval for
# the cumulative effect is wide relative to a 5% lift. It is wider than the Python notebook's
# placebo-in-time interval partly because it is a 95% rather than a 90% interval and partly because
# it includes the drift of the level. A longer pre-period, controls that track the treated total
# better, or more treated geos narrow it.
# </details>

# %% [markdown]
# ## Exercise 2 · What contaminated controls do (5 minutes)
#
# **Predict.** Suppose one treated geo were wrongly left among the controls (it got the campaign,
# but the analysis treats it as a control). Would the estimated effect for the remaining treated
# geos be larger, smaller or unchanged?
#
# **Task.** Write `contaminated_effect(panel, treated_geos, test_start, test_end, niter)` that reruns
# Exercise 1 with the first treated geo moved into the controls, and returns `cumulative()` of the
# result. (Because `run_causal_impact` uses every non-treated geo as a control, passing
# `treated_geos[-1]` does exactly that.)

# %% tags=["exercise"]
contaminated_effect <- function(panel, treated_geos, test_start, test_end, niter) {
  # TODO 2: run_causal_impact() with treated_geos[-1], then cumulative()
  stop("TODO 2")
}

# %% [markdown]
# **Solution 2 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 2 — try it yourself first { display-mode: "form" }
solution(2, "contaminated_effect", function(panel, treated_geos, test_start, test_end, niter) {
  cumulative(run_causal_impact(panel, treated_geos[-1], test_start, test_end, niter))
})

# %% [markdown]
# The checkpoint compares like with like: the same seven treated geos, once with the first treated
# geo among the controls (contaminated) and once with it removed from the data (clean).

# %% tags=["checkpoint"]
checkpoint(2, {
  set.seed(SEED)
  contaminated <- contaminated_effect(panel, TREATED, TEST_START, TEST_END, NITER)
  check_columns(as.list(contaminated), c("AbsEffect", "AbsEffect.lower", "AbsEffect.upper", "RelEffect"),
                name = "the vector contaminated_effect() returns")
  set.seed(SEED)
  clean7 <- cumulative(run_causal_impact(panel[panel$geo != TREATED[1], ], TREATED[-1],
                                         TEST_START, TEST_END, NITER))
  if (!(contaminated[["AbsEffect"]] < clean7[["AbsEffect"]])) {
    stop(sprintf(paste0("The contaminated estimate (%s) is not smaller than the clean one for the same seven ",
                        "treated geos (%s). Check that you dropped only the FIRST treated geo from the treated ",
                        "list (treated_geos[-1]) and kept it in the panel. Fix TODO 2, or run use_reference(2)."),
                 format(round(contaminated[["AbsEffect"]]), big.mark = ","),
                 format(round(clean7[["AbsEffect"]]), big.mark = ",")), call. = FALSE)
  }
})
set.seed(SEED)
impact_cont <- run_causal_impact(panel, TREATED[-1], TEST_START, TEST_END, NITER)
coefs <- impact_cont$model$bsts.model$coefficients
comparison <- data.frame(
  analysis = c("8 treated geos (Exercise 1)", "7 treated geos, first one dropped (clean)",
               "7 treated geos, first one among the controls"),
  incremental_sales = round(c(ci[["AbsEffect"]], clean7[["AbsEffect"]], contaminated[["AbsEffect"]])),
  relative_effect_pct = round(100 * c(ci[["RelEffect"]], clean7[["RelEffect"]], contaminated[["RelEffect"]]), 2)
)
print(comparison, row.names = FALSE)
cat(sprintf("Share of MCMC draws in which %s is in the regression (contaminated model): %.2f\n",
            TREATED[1], mean(coefs[, TREATED[1]] != 0)))

# %% [markdown]
# **Explain.** Compare with your prediction. How does a treated geo among the controls pull the
# estimate down? Use the last printed line.
#
# <details><summary>Why this solution works</summary>
#
# The contaminated geo got the campaign, so its sales rise in the test window. When the regression
# uses it as a predictor, part of that rise flows into the counterfactual, which is then too high,
# and actual minus counterfactual is too small. CausalImpact's documentation states this as the
# first assumption: the controls are not affected by the intervention. Spillover does the same
# damage without any mistake in the analysis: shoppers crossing into a treated region, or a
# national ad buy that reaches the controls.
# </details>

# %% [markdown]
# ## Decision · Carry the Bayesian estimate to the Python notebook
#
# The decision is made in the Python notebook; this cell states the CausalImpact view in the same
# order. **Rule** (fixed in advance): incremental if the 95% posterior interval excludes 0;
# profitable if the incremental margin at the lower end of the interval exceeds the campaign cost.
# The printout also gives CausalImpact's *posterior tail-area probability*: under the model, the
# probability of an effect at least this large arising by chance. Small means the effect is
# unlikely to be noise.

# %%
margin <- GROSS_MARGIN * ci
p_tail <- impact$summary["Cumulative", "p"]
money <- function(x) format(round(x), big.mark = ",")
cat("Number\n")
cat(sprintf("  CausalImpact: %s incremental sales (%.1f%%), 95%% posterior interval %s to %s;",
            money(ci[["AbsEffect"]]), 100 * ci[["RelEffect"]], money(ci[["AbsEffect.lower"]]),
            money(ci[["AbsEffect.upper"]])),
    sprintf("posterior tail-area probability %.3f\n", p_tail))
cat(sprintf("  Incremental margin at %.0f%%: $%s (95%% interval $%s to $%s); campaign cost $%s\n",
            100 * GROSS_MARGIN, money(margin[["AbsEffect"]]), money(margin[["AbsEffect.lower"]]),
            money(margin[["AbsEffect.upper"]]), money(CAMPAIGN_COST)))
cat("Rule\n  Incremental if the interval excludes 0; profitable if margin at its lower end > cost.\n")
cat("Recommendation\n")
cat(sprintf("  Incremental: %s. Profitable at the lower end: %s.\n",
            if (ci[["AbsEffect.lower"]] > 0) "yes" else "not shown",
            if (margin[["AbsEffect.lower"]] > CAMPAIGN_COST) "yes" else "not shown"))
cat(sprintf("  In the Python notebook's decision cell, type: causalimpact_interval = (%.0f, %.0f)\n",
            ci[["AbsEffect.lower"]], ci[["AbsEffect.upper"]]))
cat("  Mode:", if (QUICK) "QUICK run (1,000 MCMC draws): treat this as a rough answer." else
      "FULL run (5,000 MCMC draws).", "\n")

# %% [markdown]
# **Decide.** In one sentence: does the Bayesian view agree with the synthetic control on whether
# the campaign was incremental, and on whether it paid back? If not, which interval would you
# report, and why?

# %% [markdown]
# # Part D · GeoLift market selection (clinic demo; not part of the lab minutes)
#
# Before a geo test, **market selection** asks which markets to treat. GeoLift (Meta's open-source
# geo-testing package, built on *augmented synthetic control*: a synthetic control plus an
# outcome model that corrects what the weights leave unmatched; Ben-Michael, Feller & Rothstein
# 2021) simulates the test on historical data for candidate sets of N treated markets: for each
# set and each effect size it injects the effect, estimates it, and records the power, the smallest effect
# detected (minimum detectable effect, MDE), how well the other markets reproduce the set before the
# test (`AvgScaledL2Imbalance`: smaller is better), and the budget the effect would cost
# (`Investment`, from the cost per incremental unit `cpic`). The example uses GeoLift's bundled
# `GeoLift_PreTest` data: 40 US cities, 90 days.
#
# It runs many simulations, so facilitators run it **before** the clinic; its run time is on the
# readiness page once recorded. Read the table from the top: rank 1 is GeoLift's best trade-off.

# %%
suppressPackageStartupMessages(library(GeoLift))
data(GeoLift_PreTest)
geo_pretest <- GeoDataRead(data = GeoLift_PreTest, date_id = "date", location_id = "location",
                           Y_id = "Y", format = "yyyy-mm-dd", summary = FALSE)
set.seed(SEED)
selection <- GeoLiftMarketSelection(
  data = geo_pretest, treatment_periods = c(15), N = if (QUICK) c(2) else c(2, 3),
  effect_size = seq(0, 0.25, 0.05), lookback_window = 1, cpic = 7.5, budget = 100000, alpha = 0.1,
  Correlations = TRUE, fixed_effects = TRUE, side_of_test = "one_sided", parallel = FALSE, print = FALSE
)
head(selection$BestMarkets[, c("rank", "location", "duration", "EffectSize", "Power", "Average_MDE",
                               "AvgScaledL2Imbalance", "Investment", "ProportionTotal_Y")], 8)

# %% [markdown]
# The power of the top-ranked set at each effect size, from `GeoLiftPower`. With
# `lookback_window = 1` there is one simulated test per effect size, so power is 0 or 1: a quick
# screen, not a precise curve. A real design uses a longer lookback window.

# %%
best_markets <- trimws(strsplit(selection$BestMarkets$location[1], ",")[[1]])
power_best <- GeoLiftPower(data = geo_pretest, locations = best_markets, effect_size = seq(0, 0.25, 0.05),
                           lookback_window = 1, treatment_periods = 15, cpic = 7.5,
                           side_of_test = "one_sided", parallel = FALSE)
power_best[, c("location", "EffectSize", "power", "detected_lift", "Investment", "ScaledL2Imbalance")]

# %% [markdown]
# **For the clinic.** Copy the chosen markets, their MDE and the investment into your design
# template, and compare the MDE with your break-even lift.
#
# **A known issue.** In the workshop image (GeoLift 2.7.5 and augsynth 0.2.0 at the pinned commits),
# GeoLift's analysis function `GeoLift()` stopped with an error from augsynth (`Tibble columns must
# have compatible sizes`) when given two or more test markets, including GeoLift's own
# two-market example; with one market it ran (checked 2026-10-09). Market selection and power, above,
# are not affected. To analyze a multi-market test, aggregate the treated markets into one series
# first (as Parts A to C do) or use CausalImpact.

# %% [markdown]
# ## Stretch (optional) · Synthetic control with augsynth
#
# Fit the synthetic control of the treated total with the `augsynth` package (Ben-Michael, Feller &
# Rothstein 2021), which GeoLift builds on, and compare its estimate and interval with Exercise 1 and
# with the Python notebook. Build a panel with one "TREATED" unit (the treated geos' total) and the
# 32 control geos, index each unit to its pre-period mean as the Python notebook does, and call
# `augsynth(index ~ campaign, geo, week, data = ..., progfunc = "none", scm = TRUE)`; then
# `summary(fit, inf_type = "jackknife+", alpha = 0.05)`. `progfunc = "ridge"` adds the outcome-model
# augmentation the method is named for, but its default cross-validation of the penalty is much
# slower; try it if you have time.

# %% tags=["solution"]
#@title Stretch solution — try it yourself first { display-mode: "form" }
suppressPackageStartupMessages(library(augsynth))
weekly <- panel
weekly$week <- as.integer(factor(weekly$date))
weekly$post <- as.integer(weekly$date >= TEST_START)
treated_total <- aggregate(sales ~ week + post, data = weekly[weekly$geo %in% TREATED, ], FUN = sum)
treated_total$geo <- "TREATED"
units <- rbind(treated_total[c("geo", "week", "post", "sales")],
               weekly[!weekly$geo %in% TREATED, c("geo", "week", "post", "sales")])
pre_means <- tapply(units$sales[units$post == 0], units$geo[units$post == 0], mean)
units$index <- units$sales / pre_means[units$geo]
units$campaign <- as.integer(units$geo == "TREATED" & units$post == 1)
fit_as <- augsynth(index ~ campaign, geo, week, data = units, progfunc = "none", scm = TRUE)
att <- summary(fit_as, inf_type = "jackknife+", alpha = 0.05)$average_att
to_sales <- pre_means[["TREATED"]] * truth$test_weeks  # index points per week -> sales over the test
cat(sprintf("augsynth: %s incremental sales (95%% jackknife+ interval %s to %s); truth %s\n",
            money(att$Estimate * to_sales), money(att$lower_bound * to_sales),
            money(att$upper_bound * to_sales), money(truth$incremental_sales)))
