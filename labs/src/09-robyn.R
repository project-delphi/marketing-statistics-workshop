# %% [markdown]
# <!--
# APIs this notebook calls, checked on 2026-10-09 with args() and the package help inside the
# workshop image (mktstats-env, Robyn 3.12.1, reticulate 1.47.0, nevergrad 1.0.12), and against
# Robyn's demo/demo.R and demo/install_nevergrad.R on GitHub main (downloaded 2026-10-09):
# - robyn_inputs(dt_input, dep_var, dep_var_type, date_var, paid_media_spends, paid_media_vars,
#   organic_vars, context_vars, factor_vars, dt_holidays, prophet_vars, prophet_country, adstock,
#   window_start, window_end, ...) and robyn_inputs(InputCollect =, hyperparameters =).
# - robyn_run(InputCollect, cores, iterations, trials, ts_validation, add_penalty_factor,
#   seed = 123L, quiet = FALSE): quiet = TRUE fails in 3.12.1 ("object 'pb' not found"), so the
#   notebook hides the progress bar with capture.output instead. robyn_mmm seeds numpy, and so
#   Nevergrad, only `if (is.integer(seed))`: seed = 123 (a double) leaves the search unseeded
#   and the results change between runs; 123L gave identical runs (checked twice in the image).
#   The demo recommends iterations = 2000, trials = 5 for dt_simulated_weekly.
# - robyn_outputs(InputCollect, OutputModels, pareto_fronts = "auto", min_candidates, csv_out =
#   NULL, clusters = FALSE, export = FALSE, plot_pareto = FALSE, quiet): writes no files with
#   these settings (checked); $resultHypParam has solID, nrmse, decomp.rssd, robynPareto;
#   $allSolutions are the Pareto-optimal models. It has no selectID, which robyn_response would
#   otherwise use in place of select_model.
# - robyn_response(InputCollect, OutputCollect, select_model, metric_name, metric_value,
#   date_range, quiet): with metric_value (the spend over date_range) it returns sim_mean_spend
#   and sim_mean_response; the demo's marginal ROAS is the difference of two calls' responses
#   over the difference of their spends.
# - robyn_allocator(InputCollect, OutputCollect, select_model, channel_constr_low,
#   channel_constr_up, scenario = "max_response", plots, export, quiet) -> $dt_optimOut.
# - Robyn's search proposes `cores` candidates per round, so results depend on cores; the
#   notebook fixes it at 2 (Colab's CPU count) so every runtime does the same search.
# - reticulate: py_available(initialize = FALSE), py_module_available(), py_config(), import(),
#   virtualenv_create(envname, python, packages), virtualenv_install(envname, packages,
#   pip_options), virtualenv_python(envname); RETICULATE_PYTHON picks the Python before
#   reticulate starts one.
# -->
#
# # Part D · Another tool, another answer
#
# The Python notebook fitted a Bayesian marketing mix model (MMM) with PyMC-Marketing. Here you
# run **Robyn**, Meta's open-source MMM package for R, on Robyn's own example data, and ask the
# same question: which channel's return can you act on?
#
# Robyn fits a *ridge regression* (a linear regression whose coefficients are shrunk towards
# zero) of revenue on adstocked and saturated media, with trend, season and holidays from
# Prophet. The adstock and saturation parameters are not estimated by the regression: the Python
# library **Nevergrad** searches over them, scoring each candidate on two errors at once,
# prediction error (NRMSE) and how far the media's share of the effect is from its share of
# spend (DECOMP.RSSD). The models that no other model beats on both errors form the *Pareto
# front*, and the analyst picks among them. So Robyn's "uncertainty" is the spread across those
# models, not a posterior.
#
# Two caveats before you start:
#
# - **Settings.** This notebook runs 500 iterations × 1 trial (200 with QUICK). Robyn's demo
#   recommends 2,000 iterations × 5 trials for this dataset, so the results show how Robyn
#   works, not reliable estimates; Robyn itself warns that it is too few.
# - **Maintenance.** Robyn is effectively unmaintained: the last commit on its GitHub main
#   branch is from 2025-06-27 and the CRAN release 3.12.1 from 2025-07-02 (checked 2026-10-09).
#
# **Run.** First, Python's Nevergrad. On Colab this cell installs it (about a minute, not yet
# measured on Colab); in the workshop image it is already there. Its time counts as install
# time in the run record. If it fails, skip this notebook: the Python notebook stands alone.

# %%
.ng_t0 <- Sys.time()
local({
  can_import <- function(py) {
    nzchar(py) && file.exists(py) &&
      identical(suppressWarnings(system2(py, c("-c", shQuote("import nevergrad")),
                                         stdout = FALSE, stderr = FALSE)), 0L)
  }
  if (requireNamespace("reticulate", quietly = TRUE) && reticulate::py_available(initialize = FALSE)) {
    return(invisible())  # Python already runs in this session (a rerun): keep it
  }
  py <- Sys.getenv("RETICULATE_PYTHON")
  if (can_import(py)) return(invisible())  # the workshop image sets RETICULATE_PYTHON
  py <- unname(Sys.which("python3"))
  hosted <- nzchar(Sys.getenv("COLAB_RELEASE_TAG")) || nzchar(Sys.getenv("SAGEMAKER_APP_TYPE"))
  if (!can_import(py) && hosted && nzchar(py)) {
    # The same pins as the Python notebooks: environment/requirements.txt at the workshop's ref.
    root <- Sys.getenv("MKTSTATS_REPO_ROOT")
    pins <- if (nzchar(root)) file.path(root, "environment", "requirements.txt") else
      sprintf("%s/%s/environment/requirements.txt", getOption("mktstats.raw"), MKTSTATS_REF)
    message("Installing the Python library nevergrad for Robyn ...")
    system2(py, c("-m", "pip", "install", "-q", "nevergrad", "-c", shQuote(pins)))
    if (!can_import(py)) {  # this Python refuses packages: give Robyn its own environment,
      # reusing the packages already installed (numpy, pandas, scipy on Colab)
      reticulate::virtualenv_create("r-robyn", python = py, packages = NULL,
                                    system_site_packages = TRUE)
      reticulate::virtualenv_install("r-robyn", "nevergrad", pip_options = c("-c", pins))
      py <- reticulate::virtualenv_python("r-robyn")
    }
  }
  if (can_import(py)) Sys.setenv(RETICULATE_PYTHON = py)
})
.ng_secs <- as.numeric(difftime(Sys.time(), .ng_t0, units = "secs"))
.mkt_install <- suppressWarnings(as.numeric(Sys.getenv("MKTSTATS_INSTALL_SECONDS", "0")))
Sys.setenv(MKTSTATS_INSTALL_SECONDS = sprintf("%.1f", (if (is.na(.mkt_install)) 0 else .mkt_install) + .ng_secs))
cat(sprintf("Nevergrad setup took %.1f s\n", .ng_secs))

# %% [markdown]
# A provided check: Robyn loads and reticulate finds Nevergrad.

# %%
suppressPackageStartupMessages({
  library(Robyn)
  library(reticulate)
  library(ggplot2)
})
options(repr.plot.width = 9, repr.plot.height = 4, lifecycle_verbosity = "quiet")
checkpoint("Nevergrad", {
  if (!py_module_available("nevergrad")) {
    stop("Robyn needs the Python library nevergrad, and reticulate cannot import it from ",
         py_config()$python, ". Rerun the cell above; if it fails again, restart the runtime ",
         "(Runtime > Restart session) and run all cells again. To move on without Robyn, skip ",
         "this notebook: the Python notebook does not depend on it.", call. = FALSE)
  }
})
cat("Robyn", as.character(packageVersion("Robyn")), "· reticulate",
    as.character(packageVersion("reticulate")), "· Python", as.character(py_config()$version), "·",
    "nevergrad", import("nevergrad")$`__version__`, "· QUICK =", QUICK, "\n")

# %% [markdown]
# The data: `dt_simulated_weekly`, which Robyn's documentation calls "Simulated MMM data":
# 208 weeks of revenue, spend on five paid channels (tv, out-of-home, print, Facebook, search),
# Facebook impressions and search clicks, a newsletter, a competitor's sales and an events flag.
# Nobody published its true parameters, so here nothing can be scored against truth: you can
# only compare models with each other.

# %%
data("dt_simulated_weekly")
data("dt_prophet_holidays")
cat(nrow(dt_simulated_weekly), "weeks,", format(min(dt_simulated_weekly$DATE)), "to",
    format(max(dt_simulated_weekly$DATE)), "\n")
head(as.data.frame(dt_simulated_weekly), 3)

# %% [markdown]
# **Run.** Describe the model to Robyn (the demo's specification), then the ranges Nevergrad
# may search for each medium: geometric adstock `theta` (the share of a week's effect carried
# into the next week) and the Hill saturation curve's `alpha` (shape) and `gamma` (where the
# curve bends). Facebook and search are modelled through impressions and clicks; Robyn links
# them back to spend.

# %%
InputCollect <- robyn_inputs(
  dt_input = dt_simulated_weekly, dt_holidays = dt_prophet_holidays,
  date_var = "DATE", dep_var = "revenue", dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "holiday"), prophet_country = "DE",
  context_vars = c("competitor_sales_B", "events"),
  paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
  paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"),
  organic_vars = c("newsletter"), factor_vars = c("events"),
  window_start = "2016-01-01", window_end = "2018-12-31", adstock = "geometric"
)
hyperparameters <- list(  # the ranges of Robyn's demo
  facebook_I_alphas = c(0.5, 3), facebook_I_gammas = c(0.3, 1), facebook_I_thetas = c(0, 0.3),
  print_S_alphas = c(0.5, 1), print_S_gammas = c(0.3, 1), print_S_thetas = c(0.1, 0.4),
  tv_S_alphas = c(0.5, 1), tv_S_gammas = c(0.3, 1), tv_S_thetas = c(0.3, 0.8),
  search_clicks_P_alphas = c(0.5, 3), search_clicks_P_gammas = c(0.3, 1),
  search_clicks_P_thetas = c(0, 0.3),
  ooh_S_alphas = c(0.5, 1), ooh_S_gammas = c(0.3, 1), ooh_S_thetas = c(0.1, 0.4),
  newsletter_alphas = c(0.5, 3), newsletter_gammas = c(0.3, 1), newsletter_thetas = c(0.1, 0.4),
  train_size = c(0.5, 0.8)
)
InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)

# %% [markdown]
# **Run.** The search: 500 candidate models (200 with QUICK) in one trial, far below the
# 2,000 × 5 that Robyn's demo recommends; Robyn warns about it and reports, for each of its two
# errors, whether the search has converged. About a minute (an estimate for Colab; measured in
# the workshop image on 2 CPUs: see the run record). While it runs, read Exercise 1's Predict
# prompt.

# %%
ITERATIONS <- if (QUICK) 200 else 500
ROBYN_CORES <- 2  # Robyn proposes this many candidates per round: fixed so every runtime searches alike
started <- Sys.time()
invisible(capture.output({  # hides the text progress bars; quiet = TRUE fails in Robyn 3.12.1
  OutputModels <- robyn_run(InputCollect = InputCollect, cores = ROBYN_CORES,
                            iterations = ITERATIONS, trials = 1, ts_validation = FALSE,
                            add_penalty_factor = FALSE,
                            seed = 123L)  # an integer: Robyn seeds Nevergrad only then
  OutputCollect <- robyn_outputs(InputCollect, OutputModels, pareto_fronts = "auto",
                                 min_candidates = 5, csv_out = NULL, clusters = FALSE,
                                 export = FALSE, plot_pareto = FALSE, quiet = TRUE)
}))
run_seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))

# Rank the Pareto-optimal models by prediction error. Robyn asks an analyst to choose among
# them; for a reproducible notebook we take the one with the lowest NRMSE.
pareto <- as.data.frame(OutputCollect$resultHypParam)
pareto <- pareto[pareto$solID %in% OutputCollect$allSolutions, c("solID", "nrmse", "decomp.rssd")]
pareto <- pareto[order(pareto$nrmse), ]
select_model <- pareto$solID[1]
cat(sprintf("%d candidates x 1 trial in %.1f s; %d Pareto-optimal models; selected %s (NRMSE %.3f)\n",
            ITERATIONS, run_seconds, nrow(pareto), select_model, pareto$nrmse[1]))

# %% [markdown]
# Each channel's mean weekly spend over the modelling window (2016 to 2018), the spend level the
# exercise starts from:

# %%
window <- InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich
paid <- InputCollect$paid_media_spends
mean_spend <- colMeans(as.data.frame(InputCollect$dt_input)[window, paid])
round(mean_spend)

# %% [markdown]
# ## Exercise 1 · Robyn's marginal ROAS (10 minutes)
#
# The *marginal ROAS* of a channel is the revenue the next dollar brings, at the current spend:
# $(R(s_2) - R(s_1)) / (s_2 - s_1)$ for two nearby spend levels $s_1 < s_2$, where $R$ is the
# channel's response curve. It is what a budget decision needs, and it is lower than the
# average ROAS when the curve saturates.
#
# **Predict.** Robyn ran 500 iterations × 1 trial instead of 2,000 × 5. Will its five best
# models (lowest NRMSE) agree on which channel has the highest marginal ROAS: yes or no? Write
# it down.
#
# **Task.** Write `robyn_marginal_roas(InputCollect, OutputCollect, select_model, channel,
# spend1, spend2)`. Call `robyn_response(InputCollect = , OutputCollect = , select_model = ,
# metric_name = channel, metric_value = spend, date_range = "last_1", quiet = TRUE)` once at each
# spend (with `date_range = "last_1"`, `metric_value` is one week's spend) and return
# (`sim_mean_response` at spend2 − at spend1) ÷ (`sim_mean_spend` at spend2 − at spend1), the
# pattern in Robyn's demo.

# %% tags=["exercise"]
robyn_marginal_roas <- function(InputCollect, OutputCollect, select_model, channel, spend1, spend2) {
  # TODO 1: two robyn_response() calls, then the difference in response over the difference in spend
  stop("TODO 1")
}

# %% [markdown]
# **Solution 1 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 1 — try it yourself first { display-mode: "form" }
solution(1, "robyn_marginal_roas", function(InputCollect, OutputCollect, select_model, channel,
                                            spend1, spend2) {
  at <- function(spend) {
    robyn_response(InputCollect = InputCollect, OutputCollect = OutputCollect,
                   select_model = select_model, metric_name = channel, metric_value = spend,
                   date_range = "last_1", quiet = TRUE)
  }
  r1 <- at(spend1)
  r2 <- at(spend2)
  (r2$sim_mean_response - r1$sim_mean_response) / (r2$sim_mean_spend - r1$sim_mean_spend)
})

# %% [markdown]
# **Explain.** The checkpoint below computes it for every paid channel at its mean weekly spend
# and 10% above. Which channel's next dollar returns most in the selected model? Then compare
# with your prediction using the table after the checkpoint.
#
# <details><summary>Why this solution works</summary>
#
# `robyn_response` evaluates the selected model's response curve (adstock, then Hill
# saturation, times the ridge coefficient) at a given spend, so two calls a small step apart
# give the slope of the curve there: the return on the next dollar. Swapping the two spends
# changes the sign of both differences and leaves the ratio unchanged. The average ROAS, total
# response over total spend, would be higher on a saturating curve and is the wrong number for
# moving the next dollar. Facebook and search are modelled through impressions and clicks;
# `robyn_response` converts spend to them with the spend-exposure fit Robyn made in
# `robyn_inputs`.
# </details>

# %% tags=["checkpoint"]
checkpoint(1, {
  mroas <- vapply(paid, function(ch) {
    robyn_marginal_roas(InputCollect, OutputCollect, select_model, ch, mean_spend[[ch]],
                        1.1 * mean_spend[[ch]])
  }, numeric(1))
  if (any(!is.finite(mroas)) || any(mroas < 0) || !any(mroas > 0)) {
    stop(sprintf(paste0("Checkpoint 1: the marginal ROAS should be a finite number of at least 0 ",
                        "for every channel; yours are %s. This usually means a ratio the wrong way ",
                        "round or a missing difference. Check that you divide the change in ",
                        "sim_mean_response by the change in sim_mean_spend. To move on now, run ",
                        "use_reference(1)."), paste(names(mroas), signif(mroas, 3), collapse = ", ")),
         call. = FALSE)
  }
  swapped <- robyn_marginal_roas(InputCollect, OutputCollect, select_model, "tv_S",
                                 1.1 * mean_spend[["tv_S"]], mean_spend[["tv_S"]])
  check_close(swapped, mroas[["tv_S"]], rel = 1e-9,
              name = paste("tv_S's marginal ROAS with spend1 and spend2 swapped (a ratio of two",
                           "differences does not change; an average ROAS would)"))
  r1 <- robyn_response(InputCollect = InputCollect, OutputCollect = OutputCollect,
                       select_model = select_model, metric_name = "print_S",
                       metric_value = mean_spend[["print_S"]], date_range = "last_1", quiet = TRUE)
  r2 <- robyn_response(InputCollect = InputCollect, OutputCollect = OutputCollect,
                       select_model = select_model, metric_name = "print_S",
                       metric_value = 1.1 * mean_spend[["print_S"]], date_range = "last_1",
                       quiet = TRUE)
  check_close(mroas[["print_S"]],
              (r2$sim_mean_response - r1$sim_mean_response) / (r2$sim_mean_spend - r1$sim_mean_spend),
              rel = 1e-9, name = "print_S's marginal ROAS against Robyn's demo formula")
})

# %%
round(mroas, 2)

# %% [markdown]
# Do the five best models agree? The next cell repeats your function for each of them. Look at
# whether each channel's dots sit on one side of the dashed line, the break-even marginal ROAS
# of 3.3 (a 30% gross margin, as in the Python notebook; Robyn's simulated data come with no
# margin, so this is our assumption).

# %%
GROSS_MARGIN <- 0.30
BREAK_EVEN <- 1 / GROSS_MARGIN
top_models <- head(pareto$solID, 5)
by_model <- sapply(top_models, function(m) vapply(paid, function(ch) {
  robyn_marginal_roas(InputCollect, OutputCollect, m, ch, mean_spend[[ch]], 1.1 * mean_spend[[ch]])
}, numeric(1)))
by_model <- matrix(by_model, nrow = length(paid), dimnames = list(paid, top_models))
best <- apply(by_model, 2, function(x) rownames(by_model)[which.max(x)])
cat("Highest marginal ROAS in each of the", length(top_models), "best models:",
    paste(sprintf("%s: %s", names(best), best), collapse = "; "), "\n")
long <- data.frame(channel = rep(paid, times = ncol(by_model)),
                   model = rep(colnames(by_model), each = length(paid)),
                   mroas = as.vector(by_model))
ggplot(long, aes(x = mroas, y = channel)) +
  geom_point(aes(shape = model), size = 3, alpha = 0.8) +
  geom_vline(xintercept = BREAK_EVEN, linetype = "dashed") +
  annotate("text", x = BREAK_EVEN, y = Inf, label = " break-even 3.3", hjust = 0, vjust = 1.5) +
  labs(x = "marginal ROAS (revenue per extra dollar, at mean weekly spend)", y = NULL,
       shape = "Pareto-optimal model") +
  theme_minimal(base_size = 13)

# %% [markdown]
# **Explain.** Compare with your prediction. The models fit the revenue about equally well, yet
# disagree on some channels. Why can two models with similar errors give a channel very
# different marginal returns, and what would make them agree?
#
# <details><summary>What is going on</summary>
#
# Like the Bayesian model, Robyn learns a channel's curve from how revenue moves when that
# channel's spend moves. Where the data say little, many curves fit equally well, and the
# Pareto front keeps several of them. The spread across those models is Robyn's version of a
# wide posterior. More iterations and trials explore the front better, but they cannot add
# information the data do not have. A lift test does: Robyn takes experiments through the
# `calibration_input` argument of `robyn_inputs`, for the same reason the Python notebook adds
# them to the likelihood.
# </details>
#
# **Run.** Robyn's budget allocator on the selected model, as in Robyn's demo: keep the total
# spend and let each channel move between 70% and 120% (tv) or 150% (the others) of its
# current spend, to maximize the response.

# %%
AllocatorCollect <- robyn_allocator(
  InputCollect = InputCollect, OutputCollect = OutputCollect, select_model = select_model,
  channel_constr_low = 0.7, channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5),
  scenario = "max_response", plots = FALSE, export = FALSE, quiet = TRUE
)
allocation <- as.data.frame(AllocatorCollect$dt_optimOut)[
  , c("channels", "initSpendUnit", "optmSpendUnit", "initResponseUnit", "optmResponseUnit")]
allocation$spend_change <- allocation$optmSpendUnit / allocation$initSpendUnit - 1
allocation[, -1] <- round(allocation[, -1], 2)
allocation

# %% [markdown]
# ## Decision · Act on Robyn's answer, or test first?
#
# **Number.** Per paid channel: the marginal ROAS in the selected model and its lowest and
# highest value across the five best models.
#
# **Rule.** Move budget on a channel's marginal ROAS only if all five models put it on the same
# side of break-even (3.3 at a 30% margin): above means more spend pays, below means less. If
# they disagree, the channel needs a lift test before money moves.

# %%
decision <- data.frame(
  channel = paid,
  selected = round(by_model[, 1], 2),
  lowest = round(apply(by_model, 1, min), 2),
  highest = round(apply(by_model, 1, max), 2),
  spend_change_by_allocator = allocation$spend_change[match(
    InputCollect$paid_media_vars[match(paid, InputCollect$paid_media_spends)], allocation$channels)]
)
decision$verdict <- ifelse(decision$lowest > BREAK_EVEN, "all models: above break-even",
                    ifelse(decision$highest < BREAK_EVEN, "all models: below break-even",
                           "models disagree: test first"))
cat("Break-even marginal ROAS", round(BREAK_EVEN, 2), "(30% margin, assumed).",
    if (QUICK) "QUICK run (200 iterations): treat this as a rough answer." else
      "FULL run (500 iterations x 1 trial, below Robyn's recommended 2,000 x 5).", "\n")
decision

# %% [markdown]
# **Recommendation.** Write one sentence: which channels' marginal ROAS you would move budget on
# from this Robyn run, which need a test first, and what would change your answer (Robyn's
# recommended 2,000 × 5 run, or a lift test passed in `calibration_input`). Then compare it with
# your Python notebook's decision: same kind of answer, different tool.
#
# ```text
# Your sentence: ________________________________________________
# ```

# %% [markdown]
# ## Stretch (optional) · The recommended search
#
# Rerun the search with Robyn's recommended `iterations = 2000, trials = 5` (20 times the
# candidates of this notebook; expect well over 10 minutes on 2 CPUs, an estimate) and repeat the
# table of the five best models. Do they agree more? Does Robyn now report convergence?
#
# <details><summary>Code</summary>
#
# ```r
# invisible(capture.output(
#   OutputModels_full <- robyn_run(InputCollect = InputCollect, cores = ROBYN_CORES,
#                                  iterations = 2000, trials = 5, ts_validation = FALSE,
#                                  add_penalty_factor = FALSE, seed = 123L)
# ))
# OutputCollect_full <- robyn_outputs(InputCollect, OutputModels_full, pareto_fronts = "auto",
#                                     csv_out = NULL, clusters = FALSE, export = FALSE,
#                                     plot_pareto = FALSE, quiet = TRUE)
# hp <- as.data.frame(OutputCollect_full$resultHypParam)
# hp <- hp[hp$solID %in% OutputCollect_full$allSolutions, ]
# top_full <- head(hp$solID[order(hp$nrmse)], 5)
# sapply(top_full, function(m) vapply(paid, function(ch)
#   robyn_marginal_roas(InputCollect, OutputCollect_full, m, ch, mean_spend[[ch]],
#                       1.1 * mean_spend[[ch]]), numeric(1)))
# ```
# </details>
