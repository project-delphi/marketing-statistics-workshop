# Spike S5 · Robyn feasibility (1-hour timebox). Runs inside the workshop image as root:
#   docker run --rm --user root -v "$PWD":/work -w /work mktstats-env:spike Rscript spikes/s5_robyn.R
# Follows the official demo (github.com/facebookexperimental/Robyn/blob/main/demo/demo.R, read 2026-10-09):
# robyn_inputs -> hyperparameters -> robyn_run -> robyn_outputs -> robyn_allocator, with small
# iterations/trials to measure feasibility, not to produce a usable model.
t0 <- Sys.time(); tm <- list()
tick <- function(name, expr) { t <- Sys.time(); force(expr); tm[[name]] <<- round(as.numeric(difftime(Sys.time(), t, units = "secs")), 1); message("[", name, "] ", tm[[name]], " s") }
options(timeout = 900)
tick("install_robyn", install.packages(setdiff(c("Robyn", "reticulate"), rownames(installed.packages()))))
tick("install_nevergrad", system("uv pip install --python /opt/venv/bin/python nevergrad"))
Sys.setenv(RETICULATE_PYTHON = "/opt/venv/bin/python")
library(Robyn); library(reticulate)
print(reticulate::py_config())
data("dt_simulated_weekly"); data("dt_prophet_holidays")
tick("inputs", {
  InputCollect <- robyn_inputs(
    dt_input = dt_simulated_weekly, dt_holidays = dt_prophet_holidays,
    date_var = "DATE", dep_var = "revenue", dep_var_type = "revenue",
    prophet_vars = c("trend", "season", "holiday"), prophet_country = "DE",
    context_vars = c("competitor_sales_B", "events"),
    paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
    paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"),
    organic_vars = c("newsletter"), factor_vars = c("events"),
    window_start = "2016-01-01", window_end = "2018-12-31", adstock = "geometric")
  hyperparameters <- list(
    facebook_I_alphas = c(0.5, 3), facebook_I_gammas = c(0.3, 1), facebook_I_thetas = c(0, 0.3),
    print_S_alphas = c(0.5, 1), print_S_gammas = c(0.3, 1), print_S_thetas = c(0.1, 0.4),
    tv_S_alphas = c(0.5, 1), tv_S_gammas = c(0.3, 1), tv_S_thetas = c(0.3, 0.8),
    search_clicks_P_alphas = c(0.5, 3), search_clicks_P_gammas = c(0.3, 1), search_clicks_P_thetas = c(0, 0.3),
    ooh_S_alphas = c(0.5, 1), ooh_S_gammas = c(0.3, 1), ooh_S_thetas = c(0.1, 0.4),
    newsletter_alphas = c(0.5, 3), newsletter_gammas = c(0.3, 1), newsletter_thetas = c(0.1, 0.4),
    train_size = c(0.5, 0.8))
  InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)
})
tick("run_500x1", OutputModels <- robyn_run(InputCollect = InputCollect, cores = 2, iterations = 500, trials = 1, ts_validation = FALSE, add_penalty_factor = FALSE))
tick("outputs", OutputCollect <- robyn_outputs(InputCollect, OutputModels, pareto_fronts = "auto", min_candidates = 5, csv_out = NULL, clusters = FALSE, export = FALSE, plot_pareto = FALSE))
sel <- OutputCollect$allSolutions[1]
tick("allocator", AllocatorCollect <- robyn_allocator(InputCollect = InputCollect, OutputCollect = OutputCollect, select_model = sel, channel_constr_low = 0.7, channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5), scenario = "max_response", export = FALSE))
print(AllocatorCollect$dt_optimOut[, c("channels", "initSpendUnit", "optmSpendUnit")])
tm$total <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
cat("----- spike record -----\n"); cat(jsonlite::toJSON(list(spike = "s5-robyn", robyn = as.character(packageVersion("Robyn")), timings = tm), auto_unbox = TRUE), "\n"); cat("----- end -----\n")
