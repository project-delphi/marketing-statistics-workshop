# ---
# jupyter:
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # Spike S3 · Colab R runtime: repositories, binary installs, timings
#
# Throwaway notebook (see DECISIONS.md). It records how Colab's R runtime is set up, installs the
# workshop's R packages from a dated Posit Public Package Manager snapshot, installs augsynth and
# GeoLift from pinned GitHub archives (no GitHub API), runs one tiny example per package, and prints
# a JSON record between markers.

# %%
T0 <- Sys.time()
rec <- list(
  spike = "s3-colab-r",
  colab_release = Sys.getenv("COLAB_RELEASE_TAG"),
  r = R.version.string,
  os = tryCatch(system("lsb_release -ds", intern = TRUE), error = function(e) NA),
  cpus = parallel::detectCores(),
  repos_before = as.list(getOption("repos")),
  libpaths = .libPaths(),
  n_installed_before = nrow(installed.packages()),
  bspm_loaded = "bspm" %in% loadedNamespaces(),
  bspm_installed = "bspm" %in% rownames(installed.packages()),
  gsl = length(system("ldconfig -p | grep -c libgsl", intern = TRUE)) > 0,
  timings = list(),
  errors = list()
)
rec$gsl_libs <- tryCatch(system("ldconfig -p | grep libgsl", intern = TRUE), error = function(e) character())
timed <- function(name, expr) {
  t <- Sys.time()
  res <- tryCatch({ force(expr); NULL }, error = function(e) conditionMessage(e))
  rec$timings[[name]] <<- round(as.numeric(difftime(Sys.time(), t, units = "secs")), 1)
  if (!is.null(res)) { rec$errors[[name]] <<- substr(res, 1, 300); message("[", name, "] FAILED: ", res) }
  else message("[", name, "] ", rec$timings[[name]], " s")
}
str(rec[c("colab_release", "r", "os", "cpus", "repos_before", "bspm_loaded", "bspm_installed", "gsl_libs")])

# %% tags=["solution"]
# A cell marked as a folded solution (cellView form + source_hidden): is it hidden in the R runtime?
hidden_cell_ran <- TRUE

# %%
# Dated P3M snapshot with the HTTPUserAgent P3M needs to serve binaries.
options(
  repos = c(CRAN = "https://p3m.dev/cran/__linux__/noble/2026-10-01"),
  HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
                          paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])),
  timeout = 900
)
timed("apt_gsl", {
  if (!length(rec$gsl_libs)) {
    st <- system("apt-get install -y -qq libgsl27 > /tmp/apt.log 2>&1")
    if (st != 0) stop("apt-get failed: ", paste(tail(readLines("/tmp/apt.log"), 5), collapse = " | "))
  }
})

# %%
need <- function(pk) setdiff(pk, rownames(installed.packages()))
timed("install_clvtools", install.packages(need("CLVTools")))
timed("install_grf", install.packages(need("grf")))
timed("install_causalimpact", install.packages(need("CausalImpact")))
timed("install_geolift_deps", install.packages(need(c(
  "gsynth", "MarketMatching", "panelView", "directlabels", "doParallel", "foreach", "gridExtra",
  "progress", "lifecycle", "osqp", "FNN", "Formula", "dplyr", "tidyr", "ggplot2", "scales"))))
timed("install_augsynth_geolift", {
  install.packages("https://github.com/ebenmichael/augsynth/archive/7e70072232fe75057aa8b5d7af08c7e1aef53ae8.tar.gz",
                   repos = NULL, type = "source")
  install.packages("https://github.com/facebookincubator/GeoLift/archive/db34ea4299ff0e28515ebac502b78c076c93c905.tar.gz",
                   repos = NULL, type = "source")
})
rec$versions <- lapply(c(CLVTools = "CLVTools", grf = "grf", CausalImpact = "CausalImpact", bsts = "bsts",
                         augsynth = "augsynth", GeoLift = "GeoLift"),
                       function(p) tryCatch(as.character(packageVersion(p)), error = function(e) NA))
str(rec$versions)

# %%
timed("clvtools_pnbd_static_cov", {
  library(CLVTools)
  data("apparelTrans"); data("apparelStaticCov")
  clv <- clvdata(apparelTrans, date.format = "ymd", time.unit = "week", estimation.split = 40,
                 name.id = "Id", name.date = "Date", name.price = "Price")
  clv_cov <- SetStaticCovariates(clv, data.cov.life = apparelStaticCov, data.cov.trans = apparelStaticCov,
                                 names.cov.life = "Gender", names.cov.trans = "Gender", name.id = "Id")
  fit <- pnbd(clv_cov)
  print(coef(fit))
})

# %%
timed("grf_causal_forest_20k", {
  library(grf)
  n <- 20000; X <- matrix(rnorm(n * 5), n); W <- rbinom(n, 1, 0.5)
  Y <- X[, 1] + W * (0.5 + X[, 2]) + rnorm(n)
  cf <- causal_forest(X, Y, W, num.trees = 500)
  print(average_treatment_effect(cf))
})

# %%
timed("causalimpact_example", {
  library(CausalImpact)
  set.seed(1)
  x1 <- 100 + arima.sim(model = list(ar = 0.999), n = 100)
  y <- 1.2 * x1 + rnorm(100); y[71:100] <- y[71:100] + 10
  impact <- CausalImpact(cbind(y, x1), c(1, 70), c(71, 100))
  print(summary(impact))
})

# %%
timed("geolift_load", {
  library(GeoLift)
  data(GeoLift_PreTest)
  print(dim(GeoLift_PreTest))
})

# %%
rec$hidden_cell_ran <- exists("hidden_cell_ran")
rec$total_seconds <- round(as.numeric(difftime(Sys.time(), T0, units = "secs")), 1)
cat("----- spike record -----\n")
cat(jsonlite::toJSON(rec, auto_unbox = TRUE, null = "null"), "\n")
cat("----- end -----\n")
