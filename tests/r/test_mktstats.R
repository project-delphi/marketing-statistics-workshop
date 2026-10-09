# Tests of R/mktstats.R, run by tests/test_r_helpers.py (needs R >= 4.5 with jsonlite: the
# workshop image). Usage: Rscript tests/r/test_mktstats.R <repo root>
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[1] else "."
Sys.unsetenv(c("MKTSTATS_WORKED", "MKTSTATS_QUICK", "MKTSTATS_SABOTAGE", "COLAB_RELEASE_TAG"))
source(file.path(root, "R", "mktstats.R"))

failures <- 0
expect <- function(label, ok) {
  if (isTRUE(ok)) cat("ok  ", label, "\n") else { cat("FAIL", label, "\n"); failures <<- failures + 1 }
}
fails_with <- function(expr, pattern) {
  msg <- tryCatch({ force(expr); NA_character_ }, error = function(e) conditionMessage(e))
  !is.na(msg) && grepl(pattern, msg)
}
captured <- function(expr) {
  out <- character()
  err <- NULL
  out <- utils::capture.output(err <- tryCatch({ force(expr); NULL }, error = function(e) e))
  list(text = paste(out, collapse = "\n"), error = err)
}

# ---- solutions and checkpoints ---------------------------------------------------------------
.ws_start("r/toy", strrep("0", 16), strrep("1", 16), "main", list(`1` = "double"))
double <- function(x) x + x + 1  # your (wrong) code
solution(1, "double", function(x) 2 * x)
expect("a solution does not replace your code", double(3) == 7)
use_reference(1)
expect("use_reference binds the reference", double(3) == 6)

double <- function(x) stop("TODO 1")
solution(1, "double", function(x) 2 * x)
r <- captured(checkpoint(1, { stopifnot(double(2) == 4) }))
expect("an unwritten TODO re-raises", !is.null(r$error))
expect("and prints the recovery message",
       grepl("TODO 1 is not written yet", r$text) && grepl("use_reference\\(1\\)", r$text))

double <- function(x) x + 1
solution(1, "double", function(x) 2 * x)
r <- captured(checkpoint(1, { if (double(2) != 4) stop("double(2) should be 4") }))
expect("a wrong answer points at the TODO", !is.null(r$error) && grepl("failed on your code", r$text))

rm(double)
.ws_start("r/toy", strrep("0", 16), strrep("1", 16), "main", list(`1` = "double"))
.ws$stub <- list(); .ws$chosen <- character()
solution(1, "double", function(x) 2 * x)
expect("a skipped TODO is not replaced by the reference", fails_with(double(2), "run your TODO 1"))

# ---- worked mode and verify -------------------------------------------------------------------
Sys.setenv(MKTSTATS_WORKED = "1")
.ws_start("r/toy", strrep("0", 16), strrep("1", 16), "main", list(`1` = "double"))
double <- function(x) stop("TODO 1")
solution(1, "double", function(x) 2 * x)
expect("worked mode binds the reference", double(3) == 6)
r <- captured(checkpoint(1, { stopifnot(double(2) == 4) }))
expect("checkpoint passes on the reference", is.null(r$error) && grepl("REFERENCE", r$text))
.ws_verify_begin(1)
r <- captured(checkpoint(1, { stopifnot(double(2) == 4) }))
expect("the verify copy fails on the stub", !is.null(r$error) && grepl("as it should", r$text))
expect("verify end passes", is.null(captured(.ws_verify_end(1))$error))
expect("the reference is back", double(3) == 6)

double <- function(x) 4
solution(1, "double", function(x) 2 * x)
.ws_verify_begin(1)
invisible(captured(checkpoint(1, { stopifnot(double(2) == 4) })))
expect("verify end fails when the stub passes", fails_with(.ws_verify_end(1), "cannot tell"))

Sys.setenv(MKTSTATS_SABOTAGE = "1")
double <- function(x) stop("TODO 1")
solution(1, "double", function(x) 2 * x)
expect("sabotage keeps the stub in worked mode", fails_with(double(2), "TODO 1"))
Sys.unsetenv("MKTSTATS_SABOTAGE")

# ---- run record --------------------------------------------------------------------------------
Sys.setenv(MKTSTATS_INSTALL_SECONDS = "12.5")
.ws_start("r/toy", strrep("0", 16), strrep("1", 16), "main", list(`1` = "double"))
double <- function(x) stop("TODO 1")
solution(1, "double", function(x) 2 * x)
invisible(captured(checkpoint(1, { stopifnot(double(2) == 4) })))
dir <- tempfile(); dir.create(dir); old <- setwd(dir)
r <- captured(run_record())
rec <- jsonlite::fromJSON("run_record.json", simplifyVector = FALSE)
setwd(old)
expect("run_record prints between markers",
       grepl("----- run record", r$text, fixed = TRUE) && grepl("----- end of run record -----", r$text, fixed = TRUE))
expect("the record passes", identical(rec$status, "pass") && identical(rec$mode, "worked"))
expect("the record has its fields", all(c("notebook", "content_sha", "deps_sha", "ref", "settings",
  "seconds", "install_seconds", "r", "colab_release", "cpus", "packages", "checkpoints", "kernel") %in% names(rec)))
expect("install seconds come from the install cell", identical(rec$install_seconds, 12.5))
expect("settings are an object", is.list(rec$settings))
Sys.unsetenv(c("MKTSTATS_WORKED", "MKTSTATS_INSTALL_SECONDS"))

# ---- checks fail on broken input ---------------------------------------------------------------
rfm <- data.frame(frequency = c(0, 2, 5, 1), recency = c(0, 10.5, 30, 3), T = c(40, 38, 36, 20))
expect("check_rfm_table passes", isTRUE(check_rfm_table(rfm)))
swapped <- setNames(rfm, c("recency", "frequency", "T"))
expect("check_rfm_table fails on swapped columns", fails_with(check_rfm_table(swapped), "swap"))
days <- data.frame(frequency = c(0, 2, 5, 1), recency = c(0, 70, 200, 21), T = 300)
expect("whole-day recency swapped", fails_with(check_rfm_table(setNames(days, c("recency", "frequency", "T"))), "swap"))
expect("recency beyond T", fails_with(check_rfm_table(transform(rfm, recency = c(0, 10.5, 50, 3))), "exceeds T"))
set.seed(1); draws <- rnorm(20000)
expect("check_in_interval passes", length(check_in_interval(0.2, draws, prob = 0.94, kind = "hdi")) == 2)
expect("check_in_interval fails", fails_with(check_in_interval(5, draws, prob = 0.94, kind = "hdi", name = "r"), "94% HDI"))
skewed <- rexp(20000)
expect("HDI and ETI differ", all(check_in_interval(0.005, skewed, kind = "hdi") > -1) &&
         fails_with(check_in_interval(0.005, skewed, kind = "eti"), "ETI"))
expect("check_close passes", isTRUE(check_close(1.0005, 1, rel = 1e-3)))
expect("check_close fails", fails_with(check_close(1.1, 1, rel = 1e-3, name = "r"), "r is 1.1"))
expect("check_close needs a tolerance", fails_with(check_close(1, 1), "tolerance"))
a <- data.frame(id = 1:3, n = c(4, 1, 2))
expect("check_frames_agree passes", isTRUE(check_frames_agree(a, a[3:1, ], on = "id")))
expect("check_frames_agree fails", fails_with(check_frames_agree(a, transform(a, n = c(4, 2, 2)), on = "id"), "differs"))
expect("check_columns fails", fails_with(check_columns(a, c("id", "spend")), "missing"))

# ---- data --------------------------------------------------------------------------------------
Sys.setenv(MKTSTATS_REPO_ROOT = root)
expect("mkt_data reads the checkout", fails_with(mkt_data("no/such/file.csv"), "No such file"))

if (failures) quit(status = 1)
cat("all R helper tests passed\n")
