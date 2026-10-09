# R helpers for the Statistics for Marketing workshop labs (CONTRIBUTING.md, "Harness API").
#
# The generated install cell sources this file (from the checkout when MKTSTATS_REPO_ROOT is
# set, else from raw GitHub at the workshop's ref). The generated harness cell calls
# .ws_start(); the record cell calls workshop_summary() and run_record().
#
#   solution(n, name, fn)  stores a reference solution; binds it only in worked mode
#                          (WORKED_EXAMPLE <- TRUE or MKTSTATS_WORKED=1) or after use_reference(n)
#   checkpoint(n, {...})   runs the checks, says whose code they checked, prints how to
#                          recover when they fail, and re-raises so Run all stops
#   use_reference(n), workshop_summary(), run_record()
#   .ws_verify_begin(n), .ws_verify_end(n)   used by scripts/test_notebooks.py
#   check_rfm_table(), check_in_interval(), check_close(), check_columns()
#   mkt_data(path)         reads a committed file under data/
#
# No Colab forms in R: WORKED_EXAMPLE and QUICK are plain variables, overridden by the
# environment variables MKTSTATS_WORKED and MKTSTATS_QUICK.

.ws_flag <- function(name) {
  value <- tolower(trimws(Sys.getenv(name, "")))
  if (value %in% c("1", "true", "yes", "on")) return(TRUE)
  if (value %in% c("0", "false", "no", "off")) return(FALSE)
  NA
}

if (!exists("WORKED_EXAMPLE", envir = globalenv())) assign("WORKED_EXAMPLE", FALSE, envir = globalenv())
if (!exists("QUICK", envir = globalenv())) assign("QUICK", FALSE, envir = globalenv())

.ws_not_written <- "is not written yet"
.ws_record_begin <- "----- run record (paste into scripts/add_run_record.py) -----"
.ws_record_end <- "----- end of run record -----"

# Harness state lives in one environment in the global environment, so sourcing this file
# again keeps the stored solutions.
if (!exists(".ws", envir = globalenv()) || !is.environment(get(".ws", envir = globalenv()))) {
  assign(".ws", new.env(), envir = globalenv())
}
local({
  ws <- get(".ws", envir = globalenv())
  for (field in c("ref", "stub")) if (is.null(ws[[field]])) ws[[field]] <- list()
  if (is.null(ws$chosen)) ws$chosen <- character()
  if (is.null(ws$exercises)) ws$exercises <- list()
  if (is.null(ws$results)) ws$results <- list()
  if (is.null(ws$verified)) ws$verified <- list()
  if (is.null(ws$started)) ws$started <- Sys.time()
})

.ws_worked <- function() {
  env <- .ws_flag("MKTSTATS_WORKED")
  if (!is.na(env)) return(env)
  isTRUE(get0("WORKED_EXAMPLE", envir = globalenv(), inherits = FALSE))
}

.ws_quick <- function() {
  env <- .ws_flag("MKTSTATS_QUICK")
  if (!is.na(env)) return(env)
  isTRUE(get0("QUICK", envir = globalenv(), inherits = FALSE))
}

.ws_sabotaged <- function() {
  parts <- trimws(strsplit(Sys.getenv("MKTSTATS_SABOTAGE", ""), ",")[[1]])
  parts[grepl("^[0-9]+$", parts)]
}

.ws_settings <- function() {
  out <- list()
  if (.ws_quick()) out$MKTSTATS_QUICK <- "1"
  if (nzchar(Sys.getenv("MKTSTATS_SABOTAGE"))) out$MKTSTATS_SABOTAGE <- Sys.getenv("MKTSTATS_SABOTAGE")
  out
}

.ws_start <- function(notebook, content_sha, deps_sha, ref, exercises = list()) {
  ws <- .ws
  ws$notebook <- notebook
  ws$content_sha <- content_sha
  ws$deps_sha <- deps_sha
  ws$ref_name <- ref
  ws$exercises <- lapply(exercises, as.character)
  ws$results <- list()
  ws$verified <- list()
  ws$verifying <- NULL
  ws$saved <- list()
  ws$started <- Sys.time()
  assign("WORKED_EXAMPLE", .ws_worked(), envir = globalenv())
  assign("QUICK", .ws_quick(), envir = globalenv())
  mode <- if (.ws_worked()) "WORKED EXAMPLE (reference solutions)" else "checking your code"
  cat("Workshop harness ready: ", mode, if (.ws_quick()) ", QUICK settings" else "", ".\n", sep = "")
  invisible(ws)
}

.ws_is_reference <- function(obj) {
  !is.null(attr(obj, "mktstats_reference", exact = TRUE))
}

.ws_not_run <- function(n, name) {
  force(n); force(name)
  function(...) {
    stop(sprintf("%s: run your TODO %s cell first, or go on with use_reference(%s)", name, n, n),
         call. = FALSE)
  }
}

solution <- function(n, name, fn) {
  key <- as.character(n)
  ws <- .ws
  attr(fn, "mktstats_reference") <- key
  if (is.null(ws$ref[[key]])) ws$ref[[key]] <- list()
  ws$ref[[key]][[name]] <- fn
  current <- get0(name, envir = globalenv(), inherits = FALSE)
  yours_now <- !is.null(current) && !.ws_is_reference(current)
  if (yours_now) {
    if (is.null(ws$stub[[key]])) ws$stub[[key]] <- list()
    ws$stub[[key]][[name]] <- current
  }
  sabotaged <- key %in% .ws_sabotaged()
  bind <- function(value) assign(name, value, envir = globalenv())
  if (.ws_worked() && !sabotaged) {
    bind(fn)
  } else if (yours_now) {
    ws$chosen <- setdiff(ws$chosen, key)  # your TODO cell ran again: back to your code
  } else if (key %in% ws$chosen && !sabotaged) {
    bind(fn)
  } else if (!is.null(ws$stub[[key]][[name]])) {
    bind(ws$stub[[key]][[name]])
  } else if (is.function(fn)) {
    bind(.ws_not_run(key, name))
  } else {
    cat(sprintf("[workshop] Run your TODO %s cell: %s is the reference until you do.\n", key, name))
    bind(fn)
  }
  invisible(NULL)
}

use_reference <- function(n) {
  key <- as.character(n)
  ws <- .ws
  if (is.null(ws$ref[[key]])) stop(sprintf("Exercise %s has no reference solution yet: run its solution cell", key))
  for (name in names(ws$ref[[key]])) assign(name, ws$ref[[key]][[name]], envir = globalenv())
  ws$chosen <- union(ws$chosen, key)
  cat(sprintf(paste0("Exercise %s: now using the REFERENCE solution. Rerun the cells below it.",
                     " To switch back to your code, rerun your TODO %s cell.\n"), key, key))
  invisible(NULL)
}

.ws_whose <- function(key) {
  names <- .ws$exercises[[key]]
  if (is.null(names) || !length(names)) return(NA_character_)
  k <- sum(vapply(names, function(x) {
    obj <- get0(x, envir = globalenv(), inherits = FALSE)
    !is.null(obj) && .ws_is_reference(obj)
  }, logical(1)))
  if (k == length(names)) return("the REFERENCE solution")
  if (k == 0) return("your code")
  "partly the REFERENCE solution"
}

.ws_is_todo <- function(message) {
  grepl("^\\s*TODO\\s*[0-9]+", message) || grepl("run your TODO", message, fixed = TRUE)
}

checkpoint <- function(n, expr, label = NULL) {
  key <- as.character(n)
  label <- if (is.null(label)) key else as.character(label)
  ws <- .ws
  whose <- .ws_whose(key)
  err <- tryCatch({
    force(expr)
    NULL
  }, error = function(e) e)
  ok <- is.null(err)
  if (!is.null(ws$verifying)) {
    ws$verified[[as.character(ws$verifying)]] <- ok
    if (ok) cat(sprintf("[verify] Checkpoint %s PASSED on the stub of exercise %s.\n", label, key))
    else cat(sprintf("[verify] Checkpoint %s failed on the stub, as it should.\n", label))
  } else {
    ws$results[[label]] <- list(passed = ok, whose = if (is.na(whose)) NULL else whose)
    if (ok) {
      on <- if (is.na(whose)) "" else paste0(" on ", whose)
      cat(sprintf("[workshop] Checkpoint %s passed%s.\n", label, on))
    } else if (.ws_is_todo(conditionMessage(err))) {
      if (is.na(whose)) {
        cat(sprintf(paste0("[workshop] Checkpoint %s uses an exercise you have not written yet.",
                           " Finish it, or go on with use_reference(N) for that exercise.\n"), label))
      } else {
        cat(sprintf(paste0("[workshop] Checkpoint %s: TODO %s %s. Write it, run its cell, then",
                           " rerun this one. To go on without it, run use_reference(%s)\n"),
                    label, key, .ws_not_written, key))
      }
    } else if (identical(whose, "the REFERENCE solution")) {
      cat(sprintf(paste0("[workshop] Checkpoint %s failed on the reference solution: a problem with",
                         " the lab or the runtime, not with your code. Tell the instructor.\n"), label))
    } else if (is.na(whose)) {
      cat(sprintf("[workshop] Checkpoint %s failed. Read the message below.\n", label))
    } else {
      cat(sprintf(paste0("[workshop] Checkpoint %s failed on %s. Read the message below, fix TODO %s",
                         " and rerun its cell, or go on with use_reference(%s).\n"), label, whose, key, key))
    }
  }
  if (!ok) stop(err)
  invisible(TRUE)
}

.ws_verify_begin <- function(n) {
  key <- as.character(n)
  ws <- .ws
  names <- ws$exercises[[key]]
  ws$saved <- lapply(setNames(names, names), function(x) get0(x, envir = globalenv(), inherits = FALSE))
  for (name in names(ws$stub[[key]])) assign(name, ws$stub[[key]][[name]], envir = globalenv())
  ws$verified[[key]] <- NULL
  ws$verifying <- key
  invisible(NULL)
}

.ws_verify_end <- function(n) {
  key <- as.character(n)
  ws <- .ws
  for (name in names(ws$saved)) {
    value <- ws$saved[[name]]
    if (is.null(value)) {
      if (exists(name, envir = globalenv(), inherits = FALSE)) rm(list = name, envir = globalenv())
    } else {
      assign(name, value, envir = globalenv())
    }
  }
  ws$saved <- list()
  ws$verifying <- NULL
  if (!identical(ws$verified[[key]], FALSE)) {
    stop(sprintf(paste0("The checkpoint of exercise %s passed on the unfinished stub (or did not run):",
                        " it cannot tell a finished exercise from an unfinished one"), key))
  }
  invisible(NULL)
}

workshop_summary <- function() {
  ws <- .ws
  groups <- list(yours = character(), reference = character(), other = character(), failed = character())
  for (label in names(ws$results)) {
    r <- ws$results[[label]]
    if (!isTRUE(r$passed)) groups$failed <- c(groups$failed, label)
    else if (is.null(r$whose)) groups$other <- c(groups$other, label)
    else if (identical(r$whose, "your code")) groups$yours <- c(groups$yours, label)
    else groups$reference <- c(groups$reference, label)
  }
  mode <- if (.ws_worked()) "WORKED EXAMPLE: reference solutions" else "your code"
  cat(sprintf("%s, run with %s%s.\n", ws$notebook, mode, if (.ws_quick()) ", QUICK settings" else ""))
  show <- function(text, x) cat(sprintf("  %s: %s\n", text, if (length(x)) paste(x, collapse = ", ") else "none"))
  show("Checkpoints passed on your code", groups$yours)
  show("Checkpoints passed on a reference solution", groups$reference)
  show("Other checks passed", groups$other)
  show("Checkpoints failed", groups$failed)
  if (.ws_worked()) cat("  A worked-example run shows how the lab goes, not that you did it.\n")
  invisible(groups)
}

.ws_packages <- function() {
  wanted <- c("CLVTools", "grf", "CausalImpact", "bsts", "GeoLift", "augsynth", "duckdb", "DBI",
              "dplyr", "tidyr", "ggplot2", "jsonlite", "data.table", "lubridate", "readr")
  have <- rownames(utils::installed.packages())
  out <- list()
  for (p in intersect(wanted, have)) out[[p]] <- as.character(utils::packageVersion(p))
  out
}

.ws_record <- function() {
  ws <- .ws
  passed <- vapply(ws$results, function(r) isTRUE(r$passed), logical(1))
  install <- suppressWarnings(as.numeric(Sys.getenv("MKTSTATS_INSTALL_SECONDS", "")))
  colab <- Sys.getenv("COLAB_RELEASE_TAG", "")
  runtime <- if (nzchar(colab)) "colab" else if (nzchar(Sys.getenv("SAGEMAKER_APP_TYPE")) ||
                                                 file.exists("/opt/ml/metadata/resource-metadata.json")) "sagemaker" else "local"
  mem <- tryCatch({
    line <- grep("^MemTotal", readLines("/proc/meminfo"), value = TRUE)
    round(as.numeric(gsub("[^0-9]", "", line)) * 1024 / 1e9, 1)
  }, error = function(e) NULL, warning = function(w) NULL)
  rec <- list(
    date = format(Sys.time(), "%Y-%m-%d", tz = "UTC"),
    notebook = ws$notebook,
    kernel = "ir",
    content_sha = ws$content_sha,
    deps_sha = ws$deps_sha,
    ref = ws$ref_name,
    mode = if (.ws_worked()) "worked" else "learner",
    settings = .ws_settings(),
    status = if (length(passed) && all(passed)) "pass" else "fail",
    seconds = round(as.numeric(difftime(Sys.time(), ws$started, units = "secs")), 1),
    install_seconds = if (is.na(install)) NULL else install,
    r = paste(R.version$major, R.version$minor, sep = "."),
    platform = paste(Sys.info()[["sysname"]], Sys.info()[["machine"]]),
    runtime = runtime,
    colab_release = if (nzchar(colab)) colab else NULL,
    cpus = parallel::detectCores(),
    memory_gb = mem,
    packages = .ws_packages(),
    checkpoints = lapply(ws$results, function(r) list(passed = isTRUE(r$passed), whose = r$whose))
  )
  if (length(ws$verified)) rec$verified <- ws$verified
  rec
}

.ws_json <- function(x) {
  # Empty lists print as {} (objects), never [] (records are JSON objects throughout).
  fix <- function(v) {
    if (is.list(v)) {
      if (!length(v)) return(structure(list(), names = character()))
      return(lapply(v, fix))
    }
    v
  }
  jsonlite::toJSON(fix(x), auto_unbox = TRUE, null = "null", digits = NA)
}

run_record <- function(path = "run_record.json") {
  text <- as.character(.ws_json(.ws_record()))
  tryCatch(writeLines(text, path), error = function(e) cat("(Could not write ", path, ": ",
                                                             conditionMessage(e), ")\n", sep = ""))
  cat(.ws_record_begin, "\n", text, "\n", .ws_record_end, "\n", sep = "")
  invisible(NULL)
}

# ---- checks: each stops with a message that says what to look at ---------------------------

check_columns <- function(df, required, name = "The table") {
  missing <- setdiff(required, names(df))
  if (length(missing)) {
    stop(sprintf("%s is missing the column(s) %s. It has %s. Check the names your function gives the columns.",
                 name, paste(missing, collapse = ", "), paste(names(df), collapse = ", ")), call. = FALSE)
  }
  invisible(TRUE)
}

check_rfm_table <- function(rfm, frequency = "frequency", recency = "recency", T = "T", name = "The RFM table") {
  check_columns(rfm, c(frequency, recency, T), name)
  if (!nrow(rfm)) stop(name, " has no rows.", call. = FALSE)
  f <- as.numeric(rfm[[frequency]]); r <- as.numeric(rfm[[recency]]); t <- as.numeric(rfm[[T]])
  if (anyNA(c(f, r, t)) || any(!is.finite(c(f, r, t)))) stop(name, ": missing or infinite values.", call. = FALSE)
  if (any(f < 0) || any(abs(f - round(f)) > 1e-9)) {
    stop(sprintf(paste0("%s: %s must be whole numbers >= 0 (repeat purchases). Did you swap %s and %s,",
                        " or count the first purchase?"), name, frequency, frequency, recency), call. = FALSE)
  }
  if (any(r < 0)) stop(sprintf("%s: %s is negative for %d customers.", name, recency, sum(r < 0)), call. = FALSE)
  if (any(r > t + 1e-9)) {
    stop(sprintf("%s: %s exceeds %s for %d customers. Both are measured from the first purchase.",
                 name, recency, T, sum(r > t + 1e-9)), call. = FALSE)
  }
  if (any(f == 0 & r > 1e-9)) {
    stop(sprintf("%s: %d customers have no repeat purchase but %s > 0. Recency is 0 without a repeat purchase.",
                 name, sum(f == 0 & r > 1e-9), recency), call. = FALSE)
  }
  if (any(f > r + 1e-9)) {
    stop(sprintf(paste0("%s: %s exceeds %s for %d customers. Each repeat purchase falls in a later period,",
                        " so frequency <= recency in the same unit. Did you swap the columns?"),
                 name, frequency, recency, sum(f > r + 1e-9)), call. = FALSE)
  }
  invisible(TRUE)
}

.ws_interval <- function(draws, prob = 0.94, kind = c("hdi", "eti")) {
  kind <- match.arg(kind)
  if (!(prob > 0 && prob < 1)) stop("prob must be between 0 and 1")
  x <- sort(as.numeric(draws))
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 2) stop("need at least two finite draws")
  if (kind == "eti") return(unname(stats::quantile(x, c((1 - prob) / 2, 1 - (1 - prob) / 2), type = 7)))
  k <- max(1, min(n - 1, ceiling(prob * n) - 1))
  widths <- x[(k + 1):n] - x[1:(n - k)]
  i <- which.min(widths)
  c(x[i], x[i + k])
}

check_in_interval <- function(truth, draws, prob = 0.94, kind = "hdi", name = "the parameter") {
  iv <- .ws_interval(draws, prob, kind)
  label <- sprintf("%s%% %s", format(prob * 100), toupper(kind))
  if (truth < iv[1] || truth > iv[2]) {
    stop(sprintf(paste0("The true value of %s, %.4g, is outside the %s [%.4g, %.4g] of the posterior draws.",
                        " Look at the sampler diagnostics, the priors, and whether the model matches the",
                        " process that made the data."), name, truth, label, iv[1], iv[2]), call. = FALSE)
  }
  invisible(iv)
}

check_close <- function(estimate, truth, rel = NULL, abs = NULL, name = "the estimate") {
  if (is.null(rel) && is.null(abs)) stop("check_close(): give a tolerance, rel = ... and/or abs = ...")
  rel <- if (is.null(rel)) 0 else rel
  tol <- if (is.null(abs)) 0 else abs
  e <- as.numeric(estimate); t <- as.numeric(truth)
  gap <- base::abs(e - t)
  allowed <- tol + rel * base::abs(t)
  bad <- !(gap <= allowed)
  if (any(bad | is.na(bad))) {
    if (length(e) == 1 && length(t) == 1) {
      stop(sprintf("%s is %.6g; expected %.6g (allowed difference %.3g). Check the formula and the units.",
                   name, e, t, allowed), call. = FALSE)
    }
    stop(sprintf("%s: %d of %d values differ from the expected ones by more than the tolerance.",
                 name, sum(bad | is.na(bad)), length(bad)), call. = FALSE)
  }
  invisible(TRUE)
}

check_frames_agree <- function(left, right, on, tol = 1e-9,
                               names = c("the first table", "the second table")) {
  check_columns(left, on, names[1])
  check_columns(right, on, names[2])
  if (nrow(left) != nrow(right)) {
    stop(sprintf("%s has %d rows but %s has %d.", names[1], nrow(left), names[2], nrow(right)), call. = FALSE)
  }
  for (i in 1:2) {
    df <- list(left, right)[[i]]
    dup <- sum(duplicated(as.data.frame(df)[on]))
    if (dup) stop(sprintf("%s has %d repeated value(s) of %s: expected one row per key.",
                          names[i], dup, paste(on, collapse = ", ")), call. = FALSE)
  }
  shared <- setdiff(intersect(base::names(left), base::names(right)), on)
  if (!length(shared)) stop(sprintf("%s and %s have no columns in common besides the key.", names[1], names[2]),
                            call. = FALSE)
  a <- as.data.frame(left)[c(on, shared)]
  b <- as.data.frame(right)[c(on, shared)]
  m <- merge(a, b, by = on, suffixes = c("__a", "__b"), all = TRUE)
  if (nrow(m) != nrow(a)) {
    stop(sprintf("Some keys appear in only one of %s and %s.", names[1], names[2]), call. = FALSE)
  }
  for (col in shared) {
    x <- m[[paste0(col, "__a")]]; y <- m[[paste0(col, "__b")]]
    if (is.numeric(x) && is.numeric(y)) {
      bad <- !(base::abs(x - y) <= tol * pmax(1, base::abs(y)))
    } else {
      bad <- !(as.character(x) == as.character(y))
    }
    bad[is.na(bad)] <- !(is.na(x) & is.na(y))[is.na(bad)]
    if (any(bad)) {
      stop(sprintf("Column %s differs between %s and %s for %d row(s), e.g. key %s: %s vs %s. Check how each computes it.",
                   col, names[1], names[2], sum(bad), paste(m[which(bad)[1], on], collapse = "/"),
                   format(x[which(bad)[1]]), format(y[which(bad)[1]])), call. = FALSE)
    }
  }
  invisible(TRUE)
}

# ---- data -------------------------------------------------------------------------------------

mkt_data <- function(path, ref = getOption("mktstats.ref", Sys.getenv("MKTSTATS_REF", "main"))) {
  root <- Sys.getenv("MKTSTATS_REPO_ROOT")
  if (nzchar(root)) {
    file <- file.path(root, "data", path)
    if (!file.exists(file)) stop("No such file in the checkout: data/", path, call. = FALSE)
  } else {
    raw <- getOption("mktstats.raw", "https://raw.githubusercontent.com/project-delphi/marketing-statistics-workshop")
    cache <- file.path(Sys.getenv("MKTSTATS_CACHE", file.path(tempdir(), "mktstats")), ref)
    file <- file.path(cache, path)
    if (!file.exists(file)) {
      dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
      url <- sprintf("%s/%s/data/%s", raw, ref, path)
      ok <- FALSE
      for (attempt in 1:3) {
        ok <- tryCatch({ utils::download.file(url, file, quiet = TRUE, mode = "wb"); TRUE },
                       error = function(e) FALSE, warning = function(w) FALSE)
        if (ok) break
        Sys.sleep(2 * attempt)
      }
      if (!ok) { unlink(file); stop("Could not download ", url, ". Check the network and rerun.", call. = FALSE) }
    }
  }
  ext <- tolower(tools::file_ext(file))
  if (ext == "csv") return(utils::read.csv(file, stringsAsFactors = FALSE))
  if (ext == "json") return(jsonlite::fromJSON(file, simplifyVector = TRUE))
  file
}
