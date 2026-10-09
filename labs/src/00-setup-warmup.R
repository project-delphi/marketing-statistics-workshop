# %% [markdown]
# # Part A · Your runtime
#
# A *runtime* is the machine your notebook runs on. On Colab it is a virtual machine that
# Google starts for you and deletes when you disconnect, so anything you install or download
# lasts only for the session. The cell above installed what this lab needs (on Colab) and
# printed where you are running.
#
# **Predict.** How many CPUs does this runtime have, and how much memory? Write your guess
# down before you run the next cell.

# %%
suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
})
memory_gb <- tryCatch({
  line <- grep("^MemTotal", readLines("/proc/meminfo"), value = TRUE)
  round(as.numeric(gsub("[^0-9]", "", line)) * 1024 / 1e9, 1)
}, error = function(e) NA, warning = function(w) NA)
cat("R:            ", R.version.string, "\n")
cat("Platform:     ", R.version$platform, "\n")
cat("CPUs:         ", parallel::detectCores(), "\n")
cat("Memory (GB):  ", memory_gb, "\n")
cat("Colab release:", Sys.getenv("COLAB_RELEASE_TAG", "(not on Colab)"), "\n")
for (p in c("duckdb", "DBI", "dplyr")) cat(sprintf("%-14s%s\n", p, as.character(packageVersion(p))))
cat("QUICK:        ", QUICK, "(it has no effect in this lab)\n")

# %% [markdown]
# **Explain.** The CPU count and memory depend on the runtime type you choose (on Colab:
# Runtime > Change runtime type), so keep a note of them: later labs report run times, and a
# time means little without the machine it ran on. Google publishes the software each Colab
# runtime ships with (operating system, Python and R versions, preinstalled packages) in the
# [googlecolab/backend-info](https://github.com/googlecolab/backend-info) repository. The
# workshop's R labs install their packages from a dated snapshot of the Posit Public Package
# Manager, the same snapshot its Docker image uses, so CI and Colab run the same versions.

# %% [markdown]
# # Part B · From a transaction log to a customer table
#
# Most customer analytics starts from a *transaction log*: one row per purchase. Here it is
# the CDNOW log that ships with PyMC-Marketing's example data. CDNOW was an online CD
# retailer. The columns:
#
# | column | meaning |
# |---|---|
# | `_id`, `id` | two identifiers of the customer, one-to-one; we use `id` |
# | `date` | the purchase date as an integer, `YYYYMMDD` (for example `19970118`) |
# | `cds_bought` | number of CDs in the purchase |
# | `spent` | amount paid, in dollars |
#
# The next cell downloads the file once, checks it against a known SHA-256 hash (so you know
# you have the same bytes as everyone else), and caches it.

# %%
# TODO(mktstats data loaders): switch to the shared CDNOW loader once the package has one.
CDNOW_URL <- "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/cdnow_transactions.csv"
CDNOW_SHA256 <- "7311ae0fb676e9f454bba310b10bc769b1a6838aeec09bd00175eb49481dddac"

load_cdnow <- function(url = CDNOW_URL, sha256 = CDNOW_SHA256) {
  cache <- Sys.getenv("MKTSTATS_CACHE", file.path(path.expand("~"), ".cache", "mktstats"))
  path <- file.path(cache, "cdnow_transactions.csv")
  good <- function() file.exists(path) && identical(unname(tools::sha256sum(path)), sha256)
  if (!good()) {
    dir.create(cache, recursive = TRUE, showWarnings = FALSE)
    for (attempt in 1:3) {
      done <- tryCatch({
        utils::download.file(url, path, quiet = TRUE, mode = "wb")
        TRUE
      }, error = function(e) FALSE, warning = function(w) FALSE)
      if (done) break
      Sys.sleep(2 * attempt)
    }
    if (!good()) {
      unlink(path)
      stop("Could not download the CDNOW file with the expected SHA-256 hash. ",
           "Check the network and rerun; if it fails again, tell the instructor.", call. = FALSE)
    }
  }
  utils::read.csv(path, check.names = FALSE)  # keep the column name `_id`
}

tx_raw <- load_cdnow()
con <- suppressMessages(dbConnect(duckdb::duckdb()))  # quiet the storage notice
duckdb::duckdb_register(con, "transactions", tx_raw)
cat(format(nrow(tx_raw), big.mark = ","), "transactions,",
    format(length(unique(tx_raw$id)), big.mark = ","), "customers\n")
head(tx_raw)

# %% [markdown]
# DuckDB is an analytical database that runs inside this notebook: no server, and it can query
# an R data frame directly. `duckdb_register(con, "transactions", tx_raw)` made the data frame
# visible to SQL as a table named `transactions`. Try it:

# %%
dbGetQuery(con, "SELECT COUNT(*) AS n_transactions, SUM(spent) AS total_spend FROM transactions")

# %% [markdown]
# ## Exercise 1 · The customer table in SQL (12 minutes)
#
# **Predict.** The log has one row per transaction. How many rows will a table with one row
# per customer have? Roughly what share of customers do you expect bought only once?
#
# **Task.** Write `customer_table_sql(con)`: a DuckDB query on `transactions` that returns one
# row per customer with these columns:
#
# - `customer_id` (the `id` column)
# - `first_purchase` and `last_purchase`: dates, not `YYYYMMDD` integers. In DuckDB,
#   `strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE` turns `19970118` into a date.
# - `n_purchases`: the number of transactions
# - `total_spend`: the sum of `spent`
#
# Return a data frame: `dbGetQuery(con, query)`.

# %% tags=["exercise"]
customer_table_sql <- function(con) {
  # TODO 1: write the query, run it with dbGetQuery(con, query)
  stop("TODO 1")
}

# %% [markdown]
# **Solution 1 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 1 — try it yourself first { display-mode: "form" }
solution(1, "customer_table_sql", function(con) {
  dbGetQuery(con, "
    SELECT
      id AS customer_id,
      MIN(strptime(CAST(date AS VARCHAR), '%Y%m%d'))::DATE AS first_purchase,
      MAX(strptime(CAST(date AS VARCHAR), '%Y%m%d'))::DATE AS last_purchase,
      COUNT(*) AS n_purchases,
      SUM(spent) AS total_spend
    FROM transactions
    GROUP BY id
    ORDER BY customer_id
  ")
})

# %% [markdown]
# **Explain.** `GROUP BY id` collapses each customer's rows into one, and every other column
# must then be an aggregate (`MIN`, `MAX`, `COUNT`, `SUM`). Converting the date before taking
# `MIN` and `MAX` matters little here (`YYYYMMDD` integers sort in date order) but matters as
# soon as you subtract dates.
#
# <details><summary>Why convert the dates at all?</summary>
#
# Recency and customer age, the inputs of the models in Modules 1 to 4, are differences between
# dates. `19970201 - 19970131` is 70, not 1: integer dates give nonsense the moment you do
# arithmetic with them.
# </details>

# %% tags=["checkpoint"]
checkpoint(1, {
  customers_sql <- customer_table_sql(con)
  check_columns(customers_sql, c("customer_id", "first_purchase", "last_purchase", "n_purchases", "total_spend"),
                name = "customer_table_sql(con)")
  n_customers <- length(unique(tx_raw$id))
  if (nrow(customers_sql) != n_customers || anyDuplicated(customers_sql$customer_id)) {
    stop(sprintf("Expected one row per customer (%d), got %d rows. GROUP BY the customer id.",
                 n_customers, nrow(customers_sql)), call. = FALSE)
  }
  if (!inherits(customers_sql$first_purchase, "Date")) {
    stop("first_purchase should be a date, not an integer like 19970118: convert it with ",
         "strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE.", call. = FALSE)
  }
  if (any(customers_sql$first_purchase > customers_sql$last_purchase)) {
    stop("Some customers' first purchase comes after their last: check MIN and MAX.", call. = FALSE)
  }
  check_close(sum(customers_sql$n_purchases), nrow(tx_raw), abs = 0,
              name = "the number of purchases summed over customers")
  check_close(sum(customers_sql$total_spend), sum(tx_raw$spent), rel = 1e-9,
              name = "total spend summed over customers")
})

# %% [markdown]
# ## Exercise 2 · The same table in dplyr (12 minutes)
#
# Two independent implementations that agree are the cheapest test you will ever write.
#
# **Predict.** Which step is most likely to make a dplyr version disagree with the SQL one?
#
# **Task.** Write `customer_table_dplyr(tx)` returning the same columns as Exercise 1, from the
# data frame `tx` (the raw log). `as.Date(as.character(date), format = "%Y%m%d")` converts the
# dates; then `group_by()` the customer and `summarise()`.

# %% tags=["exercise"]
customer_table_dplyr <- function(tx) {
  # TODO 2: convert the dates, group by customer, summarise
  stop("TODO 2")
}

# %% [markdown]
# **Solution 2 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 2 — try it yourself first { display-mode: "form" }
solution(2, "customer_table_dplyr", function(tx) {
  tx |>
    mutate(date = as.Date(as.character(date), format = "%Y%m%d")) |>
    group_by(customer_id = id) |>
    summarise(
      first_purchase = min(date),
      last_purchase = max(date),
      n_purchases = n(),
      total_spend = sum(spent),
      .groups = "drop"
    ) |>
    arrange(customer_id)
})

# %% [markdown]
# **Explain.** `summarise()` mirrors the SQL `SELECT` list one to one, which makes the two easy
# to compare by eye. `n()` counts rows, as `COUNT(*)` does.
#
# <details><summary>What the checkpoint compares</summary>
#
# It matches the two tables on `customer_id` and compares every shared column: dates exactly,
# counts and spend within a tiny tolerance (floating-point sums can differ in the last digit
# depending on the order of addition).
# </details>

# %% tags=["checkpoint"]
checkpoint(2, {
  customers_dplyr <- customer_table_dplyr(tx_raw)
  check_frames_agree(customers_sql, customers_dplyr, on = "customer_id",
                     names = c("the SQL table", "the dplyr table"))
})

# %% [markdown]
# ## Exercise 3 · Repeat purchases count days, not rows (10 minutes)
#
# The models from Module 2 on describe *repeat purchases*: purchases after the first. They
# count purchase occasions, not rows, so two transactions by the same customer on the same day
# are one occasion. Then `repeat_purchases = (number of distinct purchase days) - 1`.
#
# **Predict.** Do any CDNOW customers have two transactions on the same day? If so, does
# counting rows instead of days change the count for many customers?
#
# **Task.** Write `repeat_purchases(tx)` with dplyr, returning one row per customer with
# `customer_id`, `purchase_days` (distinct dates) and `repeat_purchases`.

# %% tags=["exercise"]
repeat_purchases <- function(tx) {
  # TODO 3: count distinct purchase days per customer, then subtract one
  stop("TODO 3")
}

# %% [markdown]
# **Solution 3 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 3 — try it yourself first { display-mode: "form" }
solution(3, "repeat_purchases", function(tx) {
  tx |>
    group_by(customer_id = id) |>
    summarise(purchase_days = n_distinct(date), .groups = "drop") |>
    mutate(repeat_purchases = purchase_days - 1L)
})

# %% [markdown]
# **Explain.** `n_distinct()` counts distinct values, so same-day transactions collapse into
# one occasion. A customer who bought once has zero repeat purchases.
#
# <details><summary>Why the checkpoint uses a tiny made-up log</summary>
#
# Customer 1 below buys twice on 1 January and once on 1 March: one repeat purchase, although
# there are three rows. A log built to contain the tricky case tests the definition directly;
# the full CDNOW log then checks your function against an independent SQL count.
# </details>

# %% tags=["checkpoint"]
checkpoint(3, {
  toy <- data.frame(
    `_id` = c(10, 10, 10, 20), id = c(1, 1, 1, 2),
    date = c(19970101, 19970101, 19970301, 19970105),
    cds_bought = c(1, 2, 1, 1), spent = c(10, 20, 5, 7),
    check.names = FALSE
  )
  got <- repeat_purchases(toy)
  r1 <- got$repeat_purchases[got$customer_id == 1]
  r2 <- got$repeat_purchases[got$customer_id == 2]
  if (!isTRUE(all.equal(as.numeric(r1), 1))) {
    stop("Customer 1 bought twice on 1997-01-01 and once on 1997-03-01: one repeat purchase ",
         "(same-day purchases count once). Your function gives ", format(r1), ".", call. = FALSE)
  }
  if (!isTRUE(all.equal(as.numeric(r2), 0))) {
    stop("Customer 2 bought once: zero repeat purchases, not ", format(r2), ".", call. = FALSE)
  }
  expected <- dbGetQuery(con, paste("SELECT id AS customer_id, COUNT(DISTINCT date) - 1 AS repeat_purchases",
                                    "FROM transactions GROUP BY id"))
  check_frames_agree(as.data.frame(repeat_purchases(tx_raw))[c("customer_id", "repeat_purchases")], expected,
                     on = "customer_id", names = c("repeat_purchases(tx_raw)", "an independent SQL count"))
})

# %% [markdown]
# Compare with your prediction:

# %%
repeats <- repeat_purchases(tx_raw)
both <- merge(customers_sql, repeats, by = "customer_id")
cat(sprintf("Customers who bought only once: %.1f%%\n", 100 * mean(both$repeat_purchases == 0)))
cat("Customers whose count changes if you count rows instead of days:",
    sum(both$n_purchases - 1 != both$repeat_purchases), "\n")

# %% [markdown]
# # Part C · Using a coding agent to scaffold and check an analysis
#
# A *coding agent* is an AI assistant that can write code, and often run it, in your project.
# It can draft queries like the ones above from a description of the tables. Treat what it
# writes as a colleague's first draft: useful, and unverified until checked.
#
# 1. **Give it the definitions, not just the column names.** "Repeat purchases" was ambiguous
#    until Exercise 3 pinned it down (days, not rows). Ambiguity is where a plausible-looking
#    query goes wrong.
# 2. **Ask for an independent check, not only the answer.** A second implementation in another
#    tool (Exercise 2) and invariants that must hold (one row per customer, totals that match
#    the raw log) catch errors that reading the code misses.
# 3. **Run the checks yourself and read the code.** An agent saying "this works" is not
#    evidence; a passing checkpoint is.
# 4. **Mind the data.** Do not paste customer records into a tool your organization has not
#    approved for them. Describing the schema, or sharing a small made-up sample like the one
#    in checkpoint 3, is usually enough.
#
# A prompt in this spirit:
#
# > Table `transactions(_id, id, date, cds_bought, spent)`: `id` is the customer, `date` is an
# > integer `YYYYMMDD`. Write a DuckDB query with one row per customer: first and last purchase
# > dates (as dates), number of transactions, total spend. Then write a dplyr version and a
# > test that the two agree on every customer.

# %% [markdown]
# ## Decision · Sign off the customer table
#
# Module 1 starts from this table. Before anyone builds on it, decide whether it is fit to
# hand over: the SQL and dplyr versions agree, the totals match the raw log, and repeat
# purchases follow the definition the models use.

# %%
cat("Customers:", format(nrow(both), big.mark = ","), "\n")
cat("Purchases:", format(sum(both$n_purchases), big.mark = ","), "(log:", format(nrow(tx_raw), big.mark = ","), ")\n")
cat(sprintf("Spend: $%s (log: $%s)\n", format(round(sum(both$total_spend), 2), big.mark = ",", nsmall = 2),
            format(round(sum(tx_raw$spent), 2), big.mark = ",", nsmall = 2)))
cat("Window:", format(min(both$first_purchase)), "to", format(max(both$last_purchase)), "\n")
passed <- names(Filter(function(r) isTRUE(r$passed), .ws$results))
cat("Checkpoints passed:", if (length(passed)) paste(passed, collapse = ", ") else "none", "\n")

# %% [markdown]
# **Decide.** Would you hand this table to the modelling team? Write one sentence: yes or no,
# and which checks your answer rests on.

# %% [markdown]
# ## Stretch (optional) · Customers by first-purchase month
#
# Write a DuckDB query that counts customers by the month of their first purchase and gives
# each cohort's total spend over the whole window. Which month brought the most valuable
# customers per head?
#
# <details><summary>One way to write it</summary>
#
# ```sql
# WITH firsts AS (
#     SELECT id, MIN(strptime(CAST(date AS VARCHAR), '%Y%m%d'))::DATE AS first_purchase,
#            SUM(spent) AS spend
#     FROM transactions GROUP BY id
# )
# SELECT date_trunc('month', first_purchase) AS cohort, COUNT(*) AS customers,
#        SUM(spend) AS spend, SUM(spend) / COUNT(*) AS spend_per_customer
# FROM firsts GROUP BY cohort ORDER BY cohort
# ```
# </details>
