<!--
Partial, included by day-1.qmd (UI Expert). Owner: Pedagogy Expert.
Sections: #warm-up (start of the day) and #wrap-up (end of the day); the day page places them.
Durations match _variables.yml days.d1.blocks (warmup 15, wrapup 20); clock times are on the
generated timetable, so they are not repeated here.
Rows of the results table follow PLAN.md's curriculum; align them with each lab's Decision cell
once briefs/01-*.md to briefs/03-*.md are final.
-->

## Warm-up: from the pre-work to Day 1 (15 minutes) {#warm-up}

Answer alone on paper for 5 minutes, notes closed. Compare with a neighbour for 4 minutes. The facilitator takes the most-missed questions for 5 minutes, then says in one minute where today starts.

**W1.** You choose Run all in a lab and it stops at Exercise 2 with the message "TODO 2 is not written yet". What are your two options, and how will the checkpoint tell them apart?

::: {.callout-tip collapse="true" title="Answer"}
Write the function in the exercise cell and run it and its checkpoint again; or run `workshop.use_reference(2)` (in R, `use_reference(2)`) to load the reference solution and carry on. The checkpoint prints "passed on your code" in the first case and "passed on the REFERENCE solution" in the second, and the summary at the end of the notebook lists which checkpoints passed on your own code. Using the reference is not failing; it keeps you with the room.
:::

**W2.** Write the SQL expression that gives, for every order, the number of days since the same customer's previous order. What does it return for a customer's first order?

::: {.callout-tip collapse="true" title="Answer"}
`order_date - LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)`. In DuckDB, subtracting two dates gives a number of days. The first order has no previous order, so the result is `NULL`.
:::

**W3.** A customer buys at random times at an average rate of 0.5 purchases a month (a Poisson process). They have not bought for the last 4 months. What is the probability of that silence if they are still a customer? Does the silence prove they have left?

::: {.callout-tip collapse="true" title="Answer"}
The count in 4 months is Poisson with mean $0.5 \times 4 = 2$, so $P(0) = e^{-2} \approx 0.135$. About one active customer in seven with this rate would be silent this long, so silence is evidence of leaving, not proof. Module 2 turns this reasoning into P(alive), the probability that a customer is still active given their history.
:::

**W4.** You build a customer table by left-joining all customers to orders and counting rows per customer. What goes wrong for customers who have never ordered, and how do you fix it?

::: {.callout-tip collapse="true" title="Answer"}
They get a count of 1, from the single row the left join fills with missing values. Count non-missing order values instead: `COUNT(order_id)` rather than `COUNT(*)` in SQL, `count()` on an order column in pandas, `sum(!is.na(order_id))` in dplyr.
:::

**Why today.** Today's question is *{{< var days.d1.question >}}* Module 1 turns the transaction log into a table of customers with these same tools; Module 2 asks who is still a customer when nobody tells you they left; Module 3 asks whether how a customer was acquired changes that, and whether a machine-learning model does better.

## Wrap-up: fixed and left open (20 minutes) {#wrap-up}

The facilitator keeps the results table on the board (or a shared sheet) all week; Days 2 to 5 add rows to it.

**1. Fixed and left open (6 minutes: 3 in pairs, 3 with the room).** For each module, say what it made possible that the step before could not, and what it still cannot do.

- Module 1 turned a transaction log into a customer table and cohorts, but describes only the past.
- Module 2 forecast each customer's purchases and the probability that they are still active, with uncertainty, from frequency, recency and age alone.
- Module 3 added the acquisition channel and compared the model with machine learning on held-out data.
- **Left open for Day 2:** all of today's forecasts count purchases, not money. What is a customer worth in dollars, and how sure are we?

**2. The running results table (8 minutes).** Fill in your own numbers from today's labs, as each notebook's Decision cell and last cell print them. Write the interval's type and probability, and whether the run was FULL or QUICK and on your own code or the reference. On synthetic data, write the truth beside your estimate.

| Lab | Quantity | Your number | Interval (type, probability) | Truth (synthetic data) | Run (FULL or QUICK; yours or reference) |
|---|---|---|---|---|---|
| 1 | Customers in the base; share with no repeat purchase (CDNOW) |  |  | real data |  |
| 2 | BG/NBD parameters r, α, a, b against the truth |  |  |  |  |
| 2 | Holdout repeat purchases: forecast against actual (CDNOW) |  |  | actual: |  |
| 2 | Customers below the stop-retargeting threshold |  |  |  |  |
| 3 | Effect of acquisition channel on the purchase rate |  |  |  |  |
| 3 | Holdout error: BTYD against gradient boosting (same metric) |  |  | real data |  |

: Day 1 rows of the running results table (BTYD: buy-till-you-die models) {.striped}

**3. One decision claim (6 minutes: 3 writing, 3 with the room).** Write one sentence you would defend to a manager, in the form *number → rule → recommendation*: "Based on [number and interval from Lab N], I would [action], because [rule, with its cost or margin], unless [the assumption that would change it]." Two or three people read theirs; the room asks one question of each: *what would change your mind?*
