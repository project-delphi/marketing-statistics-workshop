<!--
Partial, included by day-4.qmd (UI Expert). Owner: Pedagogy Expert.
Sections: #warm-up, #wrap-up. Durations match _variables.yml days.d4.blocks (warmup 15, wrapup 20);
clock times are on the generated timetable.
-->

## Warm-up: what carried over from Day 3 (15 minutes) {#warm-up}

Answer alone on paper for 5 minutes, notes closed. Compare with a neighbor for 4 minutes. The facilitator takes the most-missed questions for 5 minutes, then says in one minute where today starts.

**W1.** An email test has a baseline conversion rate of 5%. You want to detect an increase of 0.5 percentage points with a two-sided test at the 5% level and 80% power. Using the rule of thumb $n \approx 16\,p(1-p)/\delta^2$ per arm, about how many customers per arm do you need? What happens if you want to detect 0.25 points instead?

::: {.callout-tip collapse="true" title="Answer"}
$16 \times 0.05 \times 0.95 / 0.005^2 = 30{,}400$ per arm. Halving the detectable effect multiplies the sample size by four, to about 121,600 per arm, because $n$ grows with $1/\delta^2$.
:::

**W2.** A test planned a 50/50 split. Of 100,000 customers, 50,600 landed in treatment and 49,400 in control. Is that a problem?

::: {.callout-tip collapse="true" title="Answer"}
Yes. Under a true 50/50 split, the chi-square statistic is $2 \times 600^2 / 50{,}000 = 14.4$ with one degree of freedom, $p \approx 0.0001$. This is a sample ratio mismatch: something in the assignment or the logging is broken, so the groups may not be comparable. Find the cause before reading the result.
:::

**W3.** A synthetic control estimate says a campaign raised sales in the treated region by 6%. Why run placebo tests, and what result would make you distrust the 6%?

::: {.callout-tip collapse="true" title="Answer"}
A placebo test applies the same method where there was no campaign: to control regions as if they had been treated, or to a period before the campaign. If many placebos show "effects" as large as 6%, the method produces gaps of that size from noise alone, and the 6% is not distinguishable from noise.
:::

**W4.** Last-click attribution credits paid search with 40% of online conversions. Why does that not mean pausing search would lose 40% of conversions?

::: {.callout-tip collapse="true" title="Answer"}
Attribution assigns credit to the touches before a conversion; it does not ask what would have happened without the channel. Many of those customers, such as people searching for the brand's name, would have converted anyway. The share that would be lost, the incremental effect, can only be measured by comparing with a holdout: an experiment.
:::

**Why today.** Today's question is *{{< var days.d4.question >}}* Module 8 estimates every channel's effect at once from weekly spend and sales, a marketing mix model (MMM), with carryover, diminishing returns and honest intervals. Module 9 checks that model against experiments like yesterday's and asks what it cannot identify. Module 10 uses it to divide the budget, with the uncertainty included.

## Wrap-up: fixed and left open (20 minutes) {#wrap-up}

**1. Fixed and left open (6 minutes: 3 in pairs, 3 with the room).**

- Module 8 estimated each channel's return on ad spend (ROAS) with intervals, from observational data, with carryover (adstock) and diminishing returns (saturation).
- Module 9 tied the model to experiments (lift-test calibration), tested it on later weeks it had not seen, and named what it cannot identify.
- Module 10 turned the model into a budget split under constraints, and compared choosing by the average answer with choosing by the whole posterior.
- **Left open for Day 5:** the budget treats every customer in a channel alike. Some customers buy anyway, some only with an offer, and some are put off by it. Whom should we target?

**2. The running results table (8 minutes).** Add today's rows, with interval type and probability, run mode and, on synthetic data, the truth.

| Lab | Quantity | Your number | Interval (type, probability) | Truth (synthetic data) | Run (FULL or QUICK; yours or reference) |
|---|---|---|---|---|---|
| 8 | ROAS per channel against the true ROAS; channels whose interval contains the truth |  |  |  |  |
| 8 | Divergences in the final fit |  |  | — |  |
| 9 | ROAS interval width for the tested channel, before and after calibration |  |  |  |  |
| 9 | Error on held-out weeks (time-slice cross-validation) |  |  | — |  |
| 10 | Recommended budget split |  |  |  |  |
| 10 | Expected incremental sales against the current split; probability the new split is better |  |  |  |  |

: Day 4 rows of the running results table {.striped}

**3. One decision claim (6 minutes: 3 writing, 3 with the room).** Write one sentence in the form *number → rule → recommendation*: "Based on [number and interval from Lab N], I would [action], because [rule, with its cost or margin], unless [the assumption that would change it]." Two or three people read theirs; the room asks each: *what would change your mind?*
