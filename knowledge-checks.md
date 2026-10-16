---
title: "Knowledge checks"
subtitle: "Three to five questions per module, with answers, for warm-ups, debriefs and self-checks"
---

<!--
Owner: Pedagogy Expert; reviewer: Academic Director. Written 2026-10-09.
Instructor-facing. In the render list (_quarto.yml) and linked from teach.qmd, not from the navbar.
Module titles come from _variables.yml through var shortcodes, which resolve only when the page is
rendered.
Modules 13-17 (the optional extensions) added 2026-10-10; Module 18 added 2026-10-16. They have no day page, so no "Used in"
lines. Their worked numbers follow the same rule as below; where an answer points to what a lab or
module page showed, it says so in words, and any number it quotes is on that module page. Their last
questions follow modules.m13-m17.decision as worded on 2026-10-10 and modules.m18.decision as worded on 2026-10-16.
Questions are numbered module.question, with anchors #q1-1 etc. The day pages' warm-ups use some of
them; the "Used in" line says where. Worked numbers in questions are invented for the question and
computed by hand; none is a lab result. The last question of each module follows that module's
decision as worded in modules.mNN.decision (checked 2026-10-09); if a decision changes, check it.
Library facts checked 2026-10-09: CLVTools 0.12.1 reference manual (covariates scale the Gamma rate
by exp(-gamma z) in the BG/NBD with static covariates); UCI Online Retail II page (invoice codes
starting with "c" are cancellations; CC BY 4.0); MineThatData blog (Hillstrom: 64,000 customers,
three randomized arms).
-->

Each module's block has three to five short questions that can be answered from the briefing and the lab, with the answer folded underneath. None is a trick question. Use them for:

- **Warm-ups** (Days 1 to 5): the day pages already carry four questions each, several taken from here.
- **Debriefs**: a question from the module just finished, when the room's numbers are on the board.
- **Self-checks**: participants can work through a day's modules that evening.
- **Extensions** (Modules 13 to 18): they have no day page, so use their questions as the warm-up or debrief of an extension you swap in, or as a self-check after it.

Accept a room's own numbers when they differ from a worked example here because of QUICK settings or a different runtime, as long as the reasoning holds.

## Module 1 · {{< var modules.m01.title >}}

[**1.1**]{#q1-1} In a BTYD summary table (BTYD: buy-till-you-die models), what are *frequency*, *recency* and *T* for a customer, and why does frequency leave out the first purchase?

::: {.callout-tip collapse="true" title="Answer"}
*Frequency* is the number of repeat purchases; *recency* is the customer's age at their last purchase, measured from the first purchase (0 if there was no repeat); *T* is their age at the end of the calibration period. The first purchase marks the start of the relationship (acquisition); the models describe what happens after it, so only repeat purchases count.
:::

[**1.2**]{#q1-2} "Recency" means something different in marketing RFM segments and in BTYD models. What is the difference, and why does it matter?

::: {.callout-tip collapse="true" title="Answer"}
In RFM segments, recency is the time *since* the last purchase: smaller is better. In BTYD models, it is the age *at* the last purchase: for a given age T, larger is better. Feeding one where the other is expected reverses the model's reading of who is active.

**Used in:** Day 2 warm-up (W2).
:::

[**1.3**]{#q1-3} Online Retail II has one row per product line of an invoice, and some invoice codes start with "C". What must you do before counting purchases per customer?

::: {.callout-tip collapse="true" title="Answer"}
Collapse lines to purchase occasions (one invoice, or one customer-day, is one purchase), otherwise a basket of ten products counts as ten purchases. Handle cancellations, whose invoice codes start with "C": remove them, or net them against the original invoice, and say which. Rows without a customer ID cannot be assigned to a customer.
:::

[**1.4**]{#q1-4} In a cohort table, the January cohort's month-6 retention is 18% and the June cohort's month-6 cell is empty. A colleague averages the month-6 column, counting the empty cell as 0%. What is wrong?

::: {.callout-tip collapse="true" title="Answer"}
The June cohort has not been observed for six months yet: its cell is missing (right-censored), not zero. Averaging it as 0% drags month-6 retention down. Average only cohorts observed for at least six months, and say how many there are.
:::

[**1.5**]{#q1-5} A colleague proposes to stop retargeting anyone who has not bought for 90 days. Using only what Module 1 computes, how do you decide whether the rule is good enough, and what does it miss?

::: {.callout-tip collapse="true" title="Answer"}
Count how many customers it would drop who are still active, and compare the margin their next purchases would bring with the retargeting cost the rule saves. On real data you cannot see who is still active, which is why Lab 1 measures it on the synthetic retailer, where the truth is known. The rule misses that customers buy at different rates: 90 days of silence is normal for a customer who buys twice a year and alarming for one who buys weekly. Module 2's P(alive) uses each customer's own rate.
:::

## Module 2 · {{< var modules.m02.title >}}

[**2.1**]{#q2-1} State the BG/NBD model's assumptions about purchasing while a customer is active, and about dropout. How does Pareto/NBD differ?

::: {.callout-tip collapse="true" title="Answer"}
While active, a customer buys as a Poisson process with rate λ, and λ varies across customers as a Gamma distribution (parameters r, α). After each purchase, the customer drops out with probability p, and p varies across customers as a Beta distribution (a, b). In Pareto/NBD, dropout can happen at any time, not only after a purchase: each customer's lifetime is exponential with rate μ, and μ varies as a Gamma distribution (s, β).
:::

[**2.2**]{#q2-2} Two customers each made 4 repeat purchases in a 52-week calibration period. A's last purchase was in week 50, B's in week 20. Which has the higher P(alive), and why?

::: {.callout-tip collapse="true" title="Answer"}
A. At about one purchase every 10 weeks, B's 32-week silence would be unusual for an active customer, so B has more likely left. P(alive) depends on frequency and recency together.

**Used in:** Day 2 warm-up (W1).
:::

[**2.3**]{#q2-3} In the BG/NBD model, what is P(alive) for a customer with no repeat purchases, and why? Is that a sensible answer?

::: {.callout-tip collapse="true" title="Answer"}
Exactly 1. In the BG/NBD, dropout can only happen right after a repeat purchase, so a customer who has never repeated cannot have dropped out. That is a modeling convenience, not a belief: a one-time buyer from two years ago is probably gone. Pareto/NBD, where dropout can happen at any time, gives such customers a P(alive) below 1. Check which model a P(alive) came from before acting on it for one-time buyers.
:::

[**2.4**]{#q2-4} How do you check a BTYD model before trusting its forecasts?

::: {.callout-tip collapse="true" title="Answer"}
Fit on a calibration period and compare its forecast with what happened in a holdout period: total repeat purchases, and purchases by calibration-frequency group. On synthetic data, also check that the true parameters fall inside their named intervals (for example, 94% HDI). A model that fits the calibration period but misses the holdout is not ready for decisions.
:::

[**2.5**]{#q2-5} A rule says "stop retargeting customers with P(alive) below 0.2". What does it need before it is a business decision?

::: {.callout-tip collapse="true" title="Answer"}
Money: the cost of contacting a customer, the margin per purchase, and the share of alive customers the ad makes buy (the response rate). Retarget when P(alive) × margin × response rate exceeds the cost, so the threshold is cost ÷ (margin × response rate); it follows from the economics, not from habit. And uncertainty: report how many customers fall below the threshold with an interval. Note that P(alive) says who is likely still buying, not whom retargeting would change; that is uplift, Module 11.
:::

## Module 3 · {{< var modules.m03.title >}}

[**3.1**]{#q3-1} In CLVTools with static covariates, one acquisition channel has a positive coefficient on the purchase process. What does it mean, and how big is the effect?

::: {.callout-tip collapse="true" title="Answer"}
That channel's customers buy more often than the reference channel's, other inputs held fixed. In CLVTools' parameterization, the covariate scales the Gamma rate α by $e^{-\gamma z}$, so the mean purchase rate $r/\alpha$ is multiplied by about $e^{\gamma}$ for that channel: a coefficient of 0.3 means about 35% more purchases while active.
:::

[**3.2**]{#q3-2} Why model the acquisition channel as a covariate in one model, rather than fit a separate model per channel?

::: {.callout-tip collapse="true" title="Answer"}
One model shares information across channels: small channels borrow strength from large ones, so their estimates are more stable, and the channel difference is estimated directly with its uncertainty. Separate models are more flexible (every parameter can differ) but noisy for small channels and harder to compare.
:::

[**3.3**]{#q3-3} In the machine-learning comparison, what is leakage, and how do you avoid it?

::: {.callout-tip collapse="true" title="Answer"}
Leakage is when a feature uses information from the period being predicted, for example a customer's purchase count computed over all data including the holdout. Compute every feature from the calibration period only, take the label (holdout purchases or spend) from the holdout period only, and score both models on the same customers with the same metric.
:::

[**3.4**]{#q3-4} Gradient boosting beat the BTYD model on holdout error. Name two reasons you might still use the BTYD model for some decisions.

::: {.callout-tip collapse="true" title="Answer"}
Any two of: it gives each customer a P(alive) and a forecast for any horizon, not only the one the machine-learning label was built for; its parameters have a meaning you can check and explain; it gives intervals from the model; it needs only the transaction log. Which model wins depends on the data and the metric; the lab reports the result that ran, not an assumed winner.
:::

[**3.5**]{#q3-5} The covariate effect says a channel's customers buy more often. Should acquisition budget move to that channel?

::: {.callout-tip collapse="true" title="Answer"}
Not on that alone. The effect describes the customers the channel has brought, who may differ for reasons the channel did not cause; more spend may bring in different, marginal customers; and the decision needs each channel's cost per acquired customer and their value in money (Day 2). What Module 3 decides is narrower: whether the channels differ enough to be treated differently in forecasts (in Lab 3, a rate more than 20% away from the reference channel with a 90% interval that excludes no difference).

**Used in:** Day 2 warm-up (W4).
:::

## Module 4 · {{< var modules.m04.title >}}

[**4.1**]{#q4-1} What does the Gamma-Gamma model assume about spend and frequency, and how do you check it?

::: {.callout-tip collapse="true" title="Answer"}
That a customer's average spend per transaction is unrelated to how often they buy. Check, among customers with at least one repeat purchase, that the correlation between frequency and average spend is close to zero. If it is not, the model's value forecasts are biased.

**Used in:** Day 3 warm-up (W1).
:::

[**4.2**]{#q4-2} Why is a customer's estimated average spend pulled toward the population average, and more so for customers with few purchases?

::: {.callout-tip collapse="true" title="Answer"}
The estimate combines the population distribution with the customer's own transactions (partial pooling, or shrinkage). With few transactions, the customer's own average is noisy, so the population carries more weight; with many, their own average dominates.
:::

[**4.3**]{#q4-3} A customer brings \$100 of margin at the end of each year they remain a customer, stays each year with probability 0.8, and the discount rate is 10%. What is their expected lifetime value with no horizon limit? And with retention 0.7?

::: {.callout-tip collapse="true" title="Answer"}
$\sum_{t \ge 1} 100 \times (0.8/1.1)^t = 100 \times 0.8/(1.1 - 0.8) \approx \$267$. With retention 0.7: $100 \times 0.7/0.4 = \$175$. A 10-point drop in retention cuts the value by about a third.

**Used in:** Day 3 warm-up (W2).
:::

[**4.4**]{#q4-4} In a subscription business, aggregate retention rises over time: 60% of new subscribers renew after year 1, but 80% of those still active renew after year 3. Have customers become more loyal?

::: {.callout-tip collapse="true" title="Answer"}
Not necessarily. If each customer has their own constant renewal probability, the less loyal ones leave first, so the survivors are increasingly the loyal ones and the aggregate rate rises with no change in any individual. This is the shifted-beta-geometric (sBG) model's point; projecting the year-1 rate forward would undervalue the customers.
:::

[**4.5**]{#q4-5} Finance asks for "the" customer lifetime value. Why give a range, and which assumptions should be stated beside it?

::: {.callout-tip collapse="true" title="Answer"}
The value depends on the horizon, the discount rate and whether it is margin or revenue, and the model's estimate has its own uncertainty. State margin, horizon and discount rate, give the model's interval (type and probability), show how much the value moves under other reasonable horizons or rates, and name the assumption (horizon, discount rate or spend model) that moves it most.
:::

## Module 5 · {{< var modules.m05.title >}}

[**5.1**]{#q5-1} A channel's average customer value (margin) has a 90% equal-tailed interval of \$80 to \$140. Which acquisition cost cap makes you about 95% sure that a customer is worth more than they cost, and what else must hold?

::: {.callout-tip collapse="true" title="Answer"}
\$80, the 5th percentile. The value must be margin, not revenue, and it must describe the next customer the extra spend brings, who is often worth less than the average one.

**Used in:** Day 3 warm-up (W3).
:::

[**5.2**]{#q5-2} A retention offer costs \$5 per customer and raises 12-month retention from 0.60 to 0.65. A retained customer's remaining value is \$120 margin. Does it pay, and what is the weakest number in the calculation?

::: {.callout-tip collapse="true" title="Answer"}
Expected gain per customer offered: $0.05 \times 120 = \$6 > \$5$, so it pays by \$1 per customer. The break-even lift is $5/120 \approx 4.2$ points. The weakest number is the lift of 5 points: it is an assumed causal effect, and only a randomized test measures it (Day 3). Its margin over break-even is small.
:::

[**5.3**]{#q5-3} The 90% intervals for two segments' total value overlap. Does that mean the segments are not different in value?

::: {.callout-tip collapse="true" title="Answer"}
No. Compute the difference in each posterior draw and look at the interval of the difference. Overlapping intervals can still give a difference whose interval excludes zero, especially when the two estimates are correlated.
:::

[**5.4**]{#q5-4} When customer-level values go back to the warehouse for marketing systems to use, what should be written besides the point estimate?

::: {.callout-tip collapse="true" title="Answer"}
The interval bounds (with their type and probability), P(alive), the horizon, discount rate and margin assumption, the model version and the date of the data it was fitted on. Downstream users need the uncertainty to set thresholds and the provenance to know when the numbers are stale.
:::

[**5.5**]{#q5-5} What must the first sentence of a decision memo to a CFO contain?

::: {.callout-tip collapse="true" title="Answer"}
The one recommendation, the money it concerns, and how sure you are: for example, "Cap acquisition cost at \$X for channel Y; we are about 95% confident new customers there are worth more than that, over a 3-year horizon at a 10% discount rate." Assumptions and what would change the recommendation follow.
:::

## Module 6 · {{< var modules.m06.title >}}

[**6.1**]{#q6-1} Baseline conversion is 5%. About how many customers per arm do you need to detect a 0.5-point increase at the 5% level (two-sided) with 80% power, using $n \approx 16\,p(1-p)/\delta^2$? What if you want to detect 0.25 points?

::: {.callout-tip collapse="true" title="Answer"}
$16 \times 0.05 \times 0.95/0.005^2 = 30{,}400$ per arm. Halving the detectable effect multiplies $n$ by four: about 121,600 per arm.

**Used in:** Day 4 warm-up (W1).
:::

[**6.2**]{#q6-2} A planned 50/50 split gives 50,600 and 49,400 of 100,000 customers. What do you do?

::: {.callout-tip collapse="true" title="Answer"}
Stop and investigate before reading any result. The chi-square statistic is $2 \times 600^2/50{,}000 = 14.4$, $p \approx 0.0001$: a sample ratio mismatch. Assignment or logging is broken, so the groups may not be comparable.

**Used in:** Day 4 warm-up (W2).
:::

[**6.3**]{#q6-3} Why does checking a fixed-horizon test every day, and stopping the first time $p < 0.05$, give too many false positives? Name a fix.

::: {.callout-tip collapse="true" title="Answer"}
Each look is another chance for noise to cross the threshold, so over many looks the chance of a false positive is well above 5%. Fix the sample size in advance and look once, or use a sequential method designed for repeated looks, which adjusts the threshold.
:::

[**6.4**]{#q6-4} CUPED adjusts the outcome using a pre-experiment covariate (for example, last quarter's spend). If the correlation between covariate and outcome is 0.6, by how much does the variance fall, and why is the estimate still unbiased?

::: {.callout-tip collapse="true" title="Answer"}
The variance is multiplied by $1 - 0.6^2 = 0.64$, so the same precision needs about 36% fewer customers. It stays unbiased because the covariate was measured before treatment and randomization makes it unrelated to treatment.
:::

[**6.5**]{#q6-5} Last-click attribution credits search with 40% of conversions. Why is that not the share you would lose by pausing search?

::: {.callout-tip collapse="true" title="Answer"}
Attribution assigns credit to touches; it does not ask what would have happened without them. Many of those customers would have converted anyway. Only a comparison with a holdout measures the incremental share.

**Used in:** Day 4 warm-up (W4).
:::

## Module 7 · {{< var modules.m07.title >}}

[**7.1**]{#q7-1} Treated regions sold 100 before and 120 after the campaign; control regions sold 80 before and 90 after. What is the difference-in-differences estimate, and what assumption does it rest on?

::: {.callout-tip collapse="true" title="Answer"}
$(120 - 100) - (90 - 80) = 10$. It assumes parallel trends: without the campaign, treated regions would have changed by the same amount as the controls. Check the pre-period trends; a gap that was already opening before the campaign breaks it.
:::

[**7.2**]{#q7-2} How does synthetic control build its counterfactual, and why are its weights usually non-negative and summing to one?

::: {.callout-tip collapse="true" title="Answer"}
It finds a weighted combination of control regions that tracks the treated region closely in the pre-period, and uses that combination after the start as "what would have happened". Non-negative weights summing to one keep the counterfactual inside the range of the controls (an interpolation, not an extrapolation), which makes it easier to trust and to explain.
:::

[**7.3**]{#q7-3} What is a placebo test, and what result would make you distrust an estimated 6% lift?

::: {.callout-tip collapse="true" title="Answer"}
Apply the same method where nothing happened: to control regions as if treated, or to a date before the campaign. If many placebos produce gaps as large as 6%, the method produces such gaps from noise alone and the estimate is not distinguishable from noise.

**Used in:** Day 4 warm-up (W3).
:::

[**7.4**]{#q7-4} Why estimate a geo-test's power by simulation rather than with a textbook formula?

::: {.callout-tip collapse="true" title="Answer"}
With few regions, sales that are correlated week to week and an estimator like synthetic control, no simple formula applies. Inject known lifts into past data, run the exact analysis you plan, and count how often it detects them; that gives the smallest lift the design can detect.
:::

[**7.5**]{#q7-5} The geo test's 90% interval for incremental margin is \$20k to \$140k and the campaign cost \$60k. Should you scale it?

::: {.callout-tip collapse="true" title="Answer"}
The data cannot say: the cost lies inside the interval. Under a rule fixed in advance such as "scale if the lower end exceeds the cost", the answer is no, not yet: extend the test, or redesign it so that the detectable lift is below break-even. Report the implied return on the spend with its range (margin ÷ cost: about 0.33 to 2.3 times), and the posterior probability that the margin exceeds \$60k, if the analysis gives one.
:::

## Module 8 · {{< var modules.m08.title >}}

[**8.1**]{#q8-1} A channel's geometric adstock has retention rate 0.6, normalized so the weights add to 1. What share of a week's spend effect falls in that week, and in the next?

::: {.callout-tip collapse="true" title="Answer"}
The weights are proportional to $1, 0.6, 0.36, \dots$, which add to $1/(1 - 0.6) = 2.5$. So 40% in the same week and 24% in the next.

**Used in:** Day 5 warm-up (W1).
:::

[**8.2**]{#q8-2} At current spend, a channel's average revenue ROAS is 3.0 and its marginal ROAS is 1.5; gross margin is 40%. Should spend rise or fall?

::: {.callout-tip collapse="true" title="Answer"}
Fall. Break-even revenue ROAS is $1/0.40 = 2.5$. On average the channel pays (3.0 > 2.5), but the next dollar returns only \$1.50 of revenue, \$0.60 of margin, so it loses money. Budget decisions use the marginal return.
:::

[**8.3**]{#q8-3} Why set priors on channel effects, for example in proportion to each channel's share of spend, and how do you check them?

::: {.callout-tip collapse="true" title="Answer"}
With a few years of weekly data and correlated channels, the data alone allow implausible splits of credit; weakly informative priors keep effects in a believable range. Check with a prior predictive check: simulate sales from the priors alone and confirm they cover plausible values without being absurd. Then compare prior and posterior to see what the data changed.
:::

[**8.4**]{#q8-4} The sampler reports divergent transitions. What do they mean, and what do you do?

::: {.callout-tip collapse="true" title="Answer"}
The sampler could not follow the posterior's shape somewhere, so it may have missed part of it and the estimates can be biased. Do not ignore them and do not just draw more samples. Raise the target acceptance rate, reparameterize, or tighten priors that allow implausible regions; then refit and check that the divergences are gone.
:::

[**8.5**]{#q8-5} TV's revenue ROAS has a 94% HDI of 1.8 to 3.4; margin is 40%. Is TV profitable?

::: {.callout-tip collapse="true" title="Answer"}
Break-even is 2.5, inside the interval, so the data cannot decide. Report the posterior probability that ROAS exceeds 2.5; under Lab 8's rule TV pays back only if that probability is at least 0.9, does not if it is at most 0.1, and is otherwise undecided, and an interval that straddles break-even like this one points to undecided, which a lift test would settle. Remember too that average ROAS is not the return on the next dollar.

**Used in:** Day 5 warm-up (W2).
:::

## Module 9 · {{< var modules.m09.title >}}

[**9.1**]{#q9-1} What does adding a lift test to an MMM do to the model?

::: {.callout-tip collapse="true" title="Answer"}
It adds a term to the likelihood that ties the model's predicted effect of the tested spend change to the measured lift, weighted by the test's uncertainty. Splits of credit that disagree with the experiment become unlikely, so the tested channel's ROAS interval narrows and may move, and correlated channels can shift too.
:::

[**9.2**]{#q9-2} Why validate an MMM on later weeks (time-slice cross-validation) rather than random folds?

::: {.callout-tip collapse="true" title="Answer"}
Adstock, trend and seasonality connect neighboring weeks, so a random fold lets the model see the weeks around each test week: leakage. Fit on weeks up to a cut-off and forecast the weeks after it, then move the cut-off forward.
:::

[**9.3**]{#q9-3} Two MMMs fit past sales equally well but give different ROAS for social. How is that possible?

::: {.callout-tip collapse="true" title="Answer"}
When channels move together or a channel's spend barely varies, the data identify only the combined effect: many splits of credit fit equally well. Good fit does not validate the split; experiments, priors and stability checks do.

**Used in:** Day 5 warm-up (W3).
:::

[**9.4**]{#q9-4} You refit the model on the first two years and on the last two years, and one channel's ROAS halves. What do you conclude?

::: {.callout-tip collapse="true" title="Answer"}
Either the channel's effect changed (new creative, saturation, a market change) or the data cannot pin it down. Either way, a budget decision that leans on that channel is fragile: widen its uncertainty in the allocation, constrain how far it can move, and make it the next lift test.
:::

[**9.5**]{#q9-5} Robyn and a Bayesian MMM give different answers for the same data. Name one difference in how they work.

::: {.callout-tip collapse="true" title="Answer"}
Robyn fits ridge regressions, searches the adstock and saturation hyperparameters with an evolutionary optimizer, and returns a set of candidate models that trade off prediction error against how far the channel shares are from the spend shares, from which the analyst picks one. A Bayesian MMM puts priors on those parameters and returns one posterior, with intervals from the model. So the "uncertainty" means different things in each.
:::

## Module 10 · {{< var modules.m10.title >}}

[**10.1**]{#q10-1} Without binding constraints, what condition holds at the best budget split, and why not put everything into the channel with the highest average ROAS?

::: {.callout-tip collapse="true" title="Answer"}
The marginal return of the last dollar is the same in every channel. With diminishing returns, each extra dollar in a channel returns less; the highest average ROAS often belongs to a channel already near saturation.

**Used in:** Day 5 warm-up (W4).
:::

[**10.2**]{#q10-2} Why limit how far each channel's spend may move, for example to ±30% of its current level?

::: {.callout-tip collapse="true" title="Answer"}
The model is informed only near the spend levels it has seen; far outside them, its response curves are extrapolation, often driven by priors. There are also practical limits: contracts, minimum presence, team capacity.
:::

[**10.3**]{#q10-3} How does optimizing over posterior draws differ from optimizing the response curve at the posterior mean, and what does a risk-averse objective change?

::: {.callout-tip collapse="true" title="Answer"}
Response curves are non-linear, so the average outcome over draws differs from the outcome at the average parameters. Optimizing over draws lets you report the distribution of outcomes and the probability that the new split beats the current one. A risk-averse objective (for example, a low quantile of incremental margin) moves money away from channels whose effect is uncertain.
:::

[**10.4**]{#q10-4} What changes when the objective values new customers by their lifetime value instead of first-purchase revenue?

::: {.callout-tip collapse="true" title="Answer"}
Channels that bring customers who keep buying gain budget, and channels that bring one-off buyers lose it, even at the same short-term ROAS. It connects Day 4's allocation to Days 1 and 2; the lifetime values' uncertainty should be carried into the objective, not only their means. The shift is only as good as the cost per new customer and the lifetime value behind it: in Lab 10 the costs were assumptions, so the CLV-weighted plan is reported as a sensitivity beside the recommendation, not as the recommendation.
:::

## Module 11 · {{< var modules.m11.title >}}

[**11.1**]{#q11-1} Name the four kinds of customer an offer meets, and which one targeting by uplift aims at.

::: {.callout-tip collapse="true" title="Answer"}
Persuadables (buy only with the offer), sure things (buy anyway), lost causes (never buy) and sleeping dogs (buy less because of the offer). Uplift targeting aims at persuadables. A response model, which targets those most likely to buy, mostly finds sure things and wastes the offer on them.
:::

[**11.2**]{#q11-2} Why does uplift modeling need data from a randomized experiment, such as the Hillstrom email test?

::: {.callout-tip collapse="true" title="Answer"}
Uplift is the difference between a customer's outcome with and without the offer, and only one is ever observed. Randomization makes the treated and control customers comparable, so differences in outcome between similar customers in the two groups estimate the effect. In Hillstrom, 64,000 customers were split at random into a men's email, a women's email and no email.
:::

[**11.3**]{#q11-3} What does a Qini or uplift curve show, and why must it be computed on held-out customers?

::: {.callout-tip collapse="true" title="Answer"}
Customers are sorted by predicted uplift; the curve shows the cumulative incremental responses as you target a growing share, against the straight line of random targeting. The area between them summarizes the ranking's value. On the training data, a flexible model ranks its own noise well, so the curve looks better than it is.
:::

[**11.4**]{#q11-4} A customer's predicted uplift in conversion probability is 0.4 points, each conversion brings \$50 of margin, and the email costs \$0.10. Send it? What if it costs \$0.25?

::: {.callout-tip collapse="true" title="Answer"}
Expected incremental margin is $0.004 \times 50 = \$0.20$. At \$0.10, send (gain \$0.10). At \$0.25, do not (loss \$0.05). The rule is: send when predicted uplift × margin exceeds the cost.
:::

[**11.5**]{#q11-5} grf's RATE (rank-weighted average treatment effect) for a causal forest's ranking has a confidence interval that includes zero. What do you conclude?

::: {.callout-tip collapse="true" title="Answer"}
There is no evidence that the ranking finds customers with larger effects than average. Targeting by it is not justified; decide for everyone at once from the average treatment effect (send to all if the average effect × margin exceeds the cost, otherwise to none), or collect more data.
:::

## Module 12 · {{< var modules.m12.title >}}

[**12.1**]{#q12-1} Why analyze the geo test before fitting the final MMM in the capstone?

::: {.callout-tip collapse="true" title="Answer"}
The geo test's lift, with its uncertainty, calibrates the MMM for the tested channel. Without it, the model's split of credit between correlated channels rests on priors and fit alone, and the allocation inherits that weakness.
:::

[**12.2**]{#q12-2} How do you carry uncertainty from the customer values and the MMM into the budget recommendation?

::: {.callout-tip collapse="true" title="Answer"}
Compute the allocation's outcome in each posterior draw, using the draws of the response curves and of customer value together, rather than plugging in means. Then report the interval of incremental margin and the probability that the recommended split beats the current one.
:::

[**12.3**]{#q12-3} The budget allocation and the uplift rule both use a margin figure. Why must it be the same figure, and what else must they agree on?

::: {.callout-tip collapse="true" title="Answer"}
Otherwise the two halves of the recommendation answer different economic questions and cannot be added up. They must also agree on the horizon (a quarter, a year, a lifetime), on revenue against margin, and on the units of spend.
:::

[**12.4**]{#q12-4} What belongs on the last slide of the capstone brief?

::: {.callout-tip collapse="true" title="Answer"}
The two or three assumptions the recommendation depends on most, one sensitivity result, which stages used the reference or QUICK, and the next experiment you would run to reduce the biggest uncertainty.
:::

## Module 13 · {{< var modules.m13.title >}}

[**13.1**]{#q13-1} A customer buys 4 units of a product on 3 March, and on 5 March a `C` invoice cancels all 4 (same stock code, same price). What does Module 1's cleaning do with the purchase, what does Lab 13's pipeline do, and why must the matching run before the log is split at the calibration end?

::: {.callout-tip collapse="true" title="Answer"}
Module 1's rules drop the `C` line but keep the purchase, so 3 March stays a purchase day, with its amount in the customer's mean spend, for an order the customer cancelled. The pipeline matches the cancellation to the latest earlier purchase line of the same customer, stock code and price with the opposite quantity (an ASOF join) and drops both; if nothing else was bought that day, the purchase day disappears. Matching needs the whole log: a cancellation dated after the calibration end can cancel a purchase dated before it, and if the log is split first, the calibration period keeps a purchase that was cancelled.
:::

[**13.2**]{#q13-2} A cleaning rule changes next year's total forecast revenue by 2% but moves the scores of 15% of customers by more than 10%. Does it matter for the forecast you send to finance? For a list of customers to contact?

::: {.callout-tip collapse="true" title="Answer"}
Hardly for the forecast: 2% is likely within the forecast's own error, and the customers' changes partly cancel in the total. A lot for the list: a list is decided customer by customer at a threshold, so customers near the bar move on or off it. Judge a cleaning rule by the output the decision uses (here, how many customers it adds to the list and drops from it), and report a customer's change in money: a percent change from a near-zero score can run to millions of percent.
:::

[**13.3**]{#q13-3} A few wholesale customers order about forty times as much per purchase day as retail customers. Should you cap purchase-day amounts at the 99th percentile to fit the Gamma-Gamma, to value customers, or both?

::: {.callout-tip collapse="true" title="Answer"}
To fit, not to value. The cap stops a few bulk orders from setting the spread of spend toward which every customer's estimate is shrunk, so the spend parameters land near what the retail customers' own data give. Valuing with capped amounts cuts each wholesale customer's own mean to the cap and undervalues them badly; value each customer on their own mean, with the parameters fitted on capped amounts. Compute the cap from calibration purchase days only, so nothing from the holdout leaks in. Better still, give bulk buyers a segment of their own (the lab's stretch).
:::

[**13.4**]{#q13-4} A retention contact costs £10, the margin is 30% and the contact is assumed to lift purchases by 10%. Above what expected 26-week revenue does a contact pay? A customer's expected 26-week revenue is £500: on the list or off it? And if the lift is really 5%? Would you spend next week on more cleaning or on measuring the lift?

::: {.callout-tip collapse="true" title="Answer"}
Contact when 0.10 × 0.30 × revenue > £10, so above £10 ÷ 0.03 = £333.33. At £500 the expected gain is 0.10 × 0.30 × £500 = £15 > £10: on the list. At a 5% lift the bar doubles to £10 ÷ 0.015 = £666.67 and the gain is £7.50: off. Halving the lift drops every customer between £333 and £667 from the list; on Online Retail II that moved far more customers than any cleaning or counting choice (the module page's decision). So measure the lift: contact a random part of the list and hold the rest back (Module 6). Cleaning still decides *who* is on the list, so keep the tested pipeline and report which choice moves the list most.
:::

## Module 14 · {{< var modules.m14.title >}}

[**14.1**]{#q14-1} Subscribers joined steadily over the last two years. Dashboard A reports 12-month retention as the share of all subscribers who have not cancelled within 12 months of joining. Dashboard B uses only subscribers whose 12-month status is known: those who cancelled within 12 months or were observed for more than 12. Which is too high, which too low, and what replaces both?

::: {.callout-tip collapse="true" title="Answer"}
A is too high: a subscriber who joined 3 months ago and is still active counts as retained for 12 months, though only 3 have been observed. B is too low: recent joiners who already cancelled stay in, while recent joiners still active are dropped, so the subscribers it keeps lean toward those who cancel. Both mishandle **censored** subscribers, whose outcome is known only up to the end of their observation. Kaplan-Meier uses each subscriber for exactly as long as they were observed: at risk until they cancel or their observation ends.
:::

[**14.2**]{#q14-2} An analyst compares subscribers who contacted support in their first six months with those who did not, starting both clocks at sign-up, and finds that those who contacted support churn less. What is wrong, and how do you fix it?

::: {.callout-tip collapse="true" title="Answer"}
**Immortal time.** To be in the contact group with a contact in month 5, a subscriber had to stay until month 5, so the group is credited with months it was guaranteed to survive. Define groups only by what is known at time zero, or start the clock at a **landmark**: at month 6, take the subscribers still active, group them by whether they contacted support in the first six months, and compare what happens after. A time-varying covariate is the other fix. In Lab 14, onboarding is known at time zero (the first billing date, after the trial), which is why onboarded and other subscribers can be compared from there.
:::

[**14.3**]{#q14-3} Why does the onboarding decision use the restricted mean survival time within 24 months, RMST(24), rather than the hazard ratio, and why 24 months rather than a whole lifetime?

::: {.callout-tip collapse="true" title="Answer"}
The decision is about money: months subscribed times the monthly margin. RMST(24), the area under the survival curve up to 24 months, is the expected number of months subscribed in the first 24, so a difference in it converts directly into dollars. A hazard ratio compares rates at each tenure; how many months it adds depends on the baseline churn, so it cannot be priced on its own. 24 months because the data follow nobody much longer (sign-ups began in January 2024 and observation ends in June 2026, so no one is observed for 30 months). Kaplan-Meier describes only the tenures observed; a lifetime needs a model that extrapolates, such as the sBG of Module 4, and its assumptions.
:::

[**14.4**]{#q14-4} Onboarding adds 2.5 months of subscription within 24 months for subscribers who did not onboard, standardized by plan. A month brings \$12 of margin and an onboarding call costs \$15. What share of called subscribers must the call get to onboard for it to pay? At 35%, does it pay? What must you learn before rolling calls out?

::: {.callout-tip collapse="true" title="Answer"}
Break-even $q^* = c / (\Delta\text{RMST} \times m) = 15 / (2.5 \times 12) = 15/30 = 0.50$: half of the called subscribers. At $q = 0.35$ a call returns $0.35 \times 2.5 \times 12 = \$10.50$ for \$15, so it does not pay. (A \$3 nudge would pay from $3/30 = 10\%$.) Two things are unknown. The 2.5 months is an association: onboarding was not randomized, and motivation may drive both onboarding and staying. And the rule assumes that onboarding prompted by a call works like onboarding chosen during the trial. A randomized pilot (Module 6) that measures $q$ and retention settles both before any rollout.
:::

## Module 15 · {{< var modules.m15.title >}}

[**15.1**]{#q15-1} In double machine learning, why must each customer's predicted install probability and predicted spend come from a model that never saw that customer? What happens if a flexible model is fitted on all customers and predicts the same customers?

::: {.callout-tip collapse="true" title="Answer"}
A flexible model can memorize the customers it was fitted on: its in-sample predicted install probability is close to the actual 0 or 1, so the install residual $T - \hat e$ is close to 0 for almost everyone. The estimate, $\sum \tilde T \tilde Y / \sum \tilde T^2$, then divides noise by almost nothing: the standard error balloons and the point estimate jumps from seed to seed. **Cross-fitting** (predict each fold from a model fitted on the other folds) keeps the residuals honest, and because the estimate is insensitive to small errors in the predictions (Neyman orthogonality), imperfect machine-learning predictions still give a valid interval.
:::

[**15.2**]{#q15-2} Half of the customers had a 50% chance of installing the app and would gain \$20 of spend from it; the other half had a 90% chance and would gain \$5. What is the average effect over all customers (the ATE)? What average does residual-on-residual regression estimate, and which customers does it lean on?

::: {.callout-tip collapse="true" title="Answer"}
ATE $= (20 + 5)/2 = \$12.50$. Residual-on-residual regression weights each customer's effect by $e(1 - e)$: $0.5 \times 0.5 = 0.25$ and $0.9 \times 0.1 = 0.09$. Its target is the **overlap-weighted** average, $(0.25 \times 20 + 0.09 \times 5)/(0.25 + 0.09) = 5.45/0.34 \approx \$16.03$. It leans on customers whose install was a toss-up and says least about those almost sure either way. Trimming extreme propensities changes the target again. Which average matters depends on whom the decision is about.
:::

[**15.3**]{#q15-3} Your estimate of the app's effect is \$12 of spend (95% CI \$10 to \$14), and leaving one strong observed confounder out of the covariates moved it up by \$5. A colleague asks whether a confounder you did not measure could overturn the decision, whose break-even is \$8.33. What can the data tell you, and what can they not?

::: {.callout-tip collapse="true" title="Answer"}
The data cannot test whether every common cause of install and spend was measured: no unobserved confounding is an assumption. What you can do: name the plausible missing causes (purchase intent at launch, for example); use the observed confounder as a yardstick, since one as strong, unmeasured and pushing the same way would move the interval down by about \$5, to \$5 to \$9, whose lower end is below \$8.33, so the decision would flip; run a formal sensitivity analysis (the robustness value in the lab's stretch); and settle it with a randomized test (Module 6).
:::

[**15.4**]{#q15-4} The retailer would pay a \$2.50 credit per install, the gross margin is 30%, and your estimate of the extra 26-week spend per persuaded install is \$12 (95% CI \$9 to \$15). Under Lab 15's rule, do you offer the credit, and what is the largest credit the rule allows? What if only half of the customers who install after the offer were persuaded by it?

::: {.callout-tip collapse="true" title="Answer"}
Margin per persuaded install: $0.30 \times 12 = \$3.60$ (95% CI \$2.70 to \$4.50). The rule offers the credit when the lower end exceeds it: \$2.70 > \$2.50, so offer (equivalently, the effect's interval lies above the break-even $2.50/0.30 = \$8.33$), and the largest credit the rule allows is \$2.70. If half the installers would have installed anyway, each persuaded install costs $2.50/0.5 = \$5.00$ of credits, more than \$2.70: do not offer. Only a randomized test of the credit measures the persuaded share. Say which unobserved confounder could overturn the answer (15.3).
:::

## Module 16 · {{< var modules.m16.title >}}

[**16.1**]{#q16-1} Five channels have removal effects 0.60, 0.50, 0.40, 0.30 and 0.20, and the tracked journeys hold 1,000 conversions. How many does Markov attribution credit to the first channel? Why do removal effects add up to more than 1, and what does dividing by their sum assume?

::: {.callout-tip collapse="true" title="Answer"}
The removal effects add up to 2.0, so the first channel gets $1{,}000 \times 0.60/2.0 = 300$. A converting journey through $k$ channels is lost in each of the $k$ removal scenarios, so its conversion is counted $k$ times in the sum. Dividing by the sum is a convention that makes the credit add up to the conversions; no argument about cause stands behind it. Removing a channel from a fitted chain is a counterfactual about the chain, not about what users would have done.
:::

[**16.2**]{#q16-2} Users who saw only email convert at 2%, users who saw only search at 6%, and users who saw both at 10%; users with no touch are not in the data, so the empty set is worth 0. What are the two Shapley values for a journey with both? A colleague would rather split the 10% in proportion to what each channel makes alone. Which is right?

::: {.callout-tip collapse="true" title="Answer"}
Average each channel's addition over the two orders. Email first adds 0.02, then search adds 0.08; search first adds 0.06, then email adds 0.04. So $\varphi_\text{email} = (0.02 + 0.04)/2 = 0.03$ and $\varphi_\text{search} = (0.08 + 0.06)/2 = 0.07$, adding up to 0.10: each keeps what it makes alone, and the 0.02 they make only together is split equally. The proportional split gives 0.025 and 0.075. Neither is right about cause: both are rules for sharing a value. Shapley's rules are symmetry, efficiency and the null-player property; whether the shares measure what a channel causes depends on the value function, and an observed conversion rate also reflects who each channel reaches.
:::

[**16.3**]{#q16-3} Markov and Shapley attribution agree that retargeting earns about a quarter of the conversions, and their 95% bootstrap intervals are narrow. Is that evidence that retargeting causes those conversions?

::: {.callout-tip collapse="true" title="Answer"}
No. A bootstrap interval covers the sampling noise in the credit, not bias, and both methods share the same bias: they read the same journeys, in which retargeting reaches users who were already likely to buy. When exposure follows intent, credit follows intent, and efficiency puts every tracked conversion, including those that would have happened anyway, on some channel. In Lab 16 retargeting had no effect by construction, and both methods still gave it about a quarter of the credit (the module page's results). Only users randomly kept from the channel show what it causes.
:::

[**16.4**]{#q16-4} A paid channel spends \$3,000 a month, a conversion brings \$10 of margin, and attribution credits the channel with 600 conversions a month. Withholding it from 20% of the users it reaches for a month gives a minimum detectable effect (MDE) of about 260 conversions a month, scaled to all the users it reaches; a 10% holdout gives about 350. Do you move budget now? If you test, with which holdout, and what result would justify the move?

::: {.callout-tip collapse="true" title="Answer"}
Break-even: $\$3{,}000 / \$10 = 300$ incremental conversions a month. The 20% holdout can detect 260, below 300, so it can tell a channel that pays back from one that does not; the 10% holdout (350) cannot, so it would need to be larger or longer. Test first with the 20% holdout, and move budget toward the channel only if it shows more than 300 incremental conversions a month, a cost per incremental conversion below \$10. Comparing the MDE with the 600 credited conversions would be a weaker bar: a test that can detect only 500 would pass it and still not tell break-even from nothing. The margin at risk during the test, if the credit were real: $0.20 \times 600 \times \$10 = \$1{,}200$.
:::

## Module 17 · {{< var modules.m17.title >}}

[**17.1**]{#q17-1} A channel spent \$1.0 million over its first two years at a ROAS of 5.0, and \$0.5 million in the latest year at 2.0. Gross margin is 30%. What is its ROAS over all three years, and what does each window say about next year's budget?

::: {.callout-tip collapse="true" title="Answer"}
All-period ROAS $= (5.0 \times 1.0 + 2.0 \times 0.5)/(1.0 + 0.5) = 6.0/1.5 = 4.0$, above the break-even of $1/0.30 \approx 3.33$: "keep". In the latest year each dollar returned \$2.00 of sales and \$0.60 of margin, so it lost money. ROAS over a window is a ratio of sums, so the all-period number weights each year by its spend and stays above break-even long after the channel stopped paying back. A decision about next year needs the recent window, which a static MMM cannot give: it estimates one effect per channel for every week.
:::

[**17.2**]{#q17-2} You correlate a static MMM's weekly residuals with each channel's spend, year by year. One channel's correlation goes from +0.4 in the first year to −0.5 in the last; the others stay near 0. What does it suggest, and what else could cause it?

::: {.callout-tip collapse="true" title="Answer"}
That the channel's effect per dollar has fallen. In a year when it works better than the static model's single effect, sales jump more than predicted in its high-spend weeks, so the residuals rise with its spend; when it works worse, they fall with it. A change of sign that stands out from the other channels is the warning. Other causes: a change in the baseline that lines up with the channel's spend, or a change in how the channel spends (bursts one year, steady the next). The check flags; it does not measure, and it needs spend that varies within a year. A flagged channel is a reason to fit a model whose effects can drift, or to run a lift test, not to cut it on the spot.
:::

[**17.3**]{#q17-3} Robyn fits Prophet to sales and uses its trend, seasonality and holiday components as the MMM's baseline. Why can that baseline take credit from a channel whose effect is fading, and why does the MMM's ROAS interval not show the extra uncertainty?

::: {.callout-tip collapse="true" title="Answer"}
Prophet is fitted to sales, and sales include what the media produced. A slowly fading channel lowers sales slowly, which is what a trend with changepoints follows; once the MMM takes that trend as given, part of the channel's effect already sits in the baseline, and the channel gets less credit. Prophet's components enter as fixed columns, its point estimates, so their own uncertainty never reaches the MMM: the ROAS intervals move but do not widen. A baseline estimated together with the media, in a form that cannot follow one channel's decline, avoids using the sales twice; outside evidence such as a lift test settles it.
:::

[**17.4**]{#q17-4} A drifting MMM's 4,000 posterior draws of the fading channel's last-quarter ROAS have mean 2.5, and 3,480 of them lie below the break-even of 3.33. The static model's all-period ROAS is above 3.33 in 99% of its draws. Last quarter the channel spent \$200,000. What does Lab 17's rule say, what did last quarter's spend return net of its cost, and why do the two models disagree?

::: {.callout-tip collapse="true" title="Answer"}
$P(\text{ROAS} < 3.33) = 3{,}480/4{,}000 = 0.87$, below the 0.9 needed to cut, and $P(\text{ROAS} > 3.33) = 0.13$ is far from the 0.9 needed to keep: run a lift test first, dated in the current quarter (a test from last year measures last year's effect). Last quarter's spend returned $200{,}000 \times (0.30 \times 2.5 - 1) = -\$50{,}000$ net at the mean (the net is linear in ROAS, so this is also its mean over the draws), an average return, not the next dollar's. The static model answers a question about the whole history: it pools every week into one effect per channel, so it says "keep" with near certainty about the past, not about now.
:::

## Module 18 · {{< var modules.m18.title >}}

[**18.1**]{#q18-1} A dashboard reports that, among users sent a push notification, those who opened it brought \$1.90 of margin that day and those who did not \$0.16, so "a push is worth \$1.74". In the randomized send experiment on the same days, eligible user-days that got a send brought \$0.39 and eligible user-days that did not \$0.25. What is the same-day effect of a send, why is the dashboard's number ten times larger, and what would the dashboard show if the push did nothing at all?

::: {.callout-tip collapse="true" title="Answer"}
The effect is $0.39 - 0.25 = \$0.14$ per send: among eligible user-days, a coin flip decided who was sent, so the not-sent days are a fair picture of what the sent days would have done. Opening is the user's choice, and engaged users open more often and buy more whether or not they are notified, so the opener gap measures who opens, not what the send did. If the push did nothing, the gap would barely move: the engaged users would still be the ones opening and buying. In Lab 18 the gap was about \$1.81 and the true same-day effect \$0.14, inside the experiment's 95% interval. Two details of the comparison matter: keep only eligible user-days (a user who has opted out sits in the not-sent group but could never be sent), and the effect per send is small because most sends are not opened and most sessions do not end in an order.
:::

[**18.2**]{#q18-2} A volume experiment assigns users at random to 1, 3 or 7 reminders a week for a quarter. Week-1 opens per user are 0.1, 0.3 and 0.4; the share who opted out by day 90 is 1%, 5% and 45%; 90-day margin per user is \$18, \$22 and \$18. The never-send arm earns \$13. A colleague wants the daily cadence because it has the most opens. Why do the two metrics pick different cadences, and what does the never-send arm tell you about the quarter's number?

::: {.callout-tip collapse="true" title="Answer"}
Opens count sends that were read, and they rise with every extra send; they never see what volume costs. Two things happen only with time. Sends keep the habit alive (an opened notification is a session, and a session today raises tomorrow's organic visit), which is why even one reminder a week earns \$5 per user more than never sending. And sends accumulate fatigue: at seven a week nearly half the users opt out, and an opted-out user contributes nothing more from notifications, or nothing at all after an uninstall. Three a week is where the second curve starts to overtake the first. The never-send arm (\$13) is the margin a user brings with no notifications at all, the floor the other arms are measured against. The 90-day number still understates the cost of the daily cadence: a user lost on day 60 is also lost next quarter. In Lab 18, daily sends won week-1 opens and three a week won the quarter.
:::

[**18.3**]{#q18-3} In one segment a send raises the next seven days' margin by \$0.10 on average, the probability that a send makes the user opt out is 0.002, and the value at stake is a user's \$40 of 90-day margin. A push costs nothing to send; an SMS costs \$0.05. Does the uplift rule send a push, and an SMS? Why can the same rule not choose how many sends a week?

::: {.callout-tip collapse="true" title="Answer"}
The rule charges a send its expected loss of the base plus its price: $\text{cost} = 0.002 \times 40 + \text{price}$. A push costs $0.08$, below the \$0.10 effect, so send; an SMS costs $0.08 + 0.05 = 0.13$, above it, so do not. The charge for opting out is usually the larger part, and it is what a per-send price leaves out. The rule prices one send at the fatigue level the experiment happened to have; a policy that sends every day moves every user up the fatigue curve, where the opt-out probability per send is several times higher, so the same rule would say less. It is a screen for whom to send to, not a schedule; choosing how often needs the state the user is in, which the bandit and the value-iteration policy of Lab 18 add.
:::

[**18.4**]{#q18-4} A contextual bandit's reward for a user-day is the margin earned that day, minus a penalty if the user opted out that day. In one context a send earns \$0.30 on average and a skip \$0.25, and a send makes the user opt out with probability 0.02. Which action wins with penalty 0, and with a penalty of \$10? Then: value iteration on the volume log gives a Q value for "send" in a state where the experiment never sent. Why does `greedy_policy` refuse to choose it?

::: {.callout-tip collapse="true" title="Answer"}
With penalty 0 a send wins ($0.30 > 0.25$), and it wins in every context where a send adds anything today, because the day's margin is all the bandit sees and an opt-out costs it nothing. It sends to nearly everyone nearly every day and burns the base, like the daily cadence; in Lab 18 the penalty-0 bandit lost about 41% of its users in the quarter. With a \$10 penalty the expected reward of a send is $0.30 - 0.02 \times 10 = \$0.10$, below the skip's \$0.25, so skip wins: the penalty makes the opt-out cost something today, and the bandit learns to stop in high-fatigue contexts. It still does not see the habit effect, a session's lift to tomorrow's organic visit, which arrives later and is credited to whatever action is played then. The untried action: its Q value is built from a reward and transitions that were never observed in that state, so it is a guess that value iteration treats as a fact, and off-policy methods tend to overrate exactly such actions. Keeping to actions the data tried at least 30 times in the state (the lab's bar) is the simplest guard; the production systems in the briefing use conservative Q-learning for the same reason.
:::

[**18.5**]{#q18-5} The growth team's guardrail: a policy may lose at most 10% of the reachable base to opt-outs in a quarter. Three candidates: A earns \$21,000 per 1,000 users over 90 days and loses 5%; B earns \$22,000 and loses 8%, +\$900 against the best fixed cadence with a 95% paired interval of +\$750 to +\$1,050; C earns \$22,500 and loses 14%. Which does Lab 18's rule choose? A 180-day rerun then shows C still ahead (\$44,000 per 1,000 against B's \$43,500), having lost 20% of the base. Does the rerun overturn the guardrail?

::: {.callout-tip collapse="true" title="Answer"}
The rule chooses B: the highest 90-day margin among the policies within the guardrail, and its interval excludes zero, so the gain over the fixed cadence is measured, not noise. C is out before its margin is read. The rerun does not overturn the rule: it measures what the quarter cannot, and what it shows is that C's lead shrinks (\$1,500 at 90 days, \$500 at 180) while its losses keep compounding, with a fifth of the base gone for every later quarter. Where the guardrail sits is a business choice about how much of the base to spend for margin, fixed before the numbers are read; the rerun is evidence for the next time the team sets it, and the Decision cell's "what would change it" line is where it belongs. In Lab 18 the 10 V bandit was chosen within the 10% guardrail (8.5% lost, +\$865 against three a week, 95% paired interval +\$730 to +\$999); the 3 V bandit, excluded at 13.6%, still earned the most at 180 days with 19.7% of the base gone.
:::
