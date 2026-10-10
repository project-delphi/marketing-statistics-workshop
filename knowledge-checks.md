---
title: "Knowledge checks"
subtitle: "Three to five questions per module, with answers, for warm-ups, debriefs and self-checks"
---

<!--
Owner: Pedagogy Expert; reviewer: Academic Director. Written 2026-10-09.
Instructor-facing. Not in the render list at the time of writing (agree with the UI Expert before
adding it; if it is added, point teach.qmd's link at the page instead of GitHub). Module titles come
from _variables.yml through var shortcodes, which resolve only when the page is rendered.
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
