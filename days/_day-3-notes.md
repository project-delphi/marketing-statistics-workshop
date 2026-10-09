<!--
Partial, included by day-3.qmd (UI Expert). Owner: Pedagogy Expert.
Sections: #warm-up, #clinic, #wrap-up. Durations match _variables.yml days.d3.blocks
(warmup 15, clinic 60, wrapup 20); clock times are on the generated timetable.
The clinic's demo is the GeoLift market-selection part of labs/r/07-causalimpact-geolift.ipynb;
its run time on Colab comes from the readiness page, never from this file.
-->

## Warm-up: what carried over from Day 2 (15 minutes) {#warm-up}

Answer alone on paper for 5 minutes, notes closed. Compare with a neighbor for 4 minutes. The facilitator takes the most-missed questions for 5 minutes, then says in one minute where today starts.

**W1.** The Gamma-Gamma model of spend per transaction rests on one assumption you should check before using it. What is it, and how do you check it?

::: {.callout-tip collapse="true" title="Answer"}
That a customer's average spend per transaction is unrelated to how often they buy. Check it among customers with at least one repeat purchase: the correlation between purchase frequency and average spend should be close to zero. If heavy buyers spend systematically more or less per order, the model's value forecasts are biased.
:::

**W2.** A customer brings \$100 of margin at the end of each year they are still a customer. Each year they stay with probability 0.8, and the discount rate is 10% a year. What is their expected lifetime value with no horizon limit?

::: {.callout-tip collapse="true" title="Answer"}
$\sum_{t=1}^{\infty} 100 \times 0.8^t / 1.1^t = 100 \times \dfrac{0.8}{1.1 - 0.8} \approx \$267$. If retention fell to 0.7, the value would fall to $100 \times 0.7/0.4 = \$175$: value is very sensitive to retention, which is why Module 4 shows it across assumptions.
:::

**W3.** A channel's average customer value (margin) has a 90% equal-tailed interval of \$80 to \$140. What acquisition cost cap would make you about 95% sure that a customer is worth more than they cost? What else must be true for that cap to be safe?

::: {.callout-tip collapse="true" title="Answer"}
\$80. A 90% equal-tailed interval runs from the 5th to the 95th percentile, so the value exceeds \$80 with probability about 0.95. For the cap to be safe, the value must be margin, not revenue, and it must describe the *next* customer the extra spend brings in; marginal customers are often worth less than the average one.
:::

**W4.** Module 5's retention break-even used an assumed lift in retention from the offer. Where should that number come from?

::: {.callout-tip collapse="true" title="Answer"}
From a randomized experiment: offer it to a random part of the eligible customers, hold out the rest, and compare their retention. Comparing customers who took the offer with those who did not mixes the offer's effect with who chooses to take it. That is today's subject.
:::

**Why today.** Today's question is *{{< var days.d3.question >}}* Module 6 measures effects with randomized experiments and the mistakes that break them. Module 7 measures effects when only regions can be randomized, or nothing can: geo-experiments and quasi-experiments. The afternoon clinic designs a geo-test.

## Geo-test design review (60 minutes) {#clinic}

**Goal.** Design a geo-test that can answer a budget question before any money is spent on it, and have another team try to break the design.

**The scenario.** The retailer wants to know whether a regional campaign raises sales enough to roll it out nationally. You have the weekly sales history of the same geo panel as Lab 7 and a fixed test budget. Your design must say which regions get the campaign, for how long, how you will analyze the result, and what result would make you scale, stop or redesign.

**Format.** Teams of three or four. One person in each team is the **timekeeper**; one is the **scribe** who fills the template; all review. During the review, each team is the **red team** for another.

| Minutes | What happens |
|---|---|
| 0–12 | **Demo.** The facilitator walks through GeoLift's market selection in the R notebook for Module 7, from output prepared before the session: candidate sets of treated markets, how well each can be matched by the others before the test, and the effect size each could detect. |
| 12–15 | The facilitator gives the scenario and the template; teams form. |
| 15–35 | **Design.** Teams fill in the one-page template below, using Lab 7's power-by-simulation results and the demo's output. |
| 35–51 | **Review.** Pairs of teams: each presents its design in 3 minutes, then the other team applies the review checklist for 4; then swap. Two minutes to move. |
| 51–57 | **Revise.** Fix the most serious issue the red team found. |
| 57–60 | **The room.** The facilitator names the two most common flaws. |

**The one-page template.**

1. **Decision.** "We roll out nationally if …" (in money).
2. **Primary metric.** The outcome, its unit and level (for example, weekly sales per region).
3. **Break-even lift.** Campaign cost ÷ margin per unit of sales = the incremental sales needed to pay back; as a percentage of baseline sales in the treated regions.
4. **Treated and control regions.** How they were chosen (the market-selection output, a pre-period fit) and how many are treated.
5. **Duration and power.** Test length, and the smallest lift the design can detect (from simulation). Is it below the break-even lift?
6. **Analysis plan, fixed before the test.** The estimator (synthetic control, difference-in-differences or CausalImpact), the length of the pre-period, and the interval you will report (type and probability).
7. **Threats and checks.** Spillover between neighboring regions, other campaigns or holidays in the window, placebo tests (in time and across regions), and what you will do if the pre-period fit is poor.
8. **Decision rule.** For example: "Scale if the lower end of the 90% interval for incremental margin is above the campaign cost; stop if the upper end is below; otherwise extend the test or redesign it."

**The red team's checklist.**

- [ ] The decision rule is written before the data, in money.
- [ ] The detectable lift is smaller than the break-even lift. If not, the test cannot answer the question, whatever the result.
- [ ] Treated regions are chosen by a stated rule, not by preference.
- [ ] The pre-period is long enough to fit the controls, and the design says what happens if the fit is poor.
- [ ] Spillover and contamination are addressed.
- [ ] Placebo tests are planned.
- [ ] The reported interval is named (type and probability).

**Roles of the facilitators.** Prepare the demo output before the session. Do not run market selection live: it runs many simulations and the room should not wait on it. Its measured run time, once recorded, is on the [readiness page](readiness.qmd). If a team plans to analyze several treated markets with GeoLift, point them to aggregating the markets into one series (as Lab 7 does) or to CausalImpact: GeoLift's analysis function, `GeoLift()`, fails with two or more test markets at the version the workshop pins, while market selection and power, used in the demo, work. During design and review, circulate and ask the checklist questions; do not hand out a design.

**Deliverable.** One completed template per team, photographed or pasted into a Markdown cell of your Lab 7 notebook. It is the plan you will follow in Stage 2 of the capstone on Day 5.

## Wrap-up: fixed and left open (20 minutes) {#wrap-up}

**1. Fixed and left open (6 minutes: 3 in pairs, 3 with the room).**

- Module 6 measured causal effects with randomized experiments: sample size before the test, checks on the split, the cost of peeking, and variance reduction with pre-period data. It also showed why attribution reports are not incrementality.
- Module 7 measured effects when only regions can be randomized, or nothing can: synthetic control, difference-in-differences and CausalImpact, checked with placebo tests.
- **Left open for Day 4:** an experiment measures one channel, in one period, at one spend level. We cannot test every channel all the time. Where should the next dollar go across all channels, and how do experiments keep that model honest?

**2. The running results table (8 minutes).** Add today's rows, with interval type and probability, run mode and, on synthetic data, the truth.

| Lab | Quantity | Your number | Interval (type, probability) | Truth (synthetic data) | Run (FULL or QUICK; yours or reference) |
|---|---|---|---|---|---|
| 6 | Sample size per arm for the planned detectable effect |  |  |  |  |
| 6 | Sample-ratio check on the split: p-value |  |  |  |  |
| 6 | Interval width for the lift, without and with CUPED |  |  |  |  |
| 6 | Incremental margin per email: roll out or not? |  |  |  |  |
| 7 | Geo-panel lift estimate against the true lift |  |  |  |  |
| 7 | Incremental sales and the return on the campaign spend |  |  |  |  |
| 7 | Placebo tests: how many were as large as the real estimate |  |  | real or synthetic: |  |
| 7 | Proposition 99 effect estimate (real data) |  |  | real data |  |
| Clinic | Your design's detectable lift and break-even lift |  |  |  |  |

: Day 3 rows of the running results table (CUPED: variance reduction using pre-experiment data) {.striped}

**3. One decision claim (6 minutes: 3 writing, 3 with the room).** Write one sentence in the form *number → rule → recommendation*: "Based on [number and interval from Lab N], I would [action], because [rule, with its cost or margin], unless [the assumption that would change it]." Two or three people read theirs; the room asks each: *what would change your mind?*
