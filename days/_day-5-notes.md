<!--
Partial, included by day-5.qmd (UI Expert). Owner: Pedagogy Expert.
Sections: #warm-up, #capstone, #wrap-up.
Clock times in the capstone run of show are computed from _variables.yml (recomputed 2026-10-09
after days.d5.blocks split the capstone lab around lunch): days.d5 starts 09:00; warmup 15, m11
50+55+10, break 15 put the capstone at 11:25; m12 minutes {briefing: 15, lab: 180, debrief: 60}
with lab segments of 80 and 100 minutes give briefing 11:25-11:40, pair work 11:40-13:00, lunch
13:00-13:50, pair work 13:50-15:30, break 15:30-15:40, presentations 15:40-16:40; wrapup
16:40-17:00. These match the generated timetable (_includes/day-5.md). If days.d5.blocks or
modules.m12.minutes change, recompute. briefs/12-capstone-rubric.md repeats the checkpoint
times and must be changed with this table.
The task, deliverables and rubric are in briefs/12-capstone-rubric.md.
-->

## Warm-up: what carried over from Day 4 (15 minutes) {#warm-up}

Answer alone on paper for 5 minutes, notes closed. Compare with a neighbor for 4 minutes. The facilitator takes the most-missed questions for 5 minutes, then says in one minute where today starts.

**W1.** A channel has geometric adstock with retention rate 0.6, normalized so that the weights add up to 1. What share of a week's spend effect lands in that same week? And in the following week?

::: {.callout-tip collapse="true" title="Answer"}
The weights are proportional to $1, 0.6, 0.36, \dots$, which add up to $1/(1 - 0.6) = 2.5$. So the same week gets $1/2.5 = 40\%$ and the next week $0.6/2.5 = 24\%$. More than half the effect comes after the week of spend, which is why a week-by-week comparison of spend and sales understates the channel.
:::

**W2.** TV's revenue ROAS has a 94% HDI of 1.8 to 3.4, and the gross margin is 40%. Is TV profitable?

::: {.callout-tip collapse="true" title="Answer"}
Break-even revenue ROAS is $1 / 0.40 = 2.5$: each dollar of spend must bring \$2.50 of revenue to pay for itself in margin. 2.5 lies inside the interval, so the data cannot say whether TV pays. Report the posterior probability that ROAS exceeds 2.5, computed from the draws. And remember that average ROAS is not the return on the *next* dollar: with diminishing returns, the marginal return is lower.
:::

**W3.** Two MMMs fit past weekly sales equally well, yet give different ROAS for social. Why is that possible, and what does a geo lift test add?

::: {.callout-tip collapse="true" title="Answer"}
When channels move together, or a channel's spend barely varies, different splits of credit fit the same sales history: good fit does not identify the split. A lift test measures the effect of a known change in one channel's spend, so adding it to the model (calibration) rules out splits that disagree with the experiment. The tested channel's ROAS interval narrows, and may move.
:::

**W4.** Why does the optimizer not move the whole budget to the channel with the highest average ROAS?

::: {.callout-tip collapse="true" title="Answer"}
Because of diminishing returns: as a channel's spend grows, each extra dollar returns less. Without binding constraints, the best split is where the *marginal* return of the last dollar is the same in every channel. The highest average ROAS often belongs to a channel already near saturation.
:::

**Why today.** Today's question is *{{< var days.d5.question >}}* Module 11 finds whom an offer changes, not just who buys: uplift. Then, in the capstone, you put the week together: one decision pipeline for the retailer, presented as a five-slide brief.

## Capstone run of show {#capstone}

You work in **pairs** on one notebook (one person types, both decide; swap at lunch). The task, the five-slide brief and the rubric your instructors score with are in the [capstone task and rubric]({{< var repo.url >}}/blob/{{< var repo.ref >}}/briefs/12-capstone-rubric.md); read the rubric before you start. Each stage ends with a **checkpoint** in the notebook. If a stage has not passed its checkpoint at its time, run `workshop.use_reference(n)` for that stage and move on, and say so on your last slide. A disclosed reference stage costs nothing on the rubric; a pair that falls behind loses the later stages, which cost a lot.

| Time | Minutes | Stage | Checkpoint at the end |
|---|---|---|---|
| 11:25–11:40 | 15 | **Briefing.** The task, the data, the rubric, the checkpoints. Sit with your pair (announced at the end of Day 4), open the notebook, run the setup cells. | — |
| 11:40–12:20 | 40 | **1. What each customer is worth.** Customer value forecasts by acquisition channel, with intervals. | A: values by channel, with named intervals |
| 12:20–13:00 | 40 | **2. Is search incremental?** Analyze the region test that switched search off, with the Day 3 methods. | B: lift estimate with its interval |
| 13:00–13:50 | 50 | **Lunch.** Pairs that passed B may start the calibrated MMM fit before leaving, with the Colab tab kept open. | — |
| 13:50–14:25 | 35 | **3. A calibrated MMM.** Add the geo lift to the marketing mix model; check the fit (divergences, intervals). | C: calibrated ROAS by channel |
| 14:25–14:45 | 20 | **4. Where the next dollar goes.** Allocate next quarter's budget, valuing customers by Stage 1, with uncertainty. | D: allocation and the probability it beats the current split |
| 14:45–15:05 | 20 | **5. Whom to target.** Uplift targeting: send the offer where expected incremental margin exceeds its cost. | E: targeting rule and its expected profit |
| 15:05–15:30 | 25 | **5. The decision brief.** Five slides, one per question in the rubric's brief structure. Hand in by 15:30. | Slides handed in |
| 15:30–15:40 | 10 | **Break.** | — |
| 15:40–16:40 | 60 | **Presentations.** Each pair presents; instructors score with the rubric. | — |
| 16:40–17:00 | 20 | **Wrap-up** of the day and the week (below). | — |

: Capstone run of show. The pair work runs 11:40–13:00 and 13:50–15:30: 180 minutes, around lunch. {.striped}

**Presentations.** With up to 8 pairs: 4 minutes to present, 2 for questions, 1 to change over (56 minutes). With 9 to 12 pairs: 3 minutes, 1 question, 1 to change over. With more than 12 pairs, the room splits in two with one scorer each. Your brief should still make sense if you only get to say the first sentence of each slide.

**Long fits.** If the MMM fit in Stage 3 is not finished 10 minutes before Checkpoint C, stop it, switch on `QUICK`, or use the reference for that stage. Say which on your last slide.

## Wrap-up: the week (20 minutes) {#wrap-up}

**1. Fixed and left open (6 minutes: 3 in pairs, 3 with the room).** Walk the week's line of questions, one sentence each.

- Day 1: who is still a customer, from purchase history alone.
- Day 2: what a customer is worth, in margin, with an interval.
- Day 3: whether marketing caused the sales, measured by experiments and quasi-experiments.
- Day 4: where the next dollar goes, from a mix model held to those experiments.
- Day 5: whom an offer changes, and one pipeline that connects all five.
- **Left open:** what your own data would need for this pipeline: which experiments you would run first, and which assumption of today's capstone is least safe at your company.

**2. The running results table (8 minutes).** Add today's rows, then look at the whole week's table: which number moved most when a later module improved on an earlier one?

| Lab | Quantity | Your number | Interval (type, probability) | Truth (synthetic data) | Run (FULL or QUICK; yours or reference) |
|---|---|---|---|---|---|
| 11 | Average treatment effect of the offer against the truth |  |  |  |  |
| 11 | Qini coefficient or area under the uplift curve, on held-out customers |  |  | — |  |
| 11 | Share of customers targeted; expected profit against mailing everyone |  |  |  |  |
| 12 | Your brief's headline recommendation |  |  |  |  |

: Day 5 rows of the running results table {.striped}

**3. One decision claim (6 minutes: 3 writing, 3 with the room).** Write the decision claim from your capstone brief, in the form *number → rule → recommendation*, and under it one thing you will do differently at work next week. Two or three people read theirs.
