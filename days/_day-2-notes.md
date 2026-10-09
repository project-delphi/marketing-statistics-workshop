<!--
Partial, included by day-2.qmd (UI Expert). Owner: Pedagogy Expert.
Sections: #warm-up, #clinic, #wrap-up. Durations match _variables.yml days.d2.blocks
(warmup 15, clinic 60, wrapup 20); clock times are on the generated timetable.
The memo the clinic reviews is the one Lab 5's Decision section asks for; check the memo's
required parts against briefs/05-*.md once it is final.
-->

## Warm-up: what carried over from Day 1 (15 minutes) {#warm-up}

Answer alone on paper for 5 minutes, notes closed. Compare with a neighbor for 4 minutes. The facilitator takes the most-missed questions for 5 minutes, then says in one minute where today starts.

**W1.** Two customers each made 4 repeat purchases in a 52-week calibration period. A's last purchase was in week 50, B's in week 20. Which has the higher P(alive) at week 52, and why?

::: {.callout-tip collapse="true" title="Answer"}
A. Both bought at the same rate while active, but B has been silent for 32 weeks, which would be unusual for someone buying about once every 10 weeks. The longer the silence relative to a customer's own rate, the more likely they have left. P(alive) depends on frequency and recency together.
:::

**W2.** "Recency" means two different things on Day 1. What are they, and which direction is "good" in each?

::: {.callout-tip collapse="true" title="Answer"}
In marketing RFM segments, recency is the time **since** the last purchase: smaller is better. In BTYD models (buy-till-you-die models such as BG/NBD), recency is the customer's age **at** the last purchase, measured from their first purchase: for a given age T, larger is better, because the customer was seen buying more recently. Mixing the two silently reverses a model's input.
:::

**W3.** Why were the BTYD model and the gradient-boosting model both scored on a holdout period, and what would make the machine-learning comparison unfair?

::: {.callout-tip collapse="true" title="Answer"}
A forecast is only tested on data it has not seen: both models are fitted on the calibration period and scored on purchases in the holdout period, with the same metric. The comparison is unfair if any feature for the machine-learning model uses data from the holdout period (leakage), or if the two are scored on different customers or metrics.
:::

**W4.** In Lab 3, the customers of one acquisition channel have a positive coefficient on the purchase process. Can you conclude that moving acquisition budget to that channel will bring in more frequent buyers?

::: {.callout-tip collapse="true" title="Answer"}
Not from this alone. The coefficient says that channel's customers bought more often than the reference channel's customers, holding the model's other inputs fixed. They may differ for reasons the channel did not cause (who the channel reaches), and more budget may bring in different, marginal customers. A budget decision also needs each channel's acquisition cost, which Module 5 brings in.
:::

**Why today.** Today's question is *{{< var days.d2.question >}}* Module 4 adds money to the purchase forecasts: spend per transaction, discounting and the horizon. Module 5 turns customer value into decisions (how much to pay to acquire a customer, when retention pays) and into a one-page memo, which the afternoon clinic reviews.

## CFO memo clinic (60 minutes) {#clinic}

**Goal.** Make your Module 5 decision memo survive a finance review: one clear recommendation, in money, with its uncertainty and the assumptions that would change it.

**What you bring.** The memo you wrote in Lab 5's Decision section: at most one page (about 250 words), headed by a one-sentence recommendation.

**Format.** Groups of three. Each round, one person is the **Author**, one the **CFO** and one the **Analyst**; roles rotate, so everyone plays each role once and every memo is reviewed. In a group of four, the fourth person is a second Analyst; in a pair, the CFO also uses the Analyst's checklist.

| Minutes | What happens |
|---|---|
| 0–5 | The facilitator explains the two roles, hands out the CFO card and the Analyst checklist below, and forms groups. |
| 5–44 | **Three rounds of 13 minutes**, one memo each: 2 minutes for the CFO and Analyst to read in silence; 4 for the CFO's questions; 3 for the Analyst's checklist; 2 for the Author to note what to change (no defending); 2 to rotate. |
| 44–54 | **Revise.** Each Author rewrites the headline sentence and the "what would change my mind" sentence, then swaps with one group member for a one-minute read. |
| 54–60 | **The room.** Two volunteers read their headline before and after. The facilitator names the most common gap they heard. |

**The CFO's card.** Ask up to four, and only from this card. The Author answers briefly; no slides, no notebook.

1. What exactly are you asking me to approve, in dollars, and for how long?
2. What is the range, and what kind of range is it?
3. Is that revenue or margin? Over what horizon, and at what discount rate?
4. If the true value is at the low end of your range, do we lose money?
5. What did you assume that you did not test?
6. What result would make you change the recommendation?

**The Analyst's checklist.** Tick each line that the memo meets; say which lines it misses.

- [ ] Customer value is margin, not revenue, and the margin rate is stated.
- [ ] The horizon and discount rate are stated, and the memo says how much the answer moves if they change.
- [ ] The interval is named (type and probability) and comes from the model in the notebook.
- [ ] The decision rule is stated, and it uses the uncertainty (for example, a cap set from a low quantile of customer value), not only the average.
- [ ] The money at stake is stated: how many customers, how many dollars.
- [ ] The Module 4 check that spend per transaction is unrelated to purchase frequency is mentioned.
- [ ] Average against marginal: the memo says whether its cap applies to the next customer acquired or to the average one.
- [ ] The recommendation is the first sentence, and there is only one.

**Roles of the facilitators.** One keeps time and calls each change of step aloud. The others circulate, listen, and write down the gaps they hear for the wrap-up. They do not fix anyone's numbers.

**Deliverable.** The revised memo with the Analyst's checklist ticked, pasted into a Markdown cell at the end of your Lab 5 notebook (or your own notes). You will reuse it for slide 2 of the capstone brief on Day 5.

## Wrap-up: fixed and left open (20 minutes) {#wrap-up}

**1. Fixed and left open (6 minutes: 3 in pairs, 3 with the room).**

- Module 4 put a dollar value on each customer: spend per transaction, discounted over a horizon, with an interval, and a separate model for subscription customers.
- Module 5 turned that value into decisions: acquisition cost caps, the value of segments with intervals, when a retention offer breaks even, and a memo a finance team can act on.
- **Left open for Day 3:** every decision today assumed we know what marketing *does*. The retention offer's lift and the channels' effects were assumed, not measured. Did the marketing cause the sales?

**2. The running results table (8 minutes).** Add today's rows to the table from Day 1, with interval type and probability, run mode and, on synthetic data, the truth.

| Lab | Quantity | Your number | Interval (type, probability) | Truth (synthetic data) | Run (FULL or QUICK; yours or reference) |
|---|---|---|---|---|---|
| 4 | Average spend per transaction (Gamma-Gamma) against the truth |  |  |  |  |
| 4 | Average discounted customer value (margin) |  |  |  |  |
| 4 | The same at another discount rate or horizon |  |  |  |  |
| 5 | Acquisition cost cap per channel |  |  |  |  |
| 5 | Retention offer: break-even lift in retention |  |  |  |  |
| Clinic | Your memo's headline sentence, revised |  |  |  |  |

: Day 2 rows of the running results table {.striped}

**3. One decision claim (6 minutes: 3 writing, 3 with the room).** Write one sentence in the form *number → rule → recommendation*: "Based on [number and interval from Lab N], I would [action], because [rule, with its cost or margin], unless [the assumption that would change it]." Today's is probably your memo's headline. Two or three people read theirs; the room asks each: *what would change your mind?*
