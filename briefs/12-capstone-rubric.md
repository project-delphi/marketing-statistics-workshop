# Capstone: task, deliverables, timing and rubric

<!--
Owner: Pedagogy Expert; reviewer: Academic Director. Written 2026-10-09.
This file is what pairs and instructors read on Day 5. The capstone notebook's exercises, data and
checkpoint numbering are specified in the Academic Director's lab brief for Module 12
(briefs/12-capstone.md, in progress in parallel) and built by the Technical Expert; align the stage
letters A-E below with that notebook's checkpoint numbers once both exist.
Clock times mirror the run of show in days/_day-5-notes.md, computed from _variables.yml
(days.d5.blocks, modules.m12.minutes). If those change, recompute both files.
No numbers below are results: they are formats and rules.
-->

## The task

You are the retailer's analytics team. The chief marketing officer and the chief financial officer
have one question for you:

> **How should we split next quarter's marketing budget across channels, and which customers should get
> the retention offer?**

Answer it as a pair, with the week's methods and data, in one notebook, and present the answer as a
five-slide decision brief with its uncertainty. The data are the workshop's synthetic retailer, so the
true values are known; your instructors score your estimates against them. The same pipeline would run on
real data, where the truth is not known; that is why the intervals and the limits matter.

### The five stages

Each stage reuses one day of the week and hands a number to the next stage.

| Stage | Reuses | What you produce | Handed to |
|---|---|---|---|
| **1. What each customer is worth** | Days 1–2 | Expected customer value (margin, discounted, stated horizon) by acquisition channel, with named intervals; the acquisition cost cap each implies. | Stage 4 (valuing the customers each channel brings) |
| **2. Is search incremental?** | Day 3 | The region test that switched search off in 8 of 40 regions, analyzed with the Day 3 methods: sales lost (incremental sales) with a named interval and a placebo check, written as the lift-test row the MMM calibration needs. | Stage 3 (the lift test that calibrates the model) |
| **3. A calibrated marketing mix model** | Day 4 | The MMM with Stage 2's lift test added; fit checks (divergences, posterior predictive check); ROAS by channel with named intervals, before and after calibration. | Stage 4 |
| **4. Where the next dollar goes** | Day 4 | A budget split under stated constraints, valuing new customers by Stage 1, computed over the posterior draws; expected incremental margin against the current split, and the probability the new split is better. | The brief |
| **5. Whom to target** | Day 5 | An uplift targeting rule from the email experiment: send the offer where predicted uplift × margin exceeds the offer's cost; its expected profit against mailing everyone and mailing no one, scored on held-out customers. | The brief |
| **5. The decision brief** | All | Five slides (below). | The CMO and CFO |

**Using the reference.** Each stage ends with a checkpoint (A to E). If your pair has not passed a stage's
checkpoint by its time, run `workshop.use_reference(n)` for that stage and move on. A stage run on the
reference and **disclosed on slide 5** is not penalized: correctness is then judged on how you used that
stage's result. An undisclosed one is.

### The five-slide decision brief

Each slide has a headline sentence that states the finding, at most one chart, and a sentence under the
chart saying what to look at.

1. **The recommendation.** The budget split and the targeting rule, the money at stake, and how sure you
   are (an interval and the probability the recommendation beats the current plan). This is your decision
   claim: *number → rule → recommendation*.
2. **What a customer is worth.** Value by acquisition channel with intervals, and the acquisition cost caps
   that follow. (Your Day 2 memo is a good start.)
3. **What the marketing caused.** The geo test's lift with its interval, and how it changed the MMM's ROAS
   for the tested channel.
4. **Where the next dollar goes.** The allocation against the current split: expected incremental margin
   with its interval, the constraints you imposed, and whether a risk-averse choice would differ.
5. **Whom to target, and the limits.** The targeting rule and its expected profit; the two or three
   assumptions your recommendation depends on most; which stages used the reference or QUICK; and the
   next experiment you would run.

## Deliverables (hand in by 15:30)

- [ ] **The notebook**, run top to bottom, with checkpoints A to E passed on your code or on the reference.
  Its summary cell shows which.
- [ ] **The five slides**, as a PDF or a link your instructors can open (any slide tool, or five Markdown
  cells at the end of the notebook).
- [ ] **The disclosure line** on slide 5: which stages used the reference, and FULL or QUICK.

## Timing checkpoints

| Time | Checkpoint | You should have |
|---|---|---|
| 11:40 | Start | The notebook open, setup cells run, pair roles agreed (one types, both decide; swap at lunch). |
| 12:20 | **A** | Customer value by channel, with named intervals. |
| 13:00 | **B** | The geo test's lift and its interval, in the form the MMM calibration needs. Then lunch until 13:50. |
| 14:25 | **C** | Calibrated ROAS by channel; the fit checked. If the fit is still running at 14:15, stop it and use `QUICK` or the reference. |
| 14:45 | **D** | The allocation, its expected incremental margin and the probability it beats the current split. |
| 15:05 | **E** | The targeting rule and its expected profit on held-out customers. |
| 15:30 | Hand-in | Slides and notebook handed in. Then a 10-minute break; presentations start at 15:40. |

## Rubric (for instructors: score a pair in 5 minutes)

**Before the presentations**, prepare the answer key: the true values for each stage, from
`data/synthetic/truth.json` and from a `WORKED_EXAMPLE` FULL run of the capstone notebook (the reference
pipeline's own estimates, to compare intervals with). Keep it on one sheet.

**During the presentation** (4 minutes), tick the evidence on the score sheet as you hear it.
**In the questions** (2 minutes), ask the one question that settles the criterion you are least sure of:
"What would change your recommendation?", "Is that revenue or margin?", "What kind of interval is that?",
"Which stages used the reference?". **In the change-over** (1 minute), circle a level for each criterion and
write one sentence: one thing to keep, one to change.

Score each criterion from 1 to 4. A level is reached only when everything in it is true.

| Criterion | 4 · Excellent | 3 · Solid | 2 · Developing | 1 · Not yet |
|---|---|---|---|---|
| **1. Correctness against the known truth** | Every stage's estimate is traceable to the notebook, and its interval contains the truth, or a miss is noticed and explained (a 94% interval misses 6% of the time). No unit errors. | All but one interval contains the truth; the miss is not discussed. No error that changes a decision. | Several misses, or a unit error (revenue for margin, weekly for quarterly) that changes a decision. | Numbers missing or not traceable to the notebook, or a wrong sign or order of magnitude drives the decision. |
| **2. Quantified uncertainty** | Every headline number has a named interval (type and probability). Uncertainty is carried through the stages: the allocation and targeting are computed over posterior draws, and the brief gives the probability the recommendation is right. | Every headline number has a named interval, but one decision uses only point estimates, and the brief says so. | Some intervals, unnamed or at a library default; decisions made from point estimates without saying so. | No intervals. |
| **3. Decision clarity and economic logic** | One recommendation per decision, first, in money. The rules use margin and cost correctly: break-even ROAS = 1 ÷ margin, uplift × margin > offer cost, cost caps from a low quantile of value. Marginal, not average, returns drive the budget. Constraints are stated. | Clear recommendations with correct rules, with one gap (for example, average ROAS used where marginal matters, or no constraints stated). | Vague recommendations ("invest more in search"), or a rule that uses revenue for margin or ignores a cost. | No actionable recommendation, or one that contradicts the pair's own numbers. |
| **4. Honesty about limits and assumptions** | Names the two or three assumptions the recommendation depends on most (for example, what the MMM cannot identify, spillover between test regions, the value horizon, whether the uplift model carries over to new customers); shows at least one sensitivity; discloses reference and QUICK stages; names the next experiment. | Names the key assumptions and discloses reference and QUICK stages; no sensitivity. | Generic caveats ("all models are wrong") not tied to this decision, or a reference or QUICK stage not disclosed. | No limits stated, or certainty the analysis does not support ("TV returns exactly 3.1"). |
| **5. Communication** | Five slides, each with a headline sentence that states the finding; each chart has a sentence saying what to look at; on time; the first slide answers the question. | Clear structure; one cluttered slide or a small overrun. | Notebook screenshots without headlines, or more than a minute over time. | The brief cannot be followed, or there is none. |

: Capstone rubric: five criteria, four levels {.striped}

**Total** out of 20. The score is feedback for the pair, not a grade. As a guide: 17–20, ready to present
at work; 13–16, solid with one thing to fix; 9–12, revisit the weakest criterion's day; 8 or below, talk
with the pair after the session.

### Score sheet (one per pair)

| Pair: ____________ | 1 · Correctness | 2 · Uncertainty | 3 · Decision | 4 · Honesty | 5 · Communication | Total |
|---|---|---|---|---|---|---|
| Level (circle) | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | /20 |
| Evidence heard |  |  |  |  |  |  |

Keep: ____________________________________ Change: ____________________________________

### What to listen for

These are the errors most likely to cost a level. They are the misconceptions the week's debriefs target.

- **Revenue for margin.** Customer value, ROAS break-even and uplift profit must all be in margin.
- **Average for marginal.** The next dollar's return is below the average return under diminishing returns.
- **P(alive) for uplift.** A customer likely to buy is not necessarily one the offer changes.
- **In-sample scores.** The uplift model's Qini or uplift curve must be on held-out customers.
- **A lift test without its uncertainty.** Calibration needs the test's standard error, not only its point estimate.
- **Unnamed intervals.** "The interval" with no type or probability, or a library default.
- **Units and horizons.** Weekly against quarterly, a value horizon that differs between slides.
