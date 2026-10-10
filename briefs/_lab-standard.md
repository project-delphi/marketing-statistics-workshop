# Lab standard: the rhythm every lab follows

<!--
Owner: Pedagogy Expert. Readers: Academic Director (when writing briefs/NN-*.md) and Technical
Expert (when building labs/src/NN-*.py|.R). The notebook format and the harness API are fixed in
CONTRIBUTING.md; this file adds how a lab should feel to a learner and how a reviewer checks it.
Not a rendered page (the leading underscore makes it a partial). Written 2026-10-09.
-->

A lab is 50 to 55 minutes in the room (`modules.mNN.minutes.lab` in `_variables.yml`). In that time a
working analyst, tired after a briefing, must be able to read each instruction once, know what to do,
do it, and know whether it worked. Everything below serves that.

## 1. The lab budget

Plan the minutes before writing a single exercise. For a lab of *L* minutes:

| Part | Minutes | What happens |
|---|---|---|
| Open and install | 5 | Open from the module page, Run the install and harness cells, read the header. The install runs while the learner reads. |
| Core exercises | *L* − 15 (35 to 40) | Four to six exercises of 5 to 12 minutes each. Their stated minutes must add up to no more than *L* − 15. |
| Decision cell | 5 | Number → rule → recommendation (section 7). Nobody misses it. |
| Slack | 5 | Slow installs, a restart, one wrong turn. Not planned for content. |
| Stretch | 0 | Outside the budget, for people who finish early (section 6). |

Worked example for a 55-minute lab: 5 + (8 + 6 + 10 + 8 + 8 = 40) + 5 + 5 = 55.

**Compute counts against the exercise it runs in.** A model fit that takes 3 minutes on Colab uses 3 of that
exercise's minutes. While a long cell runs, give the learner something to do: the next exercise's Predict
prompt, or a question about the plot that is about to appear. The plan's compute targets for Colab FULL
are an install under 3 minutes, no cell over 4 minutes and no more than 15 minutes of compute in total;
they are targets until a run record on the [readiness page](../readiness.qmd) says otherwise. Fit each model
once per lab (CONTRIBUTING.md); later exercises reuse that fit.

**Minutes in headings are promises.** `## Exercise 3 · Fit the BG/NBD (10 minutes)` means a typical
participant finishes it in about 10 minutes, including reading. Until a pilot measures it, the number is an
estimate and the brief says so.

**Two-notebook labs** (Modules 3, 7 and 11: two notebooks with different names, CONTRIBUTING.md) share one
slot, so the exercise minutes of both notebooks together stay within *L* − 15. The first markdown cell of
each notebook carries an **Order of work** note that says:

- which notebook comes first, in the order of `modules.mNN.notebooks` in `_variables.yml` (the order the
  lab step table shows), and which install cell to start first: the slower one, even if its notebook comes
  second;
- the exercises and minutes in each notebook;
- the minute of the lab at which to switch even if unfinished (`use_reference(n)` gets past the rest), and
  where the lab's decision is made.

Both notes say the same thing in the same words, and `teach.qmd` repeats the switch minutes.

## 2. Exercise sizing: one task, one function

- **One exercise, one task, one function.** The learner writes one function with a stated signature,
  inputs and output (column names, shapes, units). If the task needs the word "and" twice, split it.
- **5 to 12 minutes.** Under 5, merge it with a neighbor or make it a provided Run cell. Over 12, split it.
- **One new idea per exercise.** A new library call, a new statistical idea or a new piece of business
  logic, not all three at once. Scaffolding (loading data, plotting, printing tables) lives in provided Run
  cells, never in the learner's function.
- **Short solutions.** The folded solution is usually under 15 lines. If it is longer, the exercise is too big
  or some of it belongs in a provided helper.
- **No hidden state.** The function takes what it needs as arguments and returns its result. No reading
  globals, no writing files, no relying on a cell above having been run twice.
- **The stub is runnable.** It raises `NotImplementedError("TODO N")` (Python) or `stop("TODO N")` (R), so
  Run all stops there with the harness's recovery message.
- **The first checkpoint comes before any model fit or long download**, so a broken setup shows up in the
  first ten minutes, not the fortieth.

## 3. Predict prompts: a concrete, checkable guess

The learner writes the guess down before running anything. A good Predict prompt has an answer that the
next cell will prove right or wrong.

A good prompt asks for one of these:

- a **number** with a scale to anchor it ("roughly what share of customers bought only once: under 25%,
  25 to 50%, or over 50%?");
- a **sign or direction** ("will the channel coefficient for referral be positive or negative?");
- a **ranking** ("which channel will have the widest ROAS interval?");
- a **choice among two to four options** ("will CUPED make the interval narrower, wider, or leave it the same?");
- a **comparison with a stated reference** ("will the holdout forecast be above or below the 1,200 repeat
  purchases that actually happened?").

| Weak (do not write) | Strong (write this) |
|---|---|
| "Think about what the model will do." | "Before you fit: will customers with 10 or more repeat purchases have P(alive) above or below 0.9? Write one number for a typical one." |
| "What do you expect the posterior to look like?" | "Which parameter will have the widest 94% HDI relative to its mean: r, α, a or b? Pick one." |
| "Predict the effect of CUPED." | "The unadjusted 95% interval for the lift is about ±1.2 points. With CUPED, will it be narrower, wider or the same? If narrower, by about how much?" |

Rules:

- **Give the scale.** A learner cannot guess a number with no reference. State the unit and a reference value
  the notebook has already shown.
- **Ask for the guess in writing.** "Write your guess in the cell below" or "on paper". A guess only in the
  head is revised after the fact.
- **One prediction per exercise.** Two at most.
- **Never a trick.** The prompt tests the idea, not attention to a detail hidden in the data.
- **Only ask about what the learner has seen.** Every term in the prompt was defined earlier in this notebook
  or in the briefing.

## 4. Explain prompts: compare with the prediction, name the mechanism

The Explain cell comes after the checkpoint passes. It asks the learner to say, in one or two sentences,
why the result is what it is.

A good Explain prompt:

1. **sends the learner back to the prediction** ("Compare with what you wrote. Were you above or below?");
2. **asks for the mechanism**, not a description ("Which feature of the data makes the display interval
   wide: less variation in spend, a shorter history, or spend that moves with TV?");
3. **points at the evidence** ("Point to the plot or the number that shows it.").

The `<details><summary>Why this solution works</summary>` block under it gives the expected explanation
in three to six sentences, names the mechanism, and says what would change the result. It does not
re-teach the briefing; it links the section of the module page instead.

| Weak | Strong |
|---|---|
| "Explain the results." | "Your estimate of the lift is smaller than the naive before/after difference. Which part of the naive difference was not caused by the campaign? Point to the control geos' line." |
| "Why is the interval wide?" | "Search has a narrow ROAS interval and display a wide one. Compare their spend plots above: which one varies more week to week, and why does that matter for estimating its effect?" |

## 5. Checkpoints: pass or fail, and say how to recover

A checkpoint is one `with workshop.checkpoint(N):` block (Python) or `checkpoint(N, {...})` (R) per
exercise, using `mktstats.checks` functions or plain asserts with messages (CONTRIBUTING.md).

**What it compares.** On synthetic data, the estimate against the truth in `data/synthetic/truth.json`,
with a **named interval**: type and probability, for example "94% HDI" or "90% equal-tailed interval",
passed explicitly, never a library default. On real data, structural facts (row counts, column names,
ranges, monotonicity, a known published value where one exists) rather than a guessed number.

**Tolerances are justified.** The brief says why a tolerance is what it is (for example, "the reference
recovers r within 15% at FULL across 20 seeds in the monthly coverage test"). A tolerance must pass on the
reference solution in both QUICK and FULL, or the check uses a stated mode-specific tolerance.

**It must be able to fail.** Every checkpoint fails on its stub (verify mode in `scripts/test_notebooks.py`)
and on a deliberately wrong input (its `mktstats.checks` test).

**Failure messages tell the learner what to check and how to recover.** A message has four parts:

1. **What failed, with the numbers.** "Checkpoint 3: the estimated r is 0.02; the truth is 0.24, outside
   your 94% HDI [0.01, 0.04]."
2. **The likely cause.** "This usually means frequency counts all purchases, including the first."
3. **What to check.** "Check that `frequency` is the number of repeat purchases and that `T` is in weeks,
   not days."
4. **How to recover.** "To move on now, run `workshop.use_reference(3)` and come back to this later."

Never "assert failed", a bare `AssertionError`, or a traceback as the only output. A message that needs a
statistician to decode is a bug. For R, the same four parts in the `stop()` message, with
`use_reference(3)`.

The harness already handles the unwritten case: a stub's `NotImplementedError` prints "TODO N is not
written yet …" with the `use_reference(N)` line. Checkpoints only need messages for wrong answers.

## 6. QUICK and FULL

- **FULL** is the teaching run on Colab: the sampling and data sizes the module page's numbers come from.
  Module pages quote numbers only from recorded FULL runs (AGENTS.md: everything claimed to run must have
  run).
- **QUICK** (`MKTSTATS_QUICK=1`, or the `QUICK` switch in the harness cell) shortens sampling and
  subsamples data. It is for CI and for a room that is out of time. It **never skips a checkpoint**.
- **Tell the learner what QUICK changes**, in the header and again beside the first fit: "With QUICK on,
  intervals are wider and may not match the module page; a decision near its threshold can flip. That is a
  lesson about sample size, not a bug."
- **The Decision cell works in both modes.** It prints the mode next to the recommendation ("QUICK run:
  treat this as a rough answer").
- **The brief states expected run time for both modes**, marked as an estimate until a run record exists.

## 7. The Decision cell: number → rule → recommendation with uncertainty

Every lab has a `## Decision · Title` section that turns the lab's result into the module's business
decision (`modules.mNN.decision` in `_variables.yml`). It has three parts, always in this order:

1. **The number**, with its named interval, computed in the notebook: for example, the number of customers
   with P(alive) below 0.2, with a 90% interval, or channel ROAS with a 94% HDI.
2. **The rule**, stated before the result is read, with the economics that justify it: "Stop retargeting a
   customer when expected margin from the next 26 weeks of purchases is below the $1.50 cost of retargeting
   them." Thresholds come from costs and margins, not from habit.
3. **The recommendation with its uncertainty**, as one sentence a manager could act on: what to do, how much
   money or how many customers it concerns, how sure we are, and what would change it.

Where the decision depends on a posterior, apply the rule to each posterior draw and report the probability
that the recommended action is the better one, beside the interval.

Template (the notebook fills the numbers by code; the learner writes the last sentence):

```text
Number          : 1,240 customers have P(alive) < 0.2 (90% interval 1,150 to 1,330), FULL run.
Rule            : stop retargeting when expected 26-week margin < $1.50 retargeting cost.
Recommendation  : Stop retargeting about 1,240 customers, saving about $1,860 a quarter
                  (range $1,725 to $1,995). In 97% of posterior draws this beats retargeting everyone.
What would change it: a retargeting cost below $0.60, or evidence that retargeting itself
                  brings customers back (an uplift test, Module 11).
Your sentence   : ________________________________________________
```

The numbers above illustrate the format only; they are not results.

The final "Your sentence" line is the learner's one decision claim. The day wrap-up collects it (see
`days/_day-N-notes.md`).

## 8. Stretch

- **Optional, after the Decision cell**, under `## Stretch (optional) · Title`.
- **Nothing later depends on it.** No later lab, knowledge check, warm-up or capstone step needs anything
  that only the stretch teaches.
- **Good stretch material:** a sensitivity analysis (discount rate, horizon, prior), the same analysis on
  the real dataset after the synthetic one, the alternative model the briefing mentioned, a cross-check
  of the Python result against the R reference, or a harder version of the decision (constraints, risk).
- **No new required concept, no new install that can break the notebook,** and no compute that pushes the
  notebook past its run-time target. A stretch may have a checkpoint; it is not required.

## 9. Language and cognitive load

- **Define every term the first time it appears in the notebook**, even if the briefing defined it: a
  learner may open the lab without the briefing fresh in mind. One plain sentence is enough: "P(alive) is the
  probability that the customer has not yet churned, given their purchase history."
- **Symbols** are defined beside the equation, and match the module page's notation.
- **No filler words** that make a stuck learner feel slow: "simply", "just", "obviously", "trivially".
- **Imperative instructions**: "Write `build_rfm(tx)` returning …", not "We might now want to consider …".
- **Consistent names across labs**: `tx` (transactions), `customers`, `rfm`, `idata` (posterior), `truth`.
  The brief lists the names the lab uses.
- **Visible seeds.** Every random step uses a seed set in a visible cell.
- **No unexplained constants.** A threshold, a prior scale or a tolerance has a one-line reason next to it.
- **Maintainer notes stay out of the learner's way.** Notes on which API versions were checked go in the
  jupytext header or an HTML comment, never as the first paragraph a learner reads.
- **Run times quoted to learners** come from a run record of the same content (say which runtime) or are
  labelled estimates. A Colab time comes only from a Colab record.

## 10. Plots

- **Every plot has a sentence before it saying what to look at**: "Look at whether the true ROAS (black tick)
  falls inside each channel's bar." The sentence names the feature, not the plot type.
- **One message per plot.** Axes labelled with units; intervals labelled with type and probability in the
  legend or title; truth marked distinctly when shown.
- **Readable on a laptop screen** and in both light and dark notebook themes; do not rely on color alone
  (use shape or a label as well).

## 11. Reviewer checklist

The Pedagogy Expert applies this to every brief and every notebook (`labs/src/` source and the generated
notebook). The Academic Director applies it to content; the Technical Expert to the build. Copy it into the
review comment and tick each line.

**Budget and structure**

- [ ] Stated exercise minutes add up to no more than the lab minutes − 15.
- [ ] Four to six core exercises, each 5 to 12 minutes, each with one task and one function.
- [ ] The first checkpoint comes before any model fit or long download.
- [ ] Two-notebook labs: each notebook opens with the same Order of work note (order as in `_variables.yml`, install to start first, minutes per notebook, switch minute).
- [ ] Each model is fitted once; compute per cell and in total is stated (estimate or measured, labelled).
- [ ] The lab ends with a Decision cell and then a stretch section; nothing later depends on the stretch.

**Prompts**

- [ ] Every Predict prompt asks for a concrete, checkable guess with a scale or reference value.
- [ ] Every Explain prompt sends the learner back to the prediction, asks for a mechanism and points at evidence.
- [ ] Every `<details>` explanation names the mechanism and what would change the result.

**Checkpoints**

- [ ] Each checkpoint compares against truth with a named interval (type and probability), or checks structural facts on real data.
- [ ] Each failure message says what failed (with numbers), the likely cause, what to check and how to recover (`use_reference(N)`).
- [ ] Each checkpoint fails on its stub, and passes on the reference in both QUICK and FULL.

**Decision**

- [ ] Number → rule → recommendation, in that order; the rule states its costs or margins.
- [ ] The recommendation states its uncertainty (interval and, where there is a posterior, the probability the action is right).
- [ ] The decision matches `modules.mNN.decision` in `_variables.yml`.

**Cognitive load and language**

- [ ] Every term is defined at first use in the notebook; every symbol is defined beside its equation.
- [ ] One new idea per exercise; scaffolding is in provided Run cells.
- [ ] No "simply", "just", "obviously"; instructions are imperative.
- [ ] Variable names follow the shared names; seeds are visible; constants have reasons.
- [ ] Every plot has a sentence before it saying what to look at; axes have units; intervals are named.
- [ ] QUICK's effect on the results is stated where it matters, and the Decision cell prints the mode.

**Honesty**

- [ ] Run times are labelled as measured (with the run record) or estimated, in Markdown cells as in the brief.
- [ ] Numbers quoted in Markdown cells come from a recorded run, or are marked as illustrations.
