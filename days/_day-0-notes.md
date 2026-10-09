<!--
Partial, included by the UI Expert's Before Day 1 / Day 0 page. Owner: Pedagogy Expert.
Links are relative to the site root (the including page sits there). Minutes come from
_variables.yml through var shortcodes; install times are deliberately not repeated here:
the Setup page shows measured ones from run records.
-->

## Pre-work: what to do, and when {#prework}

Pre-work takes about an hour and a half if the entry check goes well, and longer if it shows an area to review. Start a week before Day 1, so there is time for the review.

| When | Step | Your time |
|---|---|---|
| A week before | [Take the entry check](prepare.qmd#entry-check): 15 questions, no web search | 15 minutes |
| A week before | [Read your score](prepare.qmd#score) and do the review for any area where you missed two or more | 0 to about 2.5 hours per area |
| A few days before | [Set up Colab](prepare.qmd#setup): open the Module 0 notebook and choose Run all | a few minutes, plus the install (measured times are on the [Setup](setup.qmd) page) |
| A few days before | [Module 0](modules/00-prework.qmd): the transaction-log warm-up in SQL and pandas or dplyr | about {{< var modules.m00.minutes.lab >}} minutes |
| The day before | [Check that your setup works](prepare.qmd#working); send any error you cannot fix to the organizers | 5 minutes |

**Why it matters.** Day 1 starts with a transaction log and turns it into a customer table within the first hour. Joins, group-bys, window functions, Poisson counts and the reading of an interval are used from the first lab onward and are not taught in the room. The entry check tells you which of these to brush up; Module 0 has you practise the first two on data like Day 1's.

**If you get stuck.** Read the [Setup](setup.qmd) page and the [FAQ](faq.qmd) first. If a cell still fails, copy the full error message and send it to the organizers before Day 1, not on the morning itself.

**On Day 1** the warm-up asks four short questions about the pre-work: how a checkpoint tells you what to do next, a window function, what a customer's silence means, and the left-join trap from the entry check.
