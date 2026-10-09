# Statistics for Marketing

A five-day hands-on workshop on statistics for sales growth and customer lifetime value, with
labs in Python and R that open in Colab.

- Site: https://project-delphi.github.io/marketing-statistics-workshop/
- How the repository fits together: [AGENTS.md](AGENTS.md), [PLAN.md](PLAN.md),
  [CONTRIBUTING.md](CONTRIBUTING.md), [DECISIONS.md](DECISIONS.md)

The `mktstats` Python package in `src/` holds the helpers the labs use (runtime checks, the
exercise harness, checks, data loaders and synthetic data). The labs install it from this
repository at the workshop's pinned ref, so this file must exist for the package to build.

## Status

<!-- BEGIN status -->
> **As of 2026-10-09: not yet ready to teach.** 2 of 17 notebooks have a current, passing, worked, full-settings run of the whole notebook on Colab (15 are not written yet). 1 have passed on another machine (1 with full settings), and 0 on CI. [What has run, where](https://project-delphi.github.io/marketing-statistics-workshop/readiness.html).
<!-- END status -->

## Licences

Code (the `mktstats` package, R helpers, scripts): [MIT](LICENSE). Teaching content (pages,
slides, figures, notebook text): [CC BY 4.0](LICENSE-content).
