# Fonts

Two variable fonts are vendored here so the site does not depend on a font CDN. Both are
latin subsets from [Fontsource](https://fontsource.org/), copied unchanged from the
project-delphi/nlp-llms workshop site (same owner), which vendored them the same way.

| File | Family | Licence | Licence text |
|---|---|---|---|
| `inter-latin-wght-normal.woff2` | [Inter](https://github.com/rsms/inter) | SIL Open Font License 1.1 | [`OFL-Inter.txt`](OFL-Inter.txt) |
| `source-serif-4-latin-wght-normal.woff2` | [Source Serif 4](https://github.com/adobe-fonts/source-serif) | SIL Open Font License 1.1 | [`OFL-SourceSerif4.md`](OFL-SourceSerif4.md) |

The licence files are the upstream `LICENSE.txt` of rsms/inter (branch `master`) and
`LICENSE.md` of adobe-fonts/source-serif (branch `release`), downloaded 2026-10-09. The
OFL permits redistribution with the font when the copyright notice and licence travel
with it; `_quarto.yml` publishes this whole folder (`resources: /fonts/**`).

The `@font-face` rules are in `fonts.css`, which `_quarto.yml` (site) and
`slides/welcome.qmd` (deck) load as a plain stylesheet. The SCSS files name the families
with system fallbacks, so the site still reads correctly if the fonts fail to load.
