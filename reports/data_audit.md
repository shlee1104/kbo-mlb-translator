# KBO data audit

- checks run: **24**
- passed: **23**
- errors: **0**
- warnings: **1**

## Warnings

- **cross_source.league_totals** — 8 of 27 seasons are short of the published HR total (source lists fewer players than it counts; worst season captures 95.3%). Use the published league totals as the model baseline, not the player sums.

  <details><summary>sample rows</summary>

  |   season |   scraped |   published |   pct_diff |   coverage |
  |---------:|----------:|------------:|-----------:|-----------:|
  |     2001 |      1022 |        1070 | -0.0448598 |   0.95514  |
  |     2006 |       643 |         660 | -0.0257576 |   0.974242 |
  |     2009 |      1117 |        1155 | -0.0329004 |   0.9671   |
  |     2015 |      1464 |        1511 | -0.0311052 |   0.968895 |
  |     2017 |      1496 |        1547 | -0.032967  |   0.967033 |

  </details>

## Passed

- **batting.required_columns** — all required columns present
- **pitching.required_columns** — all required columns present
- **batting.duplicate_player_seasons** — no duplicate player-team-seasons
- **batting.missing_ids** — every row has a register id
- **batting.negative_counts** — no negative counting stats
- **batting.range.batting_avg** — batting_avg within [0.0, 1.0]
- **batting.range.onbase_perc** — onbase_perc within [0.0, 1.0]
- **batting.range.slugging_perc** — slugging_perc within [0.0, 4.0]
- **batting.range.age** — age within [15, 55]
- **batting.range.G** — G within [0, 200]
- **pitching.duplicate_player_seasons** — no duplicate player-team-seasons
- **pitching.missing_ids** — every row has a register id
- **pitching.negative_counts** — no negative counting stats
- **pitching.range.age** — age within [15, 55]
- **pitching.range.earned_run_avg** — earned_run_avg within [0.0, 200.0]
- **pitching.range.G** — G within [0, 200]
- **batting.pa_ge_ab** — PA >= AB holds
- **batting.hits_ge_extra_base** — H >= 2B+3B+HR holds
- **batting.total_bases_identity** — TB matches H+2B+2*3B+3*HR
- **coverage.seasons** — all 27 seasons present
- **roster.date_of_birth_coverage** — 100.0% of roster rows have a usable date of birth
- **crosswalk.size** — 476 players linked (17.7% of all 2693 KBO players; most never played MLB, so a low share is expected)
- **crosswalk.uniqueness** — each KBO player links to at most one MLB record
