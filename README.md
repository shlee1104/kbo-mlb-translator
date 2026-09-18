# KBO → MLB Translator

Estimating how a KBO player's production would look in Major League Baseball,
built on a data set that is reconciled across sources and audited before
anything is modelled on it.

---

## For a scout or coordinator (no code)

**The problem.** When a Korean player is posted, the first question is always
"what do these numbers mean over here?" A .340 hitter in the KBO is not a .340
hitter in MLB, and the gap is not a fixed discount: it depends on the stat, on
how the KBO's run environment happened to be behaving that year, and on the
player's age.

**What this does.**

1. Collects every KBO player-season since 2000, plus each season's league-wide
   context, from public sources.
2. Links those players to their MLB records, which is genuinely hard because
   the same person shows up as 이정후, Lee Jung-hoo, Jung Hoo Lee and
   Lee Jeong-hu depending on who is writing.
3. Checks the result against itself and against published league totals, so we
   know what we can trust before we use it.
4. Translates each statistic separately and reports a **range**, not a single
   number, because the honest answer to "what will he hit?" is a range.

**What it does not do.** It does not scout. It has nothing to say about a
swing, a delivery, or makeup. It takes performance that already happened in
one league and puts it on the scale of another.

---

## Why these three steps, in this order

The job this was built for describes three phases, and they are in that order
for a reason: a projection model built on data nobody has audited will be
confidently wrong, and nobody will be able to tell you why.

| Phase | In this repo |
|---|---|
| **1. Data audit & quality** | `validate.py` — 20+ checks; `reports/data_audit.md` |
| **2. Exploratory tools** | Streamlit app, filter by country, league, age, position |
| **3. Valuation & projection** | Per-statistic league translations with uncertainty |

---

## Quickstart

```bash
git clone <this repo> && cd kbo-mlb-translator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Crawl KBO pages. Slow on purpose (see "Crawling politely"), ~20 min
#    for 2000-2026. Every page is cached, so re-runs are free.
PYTHONPATH=src python -m kbo_mlb.cli scrape --start 2000 --end 2026

# 2. Link KBO players to their MLB records
PYTHONPATH=src python -m kbo_mlb.cli crosswalk

# 3. Run the data quality checks and write the audit report
PYTHONPATH=src python -m kbo_mlb.cli audit

# 4. How many players have both KBO and MLB records?
PYTHONPATH=src python -m kbo_mlb.cli counts

# ...or all four
PYTHONPATH=src python -m kbo_mlb.cli all
```

Tests need no network and no fixtures beyond the repo:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
# or: PYTHONPATH=src pytest tests -v
```

---

## Data sources

| Source | Used for | Terms |
|---|---|---|
| Baseball-Reference **register** | KBO player-seasons, rosters, league totals | `/register/` is permitted by their robots.txt at a 3-second crawl delay |
| Chadwick Bureau register | Player ID crosswalk (`key_bbref_minors` → MLBAM / FanGraphs) | Open data, CC-0 |
| FanGraphs / Statcast via `pybaseball` | MLB player-seasons | Standard public API use |
| KBO official site, Statiz | **Manual spot-checks only** | koreabaseball.com disallows automated crawling in robots.txt, so it is never scraped here |

### Crawling politely

Baseball-Reference publishes two limits, and this repo respects both at once:
a 3-second crawl delay (robots.txt) and a 20-requests-per-minute ceiling
(`/bot-traffic.html`, which warns that exceeding it gets you blocked for up to
a day). `http_client.py` enforces a 3.5-second gap **and** a rolling 16/minute
token bucket, backs off hard on HTTP 429, and caches every response on disk so
that iterating on the parser costs zero additional requests.

---

## The interesting problem: who is this player?

There is no shared ID between the KBO and MLB sides, and names do not line up.
The same player appears as any of:

```
이정후        Lee Jung-hoo        Jung Hoo Lee        Lee Jeong-hu
```

Three problems are stacked on top of each other:

1. **Script.** Korean sources use Hangul; English sources romanise.
2. **Name order.** Korean puts the surname first. MLB sources usually flip it
   — but not always, and not consistently within one source.
3. **Romanisation system.** Korea moved from McCune-Reischauer to Revised
   Romanization in 2000, players pick their own spellings regardless, and
   clubs are inconsistent. Park/Bak, Lee/Yi/Rhee, Choi/Choe and
   Jung/Jeong/Chung all appear in live data.

`names.py` handles all three: a Hangul → Revised Romanization transliterator
built on Unicode syllable arithmetic, an explicit surname equivalence table
(Lee/Yi/Rhee/Ri are one class), and two matching keys — a strict one and an
aggressively phonetic one.

The subtle trap is that many Korean **given-name syllables are also surnames**.
In "Jung Ho Kang", *Jung* is a real surname but here it is half the given name;
the surname is Kang. The resolver uses the hyphen group as the strongest
signal, then weighs how strongly each token is a surname (Kim essentially
always is, Jung often isn't), rather than trusting position.

### Matching is tiered, and the tier is recorded

| Tier | Basis | Confidence | Auto-accepted |
|---|---|---|---|
| `register_id` | Chadwick ID crosswalk | 1.00 | yes |
| `name_dob` | Strict name key + exact date of birth | 0.95 | yes |
| `phonetic_dob` | Phonetic key + exact date of birth | 0.90 | yes |
| `phonetic_year` | Phonetic key + birth year only | 0.60 | **no — review queue** |

Two rules keep this honest. Phonetic keys are deliberately loose, so every
name-based match must agree on **date of birth**. And any key matching more
than one row on either side is treated as ambiguous and sent to review rather
than guessed at.

`data/processed/crosswalk_review_queue.csv` is a deliverable, not a failure.
A few dozen names resolved by hand, fed back through
`crosswalk.apply_manual_overrides`, are worth more than a silently wrong join.

---

## What the audit checks

`reports/data_audit.md` is regenerated on every run. Checks are graded
`error` (do not model on this) or `warning` (worth a look), and a run always
completes so you see every problem at once rather than the first one.

- **Structural** — required columns, duplicate player-team-seasons (a midseason
  trade is legitimate and is not flagged), missing register IDs
- **Ranges** — batting average outside [0, 1], negative counting stats,
  impossible ages
- **Internal consistency** — `PA ≥ AB`, `H ≥ 2B+3B+HR`, and
  `TB = H + 2B + 2·3B + 3·HR`. These catch column-misalignment bugs that range
  checks miss entirely, because a shifted column usually still holds
  plausible-looking numbers.
- **Cross-source reconciliation** — the strongest check: do the player rows we
  scraped from ~10 team pages add up to the league total the source publishes
  independently? A gap means we dropped a team, double-counted, or mis-parsed.
- **Coverage** — which requested seasons returned no rows at all
- **Crosswalk quality** — match rate, and the breakdown by tier

---

## Known limitations

Stated up front, because they bound what the output is worth:

1. **Small sample.** Only a few dozen players have meaningful playing time in
   both leagues. Every estimate carries wide uncertainty, which is why the
   output is a range.
2. **The players who move are not random.** Only the best KBO performers get
   MLB opportunities, and only MLB players who are struggling tend to go the
   other way. This selection cuts both ways and biases a naive translation.
   Including both directions of movement helps; it does not eliminate the
   problem.
3. **The league changed underneath the data.** The KBO altered its ball
   around 2019 and introduced an automated strike zone in 2024 — both dates
   worth re-verifying before relying on them. Comparing each player to his own
   league-year absorbs most of this, but not all.
4. **Park effects are not yet separated** from the league adjustment.
5. **Statistics translate at different rates.** Strikeout and walk rates carry
   across far more reliably than batting average on balls in play. A single
   blanket "KBO discount" is the wrong model and this repo does not use one.

---

## Layout

```
src/kbo_mlb/
  config.py        paths, source URLs, crawl limits — all tunables live here
  http_client.py   rate-limited, disk-cached fetcher
  parse_bref.py    register-page parsers (handles comment-wrapped tables)
  names.py         Hangul romanisation, surname classes, matching keys
  scrape_kbo.py    season → team → player crawl
  mlb_data.py      Chadwick register + pybaseball pulls
  crosswalk.py     tiered player matching + review queue
  validate.py      data quality checks and the audit report
  cli.py           command-line entry point
tests/             55 tests, no network required
reports/           generated audit report
```

---

## Status

Phase 1 (collection, crosswalk, audit) is implemented and tested. The
translation model and the Streamlit app are next.
