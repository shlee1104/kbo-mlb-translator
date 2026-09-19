"""Data quality and validation checks.

This is the Phase 1 deliverable in code form. Every check returns a
`CheckResult` rather than raising, so a run always produces a full report
instead of stopping at the first problem. Severity separates "this number is
impossible, do not model on it" from "worth a look".

Adding a check is deliberately cheap: write a function that takes the frames
it needs and returns CheckResults, then register it in `run_all`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str           # "error" | "warning" | "info"
    message: str
    offending_rows: pd.DataFrame | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # concise console output
        flag = "PASS" if self.passed else self.severity.upper()
        return f"[{flag}] {self.name}: {self.message}"


def _result(name, passed, severity, message, rows=None, **details):
    return CheckResult(name, passed, severity, message, rows, details)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame, required: list[str],
                           label: str) -> CheckResult:
    missing = [c for c in required if c not in df.columns]
    return _result(
        f"{label}.required_columns",
        not missing,
        "error",
        "all required columns present" if not missing
        else f"missing columns: {missing}",
        missing=missing,
    )


def check_duplicate_player_seasons(df: pd.DataFrame, label: str) -> CheckResult:
    """A player should appear once per team-season, not twice.

    Mid-season trades legitimately produce two rows for one player in one
    season, on different teams. Two rows for the *same* team-season is a
    parsing bug.
    """
    keys = ["player_register_id", "season", "team_bref_id"]
    if not all(k in df.columns for k in keys):
        return _result(f"{label}.duplicate_player_seasons", False, "error",
                       "key columns unavailable")
    dupes = df[df.duplicated(subset=keys, keep=False)]
    return _result(
        f"{label}.duplicate_player_seasons",
        dupes.empty,
        "error",
        "no duplicate player-team-seasons" if dupes.empty
        else f"{len(dupes)} duplicated player-team-season rows",
        dupes,
        duplicate_count=len(dupes),
    )


def check_missing_ids(df: pd.DataFrame, label: str) -> CheckResult:
    if "player_register_id" not in df.columns:
        return _result(f"{label}.missing_ids", False, "error",
                       "player_register_id column absent")
    missing = df[df["player_register_id"].isna()
                 | (df["player_register_id"].astype(str).str.strip() == "")]
    return _result(
        f"{label}.missing_ids",
        missing.empty,
        "error",
        "every row has a register id" if missing.empty
        else f"{len(missing)} rows without a register id",
        missing,
    )


# ---------------------------------------------------------------------------
# Value-range checks
# ---------------------------------------------------------------------------

# (column, low, high, inclusive) - bounds that are physically impossible to
# cross, not merely unusual.
_RANGES = [
    ("batting_avg", 0.0, 1.0),
    ("onbase_perc", 0.0, 1.0),
    ("slugging_perc", 0.0, 4.0),
    ("fielding_perc", 0.0, 1.0),
    ("age", 15, 55),
    ("earned_run_avg", 0.0, 200.0),
    ("G", 0, 200),
]


def check_value_ranges(df: pd.DataFrame, label: str) -> list[CheckResult]:
    out = []
    for col, low, high in _RANGES:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad = df[series.notna() & ((series < low) | (series > high))]
        out.append(_result(
            f"{label}.range.{col}",
            bad.empty,
            "error",
            f"{col} within [{low}, {high}]" if bad.empty
            else f"{len(bad)} rows where {col} is outside [{low}, {high}]",
            bad,
        ))
    return out


def check_negative_counts(df: pd.DataFrame, label: str) -> CheckResult:
    """Counting stats can never be negative."""
    count_cols = [c for c in ("G", "PA", "AB", "R", "H", "HR", "RBI", "BB",
                              "SO", "SB", "CS", "W", "L", "SV", "ER")
                  if c in df.columns]
    if not count_cols:
        return _result(f"{label}.negative_counts", True, "info",
                       "no counting columns to check")
    numeric = df[count_cols].apply(pd.to_numeric, errors="coerce")
    bad = df[(numeric < 0).any(axis=1)]
    return _result(
        f"{label}.negative_counts",
        bad.empty,
        "error",
        "no negative counting stats" if bad.empty
        else f"{len(bad)} rows with a negative counting stat",
        bad,
    )


# ---------------------------------------------------------------------------
# Internal-consistency checks
# ---------------------------------------------------------------------------

def check_batting_identities(df: pd.DataFrame, tolerance: int = 1
                             ) -> list[CheckResult]:
    """Box-score arithmetic that must hold within rounding.

    PA >= AB, H >= 2B + 3B + HR, and TB = H + 2B + 2*3B + 3*HR. These catch
    column-misalignment bugs that range checks sail straight past, because a
    shifted column usually still contains plausible-looking numbers.
    """
    out: list[CheckResult] = []
    have = set(df.columns)

    if {"PA", "AB"} <= have:
        pa = pd.to_numeric(df["PA"], errors="coerce")
        ab = pd.to_numeric(df["AB"], errors="coerce")
        bad = df[(pa.notna() & ab.notna()) & (ab > pa)]
        out.append(_result(
            "batting.pa_ge_ab", bad.empty, "error",
            "PA >= AB holds" if bad.empty
            else f"{len(bad)} rows where AB exceeds PA", bad))

    if {"H", "2B", "3B", "HR"} <= have:
        h, d, t, hr = (pd.to_numeric(df[c], errors="coerce")
                       for c in ("H", "2B", "3B", "HR"))
        extra = d + t + hr
        bad = df[(h.notna() & extra.notna()) & (extra > h)]
        out.append(_result(
            "batting.hits_ge_extra_base", bad.empty, "error",
            "H >= 2B+3B+HR holds" if bad.empty
            else f"{len(bad)} rows where extra-base hits exceed hits", bad))

    if {"TB", "H", "2B", "3B", "HR"} <= have:
        tb = pd.to_numeric(df["TB"], errors="coerce")
        h, d, t, hr = (pd.to_numeric(df[c], errors="coerce")
                       for c in ("H", "2B", "3B", "HR"))
        expected = h + d + 2 * t + 3 * hr
        diff = (tb - expected).abs()
        bad = df[diff.notna() & (diff > tolerance)]
        out.append(_result(
            "batting.total_bases_identity", bad.empty, "error",
            "TB matches H+2B+2*3B+3*HR" if bad.empty
            else f"{len(bad)} rows where total bases do not reconcile", bad))

    return out


def check_league_totals_reconcile(
    players: pd.DataFrame,
    league_totals: pd.DataFrame,
    column: str = "HR",
    tolerance_pct: float = 0.02,
) -> CheckResult:
    """Do the player rows add up to the published league total?

    This is the strongest single check in the suite: it compares a number we
    built by scraping ~10 team pages against a number the source publishes
    independently. A gap means we dropped players, double-counted a team, or
    mis-parsed a column.
    """
    if column not in players.columns or "season" not in players.columns:
        return _result("cross_source.league_totals", False, "error",
                       f"cannot reconcile: {column} or season missing")
    if league_totals.empty or column not in league_totals.columns:
        return _result("cross_source.league_totals", False, "warning",
                       "no league totals available to reconcile against")

    summed = (players.assign(**{column: pd.to_numeric(players[column],
                                                      errors="coerce")})
              .groupby("season")[column].sum())
    published = (league_totals.assign(
        **{column: pd.to_numeric(league_totals[column], errors="coerce")})
        .groupby("season")[column].sum())

    comparison = pd.DataFrame({"scraped": summed,
                               "published": published}).dropna()
    if comparison.empty:
        return _result("cross_source.league_totals", False, "warning",
                       "no overlapping seasons to reconcile")

    comparison["pct_diff"] = (
        (comparison["scraped"] - comparison["published"])
        / comparison["published"].replace(0, pd.NA)
    )
    comparison["coverage"] = (
        comparison["scraped"] / comparison["published"].replace(0, pd.NA)
    )

    # The direction of the gap tells you whose problem it is.
    #
    # scraped > published  -> we counted someone twice, or mis-parsed a row.
    #                         That is our bug, and it is an error.
    # scraped < published  -> the source's own player list is shorter than
    #                         the total it publishes. We verified this against
    #                         Baseball-Reference's register directly: a team
    #                         page's tfoot total can exceed the sum of the
    #                         players that same page lists, so marginal
    #                         players are simply absent. That is a coverage
    #                         limit of the source, not a parsing failure, so
    #                         it is a warning - but it must stay visible,
    #                         because it bounds what the data can support.
    over = comparison[comparison["pct_diff"] > tolerance_pct]
    under = comparison[comparison["pct_diff"] < -tolerance_pct]

    if not over.empty:
        return _result(
            "cross_source.league_totals", False, "error",
            f"{len(over)} seasons where scraped {column} EXCEEDS the "
            f"published total - likely double counting",
            over.reset_index(), column=column,
            worst_coverage=float(comparison["coverage"].min()),
        )

    if not under.empty:
        worst = comparison["coverage"].min()
        return _result(
            "cross_source.league_totals", False, "warning",
            f"{len(under)} of {len(comparison)} seasons are short of the "
            f"published {column} total (source lists fewer players than it "
            f"counts; worst season captures {worst:.1%}). Use the published "
            f"league totals as the model baseline, not the player sums.",
            under.reset_index(), column=column,
            worst_coverage=float(worst),
            mean_coverage=float(comparison["coverage"].mean()),
        )

    return _result(
        "cross_source.league_totals", True, "error",
        f"season {column} totals reconcile within {tolerance_pct:.0%}",
        None, column=column,
        mean_coverage=float(comparison["coverage"].mean()),
    )


def check_league_sides_agree(league_totals: pd.DataFrame,
                             tolerance: float = 0.005) -> CheckResult:
    """League strikeout rate must be the same from both sides of the ball.

    Every strikeout is recorded once by a batter and once by a pitcher, so
    SO/PA from the league batting table and SO/BF from the league pitching
    table describe the same events. They are published as two separate
    tables, which makes this a genuine cross-source check: if a column is
    mis-parsed in one of them, the two stop agreeing.
    """
    from . import rates  # local import keeps validate importable standalone

    if league_totals.empty or "side" not in league_totals.columns:
        return _result("cross_source.league_sides", False, "warning",
                       "league totals unavailable for the two-sided check")

    bat = rates.league_rates(league_totals, "batting")
    pit = rates.league_rates(league_totals, "pitching")
    if bat.empty or pit.empty:
        return _result("cross_source.league_sides", False, "warning",
                       "one side of the league totals is missing")

    merged = bat[["season", "lg_k_pct"]].merge(
        pit[["season", "lg_k_pct"]], on="season", suffixes=("_bat", "_pit")
    ).dropna()
    merged["diff"] = (merged["lg_k_pct_bat"] - merged["lg_k_pct_pit"]).abs()
    bad = merged[merged["diff"] > tolerance]
    worst = float(merged["diff"].max()) if not merged.empty else float("nan")

    return _result(
        "cross_source.league_sides",
        bad.empty,
        "error",
        f"league K% agrees between the batting and pitching tables "
        f"(worst season differs by {worst:.4f})" if bad.empty
        else f"{len(bad)} seasons where league K% differs between the "
             f"batting and pitching tables by more than {tolerance}",
        bad.reset_index(drop=True) if not bad.empty else None,
        worst_abs_diff=worst,
    )


def check_season_coverage(df: pd.DataFrame, expected_start: int,
                          expected_end: int) -> CheckResult:
    """Which seasons in the requested window produced no rows at all?"""
    if "season" not in df.columns:
        return _result("coverage.seasons", False, "error",
                       "season column absent")
    present = set(pd.to_numeric(df["season"], errors="coerce").dropna()
                  .astype(int))
    expected = set(range(expected_start, expected_end + 1))
    missing = sorted(expected - present)
    return _result(
        "coverage.seasons",
        not missing,
        "warning",
        f"all {len(expected)} seasons present" if not missing
        else f"{len(missing)} seasons returned no rows: {missing}",
        None,
        missing_seasons=missing,
    )


def check_crosswalk_quality(stats: dict, min_matches: int = 50) -> CheckResult:
    """Sanity-check the crosswalk size.

    Note on the denominator: a raw "match rate" over every KBO player is not
    a quality measure, because the large majority of KBO players never played
    in MLB and *should* not match. A low rate here is expected. What would be
    alarming is a rate near zero (the join broke) or one near 100% (the join
    is matching things it shouldn't).
    """
    matched = int(stats.get("matched", 0))
    total = int(stats.get("kbo_players", 0)) or 1
    rate = matched / total

    if matched < min_matches:
        return _result(
            "crosswalk.size", False, "error",
            f"only {matched} players linked - the join is probably broken",
            None, **stats)

    if rate > 0.75:
        return _result(
            "crosswalk.size", False, "error",
            f"{rate:.1%} of KBO players linked to MLB records, which is "
            f"implausibly high - check for a cross join on null keys",
            None, **stats)

    return _result(
        "crosswalk.size", True, "info",
        f"{matched} players linked ({rate:.1%} of all {total} KBO players; "
        f"most never played MLB, so a low share is expected)",
        None, **stats)


def check_crosswalk_uniqueness(matches: pd.DataFrame) -> CheckResult:
    """No KBO player may be linked to more than one MLB record.

    This is the check that would have caught the null-key cross join, which
    silently turned 476 genuine links into 200,180 rows.
    """
    if matches.empty or "player_register_id" not in matches.columns:
        return _result("crosswalk.uniqueness", False, "warning",
                       "no crosswalk available to check")
    counts = matches["player_register_id"].value_counts()
    dupes = counts[counts > 1]
    return _result(
        "crosswalk.uniqueness",
        dupes.empty,
        "error",
        "each KBO player links to at most one MLB record" if dupes.empty
        else f"{len(dupes)} KBO players link to multiple MLB records "
             f"(worst: {int(dupes.iloc[0])} links)",
        dupes.reset_index().head(10) if not dupes.empty else None,
    )


# ---------------------------------------------------------------------------
# Runner and report
# ---------------------------------------------------------------------------

def run_all(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    roster: pd.DataFrame,
    league_totals: pd.DataFrame,
    start_season: int,
    end_season: int,
    crosswalk_stats: dict | None = None,
    crosswalk_matches: pd.DataFrame | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(check_required_columns(
        batting, ["player", "season", "PA", "player_register_id"], "batting"))
    results.append(check_required_columns(
        pitching, ["player", "season", "IP", "player_register_id"], "pitching"))

    for df, label in ((batting, "batting"), (pitching, "pitching")):
        if df.empty:
            results.append(_result(f"{label}.non_empty", False, "error",
                                   "table is empty"))
            continue
        results.append(check_duplicate_player_seasons(df, label))
        results.append(check_missing_ids(df, label))
        results.append(check_negative_counts(df, label))
        results.extend(check_value_ranges(df, label))

    if not batting.empty:
        results.extend(check_batting_identities(batting))
        results.append(check_league_totals_reconcile(
            batting, league_totals[league_totals.get("side") == "batting"]
            if "side" in league_totals.columns else league_totals))
        results.append(check_season_coverage(batting, start_season, end_season))

    if not league_totals.empty:
        results.append(check_league_sides_agree(league_totals))

    if not roster.empty and "date_of_birth" in roster.columns:
        dob = pd.to_datetime(roster["date_of_birth"], errors="coerce")
        pct = dob.isna().mean()
        results.append(_result(
            "roster.date_of_birth_coverage",
            pct < 0.10, "warning",
            f"{1 - pct:.1%} of roster rows have a usable date of birth",
            None, missing_pct=round(float(pct), 4)))

    if crosswalk_stats:
        results.append(check_crosswalk_quality(crosswalk_stats))
    if crosswalk_matches is not None:
        results.append(check_crosswalk_uniqueness(crosswalk_matches))

    return results


def to_markdown(results: list[CheckResult]) -> str:
    """Render results as the data-audit report."""
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    passed = [r for r in results if r.passed]

    lines = [
        "# KBO data audit",
        "",
        f"- checks run: **{len(results)}**",
        f"- passed: **{len(passed)}**",
        f"- errors: **{len(errors)}**",
        f"- warnings: **{len(warnings)}**",
        "",
    ]

    for title, group in (("Errors", errors), ("Warnings", warnings),
                         ("Passed", passed)):
        if not group:
            continue
        lines += [f"## {title}", ""]
        for r in group:
            lines.append(f"- **{r.name}** — {r.message}")
            if r.offending_rows is not None and not r.offending_rows.empty:
                lines.append("")
                lines.append("  <details><summary>sample rows</summary>")
                lines.append("")
                sample = r.offending_rows.head(5).to_markdown(index=False)
                lines += ["  " + ln for ln in sample.splitlines()]
                lines += ["", "  </details>"]
        lines.append("")

    return "\n".join(lines)


def has_errors(results: list[CheckResult]) -> bool:
    return any(not r.passed and r.severity == "error" for r in results)
