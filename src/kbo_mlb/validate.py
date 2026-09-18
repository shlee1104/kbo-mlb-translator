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

    comparison["abs_pct_diff"] = (
        (comparison["scraped"] - comparison["published"]).abs()
        / comparison["published"].replace(0, pd.NA)
    )
    bad = comparison[comparison["abs_pct_diff"] > tolerance_pct]
    return _result(
        "cross_source.league_totals",
        bad.empty,
        "error",
        f"season {column} totals reconcile within "
        f"{tolerance_pct:.0%}" if bad.empty
        else f"{len(bad)} seasons where scraped {column} differs from "
             f"published by more than {tolerance_pct:.0%}",
        bad.reset_index(),
        column=column,
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


def check_crosswalk_quality(stats: dict,
                            min_match_rate: float = 0.5) -> CheckResult:
    rate = stats.get("match_rate", 0.0)
    return _result(
        "crosswalk.match_rate",
        rate >= min_match_rate,
        "warning",
        f"crosswalk matched {rate:.1%} of KBO players"
        + ("" if rate >= min_match_rate
           else f" (below the {min_match_rate:.0%} floor)"),
        None,
        **stats,
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
