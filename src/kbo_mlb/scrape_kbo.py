"""Crawl KBO season -> team -> player stats from the Baseball-Reference register.

Crawl shape (three levels, all cached on disk):

    league index          1 request
      season pages        1 per season
        team pages        ~10 per season

For 2000-2026 that is roughly 280 requests, about 17 minutes at the polite
rate. Re-runs cost nothing because every page is cached.

Outputs three tidy tables into data/interim/:
    kbo_batting.csv      one row per player-season
    kbo_pitching.csv     one row per player-season
    kbo_league_totals.csv one row per season and side of the ball
    kbo_roster.csv       one row per player-team-season (DOB, birthplace)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import config, names, parse_bref
from .http_client import CachedFetcher

log = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    batting: pd.DataFrame
    pitching: pd.DataFrame
    roster: pd.DataFrame
    league_totals: pd.DataFrame


def _url(path_with_id: str) -> str:
    return f"{config.BREF_BASE}/register/{path_with_id}"


def crawl(
    fetcher: CachedFetcher,
    start_season: int = config.DEFAULT_START_SEASON,
    end_season: int = config.DEFAULT_END_SEASON,
) -> CrawlResult:
    index_html = fetcher.get(config.KBO_LEAGUE_INDEX)
    seasons = parse_bref.parse_season_index(index_html)
    seasons = [
        s for s in seasons
        if start_season <= int(s["season"]) <= end_season
    ]
    seasons.sort(key=lambda s: int(s["season"]))
    log.info("Found %d KBO seasons in range %d-%d",
             len(seasons), start_season, end_season)

    batting_rows: list[dict] = []
    pitching_rows: list[dict] = []
    roster_rows: list[dict] = []
    league_rows: list[dict] = []

    for season_rec in seasons:
        season = int(season_rec["season"])
        season_html = fetcher.get(
            _url(f"league.cgi?id={season_rec['season_bref_id']}")
        )
        parsed = parse_bref.parse_season_page(season_html)

        # League context: the "League Totals" row on each side of the ball.
        for side in ("batting", "pitching"):
            for row in parsed[f"league_{side}"]:
                if row.get("is_aggregate") == "True":
                    league_rows.append({"season": season, "side": side, **row})

        log.info("season %d: %d teams", season, len(parsed["teams"]))

        for team in parsed["teams"]:
            team_html = fetcher.get(_url(f"team.cgi?id={team['team_bref_id']}"))
            tp = parse_bref.parse_team_page(team_html)
            base = {
                "season": season,
                "team_name": team["team_name"],
                "team_bref_id": team["team_bref_id"],
            }
            for row in tp["batting"]:
                if row.get("is_aggregate") == "True":
                    continue
                batting_rows.append({**base, **row})
            for row in tp["pitching"]:
                if row.get("is_aggregate") == "True":
                    continue
                pitching_rows.append({**base, **row})
            for row in tp["roster"]:
                roster_rows.append({**base, **row})

    return CrawlResult(
        batting=_finalise(pd.DataFrame(batting_rows)),
        pitching=_finalise(pd.DataFrame(pitching_rows)),
        roster=_finalise(pd.DataFrame(roster_rows)),
        league_totals=pd.DataFrame(league_rows),
    )


# Columns that must be numeric for anything downstream to work.
_NUMERIC = [
    "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB",
    "SO", "batting_avg", "onbase_perc", "slugging_perc",
    "onbase_plus_slugging", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "age",
    "W", "L", "earned_run_avg", "GS", "GF", "CG", "SHO", "SV", "IP", "ER",
    "BK", "WP", "batters_faced", "whip", "win_loss_perc",
]


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Add name keys, coerce numerics, and drop rows with no player ID."""
    if df.empty:
        return df

    if "player" in df.columns:
        keys = df["player"].fillna("").map(names.name_keys).apply(pd.Series)
        df = pd.concat([df, keys], axis=1)

    for col in _NUMERIC:
        if col in df.columns:
            series = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            # Baseball-Reference prints "inf" for a rate stat with a zero
            # denominator - a pitcher who gave up earned runs without
            # recording an out has an undefined ERA, not an enormous one.
            # An undefined rate is missing data, so store it as such rather
            # than letting infinity poison every downstream mean.
            df[col] = series.replace([float("inf"), float("-inf")], pd.NA)

    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(
            df["date_of_birth"], errors="coerce"
        )

    if "player_register_id" in df.columns:
        df = df[df["player_register_id"].notna()].copy()

    return df.reset_index(drop=True)


def innings_to_outs(ip_value) -> float | None:
    """Convert baseball innings notation to outs.

    Innings pitched are written in thirds: 5.1 means five and one third, not
    five and one tenth. Averaging the raw decimal is a classic silent error,
    so everything downstream works in outs.

    >>> innings_to_outs(5.1), innings_to_outs(5.2), innings_to_outs(6.0)
    (16.0, 17.0, 18.0)
    """
    if ip_value is None:
        return None
    try:
        value = float(ip_value)
    except (TypeError, ValueError):
        return None
    whole = int(value)
    fraction = round((value - whole) * 10)
    if fraction not in (0, 1, 2):
        # Not thirds notation - treat as a genuine decimal.
        return round(value * 3, 2)
    return float(whole * 3 + fraction)
