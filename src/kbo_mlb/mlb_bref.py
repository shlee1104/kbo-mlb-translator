"""MLB season stats from Baseball-Reference, for the players we matched.

Why not FanGraphs: `pybaseball` reaches FanGraphs through a legacy leaders
endpoint that now answers 403. Rather than depend on a scraped endpoint
somebody else maintains, we pull the MLB side from the same provider as the
KBO side, through the same rate-limited cached fetcher. Using one provider
for both leagues also removes a class of definitional mismatch: the same
people decided what counts as a plate appearance on both sides.

We only need the 476 players the crosswalk matched, not all of MLB, so this
is a few hundred pages rather than tens of thousands.

Two page types:
  /players/<initial>/<bbref_id>.shtml   one player's MLB career, by season
  /leagues/majors/<year>-standard-*.shtml   league totals for that season

Both are permitted by Baseball-Reference's robots.txt (only the `split.cgi`
paths under them are disallowed), at the same 3-second crawl delay.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from . import config, parse_bref
from .http_client import CachedFetcher

log = logging.getLogger(__name__)

YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# Baseball-Reference renamed its player-page data-stat keys at some point
# (PA -> b_pa, IP -> p_ip, ...) and different pages still use different
# vintages. Normalising both spellings onto one canonical name means the
# parser does not care which it meets.
_CANONICAL = {
    # identity
    "year_id": "season", "year_ID": "season",
    "team_name_abbr": "team", "team_ID": "team", "team_id": "team",
    "comp_name_abbr": "league", "lg_ID": "league", "comp_id": "league",
    "age": "age", "b_age": "age", "p_age": "age",
    # batting
    "b_games": "G", "G": "G",
    "b_pa": "PA", "PA": "PA",
    "b_ab": "AB", "AB": "AB",
    "b_r": "R", "R": "R",
    "b_h": "H", "H": "H",
    "b_doubles": "2B", "2B": "2B",
    "b_triples": "3B", "3B": "3B",
    "b_hr": "HR", "HR": "HR",
    "b_rbi": "RBI", "RBI": "RBI",
    "b_bb": "BB", "BB": "BB",
    "b_so": "SO", "SO": "SO",
    "b_tb": "TB", "TB": "TB",
    "b_hbp": "HBP", "HBP": "HBP",
    "b_sf": "SF", "SF": "SF",
    "b_sh": "SH", "SH": "SH",
    "b_batting_avg": "batting_avg", "batting_avg": "batting_avg",
    "b_onbase_perc": "onbase_perc", "onbase_perc": "onbase_perc",
    "b_slugging_perc": "slugging_perc", "slugging_perc": "slugging_perc",
    # pitching
    "p_w": "W", "W": "W",
    "p_l": "L", "L": "L",
    "p_earned_run_avg": "earned_run_avg", "earned_run_avg": "earned_run_avg",
    "p_g": "G",
    "p_gs": "GS", "GS": "GS",
    "p_ip": "IP", "IP": "IP",
    "p_h": "H",
    "p_er": "ER", "ER": "ER",
    "p_hr": "HR",
    "p_bb": "BB",
    "p_so": "SO",
    "p_bfp": "batters_faced", "batters_faced": "batters_faced", "BF": "batters_faced",
}

# A table qualifies as a season-by-season batting or pitching table only if
# its rows carry these canonical fields.
_BATTING_SIGNATURE = {"season", "PA", "AB", "HR"}
_PITCHING_SIGNATURE = {"season", "IP", "ER"}

_TOTALS_LABELS = {
    "league average", "mlb totals", "major league totals", "lg average",
    "league totals", "totals", "average", "162 game avg", "career",
}


def player_url(bbref_id: str) -> str:
    return f"{config.BREF_BASE}/players/{bbref_id[0]}/{bbref_id}.shtml"


def league_url(year: int, side: str) -> str:
    kind = "standard-batting" if side == "batting" else "standard-pitching"
    return f"{config.BREF_BASE}/leagues/majors/{year}-{kind}.shtml"


def _canonicalise(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        out[_CANONICAL.get(key, key)] = value
    return out


def _table_rows(soup, table) -> list[dict[str, str]]:
    table_id = table.get("id")
    if not table_id:
        return []
    return [_canonicalise(r) for r in parse_bref.parse_table(soup, table_id)]


def discover_tables(html: str) -> dict[str, list[dict[str, str]]]:
    """Find the season-by-season batting and pitching tables on any page.

    Matching on the *columns a table contains* rather than on its id means a
    Baseball-Reference redesign that renames tables does not break this.
    """
    soup = parse_bref.make_soup(html)
    found: dict[str, list[dict[str, str]]] = {}

    for table in soup.find_all("table"):
        rows = _table_rows(soup, table)
        if not rows:
            continue
        present = set(rows[0])
        if _BATTING_SIGNATURE <= present and "batting" not in found:
            found["batting"] = rows
        elif _PITCHING_SIGNATURE <= present and "pitching" not in found:
            found["pitching"] = rows

    return found


def _is_season_row(row: dict[str, str]) -> bool:
    season = str(row.get("season", "")).strip()
    return bool(YEAR_RE.match(season))


def _clean_seasons(rows: list[dict[str, str]], side: str) -> pd.DataFrame:
    """Keep real season rows, collapse midseason trades, coerce numerics."""
    season_rows = [r for r in rows if _is_season_row(r)]
    if not season_rows:
        return pd.DataFrame()

    df = pd.DataFrame(season_rows)
    label = df.get("team", pd.Series("", index=df.index)).fillna("").str.lower()
    df = df[~label.isin(_TOTALS_LABELS)]
    if df.empty:
        return df

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    numeric = ["PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO",
               "TB", "HBP", "SF", "SH", "G", "GS", "age", "IP", "ER",
               "batters_faced", "W", "L", "earned_run_avg", "batting_avg",
               "onbase_perc", "slugging_perc"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce")

    # A player traded midseason gets one row per team plus a combined row.
    # Keeping the row with the most playing time picks the combined one and
    # avoids counting the same season twice.
    pt = "PA" if side == "batting" else "batters_faced"
    if pt in df.columns:
        df = (df.sort_values(pt, ascending=False)
                .drop_duplicates(subset=["season"], keep="first"))
    else:
        df = df.drop_duplicates(subset=["season"], keep="first")

    return df.sort_values("season").reset_index(drop=True)


def fetch_player(fetcher: CachedFetcher, bbref_id: str
                 ) -> dict[str, pd.DataFrame]:
    """One player's MLB seasons, batting and pitching."""
    html = fetcher.get(player_url(bbref_id))
    tables = discover_tables(html)
    out: dict[str, pd.DataFrame] = {}
    for side in ("batting", "pitching"):
        df = _clean_seasons(tables.get(side, []), side)
        if not df.empty:
            df.insert(0, "key_bbref", bbref_id)
        out[side] = df
    return out


def fetch_players(fetcher: CachedFetcher, bbref_ids: list[str]
                  ) -> dict[str, pd.DataFrame]:
    """Every matched player's MLB seasons, concatenated."""
    batting, pitching, failures = [], [], []
    total = len(bbref_ids)

    for i, pid in enumerate(bbref_ids, 1):
        if i % 25 == 0 or i == 1:
            log.info("player %d/%d (%s)", i, total, pid)
        try:
            got = fetch_player(fetcher, pid)
        except Exception as exc:                      # keep going
            failures.append({"key_bbref": pid, "error": str(exc)})
            log.warning("failed %s: %s", pid, exc)
            continue
        if not got["batting"].empty:
            batting.append(got["batting"])
        if not got["pitching"].empty:
            pitching.append(got["pitching"])

    return {
        "batting": pd.concat(batting, ignore_index=True) if batting
        else pd.DataFrame(),
        "pitching": pd.concat(pitching, ignore_index=True) if pitching
        else pd.DataFrame(),
        "failures": pd.DataFrame(failures),
    }


def fetch_league_totals(fetcher: CachedFetcher, years: list[int]
                        ) -> pd.DataFrame:
    """MLB league totals per season, for the relative-rate baseline."""
    records = []
    for year in years:
        for side in ("batting", "pitching"):
            try:
                html = fetcher.get(league_url(year, side))
            except Exception as exc:
                log.warning("league %s %s failed: %s", year, side, exc)
                continue
            soup = parse_bref.make_soup(html)
            for table in soup.find_all("table"):
                rows = _table_rows(soup, table)
                if not rows:
                    continue
                sig = (_BATTING_SIGNATURE - {"season"} if side == "batting"
                       else _PITCHING_SIGNATURE - {"season"})
                if not sig <= set(rows[0]):
                    continue
                for row in rows:
                    label = str(row.get("team", "")).strip().lower()
                    if (row.get("is_aggregate") == "True"
                            or label in _TOTALS_LABELS):
                        records.append({"season": year, "side": side, **row})
                break
    return pd.DataFrame(records)
