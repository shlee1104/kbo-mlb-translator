"""MLB season stats from MLB's own Stats API.

Why not FanGraphs via pybaseball: that library fetches a legacy leaderboard
page which FanGraphs now answers with HTTP 403. More importantly, the Stats
API is a better source for this project regardless:

* It is MLB's official public endpoint - no scraping, no terms-of-use grey
  area, no HTML parsing that breaks when a page is restyled.
* It is keyed by **MLBAM id**, which the Chadwick crosswalk already gives us
  for all 476 matched players.
* We only need those 476 players, not all of MLB, so the whole pull is a few
  hundred requests instead of tens of thousands of rows.

Column names are deliberately renamed to match the KBO tables (PA, AB, H,
HR, SO, BB, TB, HBP, SF, batters_faced, ...). That is what lets `rates.py`
apply the *same* rate functions to both leagues - the property that makes a
KBO-to-MLB comparison meaningful in the first place.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from . import config
from .http_client import CachedFetcher

log = logging.getLogger(__name__)

API = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1

# Stats API field -> our schema (shared with the KBO tables).
BATTING_MAP = {
    "plateAppearances": "PA",
    "atBats": "AB",
    "hits": "H",
    "doubles": "2B",
    "triples": "3B",
    "homeRuns": "HR",
    "strikeOuts": "SO",
    "baseOnBalls": "BB",
    "intentionalWalks": "IBB",
    "hitByPitch": "HBP",
    "sacFlies": "SF",
    "sacBunts": "SH",
    "totalBases": "TB",
    "rbi": "RBI",
    "stolenBases": "SB",
    "caughtStealing": "CS",
    "runs": "R",
    "gamesPlayed": "G",
    "groundIntoDoublePlay": "GIDP",
}

PITCHING_MAP = {
    "battersFaced": "batters_faced",
    "inningsPitched": "IP",
    "strikeOuts": "SO",
    "baseOnBalls": "BB",
    "homeRuns": "HR",
    "hits": "H",
    "earnedRuns": "ER",
    "runs": "R",
    "gamesPlayed": "G",
    "gamesStarted": "GS",
    "saves": "SV",
    "wins": "W",
    "losses": "L",
}

FIELD_MAPS = {"batting": BATTING_MAP, "pitching": PITCHING_MAP}
API_GROUP = {"batting": "hitting", "pitching": "pitching"}


def make_fetcher(offline: bool = False) -> CachedFetcher:
    """A faster fetcher than the Baseball-Reference one.

    Baseball-Reference publishes a 3-second crawl delay and we honour it.
    This is a different host with different expectations: an official JSON
    API built to be queried. We still rate-limit and still cache every
    response, just at a pace suited to an API rather than a web page.
    """
    fetcher = CachedFetcher(
        cache_dir=config.RAW_DIR / "statsapi_cache",
        delay_seconds=0.15,
        max_per_minute=240,
        offline=offline,
    )
    # The shared fetcher was built for scraping HTML and asks for
    # text/html. This endpoint serves JSON and correctly answers 406 Not
    # Acceptable to that, so the header has to be replaced, not appended to.
    fetcher.session.headers["Accept"] = "application/json"
    return fetcher


def _get_json(fetcher: CachedFetcher, url: str) -> dict:
    raw = fetcher.get(url)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("non-JSON response from %s", url)
        return {}


def _clean_split(split: dict, side: str) -> dict | None:
    """Turn one Stats API split into a row in our schema."""
    stat = split.get("stat") or {}
    if not stat:
        return None

    # Keep major-league regular season only. yearByYear can include other
    # sports (minors, winter ball) depending on the player.
    sport = (split.get("sport") or {}).get("id")
    if sport is not None and sport != MLB_SPORT_ID:
        return None
    game_type = split.get("gameType")
    if game_type is not None and game_type != "R":
        return None

    row: dict = {"season": pd.to_numeric(split.get("season"), errors="coerce")}
    for api_field, ours in FIELD_MAPS[side].items():
        if api_field in stat:
            row[ours] = stat[api_field]
    if "age" in stat:
        row["age"] = stat["age"]

    team = split.get("team") or {}
    row["mlb_team"] = team.get("name")
    return row


def fetch_player_seasons(
    mlbam_ids: list[int],
    side: str,
    fetcher: CachedFetcher | None = None,
) -> pd.DataFrame:
    """Year-by-year MLB stats for the given players.

    One request per player; every response is cached, so a re-run is free.
    """
    fetcher = fetcher or make_fetcher()
    group = API_GROUP[side]
    rows: list[dict] = []

    ids = [int(i) for i in pd.Series(mlbam_ids).dropna().unique()]
    failures = 0
    for n, pid in enumerate(ids, 1):
        url = (f"{API}/people/{pid}/stats"
               f"?stats=yearByYear&group={group}")
        try:
            payload = _get_json(fetcher, url)
        except RuntimeError as exc:
            failures += 1
            # Log the first few in full, then stop repeating the same thing.
            if failures <= 3:
                log.warning("player %s failed: %s", pid, exc)
            # A handful of failures is normal (a player with no MLB record
            # on this side of the ball). Everything failing is a broken
            # request shape, and continuing would write a plausible-looking
            # empty file instead of telling anyone.
            if failures >= 10 and failures == n:
                raise RuntimeError(
                    f"every one of the first {n} requests failed for "
                    f"{side}. Last error: {exc}\nURL shape: {url}"
                ) from exc
            continue

        for block in payload.get("stats", []):
            for split in block.get("splits", []):
                row = _clean_split(split, side)
                if row is None:
                    continue
                row["key_mlbam"] = pid
                rows.append(row)

        if n % 50 == 0:
            log.info("  %d/%d players", n, len(ids))

    if failures:
        log.info("%s: %d/%d players returned nothing", side, failures, len(ids))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # A midseason trade yields one split per team; combine to a season total.
    count_cols = [c for c in df.columns
                  if c not in ("key_mlbam", "season", "age", "mlb_team", "IP")]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = {c: "sum" for c in count_cols}
    if "age" in df.columns:
        agg["age"] = "max"
    if "IP" in df.columns:
        # Innings are thirds notation; summing the decimals would be wrong,
        # so convert to outs, sum, and convert back.
        df["_outs"] = df["IP"].map(_ip_to_outs)
        agg["_outs"] = "sum"

    out = df.groupby(["key_mlbam", "season"], as_index=False).agg(agg)
    if "_outs" in out.columns:
        out["IP"] = out["_outs"].map(_outs_to_ip)
        out = out.drop(columns=["_outs"])
    return out


def _ip_to_outs(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    whole = int(v)
    frac = round((v - whole) * 10)
    if frac not in (0, 1, 2):
        return round(v * 3, 2)
    return float(whole * 3 + frac)


def _outs_to_ip(outs) -> float:
    try:
        o = float(outs)
    except (TypeError, ValueError):
        return float("nan")
    return float(int(o // 3)) + (o % 3) / 10.0


def fetch_league_totals(
    seasons: list[int],
    side: str,
    fetcher: CachedFetcher | None = None,
) -> pd.DataFrame:
    """MLB league totals per season, summed across all 30 teams.

    Mirrors the KBO side, where league context comes from the source's own
    published totals rather than from summing the players we happened to
    collect.
    """
    fetcher = fetcher or make_fetcher()
    group = API_GROUP[side]
    rows: list[dict] = []

    for season in sorted(set(int(s) for s in seasons)):
        url = (f"{API}/teams/stats?season={season}&sportIds={MLB_SPORT_ID}"
               f"&group={group}&stats=season")
        try:
            payload = _get_json(fetcher, url)
        except RuntimeError as exc:
            log.warning("season %s league totals failed: %s", season, exc)
            continue

        totals: dict = {"season": season, "side": side}
        found = False
        for block in payload.get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat") or {}
                if not stat:
                    continue
                found = True
                for api_field, ours in FIELD_MAPS[side].items():
                    if api_field not in stat:
                        continue
                    value = pd.to_numeric(stat[api_field], errors="coerce")
                    if pd.isna(value):
                        continue
                    if ours == "IP":
                        totals["_outs"] = totals.get("_outs", 0.0) + _ip_to_outs(
                            stat[api_field])
                    else:
                        totals[ours] = totals.get(ours, 0.0) + float(value)
        if found:
            if "_outs" in totals:
                totals["IP"] = _outs_to_ip(totals.pop("_outs"))
            rows.append(totals)

    return pd.DataFrame(rows)
