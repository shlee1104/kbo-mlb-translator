"""The MLB side of the join: Chadwick Bureau IDs and MLB season stats.

Chadwick Bureau's open register is the canonical player-ID crosswalk. Its
`key_bbref_minors` column holds the same register ID that KBO pages link to,
which is what makes tier-1 matching deterministic rather than name-based.

`pybaseball` is optional. If it is not installed the pipeline still runs and
produces the KBO tables, the data audit, and a crosswalk built from Chadwick
alone; only the MLB stat pull is skipped. Keeping it optional means a reviewer
can clone the repo and run the Phase 1 work without a heavy dependency tree.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

CHADWICK_COLUMNS = [
    "key_person", "key_uuid", "key_mlbam", "key_retro", "key_bbref",
    "key_bbref_minors", "key_fangraphs", "name_last", "name_first",
    "name_given", "birth_year", "birth_month", "birth_day", "pro_played_first",
    "pro_played_last", "mlb_played_first", "mlb_played_last",
]


def load_chadwick(force_refresh: bool = False) -> pd.DataFrame:
    """Download (and cache) the Chadwick register.

    The register is sharded into 16 files by the first hex digit of the UUID.
    We keep a single concatenated parquet/csv copy in data/raw/.
    """
    cache = config.RAW_DIR / "chadwick_people.csv"
    if cache.exists() and not force_refresh:
        log.info("Chadwick register: using cached copy at %s", cache)
        return pd.read_csv(cache, low_memory=False)

    frames = []
    for part in config.CHADWICK_PARTS:
        url = f"{config.CHADWICK_BASE}/{part}"
        log.info("downloading %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text), low_memory=False))

    people = pd.concat(frames, ignore_index=True)
    keep = [c for c in CHADWICK_COLUMNS if c in people.columns]
    people = people[keep]
    people.to_csv(cache, index=False)
    log.info("Chadwick register: %d people cached to %s", len(people), cache)
    return people


def chadwick_with_names(people: pd.DataFrame) -> pd.DataFrame:
    """Add a single `name` column and an ISO `date_of_birth`."""
    out = people.copy()
    first = out.get("name_first", pd.Series("", index=out.index)).fillna("")
    last = out.get("name_last", pd.Series("", index=out.index)).fillna("")
    out["name"] = (first.astype(str) + " " + last.astype(str)).str.strip()

    out["date_of_birth"] = pd.to_datetime(
        dict(
            year=pd.to_numeric(out.get("birth_year"), errors="coerce"),
            month=pd.to_numeric(out.get("birth_month"), errors="coerce"),
            day=pd.to_numeric(out.get("birth_day"), errors="coerce"),
        ),
        errors="coerce",
    )
    return out


def mlb_players_who_played(people: pd.DataFrame) -> pd.DataFrame:
    """Restrict the register to people with actual MLB service."""
    if "mlb_played_first" not in people.columns:
        return people
    return people[people["mlb_played_first"].notna()].copy()


# ---------------------------------------------------------------------------
# Season stats (optional dependency)
# ---------------------------------------------------------------------------

def pybaseball_available() -> bool:
    try:
        import pybaseball  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_mlb_batting(start: int, end: int) -> pd.DataFrame:
    """Season-level MLB batting from FanGraphs, via pybaseball.

    Returns an empty frame (with a logged warning) when pybaseball is absent,
    so callers never need to branch on it.
    """
    if not pybaseball_available():
        log.warning("pybaseball not installed - skipping the MLB batting pull. "
                    "Install it with: pip install pybaseball")
        return pd.DataFrame()

    from pybaseball import batting_stats

    cache = config.RAW_DIR / f"mlb_batting_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)

    df = batting_stats(start, end, qual=1)
    df.to_csv(cache, index=False)
    log.info("MLB batting: %d player-seasons cached", len(df))
    return df


def fetch_mlb_pitching(start: int, end: int) -> pd.DataFrame:
    """Season-level MLB pitching from FanGraphs, via pybaseball."""
    if not pybaseball_available():
        log.warning("pybaseball not installed - skipping the MLB pitching pull.")
        return pd.DataFrame()

    from pybaseball import pitching_stats

    cache = config.RAW_DIR / f"mlb_pitching_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)

    df = pitching_stats(start, end, qual=1)
    df.to_csv(cache, index=False)
    log.info("MLB pitching: %d player-seasons cached", len(df))
    return df
