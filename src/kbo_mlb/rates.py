"""Player rates, league-average rates, and the ratio between them.

The single most important design rule in this module: **a league rate and a
player rate are computed by the same function.** If league K% were computed
as a mean of player K% while player K% is SO/PA, every comparison downstream
would be quietly wrong. So each statistic is defined once, as a function of
counting totals, and applied to both a player row and a league-totals row.

Why work in ratios at all. The KBO's run environment moves a lot: the league
hit .277 in 2024 and .261 in 2013, and the ball itself changed around 2019.
A raw .300 hitter is a different player in each of those seasons. Dividing
by the player's own league-year removes that, so what the model learns is
"how does being 20% better than your league translate", not "how does .300
translate".

One caveat this module cannot fix, and which the audit already flagged: the
scraped player rows fall short of the published league totals in some
seasons, because the source omits marginal players. That is why league rates
here come from the **published league totals**, never from summing players.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rate definitions
# ---------------------------------------------------------------------------
#
# Each takes a frame of counting totals and returns a Series. Written to work
# on one row (a league total) or a million (every player-season).


def _safe(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN rather than inf where the denominator is zero."""
    denom = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    return (num / denom.where(denom > 0)).astype(float)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Fetch a column as numbers, or zeros if the source never provided it."""
    if name not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


BATTING_RATES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Plate-discipline and power rates are per plate appearance, which is the
    # denominator that survives a league change most cleanly.
    "k_pct":   lambda d: _safe(_col(d, "SO"), _col(d, "PA")),
    "bb_pct":  lambda d: _safe(_col(d, "BB"), _col(d, "PA")),
    "hr_pct":  lambda d: _safe(_col(d, "HR"), _col(d, "PA")),
    "avg":     lambda d: _safe(_col(d, "H"), _col(d, "AB")),
    "obp":     lambda d: _safe(
        _col(d, "H") + _col(d, "BB") + _col(d, "HBP"),
        _col(d, "AB") + _col(d, "BB") + _col(d, "HBP") + _col(d, "SF")),
    "slg":     lambda d: _safe(_col(d, "TB"), _col(d, "AB")),
    "iso":     lambda d: (_safe(_col(d, "TB"), _col(d, "AB"))
                          - _safe(_col(d, "H"), _col(d, "AB"))),
    # Batting average on balls in play. Isolated because it is the rate that
    # travels *worst* between leagues - it mixes contact quality with defence
    # and luck - and showing that explicitly is part of the point.
    "babip":   lambda d: _safe(
        _col(d, "H") - _col(d, "HR"),
        _col(d, "AB") - _col(d, "SO") - _col(d, "HR") + _col(d, "SF")),
}

PITCHING_RATES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "k_pct":   lambda d: _safe(_col(d, "SO"), _col(d, "batters_faced")),
    "bb_pct":  lambda d: _safe(_col(d, "BB"), _col(d, "batters_faced")),
    "hr_pct":  lambda d: _safe(_col(d, "HR"), _col(d, "batters_faced")),
    "h_pct":   lambda d: _safe(_col(d, "H"), _col(d, "batters_faced")),
    # Earned run average, rebuilt from outs rather than trusting the printed
    # value, because innings are recorded in thirds (5.1 means 5 1/3).
    "era":     lambda d: _safe(_col(d, "ER") * 27.0, _outs(d)),
}


def _outs(df: pd.DataFrame) -> pd.Series:
    """Convert innings-pitched thirds notation to outs, vectorised."""
    ip = pd.to_numeric(df.get("IP"), errors="coerce")
    whole = np.floor(ip)
    frac = np.round((ip - whole) * 10)
    thirds = np.where(np.isin(frac, [0, 1, 2]), frac, np.nan)
    outs = whole * 3 + thirds
    # A value that was never thirds notation is treated as a plain decimal.
    return pd.Series(np.where(np.isnan(outs), ip * 3, outs), index=df.index)


RATE_SETS = {"batting": BATTING_RATES, "pitching": PITCHING_RATES}

# Playing-time column per side, used for weighting and minimums.
PLAYING_TIME = {"batting": "PA", "pitching": "batters_faced"}


# ---------------------------------------------------------------------------
# Applying them
# ---------------------------------------------------------------------------

def add_rates(df: pd.DataFrame, side: str, prefix: str = "") -> pd.DataFrame:
    """Attach every rate for `side` as new columns."""
    out = df.copy()
    for name, fn in RATE_SETS[side].items():
        out[f"{prefix}{name}"] = fn(out)
    return out


def league_rates(league_totals: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per season of league-average rates, from published totals.

    `league_totals` is the table scraped from each season page's
    `league_batting` / `league_pitching` tfoot - the source's own published
    figure, which is complete even where its player lists are not.
    """
    lt = league_totals
    if "side" in lt.columns:
        lt = lt[lt["side"] == side]
    if lt.empty:
        return pd.DataFrame(columns=["season"])

    lt = lt.copy()
    lt["season"] = pd.to_numeric(lt["season"], errors="coerce").astype("Int64")

    # Guard against a season appearing twice.
    numeric_cols = [c for c in lt.columns
                    if c not in ("season", "side", "team_ID", "is_aggregate",
                                 "affiliation")]
    for c in numeric_cols:
        lt[c] = pd.to_numeric(lt[c], errors="coerce")
    grouped = lt.groupby("season", as_index=False)[numeric_cols].sum()

    out = add_rates(grouped, side)
    keep = ["season"] + list(RATE_SETS[side])
    return out[keep].rename(
        columns={k: f"lg_{k}" for k in RATE_SETS[side]}
    )


def add_relative_rates(
    players: pd.DataFrame,
    league: pd.DataFrame,
    side: str,
    min_playing_time: int = 0,
) -> pd.DataFrame:
    """Express each player's rates as a ratio to his own league-year.

    A value of 1.0 is exactly league average. 1.25 means 25% above it.
    Rows below `min_playing_time` are dropped, because a ratio built on 12
    plate appearances is noise wearing a number's clothes.
    """
    rates = RATE_SETS[side]
    df = add_rates(players, side)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    pt_col = PLAYING_TIME[side]
    if pt_col in df.columns and min_playing_time:
        df = df[pd.to_numeric(df[pt_col], errors="coerce").fillna(0)
                >= min_playing_time]

    merged = df.merge(league, on="season", how="left")
    for name in rates:
        lg = merged.get(f"lg_{name}")
        if lg is None:
            continue
        merged[f"rel_{name}"] = _safe(merged[name], lg)

    return merged


def relative_columns(side: str) -> list[str]:
    return [f"rel_{k}" for k in RATE_SETS[side]]
