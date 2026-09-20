"""Which KBO players would actually draw MLB interest?

Eligibility is not interest. Sixty players clear the posting and free-agency
filters; historically about one a year is posted. Listing all sixty as
"signing targets" is the same failure as projecting a home run rate with a
300x interval - technically derived, practically useless.

The honest way to narrow it is not to invent a threshold. It is to ask what
the players who were **actually posted** looked like in Korea beforehand,
and screen against that. Those players are in the data.

## The screen

For each Korean player who moved KBO -> MLB, take his last two KBO seasons
before the move and compute one number:

    hitters   OPS relative to his own league that season
    pitchers  strikeout rate relative to his own league

Strikeout rate is deliberately the pitcher measure. It is the only pitching
statistic this project found to carry across leagues at all (see
`reports/validation_pitching.md`), so screening on ERA would be screening
on noise.

A current player is then compared against that historical distribution at a
chosen strictness.

## Why strictness is a parameter and not a constant

The reference class is tiny - nine hitters and seven pitchers - so any cut
is fitted to a handful of careers. Worse, the two sides behave differently.
The hitters are a fairly tight group. The pitchers are not: Oh Seung-hwan
and Lim Chang-yong were relievers posted at 30 and 31, and Lim's rates sit
*below* league average, which drags a minimum-based floor down until it
admits 60% of the league.

So `screen` reports both numbers every time: how many current players it
admits, and how many of the historically posted players it would have
caught. A setting that admits everyone is not a screen, and a setting that
would have missed Lee Jung-hoo is not one either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Strictness -> which percentile of the historical cohort becomes the bar.
STRICTNESS = {
    "permissive": 0.00,   # the weakest player ever posted
    "balanced": 0.25,
    "strict": 0.50,       # the median of players actually posted
}

SCORE_NAME = {"batting": "relative OPS", "pitching": "relative K rate"}


def score(history: pd.DataFrame, side: str) -> float:
    """One number for a player's recent KBO form, relative to his league.

    Weighted by playing time across the seasons supplied, so a short
    season counts for less rather than equally.
    """
    if history.empty:
        return float("nan")

    pt_col = "PA" if side == "batting" else "batters_faced"
    w = pd.to_numeric(history.get(pt_col), errors="coerce").fillna(0.0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=history.index)

    if side == "batting":
        need = ("obp", "slg", "lg_obp", "lg_slg")
        if not all(c in history.columns for c in need):
            return float("nan")
        num = ((pd.to_numeric(history["obp"], errors="coerce")
                + pd.to_numeric(history["slg"], errors="coerce")) * w).sum()
        den = ((pd.to_numeric(history["lg_obp"], errors="coerce")
                + pd.to_numeric(history["lg_slg"], errors="coerce")) * w).sum()
        return float(num / den) if den else float("nan")

    if "rel_k_pct" not in history.columns:
        return float("nan")
    k = pd.to_numeric(history["rel_k_pct"], errors="coerce")
    if not k.notna().any():
        return float("nan")
    return float((k.fillna(0) * w).sum() / w.sum())


def recent(kbo: pd.DataFrame, player_id: str, seasons: int = 2,
           before: int | None = None) -> pd.DataFrame:
    rows = kbo[kbo["player_register_id"] == player_id]
    if before is not None:
        rows = rows[pd.to_numeric(rows["season"], errors="coerce") < before]
    return rows.sort_values("season").tail(seasons)


def historical_profile(
    kbo: pd.DataFrame,
    posted: pd.DataFrame,
    side: str,
    is_korean,
    seasons: int = 2,
) -> pd.DataFrame:
    """What the players who actually got posted looked like beforehand.

    `posted` is the cohort table from `cohorts.classify`, filtered to
    POSTED. `is_korean` is a callable (name, birth_city) -> bool; imports
    who happen to reach MLB after a KBO stint are not posted Korean stars
    and would distort the benchmark.
    """
    columns = ["player", "player_register_id", "first_mlb", "score",
               "age_last_kbo"]
    out = []
    for _, row in posted.iterrows():
        pid = row["player_register_id"]
        first_mlb = row.get("first_mlb")
        if pd.isna(first_mlb):
            continue
        history = recent(kbo, pid, seasons, before=int(first_mlb))
        if history.empty:
            continue
        name = history["player"].iloc[-1]
        if not is_korean(name, pid):
            continue
        value = score(history, side)
        if not np.isfinite(value):
            continue
        out.append({
            "player": name,
            "player_register_id": pid,
            "first_mlb": int(first_mlb),
            "score": round(value, 3),
            "age_last_kbo": float(
                pd.to_numeric(history["age"], errors="coerce").iloc[-1]),
        })
    if not out:
        # No qualifying cohort. Returning a bare empty frame here made
        # `sort_values("score")` raise KeyError, which would have taken the
        # whole app down on any season or side with nothing to benchmark
        # against. An empty profile with the right columns lets `screen`
        # do what it should in that case: admit everyone and say the bar
        # is unknown.
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out, columns=columns).sort_values(
        "score", ascending=False)


def bar(profile: pd.DataFrame, strictness: str = "balanced") -> float:
    """The score a current player has to beat."""
    if profile.empty:
        return float("nan")
    q = STRICTNESS.get(strictness, 0.25)
    return float(profile["score"].quantile(q))


def screen(scores: pd.Series, profile: pd.DataFrame,
           strictness: str = "balanced") -> dict:
    """Apply the bar, and report what it costs in both directions.

    Returns the mask plus the two numbers that make the cut auditable:
    how many current players it admits, and how many of the players who
    were actually posted it would have caught.
    """
    threshold = bar(profile, strictness)
    if not np.isfinite(threshold):
        return {"mask": pd.Series(True, index=scores.index),
                "threshold": float("nan"), "admitted": len(scores),
                "historical_caught": 0, "historical_total": 0,
                "missed": []}

    mask = scores >= threshold
    caught = profile[profile["score"] >= threshold]
    missed = profile[profile["score"] < threshold]["player"].tolist()

    return {
        "mask": mask.fillna(False),
        "threshold": round(threshold, 3),
        "admitted": int(mask.fillna(False).sum()),
        "of_total": int(len(scores)),
        "historical_caught": int(len(caught)),
        "historical_total": int(len(profile)),
        "missed": missed,
    }
