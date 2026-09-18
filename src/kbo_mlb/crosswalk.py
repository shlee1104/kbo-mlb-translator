"""Link KBO players to their MLB records.

There is no single ID that spans both sides, so we build one in tiers and
record *how* every link was made. A reviewer can then audit the weak links
without re-deriving the strong ones.

    tier 1  register_id   Chadwick Bureau maps the Baseball-Reference
                          register ID (key_bbref_minors) to MLBAM/FanGraphs.
                          Deterministic; nothing to second-guess.
    tier 2  name+dob      Strict name key plus exact date of birth.
    tier 3  phonetic+dob  Loose (phonetic) name key plus exact date of birth.
                          This is the tier that catches Lee/Yi/Rhee and
                          Jung/Jeong/Chung.
    tier 4  phonetic+year Loose name key plus birth *year* only. Proposed,
                          never auto-accepted - written to the review queue.

Anything unmatched also lands in the review queue. The queue is a deliverable,
not a failure: a scout or analyst can resolve a few dozen names by hand, and
the resolutions are read back in on the next run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import names

log = logging.getLogger(__name__)

TIER_CONFIDENCE = {
    "register_id": 1.00,
    "name_dob": 0.95,
    "phonetic_dob": 0.90,
    "phonetic_year": 0.60,
}

AUTO_ACCEPT_TIERS = ("register_id", "name_dob", "phonetic_dob")


@dataclass
class CrosswalkResult:
    matches: pd.DataFrame
    review_queue: pd.DataFrame
    stats: dict = field(default_factory=dict)


def _prep(df: pd.DataFrame, name_col: str, dob_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_strict"] = out[name_col].fillna("").map(names.strict_key)
    out["_loose"] = out[name_col].fillna("").map(names.loose_key)
    dob = pd.to_datetime(out[dob_col], errors="coerce")
    out["_dob"] = dob.dt.strftime("%Y-%m-%d")
    out["_birth_year"] = dob.dt.year
    return out


def _join_on(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    tier: str,
) -> pd.DataFrame:
    """Inner-join on `keys`, keeping only one-to-one matches.

    A key that matches several rows on either side is ambiguous by
    definition, so we drop it here and let it fall through to the review
    queue rather than guessing.
    """
    if left.empty or right.empty:
        return pd.DataFrame()

    l_counts = left.groupby(keys).size()
    r_counts = right.groupby(keys).size()
    unique_keys = set(l_counts[l_counts == 1].index) & set(
        r_counts[r_counts == 1].index
    )
    if not unique_keys:
        return pd.DataFrame()

    def _mask(df):
        idx = df.set_index(keys).index
        return idx.isin(unique_keys)

    merged = left[_mask(left)].merge(
        right[_mask(right)], on=keys, how="inner", suffixes=("_kbo", "_mlb")
    )
    if merged.empty:
        return merged
    merged["match_tier"] = tier
    merged["match_confidence"] = TIER_CONFIDENCE[tier]
    return merged


def build(
    kbo_players: pd.DataFrame,
    mlb_players: pd.DataFrame,
    chadwick: pd.DataFrame | None = None,
    kbo_name_col: str = "player",
    kbo_dob_col: str = "date_of_birth",
    mlb_name_col: str = "name",
    mlb_dob_col: str = "date_of_birth",
) -> CrosswalkResult:
    """Build the KBO <-> MLB player crosswalk.

    `kbo_players` needs `player_register_id` plus a name and date of birth.
    `mlb_players` needs `key_mlbam` (or another stable MLB id) plus the same.
    """
    left = _prep(kbo_players, kbo_name_col, kbo_dob_col)
    right = _prep(mlb_players, mlb_name_col, mlb_dob_col)

    matched_frames: list[pd.DataFrame] = []
    linked_kbo: set[str] = set()

    # --- tier 1: deterministic ID crosswalk --------------------------------
    if chadwick is not None and not chadwick.empty:
        cols = [c for c in ("key_bbref_minors", "key_mlbam", "key_fangraphs")
                if c in chadwick.columns]
        if "key_bbref_minors" in cols and "key_mlbam" in cols:
            bridge = chadwick[cols].dropna(subset=["key_bbref_minors"])
            bridge = bridge.rename(
                columns={"key_bbref_minors": "player_register_id"}
            )
            tier1 = left.merge(bridge, on="player_register_id", how="inner")
            tier1 = tier1.merge(
                right, on="key_mlbam", how="inner", suffixes=("_kbo", "_mlb")
            )
            if not tier1.empty:
                tier1["match_tier"] = "register_id"
                tier1["match_confidence"] = TIER_CONFIDENCE["register_id"]
                matched_frames.append(tier1)
                linked_kbo |= set(tier1["player_register_id"])
            log.info("tier 1 (register_id): %d matches", len(tier1))

    # --- tiers 2-4: name-based, each on the remainder ----------------------
    tier_specs = [
        (["_strict", "_dob"], "name_dob"),
        (["_loose", "_dob"], "phonetic_dob"),
        (["_loose", "_birth_year"], "phonetic_year"),
    ]

    proposals: list[pd.DataFrame] = []
    for keys, tier in tier_specs:
        remaining = left[~left["player_register_id"].isin(linked_kbo)]
        if remaining.empty:
            break
        hit = _join_on(remaining, right, keys, tier)
        log.info("tier %s: %d matches", tier, len(hit))
        if hit.empty:
            continue
        if tier in AUTO_ACCEPT_TIERS:
            matched_frames.append(hit)
            linked_kbo |= set(hit["player_register_id"])
        else:
            proposals.append(hit)

    matches = (
        pd.concat(matched_frames, ignore_index=True)
        if matched_frames else pd.DataFrame()
    )

    # --- review queue ------------------------------------------------------
    unmatched = left[~left["player_register_id"].isin(linked_kbo)].copy()
    unmatched["review_reason"] = "no_match"
    queue_parts = [unmatched]
    for prop in proposals:
        prop = prop.copy()
        prop["review_reason"] = "low_confidence_proposal"
        queue_parts.append(prop)
    review_queue = pd.concat(queue_parts, ignore_index=True)

    stats = {
        "kbo_players": len(left),
        "mlb_players": len(right),
        "matched": len(matches),
        "review_queue": len(review_queue),
        "match_rate": round(len(matches) / max(len(left), 1), 4),
    }
    if not matches.empty:
        stats["by_tier"] = matches["match_tier"].value_counts().to_dict()

    return CrosswalkResult(matches=matches, review_queue=review_queue,
                           stats=stats)


def apply_manual_overrides(
    result: CrosswalkResult, overrides: pd.DataFrame
) -> CrosswalkResult:
    """Fold hand-resolved links back in.

    `overrides` is the reviewed CSV: `player_register_id`, `key_mlbam`, and
    `decision` of "accept" or "reject". Accepted rows join the match table at
    full confidence; rejected ones leave the queue for good.
    """
    if overrides.empty:
        return result

    accepted = overrides[overrides["decision"] == "accept"].copy()
    accepted["match_tier"] = "manual"
    accepted["match_confidence"] = 1.0

    matches = pd.concat([result.matches, accepted], ignore_index=True)
    resolved = set(overrides["player_register_id"])
    queue = result.review_queue[
        ~result.review_queue["player_register_id"].isin(resolved)
    ]

    stats = dict(result.stats)
    stats["manual_overrides"] = len(accepted)
    stats["matched"] = len(matches)
    stats["review_queue"] = len(queue)
    return CrosswalkResult(matches=matches, review_queue=queue, stats=stats)
