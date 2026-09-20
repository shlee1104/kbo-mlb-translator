"""Forward projections for current KBO players, with availability.

A projected stat line on its own is only half of what a front office needs.
The other half is *when, and under which rules*, the player could actually
be signed - because the same player is worth very different amounts
depending on the timing of his post.

## Two rules drive the availability side

**Posting.** A KBO player cannot simply leave. He needs roughly seven
seasons of KBO service before his club may post him, and around nine for
domestic free agency. Service time has its own accrual rules and military
service complicates it, so the seasons-played count used here is a
**proxy**, not a service-time calculation.

**The bonus pool.** Under the current Major League agreement, a foreign
professional is exempt from the international bonus pools only if he is at
least 25 years old *and* has six or more professional seasons. Below that
he is pool-capped, which is the difference between a market contract and a
signing-bonus slot. Yamamoto was posted at 25 and signed for $325M;
Sasaki was posted at 23 and was pool-restricted.

That threshold is the single highest-leverage fact in this module. A player
posted one year early can be worth an order of magnitude less.

**Verify both before relying on them.** The posting agreement has been
revised before, and the Major League labour agreement expires 1 December
2026 with an international draft under negotiation, which would rewrite
this entirely.
"""

from __future__ import annotations

import pandas as pd

# Verified against public sources, September 2026. Still worth re-checking:
# the Major League labour agreement expires 1 December 2026.
#
#   posting at 7 KBO seasons
#     https://en.wikipedia.org/wiki/Posting_system_(KBO)
#   international free agency at 9 years, and the bonus-pool exemption at
#   25+ with 6+ professional seasons
#     https://www.mlb.com/glossary/transactions/korean-posting-system
SEASONS_FOR_POSTING = 7
SEASONS_FOR_INTERNATIONAL_FA = 9
POOL_EXEMPT_MIN_AGE = 25
POOL_EXEMPT_MIN_PRO_SEASONS = 6

# KBO free agency, per the Korean-language rules (Namu Wiki, translated).
# The structure is more intricate than a season count:
#
#   A credited season requires 145+ days registered on the first team
#   (1군). Days short of 145 in one year CARRY OVER and combine with other
#   years until they total 145. National-team call-ups are credited back.
#
#   First domestic FA:  9 seasons for high-school entrants,
#                       7 or 8 for four-year-college entrants.
#   Moving OVERSEAS:    8 seasons, high-school and college alike.
#
# Two consequences worth being explicit about.
#
# First, the seasons counted in this module are calendar seasons in which a
# player appeared, NOT 145-day credited seasons, because registered-day
# counts are not published in any source this pipeline reads. A player who
# appeared briefly gets a full season here and none from the KBO. Every
# date this module produces is therefore an ESTIMATE, not a schedule.
#
# Second, sources disagree. MLB's own glossary describes international free
# agency as requiring nine years of professional experience, while the
# Korean rules give eight seasons for an overseas move, and Namu Wiki
# hedges the college number as "7 (or 8)". Nobody should treat any of these
# as settled.
KBO_OVERSEAS_FA_SEASONS = 8      # free to leave without club consent
KBO_DOMESTIC_FA_SEASONS = 9      # high-school entrants; the common case

# For a high-school entrant the sequence runs:
#   season 7  postable, but only with the club's consent
#   season 8  overseas free agent - can leave on his own
#   season 9  domestic free agent - can take a Korean payday instead
#
# In practice club consent is rarely the binding constraint. It is refused
# when a club would lose several eligible players at once: Kiwoom held Kim
# Hye-seong back a year rather than post him alongside Lee Jung-hoo. That
# is roster congestion, not reluctance, and `posting_congestion` below
# flags it rather than modelling consent as a general barrier.

# Backwards-compatible alias.
SEASONS_FOR_FREE_AGENCY = SEASONS_FOR_INTERNATIONAL_FA


def availability(
    seasons_played: int,
    age_now: float,
    current_season: int,
    fa_seasons: int = KBO_DOMESTIC_FA_SEASONS,
) -> dict:
    """When could this player be posted, and would he be pool-capped?

    All of it keys off seasons played, which is a proxy for service time.
    """
    seasons_to_go = max(0, SEASONS_FOR_POSTING - seasons_played)
    earliest = current_season + seasons_to_go
    age_then = age_now + seasons_to_go
    seasons_then = seasons_played + seasons_to_go

    exempt = (age_then >= POOL_EXEMPT_MIN_AGE
              and seasons_then >= POOL_EXEMPT_MIN_PRO_SEASONS)

    # How much longer until the pool no longer applies, if it does now?
    years_to_exempt = max(0, POOL_EXEMPT_MIN_AGE - age_then)

    # Two different freedoms, at two different thresholds.
    overseas_season = current_season + max(
        0, KBO_OVERSEAS_FA_SEASONS - seasons_played)
    fa_season = current_season + max(0, fa_seasons - seasons_played)
    past_fa = seasons_played >= fa_seasons

    # The seasons in which he is postable but has not yet re-signed at home.
    window_start = max(current_season, earliest)
    window_end = fa_season - 1
    window = max(0, window_end - window_start + 1) if not past_fa else 0

    return {
        "seasons_played": seasons_played,
        "seasons_until_posting_eligible": seasons_to_go,
        "earliest_posting_season": earliest,
        "age_at_earliest_posting": round(age_then, 1),
        "bonus_pool_exempt_if_posted_then": exempt,
        "extra_years_for_pool_exemption": (0 if exempt
                                           else round(years_to_exempt, 1)),
        "note": ("market contract" if exempt
                 else "pool-capped: bonus slot only"),
        "domestic_fa_season": fa_season,
        "overseas_fa_season": overseas_season,
        "needs_club_consent_until": overseas_season,
        "past_first_fa": past_fa,
        "acquisition_window_seasons": window,
        "window": (f"{window_start}-{window_end}"
                   if window > 0 else "closed"),
    }


def posting_congestion(
    roster: pd.DataFrame,
    season_col: str = "earliest_posting_season",
    team_col: str = "team_name",
) -> pd.DataFrame:
    """Flag clubs that would lose several players to posting in one year.

    Club consent is almost always granted, so on paper every player who
    reaches seven seasons is available. The exception is a club with more
    than one eligible player in the same window: rather than lose both,
    it holds one back. Kiwoom posted Lee Jung-hoo after 2023 and Kim
    Hye-seong after 2024 - the same club, staggered by a year, because both
    became eligible together.

    That makes congestion a better predictor of *when* a player actually
    becomes available than the rule itself. Adds two columns:

      club_eligible_that_year  how many of the club's players reach posting
                               eligibility in the same season
      posting_likely_delayed   True when that count is above one
    """
    out = roster.copy()
    if season_col not in out.columns or team_col not in out.columns:
        out["club_eligible_that_year"] = 1
        out["posting_likely_delayed"] = False
        return out

    counts = out.groupby([team_col, season_col])[season_col].transform("size")
    out["club_eligible_that_year"] = counts.astype(int)
    out["posting_likely_delayed"] = out["club_eligible_that_year"] > 1
    return out


def project_player(
    kbo_rows: pd.DataFrame,
    model,
    draws: dict,
    mlb_league: pd.DataFrame,
    translate_mod,
    weight_recent: int = 3,
) -> pd.DataFrame:
    """Project one player's most recent KBO form into MLB terms.

    The player's league-relative rates are averaged over his last
    `weight_recent` seasons, weighted by playing time. A single season is a
    small sample, and the most recent one is not automatically the most
    representative - Kim Do-young's 2025 was 122 plate appearances.
    """
    rows = kbo_rows.sort_values("season").tail(weight_recent)
    if rows.empty:
        return pd.DataFrame()

    # Pitchers do not have plate appearances. Asking for "PA" on a pitching
    # frame returns None, and pd.to_numeric(None) is a bare nan, so this
    # line used to raise AttributeError before the weighting ever ran -
    # every pitcher's page crashed on open.
    pt_col = next((c for c in ("PA", "batters_faced") if c in rows.columns),
                  None)
    pt = (pd.to_numeric(rows[pt_col], errors="coerce").fillna(0.0)
          if pt_col else pd.Series(0.0, index=rows.index))
    if pt.sum() <= 0:
        pt = pd.Series(1.0, index=rows.index)
    age = float(pd.to_numeric(rows["age"], errors="coerce").iloc[-1])

    latest_league = mlb_league.sort_values("season").iloc[-1]
    out = []

    for stat in model.fits:
        col = f"rel_{stat}"
        if col not in rows.columns:
            continue
        vals = pd.to_numeric(rows[col], errors="coerce")
        ok = vals.notna() & (vals > 0)
        if not ok.any():
            continue
        kbo_rel = float((vals[ok] * pt[ok]).sum() / pt[ok].sum())

        interval = (translate_mod.interval_from_draws(
            draws[stat], kbo_rel, age, "kbo_to_mlb")
            if stat in draws else {})
        if not interval:
            continue

        lg = float(latest_league.get(f"lg_{stat}", float("nan")))
        out.append({
            "stat": stat,
            "kbo_relative": round(kbo_rel, 3),
            "proj_mlb_relative": round(interval["point"], 3),
            "p10_relative": round(interval["p10"], 3),
            "p90_relative": round(interval["p90"], 3),
            "mlb_league_rate": round(lg, 4),
            "proj_mlb_rate": round(interval["point"] * lg, 4),
            "p10_rate": round(interval["p10"] * lg, 4),
            "p90_rate": round(interval["p90"] * lg, 4),
        })

    return pd.DataFrame(out)


def interval_is_informative(row: pd.Series, ratio: float = 3.0) -> bool:
    """Is the range narrow enough to tell anyone anything?

    A projection whose 10th-90th percentile band spans a factor of three is
    not a projection, it is a shrug. Marking those explicitly is more useful
    than printing them next to the ones that mean something.
    """
    lo, hi = row.get("p10_relative"), row.get("p90_relative")
    if not lo or not hi or lo <= 0:
        return False
    return (hi / lo) < ratio
