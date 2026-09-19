"""KBO to MLB translator - the scout-facing front end.

Run it with:

    PYTHONPATH=src streamlit run app.py

Design intent: a scout should be able to answer two questions without
reading any code, and should not be able to come away with a false
impression of how much we know.

    1. If this KBO player came over, what might he do?
    2. When could we actually sign him, and under which rules?

Everything that can be wrong lives in `kbo_mlb.app_data`, which is covered
by tests. This file is layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kbo_mlb import app_data  # noqa: E402

st.set_page_config(page_title="KBO → MLB Translator",
                   page_icon="⚾", layout="wide")


@st.cache_data(show_spinner="Loading data and fitting the translation…")
def get_bundle(side: str, season: int, min_pt: int, korean_only: bool,
               pre_fa_only: bool, fa_seasons: int):
    return app_data.load(side=side, season=season, min_playing_time=min_pt,
                         korean_only=korean_only, pre_fa_only=pre_fa_only,
                         fa_seasons=fa_seasons)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚾ KBO → MLB")
side = st.sidebar.radio(
    "Side of the ball", ["batting", "pitching"],
    format_func=lambda s: "Hitters" if s == "batting" else "Pitchers")

if side == "pitching":
    st.sidebar.error(
        "**Pitcher projections are not trustworthy.** Validated out of "
        "sample, the pitcher model is worse than simply assuming league "
        "average. Shown for completeness only.")

season = st.sidebar.number_input("KBO season", 2000, 2030, 2026, step=1)
min_pt = st.sidebar.slider(
    "Minimum playing time", 50, 500, 150, step=25,
    help="Plate appearances (hitters) or batters faced (pitchers) for a "
         "season to count. Rates built on very little playing time are noise.")

st.sidebar.divider()
st.sidebar.caption("**Who counts as a signing target**")

korean_only = st.sidebar.checkbox(
    "Korean players only", value=True,
    help="Foreign imports in the KBO are already professionals from "
         "elsewhere, so they are not an international signing opportunity.")
pre_fa_only = st.sidebar.checkbox(
    "Before first free agency only", value=True,
    help="Once a KBO player reaches free agency and re-signs at home he is "
         "usually 30+ and locked up, and moves to MLB effectively stop. "
         "Listing them would be listing players nobody can buy.")
fa_seasons = st.sidebar.slider(
    "Seasons to domestic free agency", 6, 10,
    app_data.project.KBO_DOMESTIC_FA_SEASONS,
    help="Commonly cited as 8 for high-school entrants and 9 for college "
         "entrants. Adjustable because it is not verified to the same "
         "standard as the posting and bonus-pool rules.")

try:
    bundle = get_bundle(side, int(season), int(min_pt), korean_only,
                        pre_fa_only, int(fa_seasons))
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

roster = bundle.roster()
everyone = bundle.roster(apply_filters=False)
if roster.empty:
    st.warning(
        f"No {side} players in {season} passed every filter. Try lowering "
        f"the playing-time threshold or turning a filter off.")
    st.stop()

removed = len(everyone) - len(roster)
st.sidebar.caption(
    f"**{len(roster)}** signing targets in {season}"
    + (f"  \n({removed} of {len(everyone)} filtered out)" if removed else ""))

tab_player, tab_browse, tab_trust = st.tabs(
    ["Player", "Browse everyone", "How to read this"])


# ---------------------------------------------------------------------------
# One player
# ---------------------------------------------------------------------------

with tab_player:
    labels = {
        f"{r.player}  ({r.team_name}, age {int(r.age)})": r.player_register_id
        for r in roster.itertuples() if pd.notna(r.age)
    }
    choice = st.selectbox("Player", list(labels), index=0)
    info = app_data.project_one(bundle, labels[choice])

    if not info:
        st.warning("Not enough usable seasons to project this player.")
        st.stop()

    st.header(info["name"])
    avail = info["availability"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Age", f"{info['age']:.0f}")
    c2.metric("KBO seasons", info["seasons"])
    c3.metric("Earliest posting", avail["earliest_posting_season"])
    c4.metric("Domestic FA", avail["domestic_fa_season"])
    c5.metric("Age when posted", f"{avail['age_at_earliest_posting']:.0f}")

    # Postable at 7 seasons, a domestic free agent at about 8. The gap
    # between those is the whole opportunity.
    if avail["past_first_fa"]:
        st.error(
            "**Window closed.** He has already reached domestic free "
            "agency. Historically, KBO players who re-sign at home do not "
            "move to MLB afterwards.")
    elif avail["acquisition_window_seasons"] > 0:
        st.info(
            f"**Acquisition window: {avail['window']}** "
            f"({avail['acquisition_window_seasons']} season"
            f"{'s' if avail['acquisition_window_seasons'] != 1 else ''}). "
            f"He can be posted from {avail['earliest_posting_season']} and "
            f"reaches domestic free agency in {avail['domestic_fa_season']}. "
            f"That gap is the whole opportunity.")

    # The single most decision-relevant fact on the page.
    if avail["bonus_pool_exempt_if_posted_then"]:
        st.success(
            f"**Market contract.** Posted in "
            f"{avail['earliest_posting_season']} he would be "
            f"{avail['age_at_earliest_posting']:.0f} with "
            f"{avail['seasons_played'] + avail['seasons_until_posting_eligible']} "
            f"professional seasons, clearing the international bonus pool.")
    else:
        st.warning(
            f"**Bonus-pool capped.** Posted in "
            f"{avail['earliest_posting_season']} he would be "
            f"{avail['age_at_earliest_posting']:.0f} — under the age-25 "
            f"exemption, so he could only sign for a bonus-pool slot, not a "
            f"market contract. Waiting "
            f"{avail['extra_years_for_pool_exemption']:.0f} more year(s) "
            f"changes that entirely.")

    st.subheader("Projected MLB performance")

    proj = info["projection"]
    good = proj[proj["informative"]]
    if good.empty:
        st.error("None of this player's projections are precise enough to be "
                 "worth acting on. The ranges are too wide to distinguish him "
                 "from an average major leaguer.")
    else:
        st.caption(
            f"{len(good)} of {len(proj)} statistics say something specific "
            f"about **this** player. *Range too wide* means we cannot pin a "
            f"number down. *League average regardless* means the model "
            f"barely responds to his KBO numbers, so the projection would "
            f"look much the same for anyone.")

    show = proj.assign(
        Statistic=proj["label"],
        **{
            "vs KBO league": proj["kbo_relative"].map(lambda v: f"{v:.2f}×"),
            "Projected MLB": proj["proj_mlb_rate"].map(lambda v: f"{v:.3f}"),
            "Range (10–90%)": proj.apply(
                lambda r: f"{r['p10_rate']:.3f} – {r['p90_rate']:.3f}", axis=1),
            "Trust": proj["trust"].map({
                "usable": "✓ usable",
                "too wide": "— range too wide",
                "league average regardless": "≈ league average regardless",
            }),
        })[["Statistic", "vs KBO league", "Projected MLB",
            "Range (10–90%)", "Trust"]]

    st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander("KBO record"):
        hist = info["history"]
        cols = [c for c in ("season", "team_name", "age",
                            "PA", "batters_faced", "avg", "obp", "slg",
                            "k_pct", "bb_pct", "era")
                if c in hist.columns]
        st.dataframe(hist[cols].round(3), use_container_width=True,
                     hide_index=True)


# ---------------------------------------------------------------------------
# Everyone
# ---------------------------------------------------------------------------

with tab_browse:
    st.subheader(f"{season} signing targets")
    st.caption(
        "Korean players who have not yet reached domestic free agency. "
        "*Postable* is the first season his club could post him; *FA* is "
        "when he can re-sign at home instead. Open a name on the Player "
        "tab for the full projection.")

    f1, f2 = st.columns(2)
    ages = roster["age"].dropna()
    lo, hi = int(ages.min()), int(ages.max())
    age_range = f1.slider("Age", lo, hi, (lo, hi))
    teams = sorted(roster["team_name"].dropna().unique())
    picked = f2.multiselect("Teams", teams, default=teams)

    view = roster[
        roster["age"].between(*age_range)
        & roster["team_name"].isin(picked)]

    view = view.assign(**{
        "Postable": view["seasons"].map(
            lambda s: bundle.season
            + max(0, app_data.project.SEASONS_FOR_POSTING - s)),
        "FA": view["seasons"].map(
            lambda s: bundle.season + max(0, bundle.fa_seasons - s)),
        "Nationality from": view["nationality_source"],
    })

    cols = [c for c in ("player", "team_name", "age", "seasons", "Postable",
                        "FA", "Nationality from", "PA", "batters_faced")
            if c in view.columns]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.caption(f"{len(view)} players")


# ---------------------------------------------------------------------------
# Honesty tab
# ---------------------------------------------------------------------------

with tab_trust:
    st.subheader("What this tool can and cannot tell you")

    st.markdown("""
This takes what a player did in the KBO, compares it to **his own league in
that season**, and asks what players who made the same move have
historically done in MLB. It is not scouting. It says nothing about a swing,
a delivery, or makeup.

**It was built by checking, not by assuming.** Every statistic was tested
separately, and most of them failed.
""")

    quality = app_data.model_quality(bundle)
    if not quality.empty:
        st.markdown("**Which statistics actually carry across**")
        st.dataframe(
            quality[["statistic", "how much carries over", "explained",
                     "typical range", "pairs", "verdict"]],
            use_container_width=True, hide_index=True)
        st.caption(
            "*How much carries over*: 1.0 would mean a player keeps his edge "
            "over his league intact; 0 means the new league erases it. "
            "*Explained*: how much of the differences **between players** the "
            "model accounts for. *Typical range*: how much wider the high end "
            "of a prediction is than the low end **for one player**. "
            "A statistic can score well on the first and still be too "
            "uncertain to quote for an individual — strikeout rate is exactly "
            "that case, which is why it can read as a real pattern here and "
            "still be flagged *too wide* on a player's page.")

    st.markdown("""
### Three things to hold onto

**Plate discipline travels. Power and pitching do not.** Strikeout and walk
rates carry across with real signal. Home run rate, isolated power and
anything to do with a pitcher's run prevention do not, in this data.

**The samples are small.** Only a few dozen players have meaningful playing
time in both leagues. Wide ranges are the honest answer, not a bug.

**The players who move are not a random sample.** Only KBO stars get posted,
and mostly MLB castoffs go the other way. That selection biases any
translation, and including both directions helps but does not remove it.

### The signing rules used here

A KBO player needs roughly **seven seasons** before his club can post him.
Separately, a foreign professional is exempt from MLB's international bonus
pools only at **25 or older with six or more professional seasons** — below
that he is limited to a bonus slot rather than a market contract.

Both are approximations and should be verified. The Major League labour
agreement expires **1 December 2026** with an international draft under
negotiation, which would change all of it.
""")

    if not bundle.validation.empty:
        with st.expander("Out-of-sample test: players actually posted"):
            st.caption(
                "These players were removed from the fit entirely, then "
                "predicted. This is the only honest measure of accuracy.")
            v = bundle.validation
            cols = [c for c in ("player", "stat", "kbo_season", "mlb_season",
                                "predicted", "actual", "p10", "p90",
                                "in_interval") if c in v.columns]
            st.dataframe(v[cols], use_container_width=True, hide_index=True)
