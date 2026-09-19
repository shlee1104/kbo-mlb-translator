"""Everything the Streamlit app needs, with no Streamlit in it.

The UI layer is deliberately thin and this layer holds the logic, for one
practical reason: Streamlit cannot be imported in a test runner without a
browser session, so anything living inside `app.py` is effectively
untestable. Keeping the loading, fitting and projecting here means the part
that can be wrong is the part that is covered by tests.

It also means the app starts fast. Loading and fitting happen once and are
cached by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import cohorts, config, names, project, rates, translate

# Statistics a scout actually asks about, in the order they'd ask.
DISPLAY_ORDER = ["avg", "obp", "slg", "iso", "hr_pct", "k_pct", "bb_pct",
                 "babip", "era", "h_pct"]

PRETTY = {
    "avg": "Batting average",
    "obp": "On-base percentage",
    "slg": "Slugging",
    "iso": "Isolated power",
    "hr_pct": "Home run rate",
    "k_pct": "Strikeout rate",
    "bb_pct": "Walk rate",
    "babip": "Average on balls in play",
    "era": "Earned run average",
    "h_pct": "Hit rate allowed",
}

# Rates where a LOWER number is better, which flips how a projection reads.
LOWER_IS_BETTER = {"k_pct", "era", "h_pct"}


@dataclass
class Bundle:
    """Everything loaded and fitted, ready to project from."""
    side: str
    season: int
    kbo: pd.DataFrame           # league-relative KBO player-seasons
    kbo_raw: pd.DataFrame       # unfiltered, for service-time counts
    mlb_league: pd.DataFrame
    model: translate.TranslationModel
    draws: dict
    train_pairs: pd.DataFrame
    validation: pd.DataFrame    # out-of-sample results, may be empty

    @property
    def seasons_by_player(self) -> pd.Series:
        return (self.kbo_raw.dropna(subset=["player_register_id"])
                .groupby("player_register_id")["season"].nunique())

    def roster(self) -> pd.DataFrame:
        """One row per player active in `season`, for the picker."""
        cur = self.kbo[self.kbo["season"] == self.season]
        pt = rates.PLAYING_TIME[self.side]
        cols = [c for c in ("player_register_id", "player", "team_name",
                            "age", pt) if c in cur.columns]
        out = (cur[cols].sort_values(pt, ascending=False)
               .drop_duplicates("player_register_id"))
        out["seasons"] = out["player_register_id"].map(
            self.seasons_by_player).fillna(1).astype(int)
        return out.reset_index(drop=True)


def _interim(name: str) -> pd.DataFrame:
    path = config.INTERIM_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load(
    side: str = "batting",
    season: int = 2026,
    min_playing_time: int = 150,
    train_direction: str | None = "mlb_to_kbo",
    n_boot: int = 400,
    source_suffix: str = "",
) -> Bundle:
    """Load the data, fit the translation, and prepare bootstrap draws.

    Raises FileNotFoundError with a plain-language message when the pipeline
    has not been run, because "KeyError: season" is not a useful thing to
    show someone who just opened a web page.
    """
    kbo_raw = _interim(f"kbo_{side}.csv")
    mlb_raw = _interim(f"mlb_{side}{source_suffix}.csv")
    kbo_totals = _interim("kbo_league_totals.csv")
    mlb_totals = _interim(f"mlb_league_totals{source_suffix}.csv")

    missing = [n for n, d in (
        (f"kbo_{side}.csv", kbo_raw), (f"mlb_{side}.csv", mlb_raw),
        ("kbo_league_totals.csv", kbo_totals),
        ("mlb_league_totals.csv", mlb_totals)) if d.empty]
    if missing:
        raise FileNotFoundError(
            "The pipeline has not been run yet. Missing: "
            + ", ".join(missing)
            + ".\nRun:  python -m kbo_mlb.cli all   then   cli mlb")

    crosswalk_path = config.PROCESSED_DIR / "crosswalk.csv"
    if not crosswalk_path.exists():
        raise FileNotFoundError("Missing crosswalk.csv - run `cli crosswalk`.")
    cw = pd.read_csv(crosswalk_path, low_memory=False)
    id_map = cw[["player_register_id", "key_mlbam"]].dropna().drop_duplicates()
    id_map["key_mlbam"] = id_map["key_mlbam"].astype(int)

    mlb = mlb_raw.copy()
    mlb["key_mlbam"] = pd.to_numeric(mlb["key_mlbam"],
                                     errors="coerce").astype("Int64")
    mlb = mlb.merge(id_map, on="key_mlbam", how="inner")

    kbo_rel = rates.add_relative_rates(
        kbo_raw, rates.league_rates(kbo_totals, side), side,
        min_playing_time=min_playing_time)
    mlb_rel = rates.add_relative_rates(
        mlb, rates.league_rates(mlb_totals, side), side,
        min_playing_time=min_playing_time)

    pairs = translate.build_pairs(kbo_rel, mlb_rel)
    pairs = cohorts.attach(pairs, cohorts.classify(kbo_rel, mlb_rel))

    train = pairs[pairs["cohort"] != cohorts.POSTED]
    if train_direction:
        train = train[train["direction"] == train_direction]

    model = translate.fit(train, side)
    draws = {}
    for stat in model.fits:
        d = translate.bootstrap_coefficients(train, side, stat, n_boot=n_boot)
        if d is not None:
            draws[stat] = d

    val_path = config.PROCESSED_DIR / f"holdout_{side}.csv"
    validation = (pd.read_csv(val_path, low_memory=False)
                  if val_path.exists() else pd.DataFrame())

    return Bundle(side=side, season=season, kbo=kbo_rel, kbo_raw=kbo_raw,
                  mlb_league=rates.league_rates(mlb_totals, side),
                  model=model, draws=draws, train_pairs=train,
                  validation=validation)


def find_player(bundle: Bundle, query: str) -> pd.DataFrame:
    """Rows for a player, matched by the phonetic key so spelling is free.

    A scout typing "Ha Seong Kim" should find 김하성 without knowing which
    romanisation the source happened to use.
    """
    key = names.loose_key(query)
    df = bundle.kbo.copy()
    df["_k"] = df["player"].fillna("").map(names.loose_key)
    return df[df["_k"] == key].sort_values("season")


def project_one(bundle: Bundle, player_register_id: str) -> dict:
    """Projection plus availability for one player."""
    rows = bundle.kbo[
        bundle.kbo["player_register_id"] == player_register_id].sort_values(
        "season")
    if rows.empty:
        return {}

    proj = project.project_player(rows, bundle.model, bundle.draws,
                                  bundle.mlb_league, translate)
    if proj.empty:
        return {}

    # A narrow band is not enough on its own. If the model barely responds
    # to the player's own input (low R-squared), the "projection" is just
    # league average wearing a tight interval, and presenting that as a
    # statement about *this* player would be misleading.
    def _trust(row) -> str:
        fit = bundle.model.fits.get(row["stat"])
        narrow = project.interval_is_informative(row)
        has_signal = bool(fit and fit.r_squared >= 0.10)
        if narrow and has_signal:
            return "usable"
        if not narrow:
            return "too wide"
        return "league average regardless"

    proj["trust"] = proj.apply(_trust, axis=1)
    proj["informative"] = proj["trust"] == "usable"
    proj["label"] = proj["stat"].map(PRETTY).fillna(proj["stat"])
    proj["lower_is_better"] = proj["stat"].isin(LOWER_IS_BETTER)
    order = {s: i for i, s in enumerate(DISPLAY_ORDER)}
    proj = proj.sort_values("stat", key=lambda c: c.map(order).fillna(99))

    seasons = int(bundle.seasons_by_player.get(player_register_id,
                                               rows["season"].nunique()))
    age = float(pd.to_numeric(rows["age"], errors="coerce").iloc[-1])

    return {
        "name": rows["player"].iloc[-1],
        "age": age,
        "team": rows.get("team_name", pd.Series([""])).iloc[-1],
        "seasons": seasons,
        "history": rows,
        "projection": proj,
        "availability": project.availability(seasons, age, bundle.season),
    }


def typical_range_factor(residual_sd: float) -> float:
    """How many times wider the 90th percentile is than the 10th.

    The residual spread is in log space, so a 10th-90th band is
    plus/minus 1.2816 standard deviations and the ratio between the ends is
    exp(2 * 1.2816 * sd). A factor of 3 means the high end is triple the
    low end, which is about the point where a projection stops being worth
    quoting for an individual.
    """
    import math
    return float(math.exp(2 * 1.2816 * residual_sd))


def model_quality(bundle: Bundle) -> pd.DataFrame:
    """A plain-language read on which statistics can be trusted.

    Two different questions get two different columns, because conflating
    them is actively misleading:

      *explained* - how much of the variation BETWEEN players the model
      accounts for. High means the stat genuinely translates.

      *typical range* - how wide a prediction is for ONE player. A stat can
      translate well on average and still be too uncertain to quote for an
      individual, which is exactly the case for strikeout rate here.
    """
    if not bundle.model.fits:
        return pd.DataFrame()

    rows = []
    for stat, fit in bundle.model.fits.items():
        band = typical_range_factor(fit.residual_sd)
        rows.append({
            "statistic": PRETTY.get(stat, stat),
            "stat": stat,
            "how much carries over": round(fit.slope, 2),
            "explained": round(fit.r_squared, 3),
            "typical range": f"{band:.1f}x",
            "pairs": fit.n_pairs,
            "verdict": _verdict(fit.slope, fit.r_squared, fit.residual_sd),
        })
    return pd.DataFrame(rows)


def _verdict(slope: float, r2: float, residual_sd: float | None = None,
             max_band: float = 3.0) -> str:
    """Combine "does it translate" with "can we say it about one player"."""
    band = typical_range_factor(residual_sd) if residual_sd else None

    if r2 >= 0.25 and slope >= 0.3:
        if band is not None and band > max_band:
            return "real pattern, too wide for one player"
        return "usable signal"
    if r2 >= 0.10:
        return "weak signal"
    return "no usable signal"
