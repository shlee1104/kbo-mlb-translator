"""Out-of-sample validation: fit without the posted Koreans, then predict them.

A translation model fitted on 476 players will always look good when scored
on those same 476 players. The question a club cares about is different:
*given only what the KBO showed, would this model have called the guy we
actually signed?*

So the posted cohort is removed from the fit entirely - not down-weighted,
not cross-validated fold by fold, removed - and then predicted. Those
players are the only ones whose situation matches the forward question, and
they are the ones whose names a reader recognises.

Reported per statistic:
  * predicted vs actual league-relative rate for each held-out player
  * the share of players whose actual result landed inside the 10th-90th
    percentile interval, which is the interval's honesty check: if the
    model is calibrated, about 80% should
  * mean absolute error, against a baseline of "assume league average",
    because a model that cannot beat that baseline is not worth running
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import cohorts, translate

log = logging.getLogger(__name__)


def holdout_validate(
    pairs: pd.DataFrame,
    side: str,
    holdout_cohort: str = cohorts.POSTED,
    stats: list[str] | None = None,
    n_boot: int = 200,
    player_key: str = "player_register_id",
    train_direction: str | None = None,
) -> pd.DataFrame:
    """Fit without `holdout_cohort`, predict it, return one row per case.

    `train_direction` restricts which league crossings the fit may learn
    from:

      None          both directions, with a direction indicator
      "kbo_to_mlb"  only genuine forward moves - the right direction, but
                    a very small sample
      "mlb_to_kbo"  only MLB-to-KBO moves. Time runs backwards in these
                    pairs, so using them assumes the talent mapping between
                    the leagues is stable in both directions. That
                    assumption is exactly what validating on the held-out
                    posted players tests.

    Note that the target is always the MLB season and the predictor is
    always the KBO season, whichever direction the player travelled. That
    is deliberate: you fit the direction you intend to predict. Fitting
    KBO-from-MLB and algebraically inverting the slope would be wrong (see
    translate.inversion_diagnostic).
    """
    if "cohort" not in pairs.columns:
        raise ValueError("pairs must carry a `cohort` column; use cohorts.attach")

    train = pairs[pairs["cohort"] != holdout_cohort]
    if train_direction:
        train = train[train["direction"] == train_direction]
    test = pairs[(pairs["cohort"] == holdout_cohort)
                 & (pairs["direction"] == "kbo_to_mlb")]

    if train.empty or test.empty:
        log.warning("nothing to validate: %d train, %d test pairs",
                    len(train), len(test))
        return pd.DataFrame()

    model = translate.fit(train, side, stats=stats)

    # Resample once per statistic, not once per player. The resampling does
    # not depend on who is being predicted, and doing it per case turned a
    # validation run into something slower than the entire data pull.
    draws = {}
    for stat in model.fits:
        d = translate.bootstrap_coefficients(
            train, side, stat, n_boot=n_boot, player_key=player_key)
        if d is not None:
            draws[stat] = d

    rows: list[dict] = []

    for _, case in test.iterrows():
        for stat in model.fits:
            kcol, mcol = f"kbo_rel_{stat}", f"mlb_rel_{stat}"
            kbo_rel = pd.to_numeric(case.get(kcol), errors="coerce")
            actual = pd.to_numeric(case.get(mcol), errors="coerce")
            if not (kbo_rel and kbo_rel > 0) or pd.isna(actual):
                continue

            age = pd.to_numeric(case.get("kbo_age"), errors="coerce")
            age = translate.AGE_CENTRE if pd.isna(age) else float(age)

            point = translate.predict_relative(model, stat, float(kbo_rel),
                                               age, "kbo_to_mlb")
            interval = (translate.interval_from_draws(
                draws[stat], float(kbo_rel), age, "kbo_to_mlb")
                if stat in draws else {})

            rows.append({
                "player": case.get("player"),
                player_key: case.get(player_key),
                "stat": stat,
                "kbo_season": case.get("kbo_season"),
                "mlb_season": case.get("mlb_season"),
                "age": age,
                "kbo_relative": round(float(kbo_rel), 3),
                "predicted": round(float(point), 3),
                "actual": round(float(actual), 3),
                "p10": round(interval.get("p10", np.nan), 3),
                "p90": round(interval.get("p90", np.nan), 3),
                "in_interval": (
                    bool(interval.get("p10", np.nan) <= actual
                         <= interval.get("p90", np.nan))
                    if interval else None),
            })

    return pd.DataFrame(rows)


def scorecard(results: pd.DataFrame) -> pd.DataFrame:
    """Per-statistic accuracy, against the do-nothing baseline.

    The baseline is "predict league average" (a relative rate of 1.0). It is
    deliberately unflattering: beating it is the minimum bar for the model
    to have earned its existence.
    """
    if results.empty:
        return pd.DataFrame()

    out = []
    for stat, g in results.groupby("stat"):
        err = (g["predicted"] - g["actual"]).abs()
        naive = (1.0 - g["actual"]).abs()
        # "Assume he stays exactly as good relative to his league" - the
        # other obvious baseline, and a harder one.
        carry = (g["kbo_relative"] - g["actual"]).abs()
        covered = g["in_interval"].dropna()
        out.append({
            "stat": stat,
            "n": len(g),
            "mae_model": round(float(err.mean()), 3),
            "mae_league_average": round(float(naive.mean()), 3),
            "mae_carry_over": round(float(carry.mean()), 3),
            "beats_league_average": bool(err.mean() < naive.mean()),
            "beats_carry_over": bool(err.mean() < carry.mean()),
            "interval_coverage": (round(float(covered.mean()), 3)
                                  if len(covered) else None),
        })
    return pd.DataFrame(out).sort_values("stat").reset_index(drop=True)


def to_markdown(results: pd.DataFrame, scores: pd.DataFrame,
                model: translate.TranslationModel | None = None) -> str:
    lines = [
        "# Out-of-sample validation",
        "",
        "Players posted from the KBO were removed from the fit entirely, "
        "then predicted. Every number below is out of sample.",
        "",
    ]

    if model is not None:
        lines += [
            "## Fitted translations",
            "",
            "`retention_slope` is what share of a player's edge over his own "
            "league survives the move. 1.0 means it carries across intact; "
            "0 means the league erases it. `level_ratio` is the overall "
            "shift for a league-average player.",
            "",
            model.summary().to_markdown(index=False),
            "",
        ]

    if not scores.empty:
        lines += [
            "## Accuracy against baselines",
            "",
            "`mae_league_average` assumes every player becomes exactly "
            "league average. `mae_carry_over` assumes he keeps his KBO "
            "rate unchanged. The model has to beat both to be useful.",
            "",
            scores.to_markdown(index=False),
            "",
        ]

    if not results.empty:
        lines += ["## Held-out players", ""]
        for player, g in results.groupby("player"):
            lines.append(f"### {player}")
            lines.append("")
            cols = ["stat", "kbo_season", "mlb_season", "kbo_relative",
                    "predicted", "actual", "p10", "p90", "in_interval"]
            cols = [c for c in cols if c in g.columns]
            lines.append(g[cols].to_markdown(index=False))
            lines.append("")

    return "\n".join(lines)
