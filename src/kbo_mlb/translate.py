"""The translation model: what a KBO rate becomes in MLB.

## The shape of the model

One model per statistic, fitted on players who crossed between the leagues.
Everything is in **league-relative** terms (see `rates.py`), so a value of
1.0 is exactly league average and the model never has to care that the KBO
deadened its ball in 2019.

For each statistic s:

    log(MLB_relative_s) = a + b * log(KBO_relative_s)
                            + c * (age - 27) / 10
                            + d * moved_kbo_to_mlb

Why logs: these are ratios, bounded below by zero and roughly multiplicative.
A player 20% above league average and one 20% below should be treated
symmetrically, which logs do and raw differences do not. It also guarantees
the prediction stays positive.

Why the slope `b` matters more than the intercept: `b` is the **retention
rate** of a skill. b near 1 means a player keeps his edge intact; b near 0
means the league erases it and everyone regresses to average. The whole
point of fitting each statistic separately is that b is genuinely different
for strikeout rate than for batting average on balls in play, and a single
blanket "KBO discount" hides that.

Why `d`, the direction term: a player posted from the KBO to MLB is selected
for being a KBO star. A player going the other way is usually an MLB
organisation's castoff. Those are two different selection processes acting
on the same underlying talent gap, so pooling them without a flag would let
one contaminate the other.

## What this model is not

It is not causal and it is not scouting. It answers "players who looked like
this in the KBO have historically performed like that in MLB", on a sample
that is small and selected. The intervals are wide on purpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import rates

log = logging.getLogger(__name__)

AGE_CENTRE = 27.0     # roughly a hitter's peak; keeps the intercept meaningful
MIN_PAIRS_TO_FIT = 20


# ---------------------------------------------------------------------------
# Pairing seasons across the two leagues
# ---------------------------------------------------------------------------

def build_pairs(
    kbo: pd.DataFrame,
    mlb: pd.DataFrame,
    player_key: str = "player_register_id",
    max_gap_years: int = 2,
) -> pd.DataFrame:
    """Find each player's league *crossings* and pair the seasons either side.

    A crossing is a point in a player's timeline where consecutive seasons
    sit in different leagues. We pair the last season before it with the
    first season after it.

    This is deliberately stricter than "every KBO season against every MLB
    season". Pairing all combinations would let one good MLB year be
    explained by five different KBO years, inflating the apparent sample and
    making every interval too narrow.

    Both frames need: `player_key`, `season`, and the `rel_*` columns.
    """
    k = kbo.copy()
    m = mlb.copy()
    k["league"] = "KBO"
    m["league"] = "MLB"

    common = ["league", "season", player_key]
    rel_cols = sorted(set(c for c in k.columns if c.startswith("rel_"))
                      & set(c for c in m.columns if c.startswith("rel_")))
    extra = [c for c in ("age", "PA", "batters_faced", "player", "name")
             if c in k.columns or c in m.columns]

    def _slim(df):
        cols = common + rel_cols + [c for c in extra if c in df.columns]
        out = df[cols].copy()
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
        return out.dropna(subset=["season", player_key])

    timeline = pd.concat([_slim(k), _slim(m)], ignore_index=True)
    timeline = timeline.sort_values([player_key, "season", "league"])

    pairs: list[dict] = []
    for pid, group in timeline.groupby(player_key, sort=False):
        # One row per player-season-league; a player can't be in both at once,
        # but a midseason trade can duplicate a KBO season, so collapse first.
        group = (group.sort_values("season")
                      .drop_duplicates(subset=["season", "league"],
                                       keep="last")
                      .reset_index(drop=True))
        for i in range(len(group) - 1):
            a, b = group.iloc[i], group.iloc[i + 1]
            if a["league"] == b["league"]:
                continue
            gap = int(b["season"] - a["season"])
            if gap < 0 or gap > max_gap_years:
                continue

            kbo_row, mlb_row = (a, b) if a["league"] == "KBO" else (b, a)
            record = {
                player_key: pid,
                "direction": ("kbo_to_mlb" if a["league"] == "KBO"
                              else "mlb_to_kbo"),
                "gap_years": gap,
                "kbo_season": int(kbo_row["season"]),
                "mlb_season": int(mlb_row["season"]),
                "player": kbo_row.get("player") or mlb_row.get("name"),
            }
            for c in rel_cols:
                record[f"kbo_{c}"] = kbo_row.get(c)
                record[f"mlb_{c}"] = mlb_row.get(c)
            for c in ("age", "PA", "batters_faced"):
                if c in kbo_row.index:
                    record[f"kbo_{c}"] = kbo_row.get(c)
                if c in mlb_row.index:
                    record[f"mlb_{c}"] = mlb_row.get(c)
            pairs.append(record)

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class StatFit:
    stat: str
    intercept: float
    slope: float                 # skill retention
    age_coef: float
    direction_coef: float
    n_pairs: int
    n_players: int
    residual_sd: float
    r_squared: float

    def as_dict(self) -> dict:
        return {
            "stat": self.stat,
            "retention_slope": round(self.slope, 3),
            "intercept": round(self.intercept, 3),
            "level_ratio": round(float(np.exp(self.intercept)), 3),
            "age_coef": round(self.age_coef, 3),
            "direction_coef": round(self.direction_coef, 3),
            "n_pairs": self.n_pairs,
            "n_players": self.n_players,
            "residual_sd": round(self.residual_sd, 3),
            "r_squared": round(self.r_squared, 3),
        }


@dataclass
class TranslationModel:
    side: str
    fits: dict[str, StatFit] = field(default_factory=dict)
    stats_used: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_dict() for f in self.fits.values()])


def _design(pairs: pd.DataFrame, stat: str) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, pd.DataFrame]:
    """Build (X, y, weights, usable_rows) for one statistic."""
    kcol, mcol = f"kbo_rel_{stat}", f"mlb_rel_{stat}"
    if kcol not in pairs.columns or mcol not in pairs.columns:
        return np.empty((0, 4)), np.empty(0), np.empty(0), pairs.iloc[0:0]

    df = pairs.copy()
    df[kcol] = pd.to_numeric(df[kcol], errors="coerce")
    df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
    # Ratios must be strictly positive to take logs. A zero means the player
    # genuinely never did the thing (no home runs), which carries no
    # information about *rate* and would otherwise become -inf.
    df = df[(df[kcol] > 0) & (df[mcol] > 0)].copy()
    if df.empty:
        return np.empty((0, 4)), np.empty(0), np.empty(0), df

    age = pd.to_numeric(df.get("kbo_age"), errors="coerce")
    age = age.fillna(AGE_CENTRE) if age is not None else pd.Series(
        AGE_CENTRE, index=df.index)

    X = np.column_stack([
        np.ones(len(df)),
        np.log(df[kcol].to_numpy()),
        (age.to_numpy() - AGE_CENTRE) / 10.0,
        (df["direction"] == "kbo_to_mlb").astype(float).to_numpy(),
    ])
    y = np.log(df[mcol].to_numpy())

    # A pair where both seasons carry real playing time says more than one
    # built on a September call-up.
    pt_cols = [c for c in (f"kbo_{rates.PLAYING_TIME['batting']}",
                           f"kbo_{rates.PLAYING_TIME['pitching']}",
                           f"mlb_{rates.PLAYING_TIME['batting']}",
                           f"mlb_{rates.PLAYING_TIME['pitching']}")
               if c in df.columns]
    if pt_cols:
        pt = df[pt_cols].apply(pd.to_numeric, errors="coerce")
        w = np.sqrt(pt.min(axis=1).fillna(1.0).clip(lower=1.0).to_numpy())
    else:
        w = np.ones(len(df))

    return X, y, w, df


def fit(pairs: pd.DataFrame, side: str,
        stats: list[str] | None = None) -> TranslationModel:
    """Fit one weighted least-squares translation per statistic."""
    stats = stats or list(rates.RATE_SETS[side])
    model = TranslationModel(side=side)

    for stat in stats:
        X, y, w, used = _design(pairs, stat)
        if len(y) < MIN_PAIRS_TO_FIT:
            log.info("skipping %s: only %d usable pairs", stat, len(y))
            continue

        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ beta
        ss_res = float(np.sum(w * resid**2))
        ss_tot = float(np.sum(w * (y - np.average(y, weights=w))**2))
        dof = max(len(y) - X.shape[1], 1)

        model.fits[stat] = StatFit(
            stat=stat,
            intercept=float(beta[0]),
            slope=float(beta[1]),
            age_coef=float(beta[2]),
            direction_coef=float(beta[3]),
            n_pairs=len(y),
            n_players=int(used["player_register_id"].nunique())
            if "player_register_id" in used.columns else len(y),
            residual_sd=float(np.sqrt(ss_res / dof)),
            r_squared=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        )

    model.stats_used = list(model.fits)
    return model


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------

def predict_relative(model: TranslationModel, stat: str,
                     kbo_relative: float, age: float,
                     direction: str = "kbo_to_mlb") -> float:
    """Point estimate of the MLB league-relative rate."""
    f = model.fits[stat]
    log_pred = (f.intercept
                + f.slope * np.log(kbo_relative)
                + f.age_coef * (age - AGE_CENTRE) / 10.0
                + f.direction_coef * (1.0 if direction == "kbo_to_mlb" else 0.0))
    return float(np.exp(log_pred))


@dataclass
class BootstrapDraws:
    """Coefficient vectors from refitting on resampled players."""
    stat: str
    betas: np.ndarray        # (n_draws, 4)
    residual_sds: np.ndarray  # (n_draws,)

    def __len__(self) -> int:
        return len(self.residual_sds)


def bootstrap_coefficients(
    pairs: pd.DataFrame,
    side: str,
    stat: str,
    n_boot: int = 500,
    seed: int = 0,
    player_key: str = "player_register_id",
) -> BootstrapDraws | None:
    """Refit the model on player-resampled data, `n_boot` times.

    Resampling is by **player**, not by row. One player can contribute
    several pairs, and those are not independent observations of anything;
    resampling rows would treat them as if they were and produce intervals
    that are far too narrow.

    The design matrix is built once and rows are gathered by index on each
    draw. Refitting from the DataFrame every time instead - which is what
    this did first - made a single validation run take longer than the
    entire data pull.
    """
    X, y, w, used = _design(pairs, stat)
    if len(y) < MIN_PAIRS_TO_FIT:
        return None

    if player_key in used.columns:
        codes, _ = pd.factorize(used[player_key])
    else:
        codes = np.arange(len(used))
    n_players = int(codes.max()) + 1
    rows_by_player = [np.flatnonzero(codes == p) for p in range(n_players)]

    rng = np.random.default_rng(seed)
    betas: list[np.ndarray] = []
    sds: list[float] = []

    for _ in range(n_boot):
        pick = rng.integers(0, n_players, size=n_players)
        idx = np.concatenate([rows_by_player[p] for p in pick])
        if len(idx) <= X.shape[1]:
            continue
        Xb, yb, wb = X[idx], y[idx], w[idx]
        sw = np.sqrt(wb)
        try:
            beta, *_ = np.linalg.lstsq(Xb * sw[:, None], yb * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yb - Xb @ beta
        dof = max(len(yb) - Xb.shape[1], 1)
        betas.append(beta)
        sds.append(float(np.sqrt(np.sum(wb * resid**2) / dof)))

    if not betas:
        return None
    return BootstrapDraws(stat, np.vstack(betas), np.array(sds))


def inversion_diagnostic(pairs: pd.DataFrame, stat: str) -> dict:
    """Show why a fitted slope must not be algebraically inverted.

    For a simple regression, slope(y~x) * slope(x~y) = R-squared. So when
    the fit is weak, the two directions are wildly inconsistent, and
    flipping 1/slope overshoots by roughly 1/R-squared.

    Concretely: if predicting MLB from KBO gives a slope of 0.1 with an
    R-squared of 0.1, then the honest reverse slope is 1.0, not 10.

    The lesson is that the direction you intend to *predict* is the
    direction you must *fit*. This function exists so that the repo can
    demonstrate that rather than assert it.
    """
    kcol, mcol = f"kbo_rel_{stat}", f"mlb_rel_{stat}"
    df = pairs[[kcol, mcol]].apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df[kcol] > 0) & (df[mcol] > 0)]
    if len(df) < MIN_PAIRS_TO_FIT:
        return {}

    x = np.log(df[kcol].to_numpy())
    y = np.log(df[mcol].to_numpy())
    r = float(np.corrcoef(x, y)[0, 1])

    b_mlb_on_kbo = float(np.polyfit(x, y, 1)[0])   # predict MLB from KBO
    b_kbo_on_mlb = float(np.polyfit(y, x, 1)[0])   # predict KBO from MLB

    return {
        "stat": stat,
        "n": len(df),
        "r_squared": round(r**2, 3),
        "slope_mlb_on_kbo": round(b_mlb_on_kbo, 3),
        "slope_kbo_on_mlb": round(b_kbo_on_mlb, 3),
        "naive_inverted_slope": (round(1.0 / b_kbo_on_mlb, 3)
                                 if abs(b_kbo_on_mlb) > 1e-9 else float("inf")),
        "slope_product_equals_r2": round(b_mlb_on_kbo * b_kbo_on_mlb, 3),
    }


def _row(kbo_relative: float, age: float, direction: str) -> np.ndarray:
    return np.array([
        1.0,
        np.log(kbo_relative),
        (age - AGE_CENTRE) / 10.0,
        1.0 if direction == "kbo_to_mlb" else 0.0,
    ])


def interval_from_draws(
    draws: BootstrapDraws,
    kbo_relative: float,
    age: float,
    direction: str = "kbo_to_mlb",
    seed: int = 0,
) -> dict:
    """Turn bootstrap coefficients into a prediction interval for one player.

    Covers uncertainty in the fitted relationship *plus* the residual spread
    of individual players around it, which is what you want when asking
    "what might this player do" rather than "where is the average line".
    """
    x = _row(kbo_relative, age, direction)
    centres = draws.betas @ x
    rng = np.random.default_rng(seed)
    sample = np.exp(centres + rng.normal(0.0, draws.residual_sds))

    return {
        "point": float(np.exp(np.mean(centres))),
        "p10": float(np.percentile(sample, 10)),
        "p25": float(np.percentile(sample, 25)),
        "p50": float(np.percentile(sample, 50)),
        "p75": float(np.percentile(sample, 75)),
        "p90": float(np.percentile(sample, 90)),
        "n_draws": len(sample),
    }


def bootstrap_predict(
    pairs: pd.DataFrame,
    side: str,
    stat: str,
    kbo_relative: float,
    age: float,
    direction: str = "kbo_to_mlb",
    n_boot: int = 500,
    seed: int = 0,
    player_key: str = "player_register_id",
) -> dict:
    """Convenience wrapper: resample and predict one player in one call.

    When predicting many players from the same fit, call
    `bootstrap_coefficients` once and `interval_from_draws` per player
    instead - the resampling is the expensive part and it does not depend
    on which player you are asking about.
    """
    draws = bootstrap_coefficients(pairs, side, stat, n_boot, seed, player_key)
    if draws is None:
        return {}
    return interval_from_draws(draws, kbo_relative, age, direction, seed)


def to_absolute(relative: float, league_rate: float) -> float:
    """Turn a league-relative prediction back into a real rate."""
    return float(relative * league_rate)
