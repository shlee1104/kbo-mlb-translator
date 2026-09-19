"""Who is a posted KBO player, and who is a returning import?

The distinction matters more than it first appears. Both groups show up in
the data as "players with KBO and MLB seasons", but they arrive there by
opposite routes:

* **Posted from the KBO.** Started professionally in Korea and earned an MLB
  contract. Ryu, Kang, Hyun-soo Kim, Ha-seong Kim, Jung-hoo Lee. These are
  the top of the KBO, selected for being stars. They are also the only group
  a club actually wants a projection for.

* **Returning import.** Started in MLB (or its minors), went to the KBO when
  the MLB work dried up, and came back. Merrill Kelly, Erick Fedde, Chris
  Flexen. These are selected for having been *available*, which is close to
  the opposite filter.

Pooling the two without distinguishing them lets one selection process
contaminate the estimate of the other. The classification is derived from
the data rather than from nationality: a player's first professional season
tells you which route he took. That also avoids the trap that Jung-hoo Lee
was born in Nagoya and would be missed by any birthplace filter.
"""

from __future__ import annotations

import pandas as pd

POSTED = "posted_from_kbo"
IMPORT = "import_returnee"
UNKNOWN = "unknown"


def classify(
    kbo_seasons: pd.DataFrame,
    mlb_seasons: pd.DataFrame,
    player_key: str = "player_register_id",
) -> pd.DataFrame:
    """Label every two-league player by the route he took.

    Returns one row per player with first/last season in each league and a
    `cohort` label.
    """
    def _span(df, label):
        if df.empty:
            return pd.DataFrame(columns=[player_key, f"first_{label}",
                                         f"last_{label}", f"n_{label}"])
        g = df.copy()
        g["season"] = pd.to_numeric(g["season"], errors="coerce")
        g = g.dropna(subset=["season", player_key])
        out = g.groupby(player_key)["season"].agg(["min", "max", "nunique"])
        out.columns = [f"first_{label}", f"last_{label}", f"n_{label}"]
        return out.reset_index()

    kbo = _span(kbo_seasons, "kbo")
    mlb = _span(mlb_seasons, "mlb")
    both = kbo.merge(mlb, on=player_key, how="outer")

    def _label(row):
        fk, fm = row.get("first_kbo"), row.get("first_mlb")
        if pd.isna(fk) or pd.isna(fm):
            return UNKNOWN
        if fk < fm:
            return POSTED
        if fm < fk:
            return IMPORT
        # Same year in both leagues: a midseason move. Rare, and ambiguous
        # enough that guessing would be worse than declining to.
        return UNKNOWN

    both["cohort"] = both.apply(_label, axis=1)
    return both


def attach(pairs: pd.DataFrame, cohorts: pd.DataFrame,
           player_key: str = "player_register_id") -> pd.DataFrame:
    """Add the cohort label to a table of season pairs."""
    cols = [player_key, "cohort", "first_kbo", "first_mlb", "n_kbo", "n_mlb"]
    have = [c for c in cols if c in cohorts.columns]
    return pairs.merge(cohorts[have], on=player_key, how="left")


def summary(cohorts: pd.DataFrame) -> pd.DataFrame:
    return (cohorts["cohort"].value_counts()
            .rename_axis("cohort").reset_index(name="players"))
