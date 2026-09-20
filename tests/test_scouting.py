"""Tests for the interest screen.

The screen is the one part of this project that draws a line through a list
of real people, so the things worth testing are the ways a line like that
goes quietly wrong: a bar built only from the players who succeeded, a bar
that admits everyone, a weighting that lets a 12-plate-appearance September
outweigh a full season.
"""

import unittest

import numpy as np
import pandas as pd

from kbo_mlb import scouting


def kbo_rows(rows):
    """Minimal league-relative KBO frame: the columns `score` reads."""
    return pd.DataFrame(rows, columns=[
        "player_register_id", "player", "season", "age", "PA",
        "batters_faced", "obp", "slg", "lg_obp", "lg_slg", "rel_k_pct"])


class TestScore(unittest.TestCase):
    def test_batting_score_is_ops_over_league_ops(self):
        df = kbo_rows([("a", "A", 2025, 22, 500, np.nan,
                        0.400, 0.600, 0.350, 0.400, np.nan)])
        # (.400+.600) / (.350+.400) = 1.3333
        self.assertAlmostEqual(scouting.score(df, "batting"), 1.3333, places=3)

    def test_league_average_batter_scores_one(self):
        df = kbo_rows([("a", "A", 2025, 27, 400, np.nan,
                        0.340, 0.430, 0.340, 0.430, np.nan)])
        self.assertAlmostEqual(scouting.score(df, "batting"), 1.0, places=6)

    def test_seasons_are_weighted_by_playing_time(self):
        # A 600-PA league-average year and a 60-PA monster year must not
        # count equally. Averaging the two seasons flat would return 1.25;
        # weighting by plate appearances lands near 1.05.
        df = kbo_rows([
            ("a", "A", 2024, 26, 600, np.nan, 0.340, 0.430, 0.340, 0.430, np.nan),
            ("a", "A", 2025, 27, 60, np.nan, 0.500, 0.700, 0.340, 0.430, np.nan),
        ])
        got = scouting.score(df, "batting")
        self.assertLess(got, 1.10)
        self.assertGreater(got, 1.00)

    def test_pitching_score_uses_relative_strikeout_rate(self):
        df = kbo_rows([("p", "P", 2025, 24, np.nan, 700,
                        np.nan, np.nan, np.nan, np.nan, 1.40)])
        self.assertAlmostEqual(scouting.score(df, "pitching"), 1.40, places=6)

    def test_empty_history_is_not_a_zero(self):
        # Returning 0.0 for "no data" would silently rank an unknown player
        # below every known one instead of leaving him unscored.
        self.assertTrue(np.isnan(scouting.score(kbo_rows([]), "batting")))

    def test_missing_columns_do_not_raise(self):
        self.assertTrue(np.isnan(
            scouting.score(pd.DataFrame({"PA": [400]}), "batting")))

    def test_zero_playing_time_falls_back_to_equal_weights(self):
        df = kbo_rows([("a", "A", 2025, 22, 0, np.nan,
                        0.400, 0.600, 0.350, 0.400, np.nan)])
        self.assertAlmostEqual(scouting.score(df, "batting"), 1.3333, places=3)


class TestRecent(unittest.TestCase):
    def setUp(self):
        self.kbo = kbo_rows([
            ("a", "A", 2022, 24, 500, np.nan, 0.30, 0.40, 0.34, 0.43, np.nan),
            ("a", "A", 2023, 25, 500, np.nan, 0.35, 0.45, 0.34, 0.43, np.nan),
            ("a", "A", 2024, 26, 500, np.nan, 0.40, 0.50, 0.34, 0.43, np.nan),
            ("b", "B", 2024, 30, 500, np.nan, 0.31, 0.41, 0.34, 0.43, np.nan),
        ])

    def test_takes_the_last_n_seasons(self):
        got = scouting.recent(self.kbo, "a", seasons=2)
        self.assertEqual(sorted(got["season"]), [2023, 2024])

    def test_before_excludes_the_move_year_and_after(self):
        # Scoring a posted player on seasons he played AFTER he left would
        # be scoring him on MLB-era data. The cutoff has to be strict.
        got = scouting.recent(self.kbo, "a", seasons=2, before=2024)
        self.assertEqual(sorted(got["season"]), [2022, 2023])

    def test_unknown_player_is_empty_not_an_error(self):
        self.assertTrue(scouting.recent(self.kbo, "zzz").empty)


def _profile(scores):
    return pd.DataFrame({
        "player": [f"P{i}" for i in range(len(scores))],
        "score": scores,
    })


class TestBar(unittest.TestCase):
    def test_permissive_is_the_weakest_player_ever_posted(self):
        self.assertAlmostEqual(
            scouting.bar(_profile([1.0, 1.2, 1.5]), "permissive"), 1.0)

    def test_strict_is_the_median(self):
        self.assertAlmostEqual(
            scouting.bar(_profile([1.0, 1.2, 1.5]), "strict"), 1.2)

    def test_strictness_is_monotonic(self):
        prof = _profile([0.9, 1.1, 1.2, 1.3, 1.6])
        bars = [scouting.bar(prof, s) for s in
                ("permissive", "balanced", "strict")]
        self.assertEqual(bars, sorted(bars))

    def test_no_history_gives_no_bar(self):
        self.assertTrue(np.isnan(scouting.bar(pd.DataFrame())))

    def test_unknown_strictness_falls_back_to_balanced(self):
        prof = _profile([1.0, 1.2, 1.5])
        self.assertEqual(scouting.bar(prof, "nonsense"),
                         scouting.bar(prof, "balanced"))


class TestScreen(unittest.TestCase):
    def setUp(self):
        self.profile = _profile([1.0, 1.2, 1.5])
        self.scores = pd.Series([0.8, 1.1, 1.3, 1.9],
                                index=["w", "x", "y", "z"])

    def test_admits_players_at_or_above_the_bar(self):
        got = scouting.screen(self.scores, self.profile, "permissive")
        self.assertEqual(got["threshold"], 1.0)
        self.assertEqual(list(got["mask"]), [False, True, True, True])
        self.assertEqual(got["admitted"], 3)

    def test_reports_both_error_rates(self):
        # The count of survivors alone is not enough to judge a cut. A
        # screen that admits nobody looks impressive and is useless; the
        # historical recall is what catches that.
        got = scouting.screen(self.scores, self.profile, "strict")
        self.assertEqual(got["admitted"], 2)
        self.assertEqual(got["historical_total"], 3)
        self.assertEqual(got["historical_caught"], 2)

    def test_names_the_players_it_would_have_missed(self):
        got = scouting.screen(self.scores, self.profile, "strict")
        self.assertEqual(got["missed"], ["P0"])

    def test_permissive_catches_every_historical_signing(self):
        got = scouting.screen(self.scores, self.profile, "permissive")
        self.assertEqual(got["historical_caught"], got["historical_total"])
        self.assertEqual(got["missed"], [])

    def test_no_history_admits_everyone_rather_than_nobody(self):
        # Failing open is the right default here: with no benchmark the
        # honest answer is "this screen has nothing to say", not an empty
        # shortlist that reads as "no player qualifies".
        got = scouting.screen(self.scores, pd.DataFrame())
        self.assertTrue(got["mask"].all())
        self.assertEqual(got["admitted"], len(self.scores))
        self.assertEqual(got["historical_total"], 0)

    def test_unscored_players_are_excluded_not_admitted(self):
        scores = pd.Series([1.9, np.nan], index=["a", "b"])
        got = scouting.screen(scores, self.profile, "strict")
        self.assertEqual(list(got["mask"]), [True, False])

    def test_mask_keeps_the_index_it_was_given(self):
        got = scouting.screen(self.scores, self.profile, "balanced")
        self.assertEqual(list(got["mask"].index), list(self.scores.index))


class TestHistoricalProfile(unittest.TestCase):
    """The benchmark is where survivorship bias gets in."""

    def setUp(self):
        self.kbo = kbo_rows([
            ("star", "Star Kim", 2021, 25, 500, np.nan,
             0.42, 0.55, 0.34, 0.43, np.nan),
            ("star", "Star Kim", 2022, 26, 500, np.nan,
             0.42, 0.55, 0.34, 0.43, np.nan),
            # Played two MLB games and went home. He was still posted, so
            # dropping him would raise the bar using only the successes.
            ("bust", "Bust Park", 2016, 29, 500, np.nan,
             0.36, 0.45, 0.34, 0.43, np.nan),
            ("imp", "Eric Smith", 2019, 28, 500, np.nan,
             0.44, 0.60, 0.34, 0.43, np.nan),
        ])
        self.posted = pd.DataFrame({
            "player_register_id": ["star", "bust", "imp"],
            "first_mlb": [2023, 2017, 2020],
        })

    def _korean(self, name, pid):
        return not name.startswith("Eric")

    def test_includes_a_posted_player_who_flopped(self):
        prof = scouting.historical_profile(
            self.kbo, self.posted, "batting", self._korean, seasons=2)
        self.assertIn("Bust Park", list(prof["player"]))

    def test_the_flop_pulls_the_permissive_bar_down(self):
        prof = scouting.historical_profile(
            self.kbo, self.posted, "batting", self._korean, seasons=2)
        with_flop = scouting.bar(prof, "permissive")
        without = scouting.bar(
            prof[prof["player"] != "Bust Park"], "permissive")
        self.assertLess(with_flop, without)

    def test_excludes_foreign_imports(self):
        prof = scouting.historical_profile(
            self.kbo, self.posted, "batting", self._korean, seasons=2)
        self.assertNotIn("Eric Smith", list(prof["player"]))

    def test_scores_only_seasons_before_the_move(self):
        kbo = kbo_rows([
            ("star", "Star Kim", 2022, 26, 500, np.nan,
             0.40, 0.50, 0.34, 0.43, np.nan),
            # A KBO return years later must not count as "pre-move form".
            ("star", "Star Kim", 2028, 32, 500, np.nan,
             0.30, 0.35, 0.34, 0.43, np.nan),
        ])
        posted = pd.DataFrame({"player_register_id": ["star"],
                               "first_mlb": [2023]})
        prof = scouting.historical_profile(
            kbo, posted, "batting", self._korean, seasons=2)
        self.assertAlmostEqual(prof["score"].iloc[0], 1.169, places=2)

    def test_players_with_no_known_move_year_are_skipped(self):
        posted = pd.DataFrame({"player_register_id": ["star"],
                               "first_mlb": [np.nan]})
        prof = scouting.historical_profile(
            self.kbo, posted, "batting", self._korean)
        self.assertTrue(prof.empty)

    def test_sorted_best_first(self):
        prof = scouting.historical_profile(
            self.kbo, self.posted, "batting", self._korean, seasons=2)
        self.assertEqual(list(prof["score"]),
                         sorted(prof["score"], reverse=True))


class TestStrictnessTable(unittest.TestCase):
    def test_every_setting_has_a_percentile(self):
        for name, q in scouting.STRICTNESS.items():
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)

    def test_both_sides_have_a_readable_score_name(self):
        self.assertEqual(set(scouting.SCORE_NAME), {"batting", "pitching"})


if __name__ == "__main__":
    unittest.main()
