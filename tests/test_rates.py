"""Tests for rate computation and league-relative scaling."""

import unittest

import pandas as pd

from kbo_mlb import rates


class TestRateFormulas(unittest.TestCase):
    def setUp(self):
        self.batting = pd.DataFrame({
            "season": [2024],
            "PA": [600], "AB": [520], "H": [180], "2B": [30], "3B": [5],
            "HR": [20], "SO": [60], "BB": [70], "HBP": [5], "SF": [5],
            "TB": [280],
        })

    def test_basic_rates(self):
        out = rates.add_rates(self.batting, "batting")
        self.assertAlmostEqual(out["k_pct"].iloc[0], 60 / 600)
        self.assertAlmostEqual(out["bb_pct"].iloc[0], 70 / 600)
        self.assertAlmostEqual(out["hr_pct"].iloc[0], 20 / 600)
        self.assertAlmostEqual(out["avg"].iloc[0], 180 / 520)
        self.assertAlmostEqual(out["slg"].iloc[0], 280 / 520)

    def test_obp_uses_the_full_denominator(self):
        out = rates.add_rates(self.batting, "batting")
        expected = (180 + 70 + 5) / (520 + 70 + 5 + 5)
        self.assertAlmostEqual(out["obp"].iloc[0], expected)

    def test_babip_excludes_homers_and_strikeouts(self):
        out = rates.add_rates(self.batting, "batting")
        expected = (180 - 20) / (520 - 60 - 20 + 5)
        self.assertAlmostEqual(out["babip"].iloc[0], expected)

    def test_zero_denominator_is_missing_not_infinite(self):
        empty = pd.DataFrame({"PA": [0], "AB": [0], "H": [0], "SO": [0],
                              "BB": [0], "HR": [0], "TB": [0]})
        out = rates.add_rates(empty, "batting")
        self.assertTrue(pd.isna(out["k_pct"].iloc[0]))
        self.assertTrue(pd.isna(out["avg"].iloc[0]))


class TestInningsHandling(unittest.TestCase):
    def test_era_is_rebuilt_from_thirds_notation(self):
        # 180.1 IP means 180 and one third = 541 outs, not 180.1 innings.
        df = pd.DataFrame({"IP": [180.1], "ER": [40], "batters_faced": [700],
                           "SO": [209], "BB": [45], "HR": [15], "H": [150]})
        out = rates.add_rates(df, "pitching")
        expected = 40 * 27 / 541
        self.assertAlmostEqual(out["era"].iloc[0], expected, places=6)

    def test_thirds_are_not_read_as_decimals(self):
        df = pd.DataFrame({"IP": [5.2], "ER": [1], "batters_faced": [20],
                           "SO": [5], "BB": [1], "HR": [0], "H": [4]})
        out = rates.add_rates(df, "pitching")
        # 5.2 -> 17 outs. A decimal reading would give 5.2*3 = 15.6 outs.
        self.assertAlmostEqual(out["era"].iloc[0], 1 * 27 / 17, places=6)


class TestLeagueRelative(unittest.TestCase):
    """The core guarantee: league and player rates use the same formula."""

    def setUp(self):
        self.league_totals = pd.DataFrame({
            "season": [2024], "side": ["batting"],
            "PA": [57263], "AB": [50205], "H": [13929], "2B": [2381],
            "3B": [229], "HR": [1438], "SO": [10826], "BB": [5265],
            "HBP": [783], "SF": [528], "TB": [21082],
        })

    def test_league_rates_match_hand_computation(self):
        lg = rates.league_rates(self.league_totals, "batting")
        self.assertAlmostEqual(lg["lg_k_pct"].iloc[0], 10826 / 57263)
        self.assertAlmostEqual(lg["lg_avg"].iloc[0], 13929 / 50205)

    def test_league_average_player_scores_exactly_one(self):
        # A player whose totals ARE the league totals must come out at 1.00
        # on every rate. If the two formulas ever drift apart, this breaks.
        player = self.league_totals.drop(columns=["side"]).copy()
        player["player_register_id"] = ["lg001"]
        lg = rates.league_rates(self.league_totals, "batting")
        rel = rates.add_relative_rates(player, lg, "batting")
        for col in rates.relative_columns("batting"):
            self.assertAlmostEqual(rel[col].iloc[0], 1.0, places=9, msg=col)

    def test_above_average_player_scores_above_one(self):
        player = pd.DataFrame({
            "season": [2024], "player_register_id": ["p1"],
            "PA": [600], "AB": [520], "H": [180], "2B": [30], "3B": [5],
            "HR": [30], "SO": [60], "BB": [70], "HBP": [5], "SF": [5],
            "TB": [310],
        })
        lg = rates.league_rates(self.league_totals, "batting")
        rel = rates.add_relative_rates(player, lg, "batting")
        self.assertGreater(rel["rel_avg"].iloc[0], 1.0)     # .346 vs .277
        self.assertGreater(rel["rel_hr_pct"].iloc[0], 1.0)  # 5.0% vs 2.5%
        self.assertLess(rel["rel_k_pct"].iloc[0], 1.0)      # 10% vs 18.9%

    def test_playing_time_minimum_drops_tiny_samples(self):
        players = pd.DataFrame({
            "season": [2024, 2024], "player_register_id": ["p1", "p2"],
            "PA": [600, 12], "AB": [520, 11], "H": [180, 4], "HR": [30, 0],
            "SO": [60, 3], "BB": [70, 1], "TB": [310, 5], "HBP": [0, 0],
            "SF": [0, 0], "2B": [0, 0], "3B": [0, 0],
        })
        lg = rates.league_rates(self.league_totals, "batting")
        rel = rates.add_relative_rates(players, lg, "batting",
                                       min_playing_time=100)
        self.assertEqual(list(rel["player_register_id"]), ["p1"])


if __name__ == "__main__":
    unittest.main()
