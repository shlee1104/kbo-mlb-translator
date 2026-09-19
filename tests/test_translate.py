"""Tests for the translation model.

The centrepiece is `TestRecoversKnownCoefficients`: data is generated with a
known retention slope and level shift, and the fitter has to recover them.
If a model cannot recover a relationship it was handed, nothing it says
about real players is worth reading.
"""

import unittest

import numpy as np
import pandas as pd

from kbo_mlb import translate


class TestBuildPairs(unittest.TestCase):
    def _frames(self, kbo_seasons, mlb_seasons, pid="p1"):
        kbo = pd.DataFrame({
            "player_register_id": [pid] * len(kbo_seasons),
            "season": kbo_seasons,
            "rel_k_pct": [1.0] * len(kbo_seasons),
            "age": [25] * len(kbo_seasons),
        })
        mlb = pd.DataFrame({
            "player_register_id": [pid] * len(mlb_seasons),
            "season": mlb_seasons,
            "rel_k_pct": [1.2] * len(mlb_seasons),
            "age": [27] * len(mlb_seasons),
        })
        return kbo, mlb

    def test_finds_a_single_crossing(self):
        kbo, mlb = self._frames([2011, 2012], [2013, 2014])
        pairs = translate.build_pairs(kbo, mlb)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["kbo_season"], 2012)
        self.assertEqual(pairs.iloc[0]["mlb_season"], 2013)
        self.assertEqual(pairs.iloc[0]["direction"], "kbo_to_mlb")

    def test_finds_both_directions_for_a_returnee(self):
        # MLB -> KBO -> MLB, the typical imported pitcher's path.
        kbo, mlb = self._frames([2020, 2021], [2018, 2023])
        pairs = translate.build_pairs(kbo, mlb, max_gap_years=3)
        self.assertEqual(set(pairs["direction"]),
                         {"mlb_to_kbo", "kbo_to_mlb"})
        self.assertEqual(len(pairs), 2)

    def test_does_not_pair_across_a_long_gap(self):
        kbo, mlb = self._frames([2005], [2015])
        pairs = translate.build_pairs(kbo, mlb, max_gap_years=2)
        self.assertTrue(pairs.empty)

    def test_does_not_explode_combinatorially(self):
        # Five KBO seasons then five MLB seasons is ONE crossing, not 25
        # pairs. Over-pairing would fake a large sample.
        kbo, mlb = self._frames([2010, 2011, 2012, 2013, 2014],
                                [2015, 2016, 2017, 2018, 2019])
        pairs = translate.build_pairs(kbo, mlb)
        self.assertEqual(len(pairs), 1)

    def test_no_pairs_for_a_player_in_one_league_only(self):
        kbo = pd.DataFrame({"player_register_id": ["p1"], "season": [2020],
                            "rel_k_pct": [1.0], "age": [25]})
        mlb = pd.DataFrame({"player_register_id": ["p2"], "season": [2021],
                            "rel_k_pct": [1.0], "age": [26]})
        self.assertTrue(translate.build_pairs(kbo, mlb).empty)


def synthetic_pairs(n_players=180, slope=0.60, level=-0.15, seed=7,
                    noise=0.05):
    """Players whose MLB rate follows a known function of their KBO rate."""
    rng = np.random.default_rng(seed)
    kbo_rel = np.exp(rng.normal(0, 0.30, n_players))
    age = rng.integers(22, 33, n_players).astype(float)
    direction = rng.choice(["kbo_to_mlb", "mlb_to_kbo"], n_players)

    log_mlb = (level
               + slope * np.log(kbo_rel)
               + 0.0 * (age - translate.AGE_CENTRE) / 10.0
               + rng.normal(0, noise, n_players))
    return pd.DataFrame({
        "player_register_id": [f"p{i}" for i in range(n_players)],
        "direction": direction,
        "kbo_season": 2015, "mlb_season": 2016, "gap_years": 1,
        "kbo_rel_k_pct": kbo_rel,
        "mlb_rel_k_pct": np.exp(log_mlb),
        "kbo_age": age,
        "kbo_PA": rng.integers(300, 650, n_players),
        "mlb_PA": rng.integers(300, 650, n_players),
    })


class TestRecoversKnownCoefficients(unittest.TestCase):
    def test_recovers_slope_and_level(self):
        pairs = synthetic_pairs(slope=0.60, level=-0.15)
        model = translate.fit(pairs, "batting", stats=["k_pct"])
        f = model.fits["k_pct"]
        self.assertAlmostEqual(f.slope, 0.60, delta=0.06)
        self.assertAlmostEqual(f.intercept, -0.15, delta=0.06)
        self.assertGreater(f.r_squared, 0.8)

    def test_recovers_a_different_slope(self):
        # A statistic that barely carries over at all.
        pairs = synthetic_pairs(slope=0.20, level=-0.30, seed=11)
        f = translate.fit(pairs, "batting", stats=["k_pct"]).fits["k_pct"]
        self.assertAlmostEqual(f.slope, 0.20, delta=0.06)

    def test_refuses_to_fit_a_tiny_sample(self):
        pairs = synthetic_pairs(n_players=5)
        model = translate.fit(pairs, "batting", stats=["k_pct"])
        self.assertNotIn("k_pct", model.fits)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.pairs = synthetic_pairs(slope=0.60, level=-0.15)
        self.model = translate.fit(self.pairs, "batting", stats=["k_pct"])

    def test_league_average_input_gives_the_level_shift(self):
        # A player exactly at KBO average should land near exp(intercept).
        got = translate.predict_relative(self.model, "k_pct", 1.0,
                                         age=translate.AGE_CENTRE,
                                         direction="mlb_to_kbo")
        self.assertAlmostEqual(got, np.exp(-0.15), delta=0.06)

    def test_prediction_is_monotonic_in_input(self):
        lo = translate.predict_relative(self.model, "k_pct", 0.8, 27)
        hi = translate.predict_relative(self.model, "k_pct", 1.4, 27)
        self.assertGreater(hi, lo)

    def test_prediction_is_always_positive(self):
        for x in (0.01, 0.5, 1.0, 5.0):
            self.assertGreater(
                translate.predict_relative(self.model, "k_pct", x, 27), 0)


class TestBootstrap(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self):
        pairs = synthetic_pairs(n_players=120)
        out = translate.bootstrap_predict(pairs, "batting", "k_pct",
                                          kbo_relative=1.2, age=26,
                                          n_boot=40, seed=3)
        self.assertTrue(out)
        self.assertLess(out["p10"], out["p50"])
        self.assertLess(out["p50"], out["p90"])

    def test_noisier_data_gives_a_wider_interval(self):
        tight = translate.bootstrap_predict(
            synthetic_pairs(noise=0.03, seed=5), "batting", "k_pct",
            1.2, 26, n_boot=40, seed=1)
        loose = translate.bootstrap_predict(
            synthetic_pairs(noise=0.30, seed=5), "batting", "k_pct",
            1.2, 26, n_boot=40, seed=1)
        tight_width = tight["p90"] - tight["p10"]
        loose_width = loose["p90"] - loose["p10"]
        self.assertGreater(loose_width, tight_width)


class TestAbsoluteConversion(unittest.TestCase):
    def test_relative_times_league_rate(self):
        self.assertAlmostEqual(translate.to_absolute(1.25, 0.20), 0.25)


if __name__ == "__main__":
    unittest.main()
