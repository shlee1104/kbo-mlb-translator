"""Tests for the app's data layer.

Streamlit itself cannot be imported without a browser session, so the logic
lives in `app_data` and is tested here. `app.py` is layout only.
"""

import unittest

import pandas as pd

from kbo_mlb import app_data, translate


class TestVerdicts(unittest.TestCase):
    """The plain-language read a non-technical user sees."""

    def test_strong_fit_is_usable(self):
        self.assertEqual(app_data._verdict(0.50, 0.45), "usable signal")

    def test_band_width_matches_the_observed_interval(self):
        # Kim Do-young's batting average band came back .1317-.3794, a
        # factor of 2.88, from a residual sd of 0.429. The formula should
        # land close to what the bootstrap actually produced.
        self.assertAlmostEqual(app_data.typical_range_factor(0.429),
                               2.88, delta=0.25)

    def test_middling_fit_is_weak(self):
        self.assertEqual(app_data._verdict(0.22, 0.12), "weak signal")

    def test_flat_fit_says_so(self):
        # A pitcher's ERA: slope 0.03, R^2 0.006. The tool must not dress
        # this up as a projection.
        self.assertEqual(app_data._verdict(0.03, 0.006), "no usable signal")

    def test_high_slope_with_no_explanatory_power_is_not_usable(self):
        self.assertNotEqual(app_data._verdict(0.9, 0.01), "usable signal")


class TestLabels(unittest.TestCase):
    def test_every_display_stat_has_a_plain_name(self):
        for stat in app_data.DISPLAY_ORDER:
            self.assertIn(stat, app_data.PRETTY)
            self.assertNotEqual(app_data.PRETTY[stat], stat)

    def test_lower_is_better_set_is_sane(self):
        # Getting this backwards would invert how a scout reads the page.
        self.assertIn("k_pct", app_data.LOWER_IS_BETTER)
        self.assertIn("era", app_data.LOWER_IS_BETTER)
        self.assertNotIn("avg", app_data.LOWER_IS_BETTER)
        self.assertNotIn("obp", app_data.LOWER_IS_BETTER)


class TestModelQuality(unittest.TestCase):
    def _bundle_with(self, fits):
        model = translate.TranslationModel(side="batting")
        model.fits = fits
        return app_data.Bundle(
            side="batting", season=2026, kbo=pd.DataFrame(),
            kbo_raw=pd.DataFrame(), mlb_league=pd.DataFrame(),
            model=model, draws={}, train_pairs=pd.DataFrame(),
            validation=pd.DataFrame())

    def test_empty_model_gives_empty_table(self):
        self.assertTrue(app_data.model_quality(self._bundle_with({})).empty)

    def _fit(self, stat="k_pct", slope=0.5, r2=0.45, sd=0.6):
        return translate.StatFit(stat, intercept=0.0, slope=slope,
                                 age_coef=0.0, direction_coef=0.0,
                                 n_pairs=32, n_players=32,
                                 residual_sd=sd, r_squared=r2)

    def test_table_carries_a_verdict_per_statistic(self):
        out = app_data.model_quality(self._bundle_with({"k_pct": self._fit()}))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["statistic"], "Strikeout rate")
        self.assertIn("verdict", out.columns)
        self.assertIn("typical range", out.columns)

    def test_a_tight_fit_is_usable_for_one_player(self):
        out = app_data.model_quality(
            self._bundle_with({"k_pct": self._fit(sd=0.25)}))
        self.assertEqual(out.iloc[0]["verdict"], "usable signal")

    def test_a_real_pattern_with_wide_individual_spread_says_so(self):
        # Strikeout rate in the live data: best R-squared of any statistic,
        # yet the band for a single player spans nearly 6x. Calling that
        # "usable" would be misleading, and calling it "no signal" would be
        # wrong. It needs its own answer.
        out = app_data.model_quality(
            self._bundle_with({"k_pct": self._fit(sd=0.685)}))
        self.assertEqual(out.iloc[0]["verdict"],
                         "real pattern, too wide for one player")


class TestMissingDataMessage(unittest.TestCase):
    def test_load_explains_itself_when_the_pipeline_has_not_run(self):
        # A user opening the page before running the pipeline should get a
        # sentence, not a KeyError.
        original = app_data._interim
        app_data._interim = lambda name: pd.DataFrame()
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                app_data.load()
            message = str(ctx.exception)
            self.assertIn("pipeline has not been run", message)
            self.assertIn("cli", message)
        finally:
            app_data._interim = original


class TestRosterShape(unittest.TestCase):
    def test_roster_counts_seasons_from_unfiltered_data(self):
        # Service time accrues by being rostered, not by playing. A season
        # dropped by the playing-time filter must still count.
        kbo = pd.DataFrame({
            "player_register_id": ["p1", "p1"],
            "player": ["Do Yeong Kim", "Do Yeong Kim"],
            "team_name": ["Kia Tigers", "Kia Tigers"],
            "season": [2024, 2026],
            "age": [20, 22],
            "PA": [625, 538],
        })
        raw = pd.DataFrame({
            "player_register_id": ["p1"] * 3,
            "season": [2024, 2025, 2026],       # 2025 was injury-shortened
        })
        bundle = app_data.Bundle(
            side="batting", season=2026, kbo=kbo, kbo_raw=raw,
            mlb_league=pd.DataFrame(),
            model=translate.TranslationModel(side="batting"), draws={},
            train_pairs=pd.DataFrame(), validation=pd.DataFrame())
        roster = bundle.roster()
        self.assertEqual(len(roster), 1)
        self.assertEqual(int(roster.iloc[0]["seasons"]), 3)


if __name__ == "__main__":
    unittest.main()
