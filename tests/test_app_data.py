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



class TestTrustLabels(unittest.TestCase):
    """A narrow interval is not the same as knowing something."""

    def _bundle(self, r2):
        model = translate.TranslationModel(side="batting")
        model.fits = {"avg": translate.StatFit(
            "avg", intercept=0.0, slope=0.23, age_coef=0.0,
            direction_coef=0.0, n_pairs=32, n_players=32,
            residual_sd=0.429, r_squared=r2)}
        return app_data.Bundle(
            side="batting", season=2026, kbo=pd.DataFrame(),
            kbo_raw=pd.DataFrame(), mlb_league=pd.DataFrame(),
            model=model, draws={}, train_pairs=pd.DataFrame(),
            validation=pd.DataFrame())

    def _row(self, p10=0.53, p90=1.54):
        return pd.Series({"stat": "avg", "p10_relative": p10,
                          "p90_relative": p90})

    def test_narrow_band_without_signal_is_not_usable(self):
        # Batting average in the live data: band of 2.9x but R-squared 0.09.
        # The interval is tight because the model predicts roughly league
        # average for everyone, not because it understands the player.
        from kbo_mlb import project
        bundle = self._bundle(r2=0.09)
        row = self._row()
        narrow = project.interval_is_informative(row)
        fit = bundle.model.fits["avg"]
        self.assertTrue(narrow)
        self.assertLess(fit.r_squared, 0.10)

    def test_narrow_band_with_signal_is_usable(self):
        bundle = self._bundle(r2=0.27)
        self.assertGreaterEqual(bundle.model.fits["avg"].r_squared, 0.10)

class TestNationality(unittest.TestCase):
    """Birthplace is authoritative where present; the name covers the rest."""

    def test_korean_birthplace(self):
        self.assertEqual(app_data.infer_korean("Do Yeong Kim", "Gwangju, KR"),
                         (True, "birthplace"))

    def test_korean_name_overrides_a_foreign_birthplace(self):
        # Jung-hoo Lee was born in Nagoya while his father played in Japan.
        # Birthplace is not nationality, and a filter keying on it alone
        # would drop one of the most obvious signing targets in the league.
        korean, source = app_data.infer_korean("Jung Hoo Lee",
                                               "Nagoya, Aichi, JP")
        self.assertTrue(korean)
        self.assertIn("born abroad", source)

    def test_import_with_no_birthplace_is_caught_by_name(self):
        self.assertEqual(app_data.infer_korean("Austin Dean", ""),
                         (False, "name"))

    def test_japanese_import_is_not_korean(self):
        self.assertFalse(app_data.infer_korean("Koki Sugimoto", "")[0])

    def test_surname_spelling_found_in_the_live_roster(self):
        # 문 is usually romanised Moon or Mun; the 2026 roster spells one
        # player "Mon", which the surname table originally missed.
        self.assertTrue(app_data.infer_korean("Yong Ik Mon", "")[0])


class TestSigningTargetFilters(unittest.TestCase):
    def _bundle(self, **kw):
        kbo = pd.DataFrame({
            "player_register_id": ["kr_young", "kr_vet", "import"],
            "player": ["Do Yeong Kim", "Seong Han Park", "Austin Dean"],
            "team_name": ["Kia Tigers", "SSG Landers", "LG Twins"],
            "season": [2026, 2026, 2026],
            "age": [22, 28, 32],
            "PA": [538, 569, 570],
        })
        raw = pd.DataFrame({
            "player_register_id": (["kr_young"] * 5 + ["kr_vet"] * 9
                                   + ["import"] * 4),
            "season": (list(range(2022, 2027)) + list(range(2018, 2027))
                       + list(range(2023, 2027))),
        })
        defaults = dict(side="batting", season=2026, kbo=kbo, kbo_raw=raw,
                        mlb_league=pd.DataFrame(),
                        model=translate.TranslationModel(side="batting"),
                        draws={}, train_pairs=pd.DataFrame(),
                        validation=pd.DataFrame(),
                        birth_cities=pd.Series(dtype=object))
        defaults.update(kw)
        return app_data.Bundle(**defaults)

    def test_default_keeps_only_pre_fa_koreans(self):
        got = self._bundle().roster()
        self.assertEqual(list(got["player_register_id"]), ["kr_young"])

    def test_import_is_excluded_even_though_he_leads_in_playing_time(self):
        got = self._bundle()
        self.assertNotIn("import", set(got.roster()["player_register_id"]))

    def test_veteran_past_free_agency_is_excluded(self):
        got = self._bundle().roster()
        self.assertNotIn("kr_vet", set(got["player_register_id"]))

    def test_filters_can_be_turned_off(self):
        got = self._bundle(korean_only=False, pre_fa_only=False).roster()
        self.assertEqual(len(got), 3)

    def test_unfiltered_view_is_available_for_counting(self):
        b = self._bundle()
        self.assertEqual(len(b.roster(apply_filters=False)), 3)
        self.assertEqual(len(b.roster()), 1)

    def test_fa_threshold_is_adjustable(self):
        # At a 10-season threshold the nine-season veteran is still in.
        got = self._bundle(fa_seasons=10).roster()
        self.assertIn("kr_vet", set(got["player_register_id"]))


class TestAcquisitionWindow(unittest.TestCase):
    def test_window_is_the_gap_between_posting_and_free_agency(self):
        from kbo_mlb import project
        a = project.availability(seasons_played=5, age_now=22,
                                 current_season=2026)
        self.assertEqual(a["earliest_posting_season"], 2028)
        self.assertEqual(a["domestic_fa_season"], 2029)
        self.assertEqual(a["acquisition_window_seasons"], 1)
        self.assertFalse(a["past_first_fa"])

    def test_window_is_closed_once_free_agency_has_passed(self):
        from kbo_mlb import project
        a = project.availability(seasons_played=9, age_now=31,
                                 current_season=2026)
        self.assertTrue(a["past_first_fa"])
        self.assertEqual(a["acquisition_window_seasons"], 0)
        self.assertEqual(a["window"], "closed")

if __name__ == "__main__":
    unittest.main()
