"""Tests for the data quality checks.

Each test plants a specific, realistic defect and asserts the suite catches
it. A check that cannot fail is not a check.
"""

import unittest

import pandas as pd

from kbo_mlb import validate


def batting_frame(**overrides):
    base = pd.DataFrame({
        "player_register_id": ["a001", "b002", "c003"],
        "player": ["Jung-hoo Lee", "Erick Fedde", "Ha-seong Kim"],
        "season": [2024, 2024, 2024],
        "team_bref_id": ["t1", "t1", "t2"],
        "PA": [600, 10, 500],
        "AB": [520, 9, 450],
        "H": [180, 2, 135],
        "2B": [30, 0, 25],
        "3B": [5, 0, 2],
        "HR": [20, 0, 15],
        "TB": [280, 2, 209],
        "batting_avg": [0.346, 0.222, 0.300],
        "G": [140, 5, 120],
        "age": [25, 31, 28],
    })
    for key, value in overrides.items():
        base[key] = value
    return base


class TestStructuralChecks(unittest.TestCase):
    def test_required_columns_pass(self):
        r = validate.check_required_columns(
            batting_frame(), ["player", "season", "PA"], "batting")
        self.assertTrue(r.passed)

    def test_required_columns_fail(self):
        r = validate.check_required_columns(
            batting_frame(), ["player", "does_not_exist"], "batting")
        self.assertFalse(r.passed)
        self.assertEqual(r.severity, "error")

    def test_detects_duplicate_player_team_seasons(self):
        df = batting_frame()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        r = validate.check_duplicate_player_seasons(df, "batting")
        self.assertFalse(r.passed)
        self.assertEqual(r.details["duplicate_count"], 2)

    def test_allows_same_player_on_two_teams(self):
        # A midseason trade is legitimate, not a duplicate.
        df = batting_frame()
        traded = df.iloc[[0]].copy()
        traded["team_bref_id"] = "t9"
        r = validate.check_duplicate_player_seasons(
            pd.concat([df, traded], ignore_index=True), "batting")
        self.assertTrue(r.passed)

    def test_detects_missing_register_ids(self):
        df = batting_frame()
        df.loc[0, "player_register_id"] = None
        r = validate.check_missing_ids(df, "batting")
        self.assertFalse(r.passed)


class TestValueRanges(unittest.TestCase):
    def test_clean_frame_passes(self):
        results = validate.check_value_ranges(batting_frame(), "batting")
        self.assertTrue(all(r.passed for r in results))

    def test_catches_impossible_batting_average(self):
        df = batting_frame()
        df.loc[0, "batting_avg"] = 1.4
        results = validate.check_value_ranges(df, "batting")
        failed = [r for r in results if not r.passed]
        self.assertEqual(len(failed), 1)
        self.assertIn("batting_avg", failed[0].name)

    def test_catches_negative_counts(self):
        df = batting_frame()
        df.loc[1, "HR"] = -3
        r = validate.check_negative_counts(df, "batting")
        self.assertFalse(r.passed)


class TestBattingIdentities(unittest.TestCase):
    def test_clean_frame_passes(self):
        results = validate.check_batting_identities(batting_frame())
        self.assertTrue(all(r.passed for r in results),
                        [r.message for r in results if not r.passed])

    def test_catches_ab_greater_than_pa(self):
        df = batting_frame()
        df.loc[0, "AB"] = 700
        results = validate.check_batting_identities(df)
        failed = {r.name for r in results if not r.passed}
        self.assertIn("batting.pa_ge_ab", failed)

    def test_catches_extra_base_hits_exceeding_hits(self):
        df = batting_frame()
        df.loc[0, "HR"] = 200
        results = validate.check_batting_identities(df)
        failed = {r.name for r in results if not r.passed}
        self.assertIn("batting.hits_ge_extra_base", failed)

    def test_catches_shifted_column(self):
        # The defect this check exists for: total bases no longer reconcile
        # because a column slipped, even though every value looks plausible.
        df = batting_frame()
        df.loc[0, "TB"] = 250
        results = validate.check_batting_identities(df)
        failed = {r.name for r in results if not r.passed}
        self.assertIn("batting.total_bases_identity", failed)


class TestCrossSourceReconciliation(unittest.TestCase):
    def test_totals_reconcile(self):
        players = batting_frame()
        totals = pd.DataFrame({"season": [2024], "HR": [35], "side": ["batting"]})
        r = validate.check_league_totals_reconcile(players, totals)
        self.assertTrue(r.passed, r.message)

    def test_undercount_is_a_source_coverage_warning(self):
        # The source lists fewer players than the total it publishes. Verified
        # against Baseball-Reference directly, so it is a coverage limit to
        # surface, not a parsing error to fail on.
        players = batting_frame()
        totals = pd.DataFrame({"season": [2024], "HR": [1438],
                               "side": ["batting"]})
        r = validate.check_league_totals_reconcile(players, totals)
        self.assertFalse(r.passed)
        self.assertEqual(r.severity, "warning")
        self.assertLess(r.details["worst_coverage"], 1.0)

    def test_overcount_is_our_bug_and_errors(self):
        # Scraping MORE than the published total means we double counted.
        players = batting_frame()
        totals = pd.DataFrame({"season": [2024], "HR": [10],
                               "side": ["batting"]})
        r = validate.check_league_totals_reconcile(players, totals)
        self.assertFalse(r.passed)
        self.assertEqual(r.severity, "error")


class TestCoverage(unittest.TestCase):
    def test_reports_missing_seasons(self):
        r = validate.check_season_coverage(batting_frame(), 2022, 2024)
        self.assertFalse(r.passed)
        self.assertEqual(r.details["missing_seasons"], [2022, 2023])

    def test_full_coverage_passes(self):
        r = validate.check_season_coverage(batting_frame(), 2024, 2024)
        self.assertTrue(r.passed)


class TestReporting(unittest.TestCase):
    def test_has_errors_distinguishes_severity(self):
        ok = validate.CheckResult("a", True, "error", "fine")
        warn = validate.CheckResult("b", False, "warning", "hmm")
        err = validate.CheckResult("c", False, "error", "bad")
        self.assertFalse(validate.has_errors([ok, warn]))
        self.assertTrue(validate.has_errors([ok, warn, err]))

    def test_markdown_report_renders(self):
        results = validate.run_all(
            batting_frame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), 2024, 2024)
        md = validate.to_markdown(results)
        self.assertIn("# KBO data audit", md)
        self.assertIn("checks run", md)


if __name__ == "__main__":
    unittest.main()
