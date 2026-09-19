"""Tests for cohort classification and out-of-sample scoring."""

import unittest

import pandas as pd

from kbo_mlb import cohorts, evaluate


def seasons(pid, years):
    return pd.DataFrame({"player_register_id": [pid] * len(years),
                         "season": years})


class TestCohortClassification(unittest.TestCase):
    def test_posted_player_started_in_the_kbo(self):
        kbo = seasons("ryu", [2006, 2007, 2008])
        mlb = seasons("ryu", [2013, 2014])
        got = cohorts.classify(kbo, mlb)
        self.assertEqual(got.iloc[0]["cohort"], cohorts.POSTED)

    def test_returning_import_started_in_mlb(self):
        kbo = seasons("kelly", [2015, 2016])
        mlb = seasons("kelly", [2012, 2019])
        got = cohorts.classify(kbo, mlb)
        self.assertEqual(got.iloc[0]["cohort"], cohorts.IMPORT)

    def test_nationality_is_not_used(self):
        # Jung-hoo Lee was born in Nagoya. A birthplace filter would drop
        # him from the Korean cohort; a route-based one keeps him.
        kbo = seasons("lee", [2017, 2018, 2019])
        mlb = seasons("lee", [2024])
        got = cohorts.classify(kbo, mlb)
        self.assertEqual(got.iloc[0]["cohort"], cohorts.POSTED)

    def test_kbo_only_player_is_unknown(self):
        got = cohorts.classify(seasons("kim", [2020, 2021]), pd.DataFrame(
            columns=["player_register_id", "season"]))
        self.assertEqual(got.iloc[0]["cohort"], cohorts.UNKNOWN)

    def test_same_first_year_is_not_guessed(self):
        got = cohorts.classify(seasons("x", [2020]), seasons("x", [2020]))
        self.assertEqual(got.iloc[0]["cohort"], cohorts.UNKNOWN)

    def test_attach_adds_the_label_to_pairs(self):
        pairs = pd.DataFrame({"player_register_id": ["ryu"],
                              "direction": ["kbo_to_mlb"]})
        groups = cohorts.classify(seasons("ryu", [2008]), seasons("ryu", [2013]))
        out = cohorts.attach(pairs, groups)
        self.assertEqual(out.iloc[0]["cohort"], cohorts.POSTED)


class TestScorecard(unittest.TestCase):
    def _results(self, predicted, actual, kbo_relative=1.5):
        return pd.DataFrame({
            "stat": ["k_pct"] * len(predicted),
            "predicted": predicted,
            "actual": actual,
            "kbo_relative": [kbo_relative] * len(predicted),
            "in_interval": [True] * len(predicted),
        })

    def test_reports_both_baselines(self):
        scores = evaluate.scorecard(self._results([1.1, 1.2], [1.1, 1.2]))
        row = scores.iloc[0]
        self.assertEqual(row["mae_model"], 0.0)
        self.assertTrue(row["beats_league_average"])
        self.assertTrue(row["beats_carry_over"])

    def test_a_useless_model_is_marked_as_such(self):
        # Predicts far from the truth; "assume league average" does better.
        scores = evaluate.scorecard(self._results([3.0, 3.0], [1.0, 1.0]))
        self.assertFalse(scores.iloc[0]["beats_league_average"])

    def test_interval_coverage_is_reported(self):
        scores = evaluate.scorecard(self._results([1.0], [1.0]))
        self.assertEqual(scores.iloc[0]["interval_coverage"], 1.0)

    def test_empty_input_is_safe(self):
        self.assertTrue(evaluate.scorecard(pd.DataFrame()).empty)


class TestHoldoutGuard(unittest.TestCase):
    def test_requires_a_cohort_column(self):
        with self.assertRaises(ValueError):
            evaluate.holdout_validate(pd.DataFrame({"a": [1]}), "batting")


if __name__ == "__main__":
    unittest.main()
