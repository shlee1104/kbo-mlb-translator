"""Regressions for bugs found by running the pipeline on real data.

Each of these was a silent failure: the pipeline completed and produced
plausible-looking output while being wrong.
"""

import unittest

import pandas as pd

from kbo_mlb import crosswalk, scrape_kbo


class TestUndefinedRatesBecomeMissing(unittest.TestCase):
    """A pitcher who allows runs without recording an out has ERA = inf.

    Baseball-Reference prints "inf". Left as a float, one such row turns
    every downstream mean into infinity.
    """

    def test_infinite_era_is_stored_as_missing(self):
        raw = pd.DataFrame({
            "player": ["Zero Outs", "Normal Pitcher"],
            "player_register_id": ["a001", "b002"],
            "IP": ["0.0", "180.1"],
            "ER": ["3", "40"],
            "earned_run_avg": ["inf", "2.00"],
        })
        out = scrape_kbo._finalise(raw)
        self.assertTrue(pd.isna(out.loc[0, "earned_run_avg"]))
        self.assertEqual(out.loc[1, "earned_run_avg"], 2.00)

    def test_mean_is_not_poisoned(self):
        raw = pd.DataFrame({
            "player": ["a", "b"],
            "player_register_id": ["a001", "b002"],
            "earned_run_avg": ["inf", "4.00"],
        })
        out = scrape_kbo._finalise(raw)
        self.assertEqual(out["earned_run_avg"].mean(), 4.00)


class TestNullKeysDoNotCrossJoin(unittest.TestCase):
    """pandas merges NaN to NaN.

    Joining on an ID column that still contains nulls cross-joins every
    id-less row on one side against every id-less row on the other. On the
    real data this turned ~476 genuine links into 200,180 rows, almost all
    of them fabricated.
    """

    def test_null_mlbam_ids_do_not_multiply_rows(self):
        kbo = pd.DataFrame({
            "player_register_id": ["p1", "p2", "p3"],
            "player": ["Player One", "Player Two", "Player Three"],
            "date_of_birth": ["1990-01-01", "1991-01-01", "1992-01-01"],
        })
        # Most of the register has no MLB id, which is the normal case.
        chadwick = pd.DataFrame({
            "key_bbref_minors": ["p1", "p2", "p3"],
            "key_mlbam": [111.0, None, None],
        })
        mlb = pd.DataFrame({
            "key_mlbam": [111.0, None, None],
            "name": ["Player One", "Someone Else", "Another Person"],
            "date_of_birth": ["1990-01-01", "1970-01-01", "1971-01-01"],
        })

        result = crosswalk.build(kbo, mlb, chadwick=chadwick)

        # One real link. Without the guard this produced four id-based rows.
        id_matches = result.matches[
            result.matches["match_tier"] == "register_id"]
        self.assertEqual(len(id_matches), 1)
        self.assertEqual(id_matches.iloc[0]["player_register_id"], "p1")

    def test_no_register_id_appears_twice(self):
        kbo = pd.DataFrame({
            "player_register_id": ["p1", "p2"],
            "player": ["Player One", "Player Two"],
            "date_of_birth": ["1990-01-01", "1991-01-01"],
        })
        chadwick = pd.DataFrame({
            "key_bbref_minors": ["p1", "p1", "p2"],   # duplicated on purpose
            "key_mlbam": [111.0, 111.0, None],
        })
        mlb = pd.DataFrame({
            "key_mlbam": [111.0],
            "name": ["Player One"],
            "date_of_birth": ["1990-01-01"],
        })
        result = crosswalk.build(kbo, mlb, chadwick=chadwick)
        counts = result.matches["player_register_id"].value_counts()
        self.assertTrue((counts <= 1).all(), counts.to_dict())


if __name__ == "__main__":
    unittest.main()
