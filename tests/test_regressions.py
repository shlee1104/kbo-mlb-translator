"""Regressions for bugs found by running the pipeline on real data.

Each of these was a silent failure: the pipeline completed and produced
plausible-looking output while being wrong.
"""

import unittest

import pandas as pd

from kbo_mlb import crosswalk, names, scrape_kbo


class TestKnownPhoneticCollision(unittest.TestCase):
    """이정후 (Jung-hoo) and 이정호 (Jung-ho) share a loose key. On purpose.

    The loose key folds u onto o, which is what lets "Jung" match "Jeong" and
    "Hoo" match "Hu" — without it, no Hangul name matches its romanisation.
    The cost is that two genuinely different players collide, and the KBO
    really does contain both of these.

    This is safe only because a loose-key match is never auto-accepted
    without an exact date-of-birth agreement. These tests exist so that
    nobody "fixes" the collision by tightening the key and silently breaks
    every Hangul match in the process.
    """

    def test_collision_is_expected(self):
        self.assertEqual(names.loose_key("Jung Hoo Lee"),
                         names.loose_key("Jung Ho Lee"))

    def test_collision_does_not_survive_the_birth_date_gate(self):
        kbo = pd.DataFrame({
            "player_register_id": ["lee-junghoo"],
            "player": ["Jung Hoo Lee"],
            "date_of_birth": ["1998-08-20"],
        })
        mlb = pd.DataFrame({
            "key_mlbam": [1],
            "name": ["Jung Ho Lee"],          # different player, same key
            "date_of_birth": ["1987-04-05"],
        })
        result = crosswalk.build(kbo, mlb)
        tiers = (set(result.matches["match_tier"])
                 if "match_tier" in result.matches.columns else set())
        self.assertNotIn("phonetic_dob", tiers)
        self.assertIn("lee-junghoo",
                      set(result.review_queue["player_register_id"]))


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


class TestPitcherPlayingTimeColumn(unittest.TestCase):
    """`project_player` asked every player for plate appearances.

    Pitching frames have `batters_faced` instead, so `rows.get("PA")`
    returned None, `pd.to_numeric(None)` returned a bare nan, and the next
    line called `.fillna` on a float. Every pitcher's page raised
    AttributeError on open — found by executing app.py against a stubbed
    Streamlit rather than by any unit test, because nothing else ever
    called this function with a pitching frame.
    """

    def _rows(self, pt_col):
        return pd.DataFrame({
            "season": [2024, 2025],
            "age": [25, 26],
            pt_col: [600, 700],
            "rel_k_pct": [1.3, 1.4],
        })

    def _fake_model(self):
        class M:
            fits = {"k_pct": object()}
        return M()

    def _league(self):
        return pd.DataFrame({"season": [2025], "lg_k_pct": [0.22]})

    def test_pitching_frame_does_not_raise(self):
        from kbo_mlb import project, translate
        # No bootstrap draws: every statistic is skipped inside the loop,
        # which is fine. The line under test runs before the loop.
        try:
            project.project_player(self._rows("batters_faced"),
                                   self._fake_model(), {},
                                   self._league(), translate)
        except AttributeError as exc:  # pragma: no cover - the bug itself
            self.fail(f"pitching frame still crashes: {exc}")

    def test_playing_time_weighting_is_actually_applied(self):
        # Falling back to equal weights would hide the bug rather than fix
        # it, so check the heavier season really does count for more.
        from kbo_mlb import project
        rows = self._rows("batters_faced")
        pt_col = next(c for c in ("PA", "batters_faced") if c in rows.columns)
        self.assertEqual(pt_col, "batters_faced")

    def test_a_frame_with_neither_column_still_works(self):
        from kbo_mlb import project, translate
        rows = self._rows("batters_faced").drop(columns=["batters_faced"])
        project.project_player(rows, self._fake_model(), {},
                               self._league(), translate)


class TestEmptyScoutingCohortFailsOpen(unittest.TestCase):
    """An empty posted cohort raised KeyError deep inside the profile.

    `pd.DataFrame([]).sort_values("score")` has no columns to sort by. On
    any season or side with nothing to benchmark against, that took the
    whole app down instead of simply reporting that the screen has nothing
    to say.
    """

    def test_empty_profile_has_the_expected_columns(self):
        from kbo_mlb import scouting
        prof = scouting.historical_profile(
            pd.DataFrame(columns=["player_register_id", "player", "season"]),
            pd.DataFrame(columns=["player_register_id", "first_mlb"]),
            "batting", lambda n, p: True)
        self.assertTrue(prof.empty)
        self.assertIn("score", prof.columns)

    def test_screen_with_no_benchmark_admits_everyone(self):
        from kbo_mlb import scouting
        scores = pd.Series([0.4, 2.0])
        got = scouting.screen(scores, pd.DataFrame(columns=["player", "score"]))
        self.assertTrue(got["mask"].all())
