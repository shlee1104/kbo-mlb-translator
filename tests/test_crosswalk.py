"""Tests for the tiered KBO <-> MLB crosswalk."""

import unittest

import pandas as pd

from kbo_mlb import crosswalk


def kbo_players():
    return pd.DataFrame({
        "player_register_id": ["lee---001jun", "kim---001has", "fedde-001eri",
                               "nomatch-001xyz"],
        "player": ["Jung-hoo Lee", "Ha-seong Kim", "Erick Fedde",
                   "Sang-min Park"],
        "date_of_birth": ["1998-08-20", "1995-10-17", "1993-02-25",
                          "1999-01-01"],
    })


def mlb_players():
    return pd.DataFrame({
        "key_mlbam": [808982, 673490, 657049],
        # Deliberately different spellings and orders from the KBO side.
        "name": ["Jung Hoo Lee", "Yi Jeong-hu placeholder", "Erick Fedde"],
        "date_of_birth": ["1998-08-20", "1970-01-01", "1993-02-25"],
    })


class TestNameBasedTiers(unittest.TestCase):
    def test_matches_on_name_and_dob(self):
        result = crosswalk.build(kbo_players(), mlb_players())
        matched = set(result.matches["player_register_id"])
        self.assertIn("lee---001jun", matched)
        self.assertIn("fedde-001eri", matched)

    def test_unmatched_players_reach_the_review_queue(self):
        result = crosswalk.build(kbo_players(), mlb_players())
        queued = set(result.review_queue["player_register_id"])
        self.assertIn("nomatch-001xyz", queued)

    def test_records_the_tier_used(self):
        result = crosswalk.build(kbo_players(), mlb_players())
        self.assertIn("match_tier", result.matches.columns)
        self.assertTrue(
            set(result.matches["match_tier"]) <= set(crosswalk.TIER_CONFIDENCE))

    def test_phonetic_tier_catches_romanization_variants(self):
        kbo = pd.DataFrame({
            "player_register_id": ["lee---001jun"],
            "player": ["Yi Jeong-hu"],          # McCune-Reischauer spelling
            "date_of_birth": ["1998-08-20"],
        })
        mlb = pd.DataFrame({
            "key_mlbam": [808982],
            "name": ["Jung Hoo Lee"],           # Revised Romanization
            "date_of_birth": ["1998-08-20"],
        })
        result = crosswalk.build(kbo, mlb)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches.iloc[0]["match_tier"], "phonetic_dob")

    def test_birth_date_gates_a_phonetic_match(self):
        # Same phonetic key, different person. Must not auto-match.
        kbo = pd.DataFrame({
            "player_register_id": ["kim---001x"],
            "player": ["Kim Min-seok"],
            "date_of_birth": ["1998-08-20"],
        })
        mlb = pd.DataFrame({
            "key_mlbam": [1],
            "name": ["Min-suk Kim"],
            "date_of_birth": ["1975-03-03"],
        })
        result = crosswalk.build(kbo, mlb)
        tiers = (set(result.matches["match_tier"])
                 if "match_tier" in result.matches.columns else set())
        self.assertNotIn("phonetic_dob", tiers)
        self.assertNotIn("name_dob", tiers)
        # It should instead be offered for review, not silently dropped.
        self.assertIn("kim---001x",
                      set(result.review_queue["player_register_id"]))


class TestDeterministicTier(unittest.TestCase):
    def test_register_id_tier_takes_priority(self):
        chadwick = pd.DataFrame({
            "key_bbref_minors": ["lee---001jun"],
            "key_mlbam": [808982],
            "key_fangraphs": [32262],
        })
        result = crosswalk.build(kbo_players(), mlb_players(),
                                 chadwick=chadwick)
        row = result.matches.query("player_register_id == 'lee---001jun'")
        self.assertEqual(row.iloc[0]["match_tier"], "register_id")
        self.assertEqual(row.iloc[0]["match_confidence"], 1.0)


class TestAmbiguity(unittest.TestCase):
    def test_ambiguous_keys_are_not_auto_matched(self):
        # Two different MLB players share a name and birth date. Guessing
        # between them would be worse than leaving it for a human.
        kbo = pd.DataFrame({
            "player_register_id": ["kim---001x"],
            "player": ["Kim Min-seok"],
            "date_of_birth": ["1998-08-20"],
        })
        mlb = pd.DataFrame({
            "key_mlbam": [1, 2],
            "name": ["Min-seok Kim", "Min-seok Kim"],
            "date_of_birth": ["1998-08-20", "1998-08-20"],
        })
        result = crosswalk.build(kbo, mlb)
        self.assertTrue(result.matches.empty)
        self.assertIn("kim---001x", set(result.review_queue["player_register_id"]))


class TestManualOverrides(unittest.TestCase):
    def test_accepted_override_is_folded_in(self):
        result = crosswalk.build(kbo_players(), mlb_players())
        overrides = pd.DataFrame({
            "player_register_id": ["nomatch-001xyz"],
            "key_mlbam": [999999],
            "decision": ["accept"],
        })
        updated = crosswalk.apply_manual_overrides(result, overrides)
        self.assertIn("nomatch-001xyz",
                      set(updated.matches["player_register_id"]))
        self.assertNotIn("nomatch-001xyz",
                         set(updated.review_queue["player_register_id"]))

    def test_rejected_override_leaves_the_queue(self):
        result = crosswalk.build(kbo_players(), mlb_players())
        overrides = pd.DataFrame({
            "player_register_id": ["nomatch-001xyz"],
            "key_mlbam": [999999],
            "decision": ["reject"],
        })
        updated = crosswalk.apply_manual_overrides(result, overrides)
        self.assertNotIn("nomatch-001xyz",
                         set(updated.review_queue["player_register_id"]))
        self.assertNotIn("nomatch-001xyz",
                         set(updated.matches["player_register_id"]))


class TestStats(unittest.TestCase):
    def test_stats_are_reported(self):
        result = crosswalk.build(kbo_players(), mlb_players())
        self.assertEqual(result.stats["kbo_players"], 4)
        self.assertIn("match_rate", result.stats)
        self.assertLessEqual(result.stats["match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
