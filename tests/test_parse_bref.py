"""Tests for the Baseball-Reference register parsers.

The fixture reproduces the two quirks that break naive scrapers: tables hidden
inside HTML comments, and repeated header rows in the middle of a tbody.
"""

import unittest
from pathlib import Path

from kbo_mlb import parse_bref, scrape_kbo

FIXTURE = Path(__file__).parent / "fixtures" / "team_page.html"


class TestCommentedTables(unittest.TestCase):
    def setUp(self):
        self.html = FIXTURE.read_text(encoding="utf-8")

    def test_finds_tables_hidden_in_comments(self):
        soup = parse_bref.make_soup(self.html)
        ids = {t.get("id") for t in soup.find_all("table")}
        self.assertIn("team_batting", ids)      # live in the DOM
        self.assertIn("team_pitching", ids)     # inside a comment
        self.assertIn("standard_roster", ids)   # inside a comment

    def test_does_not_duplicate_live_tables(self):
        soup = parse_bref.make_soup(self.html)
        batting = soup.find_all("table", id="team_batting")
        self.assertEqual(len(batting), 1)


class TestParseTeamPage(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_bref.parse_team_page(
            FIXTURE.read_text(encoding="utf-8"))

    def test_skips_repeated_header_rows(self):
        # Two players plus one Team Totals row; the mid-table header must not
        # become a third "player".
        names_seen = [r["player"] for r in self.parsed["batting"]]
        self.assertNotIn("Name", names_seen)
        self.assertEqual(len(self.parsed["batting"]), 3)

    def test_flags_aggregate_rows(self):
        aggregates = [r for r in self.parsed["batting"]
                      if r["is_aggregate"] == "True"]
        self.assertEqual(len(aggregates), 1)
        self.assertEqual(aggregates[0]["player"], "Team Totals")

    def test_extracts_player_register_ids(self):
        ids = {r.get("player_register_id") for r in self.parsed["batting"]}
        self.assertIn("lee---001jun", ids)
        self.assertIn("brito-001soc", ids)

    def test_parses_commented_pitching_table(self):
        pitching = self.parsed["pitching"]
        self.assertEqual(len(pitching), 2)
        fedde = next(r for r in pitching if "Fedde" in r["player"])
        self.assertEqual(fedde["earned_run_avg"], "2.00")
        self.assertEqual(fedde["player_register_id"], "fedde-001eri")

    def test_parses_roster_birth_details(self):
        roster = self.parsed["roster"]
        lee = next(r for r in roster if "Lee" in r["player"])
        self.assertEqual(lee["date_of_birth"], "1998-08-20")
        self.assertEqual(lee["birth_city"], "Nagoya, Japan")

    def test_iter_player_ids_is_deduplicated(self):
        ids = list(parse_bref.iter_player_ids(self.parsed))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("lee---001jun", ids)


class TestInningsConversion(unittest.TestCase):
    """Innings pitched are thirds, not decimals. 5.1 is 5 1/3, not 5.1."""

    def test_thirds_notation(self):
        self.assertEqual(scrape_kbo.innings_to_outs(5.0), 15.0)
        self.assertEqual(scrape_kbo.innings_to_outs(5.1), 16.0)
        self.assertEqual(scrape_kbo.innings_to_outs(5.2), 17.0)
        self.assertEqual(scrape_kbo.innings_to_outs(180.1), 541.0)

    def test_handles_bad_input(self):
        self.assertIsNone(scrape_kbo.innings_to_outs(None))
        self.assertIsNone(scrape_kbo.innings_to_outs("not a number"))


if __name__ == "__main__":
    unittest.main()
