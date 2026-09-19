"""Tests for the MLB-side Baseball-Reference parser.

The main site's player pages use a different data-stat vintage than the
register does (`year_id` / `b_pa` rather than `year_ID` / `PA`). Tables are
found by the columns they contain rather than by id, so a redesign that
renames a table does not silently produce an empty result.
"""

import unittest

from kbo_mlb import mlb_bref

PLAYER_PAGE = """
<table id="players_standard_batting"><thead><tr>
<th data-stat="year_id">Year</th><th data-stat="b_age">Age</th>
<th data-stat="team_name_abbr">Tm</th><th data-stat="comp_name_abbr">Lg</th>
<th data-stat="b_pa">PA</th><th data-stat="b_ab">AB</th><th data-stat="b_h">H</th>
<th data-stat="b_hr">HR</th><th data-stat="b_so">SO</th><th data-stat="b_bb">BB</th>
<th data-stat="b_tb">TB</th></tr></thead>
<tbody>
<tr><th data-stat="year_id">2024</th><td data-stat="b_age">25</td>
<td data-stat="team_name_abbr">SFG</td><td data-stat="comp_name_abbr">NL</td>
<td data-stat="b_pa">37</td><td data-stat="b_ab">33</td><td data-stat="b_h">8</td>
<td data-stat="b_hr">2</td><td data-stat="b_so">6</td><td data-stat="b_bb">3</td>
<td data-stat="b_tb">14</td></tr>
<tr><th data-stat="year_id">2025</th><td data-stat="b_age">26</td>
<td data-stat="team_name_abbr">2TM</td><td data-stat="comp_name_abbr">NL</td>
<td data-stat="b_pa">400</td><td data-stat="b_ab">360</td><td data-stat="b_h">100</td>
<td data-stat="b_hr">10</td><td data-stat="b_so">50</td><td data-stat="b_bb">35</td>
<td data-stat="b_tb">160</td></tr>
<tr><th data-stat="year_id">2025</th><td data-stat="b_age">26</td>
<td data-stat="team_name_abbr">SFG</td><td data-stat="comp_name_abbr">NL</td>
<td data-stat="b_pa">250</td><td data-stat="b_ab">225</td><td data-stat="b_h">62</td>
<td data-stat="b_hr">6</td><td data-stat="b_so">31</td><td data-stat="b_bb">22</td>
<td data-stat="b_tb">100</td></tr>
</tbody>
<tfoot>
<tr><th data-stat="year_id"></th><td data-stat="team_name_abbr">Career</td>
<td data-stat="b_pa">637</td><td data-stat="b_ab">573</td><td data-stat="b_h">168</td>
<td data-stat="b_hr">17</td><td data-stat="b_so">76</td><td data-stat="b_bb">53</td>
<td data-stat="b_tb">254</td><td data-stat="b_age"></td></tr>
</tfoot></table>

<!--
<table id="players_standard_pitching"><thead><tr>
<th data-stat="year_id">Year</th><th data-stat="p_age">Age</th>
<th data-stat="team_name_abbr">Tm</th><th data-stat="p_ip">IP</th>
<th data-stat="p_er">ER</th><th data-stat="p_so">SO</th>
<th data-stat="p_bb">BB</th><th data-stat="p_hr">HR</th>
<th data-stat="p_bfp">BF</th><th data-stat="p_earned_run_avg">ERA</th></tr></thead>
<tbody>
<tr><th data-stat="year_id">2019</th><td data-stat="p_age">32</td>
<td data-stat="team_name_abbr">LAD</td><td data-stat="p_ip">182.2</td>
<td data-stat="p_er">47</td><td data-stat="p_so">163</td>
<td data-stat="p_bb">24</td><td data-stat="p_hr">17</td>
<td data-stat="p_bfp">711</td><td data-stat="p_earned_run_avg">2.32</td></tr>
</tbody></table>
-->
"""


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.tables = mlb_bref.discover_tables(PLAYER_PAGE)

    def test_finds_both_sides(self):
        self.assertIn("batting", self.tables)
        self.assertIn("pitching", self.tables)

    def test_finds_pitching_inside_an_html_comment(self):
        rows = self.tables["pitching"]
        self.assertTrue(any(r.get("season") == "2019" for r in rows))

    def test_canonicalises_the_new_data_stat_names(self):
        row = self.tables["batting"][0]
        # b_pa -> PA, year_id -> season, team_name_abbr -> team
        for key in ("season", "PA", "AB", "HR", "team", "age"):
            self.assertIn(key, row)


class TestSeasonCleaning(unittest.TestCase):
    def setUp(self):
        tables = mlb_bref.discover_tables(PLAYER_PAGE)
        self.batting = mlb_bref._clean_seasons(tables["batting"], "batting")
        self.pitching = mlb_bref._clean_seasons(tables["pitching"], "pitching")

    def test_drops_the_career_row(self):
        self.assertNotIn(637, list(self.batting["PA"]))
        self.assertEqual(len(self.batting), 2)

    def test_collapses_a_midseason_trade_to_the_combined_line(self):
        # 2025 appears twice: the combined 2TM row and one team's stint.
        # Double counting would inflate the season; dropping the wrong one
        # would understate it.
        row_2025 = self.batting[self.batting["season"] == 2025]
        self.assertEqual(len(row_2025), 1)
        self.assertEqual(int(row_2025["PA"].iloc[0]), 400)

    def test_seasons_are_numeric_and_sorted(self):
        self.assertEqual(list(self.batting["season"]), [2024, 2025])

    def test_pitching_keeps_thirds_notation_intact(self):
        # 182.2 must survive as-is so the rate layer can read it as thirds.
        self.assertAlmostEqual(float(self.pitching["IP"].iloc[0]), 182.2)
        self.assertEqual(int(self.pitching["batters_faced"].iloc[0]), 711)


class TestUrls(unittest.TestCase):
    def test_player_url_uses_the_first_letter_directory(self):
        self.assertTrue(
            mlb_bref.player_url("ryuhy01").endswith("/players/r/ryuhy01.shtml"))

    def test_league_url_by_side(self):
        self.assertIn("2024-standard-batting",
                      mlb_bref.league_url(2024, "batting"))
        self.assertIn("2024-standard-pitching",
                      mlb_bref.league_url(2024, "pitching"))


if __name__ == "__main__":
    unittest.main()
