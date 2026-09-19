"""Parsers for Baseball-Reference *register* pages.

Two quirks of these pages drive the design:

1. Most tables below the first one are wrapped in HTML comments, so a naive
   `soup.find_all("table")` sees only a fraction of the page. We un-comment
   them before parsing.
2. Every cell carries a `data-stat` attribute that is stable across pages and
   seasons, while the visible column headers are not. We key on `data-stat`
   and ignore header text entirely.

Page shapes we rely on (verified against live pages, Sept 2026):

  league.cgi?code=KBO&class=Fgn -> table#lg_history, one row per season,
                                   year cell links to league.cgi?id=<hash>
  league.cgi?id=<hash>          -> table#regular_season  (team links)
                                   table#league_batting  (+ League Totals row)
                                   table#league_pitching (+ League Totals row)
  team.cgi?id=<hash>            -> table#team_batting, table#team_pitching,
                                   table#standard_roster (DOB, birthplace)
"""

from __future__ import annotations

import re
from typing import Iterator

from bs4 import BeautifulSoup, Comment

PLAYER_ID_RE = re.compile(r"player\.fcgi\?id=([^&\"']+)")
TEAM_ID_RE = re.compile(r"team\.cgi\?id=([^&\"']+)")
LEAGUE_ID_RE = re.compile(r"league\.cgi\?id=([^&\"']+)")

# A row is a data row if it carries at least one of these. Baseball-Reference
# uses one naming vintage on the register (`year_ID`, `team_ID`) and another
# on the main site's player pages (`year_id`, `team_name_abbr`), so both are
# listed. Missing a name here makes an entire table parse as empty.
_IDENTITY_KEYS = (
    "player", "name_display",
    "team_ID", "team_id", "team_name_abbr",
    "year_ID", "year_id",
)

# Rows that are totals/averages rather than players.
_AGGREGATE_LABELS = {
    "team totals",
    "league totals",
    "league average",
    "team average",
    "totals",
    "average",
}


def make_soup(html: str) -> BeautifulSoup:
    """Parse HTML and promote comment-wrapped tables into the live tree.

    Baseball-Reference ships secondary tables inside `<!-- ... -->` so they
    render client-side. We re-parse those comments and append the tables, then
    de-duplicate by table id (the first table is often present both ways).
    """
    soup = BeautifulSoup(html, "lxml")

    seen_ids = {t.get("id") for t in soup.find_all("table") if t.get("id")}

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if "<table" not in comment:
            continue
        fragment = BeautifulSoup(comment, "lxml")
        for table in fragment.find_all("table"):
            tid = table.get("id")
            if tid and tid in seen_ids:
                continue
            if tid:
                seen_ids.add(tid)
            comment.parent.append(table)

    return soup


def _cell_text(cell) -> str:
    return cell.get_text(strip=True).replace("\xa0", " ")


def _row_is_aggregate(row: dict[str, str]) -> bool:
    for key in ("player", "name_display", "team_ID", "team_id",
                "team_name_abbr"):
        label = (row.get(key) or "").strip().lower()
        if label in _AGGREGATE_LABELS:
            return True
    return False


def parse_table(soup: BeautifulSoup, table_id: str) -> list[dict[str, str]]:
    """Return every body row of `table_id` as a data-stat -> text mapping.

    Also captures `<a href>` targets for player/team/league links, under the
    keys `player_register_id`, `team_bref_id`, `season_bref_id`.
    Header rows repeated mid-table are skipped. Aggregate rows are kept but
    flagged with `is_aggregate`, because league totals are genuinely useful.
    """
    table = soup.find("table", id=table_id)
    if table is None:
        return []

    # League and team totals live in <tfoot>, not <tbody>. Reading only the
    # body silently drops them, which costs us the independent figure the
    # cross-source reconciliation check compares against.
    sections = [s for s in (table.find("tbody"), table.find("tfoot")) if s]
    if not sections:
        sections = [table]

    rows: list[dict[str, str]] = []
    for section in sections:
        for tr in section.find_all("tr", recursive=False):
            row = _parse_row(tr)
            if row is not None:
                rows.append(row)

    return rows


def _parse_row(tr) -> dict[str, str] | None:
    """Parse one <tr> into a data-stat mapping, or None if it isn't a data row."""
    # Mid-table repeated headers carry class="thead".
    if "thead" in (tr.get("class") or []):
        return None

    cells = tr.find_all(["th", "td"], recursive=False)
    if not cells:
        return None

    row: dict[str, str] = {}
    for cell in cells:
        stat = cell.get("data-stat")
        if not stat:
            continue
        row[stat] = _cell_text(cell)

        link = cell.find("a", href=True)
        if link is not None:
            href = link["href"]
            if (m := PLAYER_ID_RE.search(href)):
                row["player_register_id"] = m.group(1)
            elif (m := TEAM_ID_RE.search(href)):
                row["team_bref_id"] = m.group(1)
            elif (m := LEAGUE_ID_RE.search(href)):
                row["season_bref_id"] = m.group(1)

    if not row:
        return None
    # A row with no identifying label is padding.
    if not any(row.get(k) for k in _IDENTITY_KEYS):
        return None

    row["is_aggregate"] = str(_row_is_aggregate(row))
    return row


def parse_season_index(html: str) -> list[dict[str, str]]:
    """From the KBO league index, return one record per season.

    Yields dicts with `season` (int-like str) and `season_bref_id`.
    """
    soup = make_soup(html)
    out = []
    for row in parse_table(soup, "lg_history"):
        year = row.get("year_ID", "").strip()
        season_id = row.get("season_bref_id")
        if not year.isdigit() or not season_id:
            continue
        out.append({"season": year, "season_bref_id": season_id})
    return out


def parse_season_page(html: str) -> dict[str, list[dict[str, str]]]:
    """From a KBO season page, return team links and league-level context.

    Returns {"teams": [...], "league_batting": [...], "league_pitching": [...]}.
    The league tables include the "League Totals" row, which is what the model
    later uses to put every player on a scale relative to his own league-year.
    """
    soup = make_soup(html)

    teams = []
    for row in parse_table(soup, "regular_season"):
        team_id = row.get("team_bref_id")
        if not team_id:
            continue
        teams.append(
            {
                "team_name": row.get("team_ID", "").strip(),
                "team_bref_id": team_id,
                "W": row.get("W", ""),
                "L": row.get("L", ""),
            }
        )

    return {
        "teams": teams,
        "league_batting": parse_table(soup, "league_batting"),
        "league_pitching": parse_table(soup, "league_pitching"),
    }


def parse_team_page(html: str) -> dict[str, list[dict[str, str]]]:
    """From a KBO team-season page, return batting, pitching and roster rows.

    The roster table is the valuable one: it carries date of birth and birth
    city, which give us exact age and a nationality signal without any extra
    requests.
    """
    soup = make_soup(html)
    return {
        "batting": parse_table(soup, "team_batting"),
        "pitching": parse_table(soup, "team_pitching"),
        "roster": parse_table(soup, "standard_roster"),
    }


def iter_player_ids(parsed_team: dict[str, list[dict[str, str]]]) -> Iterator[str]:
    """Every distinct player register ID on a parsed team page."""
    seen: set[str] = set()
    for key in ("batting", "pitching", "roster"):
        for row in parsed_team.get(key, []):
            pid = row.get("player_register_id")
            if pid and pid not in seen:
                seen.add(pid)
                yield pid
