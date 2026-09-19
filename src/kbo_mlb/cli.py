"""Command-line entry point.

    python -m kbo_mlb.cli scrape     # crawl KBO pages (slow, cached)
    python -m kbo_mlb.cli crosswalk  # build the KBO <-> MLB player links
    python -m kbo_mlb.cli audit      # run data quality checks, write report
    python -m kbo_mlb.cli counts     # how many two-league players do we have?
    python -m kbo_mlb.cli all        # scrape -> crosswalk -> audit -> counts
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import (config, crosswalk, mlb_bref, mlb_data, mlb_statsapi,
               scrape_kbo, validate)
from .http_client import CachedFetcher


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _path(name: str):
    return config.INTERIM_DIR / name


def cmd_scrape(args) -> int:
    fetcher = CachedFetcher(offline=args.offline)
    result = scrape_kbo.crawl(fetcher, args.start, args.end)

    for name, df in (
        ("kbo_batting.csv", result.batting),
        ("kbo_pitching.csv", result.pitching),
        ("kbo_roster.csv", result.roster),
        ("kbo_league_totals.csv", result.league_totals),
    ):
        df.to_csv(_path(name), index=False)
        print(f"  wrote {len(df):>6} rows -> {_path(name)}")

    print(f"\ncache hits: {fetcher.stats['cache_hits']}, "
          f"network fetches: {fetcher.stats['network_fetches']}")
    return 0


def _load(name: str, required: bool = True) -> pd.DataFrame:
    """Read an interim table.

    An empty file is a real outcome (a table the parser found nothing for),
    not a crash: return an empty frame so the audit can report the gap as a
    finding instead of dying on a traceback.
    """
    path = _path(name)
    if not path.exists():
        if required:
            print(f"missing {path}. Run `scrape` first.", file=sys.stderr)
            sys.exit(2)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        print(f"  note: {path.name} is empty", file=sys.stderr)
        return pd.DataFrame()


def _kbo_player_index(roster: pd.DataFrame) -> pd.DataFrame:
    """One row per KBO player, with the best available date of birth."""
    cols = [c for c in ("player_register_id", "player", "date_of_birth",
                        "birth_city") if c in roster.columns]
    idx = roster[cols].copy()
    idx = idx.sort_values("date_of_birth").drop_duplicates(
        subset=["player_register_id"], keep="first")
    return idx.reset_index(drop=True)


def cmd_crosswalk(args) -> int:
    roster = _load("kbo_roster.csv")
    kbo_players = _kbo_player_index(roster)

    people = mlb_data.chadwick_with_names(mlb_data.load_chadwick())
    mlb_players = mlb_data.mlb_players_who_played(people)

    result = crosswalk.build(kbo_players, mlb_players, chadwick=people)

    result.matches.to_csv(config.PROCESSED_DIR / "crosswalk.csv", index=False)
    result.review_queue.to_csv(
        config.PROCESSED_DIR / "crosswalk_review_queue.csv", index=False)

    print("crosswalk stats:")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")
    print(f"\n  wrote {config.PROCESSED_DIR / 'crosswalk.csv'}")
    print(f"  wrote {config.PROCESSED_DIR / 'crosswalk_review_queue.csv'}")
    return 0


def cmd_audit(args) -> int:
    batting = _load("kbo_batting.csv")
    pitching = _load("kbo_pitching.csv")
    roster = _load("kbo_roster.csv")
    league_totals = _load("kbo_league_totals.csv")

    stats = None
    matches = None
    cw = config.PROCESSED_DIR / "crosswalk.csv"
    if cw.exists():
        matches = pd.read_csv(cw, low_memory=False)
        n_players = roster["player_register_id"].nunique()
        stats = {
            "matched": matches["player_register_id"].nunique(),
            "crosswalk_rows": len(matches),
            "kbo_players": n_players,
        }

    results = validate.run_all(batting, pitching, roster, league_totals,
                               args.start, args.end, stats, matches)

    for r in results:
        print(r)

    report = config.REPORTS_DIR / "data_audit.md"
    report.write_text(validate.to_markdown(results), encoding="utf-8")
    print(f"\n  wrote {report}")

    return 1 if validate.has_errors(results) else 0


def cmd_mlb(args) -> int:
    """Pull MLB season stats for the matched players. Needs network.

    Two independent sources, because having both is what makes a real
    cross-source check on the MLB side possible - the same thing the KBO
    side already does against published league totals.

      statsapi (default)  MLB's official public API. JSON, keyed by MLBAM
                          id, roughly two minutes for the whole pull.
      bref                Baseball-Reference player and league pages. Same
                          provider as the KBO side, so identical
                          definitional rules, but ~30 minutes at the polite
                          crawl delay.

    FanGraphs is not used: pybaseball reaches it through a legacy endpoint
    that now answers 403.
    """
    cw_path = config.PROCESSED_DIR / "crosswalk.csv"
    if not cw_path.exists():
        print("run `crosswalk` first", file=sys.stderr)
        return 2
    cw = pd.read_csv(cw_path, low_memory=False)
    seasons = list(range(args.start, args.end + 1))
    suffix = "" if args.source == "statsapi" else "_bref"

    if args.source == "statsapi":
        ids = cw["key_mlbam"].dropna().astype(int).unique().tolist()
        print(f"fetching {len(ids)} matched players from MLB Stats API...")
        fetcher = mlb_statsapi.make_fetcher(offline=args.offline)

        for side in ("batting", "pitching"):
            df = mlb_statsapi.fetch_player_seasons(ids, side, fetcher)
            name = f"mlb_{side}{suffix}.csv"
            df.to_csv(_path(name), index=False)
            print(f"  {side}: {len(df):>6} player-seasons -> {_path(name)}")

        totals = pd.concat(
            [mlb_statsapi.fetch_league_totals(seasons, s, fetcher)
             for s in ("batting", "pitching")], ignore_index=True)
        totals.to_csv(_path(f"mlb_league_totals{suffix}.csv"), index=False)
        print(f"  league totals: {len(totals)} rows")
        print(f"\ncache hits: {fetcher.stats['cache_hits']}, "
              f"network fetches: {fetcher.stats['network_fetches']}")
        return 0

    ids = sorted(cw["key_bbref"].dropna().astype(str).unique())
    print(f"{len(ids)} matched players to pull "
          f"(~{len(ids) * 3.5 / 60:.0f} min at the polite rate)")

    fetcher = CachedFetcher(offline=args.offline)
    got = mlb_bref.fetch_players(fetcher, ids)

    # Written under a distinct suffix so the two sources never overwrite one
    # another - keeping both side by side is what allows the cross-check.
    for name, df in ((f"mlb_batting{suffix}.csv", got["batting"]),
                     (f"mlb_pitching{suffix}.csv", got["pitching"])):
        df.to_csv(_path(name), index=False)
        print(f"  wrote {len(df):>6} player-seasons -> {_path(name)}")

    if not got["failures"].empty:
        got["failures"].to_csv(_path("mlb_fetch_failures.csv"), index=False)
        print(f"  {len(got['failures'])} players failed "
              f"-> {_path('mlb_fetch_failures.csv')}")

    years = list(range(args.start, args.end + 1))
    print(f"pulling MLB league totals for {len(years)} seasons...")
    totals = mlb_bref.fetch_league_totals(fetcher, years)
    totals.to_csv(_path(f"mlb_league_totals{suffix}.csv"), index=False)
    print(f"  wrote {len(totals):>6} rows -> {_path('mlb_league_totals.csv')}")

    print(f"\ncache hits: {fetcher.stats['cache_hits']}, "
          f"network fetches: {fetcher.stats['network_fetches']}")
    return 0


def cmd_counts(args) -> int:
    """Sample size for the translation model - decides hitters vs pitchers."""
    cw = config.PROCESSED_DIR / "crosswalk.csv"
    if not cw.exists():
        print("run `crosswalk` first", file=sys.stderr)
        return 2
    matches = pd.read_csv(cw, low_memory=False)
    batting = _load("kbo_batting.csv")
    pitching = _load("kbo_pitching.csv")

    matched_ids = set(matches["player_register_id"].dropna())

    bat = batting[batting["player_register_id"].isin(matched_ids)]
    pit = pitching[pitching["player_register_id"].isin(matched_ids)]

    qualified_bat = bat[pd.to_numeric(bat.get("PA"), errors="coerce")
                        >= config.MIN_PA_FOR_MODEL]
    qualified_pit = pit[pd.to_numeric(pit.get("batters_faced"),
                                      errors="coerce")
                        >= config.MIN_BF_FOR_MODEL]

    print("Players with both KBO and MLB records")
    print(f"  crosswalk links               : {len(matched_ids)}")
    print(f"  hitters  (any KBO season)     : "
          f"{bat['player_register_id'].nunique()}")
    print(f"  hitters  (>= {config.MIN_PA_FOR_MODEL} PA season) : "
          f"{qualified_bat['player_register_id'].nunique()}")
    print(f"  pitchers (any KBO season)     : "
          f"{pit['player_register_id'].nunique()}")
    print(f"  pitchers (>= {config.MIN_BF_FOR_MODEL} BF season) : "
          f"{qualified_pit['player_register_id'].nunique()}")
    print("\nPick whichever group is larger for the Day 4 translation model.")
    return 0


def cmd_all(args) -> int:
    for fn in (cmd_scrape, cmd_crosswalk, cmd_audit, cmd_counts):
        print(f"\n{'=' * 60}\n{fn.__name__}\n{'=' * 60}")
        code = fn(args)
        if code not in (0, 1):
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared flags live on a parent parser so they are accepted either before
    # or after the subcommand: `cli --start 2000 scrape` and
    # `cli scrape --start 2000` both work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    common.add_argument("--start", type=int,
                        default=config.DEFAULT_START_SEASON)
    common.add_argument("--end", type=int, default=config.DEFAULT_END_SEASON)
    common.add_argument("--source", choices=("statsapi", "bref"),
                        default="statsapi",
                        help="MLB data source for the `mlb` command "
                             "(default: statsapi)")
    common.add_argument("--offline", action="store_true",
                        help="fail instead of hitting the network "
                             "(uses only the on-disk cache)")

    parser = argparse.ArgumentParser(
        prog="kbo_mlb", description="KBO to MLB translation pipeline",
        parents=[common])

    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_text in (
        ("scrape", cmd_scrape, "crawl KBO season/team/player pages"),
        ("crosswalk", cmd_crosswalk, "build the KBO <-> MLB player crosswalk"),
        ("mlb", cmd_mlb, "pull MLB season stats (needs network)"),
        ("audit", cmd_audit, "run data quality checks"),
        ("counts", cmd_counts, "report two-league sample sizes"),
        ("all", cmd_all, "run the whole pipeline"),
    ):
        p = sub.add_parser(name, help=help_text, parents=[common])
        p.set_defaults(func=fn)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
