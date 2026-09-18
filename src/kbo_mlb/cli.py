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

from . import config, crosswalk, mlb_data, scrape_kbo, validate
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


def _load(name: str) -> pd.DataFrame:
    path = _path(name)
    if not path.exists():
        print(f"missing {path}. Run `scrape` first.", file=sys.stderr)
        sys.exit(2)
    return pd.read_csv(path, low_memory=False)


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
    cw = config.PROCESSED_DIR / "crosswalk.csv"
    if cw.exists():
        matches = pd.read_csv(cw, low_memory=False)
        stats = {
            "matched": len(matches),
            "kbo_players": roster["player_register_id"].nunique(),
            "match_rate": round(
                len(matches) / max(roster["player_register_id"].nunique(), 1), 4
            ),
        }

    results = validate.run_all(batting, pitching, roster, league_totals,
                               args.start, args.end, stats)

    for r in results:
        print(r)

    report = config.REPORTS_DIR / "data_audit.md"
    report.write_text(validate.to_markdown(results), encoding="utf-8")
    print(f"\n  wrote {report}")

    return 1 if validate.has_errors(results) else 0


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
