"""Central configuration: paths, source URLs, and crawl politeness settings.

Everything that a reviewer might want to change lives here, so no other module
hard-codes a path, a URL, or a sleep interval.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("KBO_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"           # unmodified source payloads (HTML, CSV)
INTERIM_DIR = DATA_DIR / "interim"   # parsed but not yet reconciled
PROCESSED_DIR = DATA_DIR / "processed"  # analysis-ready tables
CACHE_DIR = DATA_DIR / "raw" / "html_cache"

REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CACHE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

BREF_BASE = "https://www.baseball-reference.com"

# The KBO lives in Baseball-Reference's *register* (minor/foreign leagues),
# not the major-league site. Register pages are allowed by their robots.txt.
KBO_LEAGUE_INDEX = f"{BREF_BASE}/register/league.cgi?code=KBO&class=Fgn"

# Chadwick Bureau register: the canonical open player-ID crosswalk.
# `key_bbref_minors` holds the same register ID that KBO pages link to.
CHADWICK_BASE = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
)
CHADWICK_PARTS = [f"people-{i}.csv" for i in "0123456789abcdef"]


# ---------------------------------------------------------------------------
# Crawl politeness
# ---------------------------------------------------------------------------
#
# Baseball-Reference publishes two limits that we respect together:
#   * robots.txt         -> Crawl-delay: 3
#   * /bot-traffic.html  -> no more than 20 requests per minute; exceeding it
#                           gets your session blocked for up to a day.
#
# We default to a 3.5s delay and cap ourselves at 16 requests/minute, which
# sits comfortably inside both. Do not lower these.

REQUEST_DELAY_SECONDS = float(os.environ.get("KBO_REQUEST_DELAY", 3.5))
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("KBO_MAX_RPM", 16))
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

USER_AGENT = os.environ.get(
    "KBO_USER_AGENT",
    # Identifying yourself is the polite default. Put a real contact address
    # here before running a large crawl.
    "kbo-mlb-translator/0.1 (research project; contact: seung82211@gmail.com)",
)


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
#
# The KBO's run environment changed enough over its history that seasons from
# the 1980s are of limited use for translating today's players. The default
# window is configurable; 2000+ keeps the crawl to roughly 250 pages.

DEFAULT_START_SEASON = int(os.environ.get("KBO_START_SEASON", 2000))
DEFAULT_END_SEASON = int(os.environ.get("KBO_END_SEASON", 2026))

# Minimum playing time for a season to be usable in the translation model.
MIN_PA_FOR_MODEL = 100
MIN_BF_FOR_MODEL = 100
