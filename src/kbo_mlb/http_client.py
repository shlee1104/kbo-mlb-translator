"""A deliberately slow, cached HTTP client.

Two jobs:

1. Stay inside Baseball-Reference's published limits (3s crawl-delay, 20
   requests/minute). We enforce *both* a per-request delay and a rolling
   per-minute token bucket, so a burst can't sneak past the delay check.
2. Cache every response on disk. A re-run of the pipeline then costs zero
   requests, which matters when you are iterating on the parser.

The cache is keyed by URL hash and stores the raw bytes exactly as received,
so the parser can be re-run and re-tested against the real payloads offline.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import deque
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)


class RateLimiter:
    """Enforces a minimum gap between requests AND a rolling per-minute cap."""

    def __init__(self, delay_seconds: float, max_per_minute: int):
        self.delay_seconds = delay_seconds
        self.max_per_minute = max_per_minute
        self._timestamps: deque[float] = deque()
        self._last_request: float | None = None

    def wait(self) -> None:
        now = time.monotonic()

        # 1. Minimum gap since the previous request.
        if self._last_request is not None:
            elapsed = now - self._last_request
            if elapsed < self.delay_seconds:
                time.sleep(self.delay_seconds - elapsed)

        # 2. Rolling one-minute window.
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 60.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_minute:
            sleep_for = 60.0 - (now - self._timestamps[0]) + 0.1
            log.info("Per-minute cap reached; sleeping %.1fs", sleep_for)
            time.sleep(max(sleep_for, 0))
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > 60.0:
                self._timestamps.popleft()

        stamp = time.monotonic()
        self._timestamps.append(stamp)
        self._last_request = stamp


class CachedFetcher:
    """Fetch URLs politely, caching responses on disk."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        delay_seconds: float | None = None,
        max_per_minute: int | None = None,
        user_agent: str | None = None,
        offline: bool = False,
    ):
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(
            delay_seconds if delay_seconds is not None else config.REQUEST_DELAY_SECONDS,
            max_per_minute if max_per_minute is not None else config.MAX_REQUESTS_PER_MINUTE,
        )
        self.offline = offline
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent or config.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.stats = {"cache_hits": 0, "network_fetches": 0, "failures": 0}

    def cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.html"

    def get(self, url: str, force: bool = False) -> str:
        """Return the page text, from cache when possible.

        Raises RuntimeError if the page is not cached and `offline` is set,
        so that tests and reruns never silently hit the network.
        """
        path = self.cache_path(url)
        if path.exists() and not force:
            self.stats["cache_hits"] += 1
            return path.read_text(encoding="utf-8", errors="replace")

        if self.offline:
            raise RuntimeError(
                f"offline=True and no cached copy of {url}\n"
                f"(expected at {path})"
            )

        last_error: Exception | None = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:  # network-level failure
                last_error = exc
                log.warning("attempt %d/%d failed for %s: %s",
                            attempt, config.MAX_RETRIES, url, exc)
                time.sleep(2**attempt + random.random())
                continue

            if resp.status_code == 200:
                self.stats["network_fetches"] += 1
                path.write_text(resp.text, encoding="utf-8")
                return resp.text

            if resp.status_code == 429:
                # We are being told to slow down; back off hard rather than
                # risking the day-long block described in bot-traffic.html.
                wait = 60 * attempt
                log.warning("429 from %s - backing off %ds", url, wait)
                time.sleep(wait)
                last_error = RuntimeError("HTTP 429")
                continue

            if 500 <= resp.status_code < 600:
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(2**attempt + random.random())
                continue

            # 403/404 and friends: no point retrying.
            self.stats["failures"] += 1
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")

        self.stats["failures"] += 1
        raise RuntimeError(f"giving up on {url}: {last_error}")
