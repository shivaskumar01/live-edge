"""Live moneylines from The Odds API (the-odds-api.com), v4. The free tier is enough to start.

We request decimal h2h prices and, for each game, line-shop the *best* (max) price available
on each side across all books — the best price is what you'd actually bet into.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {
    "nfl": "americanfootball_nfl",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
}


@dataclass
class GameOdds:
    home_team: str
    away_team: str
    home_decimal: float
    away_decimal: float
    home_book: str
    away_book: str
    commence_time: str
    is_live: bool


def _is_live(commence_time: str) -> bool:
    """A game is in-play if it started in the past and within the last ~5 hours."""
    if not commence_time:
        return False
    try:
        start = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except ValueError:
        return False
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return start < now and (now - start) < timedelta(hours=5)


class OddsClient:
    """Fetch live two-way moneylines for one sport, line-shopping the best price per side."""

    def __init__(
        self, sport: str, api_key: str | None = None, regions: str = "us", timeout: float = 10.0
    ) -> None:
        self.sport = sport.lower()
        if self.sport not in SPORT_KEYS:
            raise ValueError(f"unknown sport {sport!r}; expected one of {sorted(SPORT_KEYS)}")
        self.sport_key = SPORT_KEYS[self.sport]
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not set (pass api_key= or set the env var)")
        self.regions = regions
        self.timeout = timeout
        self.last_remaining: str | None = None  # quota tracking from response headers

    def moneylines(self) -> list[GameOdds]:
        url = f"{BASE}/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        self.last_remaining = resp.headers.get("x-requests-remaining")
        resp.raise_for_status()

        out: list[GameOdds] = []
        for game in resp.json():
            home = game.get("home_team")
            away = game.get("away_team")
            if not home or not away:
                continue

            best_home, best_away = 0.0, 0.0
            home_book, away_book = "", ""
            for bk in game.get("bookmakers", []):
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for oc in market.get("outcomes", []):
                        price = float(oc.get("price", 0) or 0)
                        name = oc.get("name")
                        if name == home and price > best_home:
                            best_home, home_book = price, bk.get("title", "")
                        elif name == away and price > best_away:
                            best_away, away_book = price, bk.get("title", "")

            if best_home <= 1.0 or best_away <= 1.0:
                continue  # incomplete / unusable line

            out.append(
                GameOdds(
                    home_team=home,
                    away_team=away,
                    home_decimal=best_home,
                    away_decimal=best_away,
                    home_book=home_book,
                    away_book=away_book,
                    commence_time=game.get("commence_time", ""),
                    is_live=_is_live(game.get("commence_time", "")),
                )
            )
        return out

    def full_markets(self) -> list[dict]:
        """Fetch h2h + spreads + totals (decimal) from EVERY book, for line-shopping. One call =
        3 credits (3 markets), so callers should cache. Returns per game the full per-book lists;
        the caller picks best prices and computes the no-vig consensus."""
        url = f"{BASE}/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        self.last_remaining = resp.headers.get("x-requests-remaining")
        resp.raise_for_status()

        out: list[dict] = []
        for game in resp.json():
            home, away = game.get("home_team"), game.get("away_team")
            if not home or not away:
                continue
            h2h, spreads, totals = [], [], []
            for bk in game.get("bookmakers", []):
                title = bk.get("title", "")
                for market in bk.get("markets", []):
                    key, ocs = market.get("key"), market.get("outcomes", [])
                    if key == "h2h":
                        h = next((o for o in ocs if o.get("name") == home), None)
                        a = next((o for o in ocs if o.get("name") == away), None)
                        if h and a:
                            h2h.append({"book": title, "home_dec": float(h.get("price", 0) or 0),
                                        "away_dec": float(a.get("price", 0) or 0)})
                    elif key == "spreads":
                        h = next((o for o in ocs if o.get("name") == home), None)
                        a = next((o for o in ocs if o.get("name") == away), None)
                        if h and a:
                            spreads.append({"book": title, "home_point": h.get("point"),
                                            "home_dec": float(h.get("price", 0) or 0),
                                            "away_point": a.get("point"),
                                            "away_dec": float(a.get("price", 0) or 0)})
                    elif key == "totals":
                        ov = next((o for o in ocs if str(o.get("name")).lower() == "over"), None)
                        un = next((o for o in ocs if str(o.get("name")).lower() == "under"), None)
                        if ov and un:
                            totals.append({"book": title, "point": ov.get("point"),
                                           "over_dec": float(ov.get("price", 0) or 0),
                                           "under_dec": float(un.get("price", 0) or 0)})
            out.append({
                "home_team": home, "away_team": away,
                "commence_time": game.get("commence_time", ""),
                "is_live": _is_live(game.get("commence_time", "")),
                "h2h": h2h, "spreads": spreads, "totals": totals,
            })
        return out
