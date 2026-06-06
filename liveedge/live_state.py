"""Live game state from ESPN's public scoreboard API, for all three sports.

ESPN's site API is unofficial and undocumented — it is best-effort and can change shape or
rate-limit at any time. Higher-fidelity swaps if you need them: NBA -> nba_api.live.nba
endpoints (scoreboard/boxscore); MLB -> MLB-StatsAPI live game feed (which gives top/bottom of
the inning cleanly, instead of the approximate text parse we do here).

This module only fills the in-game features. `pregame_home_prob` is left at its default; the
monitor seeds it (from the opening market line, or ideally from Elo ratings trained alongside
the model).
"""

from __future__ import annotations

import requests

from liveedge.features import GameState

ENDPOINTS = {
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
}

# Regulation period structure for the clock sports.
_PERIOD_SECONDS = {"nfl": 900.0, "nba": 720.0}
_N_PERIODS = {"nfl": 4, "nba": 4}


def _clock_to_seconds(clock: str) -> float:
    """Parse a display clock to seconds. Handles 'MM:SS', bare seconds, junk (-> 0.0)."""
    if not clock:
        return 0.0
    text = str(clock).strip()
    try:
        if ":" in text:
            mm, ss = text.split(":")
            return float(mm) * 60.0 + float(ss)
        return float(text)
    except ValueError:
        return 0.0


def _game_seconds_remaining(sport: str, period: int, clock_sec: float) -> float:
    """Whole-game seconds left for the clock sports: clock + the remaining full periods.

    Overtime is approximate (periods beyond regulation just contribute their own clock)."""
    ps = _PERIOD_SECONDS.get(sport)
    n = _N_PERIODS.get(sport)
    if ps is None or n is None:
        return clock_sec
    return clock_sec + max(0, n - period) * ps


class ESPNProvider:
    """Fetch in-progress games for one sport and parse them into GameState objects."""

    def __init__(self, sport: str, timeout: float = 10.0) -> None:
        self.sport = sport.lower()
        if self.sport not in ENDPOINTS:
            raise ValueError(f"unknown sport {sport!r}; expected one of {sorted(ENDPOINTS)}")
        self.url = ENDPOINTS[self.sport]
        self.timeout = timeout

    def _fetch(self) -> dict:
        resp = requests.get(
            self.url, headers={"User-Agent": "live-edge/0.1 (personal use)"}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def live_games(self) -> list[GameState]:
        """Return GameState for every in-progress game (status.type.state == 'in')."""
        data = self._fetch()
        games: list[GameState] = []
        for event in data.get("events", []):
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            status = comp.get("status") or event.get("status") or {}
            if (status.get("type") or {}).get("state") != "in":
                continue
            gs = self._parse(event, comp)
            if gs is not None:
                games.append(gs)
        return games

    def _parse(self, event: dict, comp: dict) -> GameState | None:
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if home is None or away is None:
            return None

        home_team = (home.get("team") or {})
        away_team = (away.get("team") or {})
        status = comp.get("status") or {}
        period = int(status.get("period") or 1)
        clock = _clock_to_seconds(status.get("displayClock") or "")
        detail = ((status.get("type") or {}).get("shortDetail") or "")

        gs = GameState(
            sport=self.sport,
            home_team=home_team.get("abbreviation", "HOME"),
            away_team=away_team.get("abbreviation", "AWAY"),
            score_diff=int(home.get("score") or 0) - int(away.get("score") or 0),
            period=period,
            game_id=str(event.get("id", "")),
            description=detail,
            home_full=home_team.get("displayName", ""),
            away_full=away_team.get("displayName", ""),
        )

        if self.sport in ("nfl", "nba"):
            gs.seconds_remaining = _game_seconds_remaining(self.sport, period, clock)

        if self.sport == "nfl":
            sit = comp.get("situation") or {}
            gs.down = int(sit.get("down") or 0)
            gs.ydstogo = int(sit.get("distance") or 0)
            # ESPN's yardLine is approximate as a yards-to-end-zone value; good enough as a
            # weak feature next to score/time.
            gs.yardline_100 = int(sit.get("yardLine") or 50)
            poss = sit.get("possession")
            home_id = home.get("id")
            if poss is not None and home_id is not None:
                gs.posteam_is_home = 1.0 if str(poss) == str(home_id) else 0.0

        elif self.sport == "mlb":
            sit = comp.get("situation") or {}
            gs.outs = int(sit.get("outs") or 0)
            gs.on_first = 1.0 if sit.get("onFirst") else 0.0
            gs.on_second = 1.0 if sit.get("onSecond") else 0.0
            gs.on_third = 1.0 if sit.get("onThird") else 0.0
            gs.inning = period
            # Top/bottom is parsed from the status text (approximate). The MLB-StatsAPI feed
            # provides this cleanly if you need it.
            low = detail.lower()
            gs.is_bottom = 1.0 if ("bot" in low or "bottom" in low) else 0.0

        return gs
