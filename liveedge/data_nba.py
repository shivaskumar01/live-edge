"""Historical NBA loader (nba_api / stats.nba.com).

There is no prebuilt win-prob table, so we reconstruct (score_diff, seconds_remaining) from
PlayByPlayV3 events. (PlayByPlayV2 is deprecated and the NBA API now returns empty data for it
— see nba_api issue #591 — so V3 is required.)

stats.nba.com rate-limits aggressively, so every network result (the season schedule and each
game's play-by-play) is cached to `.cache/` via `liveedge.cache`. The first full-season pull is
slow (~a quarter-hour), but it is a ONE-TIME cost and is resumable: if a run gets throttled
part-way, only the missing games are re-fetched next time. Use `max_games` to bound a first run;
raise it (e.g. 2000) for a full season. Delete `.cache/` to force a refresh.
"""

from __future__ import annotations

import re
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

from liveedge.cache import cached_frame
from liveedge.elo import EloModel
from liveedge.features import get_spec

_NBA_REGULATION_PERIODS = 4
_NBA_PERIOD_SECONDS = 720.0  # 12-minute quarters
_PBP_COLUMNS = ["period", "clock", "scoreHome", "scoreAway"]
_GAME_COLUMNS = ["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ABBREVIATION", "WL"]


def _clock_to_seconds(clock: str) -> float:
    """Parse a game clock to seconds. Handles PlayByPlayV3's ISO 8601 form 'PT11M34.00S',
    plain 'MM:SS', bare seconds, and junk (-> 0.0)."""
    if not clock:
        return 0.0
    text = str(clock).strip()
    iso = re.match(r"PT(\d+)M([\d.]+)S", text)
    if iso:
        return int(iso.group(1)) * 60.0 + float(iso.group(2))
    try:
        if ":" in text:
            mm, ss = text.split(":")
            return float(mm) * 60.0 + float(ss)
        return float(text)
    except ValueError:
        return 0.0


def _nba_seconds_remaining(period: int, clock_sec: float) -> float:
    """Whole-game seconds left. Regulation = clock + remaining full quarters; OT ~= clock only."""
    if period <= _NBA_REGULATION_PERIODS:
        return clock_sec + max(0, _NBA_REGULATION_PERIODS - period) * _NBA_PERIOD_SECONDS
    return clock_sec  # overtime is approximate — just the remaining OT clock


def _league_games(season_str: str, sleep: float) -> pd.DataFrame | None:
    """Cached season schedule (one row per team per game)."""

    def fetch() -> pd.DataFrame:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str, league_id_nullable="00"
        ).get_data_frames()[0]
        time.sleep(sleep)
        return gf[_GAME_COLUMNS]

    return cached_frame(f"nba_games_{season_str}", fetch)


def _game_pbp(game_id: str, sleep: float) -> pd.DataFrame | None:
    """Cached play-by-play for one game (only the columns we need). None on a failed pull."""

    def fetch() -> pd.DataFrame | None:
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
            time.sleep(sleep)
            return df[_PBP_COLUMNS]
        except Exception:
            time.sleep(sleep)
            return None

    return cached_frame(f"nba_pbp_{game_id}", fetch)


def load_nba_frame(seasons: list[int], max_games: int = 200, sleep: float = 0.6) -> pd.DataFrame:
    """Load NBA play-by-play into the NBA feature contract + home_win label.

    Each season string is like '2021-22'. Each game appears twice in the schedule (a home row
    and an away row); we anchor on the home row ('vs.' in MATCHUP).
    """
    spec = get_spec("nba")
    rows: list[dict] = []
    game_results: list[dict] = []

    for season in seasons:
        season_str = f"{season}-{str(season + 1)[-2:]}"
        games_df = _league_games(season_str, sleep)
        if games_df is None or not len(games_df):
            continue

        home_rows = (
            games_df[games_df["MATCHUP"].str.contains("vs.", regex=False)]
            .sort_values("GAME_DATE")
            .head(max_games)
        )

        for g in home_rows.itertuples():
            gid = g.GAME_ID
            other = games_df[
                (games_df["GAME_ID"] == gid)
                & (games_df["TEAM_ABBREVIATION"] != g.TEAM_ABBREVIATION)
            ]
            if other.empty:
                continue
            game_results.append(
                {
                    "home": g.TEAM_ABBREVIATION,
                    "away": other.iloc[0]["TEAM_ABBREVIATION"],
                    "home_won": 1.0 if g.WL == "W" else 0.0,
                    "date": g.GAME_DATE,
                    "game_id": gid,
                }
            )

            pbp = _game_pbp(gid, sleep)
            if pbp is None or not len(pbp):
                continue

            last_home, last_away = 0, 0
            for ev in pbp.itertuples():
                # V3 fills scoreHome/scoreAway only on scoring plays; carry the last values.
                sh, sa = getattr(ev, "scoreHome", ""), getattr(ev, "scoreAway", "")
                try:
                    last_home, last_away = int(sh), int(sa)
                except (ValueError, TypeError):
                    pass
                period = int(getattr(ev, "period", 1) or 1)
                clock = _clock_to_seconds(getattr(ev, "clock", "") or "")
                rows.append(
                    {
                        "score_diff": last_home - last_away,
                        "seconds_remaining": _nba_seconds_remaining(period, clock),
                        "period": period,
                        "possession_home": 0.5,  # V3 PBP has no clean possession flag
                        "game_id": gid,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No NBA rows pulled — stats.nba.com likely rate-limited or blocked this IP. "
            "Retry (cached games are skipped), increase `sleep`, or run from another network."
        )

    df = pd.DataFrame(rows)

    # Pregame Elo prior + home_win label, joined back onto the event rows by game_id.
    gr = pd.DataFrame(game_results).drop_duplicates("game_id").sort_values("date")
    elo = EloModel(k=20, home_advantage=100)  # tunable starting params
    gr["pregame_home_prob"] = elo.pregame_probs(
        [{"home": r.home, "away": r.away, "home_won": r.home_won} for r in gr.itertuples()]
    )
    df["home_win"] = df["game_id"].map(dict(zip(gr["game_id"], gr["home_won"])))
    df["pregame_home_prob"] = df["game_id"].map(dict(zip(gr["game_id"], gr["pregame_home_prob"])))

    return df[spec.features + ["home_win"]].astype(float)
