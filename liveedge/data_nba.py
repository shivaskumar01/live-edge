"""Historical NBA loader (nba_api / stats.nba.com).

There is no prebuilt win-prob table, so we reconstruct (score_diff, seconds_remaining) from
PlayByPlayV3 events. (PlayByPlayV2 is deprecated and the NBA API now returns empty data for it
— see nba_api issue #591 — so V3 is required.)

WARNINGS
--------
stats.nba.com rate-limits aggressively and frequently blocks cloud / datacenter IPs. We add a
short sleep between calls and cap the number of games (`max_games`) so a first run completes.
For real training, add a local cache (e.g. write each game's PBP to parquet) and raise the cap
rather than re-hitting the API every run.
"""

from __future__ import annotations

import re
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

from liveedge.elo import EloModel
from liveedge.features import get_spec

_NBA_REGULATION_PERIODS = 4
_NBA_PERIOD_SECONDS = 720.0  # 12-minute quarters


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


def load_nba_frame(seasons: list[int], max_games: int = 200, sleep: float = 0.6) -> pd.DataFrame:
    """Load NBA play-by-play into the NBA feature contract + home_win label.

    Each season string is like '2021-22'. Each game appears twice in LeagueGameFinder (a home
    row and an away row); we anchor on the home row ('vs.' in MATCHUP).
    """
    spec = get_spec("nba")
    rows: list[dict] = []
    game_results: list[dict] = []

    for season in seasons:
        season_str = f"{season}-{str(season + 1)[-2:]}"
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str, league_id_nullable="00"
        )
        games_df = finder.get_data_frames()[0]
        time.sleep(sleep)

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
            home_abbr = g.TEAM_ABBREVIATION
            away_abbr = other.iloc[0]["TEAM_ABBREVIATION"]
            home_won = 1.0 if g.WL == "W" else 0.0
            game_results.append(
                {"home": home_abbr, "away": away_abbr, "home_won": bool(home_won),
                 "date": g.GAME_DATE, "game_id": gid}
            )

            try:
                pbp = playbyplayv3.PlayByPlayV3(game_id=gid).get_data_frames()[0]
                time.sleep(sleep)
            except Exception:
                time.sleep(sleep)
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
                        "home_win": home_won,
                        "game_id": gid,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No NBA rows pulled — stats.nba.com likely rate-limited or blocked this IP. "
            "Retry from a residential IP, increase `sleep`, or add a local PBP cache."
        )

    df = pd.DataFrame(rows)

    gr = pd.DataFrame(game_results).sort_values("date")
    elo = EloModel(k=20, home_advantage=100)  # tunable starting params
    gr["pregame_home_prob"] = elo.pregame_probs(
        [{"home": r.home, "away": r.away, "home_won": r.home_won} for r in gr.itertuples()]
    )
    prior = dict(zip(gr["game_id"], gr["pregame_home_prob"]))
    df["pregame_home_prob"] = df["game_id"].map(prior)

    return df[spec.features + ["home_win"]].astype(float)
