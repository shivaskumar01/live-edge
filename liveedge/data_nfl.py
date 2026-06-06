"""Historical NFL loader (nflreadpy / nflfastR play-by-play).

nflreadpy is the maintained successor to nfl_data_py (which is deprecated). It returns Polars
frames, so we convert to pandas. Data goes back to ~1999; down/distance is most reliable from
~2001 onward. Every row is one play, labeled with the final home result, and seeded with an
Elo pregame prior built only from prior results.
"""

from __future__ import annotations

import nflreadpy as nfl
import numpy as np
import pandas as pd

from liveedge.elo import EloModel
from liveedge.features import get_spec


def load_nfl_frame(seasons: list[int]) -> pd.DataFrame:
    """Load NFL play-by-play for `seasons` into the NFL feature contract + home_win label."""
    spec = get_spec("nfl")

    pbp = nfl.load_pbp(seasons)
    df = pbp.to_pandas() if hasattr(pbp, "to_pandas") else pbp

    # Keep real, timed offensive plays.
    df = df[df["down"].notna() & df["game_seconds_remaining"].notna()].copy()

    df["score_diff"] = df["total_home_score"] - df["total_away_score"]
    df["seconds_remaining"] = df["game_seconds_remaining"]
    df["posteam_is_home"] = (df["posteam"] == df["home_team"]).astype(float)

    # Timeouts are stored relative to posteam/defteam; remap to home/away.
    is_home = df["posteam_is_home"] == 1.0
    df["home_timeouts"] = np.where(
        is_home, df["posteam_timeouts_remaining"], df["defteam_timeouts_remaining"]
    )
    df["away_timeouts"] = np.where(
        is_home, df["defteam_timeouts_remaining"], df["posteam_timeouts_remaining"]
    )

    df["yardline_100"] = df["yardline_100"].fillna(50)
    df["ydstogo"] = df["ydstogo"].fillna(10)

    # Label: `result` is the final home margin; keep games that have one.
    df = df[df["result"].notna()].copy()
    df["home_win"] = (df["result"] > 0).astype(float)

    # Pregame Elo prior: one row per game in date order, no result leaks into its own feature.
    games = (
        df.sort_values("game_date")
        .groupby("game_id", sort=False)
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_win=("home_win", "first"),
            game_date=("game_date", "first"),
        )
        .reset_index()
        .sort_values("game_date")
    )
    elo = EloModel(k=20, home_advantage=65)  # tunable starting params
    games["pregame_home_prob"] = elo.pregame_probs(
        [
            {"home": r.home_team, "away": r.away_team, "home_won": bool(r.home_win)}
            for r in games.itertuples()
        ]
    )
    prior = dict(zip(games["game_id"], games["pregame_home_prob"]))
    df["pregame_home_prob"] = df["game_id"].map(prior)

    return df[spec.features + ["home_win"]].astype(float)
