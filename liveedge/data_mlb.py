"""Historical MLB loader (pybaseball Statcast / Baseball Savant).

Statcast is pitch-level; we reduce it to one row per plate appearance (the last pitch of each
at-bat, which carries the resolved state going into the next batter).

WARNING
-------
Statcast pulls are heavy. We default to a ~3-week window per season so a first run isn't
enormous, and pybaseball's on-disk cache is enabled so re-runs don't re-download. For real
training, widen the window (start_md / end_md) and pull more seasons.
"""

from __future__ import annotations

import pandas as pd
from pybaseball import cache as _pybaseball_cache, statcast

from liveedge.elo import EloModel
from liveedge.features import get_spec

_pybaseball_cache.enable()  # cache Statcast pulls to disk so re-runs don't re-download


def load_mlb_frame(
    seasons: list[int], start_md: str = "07-01", end_md: str = "07-21"
) -> pd.DataFrame:
    """Load MLB Statcast into the MLB feature contract + home_win label."""
    spec = get_spec("mlb")

    frames = []
    for season in seasons:
        sc = statcast(start_dt=f"{season}-{start_md}", end_dt=f"{season}-{end_md}")
        if sc is not None and len(sc):
            frames.append(sc)
    if not frames:
        raise RuntimeError(
            "No Statcast rows returned — check the date window / network, or widen start_md/end_md."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    # One row per plate appearance: the last pitch of each at-bat.
    pa = df.groupby(["game_pk", "at_bat_number"]).tail(1).copy()
    pa["score_diff"] = pa["home_score"] - pa["away_score"]
    pa["is_bottom"] = (pa["inning_topbot"].str.lower() == "bot").astype(float)
    pa["outs"] = pa["outs_when_up"]
    pa["on_first"] = pa["on_1b"].notna().astype(float)
    pa["on_second"] = pa["on_2b"].notna().astype(float)
    pa["on_third"] = pa["on_3b"].notna().astype(float)

    # Final score per game from the last pitch (post_* is the score after the play).
    last = df.groupby("game_pk").tail(1)
    finals = pd.DataFrame(
        {
            "game_pk": last["game_pk"].to_numpy(),
            "home_final": last["post_home_score"].fillna(last["home_score"]).to_numpy(),
            "away_final": last["post_away_score"].fillna(last["away_score"]).to_numpy(),
        }
    )
    finals["home_win"] = (finals["home_final"] > finals["away_final"]).astype(float)

    # Pregame Elo prior over games in date order.
    meta = (
        df.groupby("game_pk")
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            game_date=("game_date", "first"),
        )
        .reset_index()
        .merge(finals[["game_pk", "home_win"]], on="game_pk")
        .sort_values("game_date")
    )
    elo = EloModel(k=6, home_advantage=35)  # baseball ~ coin flips; tunable
    meta["pregame_home_prob"] = elo.pregame_probs(
        [
            {"home": r.home_team, "away": r.away_team, "home_won": bool(r.home_win)}
            for r in meta.itertuples()
        ]
    )

    pa = pa.merge(finals[["game_pk", "home_win"]], on="game_pk").merge(
        meta[["game_pk", "pregame_home_prob"]], on="game_pk"
    )

    # `inning` comes straight from Statcast and is already in spec.features.
    return pa[spec.features + ["home_win"]].astype(float)
