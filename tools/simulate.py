"""Synthetic game generator — TESTING / DEMO ONLY.

This is NOT a production data source and must never be imported by the live monitor. Its
only job is to produce *calibrated-by-construction* synthetic games so the
train -> calibrate -> evaluate loop can be validated with zero network access and zero API
keys. Numbers produced by training on this data prove the machinery works; they say nothing
about real-world performance.

Generative recipe (per game)
----------------------------
1. Latent strength gap g ~ Normal(0, 0.9);  p0 = sigmoid(g).
2. Winner: home_win ~ Bernoulli(p0)  -> pregame calibration holds by construction
   (we store p0 as the pregame_home_prob feature, and the label is drawn from it).
3. final_margin = sign(home_win) * |Normal(0, scale)|.
4. For samples_per_game random times t ~ U(0,1), the in-game score_diff is drawn from the
   Brownian-bridge marginal Normal(final_margin*t, scale*sqrt(t*(1-t))). That makes
   P(home_win | score_diff, time_left, p0) a real, learnable, well-calibrated target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from liveedge.features import get_spec

# Scoring spread per sport (stddev of final margin) and regulation clock seconds.
_SCALE = {"nfl": 14.0, "nba": 13.0, "mlb": 4.0}
_TOTAL_CLOCK = {"nfl": 3600.0, "nba": 2880.0}  # MLB is inning-based, no game clock


def synthetic_frame(
    sport: str,
    n_games: int = 20000,
    samples_per_game: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic training frame for `sport`.

    Returns exactly the columns `get_spec(sport).features + [label]`, all as floats — the
    same contract the real data loaders honor.
    """
    spec = get_spec(sport)
    sport = sport.lower()
    rng = np.random.default_rng(seed)
    scale = _SCALE[sport]
    total_clock = _TOTAL_CLOCK.get(sport)

    # --- per-game latent state ---
    g = rng.normal(0.0, 0.9, n_games)
    p0 = 1.0 / (1.0 + np.exp(-g))
    home_win = (rng.random(n_games) < p0).astype(float)
    sign = np.where(home_win > 0.5, 1.0, -1.0)
    final_margin = sign * np.abs(rng.normal(0.0, scale, n_games))

    # --- expand to one row per sampled in-game moment ---
    rep = samples_per_game
    n = n_games * rep
    p0_r = np.repeat(p0, rep)
    hw_r = np.repeat(home_win, rep)
    fm_r = np.repeat(final_margin, rep)

    t = rng.uniform(0.0, 1.0, n)  # fraction of the game elapsed
    mean = fm_r * t
    sd = scale * np.sqrt(t * (1.0 - t))  # Brownian-bridge variance, 0 at the endpoints
    score_diff = np.round(rng.normal(mean, sd)).astype(int)

    data: dict[str, np.ndarray] = {
        "score_diff": score_diff,
        "pregame_home_prob": p0_r,
        "home_win": hw_r,
    }

    if sport in ("nfl", "nba"):
        data["seconds_remaining"] = (1.0 - t) * total_clock
        data["period"] = np.clip((t * 4).astype(int) + 1, 1, 4)

    if sport == "nfl":
        data["posteam_is_home"] = rng.integers(0, 2, n).astype(float)
        data["down"] = rng.integers(1, 5, n)
        data["ydstogo"] = rng.integers(1, 16, n)
        data["yardline_100"] = rng.integers(1, 100, n)
        data["home_timeouts"] = rng.integers(0, 4, n)
        data["away_timeouts"] = rng.integers(0, 4, n)
    elif sport == "nba":
        data["possession_home"] = np.full(n, 0.5)
    elif sport == "mlb":
        data["inning"] = np.clip((t * 9).astype(int) + 1, 1, 9)
        data["is_bottom"] = rng.integers(0, 2, n).astype(float)
        data["outs"] = rng.integers(0, 3, n)
        data["on_first"] = rng.integers(0, 2, n).astype(float)
        data["on_second"] = rng.integers(0, 2, n).astype(float)
        data["on_third"] = rng.integers(0, 2, n).astype(float)

    df = pd.DataFrame(data)
    return df[spec.features + [spec.label]].astype(float)
