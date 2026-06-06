"""The feature contract — the single source of truth for feature ordering.

The whole system hinges on one rule: a feature vector built from a *live* GameState must
be identical in shape and meaning to one built from a *historical* training row. Both the
data loaders and the live monitor go through `state_to_features` / `FeatureSpec.vector`, so
there is exactly one place that defines which columns exist and in what order. This is what
prevents train/serve skew.

Everything is from the HOME team's perspective. The label is `home_win` (1.0 / 0.0).
`pregame_home_prob` is a prior (filled from Elo in training, anchored from the opening market
line in the live monitor) so the tip-off number reflects team strength, not a flat 50/50.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GameState:
    """A point-in-time snapshot of one game, from the home team's perspective.

    Defaults describe a neutral pre-tip state. Sport-specific fields are simply ignored by
    sports that don't list them in their FeatureSpec, so one dataclass serves all three.
    """

    # Identity (required)
    sport: str
    home_team: str
    away_team: str

    # Common
    score_diff: int = 0  # home - away
    seconds_remaining: float = 0.0  # whole game, not just the current period
    period: int = 1
    pregame_home_prob: float = 0.5

    # Football
    posteam_is_home: float = 0.5  # 1 home has ball / 0 away has ball / 0.5 unknown
    down: int = 0
    ydstogo: int = 0
    yardline_100: int = 50  # yards to the opponent's end zone
    home_timeouts: int = 3
    away_timeouts: int = 3

    # Baseball
    inning: int = 1
    is_bottom: float = 0.0
    outs: int = 0
    on_first: float = 0.0
    on_second: float = 0.0
    on_third: float = 0.0

    # Basketball
    possession_home: float = 0.5

    # Bookkeeping (not features)
    game_id: str = ""
    description: str = ""
    # Full team names, used only to match live games to sportsbook odds (which use full
    # names) when home_team/away_team hold short abbreviations. Never fed to the model.
    home_full: str = ""
    away_full: str = ""


@dataclass
class FeatureSpec:
    """Defines the ordered feature columns and label for one sport."""

    sport: str
    features: list[str]
    label: str = "home_win"

    def vector(self, state: GameState) -> list[float]:
        """Build the model input vector for `state` in this spec's exact order."""
        return [float(getattr(state, name)) for name in self.features]


# Order defines the model's input columns and MUST stay stable across train + serve.
SPECS: dict[str, FeatureSpec] = {
    "nfl": FeatureSpec(
        sport="nfl",
        features=[
            "score_diff",
            "seconds_remaining",
            "posteam_is_home",
            "down",
            "ydstogo",
            "yardline_100",
            "home_timeouts",
            "away_timeouts",
            "pregame_home_prob",
        ],
    ),
    "nba": FeatureSpec(
        sport="nba",
        features=[
            "score_diff",
            "seconds_remaining",
            "period",
            "possession_home",
            "pregame_home_prob",
        ],
    ),
    "mlb": FeatureSpec(
        sport="mlb",
        features=[
            "score_diff",
            "inning",
            "is_bottom",
            "outs",
            "on_first",
            "on_second",
            "on_third",
            "pregame_home_prob",
        ],
    ),
}


def get_spec(sport: str) -> FeatureSpec:
    """Look up the FeatureSpec for a sport (case-insensitive). Raises on unknown."""
    key = sport.lower()
    if key not in SPECS:
        raise ValueError(f"unknown sport {sport!r}; expected one of {sorted(SPECS)}")
    return SPECS[key]


def state_to_features(state: GameState) -> list[float]:
    """Dispatch a GameState to its sport's FeatureSpec and return the feature vector."""
    return get_spec(state.sport).vector(state)
