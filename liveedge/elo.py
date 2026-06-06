"""Elo ratings -> a pregame win-probability prior.

Without a prior, every game starts ~50/50 because the in-game features carry no signal at
tip-off. Elo gives a team-strength prior built *only* from past results, so we don't need
historical betting lines to seed `pregame_home_prob`.

The per-sport starting parameters below are reasonable defaults, not tuned constants — they
are documented as tunable. Baseball games are much closer to coin flips, hence its small k
and home-field edge.
"""

from __future__ import annotations

from collections import defaultdict


class EloModel:
    """A minimal Elo rating system with a home-field advantage term.

    Parameters
    ----------
    k : update step size (how much one result moves ratings).
    home_advantage : rating points added to the home side when computing expectation.
    base : starting rating for any unseen team.
    """

    def __init__(
        self, k: float = 20.0, home_advantage: float = 65.0, base: float = 1500.0
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.base = base
        self.ratings: dict[str, float] = defaultdict(lambda: self.base)

    def expected_home(self, home: str, away: str) -> float:
        """Expected probability the home team wins given current ratings."""
        diff = (self.ratings[home] + self.home_advantage) - self.ratings[away]
        return 1 / (1 + 10 ** (-diff / 400))

    def update(self, home: str, away: str, home_won: bool) -> None:
        """Apply one game result, moving both teams' ratings by the same magnitude."""
        exp = self.expected_home(home, away)
        result = 1.0 if home_won else 0.0
        delta = self.k * (result - exp)
        self.ratings[home] += delta
        self.ratings[away] -= delta

    def pregame_probs(self, games: list[dict]) -> list[float]:
        """Walk games in chronological order, returning each game's pregame home prob.

        For each game we record `expected_home` computed from the ratings *as they stand
        before that game*, and only then call `update`. This ordering is critical: it
        prevents leaking a game's own result into its pregame feature. Each game dict needs
        keys: ``home``, ``away``, ``home_won``.
        """
        probs: list[float] = []
        for g in games:
            home, away = g["home"], g["away"]
            probs.append(self.expected_home(home, away))
            self.update(home, away, bool(g["home_won"]))
        return probs
