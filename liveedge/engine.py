"""The edge engine: combine a model probability with a live market line into an EdgeRead.

This is where the model's P(home win) meets the sportsbook's de-vigged moneyline. If the model
disagrees with the market by enough to clear the vig (and any min_ev floor), we flag the side
and size it with fractional Kelly.
"""

from __future__ import annotations

from dataclasses import dataclass

from liveedge.oddsmath import (
    decimal_to_american,
    devig_two_way,
    ev_per_dollar,
    implied_edge,
    kelly_fraction,
    overround,
)


@dataclass
class EdgeRead:
    """The full read on one game: model vs. market, per-side EV/edge, and the best bet."""

    home_team: str
    away_team: str
    model_home_prob: float
    market_home_prob: float  # de-vigged
    market_away_prob: float  # de-vigged
    book_overround: float
    home_decimal: float
    away_decimal: float
    ev_home: float
    ev_away: float
    edge_home: float
    edge_away: float
    best_side: str | None  # "home" | "away" | None
    best_ev: float
    kelly_fraction: float  # already scaled by the fractional-Kelly multiplier

    def summary(self) -> str:
        """A one-line, human-readable read, e.g.
        'BOS @ LAL: bet LAL @ -190 | EV +40.4%/$1 | Kelly 19.2%'."""
        matchup = f"{self.away_team} @ {self.home_team}"
        if self.best_side is None:
            return f"{matchup}: no +EV side (best EV {self.best_ev:+.1%}/$1)"
        if self.best_side == "home":
            team, dec = self.home_team, self.home_decimal
        else:
            team, dec = self.away_team, self.away_decimal
        amer = int(decimal_to_american(dec))
        return (
            f"{matchup}: bet {team} @ {amer:+d} "
            f"| EV {self.best_ev:+.1%}/$1 | Kelly {self.kelly_fraction:.1%}"
        )


def evaluate(
    home_team: str,
    away_team: str,
    model_home_prob: float,
    home_decimal: float,
    away_decimal: float,
    kelly_multiplier: float = 0.25,
    min_ev: float = 0.0,
) -> EdgeRead:
    """Compare a model's P(home win) against a two-way moneyline and pick the +EV side, if any."""
    model_away_prob = 1 - model_home_prob

    market_home, market_away = devig_two_way(home_decimal, away_decimal)
    book_over = overround(home_decimal, away_decimal)

    ev_home = ev_per_dollar(model_home_prob, home_decimal)
    ev_away = ev_per_dollar(model_away_prob, away_decimal)
    edge_home = implied_edge(model_home_prob, market_home)
    edge_away = implied_edge(model_away_prob, market_away)

    # Candidate is the higher-EV side; we bet it only if it clears the min_ev floor.
    if ev_home >= ev_away:
        cand_side, cand_prob, cand_dec, cand_ev = "home", model_home_prob, home_decimal, ev_home
    else:
        cand_side, cand_prob, cand_dec, cand_ev = "away", model_away_prob, away_decimal, ev_away

    if cand_ev > min_ev:
        best_side: str | None = cand_side
        best_ev = cand_ev
        kelly = kelly_fraction(cand_prob, cand_dec) * kelly_multiplier
    else:
        best_side = None
        best_ev = max(ev_home, ev_away)
        kelly = 0.0

    return EdgeRead(
        home_team=home_team,
        away_team=away_team,
        model_home_prob=model_home_prob,
        market_home_prob=market_home,
        market_away_prob=market_away,
        book_overround=book_over,
        home_decimal=home_decimal,
        away_decimal=away_decimal,
        ev_home=ev_home,
        ev_away=ev_away,
        edge_home=edge_home,
        edge_away=edge_away,
        best_side=best_side,
        best_ev=best_ev,
        kelly_fraction=kelly,
    )
