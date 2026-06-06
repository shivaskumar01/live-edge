"""Betting math: odds conversions, de-vig, EV, and Kelly staking.

Pure functions, no imports beyond the stdlib. This is the highest-risk file in the
project: a single sign error here silently turns a +EV read into a losing bet, so the
functions are intentionally tiny and exhaustively unit-tested (see tests/test_oddsmath.py).

Conventions
-----------
- American odds: integers like +150 / -200 (positive = underdog, negative = favorite).
- Decimal odds: always > 1.0 (the total return per $1 staked, stake included).
- Probabilities: floats in [0, 1].
- All stake / EV math is expressed per $1 risked.
"""

from __future__ import annotations


def american_to_decimal(a: float) -> float:
    """Convert American odds to decimal odds. American 0 is undefined."""
    if a == 0:
        raise ValueError("American odds of 0 are undefined")
    if a > 0:
        return 1 + a / 100
    return 1 + 100 / abs(a)


def decimal_to_american(d: float) -> float:
    """Convert decimal odds (>1) to American odds. >=2.0 -> positive, else negative."""
    if d <= 1:
        raise ValueError(f"decimal odds must be > 1, got {d}")
    if d >= 2:
        return round((d - 1) * 100)
    return round(-100 / (d - 1))


def decimal_to_prob(d: float) -> float:
    """Implied (vigged) probability of a single decimal price."""
    if d <= 1:
        raise ValueError(f"decimal odds must be > 1, got {d}")
    return 1 / d


def american_to_prob(a: float) -> float:
    """Implied (vigged) probability of a single American price."""
    return decimal_to_prob(american_to_decimal(a))


def prob_to_decimal(p: float) -> float:
    """Fair decimal odds for a probability (no vig). Requires 0 < p < 1."""
    if not (0 < p < 1):
        raise ValueError(f"probability must be in (0, 1), got {p}")
    return 1 / p


def prob_to_american(p: float) -> float:
    """Fair American odds for a probability (no vig). Requires 0 < p < 1."""
    return decimal_to_american(prob_to_decimal(p))


def devig_two_way(dec_a: float, dec_b: float) -> tuple[float, float]:
    """De-vig a two-way market by multiplicative normalization.

    Takes the two raw implied probabilities (1/dec) and rescales them to sum to 1,
    removing the book's hold. The returned (fair_a, fair_b) always sum to 1.0.

    Note: more sophisticated de-vig methods exist (Shin's method, the power method)
    that handle favorite-longshot bias; for two-way moneylines the simple
    proportional / normalization approach is standard and good enough.
    """
    raw_a = 1 / dec_a
    raw_b = 1 / dec_b
    o = raw_a + raw_b
    return raw_a / o, raw_b / o


def overround(dec_a: float, dec_b: float) -> float:
    """The book's hold (a.k.a. vig / juice): how much the implied probs exceed 1."""
    return 1 / dec_a + 1 / dec_b - 1


def ev_per_dollar(p: float, d: float) -> float:
    """Expected value per $1 staked at decimal odds d given true win prob p.

    Win returns (d - 1) profit with prob p; loss returns -1 with prob (1 - p):
    EV = p*(d-1) - (1-p) = p*d - 1.
    """
    return p * d - 1


def kelly_fraction(p: float, d: float) -> float:
    """Full-Kelly fraction of bankroll to stake. 0.0 when there is no edge.

    For a binary bet at decimal odds d (net odds b = d - 1) with win prob p:
    f* = edge / b where edge = p*d - 1. Returns 0.0 if edge <= 0 (never bet -EV).
    Scale this by a fractional-Kelly multiplier (e.g. 0.25) at the call site.
    """
    edge = p * d - 1
    if edge <= 0:
        return 0.0
    return edge / (d - 1)


def implied_edge(model_p: float, fair_p: float) -> float:
    """Probability edge: how much more likely we think the outcome is than the
    (de-vigged) market does. Positive means the market underrates this side."""
    return model_p - fair_p
