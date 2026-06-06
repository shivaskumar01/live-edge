"""Tests for liveedge.engine — turning model prob + live line into an EdgeRead."""

import pytest

from liveedge.engine import evaluate
from liveedge.oddsmath import american_to_decimal

# Classic -110 / -110 two-way market (de-vigs to 0.5 / 0.5, ~4.76% hold).
DEC_110 = american_to_decimal(-110)


def test_no_positive_ev_when_model_matches_fair_market():
    # If the model agrees with the de-vigged (fair) market, the vig leaves no +EV side.
    read = evaluate("LAL", "BOS", model_home_prob=0.5, home_decimal=DEC_110, away_decimal=DEC_110)
    assert read.best_side is None
    assert read.best_ev <= 0
    assert read.kelly_fraction == 0.0


def test_picks_home_when_model_high():
    read = evaluate("LAL", "BOS", model_home_prob=0.65, home_decimal=DEC_110, away_decimal=DEC_110)
    assert read.best_side == "home"
    assert read.best_ev > 0
    assert read.kelly_fraction > 0


def test_picks_away_when_model_low():
    read = evaluate("LAL", "BOS", model_home_prob=0.35, home_decimal=DEC_110, away_decimal=DEC_110)
    assert read.best_side == "away"
    assert read.best_ev > 0


def test_kelly_multiplier_scales_linearly():
    quarter = evaluate(
        "LAL", "BOS", 0.65, DEC_110, DEC_110, kelly_multiplier=0.25
    ).kelly_fraction
    half = evaluate("LAL", "BOS", 0.65, DEC_110, DEC_110, kelly_multiplier=0.5).kelly_fraction
    assert half == pytest.approx(2 * quarter)


def test_min_ev_blocks_thin_edges():
    # model_home 0.53 at -110 is a ~1.2c/$1 edge: allowed at min_ev=0, blocked at min_ev=0.02.
    allowed = evaluate("LAL", "BOS", 0.53, DEC_110, DEC_110, min_ev=0.0)
    blocked = evaluate("LAL", "BOS", 0.53, DEC_110, DEC_110, min_ev=0.02)
    assert allowed.best_side == "home"
    assert 0 < allowed.best_ev < 0.02
    assert blocked.best_side is None


def test_devigged_market_probs_sum_to_one():
    read = evaluate(
        "LAL",
        "BOS",
        0.6,
        home_decimal=american_to_decimal(-150),
        away_decimal=american_to_decimal(130),
    )
    assert read.market_home_prob + read.market_away_prob == pytest.approx(1.0)
    assert read.market_home_prob > read.market_away_prob  # -150 favorite


def test_edge_equals_model_minus_market():
    read = evaluate("LAL", "BOS", 0.62, DEC_110, DEC_110)
    model_away = 1 - read.model_home_prob
    assert read.edge_home == pytest.approx(read.model_home_prob - read.market_home_prob)
    assert read.edge_away == pytest.approx(model_away - read.market_away_prob)


def test_summary_strings():
    bet = evaluate("LAL", "BOS", 0.7, DEC_110, DEC_110).summary()
    assert "bet LAL" in bet and "Kelly" in bet and "EV" in bet
    nope = evaluate("LAL", "BOS", 0.5, DEC_110, DEC_110).summary()
    assert "no +EV side" in nope


def test_model_away_prob_is_complement():
    read = evaluate("LAL", "BOS", 0.6, DEC_110, DEC_110)
    # ev_away uses (1 - model_home_prob); verify via a known relationship.
    assert read.ev_away == pytest.approx((1 - 0.6) * DEC_110 - 1)
