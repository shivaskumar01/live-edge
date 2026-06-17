"""Tests for liveedge.oddsmath, the betting math. The most important test file:
a sign error here silently turns a +EV read into a losing bet."""

import pytest

from liveedge.oddsmath import (
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    decimal_to_prob,
    devig_two_way,
    ev_per_dollar,
    implied_edge,
    kelly_fraction,
    overround,
    prob_to_american,
    prob_to_decimal,
)


def test_american_to_decimal_known_values():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-6)


@pytest.mark.parametrize("a", [150, 100, -110, -200, 250, -350])
def test_american_decimal_roundtrip(a):
    assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)


@pytest.mark.parametrize("d", [1.5, 1.91, 2.0, 2.5, 4.0])
def test_decimal_american_roundtrip(d):
    # Roundtrip is approximate because American odds are rounded to whole numbers.
    back = american_to_decimal(decimal_to_american(d))
    assert back == pytest.approx(d, abs=0.01)


def test_implied_prob_known_values():
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(-200) == pytest.approx(0.66666667, abs=1e-6)
    assert decimal_to_prob(4.0) == pytest.approx(0.25)


def test_prob_to_odds_roundtrip():
    assert decimal_to_prob(prob_to_decimal(0.4)) == pytest.approx(0.4)
    assert american_to_prob(prob_to_american(0.6)) == pytest.approx(0.6, abs=1e-3)


def test_devig_symmetric_sums_to_one():
    a, b = devig_two_way(american_to_decimal(-110), american_to_decimal(-110))
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.5)
    assert a + b == pytest.approx(1.0)


def test_devig_asymmetric_favorite_higher_and_sums_to_one():
    fav, dog = devig_two_way(american_to_decimal(-150), american_to_decimal(130))
    assert fav + dog == pytest.approx(1.0)
    assert fav > dog  # the favorite (-150) must have the higher fair probability


def test_overround_standard_juice():
    # Two -110 sides is the classic ~4.76% hold.
    assert overround(american_to_decimal(-110), american_to_decimal(-110)) == pytest.approx(
        0.04762, abs=1e-4
    )


def test_ev_break_even_positive_negative():
    assert ev_per_dollar(0.5, 2.0) == pytest.approx(0.0)
    assert ev_per_dollar(0.6, 2.0) == pytest.approx(0.2)
    assert ev_per_dollar(0.4, 2.0) == pytest.approx(-0.2)


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.4, 2.0) == 0.0  # negative edge -> never bet


def test_kelly_known_value():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_equals_ev_over_b():
    p, d = 0.58, 1.95
    assert kelly_fraction(p, d) == pytest.approx(ev_per_dollar(p, d) / (d - 1))


def test_implied_edge():
    assert implied_edge(0.6, 0.52) == pytest.approx(0.08)
    assert implied_edge(0.45, 0.52) == pytest.approx(-0.07)


@pytest.mark.parametrize("bad", [0])
def test_american_zero_raises(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -1.0])
def test_decimal_le_one_raises(bad):
    with pytest.raises(ValueError):
        decimal_to_prob(bad)
    with pytest.raises(ValueError):
        decimal_to_american(bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_prob_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        prob_to_decimal(bad)
