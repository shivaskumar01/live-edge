"""Elo + feature-contract + synthetic end-to-end tests."""

import pytest

from liveedge.elo import EloModel
from liveedge.features import SPECS, GameState, get_spec, state_to_features
from liveedge.train import train_from_frame
from tools.simulate import synthetic_frame

# --------------------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------------------


def test_elo_favors_stronger_team():
    elo = EloModel()
    elo.ratings["A"] = 1700
    elo.ratings["B"] = 1300
    assert elo.expected_home("A", "B") > 0.5  # strong home team
    assert elo.expected_home("B", "A") < 0.5  # weak home team vs strong away


def test_elo_symmetric_without_home_advantage():
    elo = EloModel(home_advantage=0.0)
    # Equal (unseen) teams with no home edge -> dead even, and complementary when swapped.
    assert elo.expected_home("A", "B") == pytest.approx(0.5)
    assert elo.expected_home("A", "B") + elo.expected_home("B", "A") == pytest.approx(1.0)


def test_elo_update_moves_ratings_correctly():
    elo = EloModel()
    elo.update("A", "B", home_won=True)
    assert elo.ratings["A"] > 1500  # winner gains
    assert elo.ratings["B"] < 1500  # loser drops
    assert elo.ratings["A"] > elo.ratings["B"]


def test_pregame_probs_no_leak():
    games = [
        {"home": "A", "away": "B", "home_won": True},
        {"home": "A", "away": "B", "home_won": True},
    ]
    probs = EloModel().pregame_probs(games)
    prior = EloModel().expected_home("A", "B")  # from-scratch prior, no results applied
    # First prob must equal the untouched prior (result not leaked into its own feature)...
    assert probs[0] == pytest.approx(prior)
    # ...and after A wins, A's second pregame prob must be higher.
    assert probs[1] > probs[0]


# --------------------------------------------------------------------------------------
# Feature contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sport", ["nfl", "nba", "mlb"])
def test_spec_shape_and_ordering(sport):
    spec = get_spec(sport)
    state = GameState(sport=sport, home_team="H", away_team="A")
    vec = spec.vector(state)
    assert len(vec) == len(spec.features)
    assert spec.features[0] == "score_diff"  # score_diff first
    assert spec.features[-1] == "pregame_home_prob"  # prior last
    assert spec.label == "home_win"


def test_all_sports_have_specs():
    assert set(SPECS) == {"nfl", "nba", "mlb"}
    for spec in SPECS.values():
        assert "pregame_home_prob" in spec.features
        assert spec.label == "home_win"


def test_state_to_features_dispatches_and_matches_vector():
    state = GameState(sport="NBA", home_team="LAL", away_team="BOS", score_diff=5)
    assert state_to_features(state) == get_spec("nba").vector(state)
    assert state_to_features(state)[0] == 5.0  # score_diff


def test_get_spec_unknown_raises():
    with pytest.raises(ValueError):
        get_spec("cricket")


# --------------------------------------------------------------------------------------
# End-to-end: synthetic data should train to beat the base rate and calibrate well.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sport", ["nfl", "nba", "mlb"])
def test_synthetic_end_to_end_calibrates(sport):
    df = synthetic_frame(sport, n_games=4000, seed=1)
    _, _, _, metrics = train_from_frame(df, sport, epochs=8, verbose=False)
    assert metrics["log_loss"] < metrics["baseline_log_loss"]
    assert metrics["ece"] < 0.05
