import numpy as np
import pytest

from connect4 import Connect4, TDQLearningAgent


def play_moves(moves):
    game = Connect4()
    for move in moves:
        game.drop_piece(move)
    return game


def test_q_value_matches_dot_product():
    game = play_moves([3, 2, 3, 4, 5])
    agent = TDQLearningAgent(ai_player=game.current_player)
    features = agent.feature_vector(game, 0)
    weights = np.arange(features.shape[0], dtype=np.float64)
    agent.weights = weights.copy()

    expected = float(np.dot(weights, features))
    assert agent.q_value(game, 0) == expected


def test_terminal_update_matches_expected_weight_change_with_clipping():
    game = play_moves([3, 2, 3, 4, 5])
    next_game = game.copy()
    agent = TDQLearningAgent(
        ai_player=game.current_player,
        learning_rate=0.5,
        discount=0.9,
        epsilon=0.0,
    )
    initial = agent.weights.copy()
    features = agent.feature_vector(game, 0)

    td_error = agent.update(game, 0, reward=10.0, next_game=next_game, done=True)
    expected = initial + 0.5 * 1.0 * features

    assert td_error == 1.0
    assert np.allclose(agent.weights, expected)


def test_non_terminal_update_clips_large_td_error():
    game = play_moves([3, 2, 3, 4, 5])
    next_game = game.copy()
    next_game.drop_piece(0)

    agent = TDQLearningAgent(
        ai_player=game.current_player,
        learning_rate=0.25,
        discount=0.8,
        epsilon=0.0,
    )
    agent.weights = np.ones_like(agent.weights, dtype=np.float64) * 10.0

    features = agent.feature_vector(game, 0)
    current_q = float(np.dot(agent.weights, features))
    td_error = agent.update(game, 0, reward=0.3, next_game=next_game, done=False)

    assert td_error == -1.0
    assert np.allclose(agent.weights, np.ones_like(agent.weights, dtype=np.float64) * 10.0 - 0.25 * features)
    assert current_q > 1.0


def test_update_raises_on_nonfinite_weights():
    game = play_moves([3, 2, 3, 4, 5])
    next_game = game.copy()
    agent = TDQLearningAgent(ai_player=game.current_player)
    agent.weights[:] = np.nan

    with pytest.raises(ValueError, match="contains non-finite values"):
        agent.update(game, 0, reward=1.0, next_game=next_game, done=True)


def test_epsilon_zero_selects_greedy_action():
    game = play_moves([3, 2, 3, 4, 5])
    agent = TDQLearningAgent(ai_player=game.current_player, epsilon=0.0, seed=7)
    preferred = agent.feature_vector(game, 6)
    agent.weights = preferred.copy()

    assert agent.greedy_action(game) == 6
    assert agent.select_action(game) == 6


def test_epsilon_one_explores_only_legal_actions():
    game = play_moves([3, 2, 3, 4, 5])
    agent = TDQLearningAgent(ai_player=game.current_player, epsilon=1.0, seed=11)

    sampled = [agent.select_action(game) for _ in range(20)]

    assert all(action in game.actions() for action in sampled)
    assert len(set(sampled)) > 1


def test_save_and_load_round_trip(tmp_path):
    agent = TDQLearningAgent(
        ai_player=Connect4.PLAYER2,
        learning_rate=0.2,
        discount=0.85,
        epsilon=0.15,
    )
    agent.weights = np.linspace(-1.0, 1.0, agent.weights.shape[0], dtype=np.float64)
    path = tmp_path / "td_agent.npz"

    agent.save(path)
    loaded = TDQLearningAgent.load(path, ai_player=Connect4.PLAYER2, seed=5)

    assert np.array_equal(loaded.weights, agent.weights)
    assert loaded.weights.dtype == np.float64
    assert np.isclose(loaded.learning_rate, agent.learning_rate)
    assert np.isclose(loaded.discount, agent.discount)
    assert np.isclose(loaded.epsilon, agent.epsilon)


def test_td_agent_can_play_a_complete_legal_self_game():
    game = Connect4()
    agent = TDQLearningAgent(ai_player=Connect4.PLAYER1, epsilon=1.0, seed=3)
    move_count = 0

    while not game.is_terminal():
        action = agent.select_action(game)
        assert action in game.actions()
        game.drop_piece(action)
        move_count += 1

    assert 1 <= move_count <= Connect4.ROWS * Connect4.COLS
