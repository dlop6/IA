import numpy as np
import pytest

from connect4 import (
    Connect4,
    ConstantEpsilonSchedule,
    DRAW_REWARD,
    LinearDecayEpsilonSchedule,
    LOSS_REWARD,
    NON_TERMINAL_REWARD,
    TDQLearningAgent,
    TrainingStatsTracker,
    WIN_REWARD,
    terminal_reward,
    transition_reward,
)


def play_moves(moves):
    game = Connect4()
    for move in moves:
        game.drop_piece(move)
    return game


def test_sparse_terminal_rewards_are_exact():
    player1_win = play_moves([0, 1, 0, 1, 0, 1, 0])
    assert terminal_reward(player1_win, Connect4.PLAYER1) == WIN_REWARD
    assert terminal_reward(player1_win, Connect4.PLAYER2) == LOSS_REWARD

    draw_board = [
        [2, 2, 1, 2, 1, 1, 2],
        [2, 1, 2, 1, 2, 1, 2],
        [1, 1, 2, 1, 2, 2, 2],
        [2, 2, 1, 2, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 1],
        [2, 1, 1, 1, 2, 1, 2],
    ]
    draw_game = Connect4(board=draw_board, current_player=Connect4.PLAYER1)
    assert terminal_reward(draw_game, Connect4.PLAYER1) == DRAW_REWARD

    non_terminal = play_moves([3, 2, 3, 4, 5])
    assert terminal_reward(non_terminal, non_terminal.current_player) == NON_TERMINAL_REWARD
    assert transition_reward(non_terminal, non_terminal.current_player) == NON_TERMINAL_REWARD


def test_constant_epsilon_schedule_is_stable():
    schedule = ConstantEpsilonSchedule(0.2)
    assert schedule.value_at(0) == 0.2
    assert schedule.value_at(10) == 0.2
    assert schedule.value_at(1000) == 0.2


def test_linear_decay_schedule_behaves_as_configured():
    schedule = LinearDecayEpsilonSchedule(start=1.0, end=0.1, decay_steps=100)

    assert np.isclose(schedule.value_at(0), 1.0)
    assert np.isclose(schedule.value_at(50), 0.55)
    assert np.isclose(schedule.value_at(100), 0.1)
    assert np.isclose(schedule.value_at(150), 0.1)


def test_linear_decay_schedule_rejects_invalid_steps():
    with pytest.raises(ValueError, match="greater than 0"):
        LinearDecayEpsilonSchedule(start=1.0, end=0.1, decay_steps=0)


def test_td_agent_accepts_explicit_epsilon_updates():
    game = play_moves([3, 2, 3, 4, 5])
    agent = TDQLearningAgent(ai_player=game.current_player, epsilon=0.5, seed=4)
    preferred = agent.feature_vector(game, 6)
    agent.weights = preferred.copy()

    agent.set_epsilon(0.0)
    assert agent.select_action(game) == 6

    agent.set_epsilon(1.0)
    sampled = [agent.select_action(game) for _ in range(20)]
    assert all(action in game.actions() for action in sampled)


def test_training_stats_tracker_produces_checkpoint_summary():
    tracker = TrainingStatsTracker(window_size=5)
    for reward in [1.0, -1.0, 0.0, 1.0]:
        tracker.record_episode(reward)

    snapshot = tracker.snapshot(episode=4, epsilon=0.25)

    assert snapshot["episode"] == 4
    assert snapshot["epsilon"] == 0.25
    assert snapshot["window_size"] == 4
    assert np.isclose(snapshot["average_reward"], 0.25)
    assert snapshot["wins"] == 2
    assert snapshot["losses"] == 1
    assert snapshot["draws"] == 1
