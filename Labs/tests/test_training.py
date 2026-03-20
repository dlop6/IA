import json

import numpy as np

from connect4 import (
    Connect4,
    ConstantEpsilonSchedule,
    LinearDecayEpsilonSchedule,
    TDQLearningAgent,
    run_self_play_episode,
    train_self_play,
)


def test_self_play_episode_completes_and_updates_weights():
    agent = TDQLearningAgent(
        ai_player=Connect4.PLAYER1,
        learning_rate=0.2,
        discount=0.9,
        epsilon=1.0,
        seed=13,
    )
    initial = agent.weights.copy()

    result = run_self_play_episode(agent, epsilon=1.0)

    assert 1 <= result["move_count"] <= Connect4.ROWS * Connect4.COLS
    assert len(result["td_errors"]) == result["move_count"]
    assert result["winner"] in {None, Connect4.PLAYER1, Connect4.PLAYER2}
    assert not np.array_equal(agent.weights, initial)


def test_small_training_run_is_reproducible_with_fixed_seed():
    schedule = ConstantEpsilonSchedule(0.6)

    first = train_self_play(
        episodes=6,
        epsilon_schedule=schedule,
        initial_epsilon=0.6,
        checkpoint_interval=3,
        seed=21,
        stats_window_size=4,
    )
    second = train_self_play(
        episodes=6,
        epsilon_schedule=schedule,
        initial_epsilon=0.6,
        checkpoint_interval=3,
        seed=21,
        stats_window_size=4,
    )

    assert np.array_equal(first["agent"].weights, second["agent"].weights)
    assert first["summary"]["episode_logs"] == second["summary"]["episode_logs"]
    assert first["summary"]["checkpoints"] == second["summary"]["checkpoints"]


def test_training_outputs_checkpoints_and_summary_schema(tmp_path):
    result = train_self_play(
        episodes=5,
        epsilon_schedule=LinearDecayEpsilonSchedule(start=1.0, end=0.2, decay_steps=4),
        initial_epsilon=1.0,
        checkpoint_interval=2,
        output_dir=tmp_path,
        seed=8,
        stats_window_size=3,
    )

    summary = result["summary"]
    summary_path = tmp_path / "training_summary.json"
    final_weights_path = tmp_path / "td_agent_final.npz"
    checkpoints_dir = tmp_path / "checkpoints"

    assert summary_path.exists()
    assert final_weights_path.exists()
    assert checkpoints_dir.exists()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["episodes"] == 5
    assert saved_summary["seed"] == 8
    assert saved_summary["checkpoint_interval"] == 2
    assert saved_summary["stats_window_size"] == 3
    assert len(saved_summary["episode_logs"]) == 5
    assert len(saved_summary["checkpoints"]) == 3

    first_log = saved_summary["episode_logs"][0]
    assert set(first_log) == {
        "episode",
        "epsilon",
        "winner",
        "draw",
        "move_count",
        "reference_reward",
        "mean_abs_td_error",
    }

    first_checkpoint = saved_summary["checkpoints"][0]
    assert set(first_checkpoint) == {
        "episode",
        "epsilon",
        "window_size",
        "average_reward",
        "wins",
        "losses",
        "draws",
        "checkpoint_path",
    }
    assert first_checkpoint["checkpoint_path"] is not None


def test_training_rejects_invalid_episode_configuration():
    try:
        train_self_play(episodes=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected episodes=0 to raise ValueError.")
