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
        learning_rate=0.01,
        discount=0.9,
        epsilon=1.0,
        seed=13,
    )
    initial = agent.weights.copy()

    result = run_self_play_episode(agent, epsilon=1.0)

    assert 1 <= result["move_count"] <= Connect4.ROWS * Connect4.COLS
    assert len(result["td_errors"]) == result["online_updates"]
    assert result["winner"] in {None, Connect4.PLAYER1, Connect4.PLAYER2}
    assert np.isfinite(result["initial_max_abs_q"])
    assert not np.array_equal(agent.weights, initial)


def test_small_training_run_is_reproducible_with_fixed_seed():
    schedule = ConstantEpsilonSchedule(0.6)

    first = train_self_play(
        episodes=6,
        learning_rate=0.01,
        epsilon_schedule=schedule,
        initial_epsilon=0.6,
        checkpoint_interval=3,
        seed=21,
        stats_window_size=4,
    )
    second = train_self_play(
        episodes=6,
        learning_rate=0.01,
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
        episodes=12,
        learning_rate=0.01,
        epsilon_schedule=LinearDecayEpsilonSchedule(start=1.0, end=0.2, decay_steps=11),
        initial_epsilon=1.0,
        checkpoint_interval=4,
        output_dir=tmp_path,
        seed=8,
        stats_window_size=3,
        snapshot_start_episode=7,
        snapshot_interval=3,
        frozen_opponent_probability=1.0,
    )

    summary = result["summary"]
    summary_path = tmp_path / "training_summary.json"
    final_weights_path = tmp_path / "td_agent_final.npz"
    checkpoints_dir = tmp_path / "checkpoints"

    assert summary_path.exists()
    assert final_weights_path.exists()
    assert checkpoints_dir.exists()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["episodes"] == 12
    assert saved_summary["seed"] == 8
    assert saved_summary["checkpoint_interval"] == 4
    assert saved_summary["stats_window_size"] == 3
    assert saved_summary["snapshot_start_episode"] == 7
    assert saved_summary["snapshot_interval"] == 3
    assert saved_summary["frozen_opponent_probability"] == 1.0
    assert len(saved_summary["episode_logs"]) == 12
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
        "initial_max_abs_q_on_legal_actions",
        "weight_l2",
        "weight_max_abs",
        "nonfinite_detected",
        "used_frozen_opponent",
        "frozen_snapshot_index",
        "learning_player",
        "online_updates",
    }

    assert any(log["used_frozen_opponent"] for log in saved_summary["episode_logs"][6:])
    assert all(not log["nonfinite_detected"] for log in saved_summary["episode_logs"])

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
        "weight_l2",
        "weight_max_abs",
    }
    assert first_checkpoint["checkpoint_path"] is not None


def test_training_stays_finite_over_extended_smoke_run():
    result = train_self_play(
        episodes=80,
        learning_rate=0.01,
        epsilon_schedule=LinearDecayEpsilonSchedule(start=1.0, end=0.1, decay_steps=79),
        initial_epsilon=1.0,
        checkpoint_interval=20,
        seed=42,
        stats_window_size=50,
        snapshot_start_episode=41,
        snapshot_interval=20,
        frozen_opponent_probability=0.5,
    )

    assert np.isfinite(result["agent"].weights).all()
    assert all(np.isfinite(log["mean_abs_td_error"]) for log in result["summary"]["episode_logs"])
    assert all(np.isfinite(log["weight_l2"]) for log in result["summary"]["episode_logs"])
    assert all(np.isfinite(log["weight_max_abs"]) for log in result["summary"]["episode_logs"])


def test_training_rejects_invalid_episode_configuration():
    try:
        train_self_play(episodes=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected episodes=0 to raise ValueError.")
