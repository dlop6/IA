import json
from pathlib import Path

from .agents.td_agent import TDQLearningAgent
from .exploration import ConstantEpsilonSchedule
from .game import Connect4
from .rewards import transition_reward
from .training_metrics import TrainingStatsTracker


def run_self_play_episode(
    agent,
    *,
    epsilon=None,
    reference_player=Connect4.PLAYER1,
    bootstrap_scale=-1.0,
):
    """
    Run one self-play episode with a shared TD agent.

    The acting player is always the current player in the game state. Because the shared
    model encodes states from the perspective of the player to move, `bootstrap_scale`
    defaults to `-1.0` so the next-state value is interpreted from the opposite side in a
    zero-sum alternating-turn setting.
    """
    game = Connect4()
    td_errors = []
    move_count = 0

    while not game.is_terminal():
        state = game.copy()
        acting_player = state.current_player
        action = agent.select_action(state, epsilon=epsilon)
        game.drop_piece(action)
        next_state = game.copy()
        done = next_state.is_terminal()
        reward = transition_reward(next_state, acting_player)
        td_error = agent.update(
            state,
            action,
            reward,
            next_state,
            done,
            bootstrap_scale=bootstrap_scale,
        )
        td_errors.append(td_error)
        move_count += 1

    winner = None
    if game.check_winner(Connect4.PLAYER1):
        winner = Connect4.PLAYER1
    elif game.check_winner(Connect4.PLAYER2):
        winner = Connect4.PLAYER2

    if winner is None:
        reference_reward = 0.0
    elif winner == reference_player:
        reference_reward = 1.0
    else:
        reference_reward = -1.0

    return {
        "winner": winner,
        "draw": winner is None,
        "move_count": move_count,
        "reference_player": reference_player,
        "reference_reward": reference_reward,
        "td_errors": td_errors,
        "final_board": game.board.copy(),
    }


def train_self_play(
    *,
    episodes,
    agent=None,
    learning_rate=0.1,
    discount=0.99,
    initial_epsilon=0.1,
    epsilon_schedule=None,
    checkpoint_interval=0,
    output_dir=None,
    seed=None,
    stats_window_size=100,
    reference_player=Connect4.PLAYER1,
    bootstrap_scale=-1.0,
):
    """
    Train a shared TD agent via self-play and return a structured summary.

    When `output_dir` is provided, checkpoints and a machine-readable summary JSON are saved.
    """
    if episodes <= 0:
        raise ValueError("episodes must be greater than 0.")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative.")

    if agent is None:
        agent = TDQLearningAgent(
            ai_player=Connect4.PLAYER1,
            learning_rate=learning_rate,
            discount=discount,
            epsilon=initial_epsilon,
            seed=seed,
        )
    else:
        agent.set_epsilon(initial_epsilon)

    if epsilon_schedule is None:
        epsilon_schedule = ConstantEpsilonSchedule(initial_epsilon)

    tracker = TrainingStatsTracker(window_size=stats_window_size)
    episode_logs = []
    checkpoints = []

    output_path = None
    checkpoints_dir = None
    if output_dir is not None:
        output_path = Path(output_dir)
        checkpoints_dir = output_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, episodes + 1):
        epsilon = float(epsilon_schedule.value_at(episode - 1))
        agent.set_epsilon(epsilon)
        episode_result = run_self_play_episode(
            agent,
            epsilon=epsilon,
            reference_player=reference_player,
            bootstrap_scale=bootstrap_scale,
        )
        tracker.record_episode(episode_result["reference_reward"])

        episode_logs.append(
            {
                "episode": episode,
                "epsilon": epsilon,
                "winner": episode_result["winner"],
                "draw": episode_result["draw"],
                "move_count": episode_result["move_count"],
                "reference_reward": episode_result["reference_reward"],
                "mean_abs_td_error": _mean_abs(episode_result["td_errors"]),
            }
        )

        should_checkpoint = checkpoint_interval > 0 and episode % checkpoint_interval == 0
        if should_checkpoint or episode == episodes:
            snapshot = tracker.snapshot(episode=episode, epsilon=epsilon)
            snapshot["checkpoint_path"] = None

            if checkpoints_dir is not None:
                checkpoint_path = checkpoints_dir / f"td_agent_episode_{episode:05d}.npz"
                agent.save(checkpoint_path)
                snapshot["checkpoint_path"] = str(checkpoint_path.resolve())

            checkpoints.append(snapshot)

    summary = {
        "episodes": episodes,
        "seed": seed,
        "reference_player": reference_player,
        "bootstrap_scale": float(bootstrap_scale),
        "checkpoint_interval": checkpoint_interval,
        "stats_window_size": stats_window_size,
        "final_epsilon": float(agent.epsilon),
        "episode_logs": episode_logs,
        "checkpoints": checkpoints,
        "final_weights_path": None,
    }

    if output_path is not None:
        final_weights_path = output_path / "td_agent_final.npz"
        summary_path = output_path / "training_summary.json"
        agent.save(final_weights_path)
        summary["final_weights_path"] = str(final_weights_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "agent": agent,
        "summary": summary,
    }


def _mean_abs(values):
    if not values:
        return 0.0
    return float(sum(abs(value) for value in values) / len(values))
