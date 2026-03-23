import json
import random
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
    opponent_agent=None,
    learning_player=None,
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
    initial_max_abs_q = float(agent.max_abs_q_value(game))
    online_updates = 0

    if opponent_agent is not None and learning_player not in {Connect4.PLAYER1, Connect4.PLAYER2}:
        raise ValueError("learning_player must be Connect4.PLAYER1 or Connect4.PLAYER2 when using opponent_agent.")

    while not game.is_terminal():
        state = game.copy()
        acting_player = state.current_player
        if opponent_agent is None or acting_player == learning_player:
            action = agent.select_action(state, epsilon=epsilon)
            should_update = True
        else:
            action = _select_policy_action(opponent_agent, state)
            should_update = False

        game.drop_piece(action)
        next_state = game.copy()
        if should_update:
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
            online_updates += 1
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
        "initial_max_abs_q": initial_max_abs_q,
        "online_updates": online_updates,
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
    snapshot_start_episode=1001,
    snapshot_interval=500,
    frozen_opponent_probability=0.5,
):
    """
    Train a shared TD agent via self-play and return a structured summary.

    When `output_dir` is provided, checkpoints and a machine-readable summary JSON are saved.
    """
    if episodes <= 0:
        raise ValueError("episodes must be greater than 0.")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative.")
    if snapshot_start_episode <= 0:
        raise ValueError("snapshot_start_episode must be greater than 0.")
    if snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be greater than 0.")
    if not 0.0 <= frozen_opponent_probability <= 1.0:
        raise ValueError("frozen_opponent_probability must be between 0 and 1.")

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
    snapshot_pool = []
    trainer_rng = random.Random(seed)

    output_path = None
    checkpoints_dir = None
    if output_dir is not None:
        output_path = Path(output_dir)
        checkpoints_dir = output_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, episodes + 1):
        epsilon = float(epsilon_schedule.value_at(episode - 1))
        agent.set_epsilon(epsilon)
        if episode == snapshot_start_episode:
            snapshot_pool.append(agent.clone(epsilon=0.0, seed=seed))

        use_frozen_opponent = episode >= snapshot_start_episode and snapshot_pool and trainer_rng.random() < frozen_opponent_probability
        if use_frozen_opponent:
            frozen_index = trainer_rng.randrange(len(snapshot_pool))
            opponent_agent = snapshot_pool[frozen_index]
            learning_side = Connect4.PLAYER1 if episode % 2 == 1 else Connect4.PLAYER2
            episode_reference_player = learning_side
        else:
            frozen_index = None
            opponent_agent = None
            learning_side = None
            episode_reference_player = reference_player

        try:
            episode_result = run_self_play_episode(
                agent,
                epsilon=epsilon,
                reference_player=episode_reference_player,
                bootstrap_scale=bootstrap_scale,
                opponent_agent=opponent_agent,
                learning_player=learning_side,
            )
        except ValueError as exc:
            raise ValueError(f"Training failed at episode {episode}: {exc}") from exc

        tracker.record_episode(episode_result["reference_reward"])
        weight_l2 = _weight_l2(agent)
        weight_max_abs = _weight_max_abs(agent)

        episode_logs.append(
            {
                "episode": episode,
                "epsilon": epsilon,
                "winner": episode_result["winner"],
                "draw": episode_result["draw"],
                "move_count": episode_result["move_count"],
                "reference_reward": episode_result["reference_reward"],
                "mean_abs_td_error": _mean_abs(episode_result["td_errors"]),
                "initial_max_abs_q_on_legal_actions": episode_result["initial_max_abs_q"],
                "weight_l2": weight_l2,
                "weight_max_abs": weight_max_abs,
                "nonfinite_detected": False,
                "used_frozen_opponent": bool(use_frozen_opponent),
                "frozen_snapshot_index": frozen_index,
                "learning_player": learning_side,
                "online_updates": episode_result["online_updates"],
            }
        )

        if episode >= snapshot_start_episode and episode % snapshot_interval == 0:
            snapshot_pool.append(agent.clone(epsilon=0.0, seed=seed))

        should_checkpoint = checkpoint_interval > 0 and episode % checkpoint_interval == 0
        if should_checkpoint or episode == episodes:
            snapshot = tracker.snapshot(episode=episode, epsilon=epsilon)
            snapshot["checkpoint_path"] = None
            snapshot["weight_l2"] = weight_l2
            snapshot["weight_max_abs"] = weight_max_abs

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
        "snapshot_start_episode": snapshot_start_episode,
        "snapshot_interval": snapshot_interval,
        "frozen_opponent_probability": float(frozen_opponent_probability),
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


def _weight_l2(agent):
    return float((agent.weights**2).sum() ** 0.5)


def _weight_max_abs(agent):
    return float(max(abs(weight) for weight in agent.weights))


def _select_policy_action(agent, game):
    if isinstance(agent, TDQLearningAgent):
        return agent.select_action(game, epsilon=0.0)
    return agent.select_action(game)
