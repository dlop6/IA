import json
import random
from pathlib import Path

from .agents import AlphaBetaAgent, MinimaxAgent, TDQLearningAgent
from .evaluation import play_match
from .game import Connect4


CONDITIONS = {
    "A": ("TD", "Minimax"),
    "B": ("TD", "AlphaBeta"),
    "C": ("Minimax", "AlphaBeta"),
}


def run_task2_match(
    condition,
    *,
    match_index,
    seed,
    td_agent_path=None,
    td_agent=None,
    td_epsilon=0.0,
    minimax_depth=4,
    alphabeta_depth=4,
):
    """
    Run one Task 2 evaluation match with alternating starter based on match index.

    Even match indices start with the first-listed agent in the condition. Odd indices swap
    the starter to keep first-player advantage balanced.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Expected one of {sorted(CONDITIONS)}.")

    random.seed(seed)
    agent1_label, agent2_label = CONDITIONS[condition]
    starter_is_agent1 = match_index % 2 == 0

    player1_label = agent1_label if starter_is_agent1 else agent2_label
    player2_label = agent2_label if starter_is_agent1 else agent1_label

    player1_agent = _build_agent(
        player1_label,
        Connect4.PLAYER1,
        td_agent_path=td_agent_path,
        td_agent=td_agent,
        td_epsilon=td_epsilon,
        minimax_depth=minimax_depth,
        alphabeta_depth=alphabeta_depth,
        seed=seed,
    )
    player2_agent = _build_agent(
        player2_label,
        Connect4.PLAYER2,
        td_agent_path=td_agent_path,
        td_agent=td_agent,
        td_epsilon=td_epsilon,
        minimax_depth=minimax_depth,
        alphabeta_depth=alphabeta_depth,
        seed=seed + 1,
    )

    match = play_match(
        player1_agent,
        player2_agent,
        verbose=False,
        visual=False,
        result_labels={
            Connect4.PLAYER1: f"{player1_label} gana.",
            Connect4.PLAYER2: f"{player2_label} gana.",
            "draw": "Empate.",
        },
        move_labels={
            Connect4.PLAYER1: player1_label,
            Connect4.PLAYER2: player2_label,
        },
        print_result=False,
    )

    if match["winner"] == Connect4.PLAYER1:
        winner_label = player1_label
    elif match["winner"] == Connect4.PLAYER2:
        winner_label = player2_label
    else:
        winner_label = None

    if winner_label is None:
        outcome = "draw"
    elif winner_label == agent1_label:
        outcome = "win"
    else:
        outcome = "loss"

    return {
        "condition": condition,
        "match_index": match_index,
        "seed": seed,
        "agent1_label": agent1_label,
        "agent2_label": agent2_label,
        "starter_label": player1_label,
        "player1_label": player1_label,
        "player2_label": player2_label,
        "winner_label": winner_label,
        "draw": winner_label is None,
        "outcome": outcome,
        "move_count": match["move_count"],
        "moves": match["moves"],
        "final_board": match["board"].tolist(),
    }


def run_task2_evaluation(
    *,
    matches_per_condition=50,
    td_agent_path=None,
    td_agent=None,
    td_epsilon=0.0,
    minimax_depth=4,
    alphabeta_depth=4,
    seed=0,
    output_dir=None,
):
    """
    Run the 3 required Task 2 evaluation conditions and return a structured summary.

    Results are reported relative to the first-listed agent of each condition:
    - wins: first-listed agent wins
    - losses: first-listed agent loses
    - draws: tied matches
    """
    if matches_per_condition <= 0:
        raise ValueError("matches_per_condition must be greater than 0.")
    if td_agent_path is None and td_agent is None and any("TD" in pair for pair in CONDITIONS.values()):
        raise ValueError("TD evaluation requires either td_agent_path or td_agent.")

    all_results = []
    condition_summaries = {}
    base_seed = int(seed)

    for condition_index, (condition, labels) in enumerate(CONDITIONS.items()):
        matches = []
        wins = 0
        losses = 0
        draws = 0

        for match_index in range(matches_per_condition):
            match_seed = base_seed + condition_index * 10_000 + match_index
            result = run_task2_match(
                condition,
                match_index=match_index,
                seed=match_seed,
                td_agent_path=td_agent_path,
                td_agent=td_agent,
                td_epsilon=td_epsilon,
                minimax_depth=minimax_depth,
                alphabeta_depth=alphabeta_depth,
            )
            matches.append(result)
            all_results.append(result)

            if result["outcome"] == "win":
                wins += 1
            elif result["outcome"] == "loss":
                losses += 1
            else:
                draws += 1

        representative_match = _choose_representative_match(matches)
        condition_summaries[condition] = {
            "condition": condition,
            "agent1_label": labels[0],
            "agent2_label": labels[1],
            "matches": matches,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "num_matches": matches_per_condition,
            "representative_match": representative_match,
        }

    summary = {
        "matches_per_condition": matches_per_condition,
        "seed": base_seed,
        "td_epsilon": float(td_epsilon),
        "minimax_depth": minimax_depth,
        "alphabeta_depth": alphabeta_depth,
        "conditions": condition_summaries,
        "all_results": all_results,
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "task2_evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path.resolve())

    return summary


def _build_agent(
    label,
    player,
    *,
    td_agent_path,
    td_agent,
    td_epsilon,
    minimax_depth,
    alphabeta_depth,
    seed,
):
    if label == "TD":
        if td_agent is not None:
            return TDQLearningAgent(
                ai_player=player,
                learning_rate=td_agent.learning_rate,
                discount=td_agent.discount,
                epsilon=td_epsilon,
                weights=td_agent.weights.copy(),
                seed=seed,
            )
        loaded = TDQLearningAgent.load(td_agent_path, ai_player=player, seed=seed)
        loaded.set_epsilon(td_epsilon)
        return loaded

    if label == "Minimax":
        return MinimaxAgent(ai_player=player, depth=minimax_depth)

    if label == "AlphaBeta":
        return AlphaBetaAgent(ai_player=player, depth=alphabeta_depth)

    raise ValueError(f"Unsupported agent label {label!r}.")


def _choose_representative_match(matches):
    for match in matches:
        if not match["draw"]:
            return match
    return matches[0]
