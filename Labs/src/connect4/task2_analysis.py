import json
from pathlib import Path

from .game import Connect4
from .task2_report import load_task2_evaluation_summary


def extract_representative_matches(summary):
    """Return the representative match selected for each Task 2 condition."""
    return {
        condition: condition_summary["representative_match"]
        for condition, condition_summary in summary["conditions"].items()
    }


def replay_match_states(match):
    """
    Reconstruct a match into replay-ready states.

    Returns a list of frames including the initial empty board and one frame per move.
    Each frame contains only JSON-serializable data.
    """
    game = Connect4()
    frames = [
        {
            "ply": 0,
            "player": None,
            "column": None,
            "row": None,
            "last_move": None,
            "board": game.board.tolist(),
        }
    ]

    for ply_index, move in enumerate(match["moves"], start=1):
        expected_player = move["player"]
        if game.current_player != expected_player:
            raise ValueError(
                f"Replay desync at ply {ply_index}: expected player {expected_player}, "
                f"got {game.current_player}."
            )

        row = game.drop_piece(move["column"])
        if row != move["row"]:
            raise ValueError(
                f"Replay row mismatch at ply {ply_index}: expected row {move['row']}, got {row}."
            )

        frames.append(
            {
                "ply": ply_index,
                "player": expected_player,
                "column": move["column"],
                "row": row,
                "last_move": [row, move["column"]],
                "board": game.board.tolist(),
            }
        )

    return frames


def summarize_task2_analysis(summary):
    """
    Build a factual analysis summary from evaluation results.

    The output is designed to support the Task 2.3 explanation using counts already
    produced by the evaluation pipeline.
    """
    conditions = {}
    winner_counts = {}
    total_draws = 0

    for condition, condition_summary in summary["conditions"].items():
        conditions[condition] = {
            "agent1_label": condition_summary["agent1_label"],
            "agent2_label": condition_summary["agent2_label"],
            "wins": int(condition_summary["wins"]),
            "losses": int(condition_summary["losses"]),
            "draws": int(condition_summary["draws"]),
            "num_matches": int(condition_summary["num_matches"]),
        }

    for result in summary["all_results"]:
        winner_label = result["winner_label"]
        if winner_label is None:
            total_draws += 1
            continue
        winner_counts[winner_label] = winner_counts.get(winner_label, 0) + 1

    if winner_counts:
        max_wins = max(winner_counts.values())
        most_frequent_winner_labels = sorted(
            [label for label, count in winner_counts.items() if count == max_wins]
        )
    else:
        max_wins = 0
        most_frequent_winner_labels = []

    return {
        "matches_per_condition": int(summary["matches_per_condition"]),
        "conditions": conditions,
        "overall_winner_counts": winner_counts,
        "overall_draws": total_draws,
        "most_frequent_winner_labels": most_frequent_winner_labels,
        "max_wins": max_wins,
    }


def export_task2_analysis_artifacts(summary_or_path, output_dir):
    """
    Persist representative matches and a factual analysis summary for Task 2.3.
    """
    summary = (
        load_task2_evaluation_summary(summary_or_path)
        if isinstance(summary_or_path, (str, Path))
        else summary_or_path
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_matches = extract_representative_matches(summary)
    analysis_summary = summarize_task2_analysis(summary)

    representative_path = output_dir / "task2_representative_matches.json"
    analysis_path = output_dir / "task2_analysis_summary.json"

    representative_path.write_text(
        json.dumps(representative_matches, indent=2),
        encoding="utf-8",
    )
    analysis_path.write_text(
        json.dumps(analysis_summary, indent=2),
        encoding="utf-8",
    )

    return {
        "representative_matches_path": str(representative_path.resolve()),
        "analysis_summary_path": str(analysis_path.resolve()),
        "representative_matches": representative_matches,
        "analysis_summary": analysis_summary,
    }
