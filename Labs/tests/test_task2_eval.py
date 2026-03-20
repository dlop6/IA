import json
import random

from connect4 import (
    AlphaBetaAgent,
    CONDITIONS,
    Connect4,
    MinimaxAgent,
    TDQLearningAgent,
    play_match,
    run_task2_evaluation,
    run_task2_match,
    train_self_play,
)


def build_smoke_td_agent():
    result = train_self_play(
        episodes=4,
        initial_epsilon=0.8,
        checkpoint_interval=2,
        seed=17,
        stats_window_size=4,
    )
    return result["agent"]


def test_each_condition_runs_in_a_small_seeded_smoke_batch():
    td_agent = build_smoke_td_agent()

    summary = run_task2_evaluation(
        matches_per_condition=4,
        td_agent=td_agent,
        td_epsilon=0.0,
        seed=31,
    )

    assert set(summary["conditions"]) == set(CONDITIONS)
    for condition_summary in summary["conditions"].values():
        assert condition_summary["num_matches"] == 4
        assert condition_summary["wins"] + condition_summary["losses"] + condition_summary["draws"] == 4
        assert condition_summary["representative_match"] is not None


def test_starting_player_alternates_every_match():
    td_agent = build_smoke_td_agent()

    first = run_task2_match("A", match_index=0, seed=100, td_agent=td_agent)
    second = run_task2_match("A", match_index=1, seed=101, td_agent=td_agent)
    third = run_task2_match("A", match_index=2, seed=102, td_agent=td_agent)

    assert first["starter_label"] == "TD"
    assert second["starter_label"] == "Minimax"
    assert third["starter_label"] == "TD"


def test_aggregated_totals_match_executed_matches():
    td_agent = build_smoke_td_agent()
    summary = run_task2_evaluation(
        matches_per_condition=6,
        td_agent=td_agent,
        td_epsilon=0.0,
        seed=9,
    )

    total_matches = sum(
        condition_summary["wins"] + condition_summary["losses"] + condition_summary["draws"]
        for condition_summary in summary["conditions"].values()
    )

    assert total_matches == 18
    assert len(summary["all_results"]) == 18


def test_condition_c_matches_direct_control_game_regression():
    result = run_task2_match("C", match_index=0, seed=55, td_agent=TDQLearningAgent())

    random.seed(55)
    direct = play_match(
        MinimaxAgent(ai_player=Connect4.PLAYER1, depth=4),
        AlphaBetaAgent(ai_player=Connect4.PLAYER2, depth=4),
        verbose=False,
        visual=False,
        print_result=False,
    )

    if direct["winner"] == Connect4.PLAYER1:
        expected_winner = "Minimax"
    elif direct["winner"] == Connect4.PLAYER2:
        expected_winner = "AlphaBeta"
    else:
        expected_winner = None

    assert result["starter_label"] == "Minimax"
    assert result["winner_label"] == expected_winner
    assert result["move_count"] == direct["move_count"]
    assert [move["column"] for move in result["moves"]] == [move["column"] for move in direct["moves"]]


def test_evaluation_persists_summary_json(tmp_path):
    td_agent = build_smoke_td_agent()
    summary = run_task2_evaluation(
        matches_per_condition=2,
        td_agent=td_agent,
        td_epsilon=0.0,
        seed=12,
        output_dir=tmp_path,
    )

    summary_path = tmp_path / "task2_evaluation_summary.json"
    assert summary_path.exists()
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["matches_per_condition"] == 2
    assert persisted["seed"] == 12
    assert set(persisted["conditions"]) == set(CONDITIONS)
    assert summary["summary_path"] == str(summary_path.resolve())
