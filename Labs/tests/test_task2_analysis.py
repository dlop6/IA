import json

from connect4 import (
    CONDITIONS,
    export_task2_analysis_artifacts,
    extract_representative_matches,
    replay_match_states,
    summarize_task2_analysis,
    run_task2_evaluation,
    train_self_play,
)


def build_analysis_ready_summary(tmp_path):
    td_agent = train_self_play(
        episodes=4,
        initial_epsilon=0.8,
        checkpoint_interval=2,
        seed=29,
        stats_window_size=4,
    )["agent"]

    return run_task2_evaluation(
        matches_per_condition=3,
        td_agent=td_agent,
        td_epsilon=0.0,
        seed=43,
        output_dir=tmp_path,
    )


def test_representative_matches_exist_for_all_conditions(tmp_path):
    summary = build_analysis_ready_summary(tmp_path)

    representative_matches = extract_representative_matches(summary)

    assert set(representative_matches) == set(CONDITIONS)
    assert all(match is not None for match in representative_matches.values())


def test_representative_match_can_be_replayed_to_final_board(tmp_path):
    summary = build_analysis_ready_summary(tmp_path)
    representative_matches = extract_representative_matches(summary)

    for condition, match in representative_matches.items():
        frames = replay_match_states(match)
        assert len(frames) == match["move_count"] + 1
        assert frames[-1]["board"] == match["final_board"]
        assert frames[0]["board"] != frames[-1]["board"] or match["move_count"] == 0


def test_analysis_summary_is_consistent_with_evaluation_results(tmp_path):
    summary = build_analysis_ready_summary(tmp_path)

    analysis = summarize_task2_analysis(summary)

    assert set(analysis["conditions"]) == set(CONDITIONS)
    total_non_draw_wins = sum(analysis["overall_winner_counts"].values())
    expected_non_draw_wins = sum(1 for result in summary["all_results"] if result["winner_label"] is not None)
    expected_draws = sum(1 for result in summary["all_results"] if result["winner_label"] is None)

    assert total_non_draw_wins == expected_non_draw_wins
    assert analysis["overall_draws"] == expected_draws

    for condition, condition_summary in analysis["conditions"].items():
        source = summary["conditions"][condition]
        assert condition_summary["wins"] == source["wins"]
        assert condition_summary["losses"] == source["losses"]
        assert condition_summary["draws"] == source["draws"]


def test_analysis_artifacts_are_persisted(tmp_path):
    summary = build_analysis_ready_summary(tmp_path)

    export_result = export_task2_analysis_artifacts(summary, tmp_path / "analysis")

    representative_path = tmp_path / "analysis" / "task2_representative_matches.json"
    analysis_path = tmp_path / "analysis" / "task2_analysis_summary.json"

    assert representative_path.exists()
    assert analysis_path.exists()

    persisted_matches = json.loads(representative_path.read_text(encoding="utf-8"))
    persisted_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert set(persisted_matches) == set(CONDITIONS)
    assert set(persisted_analysis["conditions"]) == set(CONDITIONS)
    assert export_result["representative_matches_path"] == str(representative_path.resolve())
    assert export_result["analysis_summary_path"] == str(analysis_path.resolve())
