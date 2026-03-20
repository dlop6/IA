from connect4 import (
    DEFAULT_TASK2_CONFIG,
    build_default_task2_epsilon_schedule,
    run_task2_pipeline,
)


def test_default_schedule_builder_matches_training_defaults():
    schedule = build_default_task2_epsilon_schedule(
        episodes=DEFAULT_TASK2_CONFIG["training"]["episodes"],
        initial_epsilon=DEFAULT_TASK2_CONFIG["training"]["initial_epsilon"],
        epsilon_end=DEFAULT_TASK2_CONFIG["training"]["epsilon_end"],
    )

    assert schedule.value_at(0) == DEFAULT_TASK2_CONFIG["training"]["initial_epsilon"]
    assert schedule.value_at(DEFAULT_TASK2_CONFIG["training"]["episodes"]) == DEFAULT_TASK2_CONFIG["training"]["epsilon_end"]


def test_full_task2_pipeline_smoke_run_creates_all_key_artifacts(tmp_path):
    result = run_task2_pipeline(
        tmp_path,
        training_config={
            "episodes": 6,
            "checkpoint_interval": 3,
            "seed": 15,
            "stats_window_size": 4,
            "initial_epsilon": 0.9,
            "epsilon_end": 0.2,
        },
        evaluation_config={
            "matches_per_condition": 3,
            "seed": 27,
            "td_epsilon": 0.0,
        },
    )

    assert (tmp_path / "training" / "training_summary.json").exists()
    assert (tmp_path / "training" / "td_agent_final.npz").exists()
    assert (tmp_path / "evaluation" / "task2_evaluation_summary.json").exists()
    assert (tmp_path / "analysis" / "task2_analysis_summary.json").exists()
    assert (tmp_path / "analysis" / "task2_representative_matches.json").exists()
    assert (tmp_path / "task2_results.pdf").exists()

    assert result["training"]["episodes"] == 6
    assert result["evaluation"]["matches_per_condition"] == 3
    assert "pdf_path" in result["report"]
    assert "analysis_summary_path" in result["analysis"]
