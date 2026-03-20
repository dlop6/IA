from pathlib import Path

from connect4 import (
    CONDITIONS,
    create_task2_results_figure,
    export_task2_results_pdf,
    extract_task2_result_counts,
    load_task2_evaluation_summary,
    run_task2_evaluation,
    train_self_play,
)


def build_report_ready_summary(tmp_path):
    td_agent = train_self_play(
        episodes=4,
        initial_epsilon=0.8,
        checkpoint_interval=2,
        seed=24,
        stats_window_size=4,
    )["agent"]

    return run_task2_evaluation(
        matches_per_condition=3,
        td_agent=td_agent,
        td_epsilon=0.0,
        seed=41,
        output_dir=tmp_path,
    )


def test_report_pipeline_loads_saved_evaluation_summary(tmp_path):
    summary = build_report_ready_summary(tmp_path)

    loaded = load_task2_evaluation_summary(tmp_path / "task2_evaluation_summary.json")

    assert loaded["matches_per_condition"] == summary["matches_per_condition"]
    assert loaded["seed"] == summary["seed"]
    assert set(loaded["conditions"]) == set(CONDITIONS)


def test_extract_result_counts_matches_evaluation_totals(tmp_path):
    summary = build_report_ready_summary(tmp_path)

    counts = extract_task2_result_counts(summary)

    assert set(counts) == set(CONDITIONS)
    for condition, condition_counts in counts.items():
        expected = summary["conditions"][condition]
        assert condition_counts["wins"] == expected["wins"]
        assert condition_counts["losses"] == expected["losses"]
        assert condition_counts["draws"] == expected["draws"]
        assert (
            condition_counts["wins"] + condition_counts["losses"] + condition_counts["draws"]
            == condition_counts["num_matches"]
        )


def test_create_figure_has_required_metadata(tmp_path):
    summary = build_report_ready_summary(tmp_path)

    fig, counts = create_task2_results_figure(summary)
    ax = fig.axes[0]

    assert ax.get_title() == "Task 2 Results: Wins, Losses, and Draws by Condition"
    assert ax.get_xlabel() == "Condition"
    assert ax.get_ylabel() == "Number of Matches"
    assert ax.get_legend() is not None
    assert set(counts) == set(CONDITIONS)


def test_export_results_pdf_creates_file_and_preserves_counts(tmp_path):
    summary = build_report_ready_summary(tmp_path)
    pdf_path = tmp_path / "task2_results.pdf"

    export_result = export_task2_results_pdf(summary, pdf_path)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert export_result["pdf_path"] == str(pdf_path.resolve())
    assert set(export_result["counts"]) == set(CONDITIONS)
