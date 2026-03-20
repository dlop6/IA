from pathlib import Path

from .exploration import LinearDecayEpsilonSchedule
from .task2_analysis import export_task2_analysis_artifacts
from .task2_eval import run_task2_evaluation
from .task2_report import export_task2_results_pdf
from .training import train_self_play


DEFAULT_TASK2_CONFIG = {
    "training": {
        "episodes": 5000,
        "learning_rate": 0.1,
        "discount": 0.99,
        "initial_epsilon": 1.0,
        "epsilon_end": 0.1,
        "checkpoint_interval": 500,
        "seed": 42,
        "stats_window_size": 200,
    },
    "evaluation": {
        "matches_per_condition": 50,
        "td_epsilon": 0.0,
        "minimax_depth": 4,
        "alphabeta_depth": 4,
        "seed": 99,
    },
}


def build_default_task2_epsilon_schedule(*, episodes, initial_epsilon, epsilon_end):
    """Create the default linear epsilon schedule used by the final Task 2 pipeline."""
    decay_steps = max(1, int(episodes))
    return LinearDecayEpsilonSchedule(
        start=float(initial_epsilon),
        end=float(epsilon_end),
        decay_steps=decay_steps,
    )


def run_task2_pipeline(
    output_dir,
    *,
    training_config=None,
    evaluation_config=None,
):
    """
    Run the full Task 2 pipeline end-to-end:
    training -> evaluation -> PDF export -> analysis artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_defaults = dict(DEFAULT_TASK2_CONFIG["training"])
    evaluation_defaults = dict(DEFAULT_TASK2_CONFIG["evaluation"])

    if training_config is not None:
        training_defaults.update(training_config)
    if evaluation_config is not None:
        evaluation_defaults.update(evaluation_config)

    schedule = build_default_task2_epsilon_schedule(
        episodes=training_defaults["episodes"],
        initial_epsilon=training_defaults["initial_epsilon"],
        epsilon_end=training_defaults["epsilon_end"],
    )

    training_output_dir = output_dir / "training"
    evaluation_output_dir = output_dir / "evaluation"
    analysis_output_dir = output_dir / "analysis"
    pdf_path = output_dir / "task2_results.pdf"

    training_result = train_self_play(
        episodes=training_defaults["episodes"],
        learning_rate=training_defaults["learning_rate"],
        discount=training_defaults["discount"],
        initial_epsilon=training_defaults["initial_epsilon"],
        epsilon_schedule=schedule,
        checkpoint_interval=training_defaults["checkpoint_interval"],
        output_dir=training_output_dir,
        seed=training_defaults["seed"],
        stats_window_size=training_defaults["stats_window_size"],
    )

    evaluation_summary = run_task2_evaluation(
        matches_per_condition=evaluation_defaults["matches_per_condition"],
        td_agent=training_result["agent"],
        td_epsilon=evaluation_defaults["td_epsilon"],
        minimax_depth=evaluation_defaults["minimax_depth"],
        alphabeta_depth=evaluation_defaults["alphabeta_depth"],
        seed=evaluation_defaults["seed"],
        output_dir=evaluation_output_dir,
    )

    pdf_result = export_task2_results_pdf(evaluation_summary, pdf_path)
    analysis_result = export_task2_analysis_artifacts(evaluation_summary, analysis_output_dir)

    return {
        "config": {
            "training": training_defaults,
            "evaluation": evaluation_defaults,
        },
        "training": training_result["summary"],
        "evaluation": evaluation_summary,
        "report": pdf_result,
        "analysis": analysis_result,
    }
