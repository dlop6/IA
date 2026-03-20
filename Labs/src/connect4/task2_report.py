import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_LABELS = ("wins", "losses", "draws")
RESULT_COLORS = {
    "wins": "#2E8B57",
    "losses": "#C0392B",
    "draws": "#7F8C8D",
}


def load_task2_evaluation_summary(path):
    """Load a persisted Task 2 evaluation summary JSON file."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_task2_result_counts(summary):
    """Extract wins, losses, and draws for each Task 2 condition."""
    counts = {}
    for condition, condition_summary in summary["conditions"].items():
        counts[condition] = {
            "wins": int(condition_summary["wins"]),
            "losses": int(condition_summary["losses"]),
            "draws": int(condition_summary["draws"]),
            "num_matches": int(condition_summary["num_matches"]),
            "agent1_label": condition_summary["agent1_label"],
            "agent2_label": condition_summary["agent2_label"],
        }
    return counts


def create_task2_results_figure(summary):
    """
    Create the grouped bar chart required by Task 2.2.

    Results are plotted from the perspective of the first-listed agent in each condition.
    """
    counts = extract_task2_result_counts(summary)
    conditions = list(counts.keys())
    x_positions = list(range(len(conditions)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    for offset, label in enumerate(RESULT_LABELS):
        bar_positions = [x + (offset - 1) * width for x in x_positions]
        values = [counts[condition][label] for condition in conditions]
        ax.bar(
            bar_positions,
            values,
            width=width,
            label=label.capitalize(),
            color=RESULT_COLORS[label],
        )

    tick_labels = [
        f"{condition}: {counts[condition]['agent1_label']} vs {counts[condition]['agent2_label']}"
        for condition in conditions
    ]
    ax.set_title("Task 2 Results: Wins, Losses, and Draws by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Number of Matches")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels)
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    return fig, counts


def export_task2_results_pdf(summary_or_path, output_pdf_path):
    """Generate the Task 2 results plot and export it to PDF."""
    summary = (
        load_task2_evaluation_summary(summary_or_path)
        if isinstance(summary_or_path, (str, Path))
        else summary_or_path
    )
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, counts = create_task2_results_figure(summary)
    fig.savefig(output_pdf_path, format="pdf")
    plt.close(fig)
    return {
        "pdf_path": str(output_pdf_path.resolve()),
        "counts": counts,
    }
