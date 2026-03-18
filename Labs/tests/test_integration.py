import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from connect4 import (
    AlphaBetaAgent,
    Connect4,
    MinimaxAgent,
    RandomAgent,
    SmartAlphaBetaAgent,
    play_match,
    run_ai_vs_random_series,
)

from tests.baseline_data import SEEDED_AI_VS_RANDOM, SEEDED_AI_VS_RANDOM_SERIES


ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_match_completes_legally():
    match = play_match(
        MinimaxAgent(ai_player=Connect4.PLAYER1, depth=4),
        AlphaBetaAgent(ai_player=Connect4.PLAYER2, depth=4),
        verbose=False,
        visual=False,
        print_result=False,
    )

    assert match["move_count"] > 0
    assert match["winner"] in {None, Connect4.PLAYER1, Connect4.PLAYER2}
    assert len(match["moves"]) == match["move_count"]


def test_seeded_ai_vs_random_match_matches_frozen_baseline():
    random.seed(SEEDED_AI_VS_RANDOM["seed"])
    match = play_match(
        RandomAgent(ai_player=Connect4.PLAYER1),
        SmartAlphaBetaAgent(ai_player=Connect4.PLAYER2, depth=6),
        verbose=False,
        visual=False,
        result_labels={
            Connect4.PLAYER1: "Aleatorio (X) gana.",
            Connect4.PLAYER2: "IA (O) GANA!",
            "draw": "Empate.",
        },
        move_labels={
            Connect4.PLAYER1: "Aleatorio (X)",
            Connect4.PLAYER2: "IA (O)",
        },
        print_result=False,
    )

    assert match["winner"] == SEEDED_AI_VS_RANDOM["winner"]
    assert match["result"] == SEEDED_AI_VS_RANDOM["result"]
    assert match["move_count"] == SEEDED_AI_VS_RANDOM["move_count"]
    assert [move["column"] for move in match["moves"]] == SEEDED_AI_VS_RANDOM["columns"]
    assert np.array_equal(match["board"], np.array(SEEDED_AI_VS_RANDOM["final_board"]))


def test_seeded_ai_vs_random_series_matches_frozen_baseline(capsys):
    counts = run_ai_vs_random_series(
        num_games=SEEDED_AI_VS_RANDOM_SERIES["num_games"],
        ai_depth=6,
        seed=SEEDED_AI_VS_RANDOM_SERIES["seed"],
    )
    captured = capsys.readouterr()

    assert counts == SEEDED_AI_VS_RANDOM_SERIES["counts"]
    assert "Resultados en 10 partidas:" in captured.out


def test_notebook_smoke_executes_top_to_bottom():
    if shutil.which("jupyter") is None:
        return

    notebook = ROOT / "notebooks" / "Task2Lab6_refactored.ipynb"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        executed_notebook = tmpdir_path / "Task2Lab6_refactored.executed.ipynb"
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--output",
                str(executed_notebook),
                "--ExecutePreprocessor.timeout=600",
                str(notebook),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
