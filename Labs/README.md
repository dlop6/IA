# Connect Four Task 2 Pipeline

This project contains the refactored Connect Four baseline plus the full Task 2 TD-learning pipeline under `src/connect4`.

## What is included

- Reusable `Connect4` game state and search baselines (`Minimax`, `AlphaBeta`)
- Linear TD Q-learning agent with deterministic state-action encoding
- Self-play training with checkpoints and JSON summaries
- Task 2 evaluation for conditions A, B, and C
- Plot generation with PDF export
- Representative match extraction and replay-ready analysis artifacts
- Tests covering rules, parity, training, evaluation, reporting, and end-to-end pipeline smoke
- A thin notebook client under `notebooks/Task2Lab6_refactored.ipynb`

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full test suite:

```powershell
pytest
```

Run the notebook smoke command:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/Task2Lab6_refactored.ipynb
```

## Recommended Task 2 runbook

### 1. Train the TD agent

```powershell
@'
from pathlib import Path
from connect4 import build_default_task2_epsilon_schedule, train_self_play

output_dir = Path("artifacts/task2/training")
schedule = build_default_task2_epsilon_schedule(
    episodes=5000,
    initial_epsilon=1.0,
    epsilon_end=0.1,
)

result = train_self_play(
    episodes=5000,
    learning_rate=0.1,
    discount=0.99,
    initial_epsilon=1.0,
    epsilon_schedule=schedule,
    checkpoint_interval=500,
    output_dir=output_dir,
    seed=42,
    stats_window_size=200,
)

print(result["summary"]["final_weights_path"])
'@ | python -
```

### 2. Run Task 2 evaluation

```powershell
@'
from pathlib import Path
from connect4 import run_task2_evaluation

summary = run_task2_evaluation(
    matches_per_condition=50,
    td_agent_path=Path("artifacts/task2/training/td_agent_final.npz"),
    td_epsilon=0.0,
    minimax_depth=4,
    alphabeta_depth=4,
    seed=99,
    output_dir=Path("artifacts/task2/evaluation"),
)

print(summary["conditions"]["A"]["wins"], summary["conditions"]["A"]["losses"], summary["conditions"]["A"]["draws"])
'@ | python -
```

### 3. Export the PDF figure

```powershell
@'
from pathlib import Path
from connect4 import export_task2_results_pdf

result = export_task2_results_pdf(
    Path("artifacts/task2/evaluation/task2_evaluation_summary.json"),
    Path("artifacts/task2/task2_results.pdf"),
)

print(result["pdf_path"])
'@ | python -
```

### 4. Export representative matches and analysis summary

```powershell
@'
from pathlib import Path
from connect4 import export_task2_analysis_artifacts

result = export_task2_analysis_artifacts(
    Path("artifacts/task2/evaluation/task2_evaluation_summary.json"),
    Path("artifacts/task2/analysis"),
)

print(result["analysis_summary_path"])
'@ | python -
```

### 5. One-command smoke pipeline

For a reduced end-to-end smoke run:

```powershell
@'
from pathlib import Path
from connect4 import run_task2_pipeline

result = run_task2_pipeline(
    Path("artifacts/task2_smoke"),
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

print(result["report"]["pdf_path"])
'@ | python -
```

## Default delivery configuration

- Training episodes: `5000`
- Learning rate: `0.1`
- Discount: `0.99`
- Epsilon schedule: linear decay from `1.0` to `0.1`
- Checkpoint interval: `500`
- Evaluation matches per condition: `50`
- Evaluation TD epsilon: `0.0`
- Minimax depth: `4`
- Alpha-beta depth: `4`

## Layout

- `src/connect4/` contains reusable logic for the full Task 2 pipeline
- `tests/` verifies parity, training, evaluation, reporting, and pipeline smoke
- `notebooks/` contains the demo and analysis notebook built on top of the package
