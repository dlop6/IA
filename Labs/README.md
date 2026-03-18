# Connect Four Baseline Refactor

This refactor extracts the reusable Connect Four logic from the original lab notebook into a small Python package under `src/connect4`.

## What is included

- `Connect4` game state and rules
- `Minimax`, `AlphaBeta`, and heuristic alpha-beta agents
- Text and matplotlib board rendering helpers
- Reusable match utilities
- Tests for game rules, agent parity, and integration
- A thin notebook client under `notebooks/Task2Lab6_refactored.ipynb`

## Quick start

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Run the tests:

```powershell
pytest
```

Run the notebook smoke command from a clean environment:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/Task2Lab6_refactored.ipynb
```

## Layout

- `src/connect4/` contains reusable logic ready for future TD integration.
- `tests/` verifies parity against the frozen notebook baseline.
- `notebooks/` contains the demo and analysis notebook built on top of the package.
