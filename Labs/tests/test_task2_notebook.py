import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_task2_pipeline_notebook_executes_top_to_bottom():
    if shutil.which("jupyter") is None:
        return

    notebook = ROOT / "notebooks" / "Task2_pipeline.ipynb"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        executed_notebook = tmpdir_path / "Task2_pipeline.executed.ipynb"
        artifact_dir = tmpdir_path / "artifacts"
        env = os.environ.copy()
        env["TASK2_NOTEBOOK_PROFILE"] = "smoke"
        env["TASK2_NOTEBOOK_OUTPUT_DIR"] = str(artifact_dir)

        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--output",
                str(executed_notebook),
                "--ExecutePreprocessor.timeout=900",
                str(notebook),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert (artifact_dir / "task2_results.pdf").exists()
        assert (artifact_dir / "evaluation" / "task2_evaluation_summary.json").exists()
        assert (artifact_dir / "notebook_visuals" / "task2_results.png").exists()
        assert (artifact_dir / "notebook_visuals" / "representative_condition_A.png").exists()
