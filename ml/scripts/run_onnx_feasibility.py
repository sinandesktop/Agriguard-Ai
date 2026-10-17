"""Run the development-only synthetic SVM conversion check."""

from pathlib import Path
import json
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.feasibility import run_feasibility  # noqa: E402


if __name__ == "__main__":
    output = REPOSITORY_ROOT / "ml" / ".dev_artifacts"
    print(json.dumps(run_feasibility(output), indent=2))
