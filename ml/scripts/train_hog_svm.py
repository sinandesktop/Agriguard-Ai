"""Train the Phase 3 HOG + SVM baseline from prepared split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.dataset import configured_dataset_root  # noqa: E402
from agriguard_ml.training import train_and_evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=configured_dataset_root())
    parser.add_argument("--split-dir", type=Path, default=Path("ml/data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/models/phase3"))
    args = parser.parse_args()
    try:
        report = train_and_evaluate(args.data_root, args.split_dir, args.output_dir)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "validation_accuracy": report["validation"]["accuracy"],
        "validation_macro_f1": report["validation"]["macro_f1"],
        "test_accuracy": report["test"]["accuracy"],
        "test_macro_f1": report["test"]["macro_f1"],
        "onnx_status": report["onnx"]["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
