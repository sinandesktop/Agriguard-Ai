"""Load the saved Python model in a new process and predict one valid image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.preprocessing import HOG_FEATURE_COUNT, extract_hog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("ml/models/phase3"))
    parser.add_argument("--image", type=Path, default=None, help="Image to verify; defaults to the first test manifest image")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data/plantvillage-source/raw/color"))
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    image_path = args.image
    if image_path is None:
        import csv

        with Path("ml/data/splits/test.csv").open(newline="", encoding="utf-8") as stream:
            image_path = args.data_root / Path(next(csv.DictReader(stream))["path"])
    feature = extract_hog(image_path)
    if feature.shape != (HOG_FEATURE_COUNT,):
        raise RuntimeError(f"Unexpected feature shape: {feature.shape}")
    model = joblib.load(artifact_dir / metadata["model_artifact"])
    class_id = int(model.predict(feature.reshape(1, -1))[0])
    labels = metadata["class_id_to_label"]
    label = labels[str(class_id)]
    print(json.dumps({"image": str(image_path.resolve()), "feature_count": int(feature.size), "class_id": class_id, "label": label, "load_and_predict": "passed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
