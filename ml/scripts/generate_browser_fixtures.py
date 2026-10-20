"""Create ignored held-out image and numeric fixtures for browser HOG parity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/src"))
from agriguard_ml.preprocessing import extract_hog, preprocess_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "ml/data/plantvillage-source/raw/color")
    parser.add_argument("--manifest", type=Path, default=ROOT / "ml/data/splits/test.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "ml/models/phase3/hog_svm.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "ml/.dev_artifacts/phase4/browser-fixtures")
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()
    if args.per_class < 1 or not args.data_root.is_dir() or not args.manifest.is_file() or not args.model.is_file():
        parser.error("Need positive --per-class and existing held-out dataset, manifest, and Phase 3 model files.")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    class_ids = {label: index for index, label in enumerate(sorted({row["label"] for row in rows}))}
    selected: list[dict[str, str]] = []
    counts = {label: 0 for label in class_ids}
    for row in rows:
        if counts.get(row["label"], args.per_class) < args.per_class:
            selected.append(row)
            counts[row["label"]] += 1
    if not selected or any(count != args.per_class for count in counts.values()):
        parser.error(f"Manifest does not contain {args.per_class} examples in every class: {counts}")
    model = joblib.load(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for index, row in enumerate(selected):
        source = args.data_root / row["path"]
        relative = Path("images") / f"{index:02d}{source.suffix.lower()}"
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        pixels = preprocess_image(source)
        with Image.open(source) as image:
            source_gray = np.asarray(image.convert("L"), dtype=np.uint8)
        feature = extract_hog(source)
        predicted = int(model.predict(feature.reshape(1, -1))[0])
        decision = model.decision_function(feature.reshape(1, -1))[0]
        fixtures.append({
            "source": relative.as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_size": [int(source_gray.shape[1]), int(source_gray.shape[0])],
            "manifest_path": row["path"], "label": row["label"],
            "true_class_id": class_ids[row["label"]], "predicted_class_id": predicted,
            "grayscale_source": source_gray.reshape(-1).astype(int).tolist(),
            "grayscale_128": pixels.reshape(-1).astype(int).tolist(),
            "hog_float32": feature.tolist(),
            "svm_decision_values": np.asarray(decision, dtype=float).tolist(),
        })
    (args.output / "fixtures.json").write_text(json.dumps({
        "provenance": "Held-out Phase 3 test split. Development-only. Predictions are saved-model outputs, not newly measured metrics.",
        "image_size": [128, 128], "feature_count": 8100, "class_id_to_label": {str(i): label for label, i in class_ids.items()},
        "fixtures": fixtures,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fixtures": len(fixtures), "per_class": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
