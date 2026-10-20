"""Generate ignored RGB/feature fixtures from synthetic patterns and Phase 6.6 non-test images."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import joblib
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/scripts"))
import run_phase5_experiments as phase5  # noqa: E402

EXPERIMENT = ROOT / "ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1"
DATA_ROOT = ROOT / "ml/data/plantvillage-source/raw/color"
SPLITS = ROOT / "ml/data/splits/phase6_6"
OUTPUT = ROOT / "ml/.dev_artifacts/phase14/browser-feature-fixtures.json"
CLASSES = [
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fixture_pixels() -> list[dict]:
    y, x = np.indices((128, 128))
    checks = np.empty((128, 128, 3), dtype=np.uint8)
    checks[:] = [0, 0, 0]
    checks[(x // 16 + y // 16) % 2 == 0] = [255, 0, 0]
    checks[(x // 32) % 4 == 1] = [0, 255, 0]
    checks[(y // 32) % 4 == 2] = [0, 0, 255]

    y2, x2 = np.indices((173, 91))
    gradient = np.stack(((x2 * 13 + y2 * 7) % 256, (x2 * 3 + y2 * 19) % 256, (x2 * 23 + y2 * 5) % 256), axis=-1).astype(np.uint8)
    return [
        {"id": "synthetic-color-checker-128", "source_kind": "synthetic", "label": None, "rgb": checks},
        {"id": "synthetic-resize-gradient-91x173", "source_kind": "synthetic", "label": None, "rgb": gradient},
    ]


def representative_pixels() -> list[dict]:
    train = read_csv(SPLITS / "train.csv")
    validation = read_csv(SPLITS / "validation.csv")
    entries = []
    for label in CLASSES:
        source_split = "validation"
        candidates = [row for row in validation if row["label"] == label]
        if not candidates:
            source_split = "train"
            candidates = [row for row in train if row["label"] == label]
        if not candidates:
            raise ValueError(f"No non-test representative available for {label}")
        row = sorted(candidates, key=lambda item: item["path"])[0]
        path = DATA_ROOT / row["path"]
        with Image.open(path) as source:
            source.load()
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        entries.append({
            "id": "plantvillage-" + label,
            "source_kind": "plantvillage",
            "source_split": source_split,
            "source_path": row["path"],
            "label": label,
            "rgb": rgb,
        })
    return entries


def main() -> None:
    model = joblib.load(EXPERIMENT / "selected_model.joblib")
    entries = fixture_pixels() + representative_pixels()
    records = []
    with tempfile.TemporaryDirectory(prefix="agriguard-phase14-") as temporary:
        for index, entry in enumerate(entries):
            rgb = entry.pop("rgb")
            height, width, _ = rgb.shape
            image_path = Path(temporary) / f"fixture-{index}.png"
            Image.fromarray(rgb, mode="RGB").save(image_path, format="PNG")
            hog = phase5.extract_hog(image_path)
            hsv = phase5.color_histogram(image_path)
            combined = np.concatenate([hog, hsv]).astype(np.float32)
            prediction = int(model.predict(combined.reshape(1, -1))[0])
            decisions = model.decision_function(combined.reshape(1, -1))[0].astype(float).tolist()
            records.append({
                **entry,
                "width": width,
                "height": height,
                "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                "hog": hog.astype(np.float32).tolist(),
                "hsv": hsv.astype(np.float32).tolist(),
                "class_id": prediction,
                "decision_values": decisions,
            })
    session = ort.InferenceSession(str(EXPERIMENT / "selected_model.onnx"), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "X" or inputs[0].type != "tensor(float)" or inputs[0].shape != [None, 8132]:
        raise ValueError(f"Unexpected Phase 9 ONNX input contract: {[(i.name, i.type, i.shape) for i in inputs]}")
    if len(outputs) != 2 or outputs[0].name != "label" or outputs[0].type != "tensor(int64)" or outputs[1].name != "probabilities" or outputs[1].type != "tensor(float)":
        raise ValueError(f"Unexpected Phase 9 ONNX output contract: {[(o.name, o.type, o.shape) for o in outputs]}")
    feature_matrix = np.asarray([[*r["hog"], *r["hsv"]] for r in records], dtype=np.float32)
    py_labels = np.asarray([r["class_id"] for r in records], dtype=np.int64)
    py_decisions = np.asarray([r["decision_values"] for r in records], dtype=np.float32)
    onnx_labels, onnx_scores = session.run(None, {"X": feature_matrix})
    if not np.array_equal(py_labels, onnx_labels):
        raise ValueError("Python and ONNX class predictions differ on the parity fixtures")
    score_difference = np.abs(py_decisions - onnx_scores)
    if float(score_difference.max()) > 5e-4:
        raise ValueError(f"ONNX SVM decision outputs differ from Python: max abs {score_difference.max()}")

    model_hash = hashlib.sha256()
    with (EXPERIMENT / "selected_model.onnx").open("rb") as model_stream:
        for chunk in iter(lambda: model_stream.read(1024 * 1024), b""):
            model_hash.update(chunk)
    payload = {
        "fixture_version": 1,
        "phase9_model_sha256": model_hash.hexdigest(),
        "onnx_interface": {
            "inputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in inputs],
            "outputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in outputs],
            "python_onnx_prediction_matches": int(np.sum(py_labels == onnx_labels)),
            "prediction_count": len(records),
            "score_output_max_abs_difference": float(score_difference.max()),
            "score_output_semantics": "Raw SVC decision_function scores; not probabilities despite the ONNX output name.",
        },
        "class_id_to_label": {str(i): value for i, value in enumerate(CLASSES)},
        "feature_config_path": "ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1/feature_config.json",
        "python_feature_source": "ml/scripts/run_phase5_experiments.py: extract_hog + color_histogram",
        "entries": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "entries": len(records), "samples": [{"id": r["id"], "path": r.get("source_path"), "split": r.get("source_split"), "label": r.get("label"), "width": r["width"], "height": r["height"]} for r in records]}, indent=2))


if __name__ == "__main__":
    main()
