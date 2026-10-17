"""Synthetic SVM-to-ONNX feasibility check; contains no plant data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 31
FEATURE_COUNT = 8
CLASS_COUNT = 3


def make_synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic generic features, integer classes, and held-out vectors."""
    rng = np.random.default_rng(SEED)
    centers = rng.normal(0, 1, size=(CLASS_COUNT, FEATURE_COUNT)).astype(np.float32)
    centers *= 2.5
    train_x = np.vstack(
        [center + rng.normal(0, 0.55, size=(48, FEATURE_COUNT)) for center in centers]
    ).astype(np.float32)
    train_y = np.repeat(np.arange(CLASS_COUNT, dtype=np.int64), 48)
    test_x = np.vstack(
        [center + rng.normal(0, 0.55, size=(8, FEATURE_COUNT)) for center in centers]
    ).astype(np.float32)
    test_y = np.repeat(np.arange(CLASS_COUNT, dtype=np.int64), 8)
    return train_x, train_y, test_x, test_y


def build_classifier() -> Any:
    """Create a scaled linear SVM with a converter-friendly numeric input."""
    return make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=SEED))


def run_feasibility(output_dir: Path) -> dict[str, Any]:
    """Train, export, run in ONNX Runtime, and save a numeric parity report."""
    from onnxruntime import InferenceSession
    from skl2onnx import to_onnx

    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, _ = make_synthetic_data()
    model = build_classifier().fit(train_x, train_y)
    python_labels = model.predict(test_x).astype(np.int64)

    # ZipMap wraps SVC probabilities in a sequence of maps. Browsers can read
    # tensor outputs directly, so export class probabilities as a numeric tensor.
    onnx_model = to_onnx(
        model,
        train_x[:1].astype(np.float32),
        options={"svc__zipmap": False},
        target_opset=17,
    )
    model_path = output_dir / "synthetic_svm.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())

    session = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_types = [output.type for output in session.get_outputs()]
    if not all(output_type.startswith("tensor(") for output_type in output_types):
        raise AssertionError(f"ONNX outputs must be tensors for browser inference: {output_types}")
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: test_x.astype(np.float32)})
    onnx_labels = np.asarray(onnx_outputs[0], dtype=np.int64).reshape(-1)
    matches = np.array_equal(python_labels, onnx_labels)

    vectors_path = output_dir / "test_vectors.json"
    vectors_path.write_text(
        json.dumps(
            {
                "feature_count": FEATURE_COUNT,
                "vectors": test_x.tolist(),
                "expected_class_ids": python_labels.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "purpose": "synthetic_numeric_onnx_feasibility_only",
        "plant_images_or_labels_used": False,
        "feature_count": FEATURE_COUNT,
        "test_vector_count": int(len(test_x)),
        "python_onnx_predictions_match": bool(matches),
        "onnx_output_types": output_types,
        "onnx_model": model_path.name,
        "test_vectors": vectors_path.name,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not matches:
        raise AssertionError("Python and ONNX class predictions differ")
    return report
