"""Train and evaluate a real HOG + SVM baseline using prepared CSV manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import (
    BLOCK_NORM,
    CELLS_PER_BLOCK,
    HOG_FEATURE_COUNT,
    HOG_ORIENTATIONS,
    HOG_TRANSFORM_SQRT,
    IMAGE_SIZE,
    PIXELS_PER_CELL,
    extract_hog,
)


RANDOM_SEED = 2026
SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_CLASSES = {
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
}
SVC_PARAMETERS: dict[str, Any] = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": False,
    "random_state": RANDOM_SEED,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(manifest_path: Path, data_root: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "sha256", "group_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest {manifest_path} must contain columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    for row in rows:
        path = data_root / Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {path}")
        if Path(row["path"]).parts[0] != row["label"]:
            raise ValueError(f"Manifest label does not match image class folder: {row['path']}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"Manifest image hash no longer matches: {row['path']}")
    return rows


def _features(rows: list[dict[str, str]], data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    for index, row in enumerate(rows, start=1):
        features.append(extract_hog(data_root / Path(row["path"])))
        labels.append(row["label"])
        if index % 250 == 0 or index == len(rows):
            print(f"HOG {split}: {index}/{len(rows)} images", flush=True)
    return np.vstack(features).astype(np.float32, copy=False), np.asarray(labels)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]) -> dict[str, Any]:
    label_ids = np.arange(len(class_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(class_labels)
        },
        "confusion_matrix": {
            "labels": class_labels,
            "values": matrix.astype(int).tolist(),
        },
    }


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _write_report(output_path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# AgriGuard HOG + SVM baseline", "",
        "This is a classical machine-learning model trained on the selected PlantVillage color images. It has not been validated on field photographs.", "",
        f"- Dataset root: `{result['dataset_root']}`",
        f"- Seed: {result['random_seed']}",
        f"- Training images: {result['split_sizes']['train']}",
        f"- Validation images: {result['split_sizes']['validation']}",
        f"- Test images: {result['split_sizes']['test']}",
        f"- HOG extraction seconds: {result['durations_seconds']['hog_extraction_total']:.2f}",
        f"- SVM fit seconds: {result['durations_seconds']['fit']:.2f}",
        f"- Validation evaluation seconds: {result['durations_seconds']['validation_evaluation']:.2f}",
        f"- Test evaluation seconds: {result['durations_seconds']['test_evaluation']:.2f}", "",
        "## Validation metrics", "",
    ]
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        lines.append(f"- {key.replace('_', ' ').title()}: {result['validation'][key]:.4f}")
    lines.extend(["", "## Test metrics (held out)", ""])
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        lines.append(f"- {key.replace('_', ' ').title()}: {result['test'][key]:.4f}")
    lines.extend(["", "### Per-class test metrics", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
    for label, metrics in result["test"]["per_class"].items():
        lines.append(f"| `{label}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |")
    labels = result["test"]["confusion_matrix"]["labels"]
    matrix = result["test"]["confusion_matrix"]["values"]
    lines.extend(["", "### Test confusion matrix", "", "Rows are actual classes and columns are predicted classes.", ""])
    lines.append("| Actual \\ Predicted | " + " | ".join(f"`{label}`" for label in labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for label, values in zip(labels, matrix):
        lines.append(f"| `{label}` | " + " | ".join(str(value) for value in values) + " |")
    lines.extend(["", "## ONNX", "", f"- Conversion/parity status: **{result['onnx']['status']}**"])
    if result["onnx"].get("error"):
        lines.append(f"- Details: `{result['onnx']['error']}`")
    elif result["onnx"].get("parity"):
        lines.append(
            f"- Test-vector class-label parity: {result['onnx']['parity']['matched_count']}/{result['onnx']['parity']['vector_count']} matched; "
            f"{result['onnx']['parity']['mismatch_count']} mismatches."
        )
    lines.extend(["", "## Limitations", "", "PlantVillage images use controlled backgrounds. These scores do not establish performance on real farm images."])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_evaluate(data_root: Path, split_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Train one fixed baseline, evaluate validation/test once, save model and ONNX parity."""
    data_root = data_root.expanduser().resolve()
    split_dir = split_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty artifact directory: {output_dir}")
    manifests = {name: split_dir / f"{name}.csv" for name in SPLIT_NAMES}
    rows = {name: _read_manifest(path, data_root) for name, path in manifests.items()}
    label_sets = {name: {row["label"] for row in split_rows} for name, split_rows in rows.items()}
    if any(labels != EXPECTED_CLASSES for labels in label_sets.values()):
        raise ValueError(f"Each split must contain the six expected classes; found {label_sets}")

    # Each split's features are computed once in memory; the test features are not used for fitting.
    extraction_start = time.perf_counter()
    features = {name: _features(rows[name], data_root, name) for name in SPLIT_NAMES}
    extraction_seconds = time.perf_counter() - extraction_start
    class_labels = sorted(EXPECTED_CLASSES)
    class_to_id = {label: index for index, label in enumerate(class_labels)}
    x_train, train_labels = features["train"]
    x_validation, validation_labels = features["validation"]
    x_test, test_labels = features["test"]
    y_train = np.asarray([class_to_id[label] for label in train_labels], dtype=np.int64)
    y_validation = np.asarray([class_to_id[label] for label in validation_labels], dtype=np.int64)
    y_test = np.asarray([class_to_id[label] for label in test_labels], dtype=np.int64)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**SVC_PARAMETERS)),
    ])
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    validation_start = time.perf_counter()
    validation_pred = model.predict(x_validation).astype(np.int64)
    validation_metrics = _metrics(y_validation, validation_pred, class_labels)
    validation_seconds = time.perf_counter() - validation_start

    # Configuration is fixed before this one held-out test evaluation.
    test_start = time.perf_counter()
    test_pred = model.predict(x_test).astype(np.int64)
    test_metrics = _metrics(y_test, test_pred, class_labels)
    test_seconds = time.perf_counter() - test_start

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hog_svm.joblib"
    joblib.dump(model, model_path)

    onnx_result: dict[str, Any]
    try:
        from onnxruntime import InferenceSession
        from skl2onnx import to_onnx

        onnx_model = to_onnx(
            model,
            x_train[:1].astype(np.float32),
            target_opset=17,
            options={"svc__zipmap": False},
        )
        onnx_bytes = onnx_model.SerializeToString()
        session = InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_types = [item.type for item in session.get_outputs()]
        if input_meta.type != "tensor(float)" or len(input_meta.shape) != 2 or input_meta.shape[1] != HOG_FEATURE_COUNT:
            raise ValueError(f"Unexpected ONNX input schema: {input_meta.type}, {input_meta.shape}")
        if not all(value.startswith("tensor(") for value in output_types):
            raise ValueError(f"ONNX model has a non-tensor output: {output_types}")
        onnx_outputs = session.run(None, {input_meta.name: x_test.astype(np.float32)})
        onnx_pred = np.asarray(onnx_outputs[0], dtype=np.int64).reshape(-1)
        mismatches = np.flatnonzero(onnx_pred != test_pred)
        parity = {
            "vector_count": int(len(test_pred)),
            "matched_count": int(len(test_pred) - len(mismatches)),
            "mismatch_count": int(len(mismatches)),
            "mismatch_indices": mismatches.astype(int).tolist(),
            "python_class_ids": test_pred.tolist(),
            "onnx_class_ids": onnx_pred.tolist(),
            "class_id_to_label": {str(value): label for label, value in class_to_id.items()},
        }
        (output_dir / "hog_svm.onnx").write_bytes(onnx_bytes)
        onnx_result = {
            "status": "passed" if len(mismatches) == 0 else "mismatch",
            "input_type": input_meta.type,
            "input_shape": input_meta.shape,
            "output_types": output_types,
            "parity": parity,
        }
        (output_dir / "onnx_parity.json").write_text(json.dumps(onnx_result, indent=2) + "\n", encoding="utf-8")
    except Exception as error:  # Preserve the already-saved Python model on converter/runtime failure.
        onnx_result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        (output_dir / "onnx_parity.json").write_text(json.dumps(onnx_result, indent=2) + "\n", encoding="utf-8")

    class_counts = {
        split: dict(sorted(Counter(row["label"] for row in split_rows).items()))
        for split, split_rows in rows.items()
    }
    metrics_result = {
        "dataset_root": str(data_root),
        "model": "StandardScaler + RBF SVC (class_weight=balanced)",
        "random_seed": RANDOM_SEED,
        "svc_parameters": SVC_PARAMETERS,
        "class_id_to_label": {str(value): label for label, value in class_to_id.items()},
        "class_counts": class_counts,
        "split_sizes": {split: len(split_rows) for split, split_rows in rows.items()},
        "durations_seconds": {
            "hog_extraction_total": extraction_seconds,
            "fit": fit_seconds,
            "validation_evaluation": validation_seconds,
            "test_evaluation": test_seconds,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "onnx": onnx_result,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_result, indent=2) + "\n", encoding="utf-8")

    preprocessing = {
        "image_size": list(IMAGE_SIZE),
        "resize_interpolation": "Pillow bilinear",
        "color_handling": "Pillow convert('L') grayscale, uint8 0-255",
        "exif_orientation_applied": False,
        "hog_orientations": HOG_ORIENTATIONS,
        "pixels_per_cell": list(PIXELS_PER_CELL),
        "cells_per_block": list(CELLS_PER_BLOCK),
        "block_norm": BLOCK_NORM,
        "transform_sqrt": HOG_TRANSFORM_SQRT,
        "feature_count": HOG_FEATURE_COUNT,
        "feature_dtype": "float32",
    }
    metadata = {
        "model_artifact": model_path.name,
        "onnx_artifact": "hog_svm.onnx" if onnx_result["status"] in {"passed", "mismatch"} else None,
        "class_id_to_label": metrics_result["class_id_to_label"],
        "preprocessing": preprocessing,
        "training_parameters": {"pipeline": ["StandardScaler", "SVC"], "svc": SVC_PARAMETERS},
        "random_seed": RANDOM_SEED,
        "dataset_root": str(data_root),
        "split_manifests": {
            name: {"path": str(manifests[name]), "sha256": _sha256(manifests[name])}
            for name in SPLIT_NAMES
        },
        "library_versions": {
            name: _package_version(distribution)
            for name, distribution in {
                "python": "pip", "numpy": "numpy", "scikit_learn": "scikit-learn",
                "scikit_image": "scikit-image", "pillow": "Pillow", "joblib": "joblib",
                "skl2onnx": "skl2onnx", "onnxruntime": "onnxruntime",
            }.items()
        },
        "evaluation_metrics": {"validation": validation_metrics, "test": test_metrics},
        "durations_seconds": metrics_result["durations_seconds"],
        "limitations": ["PlantVillage uses controlled image backgrounds; field performance is unverified."],
    }
    metadata["library_versions"]["python"] = platform.python_version()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / "report.md", metrics_result)
    return metrics_result
