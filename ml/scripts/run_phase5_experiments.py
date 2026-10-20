"""Run a small validation-only HOG/SVM experiment, then finalize one frozen candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/src"))
from agriguard_ml.preprocessing import (  # noqa: E402
    BLOCK_NORM, CELLS_PER_BLOCK, HOG_FEATURE_COUNT, HOG_ORIENTATIONS,
    HOG_TRANSFORM_SQRT, IMAGE_SIZE, PIXELS_PER_CELL, extract_hog,
)
from agriguard_ml.training import RANDOM_SEED, SPLIT_NAMES, _read_manifest  # noqa: E402

FEATURE_SPEC = {
    "hog": {"orientations": HOG_ORIENTATIONS, "pixels_per_cell": list(PIXELS_PER_CELL),
            "cells_per_block": list(CELLS_PER_BLOCK), "block_norm": BLOCK_NORM,
            "transform_sqrt": HOG_TRANSFORM_SQRT, "length": HOG_FEATURE_COUNT},
    "color": {"space": "HSV via scikit-image rgb2hsv", "bins": {"H": 16, "S": 8, "V": 8},
              "range": [0, 1], "normalization": "per-channel histogram divided by pixel count",
              "image": "Pillow RGB, bilinear resize to 128x128 before HSV conversion"},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(data_root: Path, split_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows = {name: _read_manifest(split_dir / f"{name}.csv", data_root) for name in SPLIT_NAMES}
    sets = {name: {row["label"] for row in content} for name, content in rows.items()}
    if len(set.union(*sets.values())) != 6 or any(s != sets["train"] for s in sets.values()):
        raise ValueError(f"Splits must share the same six classes: {sets}")
    paths = {name: {row["path"] for row in content} for name, content in rows.items()}
    if any(paths[a] & paths[b] for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]):
        raise ValueError("Manifest image paths overlap across splits")
    return rows


def color_histogram(path: Path) -> np.ndarray:
    from skimage.color import rgb2hsv

    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    parts = [np.histogram(hsv[..., channel], bins=bins, range=(0.0, 1.0))[0] / hsv.shape[0] / hsv.shape[1]
             for channel, bins in enumerate((16, 8, 8))]
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError(f"Unexpected color histogram for {path}: {result.shape}")
    return result


def extract_features(rows: list[dict[str, str]], root: Path, tag: str, include_color: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    hog_rows, color_rows = [], []
    labels = []
    durations = {"hog": 0.0, "color_histogram": 0.0}
    for i, row in enumerate(rows, 1):
        image = root / row["path"]
        started = time.perf_counter()
        hog_rows.append(extract_hog(image))
        durations["hog"] += time.perf_counter() - started
        if include_color:
            started = time.perf_counter()
            color_rows.append(color_histogram(image))
            durations["color_histogram"] += time.perf_counter() - started
        labels.append(row["label"])
        if i % 250 == 0 or i == len(rows):
            print(f"features {tag}: {i}/{len(rows)}", flush=True)
    colors = np.vstack(color_rows).astype(np.float32) if color_rows else np.empty((len(rows), 0), dtype=np.float32)
    return np.vstack(hog_rows).astype(np.float32), colors, np.asarray(labels), durations


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    ids = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=ids, average=None, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=ids, average="macro", zero_division=0)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(labels)},
        "confusion_matrix": {"labels": labels, "values": confusion_matrix(y_true, y_pred, labels=ids).astype(int).tolist()},
    }


def candidates(class_labels: list[str], y_train: np.ndarray) -> dict[str, dict[str, Any]]:
    healthy_id = class_labels.index("Potato___healthy")
    counts = np.bincount(y_train, minlength=len(class_labels))
    balanced = len(y_train) / (len(class_labels) * counts)
    boosted = {i: float(value) for i, value in enumerate(balanced)}
    boosted[healthy_id] *= 2
    common = {"kernel": "rbf", "gamma": "scale", "probability": False, "random_state": RANDOM_SEED}
    return {
        "baseline_hog_C1_balanced": {"features": "hog", "svc": {**common, "C": 1.0, "class_weight": "balanced"}},
        "hog_hsv_C1_balanced": {"features": "hog_hsv", "svc": {**common, "C": 1.0, "class_weight": "balanced"}},
        "hog_C2_balanced": {"features": "hog", "svc": {**common, "C": 2.0, "class_weight": "balanced"}},
        "hog_C4_balanced": {"features": "hog", "svc": {**common, "C": 4.0, "class_weight": "balanced"}},
        "hog_C1_unweighted": {"features": "hog", "svc": {**common, "C": 1.0, "class_weight": None}},
        "hog_C1_healthy_2x": {"features": "hog", "svc": {**common, "C": 1.0, "class_weight": boosted}},
    }


def model_for(spec: dict[str, Any]) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("svc", SVC(**spec["svc"]) )])


def baseline_hashes() -> dict[str, str]:
    folder = ROOT / "ml/models/phase3"
    names = ("hog_svm.joblib", "hog_svm.onnx", "metadata.json", "metrics.json", "onnx_parity.json", "report.md")
    return {name: digest(folder / name) for name in names}


def run_validation(args: argparse.Namespace) -> None:
    if args.experiment_dir.exists() and any(args.experiment_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment output: {args.experiment_dir}")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()
    split_dir = args.split_dir.resolve()
    rows = load_rows(data_root, split_dir)
    class_labels = sorted({row["label"] for row in rows["train"]})
    class_to_id = {label: i for i, label in enumerate(class_labels)}
    hashes = {name: digest(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    base_hashes = baseline_hashes()
    cache_key = hashlib.sha256(json.dumps({"manifests": hashes, "features": FEATURE_SPEC}, sort_keys=True).encode()).hexdigest()[:16]
    cache = args.cache_dir / f"features_{cache_key}.npz"
    cache_sidecar = cache.with_suffix(".json")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if cache.exists() != cache_sidecar.exists():
        raise ValueError(f"Incomplete feature cache pair: {cache} and {cache_sidecar}")
    if cache.is_file() and cache_sidecar.is_file():
        meta = json.loads(cache_sidecar.read_text(encoding="utf-8"))
        if meta.get("cache_key") != cache_key:
            raise ValueError("Feature cache metadata mismatch; refusing to reuse it")
        archive = np.load(cache)
        x_train_hog, x_train_color = archive["train_hog"], archive["train_color"]
        x_valid_hog, x_valid_color = archive["valid_hog"], archive["valid_color"]
        y_train = np.asarray([class_to_id[x] for x in archive["train_labels"].tolist()], dtype=np.int64)
        y_valid = np.asarray([class_to_id[x] for x in archive["valid_labels"].tolist()], dtype=np.int64)
        extraction = meta["extraction_seconds"]
        print(f"Loaded validated feature cache: {cache}")
    else:
        x_train_hog, x_train_color, train_labels, train_durations = extract_features(rows["train"], data_root, "train")
        x_valid_hog, x_valid_color, valid_labels, valid_durations = extract_features(rows["validation"], data_root, "validation")
        extraction = {
            "train_hog": train_durations["hog"], "train_color_histogram": train_durations["color_histogram"],
            "validation_hog": valid_durations["hog"], "validation_color_histogram": valid_durations["color_histogram"],
        }
        extraction["train_and_validation_total"] = sum(extraction.values())
        y_train = np.asarray([class_to_id[x] for x in train_labels], dtype=np.int64)
        y_valid = np.asarray([class_to_id[x] for x in valid_labels], dtype=np.int64)
        temp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temp, train_hog=x_train_hog, train_color=x_train_color, train_labels=train_labels,
                            valid_hog=x_valid_hog, valid_color=x_valid_color, valid_labels=valid_labels)
        temp.replace(cache)
        cache_sidecar.write_text(json.dumps({"cache_key": cache_key, "manifest_hashes": hashes,
                                             "feature_spec": FEATURE_SPEC, "extraction_seconds": extraction}, indent=2) + "\n", encoding="utf-8")
    result_candidates: dict[str, Any] = {}
    for name, spec in candidates(class_labels, y_train).items():
        train_x = np.hstack((x_train_hog, x_train_color)) if spec["features"] == "hog_hsv" else x_train_hog
        valid_x = np.hstack((x_valid_hog, x_valid_color)) if spec["features"] == "hog_hsv" else x_valid_hog
        fit = model_for(spec)
        start = time.perf_counter(); fit.fit(train_x, y_train); fit_seconds = time.perf_counter() - start
        start = time.perf_counter(); valid_pred = fit.predict(valid_x); eval_seconds = time.perf_counter() - start
        result_candidates[name] = {
            "spec": spec, "validation": metrics(y_valid, valid_pred, class_labels),
            "training_seconds": fit_seconds, "validation_evaluation_seconds": eval_seconds,
            "feature_extraction_seconds": extraction,
        }
        del fit
        print(f"{name}: val accuracy={result_candidates[name]['validation']['accuracy']:.4f}, macroF1={result_candidates[name]['validation']['macro_f1']:.4f}, potato_healthy_recall={result_candidates[name]['validation']['per_class']['Potato___healthy']['recall']:.4f}, fit={fit_seconds:.2f}s", flush=True)
    baseline = json.loads((ROOT / "ml/models/phase3/metrics.json").read_text(encoding="utf-8"))
    result = {
        "phase": "validation_only", "random_seed": RANDOM_SEED, "data_root": str(data_root),
        "split_hashes": hashes, "split_sizes": {k: len(v) for k, v in rows.items()},
        "class_id_to_label": {str(i): label for i, label in enumerate(class_labels)},
        "class_counts": {split: dict(Counter(row["label"] for row in content)) for split, content in rows.items()},
        "baseline_artifact_hashes_before": base_hashes, "baseline_test_metrics_reference": baseline["test"],
        "baseline_validation_metrics_reference": baseline["validation"], "feature_spec": FEATURE_SPEC,
        "candidates": result_candidates, "candidate_selection_rule": "Choose by validation macro-F1; consider Potato healthy recall and per-class trade-offs. No test metrics were used in candidate ranking.",
        "feature_cache": str(cache), "python": platform.python_version(), "scikit_learn": version("scikit-learn"),
    }
    (args.experiment_dir / "validation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_validation_report(args.experiment_dir / "report.md", result)
    print(f"Validation results saved to {args.experiment_dir / 'validation_results.json'}")


def write_validation_report(path: Path, result: dict[str, Any]) -> None:
    lines = ["# Phase 5 traditional ML experiments", "", "Validation-only candidate comparison; no test set was used for model selection.", "",
             "| Candidate | Features | Val accuracy | Macro precision | Macro recall | Macro F1 | Potato healthy recall | Fit sec | HOG extract sec | Color extract sec |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in result["candidates"].items():
        m = item["validation"]
        extraction = item["feature_extraction_seconds"]
        lines.append(f"| `{name}` | {item['spec']['features']} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {m['per_class']['Potato___healthy']['recall']:.4f} | {item['training_seconds']:.2f} | {extraction.get('train_hog', 0) + extraction.get('validation_hog', 0):.2f} | {extraction.get('train_color_histogram', 0) + extraction.get('validation_color_histogram', 0):.2f} |")
    for name, item in result["candidates"].items():
        lines.extend(["", f"## {name} validation details", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
        for label, score in item["validation"]["per_class"].items():
            lines.append(f"| `{label}` | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} | {score['support']} |")
        lines.extend(["", "Confusion matrix rows are actual classes; columns are predicted classes, in the order:", "", ", ".join(f"`{x}`" for x in item["validation"]["confusion_matrix"]["labels"]), "", "```json", json.dumps(item["validation"]["confusion_matrix"]["values"]), "```"])
    lines.extend(["", "## Baseline integrity", "", "Original Phase 3 model files and split manifests were hashed before experiment fitting. See `validation_results.json` for their hashes and split checksums.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def finalize(args: argparse.Namespace) -> None:
    results_path = args.experiment_dir / "validation_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Validation result file missing: {results_path}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if args.candidate not in result["candidates"]:
        raise ValueError(f"Unknown candidate {args.candidate}; available: {sorted(result['candidates'])}")
    if args.candidate == "baseline_hog_C1_balanced":
        raise ValueError("Baseline is already saved under Phase 3; use its held-out test results and do not replace it.")
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite versioned artifact directory: {args.artifact_dir}")
    data_root = args.data_root.resolve(); split_dir = args.split_dir.resolve()
    current_hashes = {name: digest(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    if current_hashes != result["split_hashes"]:
        raise ValueError("Split manifests changed since validation; refusing finalization")
    current_base_hashes = baseline_hashes()
    if current_base_hashes != result["baseline_artifact_hashes_before"]:
        raise ValueError("Phase 3 baseline changed since validation; refusing finalization")
    rows = load_rows(data_root, split_dir)
    labels = [result["class_id_to_label"][str(i)] for i in range(6)]
    class_to_id = {label: i for i, label in enumerate(labels)}
    cache_key = hashlib.sha256(json.dumps({"manifests": current_hashes, "features": FEATURE_SPEC}, sort_keys=True).encode()).hexdigest()[:16]
    cache = args.cache_dir / f"features_{cache_key}.npz"
    cache_sidecar = cache.with_suffix(".json")
    if not cache.is_file() or not cache_sidecar.is_file():
        raise FileNotFoundError(f"Validated training/validation feature cache missing: {cache}")
    cache_meta = json.loads(cache_sidecar.read_text(encoding="utf-8"))
    if cache_meta.get("cache_key") != cache_key or cache_meta.get("manifest_hashes") != current_hashes:
        raise ValueError("Feature cache does not match the exact split manifests")
    archive = np.load(cache)
    x_train_hog, x_train_color = archive["train_hog"], archive["train_color"]
    train_labels = archive["train_labels"].tolist()
    spec = result["candidates"][args.candidate]["spec"]
    train_x = np.hstack((x_train_hog, x_train_color)) if spec["features"] == "hog_hsv" else x_train_hog
    y_train = np.asarray([class_to_id[label] for label in train_labels], dtype=np.int64)
    model = model_for(spec)
    fit_start = time.perf_counter(); model.fit(train_x, y_train); fit_seconds = time.perf_counter() - fit_start
    x_test_hog, x_test_color, test_labels, test_feature_durations = extract_features(
        rows["test"], data_root, "test (frozen candidate)", include_color=spec["features"] == "hog_hsv"
    )
    test_extraction = sum(test_feature_durations.values())
    test_x = np.hstack((x_test_hog, x_test_color)) if spec["features"] == "hog_hsv" else x_test_hog
    y_test = np.asarray([class_to_id[label] for label in test_labels], dtype=np.int64)
    eval_start = time.perf_counter(); test_pred = model.predict(test_x); test_eval_seconds = time.perf_counter() - eval_start
    test_metrics = metrics(y_test, test_pred, labels)
    from onnxruntime import InferenceSession
    from skl2onnx import to_onnx
    onnx_model = to_onnx(model, train_x[:1].astype(np.float32), target_opset=17, options={"svc__zipmap": False})
    onnx_path_bytes = onnx_model.SerializeToString()
    session = InferenceSession(onnx_path_bytes, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    onnx_outputs = session.run(None, {input_info.name: test_x.astype(np.float32)})
    onnx_pred = np.asarray(onnx_outputs[0], dtype=np.int64).reshape(-1)
    parity = {"vector_count": len(test_pred), "matched_count": int(np.sum(onnx_pred == test_pred)),
              "mismatch_count": int(np.sum(onnx_pred != test_pred)),
              "python_class_ids": test_pred.tolist(), "onnx_class_ids": onnx_pred.tolist(),
              "class_id_to_label": {str(i): label for i, label in enumerate(labels)},
              "decision_score_max_abs_error": float(np.max(np.abs(np.asarray(onnx_outputs[1], dtype=np.float64) - model.decision_function(test_x))))}
    if parity["mismatch_count"]:
        raise RuntimeError(f"Python/ONNX class prediction parity failed: {parity['mismatch_count']} mismatches; refusing to save candidate")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifact_dir / "model.joblib")
    (args.artifact_dir / "model.onnx").write_bytes(onnx_path_bytes)
    feature_info = {"pipeline_features": spec["features"], "preprocessing": {
        "pillow_image_size": list(IMAGE_SIZE), "grayscale_contract_for_hog": True,
        "color_histogram": FEATURE_SPEC["color"] if spec["features"] == "hog_hsv" else None,
        "hog": FEATURE_SPEC["hog"] if "hog" in spec["features"] else None,
        "feature_count": int(train_x.shape[1]), "input_dtype": "float32"}}
    validation_metrics = result["candidates"][args.candidate]["validation"]
    final = {"candidate": args.candidate, "selection_basis": "Validation-only metrics; model/configuration frozen before test evaluation.",
             "model_artifact": "model.joblib", "onnx_artifact": "model.onnx", "class_id_to_label": {str(i): label for i, label in enumerate(labels)},
             "feature_config": feature_info, "training_parameters": spec, "random_seed": RANDOM_SEED,
             "split_hashes": current_hashes, "validation_metrics": validation_metrics, "test_metrics": test_metrics,
             "durations_seconds": {"fit": fit_seconds, "train_validation_feature_extraction": result["candidates"][args.candidate]["feature_extraction_seconds"],
                                   "test_feature_extraction": test_extraction, "test_evaluation": test_eval_seconds},
             "onnx": {"input_name": input_info.name, "input_type": input_info.type, "input_shape": input_info.shape,
                      "outputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in session.get_outputs()], "parity": parity},
             "browser_compatibility": "Not browser compatible until the TypeScript feature path reproduces this candidate's feature vector. Phase 3 remains the browser model.",
             "limitations": ["Evaluated on controlled PlantVillage images only; field performance is unknown."]}
    (args.artifact_dir / "metadata.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (args.artifact_dir / "metrics.json").write_text(json.dumps({"validation": validation_metrics, "test": test_metrics, "onnx_parity": parity}, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(results_path, args.artifact_dir / "candidate_validation_results.json")
    write_final_report(args.artifact_dir / "report.md", result, args.candidate, test_metrics, parity, final["durations_seconds"])
    print(json.dumps({"candidate": args.candidate, "test": test_metrics, "durations_seconds": final["durations_seconds"], "onnx_parity": parity, "artifact_dir": str(args.artifact_dir)}, indent=2))


def write_final_report(path: Path, validation_result: dict[str, Any], selected: str,
                       test_metrics: dict[str, Any], parity: dict[str, Any], durations: dict[str, Any]) -> None:
    base = validation_result["baseline_test_metrics_reference"]
    lines = ["# Phase 5 selected model report", "", f"Selected by validation: `{selected}`. Test data was accessed only after selection was frozen.", "",
             "## Validation candidate comparison", "",
             "| Candidate | Accuracy | Macro precision | Macro recall | Macro F1 | Potato healthy recall | SVM fit (s) | HOG extraction (s) | Color histogram extraction (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in validation_result["candidates"].items():
        m, ext = item["validation"], item["feature_extraction_seconds"]
        lines.append(f"| `{name}` | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {m['per_class']['Potato___healthy']['recall']:.4f} | {item['training_seconds']:.2f} | {ext['train_hog'] + ext['validation_hog']:.2f} | {ext['train_color_histogram'] + ext['validation_color_histogram']:.2f} |")
    lines.extend(["", "## Selected test comparison", "", "| Model | Accuracy | Macro precision | Macro recall | Macro F1 |", "|---|---:|---:|---:|---:|"])
    lines.append(f"| Phase 3 baseline | {base['accuracy']:.4f} | {base['macro_precision']:.4f} | {base['macro_recall']:.4f} | {base['macro_f1']:.4f} |")
    lines.append(f"| Phase 5 HOG + HSV SVM | {test_metrics['accuracy']:.4f} | {test_metrics['macro_precision']:.4f} | {test_metrics['macro_recall']:.4f} | {test_metrics['macro_f1']:.4f} |")
    lines.extend(["", "## Per-class held-out test results", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
    for label, score in test_metrics["per_class"].items():
        lines.append(f"| `{label}` | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} | {score['support']} |")
    labels = test_metrics["confusion_matrix"]["labels"]
    lines.extend(["", "## Held-out test confusion matrix", "", "Rows are actual classes and columns are predictions.", "", "| Actual \\ Predicted | " + " | ".join(f"`{label}`" for label in labels) + " |", "|---|" + "---:|" * len(labels)])
    for label, row in zip(labels, test_metrics["confusion_matrix"]["values"]):
        lines.append(f"| `{label}` | " + " | ".join(str(value) for value in row) + " |")
    lines.extend(["", "## Deployment checks", "", f"- ONNX class ID parity: {parity['matched_count']}/{parity['vector_count']}; decision-score max absolute error {parity['decision_score_max_abs_error']:.3g}.",
                  f"- Fit time: {durations['fit']:.2f}s; test feature extraction: {durations['test_feature_extraction']:.2f}s; test evaluation: {durations['test_evaluation']:.2f}s.",
                  "- This HOG + HSV feature vector requires a new TypeScript HSV histogram implementation before browser deployment. Phase 3 remains the browser-compatible model.",
                  "- PlantVillage backgrounds are controlled; farm-photo generalization has not been established.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "ml/data/plantvillage-source/raw/color")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "ml/data/splits")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "ml/.dev_artifacts/phase5/cache")
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "ml/.dev_artifacts/phase5/run1")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "ml/models/phase5/hog_svm_candidate_v1")
    parser.add_argument("--finalize-candidate", help="Freeze and test one candidate listed in validation_results.json")
    args = parser.parse_args()
    try:
        if args.finalize_candidate:
            args.candidate = args.finalize_candidate
            finalize(args)
        else:
            run_validation(args)
    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError) as error:
        print(f"Phase 5 experiment stopped: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
