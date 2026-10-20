"""Derive Phase 15 metrics and component-bootstrap intervals from saved predictions only."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ml/models/phase10_external_test"
OUT = ROOT / "ml/models/phase15_external_test"
LABELS = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}
SUPPORTED = [0, 1, 2]
BOOTSTRAP_REPLICATES = 2000
SEED = 20261019


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.bincount(y_true * len(LABELS) + y_pred, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.int64)
    predicted = cm.sum(axis=0).astype(np.int64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    present = [i for i in SUPPORTED if support[i] > 0]
    accuracy = float(tp.sum() / support.sum())
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall[present])),
        "supported_class_macro_precision": float(np.mean(precision[present])),
        "supported_class_macro_recall": float(np.mean(recall[present])),
        "supported_class_macro_f1": float(np.mean(f1[present])),
        "six_label_macro_precision_zero_division_0": float(np.mean(precision)),
        "six_label_macro_recall_zero_division_0": float(np.mean(recall)),
        "six_label_macro_f1_zero_division_0": float(np.mean(f1)),
        "per_class": {LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(len(LABELS))},
        "confusion_matrix_labels_true_rows_predicted_columns": cm.astype(int).tolist(),
    }


def main() -> None:
    raw = read_json(SOURCE / "metrics.json")
    with (SOURCE / "predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != raw["unique_hash_count_evaluated"]:
        raise ValueError("Saved prediction rows do not match the one-time evaluation count")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Saved predictions contain duplicate hashes; expected one representative per exact hash")
    if any(r["true_label"] not in LABEL_INDEX or r["predicted_label"] not in LABEL_INDEX or not r["component_id"] for r in rows):
        raise ValueError("Saved prediction labels or component IDs are invalid")
    y_true = np.asarray([LABEL_INDEX[r["true_label"]] for r in rows], dtype=np.int64)
    y_pred = np.asarray([LABEL_INDEX[r["predicted_label"]] for r in rows], dtype=np.int64)
    if set(y_true.tolist()) != set(SUPPORTED):
        raise ValueError("Expected exactly the three potato classes to have support")
    if int(np.sum((y_true == 2) & np.isin(y_pred, [0, 1, 3, 4]))) != 220:
        raise ValueError("Healthy-potato disease count did not reproduce the correctly defined disease-label set")
    if int(np.sum((y_true == 2) & (y_pred == 5))) != 1:
        raise ValueError("Expected one potato-healthy example predicted as tomato healthy; check the corrected disease count")
    point = summarize(y_true, y_pred)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["component_id"]].append(i)
    # Components are single-label in this test split; keep every component intact.
    components_by_class: dict[int, list[list[int]]] = defaultdict(list)
    for members in groups.values():
        labels = {int(y_true[i]) for i in members}
        if len(labels) != 1:
            raise ValueError("A connected component contains multiple true classes")
        components_by_class[labels.pop()].append(members)
    rng = np.random.default_rng(SEED)
    samples = {key: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for key in ["accuracy", "balanced_accuracy", "supported_class_macro_precision", "supported_class_macro_recall", "supported_class_macro_f1"]}
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_indices = []
        for class_id in SUPPORTED:
            components = components_by_class[class_id]
            chosen = rng.integers(0, len(components), size=len(components))
            for chosen_index in chosen:
                selected_indices.extend(components[int(chosen_index)])
        result = summarize(y_true[selected_indices], y_pred[selected_indices])
        for key in samples:
            samples[key][replicate] = result[key]
    intervals = {key: {"lower_95_percentile": float(np.quantile(values, 0.025)), "upper_95_percentile": float(np.quantile(values, 0.975))} for key, values in samples.items()}
    output = {
        "experiment_id": "phase15_frozen_external_test_20261019_v1",
        "source_evaluation": str(SOURCE / "metrics.json"),
        "selected_model_joblib_sha256": raw["frozen_model_sha256"],
        "selected_model_onnx_sha256": "1d741fbe7ee6d9a81f0bba78e7a2e84e63f4636000bda7ff8bd3223ffa28f212",
        "test_manifest_sha256": raw["test_manifest_sha256"],
        "test_manifest_paths": raw["test_path_count"],
        "unique_exact_hash_images_evaluated": raw["unique_hash_count_evaluated"],
        "exact_duplicate_paths_omitted": raw["duplicate_paths_omitted"],
        "known_component_count": len(groups),
        "component_counts_by_class": {LABELS[k]: len(v) for k, v in components_by_class.items()},
        "class_order": LABELS,
        "metrics": point,
        "confidence_intervals": {
            "method": "Stratified cluster bootstrap over the saved connected components; sample components with replacement within each of the three supported potato classes, retaining all distinct-hash representatives in each selected component. 2,000 replicates; NumPy default_rng seed 20261019; percentile 95% intervals.",
            "limitations": "Intervals reflect only components found by the Phase 8 exact-hash and perceptual-hash grouping; absent leaf/capture IDs and missed near-duplicates are not represented.",
            "values": intervals,
        },
        "healthy_potato_predicted_as_diseased_corrected": int(np.sum((y_true == 2) & np.isin(y_pred, [0, 1, 3, 4]))),
        "healthy_potato_predicted_as_tomato_healthy_not_counted_as_diseased": int(np.sum((y_true == 2) & (y_pred == 5))),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supplementary_metrics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
