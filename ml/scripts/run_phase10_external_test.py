"""One-time final test evaluation for the frozen Phase 9 candidate.

Run `--preflight` first. It records the model and test-manifest hashes without
opening test images. Then run `--evaluate-once`; that command creates a
permanent start marker before it reads test images and refuses a second run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/scripts"))
sys.path.insert(0, str(ROOT / "ml/src"))
from run_phase5_experiments import extract_features  # noqa: E402

MODEL_DIR = ROOT / "ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1"
OUT = ROOT / "ml/models/phase10_external_test"
SPLIT_DIR = ROOT / "ml/data/splits/phase8_external/phase8_external_20261019_v2"
IMAGE_ROOT = ROOT / "ml/data/external/irish_potato/images"
MODEL_PATH = MODEL_DIR / "selected_model.joblib"
TEST_MANIFEST = SPLIT_DIR / "test.csv"
CLASS_ORDER = [
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
]
SOURCE_TO_LABEL = {"earlyblt": CLASS_ORDER[0], "lateblt": CLASS_ORDER[1], "healthy": CLASS_ORDER[2]}
SOURCE_TO_META = {"earlyblt": "0", "healthy": "1", "lateblt": "2"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"Phase 10 output directory must be empty for freeze: {OUT}")
    phase9_meta = read_json(MODEL_DIR / "metadata.json")
    phase9_metrics = read_json(MODEL_DIR / "validation_metrics.json")
    labels = read_json(MODEL_DIR / "class_labels.json")["class_id_to_label"]
    config = read_json(MODEL_DIR / "feature_config.json")
    phase8_summary = read_json(SPLIT_DIR / "summary.json")
    if phase9_meta.get("selected_candidate") != "hog_hsv_svm" or phase9_metrics.get("selected_candidate") != "hog_hsv_svm":
        raise ValueError("Phase 9 metadata does not identify HOG + HSV + SVM as the selected candidate")
    if phase9_meta.get("external_test_accessed") is not False:
        raise ValueError("Phase 9 metadata does not confirm that external test data remained untouched")
    if config.get("combined_feature_count") != 8132 or config.get("combined_order") != "HOG 8100 then H(16 bins), S(8 bins), V(8 bins)":
        raise ValueError("Unexpected Phase 9 feature contract")
    if {int(k): v for k, v in labels.items()} != dict(enumerate(CLASS_ORDER)):
        raise ValueError("Unexpected Phase 9 class map")
    if not MODEL_PATH.is_file() or not TEST_MANIFEST.is_file():
        raise FileNotFoundError("Selected Phase 9 model or Phase 8 test manifest is missing")
    manifest_hash = sha256(TEST_MANIFEST)
    expected_hash = phase8_summary["manifest_sha256"]["test"]
    if manifest_hash != expected_hash:
        raise ValueError("External test manifest SHA-256 differs from Phase 8 integrity summary")
    if phase8_summary.get("fresh_test_evaluated") is not False:
        raise ValueError("Phase 8 summary does not mark its external test split untouched")
    model_hash = sha256(MODEL_PATH)
    selected_candidate_hash = sha256(MODEL_DIR / "candidate_hog_hsv_svm.joblib")
    freeze = {
        "freeze_time_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": phase9_meta["experiment_id"],
        "selected_candidate": phase9_meta["selected_candidate"],
        "selection_rule": phase9_meta["selection_rule"],
        "selected_model_path": str(MODEL_PATH),
        "selected_model_sha256": model_hash,
        "phase9_candidate_hog_hsv_sha256": selected_candidate_hash,
        "feature_config_sha256": sha256(MODEL_DIR / "feature_config.json"),
        "class_labels_sha256": sha256(MODEL_DIR / "class_labels.json"),
        "test_manifest_path": str(TEST_MANIFEST),
        "test_manifest_sha256": manifest_hash,
        "phase8_summary_manifest_sha256": expected_hash,
        "expected_test_path_count": sum(phase8_summary["split_class_counts"]["test"].values()),
        "expected_test_class_path_counts": phase8_summary["split_class_counts"]["test"],
        "test_manifest_integrity_status": phase8_summary.get("component_integrity"),
        "phase9_external_test_accessed": phase9_meta["external_test_accessed"],
        "test_images_opened_before_freeze": False,
        "model_loaded_before_freeze": False,
        "test_metrics_used_for_selection": False,
    }
    write_json(OUT / "freeze_record.json", freeze)
    print(json.dumps(freeze, indent=2))


def evaluate_once() -> None:
    freeze_path = OUT / "freeze_record.json"
    start_path = OUT / "evaluation_started.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Run --preflight to freeze the candidate before evaluation")
    if start_path.exists() or (OUT / "metrics.json").exists() or (OUT / "predictions.csv").exists():
        raise FileExistsError("Phase 10 evaluation already started or completed; refusing a second test run")
    freeze = read_json(freeze_path)
    if sha256(MODEL_PATH) != freeze["selected_model_sha256"]:
        raise ValueError("Frozen selected model hash changed after preflight")
    if sha256(TEST_MANIFEST) != freeze["test_manifest_sha256"]:
        raise ValueError("External test manifest changed after preflight")
    # From this point the one-time test evaluation is consumed, even if a later
    # operation fails. No result-based rerun/selection is allowed.
    write_json(start_path, {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model_sha256": freeze["selected_model_sha256"],
        "test_manifest_sha256": freeze["test_manifest_sha256"],
        "one_time_evaluation_consumed": True,
    })

    rows = read_csv(TEST_MANIFEST)
    if not rows:
        raise ValueError("External test manifest is empty")
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"Non-test row in test manifest: {row['path']}")
        if row["label"] not in CLASS_ORDER or SOURCE_TO_LABEL.get(row.get("source_label")) != row["label"]:
            raise ValueError(f"Invalid source/normalized label mapping: {row['path']}")
        if row.get("source_metadata_label") != SOURCE_TO_META.get(row["source_label"]):
            raise ValueError(f"External source metadata label mismatch: {row['path']}")
        if Path(row["path"]).parts[1] != row["source_label"]:
            raise ValueError(f"Source folder/label mismatch: {row['path']}")
        if not row.get("component_id") or not row.get("sha256"):
            raise ValueError(f"Missing exact hash or component ID: {row['path']}")

    # Confirm the reserved test has no known exact-hash or component overlap
    # with external train, external validation, or Phase 9's selected training sample.
    train_rows = read_csv(SPLIT_DIR / "train.csv")
    validation_rows = read_csv(SPLIT_DIR / "validation.csv")
    selected_train_rows = read_csv(MODEL_DIR / "selected_external_training_manifest.csv")
    phase9_combined_train = read_csv(MODEL_DIR / "combined_training_manifest.csv")
    test_hashes = {r["sha256"] for r in rows}
    test_components = {r["component_id"] for r in rows}
    train_hashes = {r["sha256"] for r in train_rows}
    validation_hashes = {r["sha256"] for r in validation_rows}
    train_components = {r["component_id"] for r in train_rows}
    validation_components = {r["component_id"] for r in validation_rows}
    selected_hashes = {r["sha256"] for r in selected_train_rows}
    phase9_train_hashes = {r["sha256"] for r in phase9_combined_train}
    selected_components = {r["component_id"] for r in selected_train_rows}
    if test_hashes & (train_hashes | validation_hashes | selected_hashes | phase9_train_hashes):
        raise ValueError("External test exact hash overlaps known train/validation data")
    if test_components & (train_components | validation_components | selected_components):
        raise ValueError("External test component overlaps known train/validation data")

    # One representative per exact hash, preserving manifest order, as used by
    # Phase 8 validation. Copies must have consistent labels and group IDs.
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    canonical = []
    duplicate_records = []
    for digest, members in groups.items():
        if len({r["label"] for r in members}) != 1 or len({r["component_id"] for r in members}) != 1:
            raise ValueError(f"Conflicting exact duplicate labels/components in held-out test: {digest}")
        canonical.append(members[0])
        for omitted in members[1:]:
            duplicate_records.append({"sha256":digest,"label":members[0]["label"],"retained_path":members[0]["path"],"omitted_path":omitted["path"],"component_id":members[0]["component_id"]})

    # Hash and fully decode each unique image after the freeze marker is saved.
    for i, row in enumerate(canonical, 1):
        image = (IMAGE_ROOT / Path(row["path"])).resolve()
        if not image.is_relative_to(IMAGE_ROOT.resolve()) or not image.is_file():
            raise FileNotFoundError(image)
        if sha256(image) != row["sha256"]:
            raise ValueError(f"Test image SHA-256 mismatch: {row['path']}")
        with Image.open(image) as im:
            im.load()
        if i % 1000 == 0:
            print(f"verified test images: {i}/{len(canonical)}", flush=True)

    model = joblib.load(MODEL_PATH)
    model_classes = model.named_steps["svc"].classes_.tolist()
    if model_classes != list(range(6)) or model.named_steps["scaler"].n_features_in_ != 8132:
        raise ValueError("Frozen model classes or scaler input feature count do not match Phase 9 config")
    start = time.perf_counter()
    X_hog, X_hsv, labels, extraction_times = extract_features(
        [{"path":r["path"],"label":r["label"]} for r in canonical], IMAGE_ROOT,
        "Phase 10 one-time external test", include_color=True,
    )
    X = np.hstack((X_hog, X_hsv)).astype(np.float32, copy=False)
    if X.shape != (len(canonical),8132) or X.dtype != np.float32:
        raise ValueError(f"Unexpected test feature matrix: {X.shape}, {X.dtype}")
    if labels.tolist() != [r["label"] for r in canonical]:
        raise ValueError("Test image feature labels became misaligned")
    extraction_elapsed = time.perf_counter()-start
    y_true = np.asarray([CLASS_ORDER.index(r["label"]) for r in canonical], dtype=np.int64)
    start = time.perf_counter()
    y_pred = model.predict(X).astype(np.int64)
    prediction_elapsed = time.perf_counter()-start

    p, r, f, support = precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2],average=None,zero_division=0)
    macro = precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2],average="macro",zero_division=0)
    cm = confusion_matrix(y_true,y_pred,labels=list(range(6)))
    correct = int(np.sum(y_true==y_pred))
    healthy_diseased = int(sum(t==2 and pred in {0,1,3,4,5} for t,pred in zip(y_true,y_pred)))
    diseased_healthy = int(sum(t in {0,1} and pred==2 for t,pred in zip(y_true,y_pred)))
    potato_as_tomato_disease = int(sum(t in {0,1,2} and pred in {3,4} for t,pred in zip(y_true,y_pred)))
    potato_as_any_tomato = int(sum(t in {0,1,2} and pred in {3,4,5} for t,pred in zip(y_true,y_pred)))
    metrics = {
        "experiment_id":"phase10_external_test_20261019_v1",
        "selected_candidate":"hog_hsv_svm",
        "frozen_model_path":str(MODEL_PATH),"frozen_model_sha256":freeze["selected_model_sha256"],
        "test_manifest_path":str(TEST_MANIFEST),"test_manifest_sha256":freeze["test_manifest_sha256"],
        "evaluated_split":"reserved external test; evaluated once after candidate freeze",
        "test_path_count":len(rows),"unique_hash_count_evaluated":len(canonical),"duplicate_paths_omitted":len(duplicate_records),
        "class_counts_by_manifest_path":dict(Counter(row["label"] for row in rows)),
        "class_counts_unique_hashes_evaluated":dict(Counter(row["label"] for row in canonical)),
        "accuracy":correct/len(canonical),"correct_count":correct,"denominator":len(canonical),
        "macro_precision":float(macro[0]),"macro_recall":float(macro[1]),"macro_f1":float(macro[2]),
        "macro_classes_with_support":[CLASS_ORDER[i] for i,n in enumerate(support) if n],
        "per_class":{CLASS_ORDER[i]:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"support":int(support[i])} for i in range(3)},
        "confusion_matrix":{"labels":CLASS_ORDER,"values":cm.astype(int).tolist()},
        "healthy_potato_predicted_as_diseased":healthy_diseased,
        "diseased_potato_predicted_as_healthy":diseased_healthy,
        "potato_predicted_as_tomato_disease":potato_as_tomato_disease,
        "potato_predicted_as_any_tomato_class":potato_as_any_tomato,
        "predicted_class_counts":dict(Counter(CLASS_ORDER[int(i)] for i in y_pred)),
        "feature_extraction_seconds":extraction_elapsed,"feature_extraction_breakdown_seconds":extraction_times,"prediction_seconds":prediction_elapsed,
        "test_component_overlap_external_train":len(test_components & {r["component_id"] for r in train_rows}),
        "test_component_overlap_external_validation":len(test_components & {r["component_id"] for r in validation_rows}),
        "test_component_overlap_phase9_selected_external_training":len(test_components & selected_components),
        "test_hash_overlap_known_train_validation":0,
        "test_hash_overlap_phase9_combined_training":0,
        "grouping_limitations":"External dataset metadata has no reliable plant/leaf/capture IDs. Phase 8 exact-hash and pHash constraints keep known links within one split, but undetected near-duplicate or related captures cannot be ruled out. Cross-source near-duplicate links are also not established.",
        "geographic_limitations":"Images were collected in Tanzania, predominantly from the Shangi variety and Samsung Galaxy A03 Core phones. This does not establish reliability in Kerala, across potato varieties, or across phone cameras.",
        "probability_note":"SVM outputs are not calibrated probabilities.",
    }
    prediction_rows = []
    for row, truth, pred in zip(canonical,y_true,y_pred):
        prediction_rows.append({"path":row["path"],"source_label":row["source_label"],"source_metadata_label":row["source_metadata_label"],"true_label":CLASS_ORDER[int(truth)],"predicted_label":CLASS_ORDER[int(pred)],"correct":str(int(truth)==int(pred)).lower(),"sha256":row["sha256"],"component_id":row["component_id"]})
    write_json(OUT/"metrics.json",metrics)
    write_csv(OUT/"predictions.csv",prediction_rows,["path","source_label","source_metadata_label","true_label","predicted_label","correct","sha256","component_id"])
    write_csv(OUT/"exact_duplicate_exclusions.csv",duplicate_records,["sha256","label","retained_path","omitted_path","component_id"])
    matrix_rows=[]
    for i,label in enumerate(CLASS_ORDER):
        matrix_rows.append({"true_label":label,**{CLASS_ORDER[j]:int(cm[i,j]) for j in range(6)}})
    write_csv(OUT/"confusion_matrix.csv",matrix_rows,["true_label",*CLASS_ORDER])
    (OUT/"report.md").write_text(make_report(metrics),encoding="utf-8")
    write_json(OUT/"evaluation_complete.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),"test_evaluation_count":1,"test_images_evaluated":len(canonical),"model_sha256":freeze["selected_model_sha256"],"manifest_sha256":freeze["test_manifest_sha256"]})
    print(json.dumps(metrics,indent=2),flush=True)


def make_report(m: dict) -> str:
    lines=["# Phase 10: final held-out external test evaluation","",f"Frozen candidate: `{m['selected_candidate']}`",f"Model SHA-256: `{m['frozen_model_sha256']}`",f"Test manifest SHA-256: `{m['test_manifest_sha256']}`","","## Test metrics","",f"- Image paths in manifest: {m['test_path_count']}",f"- Unique SHA-256 images evaluated: {m['unique_hash_count_evaluated']}",f"- Exact duplicate paths omitted: {m['duplicate_paths_omitted']}",f"- Accuracy: {m['accuracy']:.4f} ({m['correct_count']}/{m['denominator']})",f"- Macro precision: {m['macro_precision']:.4f}",f"- Macro recall: {m['macro_recall']:.4f}",f"- Macro F1: {m['macro_f1']:.4f}","","### Class metrics","","| Potato class | Support | Precision | Recall | F1 |","|---|---:|---:|---:|---:|"]
    for label,score in m["per_class"].items():
        lines.append(f"| {label} | {score['support']} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} |")
    lines += ["","### Confusion matrix","","Rows are true labels and columns are predicted labels in this order: "+", ".join(m["confusion_matrix"]["labels"])+".","","```text"]
    lines += [str(row) for row in m["confusion_matrix"]["values"]]
    lines += ["```","",f"- Healthy potato predicted diseased: {m['healthy_potato_predicted_as_diseased']}",f"- Diseased potato predicted healthy: {m['diseased_potato_predicted_as_healthy']}",f"- Potato images predicted as tomato disease: {m['potato_predicted_as_tomato_disease']}",f"- Potato images predicted as any tomato class: {m['potato_predicted_as_any_tomato_class']}","","## Comparison with external validation","", "Validation is development data because it informed candidate selection. The figures below are not an independent test of generalization.","", "| Metric | External validation | Held-out external test |", "|---|---:|---:|"]
    phase9=read_json(MODEL_DIR/"validation_metrics.json")["candidate_results"]["hog_hsv_svm"]["external"]
    for key,label in (("accuracy","Accuracy"),("macro_precision","Macro precision"),("macro_recall","Macro recall"),("macro_f1","Macro F1")):
        lines.append(f"| {label} | {phase9[key]:.4f} | {m[key]:.4f} |")
    lines += ["","## Limitations","", "The source images were collected in Tanzania, mainly on Samsung Galaxy A03 Core smartphones and predominantly the Shangi variety. These results do not establish reliability in Kerala or for other varieties and phone cameras. The source lacks reliable leaf/plant/capture identifiers; exact-hash and available perceptual-hash grouping reduce known leakage, but undetected related captures remain possible. The external test was evaluated once and was not used to tune or choose the model.","","The selected model uses HSV features. Browser deployment still requires the matching TypeScript HSV extractor and browser parity checks. One held-out source-specific result is not enough to call the model production-ready.",""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight",action="store_true",help="Freeze model and verify test manifest hash without opening test images")
    group.add_argument("--evaluate-once",action="store_true",help="Run the one-time held-out external test evaluation")
    args=parser.parse_args()
    preflight() if args.preflight else evaluate_once()
