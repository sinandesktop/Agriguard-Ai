"""Train controlled HOG/SVM candidates with a component-safe external sample.

This experiment reads only Phase 6.6 and Phase 8 v2 train/validation manifests.
It intentionally never opens either dataset's test manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/scripts"))
sys.path.insert(0, str(ROOT / "ml/src"))
import run_phase5_experiments as phase5  # noqa: E402

PV_ROOT = ROOT / "ml/data/plantvillage-source/raw/color"
EXT_ROOT = ROOT / "ml/data/external/irish_potato/images"
PV_SPLITS = ROOT / "ml/data/splits/phase6_6"
EXT_SPLITS = ROOT / "ml/data/splits/phase8_external/phase8_external_20261019_v2"
EXPERIMENT_ID = "phase9_domain_adaptation_20261019_v1"
SEED = 20261019
TARGET_PER_EXTERNAL_CLASS = 1000
CLASSES = [
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
]
EXT_SOURCE_LABEL = {
    "Potato___Early_blight": "earlyblt",
    "Potato___Late_blight": "lateblt",
    "Potato___healthy": "healthy",
}
EXT_METADATA_ID = {"earlyblt": "0", "healthy": "1", "lateblt": "2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_digest(path: Path) -> str:
    return sha256(path)


def validate_rows(rows: list[dict[str, str]], root: Path, source: str, split: str) -> dict:
    if not rows:
        raise ValueError(f"Empty {source}/{split} manifest")
    needed = {"path", "label", "sha256", "component_id"}
    if not needed.issubset(rows[0]):
        raise ValueError(f"{source}/{split} is missing required columns: {needed - set(rows[0])}")
    hash_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    labels = Counter()
    for i, row in enumerate(rows, 1):
        if not row["component_id"]:
            raise ValueError(f"Missing component ID in {source}/{split}: {row['path']}")
        if row["label"] not in CLASSES:
            raise ValueError(f"Unexpected normalized label in {source}/{split}: {row['label']}")
        path = (root / Path(row["path"])).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing/unsafe {source}/{split} image: {row['path']}")
        actual_hash = sha256(path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {source}/{split}: {row['path']}")
        hash_members[actual_hash].append(row)
        with Image.open(path) as im:
            im.load()
            if im.width <= 0 or im.height <= 0:
                raise ValueError(f"Invalid image dimensions in {source}/{split}: {row['path']}")
        if source == "external":
            expected = EXT_SOURCE_LABEL[row["label"]]
            if row.get("source_label") != expected:
                raise ValueError(f"External source label mismatch: {row['path']}")
            if row.get("source_metadata_label") != EXT_METADATA_ID[expected]:
                raise ValueError(f"External metadata label mismatch: {row['path']}")
            if Path(row["path"]).parts[1] != expected:
                raise ValueError(f"External folder/label mismatch: {row['path']}")
            if row.get("split") != split:
                raise ValueError(f"External split marker mismatch: {row['path']}")
        else:
            if row["path"].split("/", 1)[0] != row["label"]:
                raise ValueError(f"PlantVillage folder/label mismatch: {row['path']}")
            if row.get("split_assignment") != split:
                raise ValueError(f"PlantVillage split marker mismatch: {row['path']}")
        labels[row["label"]] += 1
        if i % 5000 == 0:
            print(f"verified {source}/{split}: {i}/{len(rows)}", flush=True)
    duplicate_groups = 0
    extra_paths = 0
    if source == "plantvillage":
        for digest, members in hash_members.items():
            if len(members) == 1:
                if members[0].get("canonical_hash_member", "true").lower() != "true":
                    raise ValueError(f"Singleton hash incorrectly marked noncanonical: {members[0]['path']}")
                continue
            duplicate_groups += 1
            extra_paths += len(members) - 1
            if len({r["label"] for r in members}) != 1 or len({r["component_id"] for r in members}) != 1:
                raise ValueError(f"PlantVillage exact-duplicate copies disagree on label/component: {digest}")
            canon = [r for r in members if r.get("canonical_hash_member", "").lower() == "true"]
            if len(canon) != 1 or any(r.get("canonical_hash_member", "").lower() != "false" for r in members if r not in canon):
                raise ValueError(f"PlantVillage exact-duplicate canonical flags invalid: {digest}")
    else:
        for digest, members in hash_members.items():
            if len(members) == 1:
                continue
            duplicate_groups += 1
            extra_paths += len(members) - 1
            if len({r["label"] for r in members}) != 1 or len({r["component_id"] for r in members}) != 1:
                raise ValueError(f"External exact-duplicate copies disagree on label/component: {digest}")
    return {"manifest_rows": len(rows), "labels_by_path": dict(labels), "distinct_hashes": len(hash_members), "exact_duplicate_groups": duplicate_groups, "extra_duplicate_paths": extra_paths}


def choose_external_components(rows: list[dict[str, str]], forbidden_validation_hashes: set[str], forbidden_train_hashes: set[str]) -> tuple[list[dict[str, str]], list[str], dict]:
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    retained_by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        retained_by_hash.setdefault(row["sha256"], row)
    for row in retained_by_hash.values():
        by_component[row["component_id"]].append(row)
    excluded_validation = {
        component for component, members in by_component.items()
        if any(row["sha256"] in forbidden_validation_hashes for row in members)
    }
    excluded_duplicate_train = {
        component for component, members in by_component.items()
        if component not in excluded_validation and any(row["sha256"] in forbidden_train_hashes for row in members)
    }
    excluded_components = excluded_validation | excluded_duplicate_train
    allowed = {c: members for c, members in by_component.items() if c not in excluded_components}
    rng = random.Random(SEED)
    shuffled_by_label = {}
    for label in EXT_SOURCE_LABEL:
        choices = [c for c, members in allowed.items() if any(r["label"] == label for r in members)]
        rng.shuffle(choices)
        shuffled_by_label[label] = choices

    selected_components: set[str] = set()
    counts = Counter()
    for label in ("Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"):
        for component in shuffled_by_label[label]:
            if counts[label] >= TARGET_PER_EXTERNAL_CLASS:
                break
            if component in selected_components:
                continue
            selected_components.add(component)
            for row in allowed[component]:
                counts[row["label"]] += 1
        if counts[label] < TARGET_PER_EXTERNAL_CLASS:
            raise ValueError(f"Could not reach target sample for {label}: {counts[label]}")
    chosen = [row for component in sorted(selected_components) for row in allowed[component]]
    chosen.sort(key=lambda r: (r["label"], r["component_id"], r["path"]))
    if any(r["sha256"] in (forbidden_validation_hashes | forbidden_train_hashes) for r in chosen):
        raise AssertionError("Selected external training hash overlaps validation")
    summary = {
        "target_distinct_images_per_class": TARGET_PER_EXTERNAL_CLASS,
        "selected_counts_by_class": dict(Counter(r["label"] for r in chosen)),
        "selected_component_count": len(selected_components),
        "excluded_components_due_to_plantvillage_validation_hash_overlap": len(excluded_validation),
        "excluded_rows_due_to_plantvillage_validation_hash_overlap": sum(len(by_component[c]) for c in excluded_validation),
        "excluded_components_due_to_plantvillage_training_exact_duplicate_overlap": len(excluded_duplicate_train),
        "excluded_rows_due_to_plantvillage_training_exact_duplicate_overlap": sum(len(by_component[c]) for c in excluded_duplicate_train),
        "largest_selected_component": max(Counter(r["component_id"] for r in chosen).values()),
        "seed": SEED,
    }
    return chosen, sorted(selected_components), summary


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label_order: list[str]) -> dict:
    ids = np.arange(len(label_order))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=ids, average=None, zero_division=0)
    present = np.flatnonzero(support > 0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "macro_classes_with_support": [label_order[i] for i in present],
        "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, label in enumerate(label_order)},
        "confusion_matrix": {"labels": label_order, "values": cm.astype(int).tolist()},
        "example_count": int(len(y_true)),
        "predicted_counts": {label_order[i]: int(n) for i, n in enumerate(np.bincount(y_pred, minlength=len(label_order)))},
        "potato_true_predicted_as_tomato": int(sum(label_order[int(t)].startswith("Potato___") and label_order[int(p_)].startswith("Tomato___") for t, p_ in zip(y_true, y_pred))),
    }


def feature_matrix(rows: list[dict[str, str]], root: Path, tag: str, use_color: bool) -> tuple[np.ndarray, np.ndarray]:
    hog, hsv, labels, durations = phase5.extract_features(rows, root, tag, include_color=use_color)
    if labels.tolist() != [r["label"] for r in rows]:
        raise ValueError(f"Feature extractor changed label order for {tag}")
    if hog.shape != (len(rows), 8100) or hog.dtype != np.float32:
        raise ValueError(f"Unexpected HOG feature matrix for {tag}: {hog.shape}, {hog.dtype}")
    if use_color:
        if hsv.shape != (len(rows), 32) or hsv.dtype != np.float32:
            raise ValueError(f"Unexpected HSV matrix for {tag}: {hsv.shape}, {hsv.dtype}")
        X = np.hstack((hog, hsv)).astype(np.float32, copy=False)
    else:
        X = hog
    print(f"feature extraction {tag}: {len(rows)} rows, {X.shape[1]} features, {sum(durations.values()):.1f}s", flush=True)
    return X, durations


def mixed_source_feature_matrix(rows: list[dict[str, str]], tag: str) -> tuple[np.ndarray, dict[str, float]]:
    """Extract rows from their actual source roots, restoring manifest order."""
    groups = {
        "PlantVillage": (PV_ROOT, [i for i, r in enumerate(rows) if r["source_dataset"] == "PlantVillage"]),
        "external": (EXT_ROOT, [i for i, r in enumerate(rows) if r["source_dataset"] != "PlantVillage"]),
    }
    hog_out = np.empty((len(rows), 8100), dtype=np.float32)
    hsv_out = np.empty((len(rows), 32), dtype=np.float32)
    durations = {"hog": 0.0, "color_histogram": 0.0}
    for source, (root, indices) in groups.items():
        if not indices:
            continue
        subset = [rows[i] for i in indices]
        hog, hsv, got_labels, took = phase5.extract_features(
            [{"path": r["path"], "label": r["label"]} for r in subset], root,
            f"{tag} {source}", include_color=True,
        )
        if got_labels.tolist() != [r["label"] for r in subset]:
            raise ValueError(f"Extractor label order mismatch for {source}")
        hog_out[indices] = hog
        hsv_out[indices] = hsv
        for key, value in took.items():
            durations[key] += value
    combined = np.hstack((hog_out, hsv_out)).astype(np.float32, copy=False)
    if combined.shape != (len(rows), 8132):
        raise ValueError(f"Unexpected mixed-source feature shape: {combined.shape}")
    print(f"feature extraction {tag}: {len(rows)} mixed-source rows, 8132 features", flush=True)
    return combined, durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Phase 9 output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifests = {
        "pv_train": PV_SPLITS / "train.csv", "pv_validation": PV_SPLITS / "validation.csv",
        "external_train": EXT_SPLITS / "train.csv", "external_validation": EXT_SPLITS / "validation.csv",
    }
    for name, path in manifests.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    pv_train = read_csv(manifests["pv_train"])
    pv_val = read_csv(manifests["pv_validation"])
    ext_train = read_csv(manifests["external_train"])
    ext_val = read_csv(manifests["external_validation"])
    # Deliberately no reference to either test manifest/path in this script.

    quarantine_file = EXT_SPLITS / "quarantined_components.csv"
    unreadable_file = EXT_SPLITS / "unreadable_images.csv"
    quarantined_paths = {r["path"] for r in read_csv(quarantine_file)}
    unreadable_paths = {r["path"] for r in read_csv(unreadable_file)}
    if not ext_train or not ext_val:
        raise ValueError("External train/validation manifests must be nonempty")
    bad_manifest_paths = {r["path"] for r in ext_train + ext_val} & (quarantined_paths | unreadable_paths)
    if bad_manifest_paths:
        raise ValueError(f"External manifest includes quarantined/unreadable paths: {sorted(bad_manifest_paths)[:5]}")

    t_audit = time.perf_counter()
    audit = {
        "plantvillage_train": validate_rows(pv_train, PV_ROOT, "plantvillage", "train"),
        "plantvillage_validation": validate_rows(pv_val, PV_ROOT, "plantvillage", "validation"),
        "external_train": validate_rows(ext_train, EXT_ROOT, "external", "train"),
        "external_validation": validate_rows(ext_val, EXT_ROOT, "external", "validation"),
    }
    pv_train_manifest = pv_train
    pv_val_manifest = pv_val
    ext_train_manifest = ext_train
    ext_val_manifest = ext_val
    pv_train = [r for r in pv_train_manifest if r.get("canonical_hash_member", "true").lower() == "true"]
    pv_val = [r for r in pv_val_manifest if r.get("canonical_hash_member", "true").lower() == "true"]
    def first_per_hash(rows):
        retained = {}
        for item in rows:
            retained.setdefault(item["sha256"], item)
        return list(retained.values()), retained
    ext_train, ext_train_canonical = first_per_hash(ext_train_manifest)
    ext_val, ext_val_canonical = first_per_hash(ext_val_manifest)
    duplicate_handling_rows = []
    for split_name, source_rows in (("plantvillage_train", pv_train_manifest), ("plantvillage_validation", pv_val_manifest)):
        canonical_by_hash = {r["sha256"]: r["path"] for r in source_rows if r.get("canonical_hash_member", "true").lower() == "true"}
        for r in source_rows:
            if r.get("canonical_hash_member", "true").lower() == "false":
                duplicate_handling_rows.append({"split": split_name, "sha256": r["sha256"], "label": r["label"], "retained_path": canonical_by_hash[r["sha256"]], "omitted_path": r["path"], "component_id": r["component_id"]})
    for split_name, source_rows, retained in (("external_train", ext_train_manifest, ext_train_canonical), ("external_validation", ext_val_manifest, ext_val_canonical)):
        first_path = set()
        for r in source_rows:
            if r["sha256"] in first_path:
                duplicate_handling_rows.append({"split": split_name, "sha256": r["sha256"], "label": r["label"], "retained_path": retained[r["sha256"]]["path"], "omitted_path": r["path"], "component_id": r["component_id"]})
            else:
                first_path.add(r["sha256"])
    pv_train_hashes = {r["sha256"] for r in pv_train}
    pv_val_hashes = {r["sha256"] for r in pv_val}
    ext_train_hashes = {r["sha256"] for r in ext_train}
    ext_val_hashes = {r["sha256"] for r in ext_val}
    if pv_train_hashes & pv_val_hashes or ext_train_hashes & ext_val_hashes:
        raise ValueError("An existing source has exact-hash train/validation leakage")
    pv_comp = defaultdict(list)
    for row in pv_train:
        pv_comp[row["component_id"]].append(row)
    pv_components_blocked_by_ext_val = {
        c for c, rs in pv_comp.items() if any(r["sha256"] in ext_val_hashes for r in rs)
    }
    pv_fit = [r for r in pv_train if r["component_id"] not in pv_components_blocked_by_ext_val]
    if not pv_fit:
        raise ValueError("No PlantVillage training data remains after component-level cross-source overlap exclusion")
    # No training hash may appear in either validation source. Block complete external components.
    forbidden_validation = pv_val_hashes
    forbidden_pv_train = {r["sha256"] for r in pv_fit}
    external_sample, selected_components, sample_info = choose_external_components(ext_train, forbidden_validation, forbidden_pv_train)
    combined_hashes = [r["sha256"] for r in pv_fit] + [r["sha256"] for r in external_sample]
    all_validation_hashes = pv_val_hashes | ext_val_hashes
    if set(combined_hashes) & all_validation_hashes:
        raise ValueError("Combined training has exact-hash overlap with validation")
    if len(combined_hashes) != len(set(combined_hashes)):
        raise ValueError("Exact hash duplicated across combined training sources")
    for r in external_sample:
        if r["component_id"] not in selected_components:
            raise AssertionError("External component was split during sampling")

    pv_train_used = [dict(r, source_dataset="PlantVillage", original_label=r["label"], source_label="", source_metadata_label="") for r in pv_fit]
    ext_train_used = [dict(r, source_dataset="Project-AgML/potato_leaf_blight_classification", original_label=r["source_label"]) for r in external_sample]
    train_rows = pv_train_used + ext_train_used
    train_rows.sort(key=lambda r: (r["label"], r["source_dataset"], r["path"]))
    if set(r["label"] for r in train_rows) != set(CLASSES):
        raise ValueError("Combined train pool does not have all six classes")

    # Persist the exact samples and audit record before feature generation/fitting.
    sample_fields = ["source_dataset", "path", "label", "original_label", "source_label", "source_metadata_label", "sha256", "component_id", "grouping_evidence", "source_root", "split", "width", "height", "format"]
    write_csv(out / "combined_training_manifest.csv", train_rows, sample_fields)
    write_csv(out / "selected_external_training_manifest.csv", ext_train_used, sample_fields)
    write_csv(out / "exact_duplicate_handling.csv", duplicate_handling_rows, ["split","sha256","label","retained_path","omitted_path","component_id"])
    write_csv(out / "excluded_plantvillage_components.csv", [r for c in sorted(pv_components_blocked_by_ext_val) for r in pv_comp[c]], list(pv_train[0]))
    for split_name, rows in (("plantvillage_validation", pv_val), ("external_validation", ext_val)):
        for r in rows:
            if r["sha256"] in set(combined_hashes):
                raise ValueError(f"Training/validation overlap in {split_name}: {r['path']}")

    labels = CLASSES
    to_id = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([to_id[r["label"]] for r in train_rows], dtype=np.int64)
    y_pv_val = np.asarray([to_id[r["label"]] for r in pv_val], dtype=np.int64)
    y_ext_val = np.asarray([to_id[r["label"]] for r in ext_val], dtype=np.int64)
    print(f"Verified manifests and filtered cross-source overlap in {time.perf_counter()-t_audit:.1f}s", flush=True)
    print(f"Combined fit rows: {len(train_rows)} ({dict(Counter((r['source_dataset'],r['label']) for r in train_rows))})", flush=True)

    train_features, extraction_train = mixed_source_feature_matrix(train_rows, "phase9 combined train")
    pv_val_features, extraction_pv_val = feature_matrix(pv_val, PV_ROOT, "phase9 PlantVillage validation", True)
    ext_val_features, extraction_ext_val = feature_matrix(ext_val, EXT_ROOT, "phase9 external validation", True)
    # HOG is the leading 8,100 columns of the combined extraction, so both
    # candidates use identical precomputed HOG values without a second pass.
    hog_train = train_features[:, :8100]
    hog_pv = pv_val_features[:, :8100]
    hog_ext = ext_val_features[:, :8100]

    write_json(out / "data_audit.json", {
        "experiment_id": EXPERIMENT_ID, "audit_seed": SEED, "manifest_audit": audit,
        "manifest_sha256": {key: manifest_digest(path) for key, path in manifests.items()},
        "external_sampling": sample_info,
        "external_manifest_quarantine_check": {
            "conflicted_or_metadata_mismatch_paths_in_train_or_validation": 0,
            "unreadable_paths_in_train_or_validation": 0,
            "source_audit_quarantined_path_count": len(quarantined_paths),
            "source_audit_unreadable_path_count": len(unreadable_paths),
        },
        "cross_source_hash_exclusions": {
            "pv_train_components_removed_due_to_external_validation_hash": len(pv_components_blocked_by_ext_val),
            "pv_train_rows_removed_due_to_external_validation_hash": len(pv_train)-len(pv_fit),
            "external_components_removed_due_to_plantvillage_validation_hash": sample_info["excluded_components_due_to_plantvillage_validation_hash_overlap"],
            "external_components_removed_due_to_plantvillage_train_exact_duplicate": sample_info["excluded_components_due_to_plantvillage_training_exact_duplicate_overlap"],
            "all_combined_train_hashes_disjoint_from_both_validation_sets": True,
            "train_hash_duplicates_across_sources": 0,
        },
        "combined_training_counts_by_source_and_class": {source: dict(Counter(r["label"] for r in train_rows if r["source_dataset"] == source)) for source in sorted({r["source_dataset"] for r in train_rows})},
        "combined_validation_counts": {"PlantVillage": dict(Counter(r["label"] for r in pv_val)), "external": dict(Counter(r["label"] for r in ext_val))},
        "grouping_note": "External sample selects complete existing connected components. External grouping metadata contains no capture/leaf/plant IDs; pHash-connected components are conservative constraints, not confirmed same-leaf groups. PlantVillage component IDs are preserved.",
        "audit_seconds": time.perf_counter()-t_audit,
    })

    candidate_results = {}
    fitted_models = {}
    feature_sets = {
        "hog_svm": (hog_train, hog_pv, hog_ext, 8100),
        "hog_hsv_svm": (train_features, pv_val_features, ext_val_features, 8132),
    }
    total_train_seconds = {}
    prediction_seconds = {}
    for name, (xtr, xpv, xext, n_features) in feature_sets.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=False, random_state=SEED, cache_size=512)),
        ])
        start = time.perf_counter()
        pipeline.fit(xtr, y_train)
        fit_seconds = time.perf_counter()-start
        total_train_seconds[name] = fit_seconds
        fitted_models[name] = pipeline
        model_predictions = {}
        candidate_results[name] = {"feature_count": n_features, "svc_parameters": {"kernel":"rbf","C":1.0,"gamma":"scale","class_weight":"balanced","probability":False,"random_state":SEED,"cache_size_mb":512}, "training_seconds": fit_seconds}
        for source, X, y, rows in (("plantvillage", xpv, y_pv_val, pv_val), ("external", xext, y_ext_val, ext_val)):
            start = time.perf_counter()
            pred = pipeline.predict(X).astype(np.int64)
            elapsed = time.perf_counter()-start
            prediction_seconds[f"{name}_{source}"] = elapsed
            model_predictions[source] = pred
            candidate_results[name][source] = metrics(y, pred, labels)
            candidate_results[name][source]["class_counts"] = dict(Counter(r["label"] for r in rows))
            candidate_results[name][source]["source_counts"] = dict(Counter(r.get("source_dataset", r.get("source_label", "PlantVillage")) for r in rows))
            pred_rows = []
            for row, truth, predicted in zip(rows, y, pred):
                pred_rows.append({"path": row["path"], "source_dataset": "PlantVillage" if source == "plantvillage" else row["source_repository"], "source_label": row.get("source_label", row["label"]), "true_label": labels[int(truth)], "predicted_label": labels[int(predicted)], "correct": str(int(truth)==int(predicted)).lower(), "sha256": row["sha256"], "component_id": row["component_id"]})
            write_csv(out / f"{name}_{source}_validation_predictions.csv", pred_rows, ["path","source_dataset","source_label","true_label","predicted_label","correct","sha256","component_id"])
        print(f"fit {name}: {fit_seconds:.1f}s; PV val {candidate_results[name]['plantvillage']['accuracy']:.4f}; external val {candidate_results[name]['external']['accuracy']:.4f}", flush=True)

    # Source-balanced selection metric; external potato macro-F1 and PV supported-class macro-F1 have equal weight.
    for name, result in candidate_results.items():
        result["source_balanced_validation_macro_f1"] = float((result["plantvillage"]["macro_f1"] + result["external"]["macro_f1"])/2)
    selected = max(candidate_results, key=lambda n: (candidate_results[n]["source_balanced_validation_macro_f1"], candidate_results[n]["external"]["macro_recall"], candidate_results[n]["plantvillage"]["macro_f1"]))
    selected_model = fitted_models[selected]
    joblib.dump(fitted_models["hog_svm"], out / "candidate_hog_svm.joblib", compress=3)
    joblib.dump(fitted_models["hog_hsv_svm"], out / "candidate_hog_hsv_svm.joblib", compress=3)
    joblib.dump(selected_model, out / "selected_model.joblib", compress=3)

    # Convert winner and compare Python/ONNX class IDs on both validation sources only.
    onnx_result = {"status": "failed"}
    selected_sets = feature_sets[selected]
    onnx_inputs = {"plantvillage_validation": selected_sets[1], "external_validation": selected_sets[2]}
    try:
        from onnxruntime import InferenceSession
        from skl2onnx import to_onnx
        onnx = to_onnx(selected_model, selected_sets[0][:1].astype(np.float32), target_opset=17, options={"svc__zipmap": False})
        raw = onnx.SerializeToString()
        session = InferenceSession(raw, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        if input_meta.type != "tensor(float)" or input_meta.shape[1] != selected_sets[0].shape[1]:
            raise ValueError(f"ONNX input contract mismatch: {input_meta.type} {input_meta.shape}")
        parity = {}
        for key, X in onnx_inputs.items():
            python_ids = selected_model.predict(X).astype(np.int64)
            output = session.run(None, {input_meta.name: X.astype(np.float32, copy=False)})
            onnx_ids = np.asarray(output[0], dtype=np.int64).reshape(-1)
            mismatches = np.flatnonzero(python_ids != onnx_ids)
            parity[key] = {"count": len(python_ids), "matched": int(len(python_ids)-len(mismatches)), "mismatch_count": int(len(mismatches)), "mismatch_indices_first_50": mismatches[:50].tolist()}
        total_count = sum(v["count"] for v in parity.values())
        total_matches = sum(v["matched"] for v in parity.values())
        onnx_result = {"status": "passed" if total_count == total_matches else "mismatch", "opset": 17, "input_name": input_meta.name, "input_type": input_meta.type, "input_shape": input_meta.shape, "validation_parity": parity, "total_vectors": total_count, "matched_vectors": total_matches, "mismatches": total_count-total_matches}
        if onnx_result["status"] == "passed":
            (out / "selected_model.onnx").write_bytes(raw)
    except Exception as exc:
        onnx_result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    class_map = {str(i): label for i, label in enumerate(labels)}
    manifest_hashes = {key: manifest_digest(path) for key, path in manifests.items()}
    config = {
        "hog": {"image_size":[128,128],"resize_interpolation":"Pillow bilinear","grayscale":"Pillow convert L uint8, no EXIF orientation","orientations":9,"pixels_per_cell":[8,8],"cells_per_block":[2,2],"block_norm":"L2-Hys","transform_sqrt":False,"feature_count":8100,"dtype":"float32"},
        "hsv_histogram":{"space":"HSV via scikit-image rgb2hsv","input":"Pillow RGB bilinear 128x128; float32 /255 in [0,1]","bins":{"H":16,"S":8,"V":8},"range":[0,1],"normalization":"each channel histogram / pixel count","feature_count":32},
        "combined_order":"HOG 8100 then H(16 bins), S(8 bins), V(8 bins)","combined_feature_count":8132,
        "scaler":"StandardScaler fitted only on combined training rows","feature_implementation":"ml/scripts/run_phase5_experiments.py; unchanged Phase 5 implementation",
    }
    write_json(out / "class_labels.json", {"class_id_to_label": class_map})
    write_json(out / "feature_config.json", config)
    write_json(out / "validation_metrics.json", {"experiment_id": EXPERIMENT_ID, "selected_candidate": selected, "selection_rule":"Equal-weight mean of PlantVillage supported-class validation macro-F1 and external potato validation macro-F1; tie-break by external macro-recall then PlantVillage macro-F1. No test data used.", "candidate_results": candidate_results})
    write_json(out / "onnx_parity.json", onnx_result)
    try:
        packages = {name: version(name) for name in ("numpy","Pillow","scikit-image","scikit-learn","onnxruntime","skl2onnx","joblib")}
    except PackageNotFoundError:
        packages = {}
    metadata = {
        "experiment_id": EXPERIMENT_ID,"seed":SEED,"selected_candidate":selected,"selection_rule":"Equal-weight mean of the two source-specific supported-class validation macro-F1 values; tie-break external macro-recall then PlantVillage macro-F1.",
        "manifest_sha256":manifest_hashes,"source_counts":json.loads((out/"data_audit.json").read_text(encoding="utf-8"))["combined_training_counts_by_source_and_class"],
        "training_rows":len(train_rows),"training_class_counts":dict(Counter(r["label"] for r in train_rows)),"external_sampling":sample_info,
        "duplicate_handling":"Phase 6.6 canonical_hash_member rows only; Phase 8 v2 manifest contains distinct exact hashes. Cross-source exact-hash overlaps with either validation set block the entire training component. No exact-hash duplicates cross combined training and either validation set.",
        "parameters":candidate_results[selected]["svc_parameters"],"class_weight":"balanced across six classes using combined training labels","timing_seconds":{"data_audit":audit,"audit_seconds":json.loads((out/"data_audit.json").read_text(encoding="utf-8"))["audit_seconds"],"feature_extraction_train":extraction_train,"feature_extraction_plantvillage_validation":extraction_pv_val,"feature_extraction_external_validation":extraction_ext_val,"fit_by_candidate":total_train_seconds,"validation_prediction":prediction_seconds},
        "onnx":onnx_result,"software_versions":{"python":platform.python_version(),**packages},"external_test_accessed":False,
        "limitations":["External source has no reliable leaf/plant/capture identifiers; its existing exact-hash and perceptual-hash components were preserved, but related-image leakage cannot be ruled out.","External validation has been used for model selection and is not an untouched final test.","PlantVillage validation has no Potato healthy examples; external potato validation provides the healthy metric.","No independent external test evaluation was performed in Phase 9."],
    }
    write_json(out / "metadata.json", metadata)
    create_report(out, candidate_results, selected, onnx_result, sample_info, train_rows, pv_val, ext_val, total_train_seconds)
    print(json.dumps({"output":str(out),"selected":selected,"training_class_counts":metadata["training_class_counts"],"external_sample":sample_info,"candidate_summary":{n:{s:{k:v[k] for k in ("accuracy","macro_precision","macro_recall","macro_f1","example_count","potato_true_predicted_as_tomato")} for s,v in d.items() if s in ("plantvillage","external")} for n,d in candidate_results.items()},"onnx":onnx_result},indent=2),flush=True)


def create_report(out: Path, results: dict, selected: str, onnx: dict, sample: dict, train_rows: list[dict], pv_val: list[dict], ext_val: list[dict], fit_times: dict) -> None:
    lines = ["# Phase 9 domain-adaptation experiment", "", f"Experiment: `{EXPERIMENT_ID}`. External test data was not read.", "", "## Data used", "", f"- PlantVillage training: {sum(r['source_dataset']=='PlantVillage' for r in train_rows)} rows after cross-source component exclusion.", f"- External training: {sum(r['source_dataset']!='PlantVillage' for r in train_rows)} sampled rows across complete existing components; target was 1,000 per potato class.", f"- PlantVillage validation: {len(pv_val)} rows; external validation: {len(ext_val)} rows. These validation sources are reported separately.", f"- External sampling seed: {SEED}; selected counts: `{json.dumps(sample['selected_counts_by_class'])}`.", "- Source weights: no separate source weights. `class_weight='balanced'` is computed over combined training labels.", "", "## Validation results", "", "Metrics are source-specific; macro metrics average only classes represented in that validation set. Potato healthy is absent from PlantVillage validation.", ""]
    for candidate, data in results.items():
        lines.extend([f"### {candidate}", "", f"Fit time: {fit_times[candidate]:.1f}s; features: {data['feature_count']}; source-balanced macro-F1 selection score: {data['source_balanced_validation_macro_f1']:.4f}.", ""])
        for source in ("plantvillage", "external"):
            m = data[source]
            lines.append(f"**{source.title()} validation** — n={m['example_count']}; accuracy {m['accuracy']:.4f}; macro precision {m['macro_precision']:.4f}; macro recall {m['macro_recall']:.4f}; macro F1 {m['macro_f1']:.4f}; potato-to-tomato predictions {m['potato_true_predicted_as_tomato']}.")
            lines.append("")
            lines.append("| Class | Count | Precision | Recall | F1 |")
            lines.append("|---|---:|---:|---:|---:|")
            for label, score in m["per_class"].items():
                if score["support"]:
                    lines.append(f"| {label} | {score['support']} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} |")
            lines.extend(["", "Confusion matrix (rows true, columns predicted; labels: " + ", ".join(m["confusion_matrix"]["labels"]) + "):", "", "```json", json.dumps(m["confusion_matrix"]["values"], indent=2), "```", ""])
    lines.extend(["## Selection and deployment", "", f"Selected candidate: **{selected}** by the predeclared equal-weight mean of the PlantVillage and external validation macro-F1 scores, with external macro-recall then PlantVillage macro-F1 as tie-breakers.", "", f"ONNX: `{onnx.get('status')}`. Details are in `onnx_parity.json`.", "", "If HOG + HSV wins, browser deployment still requires a TypeScript HSV histogram extractor and browser parity verification. External validation has informed selection and is not a final independent test. External test images were not accessed.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
