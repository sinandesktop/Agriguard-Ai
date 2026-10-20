"""Prepare and evaluate the Phase 8 Irish potato external dataset.

All generated data outputs are written under ml/data/splits/phase8_external/;
external images remain under the Git-ignored ml/data/external/ tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "ml/data/external/irish_potato"
ARCHIVE = EXTERNAL_ROOT / "source_archive/potato_leaf_blight.zip"
IMAGE_ROOT = EXTERNAL_ROOT / "images"
INITIAL_OUT = ROOT / "ml/data/splits/phase8_external/phase8_external_20261019_v1"
OUT = ROOT / "ml/data/splits/phase8_external/phase8_external_20261019_v2"
MODEL_DIR = ROOT / "ml/models/phase7/phase7_exp_20261019_v1"
SPLITS = ("train", "validation", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_MAP = {
    "healthy": "Potato___healthy",
    "earlyblt": "Potato___Early_blight",
    "early_blight": "Potato___Early_blight",
    "early-blight": "Potato___Early_blight",
    "lateblt": "Potato___Late_blight",
    "late_blight": "Potato___Late_blight",
    "late-blight": "Potato___Late_blight",
}
SOURCE = "Project-AgML/potato_leaf_blight_classification"
SEED = 20261019
PHASH_MAX_DISTANCE = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


class BKTree:
    """Small Hamming-distance index for 63-bit perceptual hashes."""
    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict[int, Any]] | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            dist = self.distance(value, node[0])
            if dist == 0:
                node[1].append(index)
                return
            child = node[2].get(dist)
            if child is None:
                node[2][dist] = (value, [index], {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        if self.root is None:
            return found
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = self.distance(value, node[0])
            if dist <= radius:
                found.extend((i, dist) for i in node[1])
            lo, hi = dist - radius, dist + radius
            stack.extend(child for edge, child in node[2].items() if lo <= edge <= hi)
        return found


def safe_extract_and_hash() -> tuple[str, int, int]:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Dataset archive is missing: {ARCHIVE}")
    expected = 7_170_478_690  # current Hugging Face response Content-Length
    actual = ARCHIVE.stat().st_size
    if actual != expected:
        raise ValueError(f"Archive size is {actual}, expected {expected}; refusing extraction")
    archive_hash = sha256_file(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"ZIP CRC verification failed: {bad}")
        infos = zf.infolist()
        expanded = sum(i.file_size for i in infos if not i.is_dir())
        for info in infos:
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            mode = info.external_attr >> 16
            if (mode & 0o170000) == 0o120000:
                raise ValueError(f"Symlink member rejected: {info.filename}")
        IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        marker = IMAGE_ROOT / ".phase8_extraction_complete.json"
        if marker.is_file():
            mark = json.loads(marker.read_text(encoding="utf-8"))
            if mark.get("archive_sha256") != archive_hash:
                raise ValueError("Existing extraction marker refers to a different archive")
            return archive_hash, len(infos), expanded
        if any(IMAGE_ROOT.iterdir()):
            raise FileExistsError(f"Extraction destination is non-empty without a completion marker: {IMAGE_ROOT}")
        free = shutil.disk_usage(IMAGE_ROOT.drive + "\\").free
        reserve = 20 * 1024**3
        if free < expanded + reserve:
            raise OSError(f"Need {expanded + reserve} bytes free for extraction plus 20 GiB reserve; found {free}")
        zf.extractall(IMAGE_ROOT)
    write_json(IMAGE_ROOT / ".phase8_extraction_complete.json", {
        "archive": str(ARCHIVE), "archive_sha256": archive_hash, "archive_bytes": actual,
        "zip_entries": len(infos), "uncompressed_bytes": expanded,
    })
    return archive_hash, len(infos), expanded


def label_for(path: Path, base: Path) -> tuple[str | None, str | None]:
    parts = [p.lower().strip() for p in path.relative_to(base).parts[:-1]]
    found = [(part, LABEL_MAP[part]) for part in parts if part in LABEL_MAP]
    labels = {v for _, v in found}
    if len(labels) == 1:
        raw = found[-1][0]
        return next(iter(labels)), raw
    return None, None


def phash(path: Path) -> int:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(a, norm="ortho")[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    bits = low[1:] > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def metadata_inventory(base: Path) -> list[str]:
    return [p.relative_to(base).as_posix() for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".xml", ".txt"}
            and p.name != ".phase8_extraction_complete.json"]


def is_archive_noise(path: Path, base: Path) -> bool:
    rel = path.relative_to(base)
    return "__MACOSX" in rel.parts or path.name.startswith("._")


def read_jsonl_labels(base: Path) -> tuple[dict[str, tuple[int, str]], list[str]]:
    """Read JSONL image names and labels; this source file has no group IDs."""
    mapping: dict[str, tuple[int, str]] = {}
    paths = metadata_inventory(base)
    for rel in paths:
        if not rel.lower().endswith(".jsonl"):
            continue
        with (base / Path(rel)).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                name = str(item.get("file_name", "")).replace("\\", "/")
                if not name:
                    continue
                try:
                    numeric = int(item["label"])
                except (KeyError, TypeError, ValueError):
                    numeric = -1
                if name in mapping:
                    raise ValueError(f"Duplicate metadata file_name at {rel}:{line_no}: {name}")
                mapping[name] = (numeric, str(item.get("label", "")))
    return mapping, paths


def assign_splits(groups_by_class: dict[str, list[list[int]]], records: list[dict[str, Any]]) -> dict[str, str]:
    rng = random.Random(SEED)
    assignment: dict[str, str] = {}
    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for label in sorted(groups_by_class):
        groups = groups_by_class[label][:]
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        targets = {s: ratios[s] * total for s in SPLITS}
        counts = Counter()
        # Largest components first; seeded shuffling resolves equal-size ties.
        groups.sort(key=len, reverse=True)
        for group in groups:
            # Assign to the split with the lowest fraction of its class target
            # already filled. Squared absolute error badly overfilled small
            # splits when components were numerous.
            split = min(SPLITS, key=lambda s: counts[s] / max(targets[s], 1))
            assignment[records[group[0]]["component_id"]] = split
            counts[split] += len(group)
    return assignment


def prepare() -> None:
    archive_hash, zip_entries, expanded_bytes = safe_extract_and_hash()
    metadata_labels, metadata_files = read_jsonl_labels(IMAGE_ROOT)
    files = sorted(p for p in IMAGE_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                   and not is_archive_noise(p, IMAGE_ROOT))
    metadata_set = set(metadata_files)
    unsupported = [p for p in IMAGE_ROOT.rglob("*") if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES
                   and p.name != ".phase8_extraction_complete.json" and not is_archive_noise(p, IMAGE_ROOT)
                   and p.relative_to(IMAGE_ROOT).as_posix() not in metadata_set]
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    unlabeled: list[dict[str, str]] = []
    label_conflicts: list[dict[str, str]] = []
    observed_metadata_paths: set[str] = set()
    metadata_numeric_to_source = {0: "earlyblt", 1: "healthy", 2: "lateblt"}
    for i, path in enumerate(files, 1):
        label, source_label = label_for(path, IMAGE_ROOT)
        if label is None:
            unlabeled.append({"path": path.relative_to(IMAGE_ROOT).as_posix(), "reason": "no unambiguous recognized class directory"})
            continue
        rel_path = path.relative_to(IMAGE_ROOT).as_posix()
        meta = metadata_labels.get(rel_path)
        if meta is None:
            metadata_label_raw = ""
            label_conflict = True
            label_conflicts.append({"path": rel_path, "folder_label": source_label, "metadata_label": "",
                                    "reason": "image has no matching metadata.jsonl entry"})
        else:
            observed_metadata_paths.add(rel_path)
            numeric, metadata_label_raw = meta
            expected_source = metadata_numeric_to_source.get(numeric)
            label_conflict = expected_source != source_label
            if label_conflict:
                label_conflicts.append({"path": rel_path, "folder_label": source_label,
                                        "metadata_label": str(metadata_label_raw),
                                        "reason": f"metadata numeric label maps to {expected_source!r}"})
        try:
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                fmt = im.format or "unknown"
            records.append({
                "path": rel_path, "absolute_path": path,
                "label": label, "source_label": source_label, "sha256": sha256_file(path),
                "source_metadata_label": metadata_label_raw, "label_conflict": label_conflict,
                "width": width, "height": height, "format": fmt,
            })
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            unreadable.append({"path": path.relative_to(IMAGE_ROOT).as_posix(), "error": str(exc)})
        if i % 5000 == 0:
            print(f"audited {i}/{len(files)} candidate image files", flush=True)

    # Exact-hash and pHash relationship components.
    uf = UnionFind(len(records))
    by_hash: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        by_hash[row["sha256"]].append(idx)
    exact_groups = []
    conflicting_exact_hashes: set[str] = set()
    for digest, members in by_hash.items():
        if len(members) > 1:
            for idx in members[1:]:
                uf.union(members[0], idx)
            member_labels = sorted({records[i]["label"] for i in members})
            if len(member_labels) > 1:
                conflicting_exact_hashes.add(digest)
            exact_groups.append({"sha256": digest, "paths": json.dumps([records[i]["path"] for i in members]),
                                 "labels": json.dumps(member_labels)})

    phashes: list[int] = []
    for i, row in enumerate(records, 1):
        phashes.append(phash(row["absolute_path"]))
        if i % 5000 == 0:
            print(f"computed perceptual hashes {i}/{len(records)}", flush=True)
    tree = BKTree()
    near_pairs: list[dict[str, Any]] = []
    for idx, value in enumerate(phashes):
        for other, distance in tree.query(value, PHASH_MAX_DISTANCE):
            if records[idx]["sha256"] != records[other]["sha256"]:
                uf.union(idx, other)
                near_pairs.append({"path_a": records[other]["path"], "label_a": records[other]["label"],
                                   "path_b": records[idx]["path"], "label_b": records[idx]["label"],
                                   "phash_distance": distance, "sha256_a": records[other]["sha256"],
                                   "sha256_b": records[idx]["sha256"],
                                   "reason": "pHash Hamming distance <= 4; candidate only, not confirmed duplicate"})
        tree.add(value, idx)
        if idx and idx % 5000 == 0:
            print(f"near-duplicate screen {idx}/{len(records)}; candidates {len(near_pairs)}", flush=True)

    near_related_paths = {p for pair in near_pairs for p in (pair["path_a"], pair["path_b"])}
    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(records)):
        components[uf.find(idx)].append(idx)
    group_lists: dict[str, list[list[int]]] = defaultdict(list)
    quarantined: list[dict[str, str]] = []
    forced_train_components: set[str] = set()
    for members in components.values():
        labels = {records[i]["label"] for i in members}
        has_exact_label_conflict = any(records[i]["sha256"] in conflicting_exact_hashes for i in members)
        has_metadata_label_conflict = any(records[i]["label_conflict"] for i in members)
        digest = hashlib.sha256("\n".join(sorted(records[i]["sha256"] for i in members)).encode()).hexdigest()[:16]
        cid = f"ext_{digest}"
        for i in members:
            records[i]["component_id"] = cid
            evidence = []
            if any(len(by_hash[records[j]["sha256"]]) > 1 for j in members):
                evidence.append("exact SHA-256 relation")
            if records[i]["path"] in near_related_paths:
                evidence.append("pHash distance <= 4 candidate relation")
            records[i]["grouping_evidence"] = "; ".join(evidence) if evidence else "no related image found by exact hash or pHash screen"
        if has_exact_label_conflict or has_metadata_label_conflict:
            for i in members:
                quarantined.append({"path": records[i]["path"], "label": records[i]["label"],
                                    "component_id": cid,
                                    "reason": "exact duplicate has conflicting class labels or metadata disagrees with folder label"})
        elif len(labels) != 1:
            # A pHash match is only a conservative relationship constraint.
            # Keep cross-label components together in training, without calling
            # them duplicate labels or contaminating fresh evaluation splits.
            forced_train_components.add(cid)
        else:
            group_lists[next(iter(labels))].append(members)

    assignments = assign_splits(group_lists, records)
    assignments.update({component_id: "train" for component_id in forced_train_components})
    for row in records:
        row["split"] = assignments.get(row.get("component_id"), "quarantined")
    component_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["split"] in SPLITS:
            component_splits[row["component_id"]].add(row["split"])
            hash_splits[row["sha256"]].add(row["split"])
    if any(len(splits) != 1 for splits in component_splits.values()):
        raise AssertionError("A connected component crossed splits")
    if any(len(splits) != 1 for splits in hash_splits.values()):
        raise AssertionError("An exact hash crossed splits")
    record_by_path = {r["path"]: r for r in records}
    for pair in near_pairs:
        a = record_by_path[pair["path_a"]]
        b = record_by_path[pair["path_b"]]
        if a["split"] != b["split"]:
            raise AssertionError(f"Near-duplicate candidate crossed splits: {pair['path_a']} / {pair['path_b']}")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_fields = ["path", "label", "source_label", "source_metadata_label", "sha256", "component_id", "grouping_evidence",
                       "source_repository", "source_archive", "split", "width", "height", "format"]
    counts = {s: Counter() for s in SPLITS}
    manifest_hashes: dict[str, str] = {}
    for split in SPLITS:
        rows = []
        for row in records:
            if row["split"] != split:
                continue
            counts[split][row["label"]] += 1
            rows.append({**row, "source_repository": SOURCE, "source_archive": ARCHIVE.name, "split": split})
        target = OUT / f"{split}.csv"
        write_csv(target, rows, manifest_fields)
        manifest_hashes[split] = sha256_file(target)

    write_csv(OUT / "exact_duplicates.csv", exact_groups, ["sha256", "paths", "labels"])
    write_csv(OUT / "near_duplicate_candidates.csv", near_pairs,
              ["path_a", "label_a", "path_b", "label_b", "phash_distance", "sha256_a", "sha256_b", "reason"])
    write_csv(OUT / "unreadable_images.csv", unreadable, ["path", "error"])
    write_csv(OUT / "unmapped_images.csv", unlabeled, ["path", "reason"])
    write_csv(OUT / "label_conflicts.csv", label_conflicts, ["path", "folder_label", "metadata_label", "reason"])
    write_csv(OUT / "quarantined_components.csv", quarantined, ["path", "label", "component_id", "reason"])
    inventory = []
    dimensions = Counter(f"{r['width']}x{r['height']}" for r in records)
    formats = Counter(r["format"] for r in records)
    dimension_pairs = [(r["width"], r["height"]) for r in records]
    dimension_range = {
        "min_width": min((x for x, _ in dimension_pairs), default=0),
        "max_width": max((x for x, _ in dimension_pairs), default=0),
        "min_height": min((y for _, y in dimension_pairs), default=0),
        "max_height": max((y for _, y in dimension_pairs), default=0),
    }
    for label in sorted(set(LABEL_MAP.values())):
        inventory.append({"label": label, "train": counts["train"][label],
                          "validation": counts["validation"][label], "test": counts["test"][label],
                          "readable_labeled": sum(1 for r in records if r["label"] == label),
                          "quarantined": sum(1 for r in quarantined if r["label"] == label)})
    write_csv(OUT / "class_inventory.csv", inventory,
              ["label", "train", "validation", "test", "readable_labeled", "quarantined"])
    summary = {
        "experiment_id": "phase8_external_20261019_v1", "source_repository": SOURCE,
        "original_zenodo_doi": "10.5281/zenodo.8286529", "archive_path": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256_local_only": archive_hash,
        "zip_entries": zip_entries, "expanded_bytes": expanded_bytes,
        "image_root": str(IMAGE_ROOT), "candidate_image_files": len(files),
        "readable_labeled_images": len(records), "unreadable_images": len(unreadable),
        "unsupported_files": len(unsupported), "unmapped_images": len(unlabeled),
        "metadata_files_found": metadata_files, "metadata_entries": len(metadata_labels),
        "metadata_entries_without_image": len(set(metadata_labels) - observed_metadata_paths),
        "metadata_label_issues": len(label_conflicts),
        "dimensions": dict(dimensions), "image_formats": dict(formats),
        "dimension_range": dimension_range,
        "metadata_numeric_label_mapping": {"0": "earlyblt", "1": "healthy", "2": "lateblt"},
        "exact_duplicate_groups": len(exact_groups), "extra_exact_duplicate_paths": sum(len(g["paths"]) - 1 for g in exact_groups),
        "exact_duplicate_label_conflict_groups": sum(len(g["labels"]) > 1 for g in exact_groups),
        "near_duplicate_candidate_pairs_phash_le4": len(near_pairs), "phash_max_distance": PHASH_MAX_DISTANCE,
        "group_metadata": "metadata.jsonl contains file_name and numeric class label only; no leaf, plant, capture, or group identifiers are present. Exact hashes and pHash candidates are conservative same-split constraints.",
        "split_seed": SEED, "split_ratios_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_class_counts": {s: dict(counts[s]) for s in SPLITS},
        "manifest_sha256": manifest_hashes,
        "conflicted_components_quarantined": len(quarantined),
        "mixed_label_components_training_only": len(forced_train_components),
        "mixed_label_training_only_paths": sum(1 for r in records if r["component_id"] in forced_train_components),
        "interpretation": "pHash edges are conservative same-split constraints, not confirmed duplicate labels.",
        "licence_displayed_by_huggingface": "CC BY 4.0", "zenodo_license_field": "blank; discrepancy recorded",
    }
    write_json(OUT / "summary.json", summary)
    report = ["# Phase 8 external potato dataset audit", "",
              f"- Source: `{SOURCE}` (Zenodo DOI `10.5281/zenodo.8286529`).",
              f"- Local archive SHA-256: `{archive_hash}` (locally computed; not an official checksum).",
              f"- Candidate image files: {len(files)}; readable, labeled: {len(records)}; unreadable: {len(unreadable)}; unsupported files: {len(unsupported)}; unmapped: {len(unlabeled)}.",
              f"- Image dimensions range: width {dimension_range['min_width']}-{dimension_range['max_width']}, height {dimension_range['min_height']}-{dimension_range['max_height']}; most common sizes: {dimensions.most_common(10)}. Formats: {dict(formats)}.",
              f"- Exact duplicate groups: {len(exact_groups)}; extra duplicate paths: {summary['extra_exact_duplicate_paths']}; cross-label exact groups: {summary['exact_duplicate_label_conflict_groups']}.",
              f"- pHash candidate pairs at distance <= {PHASH_MAX_DISTANCE}: {len(near_pairs)}. These are conservative grouping constraints, not confirmed duplicates.",
              f"- Cross-label pHash components assigned training-only: {len(forced_train_components)} components / {summary['mixed_label_training_only_paths']} paths. Exact-hash label conflicts and metadata disagreements were quarantined: {len(quarantined)} paths.",
              f"- Metadata files: {len(metadata_files)}; JSONL entries: {len(metadata_labels)}; metadata entries without matching images: {summary['metadata_entries_without_image']}; image/metadata label issues: {len(label_conflicts)}.",
              "- Metadata has no leaf/plant/capture grouping identifier; missing group metadata is not treated as independence.",
              "- Split seed: 20261019; target ratios by component: 70/15/15. Exact hashes and screened pHash pairs stay within a split.",
              "", "## Class counts by split", "", "| Label | Train | Validation | Test | Quarantined |", "|---|---:|---:|---:|---:|"]
    for row in inventory:
        report.append(f"| {row['label']} | {row['train']} | {row['validation']} | {row['test']} | {row['quarantined']} |")
    report.extend(["", "## Source and geographic limitations", "",
                   "The Hugging Face card displays CC BY 4.0; the original Zenodo record's licence field is blank. Local academic evaluation only; do not redistribute the archive or image contents in the public application. Attribute all dataset creators and cite the Zenodo DOI and associated Data in Brief article. The source is Tanzanian smallholder-farm imagery, primarily Shangi potato, so this external validation does not establish performance in Kerala.",
                   "", "See `source_provenance.md`, `summary.json`, and the CSV reports for details."])
    (OUT / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Preparation outputs saved: {OUT}")
    print(json.dumps(summary["split_class_counts"], indent=2))


def repartition() -> None:
    """Regenerate corrected splits from the completed Phase 8 audit artifacts.

    Reuses the audited pHash edge list rather than recomputing the expensive
    perceptual hashes. Writes a new v2 directory and leaves v1 untouched.
    """
    if not INITIAL_OUT.is_dir():
        raise FileNotFoundError(f"Initial audit output is missing: {INITIAL_OUT}")
    source_summary = json.loads((INITIAL_OUT / "summary.json").read_text(encoding="utf-8"))
    records_by_path: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        with (INITIAL_OUT / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                records_by_path[row["path"]] = row
    initial_quarantine: dict[str, dict[str, str]] = {}
    with (INITIAL_OUT / "quarantined_components.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            initial_quarantine[row["path"]] = row

    metadata_labels, metadata_files = read_jsonl_labels(IMAGE_ROOT)
    metadata_numeric_to_source = {0: "earlyblt", 1: "healthy", 2: "lateblt"}
    exact_group_paths: dict[str, list[str]] = defaultdict(list)
    for row in records_by_path.values():
        exact_group_paths[row["sha256"]].append(row["path"])

    # The first run quarantined mixed-label pHash components too broadly.
    # Recover those images here; quarantine only exact-hash label conflicts
    # and actual metadata/folder label disagreements.
    records: list[dict[str, Any]] = []
    for path, old in records_by_path.items():
        image_path = IMAGE_ROOT / path
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        actual_hash = sha256_file(image_path)
        if actual_hash != old["sha256"]:
            raise ValueError(f"Previously audited image hash changed: {path}")
        label, source_label = label_for(image_path, IMAGE_ROOT)
        if label != old["label"] or source_label != old["source_label"]:
            raise ValueError(f"Manifest label does not match source directory: {path}")
        records.append({**old, "absolute_path": image_path, "metadata_conflict": False})

    for path, old in initial_quarantine.items():
        if path in records_by_path:
            continue
        image_path = IMAGE_ROOT / path
        if not image_path.is_file():
            raise FileNotFoundError(f"Quarantined image is missing: {path}")
        actual_hash = sha256_file(image_path)
        label, source_label = label_for(image_path, IMAGE_ROOT)
        if label != old["label"]:
            raise ValueError(f"Quarantine label does not match source directory: {path}")
        meta = metadata_labels.get(path)
        if meta is None:
            metadata_raw = ""
            metadata_conflict = True
        else:
            numeric, metadata_raw = meta
            metadata_conflict = metadata_numeric_to_source.get(numeric) != source_label
        with Image.open(image_path) as im:
            im.load()
            width, height, fmt = im.width, im.height, im.format or "unknown"
        records.append({"path": path, "label": label, "source_label": source_label,
                        "source_metadata_label": metadata_raw, "sha256": actual_hash,
                        "absolute_path": image_path, "width": str(width), "height": str(height),
                        "format": fmt, "metadata_conflict": metadata_conflict})

    if len(records) != source_summary["readable_labeled_images"]:
        raise ValueError(f"Recovered {len(records)} readable records, expected {source_summary['readable_labeled_images']}")
    by_path = {row["path"]: i for i, row in enumerate(records)}
    if len(by_path) != len(records):
        raise ValueError("Duplicate image paths found while reconstructing records")

    uf = UnionFind(len(records))
    by_hash: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        by_hash[row["sha256"]].append(idx)
    conflicting_hashes: set[str] = set()
    exact_groups = []
    for digest, members in by_hash.items():
        if len(members) > 1:
            for idx in members[1:]:
                uf.union(members[0], idx)
            labels = sorted({records[i]["label"] for i in members})
            if len(labels) > 1:
                conflicting_hashes.add(digest)
            exact_groups.append({"sha256": digest,
                                 "paths": json.dumps([records[i]["path"] for i in members]),
                                 "labels": json.dumps(labels)})

    near_pairs: list[dict[str, Any]] = []
    with (INITIAL_OUT / "near_duplicate_candidates.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for edge in csv.DictReader(f):
            a, b = edge["path_a"], edge["path_b"]
            if a not in by_path or b not in by_path:
                raise ValueError(f"Near-duplicate edge refers to an unknown image: {a}, {b}")
            uf.union(by_path[a], by_path[b])
            near_pairs.append(edge)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        components[uf.find(i)].append(i)
    groups_by_class: dict[str, list[list[int]]] = defaultdict(list)
    forced_train: set[str] = set()
    quarantined: list[dict[str, str]] = []
    near_related_paths = {p for edge in near_pairs for p in (edge["path_a"], edge["path_b"])}
    component_splits_pre: dict[str, str] = {}
    for members in components.values():
        labels = {records[i]["label"] for i in members}
        has_exact_conflict = any(records[i]["sha256"] in conflicting_hashes for i in members)
        has_metadata_conflict = any(records[i]["metadata_conflict"] for i in members)
        cid_digest = hashlib.sha256("\n".join(sorted(records[i]["sha256"] for i in members)).encode()).hexdigest()[:16]
        cid = f"ext_{cid_digest}"
        for i in members:
            row = records[i]
            row["component_id"] = cid
            evidence = []
            if len(by_hash[row["sha256"]]) > 1:
                evidence.append("exact SHA-256 relation")
            if row["path"] in near_related_paths:
                evidence.append("pHash distance <= 4 candidate relation")
            row["grouping_evidence"] = "; ".join(evidence) if evidence else "no relation found by exact hash or pHash screen"
        if has_exact_conflict or has_metadata_conflict:
            for i in members:
                quarantined.append({"path": records[i]["path"], "label": records[i]["label"],
                                    "component_id": cid,
                                    "reason": "exact hash has conflicting labels or metadata label disagrees with folder"})
            component_splits_pre[cid] = "quarantined"
        elif len(labels) > 1:
            forced_train.add(cid)
            component_splits_pre[cid] = "train"
        else:
            groups_by_class[next(iter(labels))].append(members)

    assignments = assign_splits(groups_by_class, records)
    assignments.update({cid: "train" for cid in forced_train})
    for row in records:
        row["split"] = "quarantined" if row["component_id"] in component_splits_pre and component_splits_pre[row["component_id"]] == "quarantined" else assignments[row["component_id"]]

    # Verify every known relationship remains contained in exactly one split.
    component_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["split"] != "quarantined":
            component_splits[row["component_id"]].add(row["split"])
            hash_splits[row["sha256"]].add(row["split"])
    if any(len(v) != 1 for v in component_splits.values()) or any(len(v) != 1 for v in hash_splits.values()):
        raise AssertionError("A connected component or exact hash crossed splits")
    record_by_path = {row["path"]: row for row in records}
    for group in exact_groups:
        members = [record_by_path[path] for path in json.loads(group["paths"])]
        splits = {row["split"] for row in members}
        if len(splits) != 1:
            raise AssertionError(f"Exact duplicate group crossed splits: {group['sha256']}")
        group["split"] = next(iter(splits))
        group["component_id"] = members[0]["component_id"]
        group["conflicting_labels"] = str(len({row["label"] for row in members}) > 1).lower()
        group["image_count"] = len(members)
    for edge in near_pairs:
        a, b = record_by_path[edge["path_a"]], record_by_path[edge["path_b"]]
        if a["split"] != b["split"]:
            raise AssertionError(f"pHash candidate crossed splits: {a['path']} / {b['path']}")
        edge["split_a"] = a["split"]
        edge["split_b"] = b["split"]

    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ("source_provenance.md", "unreadable_images.csv", "unmapped_images.csv", "label_conflicts.csv"):
        source = INITIAL_OUT / filename
        if source.is_file():
            shutil.copy2(source, OUT / filename)
    manifest_fields = ["path", "label", "source_label", "source_metadata_label", "sha256", "component_id",
                       "grouping_evidence", "source_repository", "source_archive", "split", "width", "height", "format"]
    split_counts = {s: Counter() for s in SPLITS}
    manifest_hashes = {}
    for split in SPLITS:
        rows = []
        for row in records:
            if row["split"] != split:
                continue
            split_counts[split][row["label"]] += 1
            rows.append({**row, "source_repository": SOURCE, "source_archive": ARCHIVE.name})
        target = OUT / f"{split}.csv"
        write_csv(target, rows, manifest_fields)
        manifest_hashes[split] = sha256_file(target)
    write_csv(OUT / "exact_duplicates.csv", exact_groups,
              ["sha256", "paths", "labels", "split", "component_id", "conflicting_labels", "image_count"])
    write_csv(OUT / "near_duplicate_candidates.csv", near_pairs,
              ["path_a", "label_a", "path_b", "label_b", "phash_distance", "sha256_a", "sha256_b", "reason", "split_a", "split_b"])
    write_csv(OUT / "quarantined_components.csv", quarantined, ["path", "label", "component_id", "reason"])
    inventory = []
    for label in sorted(set(LABEL_MAP.values())):
        inventory.append({"label": label, "train": split_counts["train"][label],
                          "validation": split_counts["validation"][label], "test": split_counts["test"][label],
                          "readable_labeled": sum(1 for r in records if r["label"] == label),
                          "quarantined": sum(1 for r in quarantined if r["label"] == label),
                          "distinct_hashes": len({r["sha256"] for r in records if r["label"] == label})})
    write_csv(OUT / "class_inventory.csv", inventory,
              ["label", "train", "validation", "test", "readable_labeled", "quarantined", "distinct_hashes"])
    dimension_counts = Counter(f"{int(r['width'])}x{int(r['height'])}" for r in records)
    image_formats = Counter(r["format"] for r in records)
    widths = [int(r["width"]) for r in records]
    heights = [int(r["height"]) for r in records]
    dimension_range = {"min_width": min(widths), "max_width": max(widths),
                       "min_height": min(heights), "max_height": max(heights)}
    summary = dict(source_summary)
    summary.update({
        "experiment_id": "phase8_external_20261019_v2",
        "initial_attempt": "v1 used a defective greedy split allocator and is not valid for evaluation; v2 supersedes it.",
        "split_seed": SEED, "split_ratios_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_class_counts": {s: dict(split_counts[s]) for s in SPLITS},
        "manifest_sha256": manifest_hashes,
        "exact_duplicate_groups": len(exact_groups),
        "extra_exact_duplicate_paths": sum(len(json.loads(g["paths"])) - 1 for g in exact_groups),
        "exact_duplicate_label_conflict_groups": len(conflicting_hashes),
        "near_duplicate_candidate_pairs_phash_le4": len(near_pairs),
        "conflicted_components_quarantined": len(quarantined),
        "mixed_label_components_training_only": len(forced_train),
        "mixed_label_training_only_paths": sum(1 for r in records if r["component_id"] in forced_train and r["split"] == "train"),
        "component_integrity": "passed: exact hashes, pHash edges, and connected components each assigned to one split; mixed-label pHash-only components assigned training-only; conflicting exact-hash components quarantined.",
        "dimensions": dict(dimension_counts), "dimension_range": dimension_range, "image_formats": dict(image_formats),
        "metadata_files_found": metadata_files,
        "group_metadata": "metadata.jsonl contains image path and numeric class label only; no reliable capture, leaf, plant, or group identifier exists.",
        "fresh_test_evaluated": False,
    })
    write_json(OUT / "summary.json", summary)
    lines = ["# Phase 8 external potato dataset audit and split plan (v2)", "",
             f"- Dataset archive: `{ARCHIVE}`; bytes: {ARCHIVE.stat().st_size}; local SHA-256: `{summary['archive_sha256_local_only']}` (not an official checksum).",
             f"- Readable labeled images: {summary['readable_labeled_images']}; unreadable: {summary['unreadable_images']}; unsupported: {summary['unsupported_files']}; metadata label issues: {summary['metadata_label_issues']}.",
             f"- Dimensions: width {dimension_range['min_width']}-{dimension_range['max_width']}, height {dimension_range['min_height']}-{dimension_range['max_height']}; common sizes: {dimension_counts.most_common(10)}; formats: {dict(image_formats)}.",
             f"- Exact duplicate groups: {len(exact_groups)}; extra duplicate paths: {summary['extra_exact_duplicate_paths']}; conflicting-label exact hashes: {len(conflicting_hashes)}.",
             f"- pHash candidate edges (distance <= 4): {len(near_pairs)}. These are conservative split constraints, not confirmed duplicates.",
             f"- Conflicting exact-hash components quarantined: {len(quarantined)} paths. Cross-label pHash-only components sent to train: {len(forced_train)} components / {summary['mixed_label_training_only_paths']} paths.",
             "- Group metadata: metadata.jsonl contains `file_name` and numeric `label`, no leaf/plant/capture/group identifier. Missing grouping metadata is treated as unknown; exact hashes and pHash candidates are constrained to one split.",
             f"- Seed: {SEED}; target ratios 70/15/15 at connected-component level. The previous v1 split manifests are invalid and must not be used.",
             "", "## Class counts by split", "", "| Label | Train paths | Validation paths | Test paths | Readable | Distinct hashes | Quarantined |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in inventory:
        lines.append(f"| {row['label']} | {row['train']} | {row['validation']} | {row['test']} | {row['readable_labeled']} | {row['distinct_hashes']} | {row['quarantined']} |")
    lines.extend(["", "## Source, licence, and scope", "",
                  "The Hugging Face card displays CC BY 4.0, while the original Zenodo record's licence field is blank. This archive is for local academic evaluation only and must not be redistributed with the public application. The dataset was collected on Tanzanian smallholder farms, mainly using Samsung Galaxy A03 Core phones, and predominantly represents the Shangi variety. Results do not establish performance in Kerala.",
                  "", "The external test manifest is reserved. Phase 8 inference uses the validation manifest only."])
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (INITIAL_OUT / "split_generation_note.md").write_text(
        "# Phase 8 v1 split status\n\nThe v1 audit detected an allocator defect: it produced approximately 5% training and 47.5% each validation and test for the per-class groups. Do not use v1 train/validation/test manifests for evaluation. The corrected v2 manifests are in `../phase8_external_20261019_v2/`. The v1 audit counts and pHash/exact-duplicate candidate outputs are retained as an intermediate audit record.\n",
        encoding="utf-8")
    print(f"Corrected manifests saved to {OUT}")
    print(json.dumps(summary["split_class_counts"], indent=2))


def evaluate() -> None:
    sys.path.insert(0, str(ROOT / "ml/src"))
    sys.path.insert(0, str(ROOT / "ml/scripts"))
    import joblib
    from run_phase5_experiments import extract_features

    labels_payload = json.loads((MODEL_DIR / "class_labels.json").read_text(encoding="utf-8"))
    id_to_label = {int(k): v for k, v in labels_payload["class_id_to_label"].items()}
    if len(id_to_label) != 6:
        raise ValueError("Phase 7 class map is not the expected six-label map")
    model = joblib.load(MODEL_DIR / "selected_model.joblib")
    feature_config = json.loads((MODEL_DIR / "feature_config.json").read_text(encoding="utf-8"))
    if feature_config.get("combined_feature_count") != 8132:
        raise ValueError("Saved Phase 7 feature configuration is not 8,132 values")
    if feature_config["combined_order"] != "HOG 8100 then H(16 bins), S(8 bins), V(8 bins)":
        raise ValueError("Unexpected Phase 7 feature order")

    rows = list(csv.DictReader((OUT / "validation.csv").open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("External validation manifest is empty")
    seen_hash: set[str] = set()
    eval_rows = []
    for row in rows:
        path = IMAGE_ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"External validation image missing/hash mismatch: {row['path']}")
        if row["sha256"] in seen_hash:
            continue
        seen_hash.add(row["sha256"])
        eval_rows.append(row)
    if not eval_rows:
        raise ValueError("No distinct exact-hash validation images remain")

    start = time.perf_counter()
    hog_x, hsv_x, feature_labels, durations = extract_features(
        [{"path": r["path"], "label": r["label"]} for r in eval_rows], IMAGE_ROOT,
        "Phase 8 external validation", include_color=True)
    features = np.hstack((hog_x, hsv_x)).astype(np.float32, copy=False)
    if features.shape != (len(eval_rows), 8132):
        raise ValueError(f"Expected validation feature matrix (*,8132), found {features.shape}")
    extraction_seconds = time.perf_counter() - start

    label_to_id = {v: k for k, v in id_to_label.items()}
    y_true = np.asarray([label_to_id[r["label"]] for r in eval_rows], dtype=np.int64)
    start = time.perf_counter()
    y_pred = model.predict(features).astype(np.int64)
    prediction_seconds = time.perf_counter() - start
    potato_labels = [label_to_id["Potato___Early_blight"], label_to_id["Potato___Late_blight"], label_to_id["Potato___healthy"]]
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=potato_labels, average=None, zero_division=0)
    present_indices = [i for i, n in enumerate(support) if n]
    macro = (float(np.mean(p[present_indices])), float(np.mean(r[present_indices])), float(np.mean(f[present_indices])))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    support_by_label = Counter(r["label"] for r in eval_rows)
    tomato_ids = {i for i, label in id_to_label.items() if label.startswith("Tomato___")}
    healthy_id = label_to_id["Potato___healthy"]
    diseased_ids = {label_to_id["Potato___Early_blight"], label_to_id["Potato___Late_blight"]}
    metrics = {
        "experiment_id": "phase8_external_20261019_v2", "frozen_model": str(MODEL_DIR / "selected_model.joblib"),
        "evaluated_split": "validation only; external test manifest was not loaded",
        "metrics_scope": "Three supported potato labels; tomato-output rows remain incorrect errors and appear in the six-label confusion matrix.",
        "accuracy": float(np.mean(y_true == y_pred)), "macro_precision": macro[0], "macro_recall": macro[1], "macro_f1": macro[2],
        "per_class": {id_to_label[label_id]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, label_id in enumerate(potato_labels)},
        "confusion_matrix": {"labels": [id_to_label[i] for i in range(6)], "values": cm.astype(int).tolist()},
        "class_counts": dict(support_by_label),
        "validation_manifest_path_counts": dict(Counter(r["label"] for r in rows)),
        "example_count": len(eval_rows),
        "predicted_class_counts": dict(Counter(id_to_label[int(i)] for i in y_pred)),
        "tomato_class_predictions_on_potato_images": int(sum(pred in tomato_ids for pred in y_pred)),
        "healthy_potato_predicted_diseased": int(sum(t == healthy_id and p_ in diseased_ids | tomato_ids for t, p_ in zip(y_true, y_pred))),
        "diseased_potato_predicted_healthy": int(sum(t in diseased_ids and p_ == healthy_id for t, p_ in zip(y_true, y_pred))),
        "feature_extraction_seconds": extraction_seconds,
        "feature_extraction_breakdown_seconds": durations,
        "prediction_seconds": prediction_seconds,
        "duplicates_omitted_from_metrics": len(rows) - len(eval_rows),
        "probability_note": "The SVM is not calibrated; no raw decision values are reported as probabilities.",
    }
    write_json(OUT / "validation_metrics.json", metrics)
    pred_rows = []
    for source_row, true_id, pred_id in zip(eval_rows, y_true, y_pred):
        pred_rows.append({"path": source_row["path"], "source_label": source_row["source_label"],
                          "true_label": id_to_label[int(true_id)], "predicted_label": id_to_label[int(pred_id)],
                          "correct": str(int(true_id) == int(pred_id)).lower(), "sha256": source_row["sha256"],
                          "component_id": source_row["component_id"]})
    write_csv(OUT / "validation_predictions.csv", pred_rows,
              ["path", "source_label", "true_label", "predicted_label", "correct", "sha256", "component_id"])
    text = ["# Frozen Phase 7 model: external validation", "",
            "Dataset: independent Tanzania-source potato photographs. Metrics below use one representative per exact SHA-256 hash; all validation paths are listed separately in `summary.json`.", "",
            f"- Accuracy: {metrics['accuracy']:.4f}", f"- Macro precision: {macro[0]:.4f}",
            f"- Macro recall: {macro[1]:.4f}", f"- Macro F1: {macro[2]:.4f}",
            f"- Valid distinct exact-hash examples: {len(eval_rows)}", f"- Tomato predictions: {metrics['tomato_class_predictions_on_potato_images']}",
            f"- Healthy predicted diseased: {metrics['healthy_potato_predicted_diseased']}",
            f"- Diseased predicted healthy: {metrics['diseased_potato_predicted_healthy']}",
            f"- Feature extraction: {extraction_seconds:.2f}s; prediction: {prediction_seconds:.2f}s", "",
            "## Per-class metrics (exact-hash distinct images)", "",
            "| Label | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in ("Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"):
        item = metrics["per_class"][label]
        text.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
    text.extend(["", "## Prediction counts", "",
            "| Predicted label | Count |", "|---|---:|"])
    for label, count in sorted(metrics["predicted_class_counts"].items()):
        text.append(f"| {label} | {count} |")
    text.extend(["", 
            "## Confusion matrix (rows=true, columns=predicted)", "",
            "Labels: " + ", ".join(id_to_label[i] for i in range(6)), "",
            "```text", json.dumps(cm.astype(int).tolist(), indent=2), "```", "",
            "This is a Tanzania-source validation result, separate from PlantVillage metrics. It does not establish Kerala field performance. The held-out external test split was not evaluated."])
    (OUT / "evaluation_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "repartition", "evaluate"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "repartition":
        repartition()
    else:
        evaluate()


if __name__ == "__main__":
    main()
