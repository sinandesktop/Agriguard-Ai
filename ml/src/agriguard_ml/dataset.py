"""PlantVillage inspection and leakage-aware split manifest utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UNSUPPORTED_IMAGE_FORMATS = {".gif", ".heic", ".heif", ".avif", ".jfif"}
CANONICAL_EXPERIMENT_CLASSES = (
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(root: Path) -> dict[str, Any]:
    """Inspect immediate class directories and report image/data quality facts."""
    root = root.expanduser().resolve()
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "available": root.is_dir(),
        "supported_formats": sorted(SUPPORTED_FORMATS),
        "total_supported_images": 0,
        "classes": {},
        "empty_classes": [],
        "unreadable_images": [],
        "unsupported_formats": [],
        "exact_duplicate_groups": [],
        "exact_duplicate_file_count": 0,
        "near_duplicate_analysis": "Not run; exact byte hashes only. Supply leaf/variant group metadata to prevent known related views from crossing splits.",
        "structure_warnings": [],
        "class_name_warnings": [],
        "image_records": [],
    }
    if not root.is_dir():
        report["status"] = "dataset_missing"
        report["structure_warnings"].append("Dataset root directory does not exist.")
        return report

    class_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    if not class_dirs:
        report["structure_warnings"].append("No immediate class directories found under dataset root.")
    # Class directories must be one level below the configured image root.
    for item in root.iterdir():
        if item.is_file() and item.suffix.casefold() in SUPPORTED_FORMATS:
            report["structure_warnings"].append(f"Image found at dataset root, outside a class directory: {item.name}")

    hashes: dict[str, list[str]] = defaultdict(list)
    for class_dir in class_dirs:
        class_report = {"image_count": 0, "readable_count": 0, "unreadable_count": 0, "unsupported_count": 0}
        direct_files = sorted((p for p in class_dir.iterdir() if p.is_file()), key=lambda p: p.name.casefold())
        nested_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        if nested_dirs:
            report["structure_warnings"].append(
                f"Nested directories under class '{class_dir.name}'; expected images directly in each class directory."
            )
        for image_path in direct_files:
            suffix = image_path.suffix.casefold()
            relative = image_path.relative_to(root).as_posix()
            if suffix in UNSUPPORTED_IMAGE_FORMATS:
                class_report["unsupported_count"] += 1
                report["unsupported_formats"].append({"path": relative, "extension": suffix})
                continue
            if suffix not in SUPPORTED_FORMATS:
                continue
            class_report["image_count"] += 1
            report["total_supported_images"] += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
            except (OSError, ValueError, UnidentifiedImageError) as error:
                class_report["unreadable_count"] += 1
                report["unreadable_images"].append({"path": relative, "error": str(error)})
                continue
            class_report["readable_count"] += 1
            digest = _sha256(image_path)
            hashes[digest].append(relative)
            report["image_records"].append(
                {"path": relative, "label": class_dir.name, "sha256": digest}
            )
        if class_report["image_count"] == 0:
            report["empty_classes"].append(class_dir.name)
        report["classes"][class_dir.name] = class_report
        if "___" not in class_dir.name:
            report["class_name_warnings"].append(
                f"Class '{class_dir.name}' does not follow the common PlantVillage Crop___Condition naming pattern; verify its label."
            )

    report["exact_duplicate_groups"] = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(hashes.items())
        if len(paths) > 1
    ]
    report["exact_duplicate_file_count"] = sum(
        len(group["paths"]) - 1 for group in report["exact_duplicate_groups"]
    )
    report["status"] = "inspected"
    return report


def _read_group_manifest(path: Path | None, root: Path) -> dict[str, str]:
    """Read optional CSV with path,group_id columns for leaf/variant grouping."""
    groups: dict[str, str] = {}
    if path is None:
        return groups
    with path.open(newline="", encoding="utf-8-sig") as manifest:
        reader = csv.DictReader(manifest)
        if not reader.fieldnames or not {"path", "group_id"}.issubset(reader.fieldnames):
            raise ValueError("Group manifest CSV must contain path and group_id columns")
        for row in reader:
            raw_path, group_id = (row.get("path") or "").strip(), (row.get("group_id") or "").strip()
            if raw_path and group_id:
                candidate = Path(raw_path)
                try:
                    key = candidate.resolve().relative_to(root.resolve()).as_posix() if candidate.is_absolute() else candidate.as_posix()
                except ValueError:
                    key = candidate.as_posix()
                groups[key] = group_id
    return groups


def create_upstream_leaf_group_manifest(
    root: Path, leaf_map_path: Path, output_csv: Path
) -> dict[str, int]:
    """Translate upstream PlantVillage leaf-map.json into this project's CSV format."""
    root = root.expanduser().resolve()
    upstream_map = json.loads(leaf_map_path.read_text(encoding="utf-8"))
    if not isinstance(upstream_map, dict):
        raise ValueError("Upstream leaf map must be a JSON object")
    audit = inspect_dataset(root)
    if not audit["available"]:
        raise ValueError(f"Dataset directory is missing: {audit['dataset_root']}")

    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    matched_paths: set[str] = set()
    for record in audit["image_records"]:
        image_path = root / Path(record["path"])
        # Upstream keys are the original leaf filename after the UUID + "___" prefix.
        leaf_key = image_path.stem.split("___", 1)[-1].casefold().strip()
        raw_ids = upstream_map.get(leaf_key, [])
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        group_ids = {
            value for value in raw_ids
            if isinstance(value, str) and value.startswith(record["label"] + ":::")
        }
        if not group_ids:
            continue
        image_key = "image:" + record["path"]
        find(image_key)
        matched_paths.add(record["path"])
        for group_id in group_ids:
            union(image_key, "leaf:" + group_id)

    components: dict[str, list[str]] = defaultdict(list)
    for path in matched_paths:
        components[find("image:" + path)].append(path)
    related_components = [sorted(paths) for paths in components.values() if len(paths) > 1]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "group_id"))
        writer.writeheader()
        for paths in sorted(related_components, key=lambda group: group[0]):
            component_id = "upstream-leaf:" + hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:16]
            for path in paths:
                writer.writerow({"path": path, "group_id": component_id})
    return {
        "matched_image_count": len(matched_paths),
        "related_group_count": len(related_components),
        "grouped_image_count": sum(len(paths) for paths in related_components),
    }


def create_splits(
    root: Path,
    output_dir: Path,
    classes: tuple[str, ...] | list[str] = CANONICAL_EXPERIMENT_CLASSES,
    per_class_limit: int | None = None,
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    group_manifest: Path | None = None,
) -> dict[str, Any]:
    """Write reproducible, class-stratified split CSVs grouped by duplicate/leaf ID."""
    audit = inspect_dataset(root)
    if not audit["available"]:
        raise ValueError(f"Dataset directory is missing: {audit['dataset_root']}")
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Split ratios must be three positive values that sum to 1")
    missing = [name for name in classes if name not in audit["classes"]]
    if missing:
        raise ValueError(f"Requested class directories were not found: {', '.join(missing)}")
    group_ids = _read_group_manifest(group_manifest, root)
    records = [record for record in audit["image_records"] if record["label"] in classes]
    if per_class_limit is not None:
        if per_class_limit < 1:
            raise ValueError("per_class_limit must be a positive integer")
        # Select whole identity groups so a cap cannot split duplicates/variants.
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for record in records:
            identity = group_ids.get(record["path"], f"sha256:{record['sha256']}")
            buckets[record["label"] + "\0" + identity].append(record)
        rng = random.Random(seed)
        chosen: list[dict[str, str]] = []
        for class_name in classes:
            groups = [items for key, items in buckets.items() if key.startswith(class_name + "\0")]
            rng.shuffle(groups)
            count = 0
            for group in groups:
                if count >= per_class_limit:
                    break
                chosen.extend(group)
                count += len(group)
        records = chosen

    if not records:
        raise ValueError("No readable images are available in the selected classes")
    # Identity keys union exact duplicates and externally supplied leaf/variant IDs.
    uf_parent: dict[str, str] = {}

    def find(key: str) -> str:
        uf_parent.setdefault(key, key)
        if uf_parent[key] != key:
            uf_parent[key] = find(uf_parent[key])
        return uf_parent[key]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            uf_parent[right_root] = left_root

    for record in records:
        union("sha256:" + record["sha256"], "metadata:" + group_ids[record["path"]]) if record["path"] in group_ids else find("sha256:" + record["sha256"])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        identity = "sha256:" + record["sha256"]
        if record["path"] in group_ids:
            identity = "metadata:" + group_ids[record["path"]]
        grouped[find(identity)].append(record)

    split_names = ("train", "validation", "test")
    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    label_totals = Counter(record["label"] for record in records)
    split_counts = {name: Counter() for name in split_names}
    assignments: dict[str, list[dict[str, str]]] = {name: [] for name in split_names}
    for _, group_records in group_items:
        label_counts = Counter(record["label"] for record in group_records)
        def cost(candidate: str) -> float:
            score = 0.0
            for label, total in label_totals.items():
                for split_index, split_name in enumerate(split_names):
                    after = split_counts[split_name][label]
                    if split_name == candidate:
                        after += label_counts[label]
                    target = total * ratios[split_index]
                    score += ((after - target) / max(total, 1)) ** 2
            for split_index, split_name in enumerate(split_names):
                size_after = len(assignments[split_name])
                if split_name == candidate:
                    size_after += len(group_records)
                size_target = len(records) * ratios[split_index]
                score += ((size_after - size_target) / max(len(records), 1)) ** 2
            return score
        selected_split = min(split_names, key=lambda name: (cost(name), split_names.index(name)))
        assignments[selected_split].extend(group_records)
        split_counts[selected_split].update(label_counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    root_path = Path(root).expanduser().resolve()
    for split_name, split_records in assignments.items():
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "label", "sha256", "group_id"))
            writer.writeheader()
            for record in sorted(split_records, key=lambda row: row["path"]):
                writer.writerow({
                    **record,
                    "group_id": group_ids.get(record["path"], "sha256:" + record["sha256"]),
                })
    split_report = {
        "dataset_root": str(root_path),
        "seed": seed,
        "ratios": dict(zip(split_names, ratios)),
        "selected_classes": list(classes),
        "per_class_limit": per_class_limit,
        "group_manifest": str(group_manifest.resolve()) if group_manifest else None,
        "grouping": "exact content hashes plus optional supplied leaf/variant group IDs",
        "near_duplicate_method": "not run; exact hashes and optional metadata only",
        "split_counts": {name: dict(Counter(row["label"] for row in rows)) for name, rows in assignments.items()},
        "split_sizes": {name: len(rows) for name, rows in assignments.items()},
        "manifests": {name: f"{name}.csv" for name in split_names},
    }
    (output_dir / "splits.json").write_text(json.dumps(split_report, indent=2) + "\n", encoding="utf-8")
    return split_report


def write_audit_reports(root: Path, output_dir: Path) -> dict[str, Any]:
    """Save JSON and readable Markdown reports for a dataset inspection."""
    report = inspect_dataset(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PlantVillage dataset inspection", "", f"- Root: `{report['dataset_root']}`",
        f"- Available: **{report['available']}**", f"- Status: **{report['status']}**",
        f"- Supported images: **{report['total_supported_images']}**", "", "## Classes", "",
    ]
    if report["classes"]:
        lines.extend(["| Class directory | Images | Readable | Unreadable | Unsupported |", "|---|---:|---:|---:|---:|"])
        for label, counts in report["classes"].items():
            lines.append(f"| `{label}` | {counts['image_count']} | {counts['readable_count']} | {counts['unreadable_count']} | {counts['unsupported_count']} |")
    else:
        lines.append("No class directories were inspected.")
    lines.extend([
        "", f"## Data quality", "", f"- Empty classes: {len(report['empty_classes'])}",
        f"- Unreadable images: {len(report['unreadable_images'])}",
        f"- Unsupported formats: {len(report['unsupported_formats'])}",
        f"- Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
        f"- Extra exact duplicate files: {report.get('exact_duplicate_file_count', 0)}", "",
        f"- Related-view / near-duplicate assessment: {report['near_duplicate_analysis']}", "",
    ])
    for heading, key in (("Structure warnings", "structure_warnings"), ("Class-name warnings", "class_name_warnings")):
        lines.extend([f"## {heading}", ""])
        lines.extend([f"- {entry}" for entry in report[key]] or ["- None."])
        lines.append("")
    if report["exact_duplicate_groups"]:
        lines.extend(["## Exact duplicate paths", ""])
        for group in report["exact_duplicate_groups"]:
            lines.append(f"- `{group['sha256']}`: " + ", ".join(f"`{path}`" for path in group["paths"]))
    (output_dir / "dataset_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report


def configured_dataset_root() -> Path:
    """Resolve explicit environment configuration or the documented default."""
    configured = os.environ.get("PLANTVILLAGE_ROOT")
    return Path(configured) if configured else Path("ml/data/plantvillage/color")
