"""Audit a local PlantVillage-style class-folder dataset; never downloads data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.dataset import (  # noqa: E402
    CANONICAL_EXPERIMENT_CLASSES,
    configured_dataset_root,
    create_upstream_leaf_group_manifest,
    create_splits,
    write_audit_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=configured_dataset_root(), help="Image root with one directory per class (or set PLANTVILLAGE_ROOT)")
    parser.add_argument("--output", type=Path, default=Path("ml/data/reports"), help="Directory for JSON and Markdown audit reports")
    parser.add_argument("--make-splits", action="store_true", help="Also create train/validation/test CSV manifests")
    parser.add_argument("--splits-output", type=Path, default=Path("ml/data/splits"))
    parser.add_argument("--classes", nargs="+", default=list(CANONICAL_EXPERIMENT_CLASSES), help="Exact folder names to select; labels are not guessed or normalized")
    parser.add_argument("--per-class-limit", type=int, default=None, help="Maximum selected images per class (identity groups remain together)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group-manifest", type=Path, default=None, help="Optional CSV with path,group_id columns for related leaves/variants")
    parser.add_argument("--upstream-leaf-map", type=Path, default=None, help="Convert upstream leaf-map.json and use it for leakage-aware splits")
    args = parser.parse_args()

    audit = write_audit_reports(args.root, args.output)
    print(json.dumps({key: audit[key] for key in ("dataset_root", "available", "status", "total_supported_images", "empty_classes", "exact_duplicate_file_count")}, indent=2))
    print(f"Reports written to: {args.output.resolve()}")
    if args.make_splits:
        try:
            if args.group_manifest is not None and args.upstream_leaf_map is not None:
                raise ValueError("Use either --group-manifest or --upstream-leaf-map, not both")
            group_manifest = args.group_manifest
            if args.upstream_leaf_map is not None:
                generated_manifest = args.splits_output / "leaf_group_manifest.csv"
                group_summary = create_upstream_leaf_group_manifest(args.root, args.upstream_leaf_map, generated_manifest)
                print("Upstream leaf grouping:")
                print(json.dumps(group_summary, indent=2))
                group_manifest = generated_manifest
            split_report = create_splits(
                args.root,
                args.splits_output,
                classes=args.classes,
                per_class_limit=args.per_class_limit,
                seed=args.seed,
                group_manifest=group_manifest,
            )
        except (OSError, ValueError) as error:
            print(f"Could not create split manifests: {error}", file=sys.stderr)
            return 2
        print(json.dumps(split_report, indent=2))
        print(f"Split manifests written to: {args.splits_output.resolve()}")
    return 0 if audit["available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
