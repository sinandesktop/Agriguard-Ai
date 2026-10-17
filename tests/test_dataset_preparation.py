"""Synthetic-fixture tests for PlantVillage inspection and split preparation."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.dataset import (  # noqa: E402
    create_splits,
    create_upstream_leaf_group_manifest,
    inspect_dataset,
    write_audit_reports,
)


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "dataset"
        self.root.mkdir()
        self.classes = ("Crop___healthy", "Crop___disease")
        for class_index, label in enumerate(self.classes):
            class_dir = self.root / label
            class_dir.mkdir()
            for image_index in range(10):
                color = ((class_index * 71 + image_index * 11) % 256, image_index * 13, class_index * 101)
                Image.new("RGB", (12, 12), color).save(class_dir / f"image_{image_index}.png")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_missing_dataset_directory_is_reported(self) -> None:
        report = inspect_dataset(self.root / "missing")
        self.assertFalse(report["available"])
        self.assertEqual(report["status"], "dataset_missing")

    def test_missing_classes_prevent_split_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested class directories were not found"):
            create_splits(self.root, self.root / "splits", classes=("Missing___class",))

    def test_corrupted_image_is_reported_unreadable(self) -> None:
        path = self.root / self.classes[0] / "broken.png"
        path.write_bytes(b"not an image")
        report = inspect_dataset(self.root)
        self.assertEqual(report["classes"][self.classes[0]]["unreadable_count"], 1)
        self.assertEqual(report["unreadable_images"][0]["path"], f"{self.classes[0]}/broken.png")

    def test_unsupported_image_format_is_reported(self) -> None:
        (self.root / self.classes[0] / "animation.gif").write_bytes(b"GIF89a")
        report = inspect_dataset(self.root)
        self.assertEqual(report["unsupported_formats"][0]["extension"], ".gif")

    def test_empty_class_is_reported(self) -> None:
        (self.root / "Empty___class").mkdir()
        report = inspect_dataset(self.root)
        self.assertIn("Empty___class", report["empty_classes"])

    def test_exact_duplicate_files_are_detected(self) -> None:
        source = self.root / self.classes[0] / "image_0.png"
        shutil.copyfile(source, self.root / self.classes[0] / "copy.png")
        report = inspect_dataset(self.root)
        self.assertEqual(report["exact_duplicate_file_count"], 1)
        self.assertEqual(len(report["exact_duplicate_groups"]), 1)

    def test_splits_are_reproducible_with_fixed_seed(self) -> None:
        first, second = self.root / "first", self.root / "second"
        report = create_splits(self.root, first, classes=self.classes, seed=17)
        create_splits(self.root, second, classes=self.classes, seed=17)
        self.assertTrue(all(report["split_sizes"][name] > 0 for name in ("train", "validation", "test")))
        for filename in ("train.csv", "validation.csv", "test.csv", "splits.json"):
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_exact_duplicates_never_cross_splits(self) -> None:
        source = self.root / self.classes[0] / "image_0.png"
        shutil.copyfile(source, self.root / self.classes[1] / "duplicate_from_other_class.png")
        output = self.root / "splits"
        create_splits(self.root, output, classes=self.classes, seed=5)
        digest_splits: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            with (output / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    digest_splits.setdefault(row["sha256"], set()).add(split)
        self.assertTrue(all(len(split_names) == 1 for split_names in digest_splits.values()))

    def test_supplied_leaf_groups_never_cross_splits(self) -> None:
        original = self.root / self.classes[0] / "image_0.png"
        variant = self.root / self.classes[0] / "image_1.png"
        group_file = self.root / "leaf_groups.csv"
        group_file.write_text(
            "path,group_id\n"
            f"{self.classes[0]}/image_0.png,leaf-001\n"
            f"{self.classes[0]}/image_1.png,leaf-001\n",
            encoding="utf-8",
        )
        output = self.root / "grouped_splits"
        create_splits(self.root, output, classes=self.classes, seed=1, group_manifest=group_file)
        found: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            with (output / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row["path"] in {original.relative_to(self.root).as_posix(), variant.relative_to(self.root).as_posix()}:
                        found[row["path"]] = split
        self.assertEqual(found["Crop___healthy/image_0.png"], found["Crop___healthy/image_1.png"])

    def test_upstream_leaf_map_is_converted_to_path_groups(self) -> None:
        class_dir = self.root / self.classes[0]
        Image.new("RGB", (12, 12), (10, 20, 30)).save(class_dir / "uuid-a___Leaf A.JPG")
        Image.new("RGB", (12, 12), (30, 20, 10)).save(class_dir / "uuid-b___Leaf B.JPG")
        leaf_map = self.root / "leaf-map.json"
        leaf_map.write_text(
            '{"leaf a": ["Crop___healthy:::1"], "leaf b": ["Crop___healthy:::1"]}',
            encoding="utf-8",
        )
        output = self.root / "groups.csv"
        summary = create_upstream_leaf_group_manifest(self.root, leaf_map, output)
        self.assertEqual(summary["related_group_count"], 1)
        self.assertEqual(summary["grouped_image_count"], 2)

    def test_machine_and_human_reports_are_written(self) -> None:
        output = self.root / "reports"
        write_audit_reports(self.root, output)
        self.assertTrue((output / "dataset_summary.json").is_file())
        self.assertTrue((output / "dataset_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
