"""Tests for deterministic synthetic data and SVM-to-ONNX parity."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.feasibility import make_synthetic_data, run_feasibility  # noqa: E402


class FeasibilityTests(unittest.TestCase):
    def test_synthetic_data_is_reproducible_and_numeric(self) -> None:
        first = make_synthetic_data()
        second = make_synthetic_data()
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first[0].shape[1], 8)

    def test_pipeline_exports_and_predictions_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = run_feasibility(Path(temporary_directory))
            self.assertTrue(report["python_onnx_predictions_match"])
            self.assertTrue(report["onnx_output_types"])
            self.assertTrue(
                all(output_type.startswith("tensor(") for output_type in report["onnx_output_types"])
            )
            self.assertTrue((Path(temporary_directory) / "synthetic_svm.onnx").is_file())
            self.assertTrue((Path(temporary_directory) / "test_vectors.json").is_file())


if __name__ == "__main__":
    unittest.main()
