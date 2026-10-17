"""Tests for the real image-to-HOG preprocessing contract."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from agriguard_ml.preprocessing import (  # noqa: E402
    HOG_FEATURE_COUNT,
    IMAGE_SIZE,
    extract_hog,
    preprocess_image,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "fixture.png"
        image = Image.new("RGB", (73, 49), "#3a9b5c")
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 6, 58, 43), fill="#d8c33f", outline="#202020", width=2)
        image.save(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_preprocessing_is_fixed_size_grayscale_uint8(self) -> None:
        pixels = preprocess_image(self.path)
        self.assertEqual(pixels.shape, IMAGE_SIZE)
        self.assertEqual(pixels.dtype, np.uint8)
        np.testing.assert_array_equal(pixels, preprocess_image(self.path))

    def test_hog_vector_is_deterministic_float32_and_expected_length(self) -> None:
        first = extract_hog(self.path)
        second = extract_hog(self.path)
        self.assertEqual(first.shape, (HOG_FEATURE_COUNT,))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_invalid_image_raises_clear_value_error(self) -> None:
        invalid = Path(self.temporary.name) / "invalid.jpg"
        invalid.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            extract_hog(invalid)

    def test_identical_pixels_produce_identical_features_across_paths(self) -> None:
        second_path = Path(self.temporary.name) / "same_pixels.png"
        with Image.open(self.path) as image:
            image.save(second_path)
        np.testing.assert_array_equal(extract_hog(self.path), extract_hog(second_path))


if __name__ == "__main__":
    unittest.main()
