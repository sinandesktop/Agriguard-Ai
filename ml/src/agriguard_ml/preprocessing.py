"""Deterministic Pillow and HOG preprocessing shared by training and Python inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog


IMAGE_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
BLOCK_NORM = "L2-Hys"
HOG_TRANSFORM_SQRT = False
HOG_FEATURE_COUNT = 8100


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load, convert to grayscale, and bilinearly resize to uint8 128x128.

    EXIF orientation is not applied. This choice is part of the model input contract.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as source:
            grayscale = source.convert("L")
            resized = grayscale.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
            return np.asarray(resized, dtype=np.uint8).copy()
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not decode image '{path}': {error}") from error


def extract_hog(image_path: str | Path) -> np.ndarray:
    """Return a flat, deterministic 8,100-element float32 HOG vector."""
    image = preprocess_image(image_path)
    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm=BLOCK_NORM,
        transform_sqrt=HOG_TRANSFORM_SQRT,
        feature_vector=True,
        channel_axis=None,
    )
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    if vector.shape != (HOG_FEATURE_COUNT,):
        raise RuntimeError(f"Unexpected HOG feature shape {vector.shape}; expected ({HOG_FEATURE_COUNT},)")
    if not np.isfinite(vector).all():
        raise RuntimeError("HOG feature vector contains non-finite values")
    return vector
