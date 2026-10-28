
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray


IMAGE_SIZE = (128, 128)
COLOR_BINS = 8


def extract_features(image_path):
    """
    Convert a leaf image into a numerical feature vector.

    Features:
    - HOG: local shapes and edges
    - RGB histograms: colour distribution
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        rgb = np.asarray(image, dtype=np.uint8)

    # Colour histograms: 8 bins per channel.
    color_features = []

    for channel in range(3):
        histogram, _ = np.histogram(
            rgb[:, :, channel],
            bins=COLOR_BINS,
            range=(0, 256),
            density=True,
        )
        color_features.extend(histogram)

    # HOG features: edges and local texture.
    gray = rgb2gray(rgb)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return np.concatenate([
        np.asarray(color_features, dtype=np.float32),
        hog_features.astype(np.float32),
    ])
