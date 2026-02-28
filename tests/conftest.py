"""
Pytest configuration and shared fixtures for Vision2Voice tests.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def sample_rgb_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_path(tmp_path, sample_rgb_image):
    from PIL import Image
    img = Image.fromarray(sample_rgb_image)
    path = tmp_path / "test_image.jpg"
    img.save(str(path))
    return str(path)


@pytest.fixture
def empty_models_dir(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    return str(models)
