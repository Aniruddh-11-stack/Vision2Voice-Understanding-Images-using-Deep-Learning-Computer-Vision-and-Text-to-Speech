"""Unit tests for vision2voice.utils.image_utils."""

import numpy as np
import pytest

from vision2voice.utils.image_utils import load_image_rgb, resize_and_pad, validate_image_path


class TestValidateImagePath:
    def test_valid_jpeg(self, sample_image_path):
        assert validate_image_path(sample_image_path).exists()

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_image_path(tmp_path / "nonexistent.jpg")

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "file.xyz"; bad_file.write_text("dummy")
        with pytest.raises(ValueError): validate_image_path(bad_file)


class TestLoadImageRgb:
    def test_returns_rgb_array(self, sample_image_path):
        img = load_image_rgb(sample_image_path)
        assert isinstance(img, np.ndarray) and img.shape[2] == 3

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError): load_image_rgb(tmp_path / "ghost.jpg")


class TestResizeAndPad:
    def test_output_shape(self, sample_rgb_image):
        assert resize_and_pad(sample_rgb_image).shape == (224, 224, 3)
