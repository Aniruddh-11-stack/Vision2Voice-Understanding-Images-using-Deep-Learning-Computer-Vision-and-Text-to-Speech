"""Unit tests for Vision2VoicePredictor."""
import pytest
from unittest.mock import MagicMock, patch
from vision2voice.predictor import ModelNotReadyError, Vision2VoicePredictor

class TestVision2VoicePredictorInit:
    @patch("vision2voice.predictor.YOLO")
    @patch("vision2voice.predictor.VGG16")
    @patch("vision2voice.predictor.Model")
    def test_not_ready(self, mm, mv, my, empty_models_dir):
        mm.return_value=mv.return_value=my.return_value=MagicMock()
        assert not Vision2VoicePredictor(empty_models_dir).ready

    @patch("vision2voice.predictor.YOLO")
    @patch("vision2voice.predictor.VGG16")
    @patch("vision2voice.predictor.Model")
    def test_raises_when_not_ready(self, mm, mv, my, empty_models_dir):
        mm.return_value=mv.return_value=my.return_value=MagicMock()
        p = Vision2VoicePredictor(empty_models_dir)
        with pytest.raises(ModelNotReadyError): p.analyze_full_image("any.jpg")
