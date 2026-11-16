"""Unit tests for Vision2VoicePredictor."""
import numpy as np
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


class FakeDecoder:
    """Always prefers: start -> 'dog' -> 'runs' -> end."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.plan = {2: 10, 10: 11, 11: 1}  # start->dog->runs->end

    def predict(self, inputs, verbose=0):
        _, seqs = inputs
        out = np.full((len(seqs), self.vocab), 1e-6)
        for i, s in enumerate(seqs):
            out[i, self.plan.get(int(s[s > 0][-1]), 1)] = 0.9
        return out / out.sum(axis=1, keepdims=True)


class TestBeamSearch:
    def _predictor(self):
        p = Vision2VoicePredictor.__new__(Vision2VoicePredictor)
        p.max_length, p.beam_width = 33, 3
        p.start_id, p.end_id = 2, 1
        p.index_word = {1: "end", 2: "start", 10: "dog", 11: "runs"}
        p.model = FakeDecoder(20)
        return p

    def test_returns_full_caption_not_first_word(self):
        caps = self._predictor().beam_search(np.zeros((1, 4096)))
        assert caps[0][0] == "dog runs"
        assert caps[0][1] > caps[-1][1]

    def test_generate_best_caption_uses_features(self):
        p = self._predictor()
        p._extract_feature = lambda img: np.zeros((1, 4096))
        assert p.generate_best_caption(np.zeros((50, 50, 3), dtype="uint8")) == "dog runs"
