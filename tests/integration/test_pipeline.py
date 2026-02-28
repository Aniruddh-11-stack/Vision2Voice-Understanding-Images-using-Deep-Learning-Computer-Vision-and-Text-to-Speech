"""Integration tests for Vision2Voice pipeline."""
import os, pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
WEIGHTS_PRESENT = (MODELS_DIR/"modelConcat_1_89.h5").exists() and (MODELS_DIR/"caption_train_tokenizer.pkl").exists()
skip_if_no_weights = pytest.mark.skipif(not WEIGHTS_PRESENT, reason="No weights")

@skip_if_no_weights
def test_full_pipeline(sample_image_path):
    from vision2voice.predictor import Vision2VoicePredictor
    p = Vision2VoicePredictor(str(MODELS_DIR))
    assert p.ready
    b,e = p.analyze_full_image(sample_image_path)
    assert len(b) > 0 and len(e) > 0
