"""Unit tests for vision2voice.audio.TextToSpeechEngine."""

from unittest.mock import MagicMock, patch

import pytest

from vision2voice.audio import TextToSpeechEngine


class TestTextToSpeechEngine:
    def test_raises_on_empty(self, tmp_path):
        e = TextToSpeechEngine(output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            e.synthesize("")

    @patch("vision2voice.audio.gTTS")
    def test_synthesize(self, mg, tmp_path):
        mi = MagicMock()
        mg.return_value = mi
        e = TextToSpeechEngine(output_dir=str(tmp_path))
        p = e.synthesize("hi", filename="t.mp3")
        mi.save.assert_called_once()
        assert p.endswith("t.mp3")
