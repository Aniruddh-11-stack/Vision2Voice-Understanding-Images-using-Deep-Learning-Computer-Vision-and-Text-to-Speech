"""
Text-to-Speech Engine Module.

Wraps Google Text-to-Speech (gTTS) and provides a clean interface
for audio synthesis with configurable language and speed settings.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from gtts import gTTS

logger = logging.getLogger(__name__)


class TextToSpeechEngine:
    def __init__(self, language="en", slow=False, output_dir=None):
        self.language = language
        self.slow = slow
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text, filename="output_speech.mp3"):
        if not text or not text.strip(): raise ValueError("Cannot synthesise empty text.")
        output_path = self.output_dir / filename
        gTTS(text=text, lang=self.language, slow=self.slow).save(str(output_path))
        return str(output_path)

    def synthesize_to_bytes(self, text):
        import io
        if not text or not text.strip(): raise ValueError("Cannot synthesise empty text.")
        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        buf = io.BytesIO(); tts.write_to_fp(buf); return buf.getvalue()
