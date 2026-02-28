"""
Vision2Voice: AI-powered image analysis and speech synthesis pipeline.

This package provides a modular pipeline for:
- Object detection using YOLOv8
- Image feature extraction using VGG16
- Caption generation using LSTM with Beam Search
- Text-to-speech synthesis using gTTS
"""

__version__ = "1.0.0"
__author__ = "Aniruddh Kulkarni"
__email__ = "anikulks@gmail.com"

from vision2voice.predictor import Vision2VoicePredictor
from vision2voice.audio import TextToSpeechEngine

__all__ = ["Vision2VoicePredictor", "TextToSpeechEngine"]
