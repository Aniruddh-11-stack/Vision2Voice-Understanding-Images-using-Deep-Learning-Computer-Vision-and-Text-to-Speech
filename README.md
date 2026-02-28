# Vision2Voice

> AI-powered image understanding pipeline that bridges visual perception and spoken language ÃÂ¢ÃÂÃÂ combining deep CNNs, sequence models, object detection, and text-to-speech into a single cohesive system.

[![CI](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/ci.yml/badge.svg)](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/ci.yml)
[![Docker](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/docker-build.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Vision2Voice App Demo](docs/screenshots/app_demo.png)

## ðï¸ Model Architecture

The model uses a dual-input architecture combining:
- **Image encoder**: VGG16 (4096-dim features) â Dense(256)
- **Language model**: Embedding + LSTM (256-dim)
- **Fusion**: Concatenate â Dense(256) â Dense(vocab_size, softmax)

![Model Architecture](docs/model_architecture.png)

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Docker Deployment](#docker-deployment)
- [Dataset & Model Weights](#dataset--model-weights)
- [Development](#development)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Overview

**Vision2Voice** goes beyond simple image captioning. Rather than describing a scene as a whole, it:

1. **Detects** every distinct object in the image using **YOLOv8**
2. **Crops** each detected object into its own sub-image
3. **Captions** every crop independently using a trained **VGG16 + LSTM Ensemble**
4. **Aggregates** all captions into a rich, object-aware scene description
5. **Speaks** the result aloud via **Google Text-to-Speech (gTTS)**

The result is a narration that captures individual entities and their relationships ÃÂ¢ÃÂÃÂ far richer than a single global description.

---

## Architecture

```
Input Image
    ÃÂ¢ÃÂÃÂ
    ÃÂ¢ÃÂÃÂ¼
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ               Vision2Voice Pipeline                  ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ                                                      ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ               ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂ  VGG16   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¶ÃÂ¢ÃÂÃÂ  LSTM + Beam     ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¶ Base CaptionÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂ  (fc2)   ÃÂ¢ÃÂÃÂ    ÃÂ¢ÃÂÃÂ  Search          ÃÂ¢ÃÂÃÂ               ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ               ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂ                                              ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ               ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂ YOLOv8  ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¶ÃÂ¢ÃÂÃÂ  Crop N Objects  ÃÂ¢ÃÂÃÂ               ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂ Detect  ÃÂ¢ÃÂÃÂ    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¬ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ               ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ  ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ            ÃÂ¢ÃÂÃÂ                          ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ                    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¼ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ                   ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ                    ÃÂ¢ÃÂÃÂ VGG16+LSTM ÃÂ¢ÃÂÃÂ ÃÂÃÂ N captions       ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ                    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¬ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ                   ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ                          ÃÂ¢ÃÂÃÂ                          ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ              ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¼ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ             ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ              ÃÂ¢ÃÂÃÂ  Caption Aggregation   ÃÂ¢ÃÂÃÂ             ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂ              ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¬ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ             ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¼ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ
                           ÃÂ¢ÃÂÃÂ
                    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¼ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ
                    ÃÂ¢ÃÂÃÂ    gTTS     ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ¶ ÃÂ°ÃÂÃÂÃÂ MP3 Audio
                    ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Feature Extraction | VGG16 (ImageNet) | 4096-dim visual features |
| Caption Generation | LSTM + Beam Search (width=10) | Natural language description |
| Object Detection | YOLOv8-nano | Bounding box detection |
| Speech Synthesis | Google TTS | Audio narration |
| UI | Streamlit | Interactive web dashboard |

---

## Project Structure

```
vision2voice/
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ .github/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ workflows/
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ ci.yml               # Lint + test on every push/PR
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ docker-build.yml     # Build & push Docker image on release
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ src/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ vision2voice/            # Core Python package
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ __init__.py
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ predictor.py         # Vision2VoicePredictor (VGG16 + LSTM + YOLO)
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ audio.py             # TextToSpeechEngine (gTTS wrapper)
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ utils/
ÃÂ¢ÃÂÃÂ           ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ __init__.py
ÃÂ¢ÃÂÃÂ           ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ image_utils.py   # Image loading & pre-processing helpers
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ app/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ streamlit_app.py         # Streamlit dashboard entry point
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ notebooks/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ 01_model_training.ipynb  # Original research & training notebook
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ models/                      # ÃÂ¢ÃÂÃÂ ÃÂ¯ÃÂ¸ÃÂ Place pre-trained weights here (not committed)
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ .gitkeep
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ data/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ README.md                # Dataset download instructions
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ samples/                 # Sample images for quick testing
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ outputs/                     # Generated audio & temp files (git-ignored)
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ .gitkeep
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ tests/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ conftest.py              # Shared pytest fixtures
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ unit/                    # Fast, dependency-free unit tests
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ test_image_utils.py
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ test_audio.py
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ test_predictor.py
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ integration/             # Full-pipeline tests (requires model weights)
ÃÂ¢ÃÂÃÂ       ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ test_pipeline.py
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ docs/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ architecture.md          # Detailed system design
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ api_reference.md         # Module-level API docs
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ dataset.md               # Dataset details & pre-processing
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ configs/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ default.yaml             # Runtime configuration
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ scripts/
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ run.sh                   # Linux/macOS launcher
ÃÂ¢ÃÂÃÂ   ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ run.bat                  # Windows launcher
ÃÂ¢ÃÂÃÂ
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ .flake8                      # Linter config
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ .gitignore
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ .pre-commit-config.yaml      # Pre-commit hooks
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ Dockerfile                   # Multi-stage Docker build
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ docker-compose.yml
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ Makefile                     # Developer shortcuts
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ pyproject.toml               # Package metadata & tool config
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ requirements.txt             # Runtime dependencies
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ requirements-dev.txt         # Dev/test dependencies
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ README.md
```

---

## Quickstart

### Prerequisites
- Python 3.9 ÃÂ¢ÃÂÃÂ 3.11
- Git
- (Optional) CUDA-capable GPU for faster inference

### 1. Clone the repo

```bash
git clone https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech.git
cd Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech
```

### 2. Install dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
```

### 3. Add model weights

Download the pre-trained weights (see [Dataset & Model Weights](#dataset--model-weights)) and place them in the `models/` directory:

```
models/
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ modelConcat_1_89.h5
ÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂÃÂ¢ÃÂÃÂ caption_train_tokenizer.pkl
```

### 4. Run the app

**Using Make (recommended):**
```bash
make run
```

**Using the launch script:**
```bash
bash scripts/run.sh          # Linux/macOS
scripts\run.bat              # Windows
```

**Manually:**
```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Docker Deployment

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

The app is available at http://localhost:8501. Mount your model weights:

```yaml
# docker-compose.yml (already configured)
volumes:
  - ./models:/app/models:ro
```

---

## Dataset & Model Weights

The model was trained on the **Flickr8k dataset** (8,000 images with 5 captions each).

| Resource | Link |
|----------|------|
| Training Dataset (Google Drive) | [Download](https://drive.google.com/drive/folders/1Q_NAnfN1bQWna2wY7wT8DinYlJe0Dtuk?usp=sharing) |
| Pre-trained Weights | Contact the author |

See [`docs/dataset.md`](docs/dataset.md) for full dataset description and pre-processing steps.

---

## Development

### Install dev dependencies and pre-commit hooks

```bash
make install-dev
```

### Common developer commands

| Command | Description |
|---------|-------------|
| `make run` | Launch the Streamlit dashboard |
| `make test` | Run unit tests with coverage |
| `make test-all` | Run unit + integration tests |
| `make lint` | Run flake8 linter |
| `make format` | Auto-format with black + isort |
| `make type-check` | Run mypy type checker |
| `make docker-build` | Build Docker image |
| `make clean` | Remove generated artefacts |

### Running tests manually

```bash
# Unit tests only (no weights required)
PYTHONPATH=src pytest tests/unit -v

# All tests (integration requires model weights in models/)
PYTHONPATH=src pytest tests/ -v
```

---

## API Reference

### `Vision2VoicePredictor`

```python
from vision2voice.predictor import Vision2VoicePredictor

predictor = Vision2VoicePredictor(models_dir="models/")

# Check if weights were loaded
if predictor.ready:
    base_caption, ensemble_caption = predictor.analyze_full_image("photo.jpg")
```

### `TextToSpeechEngine`

```python
from vision2voice.audio import TextToSpeechEngine

tts = TextToSpeechEngine(language="en", output_dir="outputs/")
audio_path = tts.synthesize("A dog is playing in the park.")
```

See [`docs/api_reference.md`](docs/api_reference.md) for the full API.

---

## Contributing

1. Fork the repo and create your feature branch: `git checkout -b feature/your-feature`
2. Install dev dependencies: `make install-dev`
3. Make your changes and ensure tests pass: `make test`
4. Ensure code is formatted: `make format && make lint`
5. Open a Pull Request targeting `main`

---

## License

MIT ÃÂÃÂ© [Aniruddh Kulkarni](https://github.com/Aniruddh-11-stack)
