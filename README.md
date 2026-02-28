# Vision2Voice Ã°ÂÂÂÃ¯Â¸ÂÃ°ÂÂÂ

> AI-powered image understanding pipeline that bridges visual perception and spoken language Ã¢ÂÂ combining deep CNNs, sequence models, object detection, and text-to-speech into a single cohesive system.

[![CI](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/ci.yml/badge.svg)](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/ci.yml)
[![Docker](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Aniruddh-11-stack/Vision2Voice-Understanding-Images-using-Deep-Learning-Computer-Vision-and-Text-to-Speech/actions/workflows/docker-build.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Vision2Voice App Demo](docs/screenshots/app_demo.png)

## 🏗️ Model Architecture

The model uses a dual-input architecture combining:
- **Image encoder**: VGG16 (4096-dim features) → Dense(256)
- **Language model**: Embedding + LSTM (256-dim)
- **Fusion**: Concatenate → Dense(256) → Dense(vocab_size, softmax)

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

The result is a narration that captures individual entities and their relationships Ã¢ÂÂ far richer than a single global description.

---

## Architecture

```
Input Image
    Ã¢ÂÂ
    Ã¢ÂÂ¼
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
Ã¢ÂÂ               Vision2Voice Pipeline                  Ã¢ÂÂ
Ã¢ÂÂ                                                      Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ               Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂ  VGG16   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¶Ã¢ÂÂ  LSTM + Beam     Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¶ Base CaptionÃ¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂ  (fc2)   Ã¢ÂÂ    Ã¢ÂÂ  Search          Ã¢ÂÂ               Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ               Ã¢ÂÂ
Ã¢ÂÂ       Ã¢ÂÂ                                              Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ               Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂ YOLOv8  Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¶Ã¢ÂÂ  Crop N Objects  Ã¢ÂÂ               Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂ Detect  Ã¢ÂÂ    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¬Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ               Ã¢ÂÂ
Ã¢ÂÂ  Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ            Ã¢ÂÂ                          Ã¢ÂÂ
Ã¢ÂÂ                    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¼Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ                   Ã¢ÂÂ
Ã¢ÂÂ                    Ã¢ÂÂ VGG16+LSTM Ã¢ÂÂ ÃÂ N captions       Ã¢ÂÂ
Ã¢ÂÂ                    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¬Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ                   Ã¢ÂÂ
Ã¢ÂÂ                          Ã¢ÂÂ                          Ã¢ÂÂ
Ã¢ÂÂ              Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¼Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ             Ã¢ÂÂ
Ã¢ÂÂ              Ã¢ÂÂ  Caption Aggregation   Ã¢ÂÂ             Ã¢ÂÂ
Ã¢ÂÂ              Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¬Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ             Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¼Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
                           Ã¢ÂÂ
                    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¼Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
                    Ã¢ÂÂ    gTTS     Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ¶ Ã°ÂÂÂ MP3 Audio
                    Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
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
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ .github/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ workflows/
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ ci.yml               # Lint + test on every push/PR
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ docker-build.yml     # Build & push Docker image on release
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ src/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ vision2voice/            # Core Python package
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ __init__.py
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ predictor.py         # Vision2VoicePredictor (VGG16 + LSTM + YOLO)
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ audio.py             # TextToSpeechEngine (gTTS wrapper)
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ utils/
Ã¢ÂÂ           Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ __init__.py
Ã¢ÂÂ           Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ image_utils.py   # Image loading & pre-processing helpers
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ app/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ streamlit_app.py         # Streamlit dashboard entry point
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ notebooks/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ 01_model_training.ipynb  # Original research & training notebook
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ models/                      # Ã¢ÂÂ Ã¯Â¸Â Place pre-trained weights here (not committed)
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ .gitkeep
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ data/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ README.md                # Dataset download instructions
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ samples/                 # Sample images for quick testing
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ outputs/                     # Generated audio & temp files (git-ignored)
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ .gitkeep
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ tests/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ conftest.py              # Shared pytest fixtures
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ unit/                    # Fast, dependency-free unit tests
Ã¢ÂÂ   Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ test_image_utils.py
Ã¢ÂÂ   Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ test_audio.py
Ã¢ÂÂ   Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ test_predictor.py
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ integration/             # Full-pipeline tests (requires model weights)
Ã¢ÂÂ       Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ test_pipeline.py
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ docs/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ architecture.md          # Detailed system design
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ api_reference.md         # Module-level API docs
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ dataset.md               # Dataset details & pre-processing
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ configs/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ default.yaml             # Runtime configuration
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ scripts/
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ run.sh                   # Linux/macOS launcher
Ã¢ÂÂ   Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ run.bat                  # Windows launcher
Ã¢ÂÂ
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ .flake8                      # Linter config
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ .gitignore
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ .pre-commit-config.yaml      # Pre-commit hooks
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Dockerfile                   # Multi-stage Docker build
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ docker-compose.yml
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ Makefile                     # Developer shortcuts
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ pyproject.toml               # Package metadata & tool config
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ requirements.txt             # Runtime dependencies
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ requirements-dev.txt         # Dev/test dependencies
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ README.md
```

---

## Quickstart

### Prerequisites
- Python 3.9 Ã¢ÂÂ 3.11
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
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ modelConcat_1_89.h5
Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂ caption_train_tokenizer.pkl
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

MIT ÃÂ© [Aniruddh Kulkarni](https://github.com/Aniruddh-11-stack)
