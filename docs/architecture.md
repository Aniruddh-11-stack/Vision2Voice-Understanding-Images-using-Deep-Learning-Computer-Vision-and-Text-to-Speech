# Vision2Voice — System Architecture

## 1. Overview

Vision2Voice is a multi-stage AI pipeline that takes a single image as input
and produces both a textual description and an audio narration of its content.
The system is deliberately modular: each stage is implemented as an
independently testable Python class so that components can be swapped or
upgraded without touching the rest of the pipeline.

## 2. Component Breakdown

### 2.1 Feature Extractor — VGG16 (CNN)

- **Library:** Keras / TensorFlow
- **Pre-training:** ImageNet
- **Layer used:** `fc2` (4096-dimensional output vector)
- **Input:** 224 × 224 RGB image (resized from any resolution)
- **Purpose:** Transforms a raw pixel grid into a rich semantic vector that
  captures objects, textures, and scene context.

### 2.2 Caption Generator — LSTM with Beam Search

- **Architecture:** VGG16 features → embedding → LSTM → dense (vocab_size)
- **Training corpus:** Flickr8k (8,000 images × 5 captions)
- **Vocabulary size:** 7,506 tokens
- **Max sequence length:** 33 tokens
- **Decoding strategy:** Beam Search (width = 10) for higher-quality captions
  vs. greedy decoding.

The model is a concat-merge encoder-decoder:
1. VGG16 features are projected via a `Dense` layer into the caption space.
2. The caption prefix is embedded and encoded by the LSTM.
3. Both representations are concatenated and decoded token-by-token.

### 2.3 Object Detector — YOLOv8

- **Variant:** YOLOv8-nano (fastest; ~3 ms/image on CPU)
- **Purpose:** Detect bounding boxes for every salient entity.
- **Output used:** `xyxy` bounding boxes (absolute pixel coordinates).
- Cropped sub-images are passed individually through the VGG16 + LSTM pipeline,
  yielding one caption per detected object.

### 2.4 Caption Aggregator

A simple join of all captions (base + one per YOLO crop) into a single
flowing description string. This design is intentionally straightforward to
allow future replacement with a seq2seq fusion model.

### 2.5 TTS — gTTS

- **Provider:** Google Text-to-Speech API (requires internet access).
- **Output:** MP3 file saved to the configured `outputs/` directory.
- The `TextToSpeechEngine` wrapper supports in-memory synthesis
  (`synthesize_to_bytes`) for streaming use cases.

## 3. Data Flow

```
Image File (JPEG/PNG)
    │
    ├─────────────────────────────────────────────────────┐
    │                                                     │
    ▼                                                     ▼
VGG16 (224×224 resize)                            YOLOv8 Detection
    │                                                     │
4096-dim vector                                   N bounding boxes
    │                                                     │
LSTM Beam Search ──────────────────────── crop₁ … cropₙ → VGG16 → LSTM
    │                                                     │
base_caption                                 [crop_cap₁, …, crop_capₙ]
    │                                                     │
    └─────────────── Caption Aggregator ─────────────────┘
                              │
                    ensemble_caption (str)
                                │
                             gTTS API
                                │
                         output_speech.mp3
```

## 4. Module Dependency Graph

```
app/streamlit_app.py
    ├── vision2voice.predictor.Vision2VoicePredictor
    │        ├── keras (VGG16, LSTM model)
    │        ├── ultralytics (YOLO)
    │       └── vision2voice.utils.image_utils
    └── vision2voice.audio.TextToSpeechEngine
            └── gtts
```

## 5. Performance Notes

| Hardware | Approx. inference time per image |
|----------|----------------------------------|
| CPU only | 15 – 60 s (beam_width=10, 3+ YOLO crops) |
| GPU (CUDA) | 2 – 8 s |

Reducing `beam_width` in `configs/default.yaml` gives a significant speedup
at modest quality cost.

## 6. Future Improvements

- Replace LSTM with a Transformer-based captioner (BLIP, LLaVA).
- Add async inference to keep the Streamlit UI responsive.
- Implement caption fusion with an LLM for grammatically coherent aggregation.
- Add streaming TTS for low-latency audio playback.
- Expose a FastAPI REST endpoint alongside the Streamlit UI.
