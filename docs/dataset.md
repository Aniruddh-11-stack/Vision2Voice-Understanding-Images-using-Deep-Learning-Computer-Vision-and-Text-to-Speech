# Vision2Voice — Dataset Guide

## Training Dataset: Flickr8k

| Attribute | Value |
|-----------|-------|
| Images | 8,092 photographs |
| Captions per image | 5 (human-annotated) |
| Total caption-image pairs | 40,460 |
| Vocabulary (after cleaning) | ~7,506 unique tokens |
| Split | 6,000 train / 1,000 dev / 1,000 test |

### Download

The full dataset (images + captions) used to train Vision2Voice is available on Google Drive:

🔗 [Download Dataset](https://drive.google.com/drive/folders/1Q_NAnfN1bQWna2wY7wT8DinYlJe0Dtuk?usp=sharing)

The drive folder contains:
- `Images/` — 8,092 JPEG photographs
- `captions.txt` — image-caption mappings
- `caption_train_tokenizer.pkl` — pre-fitted Keras tokenizer
- `modelConcat_1_89.h5` — the trained VGG16+LSTM caption model

### Pre-processing Pipeline (from training notebook)

1. **Load captions** from `captions.txt`; strip punctuation and lowercase.
2. **Filter vocabulary** — words appearing fewer than 5 times are replaced with `<unk>`.
3. **Add sequence tokens** — `<START>` prefix and `end` suffix to every caption.
4. **Tokenize** with `keras.preprocessing.text.Tokenizer`; save as `.pkl`.
5. **Extract VGG16 features** — pass every training image through VGG16 (fc2 layer);
   store as a `{filename: feature_vector}` dictionary.
6. **Train LSTM** — encoder-decoder with concat merge; 20 epochs, Adam optimizer,
   categorical cross-entropy loss; best checkpoint saved as `.h5`.

### File Layout after Download

```
models/
├── modelConcat_1_89.h5          ← LSTM caption model
└── caption_train_tokenizer.pkl  ← Fitted tokenizer

data/
└── samples/                     ← (Optional) a few test images
```

### Adding Sample Images

Place a few test images in `data/samples/` for quick local testing without
needing to upload through the Streamlit UI:

```bash
cp ~/Downloads/my_photo.jpg data/samples/
PYTHONPATH=src python - <<'EOF'
from vision2voice.predictor import Vision2VoicePredictor
p = Vision2VoicePredictor("models/")
base, ensemble = p.analyze_full_image("data/samples/my_photo.jpg")
print("Base:", base)
print("Ensemble:", ensemble)
EOF
```
