# Vision2Voice — API Reference

## `vision2voice.predictor`

### `class Vision2VoicePredictor`

Full image-understanding pipeline orchestrator.

#### Constructor

```python
Vision2VoicePredictor(models_dir: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `models_dir` | `str` | Path to the directory containing model weight files. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ready` | `bool` | `True` when all weights loaded successfully. |
| `max_length` | `int` | Maximum caption token length (default: 33). |
| `vocab_size` | `int` | Tokenizer vocabulary size (default: 7506). |
| `beam_width` | `int` | Beam Search width (default: 10). |

#### Methods

##### `generate_best_caption(np_img) -> str`

Returns the single highest-probability caption for a raw RGB image array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `np_img` | `np.ndarray` | RGB image array, shape (H, W, 3). |

**Returns:** `str` — best caption string.

##### `analyze_full_image(image_path) -> Tuple[str, str]`

Runs the full Vision2Voice pipeline on a disk image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to JPEG or PNG image file. |

**Returns:** `Tuple[str, str]` — `(base_caption, ensemble_caption)`.

**Raises:**
- `ModelNotReadyError` — if model weights were not loaded.

---

## `vision2voice.audio`

### `class TextToSpeechEngine`

Wrapper around Google Text-to-Speech (gTTS).

#### Constructor

```python
TextToSpeechEngine(language="en", slow=False, output_dir=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `"en"` | BCP-47 language code. |
| `slow` | `bool` | `False` | Synthesise at reduced speed. |
| `output_dir` | `str \| None` | system temp | Directory for saved audio files. |

#### Methods

##### `synthesize(text, filename="output_speech.mp3") -> str`

Write MP3 audio to disk and return its path.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to synthesise. |
| `filename` | `str` | Output filename. |

**Returns:** `str` — absolute path to the generated MP3 file.

**Raises:**
- `ValueError` — if `text` is empty.

##### `synthesize_to_bytes(text) -> bytes`

Return MP3 audio as raw bytes without writing to disk.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to synthesise. |

**Returns:** `bytes` — raw MP3 data.

---

## `vision2voice.utils.image_utils`

### `validate_image_path(image_path) -> Path`

Validate existence and extension of an image path.

**Raises:** `FileNotFoundError`, `ValueError`.

### `load_image_rgb(image_path) -> np.ndarray`

Load an image from disk as an RGB NumPy array (H, W, 3).

**Raises:** `FileNotFoundError`, `IOError`.

### `resize_and_pad(img, target_size=(224, 224), pad_colour=(0,0,0)) -> np.ndarray`

Letterbox-resize an image to `target_size` while preserving aspect ratio.
