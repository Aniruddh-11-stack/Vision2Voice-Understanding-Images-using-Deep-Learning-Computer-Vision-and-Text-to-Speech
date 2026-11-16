"""
Vision2Voice Predictor Module.

This module contains the core Vision2VoicePredictor class which orchestrates
the full image-understanding pipeline:
  - VGG16 feature extraction
  - LSTM-based caption generation with Beam Search
  - YOLOv8 object detection and crop analysis
"""

import logging
import os
import pickle
import re
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Keras is imported lazily (see _import_keras): YOLO/PyTorch must run once BEFORE TensorFlow is
# imported, otherwise the process can abort with "LLVM ERROR: inconsistency in registered
# CommandLine options". The names exist at module level so tests can patch them.
VGG16 = preprocess_input = Model = load_model = None


def _import_keras() -> None:
    global VGG16, preprocess_input, Model, load_model
    if VGG16 is None:
        from keras.applications.vgg16 import VGG16 as _vgg
        VGG16 = _vgg
    if preprocess_input is None:
        from keras.applications.vgg16 import preprocess_input as _pp
        preprocess_input = _pp
    if Model is None:
        from keras.models import Model as _model
        Model = _model
    if load_model is None:
        from keras.models import load_model as _lm
        load_model = _lm

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when model weights are not found in the expected directory."""
    pass


class _TokenizerState:
    """Attributes of a pickled Keras 2 Tokenizer, readable without Keras 2."""
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _TokenizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "preprocessing" in module and name == "Tokenizer":
            return _TokenizerState
        return super().find_class(module, name)


def _pad(seqs: List[List[int]], maxlen: int) -> np.ndarray:
    """Same as keras pad_sequences(padding='pre', truncating='pre')."""
    out = np.zeros((len(seqs), maxlen), dtype="int32")
    for i, s in enumerate(seqs):
        s = s[-maxlen:]
        if s:
            out[i, maxlen - len(s):] = s
    return out


class Vision2VoicePredictor:
    MODEL_FILENAME = "modelConcat_1_89.h5"
    TOKENIZER_FILENAME = "caption_train_tokenizer.pkl"

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.ready: bool = False
        self.max_length: int = 33
        self.vocab_size: int = 7506
        self.beam_width: int = 10
        logger.info("Loading YOLOv8 object detector...")
        # YOLO (PyTorch) before TensorFlow work: the reverse order can abort with
        # "LLVM ERROR: inconsistency in registered CommandLine options".
        self.yolo = YOLO("yolov8n.pt")
        try:  # one tiny warm-up run so PyTorch is fully initialised first
            self.yolo.predict(np.zeros((64, 64, 3), dtype="uint8"), verbose=False)
        except Exception as exc:  # e.g. a mocked YOLO in unit tests
            logger.debug("YOLO warm-up skipped: %s", exc)
        _import_keras()
        logger.info("Initializing VGG16 feature extractor...")
        base_model = VGG16(include_top=True)
        self.feature_extractor: Model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)
        self._load_weights()

    # ------------------------------------------------------------------ loading
    def _load_weights(self) -> None:
        t = os.path.join(self.models_dir, self.TOKENIZER_FILENAME)
        m = os.path.join(self.models_dir, self.MODEL_FILENAME)
        if not os.path.exists(t) or not os.path.exists(m):
            logger.warning("Model weights not found in %s", self.models_dir)
            return
        with open(t, "rb") as fh:
            tok = _TokenizerUnpickler(fh).load()
        self.word_index = tok.word_index
        self.index_word = {int(k): v for k, v in tok.index_word.items()}
        self._filters = re.compile("[" + re.escape(getattr(tok, "filters", "")) + "]") if getattr(tok, "filters", "") else None
        self.start_id = self.word_index["start"]
        self.end_id = self.word_index["end"]
        self.model = self._load_decoder(m)
        self.ready = True

    def _load_decoder(self, path: str):
        try:
            return load_model(path, compile=False)
        except Exception:  # Keras 3 can't read this Keras 2 file directly
            logger.info("Rebuilding decoder architecture to load Keras 2 weights")
            from keras import Input
            from keras.layers import LSTM, Dense, Dropout, Embedding, concatenate
            import h5py
            i1 = Input(shape=(4096,))
            a = Dense(256, activation="relu")(Dropout(0.5)(i1))
            i2 = Input(shape=(self.max_length,))
            b = LSTM(256)(Dropout(0.5)(Embedding(self.vocab_size, 50, trainable=False)(i2)))
            out = Dense(self.vocab_size, activation="softmax")(Dense(256, activation="relu")(concatenate([a, b])))
            model = Model([i1, i2], out)
            with h5py.File(path, "r") as f:
                g = f["model_weights"] if "model_weights" in f else f
                saved = []
                for n in g.attrs["layer_names"]:
                    n = n.decode() if isinstance(n, bytes) else n
                    ws = [np.array(g[n][w.decode() if isinstance(w, bytes) else w]) for w in g[n].attrs["weight_names"]]
                    if ws:
                        saved.append(ws)
            for layer in [l for l in model.layers if l.weights]:
                shapes = [tuple(w.shape) for w in layer.get_weights()]
                k = next(i for i, ws in enumerate(saved) if [tuple(w.shape) for w in ws] == shapes)
                layer.set_weights(saved.pop(k))
            return model

    # ---------------------------------------------------------------- inference
    def _extract_feature(self, np_img: np.ndarray) -> np.ndarray:
        """RGB array -> 4096-d VGG16 fc2 features (224x224, like keras load_img)."""
        x = cv2.resize(np_img, (224, 224), interpolation=cv2.INTER_NEAREST).astype("float32")
        x = preprocess_input(np.expand_dims(x, 0))
        return self.feature_extractor.predict(x, verbose=0)

    def beam_search(self, photo: np.ndarray, beam_width: int = None) -> List[Tuple[str, float]]:
        """All finished beams as (caption, length-normalised log-prob), best first.

        Mirrors generate_caption_beam in Project.ipynb, batching every live beam
        into one decoder call per step.
        """
        k = beam_width or self.beam_width
        beams = [([self.start_id], 0.0, False)]
        for _ in range(self.max_length):
            live = [b for b in beams if not b[2]]
            if not live:
                break
            probs = self.model.predict(
                [np.repeat(photo, len(live), axis=0), _pad([b[0] for b in live], self.max_length)], verbose=0)
            candidates = [b for b in beams if b[2]]
            for (seq, logp, _), p in zip(live, probs):
                for idx in np.argsort(p)[-k:]:
                    idx = int(idx)
                    if idx == 0:
                        continue
                    candidates.append((seq + [idx], logp + float(np.log(p[idx] + 1e-12)), idx == self.end_id))
            beams = sorted(candidates, key=lambda b: b[1] / len(b[0]), reverse=True)[:k]
        results = []
        for seq, logp, _ in sorted(beams, key=lambda b: b[1] / len(b[0]), reverse=True):
            words = [self.index_word.get(i, "") for i in seq[1:] if i not in (self.start_id, self.end_id)]
            results.append((" ".join(w for w in words if w), logp / len(seq)))
        return results

    def generate_best_caption(self, np_img: np.ndarray) -> str:
        """Full caption for an RGB image array (previously returned only the first word)."""
        return self.beam_search(self._extract_feature(np_img))[0][0]

    def analyze_full_image(self, image_path: str) -> Tuple[str, str]:
        """Returns (caption of the whole image, whole caption + captions of each detected object)."""
        if not self.ready:
            raise ModelNotReadyError("Model weights not found.")
        base_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        base_cap = self.generate_best_caption(base_img)
        results = self.yolo.predict(image_path, verbose=False)
        crops: List[str] = []
        pil_image = Image.open(image_path).convert("RGB")
        for box in results[0].boxes:
            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            if x2 - x1 < 32 or y2 - y1 < 32:  # tiny crops give meaningless captions
                continue
            crops.append(self.generate_best_caption(np.array(pil_image.crop((x1, y1, x2, y2)))))
        return base_cap, " ".join([base_cap] + crops)
