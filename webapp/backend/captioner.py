"""Vision2Voice captioner: the project's own trained VGG16 + LSTM model, plus YOLOv8 crops.

Matches the training notebook (Project.ipynb):
- VGG16 (ImageNet) `fc2` layer -> 4096-d feature; images resized to 224x224 like `load_img`
- LSTM decoder `modelConcat_1_89.h5`, max length 33, vocab 7506, tokens `start` ... `end`
- Beam search (default width 10) with length-normalised log-probability
- YOLOv8-nano detects objects; each crop is captioned too
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.getenv("V2V_WEIGHTS_DIR", REPO_ROOT / "capstone_data_2"))
MODEL_FILE = os.getenv("V2V_MODEL_FILE", "Saved_Models3/modelConcat_1_89.h5")
TOKENIZER_FILE = os.getenv("V2V_TOKENIZER_FILE", "caption_train_tokenizer.pkl")
MAX_LEN = 33


class _TokenizerState:
    """Holds a pickled Keras 2 Tokenizer's attributes without needing Keras 2 installed."""
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "preprocessing" in module and name == "Tokenizer":
            return _TokenizerState
        return super().find_class(module, name)


class Vocabulary:
    def __init__(self, path: Path):
        with open(path, "rb") as fh:
            st = _Unpickler(fh).load()
        self.word_index: dict[str, int] = st.word_index
        self.index_word: dict[int, str] = {int(k): v for k, v in st.index_word.items()}
        self._filters = re.compile("[" + re.escape(st.filters) + "]")
        self.start, self.end = self.word_index["start"], self.word_index["end"]

    def encode(self, text: str) -> list[int]:
        words = self._filters.sub(" ", text.lower()).split()
        return [self.word_index[w] for w in words if w in self.word_index]


def _pad(seqs: list[list[int]], maxlen: int = MAX_LEN) -> np.ndarray:
    out = np.zeros((len(seqs), maxlen), dtype="int32")  # Keras pad_sequences: 'pre' padding/truncating
    for i, s in enumerate(seqs):
        s = s[-maxlen:]
        out[i, maxlen - len(s):] = s
    return out


def build_decoder(vocab_size: int = 7506, max_len: int = MAX_LEN):
    """Same architecture as `define_model_concat` in Project.ipynb."""
    from keras import Input, Model
    from keras.layers import LSTM, Dense, Dropout, Embedding, concatenate
    i1 = Input(shape=(4096,))
    a = Dense(256, activation="relu")(Dropout(0.5)(i1))
    i2 = Input(shape=(max_len,))
    b = LSTM(256)(Dropout(0.5)(Embedding(vocab_size, 50, trainable=False)(i2)))
    out = Dense(vocab_size, activation="softmax")(Dense(256, activation="relu")(concatenate([a, b])))
    return Model([i1, i2], out)


def load_decoder(path: Path):
    """Loads the Keras 2 .h5 decoder; works on Keras 2 and Keras 3."""
    from keras.models import load_model
    try:
        return load_model(path, compile=False)
    except Exception:
        model = build_decoder()
        _load_h5_weights(model, path)
        return model


def _load_h5_weights(model, path: Path):
    """Copies weights layer by layer from a Keras 2 HDF5 file (in saved layer order)."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
        saved = []
        for n in names:
            wn = g[n].attrs["weight_names"]
            ws = [np.array(g[n][w.decode() if isinstance(w, bytes) else w]) for w in wn]
            if ws:
                saved.append(ws)
    targets = [l for l in model.layers if l.weights]
    # match by weight shapes, preserving order (Dense layers have distinct shapes here)
    remaining = list(saved)
    for layer in targets:
        shapes = [tuple(w.shape) for w in layer.get_weights()]
        for k, ws in enumerate(remaining):
            if [tuple(w.shape) for w in ws] == shapes:
                layer.set_weights(ws)
                remaining.pop(k)
                break
        else:
            raise ValueError(f"No saved weights match layer {layer.name} {shapes}")


class Captioner:
    def __init__(self, yolo_weights: str = os.getenv("YOLO_MODEL", "yolov8n.pt")):
        from keras.applications.vgg16 import VGG16
        from keras.models import Model

        self.vocab = Vocabulary(WEIGHTS_DIR / TOKENIZER_FILE)
        self.decoder = load_decoder(WEIGHTS_DIR / MODEL_FILE)
        base = VGG16(include_top=True, weights="imagenet")
        self.encoder = Model(base.input, base.get_layer("fc2").output)
        self._yolo_weights = yolo_weights
        self._yolo = None

    # ---- vision ----
    def features(self, img: Image.Image) -> np.ndarray:
        from keras.applications.vgg16 import preprocess_input
        x = np.asarray(img.convert("RGB").resize((224, 224), Image.NEAREST), dtype="float32")
        return self.encoder.predict(preprocess_input(x[None]), verbose=0)

    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self._yolo_weights)
        return self._yolo

    def detect(self, img: Image.Image, conf: float = 0.35, max_objects: int = 20) -> list[dict]:
        res = self.yolo.predict(img, conf=conf, verbose=False, max_det=max_objects)[0]
        return [{"label": res.names[int(c)], "conf": round(float(p), 3), "box": [round(float(v), 4) for v in b]}
                for b, c, p in zip(res.boxes.xyxyn.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]

    # ---- language ----
    def beam_search(self, feat: np.ndarray, beam_width: int = 10) -> tuple[str, float]:
        beams = [([self.vocab.start], 0.0, False)]
        for _ in range(MAX_LEN):
            live = [b for b in beams if not b[2]]
            if not live:
                break
            probs = self.decoder.predict([np.repeat(feat, len(live), 0), _pad([b[0] for b in live])], verbose=0)
            cand = [b for b in beams if b[2]]
            for (seq, lp, _), p in zip(live, probs):
                for idx in np.argsort(p)[-beam_width:]:
                    idx = int(idx)
                    if idx == 0:
                        continue
                    cand.append((seq + [idx], lp + float(np.log(p[idx] + 1e-12)), idx == self.vocab.end))
            beams = sorted(cand, key=lambda b: b[1] / len(b[0]), reverse=True)[:beam_width]
        seq, lp, _ = max(beams, key=lambda b: b[1] / len(b[0]))
        words = [self.vocab.index_word[i] for i in seq[1:] if i not in (self.vocab.end, self.vocab.start)]
        return " ".join(words), lp / len(seq)

    def caption(self, img: Image.Image, beam_width: int = 10) -> str:
        return self.beam_search(self.features(img), beam_width)[0]

    def caption_scored(self, img: Image.Image, beam_width: int = 10) -> tuple[str, float]:
        """Caption plus its length-normalised log-probability (higher = more confident)."""
        return self.beam_search(self.features(img), beam_width)

    def analyze(self, img: Image.Image, beam_width: int = 10, crops: bool = True,
                min_crop_px: int = 48, max_crops: int = 4) -> dict:
        """Full-image caption, detections, and captions of the largest object crops."""
        main, main_score = self.caption_scored(img, beam_width)
        dets = self.detect(img)
        crop_caps = []
        if crops:
            w, h = img.size
            ranked = sorted(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), reverse=True)
            for d in ranked[:max_crops]:
                x1, y1, x2, y2 = d["box"]
                box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
                if box[2] - box[0] < min_crop_px or box[3] - box[1] < min_crop_px:
                    continue
                text, score = self.caption_scored(img.crop(box), max(3, beam_width // 2))
                d["caption"] = text
                crop_caps.append({"label": d["label"], "caption": text, "score": round(score, 3)})
        return {"caption": main, "score": round(main_score, 3), "detections": dets, "crop_captions": crop_caps}
