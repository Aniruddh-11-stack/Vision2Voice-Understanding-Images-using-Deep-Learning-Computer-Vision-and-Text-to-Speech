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
from pickle import load
from typing import Tuple, List, Optional

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when model weights are not found in the expected directory."""
    pass


class Vision2VoicePredictor:
    MODEL_FILENAME = "modelConcat_1_89.h5"
    TOKENIZER_FILENAME = "caption_train_tokenizer.pkl"

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.ready: bool = False
        self.max_length: int = 33
        self.vocab_size: int = 7506
        self.beam_width: int = 10
        logger.info("Initializing VGG16 feature extractor...")
        base_model = VGG16(include_top=True)
        self.feature_extractor: Model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)
        self._load_weights()
        logger.info("Loading YOLOv8 object detector...")
        self.yolo = YOLO("yolov8n.pt")

    def _load_weights(self) -> None:
        t = os.path.join(self.models_dir, self.TOKENIZER_FILENAME)
        m = os.path.join(self.models_dir, self.MODEL_FILENAME)
        if not os.path.exists(t) or not os.path.exists(m): return
        with open(t, "rb") as fh: self.tokenizer = load(fh)
        self.model = load_model(m)
        self.ready = True

    def _extract_feature(self, np_img):
        import keras.preprocessing.image as kpi
        x = kpi.img_to_array(cv2.resize(np_img, (224,224)))
        x = preprocess_input(np.expand_dims(x, 0))
        return self.feature_extractor.predict(x, verbose=0)

    def generate_best_caption(self, np_img):
        from keras.preprocessing.sequence import pad_sequences
        photo = self._extract_feature(np_img)
        seq = pad_sequences([self.tokenizer.texts_to_sequences(["<START>"])[0]], maxlen=self.max_length)
        out = np.squeeze(self.model.predict([photo, seq], verbose=0))
        top = np.argsort(out)[-self.beam_width:]
        return self.tokenizer.index_word.get(top[-1], "")

    def analyze_full_image(self, image_path):
        if not self.ready: raise ModelNotReadyError("Model weights not found.")
        base_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        base_cap = self.generate_best_caption(base_img)
        results = self.yolo.predict(image_path, verbose=False)
        crops = []
        pil_image = Image.open(image_path)
        for box in results[0].boxes:
            x1,y1,x2,y2 = [round(v) for v in box.xyxy[0].tolist()]
            crop = np.array(pil_image.crop((x1,y1,x2,y2)))
            if crop.ndim==2: crop=cv2.cvtColor(crop,cv2.COLOR_GRAY2RGB)
            elif crop.shape[2]==4: crop=cv2.cvtColor(crop,cv2.COLOR_RGBA2RGB)
            crops.append(self.generate_best_caption(crop))
        return base_cap, " ".join([base_cap]+crops)
