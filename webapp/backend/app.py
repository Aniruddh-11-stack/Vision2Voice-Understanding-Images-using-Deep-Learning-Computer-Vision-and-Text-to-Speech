"""Vision2Voice web backend.

image -> VGG16 (fc2) + LSTM beam-search caption  (the project's trained model)
      -> YOLOv8-nano objects + captions of the largest object crops
      -> text_polish (rules + optional T5 grammar fix) -> translation -> gTTS / pyttsx3
Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from text_polish import compose

MAX_IMAGE_MB = 10
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ACCENTS = {"in": "co.in", "us": "com", "uk": "co.uk", "au": "com.au"}
DETAIL = {"brief": {"beam_width": 3, "crops": False}, "rich": {"beam_width": 10, "crops": True}}

_state: dict = {"captioner": None, "error": None}
_lock = threading.Lock()  # Keras/YOLO models aren't thread-safe; serialise inference


def get_captioner():
    if _state["captioner"] is None:
        with _lock:
            if _state["captioner"] is None:
                # Load YOLO (PyTorch) and run it once BEFORE TensorFlow starts; the reverse order
                # can crash with "LLVM ERROR: inconsistency in registered CommandLine options".
                from ultralytics import YOLO
                yolo = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))
                yolo.predict(Image.new("RGB", (64, 64)), verbose=False)
                from captioner import Captioner
                c = Captioner()
                c._yolo = yolo
                _state["captioner"] = c
    return _state["captioner"]


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("PRELOAD", "1") == "1":
        try:
            get_captioner()
        except Exception as e:  # keep the server up so /api/health can explain the problem
            _state["error"] = str(e)
            print(f"[error] model load failed: {e}")
    yield


app = FastAPI(title="Vision2Voice API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_image(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_IMAGE_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file couldn't be read as an image.")
    img.thumbnail((1280, 1280))
    return img


def translate(text: str, lang: str) -> str:
    if lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=lang).translate(text)
    except Exception:
        return text


def synthesize(text: str, lang: str = "en", accent: str = "in", slow: bool = False) -> tuple[bytes, str]:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, tld=ACCENTS.get(accent, "com"), slow=slow).write_to_fp(buf)
        return buf.getvalue(), "audio/mpeg"
    except Exception:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 130 if slow else 170)
        path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.wav"
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        data = path.read_bytes()
        path.unlink(missing_ok=True)
        return data, "audio/wav"


def audio_payload(text: str, lang: str, accent: str, slow: bool) -> dict:
    try:
        audio, mime = synthesize(text, lang, accent, slow)
    except Exception as e:
        print(f"[warn] speech failed: {e}")
        return {"audio_mime": None, "audio_base64": None}
    return {"audio_mime": mime, "audio_base64": base64.b64encode(audio).decode()}


@app.get("/api/health")
def health():
    ok = _state["captioner"] is not None
    return {"status": "ok" if ok else ("error" if _state["error"] else "loading"), "error": _state["error"],
            "device": "CPU", "models": {"encoder": "VGG16 (ImageNet, fc2)", "decoder": "LSTM (Flickr8k)",
                                       "detector": os.getenv("YOLO_MODEL", "yolov8n.pt"), "tts": "gTTS"}}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    mode: str = Form("describe"),          # describe | live
    detail: str = Form("rich"),            # rich | brief
    lang: str = Form("en"),
    accent: str = Form("in"),
    slow: bool = Form(False),
    speak: bool = Form(True),
):
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "Choose an image file.")
    img = load_image(await image.read())
    try:
        cap = get_captioner()
    except Exception as e:
        raise HTTPException(503, f"Models couldn't be loaded: {e}")
    t0 = time.perf_counter()
    opts = DETAIL["brief"] if mode == "live" else DETAIL.get(detail, DETAIL["rich"])
    with _lock:
        raw = cap.analyze(img, beam_width=opts["beam_width"], crops=opts["crops"])
    vision_ms = int((time.perf_counter() - t0) * 1000)

    result = compose(raw["caption"], [d["label"] for d in raw["detections"]], raw["crop_captions"], mode)
    spoken = translate(result["narration"], lang)
    payload = {**result, "raw_caption": raw["caption"], "raw_crop_captions": raw["crop_captions"],
               "detections": raw["detections"], "spoken_text": spoken, "lang": lang, "image_size": img.size}
    if speak:
        payload.update(audio_payload(spoken, lang, accent, slow))
    payload["timings_ms"] = {"vision": vision_ms, "total": int((time.perf_counter() - t0) * 1000)}
    return payload


@app.post("/api/tts")
async def tts(text: str = Form(...), lang: str = Form("en"), accent: str = Form("in"), slow: bool = Form(False)):
    return audio_payload(text[:2000], lang, accent, slow)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
