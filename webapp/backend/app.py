"""Vision to Voice backend.

image -> BLIP caption (+ BLIP VQA) + YOLOv8 objects + EasyOCR text
      -> text_polish (rules + T5 grammar fix) -> translation -> gTTS / pyttsx3 audio
Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from text_polish import answer_sentence, compose

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
VQA_MODEL = os.getenv("VQA_MODEL", "Salesforce/blip-vqa-base")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
MAX_IMAGE_MB = 10
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ACCENTS = {"in": "co.in", "us": "com", "uk": "co.uk", "au": "com.au"}

@asynccontextmanager
async def lifespan(_app):
    if os.getenv("PRELOAD", "1") == "1":  # load models once at startup
        caption_model(); vqa_model(); safe(yolo_model); safe(ocr_reader)
    yield


app = FastAPI(title="Vision to Voice API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- models (lazy, cached) ----------
def _to_device(inputs):
    return {k: v.to(DEVICE, DTYPE) if v.dtype.is_floating_point else v.to(DEVICE) for k, v in inputs.items()}


@lru_cache(maxsize=1)
def caption_model():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    return (BlipProcessor.from_pretrained(CAPTION_MODEL),
            BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL, torch_dtype=DTYPE).to(DEVICE).eval())


@lru_cache(maxsize=1)
def vqa_model():
    from transformers import BlipForQuestionAnswering, BlipProcessor
    return (BlipProcessor.from_pretrained(VQA_MODEL),
            BlipForQuestionAnswering.from_pretrained(VQA_MODEL, torch_dtype=DTYPE).to(DEVICE).eval())


@lru_cache(maxsize=1)
def yolo_model():
    from ultralytics import YOLO
    return YOLO(YOLO_MODEL)


@lru_cache(maxsize=1)
def ocr_reader():
    import easyocr
    return easyocr.Reader(OCR_LANGS, gpu=DEVICE == "cuda", verbose=False)


# ---------- vision steps ----------
def load_image(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_IMAGE_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file couldn't be read as an image.")
    img.thumbnail((1024, 1024))
    return img


@torch.inference_mode()
def caption(img: Image.Image, rich: bool) -> str:
    proc, model = caption_model()
    inputs = proc(img, "a picture of", return_tensors="pt") if rich else proc(img, return_tensors="pt")
    out = model.generate(**_to_device(inputs), max_new_tokens=50 if rich else 25, min_new_tokens=6,
                         num_beams=5, repetition_penalty=1.4, no_repeat_ngram_size=3)
    return proc.decode(out[0], skip_special_tokens=True)


@torch.inference_mode()
def vqa(img: Image.Image, question: str) -> str:
    proc, model = vqa_model()
    out = model.generate(**_to_device(proc(img, question, return_tensors="pt")), max_new_tokens=15)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def detect(img: Image.Image, conf: float = 0.35) -> list[dict]:
    res = yolo_model().predict(img, conf=conf, verbose=False, device=0 if DEVICE == "cuda" else "cpu")[0]
    names = res.names
    return [{"label": names[int(c)], "conf": round(float(p), 3), "box": [round(float(v), 4) for v in b]}
            for b, c, p in zip(res.boxes.xyxyn.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]


def read_text(img: Image.Image, min_conf: float = 0.4) -> list[dict]:
    w, h = img.size
    items = []
    for pts, text, p in ocr_reader().readtext(np.array(img)):
        if p < min_conf or len(text.strip()) < 2:
            continue
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        items.append({"text": text.strip(), "conf": round(float(p), 3),
                      "box": [round(min(xs) / w, 4), round(min(ys) / h, 4), round(max(xs) / w, 4), round(max(ys) / h, 4)]})
    items.sort(key=lambda i: (round(i["box"][1], 1), i["box"][0]))  # reading order
    return items


def safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception as e:  # a missing optional model must not break the whole request
        print(f"[warn] {fn.__name__} failed: {e}")
        return default


# ---------- translation + speech ----------
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
    audio, mime = synthesize(text, lang, accent, slow)
    return {"audio_mime": mime, "audio_base64": base64.b64encode(audio).decode()}


# ---------- API ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "device": DEVICE.upper(),
            "models": {"caption": CAPTION_MODEL, "vqa": VQA_MODEL, "detector": YOLO_MODEL, "ocr": "EasyOCR"}}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    mode: str = Form("describe"),          # describe | read | live
    detail: str = Form("rich"),            # rich | brief
    question: str = Form(""),
    lang: str = Form("en"),
    accent: str = Form("in"),
    slow: bool = Form(False),
    speak: bool = Form(True),
):
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "Choose an image file.")
    img = load_image(await image.read())
    t0 = time.perf_counter()

    raw_caption = caption(img, rich=detail == "rich" and mode != "live")
    objects = safe(detect, img, default=[]) if mode != "read" else []
    ocr = safe(read_text, img, default=[]) if mode != "live" else []
    answer = vqa(img, question) if question.strip() and mode != "live" else ""
    vision_ms = int((time.perf_counter() - t0) * 1000)

    result = compose(raw_caption, [o["label"] for o in objects], [o["text"] for o in ocr],
                     question, answer, mode)
    spoken = translate(result["narration"], lang)
    payload = {
        **result, "raw_caption": raw_caption, "detections": objects, "ocr": ocr,
        "question": question or None, "spoken_text": spoken, "lang": lang,
        "image_size": img.size,
    }
    if speak:
        payload.update(audio_payload(spoken, lang, accent, slow))
    payload["timings_ms"] = {"vision": vision_ms, "total": int((time.perf_counter() - t0) * 1000)}
    return payload


@app.post("/api/ask")
async def ask(image: UploadFile = File(...), question: str = Form(...), lang: str = Form("en"),
              accent: str = Form("in"), slow: bool = Form(False)):
    img = load_image(await image.read())
    answer = answer_sentence(question, vqa(img, question))
    spoken = translate(answer, lang)
    return {"question": question, "answer": answer, "spoken_text": spoken, **audio_payload(spoken, lang, accent, slow)}


@app.post("/api/tts")
async def tts(text: str = Form(...), lang: str = Form("en"), accent: str = Form("in"), slow: bool = Form(False)):
    return audio_payload(text[:2000], lang, accent, slow)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
