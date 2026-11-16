"""API tests with the heavy models stubbed out (fast, no downloads)."""
import io
import os
import sys
import types

os.environ["PRELOAD"] = "0"
os.environ["ENABLE_GRAMMAR_MODEL"] = "0"

try:
    import torch  # noqa: F401
except ImportError:  # allow running tests without installing torch
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake.float16 = fake.float32 = object()
    fake.inference_mode = lambda: (lambda f: f)
    sys.modules["torch"] = fake

from fastapi.testclient import TestClient
from PIL import Image

import app as A

A.caption = lambda img, rich: "arafed brown dog sitting on a bench bench next to a bicycle with a"
A.vqa = lambda img, q: "2"
A.detect = lambda img: [{"label": "dog", "conf": 0.96, "box": [0.2, 0.3, 0.4, 0.6]},
                        {"label": "person", "conf": 0.9, "box": [0.6, 0.2, 0.8, 0.9]},
                        {"label": "person", "conf": 0.8, "box": [0.7, 0.2, 0.9, 0.9]}]
A.read_text = lambda img: [{"text": "PARK", "conf": 0.9, "box": [0.7, 0.1, 0.8, 0.2]}]
A.synthesize = lambda text, lang="en", accent="in", slow=False: (b"ID3fake", "audio/mpeg")
client = TestClient(A.app)


def _img():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, "JPEG")
    return ("x.jpg", buf.getvalue(), "image/jpeg")


def test_health_and_index():
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Vision to Voice" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_analyze_describe():
    d = client.post("/api/analyze", files={"image": _img()}, data={"question": "How many people are there?"}).json()
    assert d["caption"] == "A brown dog sitting on a bench next to a bicycle."
    assert d["narration"] == ("I can see a brown dog sitting on a bench next to a bicycle. I can also see two people. "
                              "There is text that reads “PARK”. There are two people.")
    assert d["audio_base64"] and len(d["detections"]) == 3


def test_analyze_read_and_live():
    r = client.post("/api/analyze", files={"image": _img()}, data={"mode": "read"}).json()
    assert r["narration"].startswith("The text reads: “PARK”.")
    l = client.post("/api/analyze", files={"image": _img()}, data={"mode": "live"}).json()
    assert "PARK" not in l["narration"] and l["ocr"] == []


def test_ask_and_errors():
    a = client.post("/api/ask", files={"image": _img()}, data={"question": "How many dogs are there?"}).json()
    assert a["answer"] == "There are two dogs."
    bad = client.post("/api/analyze", files={"image": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
