"""API tests with the VGG16 + LSTM + YOLO pipeline stubbed (fast, no TensorFlow or weights needed)."""
import io
import os

os.environ["PRELOAD"] = "0"
os.environ["ENABLE_GRAMMAR_MODEL"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import app as A  # noqa: E402


class FakeCaptioner:
    def analyze(self, img, beam_width=10, crops=True):
        return {"caption": "people are standing on the side of bus",
                "detections": [{"label": "bus", "conf": 0.9, "box": [0.0, 0.2, 1.0, 0.8]},
                               {"label": "person", "conf": 0.8, "box": [0.1, 0.4, 0.3, 0.9]},
                               {"label": "dog", "conf": 0.7, "box": [0.5, 0.6, 0.6, 0.9]}],
                "crop_captions": ["man in blue shirt is standing next to several basket poles"] if crops else []}


A._state["captioner"] = FakeCaptioner()
A.synthesize = lambda text, lang="en", accent="in", slow=False: (b"ID3fake", "audio/mpeg")
client = TestClient(A.app)


def _img():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, "JPEG")
    return ("x.jpg", buf.getvalue(), "image/jpeg")


def test_health_and_static():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "VGG16" in h["models"]["encoder"]
    assert "Vision to Voice" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_analyze_rich():
    d = client.post("/api/analyze", files={"image": _img()}).json()
    assert d["caption"] == "People are standing on the side of a bus."
    assert d["narration"] == ("People are standing on the side of a bus. I can also see a dog. "
                              "Up close, a man in a blue shirt is standing next to several basket poles.")
    assert d["raw_caption"] == "people are standing on the side of bus"
    assert d["audio_base64"] and len(d["detections"]) == 3


def test_analyze_brief_and_live():
    b = client.post("/api/analyze", files={"image": _img()}, data={"detail": "brief"}).json()
    assert b["closeups"] == []
    live = client.post("/api/analyze", files={"image": _img()}, data={"mode": "live"}).json()
    assert "Up close" not in live["narration"]


def test_bad_upload_and_tts():
    assert client.post("/api/analyze", files={"image": ("x.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/tts", data={"text": "Hello"}).json()["audio_mime"] == "audio/mpeg"
