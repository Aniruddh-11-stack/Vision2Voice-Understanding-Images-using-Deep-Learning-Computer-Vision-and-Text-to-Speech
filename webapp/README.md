# Vision to Voice

Describes images out loud, built for accessibility (for example, helping visually impaired users).

## Pipeline
```
image ─┬─ BLIP-large ............ caption
       ├─ BLIP-VQA .............. answers your questions
       ├─ YOLOv8 ................ objects + boxes
       └─ EasyOCR ............... printed text
              │
              ▼
   text_polish.py  →  grammatical narration
   • rules: strips BLIP artifacts ("arafed"), echoed prompts, stutters ("bench bench"),
     repeated phrases, dangling words ("…with a"); adds missing articles; fixes a/an;
     plurals and counts ("two people"); natural lists ("a cup, two chairs, and a dog")
   • neural: T5 grammar correction, accepted only when the edit is small (can't hallucinate)
              │
              ▼
   translation (optional) → gTTS (Indian/US/UK/AU English + Indian languages) / pyttsx3 offline
```
Example: `arafed brown dog sitting on a bench bench next to a bicycle with a`
→ *"I can see a brown dog sitting on a bench next to a bicycle. I can also see two people. There is text that reads “PARK”."*

## Features
- **Describe**: upload, drag and drop, paste, or capture with the camera; brief or rich detail
- **Live camera**: narrates every 4 s and only speaks when the scene changes
- **Read text**: reads signs, labels, and medicine boxes aloud
- **Ask**: type or speak a question ("How many people are there?" → "There are three people.")
- Detection boxes overlaid on the image, a waveform audio player with speed control, MP3 download, history
- Responsive: two-panel laptop layout, one-column phone layout with a bottom navigation bar
- Accessible: real buttons and labels, keyboard support (Space plays and pauses), live region for results

## Run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. The first start downloads about 3 GB of models (BLIP, YOLOv8n, EasyOCR, T5).
Camera access needs `localhost` or HTTPS. On Linux, offline TTS needs `sudo apt install espeak-ng`.

Docker: `docker build -t vision2voice . && docker run -p 8000:8000 vision2voice` (add `--gpus all` with a CUDA image)

## Tests
`cd backend && pytest -q` covers the grammar rules and every API endpoint, with the models stubbed, so it runs in under a second.

## API
| Method | Path | Fields (multipart) |
|---|---|---|
| GET | `/api/health` | |
| POST | `/api/analyze` | `image`, `mode` (describe/read/live), `detail` (rich/brief), `question`, `lang`, `accent` (in/us/uk/au), `slow`, `speak` |
| POST | `/api/ask` | `image`, `question`, `lang`, `accent` |
| POST | `/api/tts` | `text`, `lang`, `accent` |

`/api/analyze` returns `caption`, `narration`, `objects`, `detections` (normalised boxes), `ocr`, `text`, `answer`, `raw_caption`, `audio_base64`, `audio_mime`, `timings_ms`.

## Configuration (env vars)
`CAPTION_MODEL` (use `Salesforce/blip-image-captioning-base` for faster CPU) · `VQA_MODEL` · `YOLO_MODEL` (`yolov8s.pt` for higher accuracy) · `OCR_LANGS` (`en,hi`) · `GRAMMAR_MODEL` · `ENABLE_GRAMMAR_MODEL=0` (rules only) · `PRELOAD=0`
