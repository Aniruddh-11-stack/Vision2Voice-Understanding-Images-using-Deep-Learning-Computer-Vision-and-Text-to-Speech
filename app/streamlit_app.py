"""
Vision2Voice — Streamlit Dashboard.

Entry point for the enterprise Streamlit web application.
Run with:
    streamlit run app/streamlit_app.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vision2voice.predictor import Vision2VoicePredictor
from vision2voice.audio import TextToSpeechEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = str(ROOT / "models")
OUTPUTS_DIR = str(ROOT / "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

st.set_page_config(page_title="Vision2Voice AI", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center;color:#1E3A8A;font-weight:700'>Vision2Voice AI Dashboard</h1>", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI models…")
def load_predictor(): return Vision2VoicePredictor(MODELS_DIR)

@st.cache_resource
def load_tts(): return TextToSpeechEngine(output_dir=OUTPUTS_DIR)

predictor = load_predictor()
tts_engine = load_tts()

with st.sidebar:
    st.title("Settings")
    st.success("✅ Models Ready") if predictor.ready else st.error("⚠️ Weights Missing!")

col_upload, col_results = st.columns([1, 1], gap="large")
with col_upload:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Select a JPG or PNG", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_container_width=True)
        run_btn = st.button("▶ Run Vision2Voice Analysis", use_container_width=True) if predictor.ready else None

if uploaded_file and predictor.ready and run_btn:
    temp_path = str(ROOT / "outputs" / "temp_upload.jpg")
    image.save(temp_path)
    with st.spinner("Analysing image..."):
        start = time.perf_counter()
        try:
            base_cap, ens_cap = predictor.analyze_full_image(temp_path)
            audio_path = tts_engine.synthesize(ens_cap)
            elapsed = time.perf_counter() - start
            with col_results:
                st.subheader("📈 Analysis Results")
                st.markdown("#### 🧠 Base Caption"); st.info(f"*{base_cap}*")
                st.markdown("#### ✨ Ensemble Description"); st.success(f"**{ens_cap}**")
                st.markdown("#### 🔊 Audio"); st.audio(audio_path, format="audio/mp3")
                st.caption(f"⏱ Processing: {elapsed:.2f}s")
        except Exception as e: st.error(f"Error: {e}")
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
