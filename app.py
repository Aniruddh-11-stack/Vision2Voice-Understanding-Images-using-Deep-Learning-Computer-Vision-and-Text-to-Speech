import streamlit as st
import os
import time
from PIL import Image
from model import Vision2VoicePredictor
import pandas as pd
from gtts import gTTS

# Configure Streamlit page
st.set_page_config(
    page_title="Vision2Voice AI",
    page_icon="👀",
    layout="wide"
)

# Custom styling for enterprise look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Vision2Voice AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6B7280; margin-bottom: 2rem;'>Understanding Images using Deep Learning, Computer Vision, and Text-to-Speech</p>", unsafe_allow_html=True)

# Define models directory (relative to script)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

@st.cache_resource
def load_predictor():
    """Cache the predictor model to avoid reloading on every interaction."""
    return Vision2VoicePredictor(MODELS_DIR)

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to initialize the model backend: {e}")
    predictor = None

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Settings")
    
    if not predictor or not getattr(predictor, 'ready', False):
        st.error("⚠️ Model Weights Missing!")
        st.info("The required `.h5` and `.pkl` weights were not found in the `models/` directory.")
        st.markdown("""
        **Required Files:**
        - `modelConcat_1_89.h5`
        - `caption_train_tokenizer.pkl`
        """)
        model_ready = False
    else:
        st.success("✅ Models Loaded & Ready")
        model_ready = True
        
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Upload any RGB image.")
    st.markdown("2. The **Base Vision Model** reads the whole scene.")
    st.markdown("3. **YOLOv8** crops individual objects.")
    st.markdown("4. The **VGG16+LSTM Ensemble** captions every object and joins them.")
    st.markdown("5. **gTTS** synthesizes the output into speech.")

# Main content layout
col1, col2 = st.columns([1, 1])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Save a temporary copy for YOLO
    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        
    if model_ready:
        if st.button("Generate Vision2Voice Analysis", use_container_width=True):
            with st.spinner("Analyzing image features and generating captions..."):
                start_time = time.time()
                
                try:
                    # Run the prediction pipeline
                    base_cap, collective_cap = predictor.analyze_full_image(temp_path)
                    
                    # Generate Audio
                    tts = gTTS(collective_cap)
                    audio_path = "output_speech.mp3"
                    tts.save(audio_path)
                    
                    process_time = time.time() - start_time
                    
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Display Text Results
                        st.markdown("#### 🧠 Traditional Model Output:")
                        st.info(f"*{base_cap}*")
                        
                        st.markdown("#### ✨ Vision2Voice Output (Ensemble):")
                        st.success(f"**{collective_cap}**")
                        
                        # Display Audio
                        st.markdown("#### 🔊 Audio Synthesis:")
                        st.audio(audio_path, format="audio/mp3")
                        
                        st.caption(f"Processing time: {process_time:.2f} seconds")
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                
                finally:
                    # Cleanup temporary files
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.warning("Please place the pre-trained model weights in the `models/` folder to enable inference.")
