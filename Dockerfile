# ============================================================
# Stage 1 — Base image with system-level dependencies
# ============================================================
FROM python:3.10-slim AS base

LABEL maintainer="Aniruddh Kulkarni <anikulks@gmail.com>"
LABEL description="Vision2Voice: AI image analysis and speech synthesis"

# System dependencies for OpenCV and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 2 — Python dependencies
# ============================================================
FROM base AS dependencies

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 3 — Application
# ============================================================
FROM dependencies AS app

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/

# Create required directories
RUN mkdir -p models outputs

# Set Python path
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
