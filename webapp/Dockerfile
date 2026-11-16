# Build from the REPOSITORY ROOT so the trained weights can be copied in:
#   git lfs pull --include="capstone_data_2/Saved_Models3/modelConcat_1_89.h5,capstone_data_2/caption_train_tokenizer.pkl"
#   docker build -f webapp/Dockerfile -t vision2voice-web .
#   docker run -p 8000:8000 vision2voice-web
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY webapp/backend/requirements.txt webapp/backend/
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r webapp/backend/requirements.txt
COPY capstone_data_2/Saved_Models3/modelConcat_1_89.h5 capstone_data_2/Saved_Models3/
COPY capstone_data_2/caption_train_tokenizer.pkl capstone_data_2/
COPY webapp webapp
WORKDIR /app/webapp/backend
ENV ENABLE_GRAMMAR_MODEL=0
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
