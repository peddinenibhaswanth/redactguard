# RedactGuard API (FastAPI). The Streamlit demo has its own image (see
# demo/Dockerfile) since it's deployed separately (Hugging Face Spaces).
FROM python:3.11-slim

# Tesseract binary is required for OCR extraction - pip installing
# pytesseract alone does NOT install this, which is the single most common
# setup failure per the implementation guide.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY phase3_finetune/prompt_template.py phase3_finetune/prompt_template.py

ENV OUTPUT_DIR=/app/outputs
RUN mkdir -p /app/outputs

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
