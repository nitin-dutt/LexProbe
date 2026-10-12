FROM python:3.11-slim

# Install system deps for bitsandbytes + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api/     ./api/
COPY src/     ./src/
COPY models/  ./models/
# Secrets (HF_TOKEN) are passed at runtime: docker run -e HF_TOKEN=... — never copied into the image

# Create empty __init__ if missing
RUN touch api/__init__.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
