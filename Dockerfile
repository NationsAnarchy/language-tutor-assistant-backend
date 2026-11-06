FROM python:3.12-slim

# Install system dependencies: ffmpeg for PCM->MP3 TTS transcoding, curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies using pinned lockfile
COPY requirements.txt requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock

# Copy application source
COPY app/ ./app/
COPY README.md ./

# Create default cache and data directories
RUN mkdir -p /app/audio_cache /app/data

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=5s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
