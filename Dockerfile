FROM python:3.12-slim

# Install system dependencies: ffmpeg for PCM->MP3 TTS transcoding, curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and seed data
COPY app/ ./app/
COPY data/ ./data/
COPY README.md ./

# Create default data directory with non-root ownership
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

VOLUME ["/app/data"]

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=5s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
