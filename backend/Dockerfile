# Dockerfile for Autonomous Insurance Credentialing Agent
# Optimized for Google Cloud Run

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies + production server
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn eventlet

# Copy application code
COPY api_server.py .
COPY auth.py .
COPY credentialing_agent.py .
COPY knowledge_base.py .
COPY create_admin.py .
COPY cli.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/exports /app/temp_audio

# Cloud Run injects PORT (default 8080) — do NOT hardcode 5000
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run with gunicorn + eventlet for WebSocket (flask-socketio) support
# -w 1 is required with eventlet/gevent workers
CMD exec gunicorn --worker-class eventlet -w 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --keep-alive 5 \
    "api_server:app"
