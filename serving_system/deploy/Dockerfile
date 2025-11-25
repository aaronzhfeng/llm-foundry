# Dockerfile for Qwen3-1.8B Inference Server
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serve_qwen3.py .
COPY static/ ./static/

# The training system is mounted as a volume at runtime
# Checkpoints are also mounted as a volume

# Environment variables (can be overridden)
ENV CHECKPOINT_DIR=/checkpoints
ENV CHECKPOINT=ckpt_160000.pt
ENV USE_COMPILE=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "serve_qwen3:app", "--host", "0.0.0.0", "--port", "8000"]

