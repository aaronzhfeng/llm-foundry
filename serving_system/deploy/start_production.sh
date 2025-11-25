#!/bin/bash
# Production deployment script for Qwen3-1.8B Inference Server
# Usage: ./start_production.sh [gpu_id] [port]

set -e

GPU_ID=${1:-0}
PORT=${2:-8000}
CHECKPOINT=${CHECKPOINT:-ckpt_160000.pt}

echo "🚀 Starting Qwen3-1.8B Production Server"
echo "   GPU: $GPU_ID"
echo "   Port: $PORT"
echo "   Checkpoint: $CHECKPOINT"

cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "/raid/zhf004/llm_TII/venv" ]; then
    source /raid/zhf004/llm_TII/venv/bin/activate
else
    echo "❌ No virtual environment found!"
    exit 1
fi

# Export configuration
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CHECKPOINT=$CHECKPOINT
export USE_COMPILE=true

# Start with Gunicorn for production (better process management)
# Falls back to uvicorn if gunicorn not installed
if command -v gunicorn &> /dev/null; then
    echo "📦 Starting with Gunicorn..."
    exec gunicorn serve_qwen3:app \
        --bind 0.0.0.0:$PORT \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers 1 \
        --timeout 120 \
        --keep-alive 30 \
        --access-logfile - \
        --error-logfile -
else
    echo "📦 Starting with Uvicorn..."
    exec uvicorn serve_qwen3:app \
        --host 0.0.0.0 \
        --port $PORT \
        --timeout-keep-alive 30
fi

