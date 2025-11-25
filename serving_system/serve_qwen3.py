#!/usr/bin/env python3
"""
Qwen3-1.8B Inference Server (FastAPI + torch.compile)

A lightweight serving solution for the custom-trained Qwen3 model.
No HuggingFace conversion required - loads raw PyTorch checkpoints directly.

Usage:
    # Default (first available GPU)
    uvicorn serve_qwen3:app --host 0.0.0.0 --port 8000
    
    # Specific GPU
    CUDA_VISIBLE_DEVICES=2 uvicorn serve_qwen3:app --host 0.0.0.0 --port 8000
    
    # Custom checkpoint
    CHECKPOINT=ckpt_080000.pt uvicorn serve_qwen3:app --port 8000

Test:
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "The capital of France is", "max_new_tokens": 50}'
"""

import os
import sys
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────────────────────────────────────
SERVING_ROOT = Path(__file__).parent.resolve()
TRAINING_SYSTEM = SERVING_ROOT.parent / "enhanced_training_system"
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/raid/zhf004/out-qwen3-1.8b-b200-50h"))

# Add training system to path for model imports
sys.path.insert(0, str(TRAINING_SYSTEM))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (Environment Variable Overrides)
# ─────────────────────────────────────────────────────────────────────────────
class ServerConfig:
    # GPU Selection: Use CUDA_VISIBLE_DEVICES or GPU_ID env var
    # CUDA_VISIBLE_DEVICES=2 restricts visibility to GPU 2 only
    # GPU_ID=2 uses cuda:2 directly (all GPUs still visible)
    GPU_ID = os.environ.get("GPU_ID", "0")
    
    if torch.cuda.is_available():
        DEVICE = f"cuda:{GPU_ID}" if "CUDA_VISIBLE_DEVICES" not in os.environ else "cuda:0"
        DTYPE = torch.bfloat16
    else:
        DEVICE = "cpu"
        DTYPE = torch.float32
    
    # Checkpoint selection via env var
    CHECKPOINT_NAME = os.environ.get("CHECKPOINT", "ckpt_160000.pt")
    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
    TOKENIZER_PATH = TRAINING_SYSTEM / "qwen3_tokenizer"
    
    # Inference defaults
    DEFAULT_MAX_TOKENS = 128
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    
    # Performance
    USE_COMPILE = os.environ.get("USE_COMPILE", "true").lower() == "true"

config = ServerConfig()

# Print config on import
print(f"📋 Server Configuration:")
print(f"   Device: {config.DEVICE}")
print(f"   Checkpoint: {config.CHECKPOINT_PATH}")
print(f"   torch.compile: {config.USE_COMPILE}")

# ─────────────────────────────────────────────────────────────────────────────
# Global Model State
# ─────────────────────────────────────────────────────────────────────────────
model = None
tokenizer = None
checkpoint_info = {}

def load_model():
    """Load model and tokenizer into memory."""
    global model, tokenizer, checkpoint_info
    
    from model_builder import ConfigurableGPT
    from model_config import ModelArchitectureConfig
    from transformers import AutoTokenizer
    
    print(f"🔄 Loading checkpoint from {config.CHECKPOINT_PATH}...")
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    
    # Store checkpoint metadata
    checkpoint_info = {
        "path": str(config.CHECKPOINT_PATH),
        "iter_num": ckpt.get("iter_num", "unknown"),
        "best_val_loss": ckpt.get("best_val_loss", "unknown"),
    }
    
    # Reconstruct model from saved config
    model_args = ckpt["model_args"]
    arch_config = ModelArchitectureConfig(**model_args)
    model = ConfigurableGPT(arch_config)
    
    # Strip "_orig_mod." prefix from torch.compile checkpoint
    state_dict = ckpt["model"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[clean_key] = v
    
    # Load weights (strict=False to skip RoPE cached tensors if shapes differ)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys (expected for RoPE cache): {len(missing)}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected}")
    
    model.to(config.DEVICE, dtype=config.DTYPE)
    model.eval()
    
    # Optional: recompile for inference speed
    if config.USE_COMPILE and config.DEVICE == "cuda":
        print("🔧 Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Load tokenizer
    print(f"🔄 Loading tokenizer from {config.TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.TOKENIZER_PATH), 
        trust_remote_code=True
    )
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model loaded: {param_count:.2f}B parameters on {config.DEVICE}")
    print(f"   Checkpoint iteration: {checkpoint_info['iter_num']}")
    print(f"   Best validation loss: {checkpoint_info['best_val_loss']}")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    load_model()
    yield
    # Cleanup (if needed)
    print("🛑 Shutting down server...")

app = FastAPI(
    title="Qwen3-1.8B Inference Server",
    description="Custom-trained Qwen3 model serving via FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(default=128, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0 = greedy)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling (None = disabled)")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="Repetition penalty")

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    full_response: str
    tokens_generated: int
    
class HealthResponse(BaseModel):
    status: str
    device: str
    checkpoint: str
    iteration: int
    best_val_loss: float
    parameters_billions: float

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text completion for the given prompt."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Tokenize input
    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(config.DEVICE)
    input_length = input_ids.shape[1]
    
    # ConfigurableGPT.generate() signature: (idx, max_new_tokens, temperature, top_k, eos_token_id, repetition_penalty)
    temperature = req.temperature if req.temperature > 0 else 1e-8  # Avoid div by zero for greedy
    top_k = req.top_k  # Can be None
    repetition_penalty = req.repetition_penalty  # Default 1.0 = no penalty
    
    # Generate with early stopping on EOS and repetition penalty
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,  # Stop when model generates EOS
            repetition_penalty=repetition_penalty,  # Discourage repetition
        )
    
    # Decode
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    tokens_generated = output_ids.shape[1] - input_length
    
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=generated_text,
        full_response=full_response,
        tokens_generated=tokens_generated,
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with model info."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    
    return HealthResponse(
        status="ok",
        device=config.DEVICE,
        checkpoint=str(config.CHECKPOINT_PATH),
        iteration=checkpoint_info.get("iter_num", 0),
        best_val_loss=float(checkpoint_info.get("best_val_loss", 0)),
        parameters_billions=round(param_count, 2),
    )

@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "service": "Qwen3-1.8B Inference Server",
        "endpoints": {
            "/generate": "POST - Generate text completion",
            "/health": "GET - Health check and model info",
            "/docs": "GET - OpenAPI documentation",
            "/chat": "GET - Chat UI",
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# Static Files & Chat UI
# ─────────────────────────────────────────────────────────────────────────────
STATIC_DIR = SERVING_ROOT / "static"

# Serve chat UI at root
@app.get("/")
async def chat_ui():
    """Serve the chat interface."""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/chat")
async def chat_redirect():
    """Alias for chat UI."""
    return FileResponse(STATIC_DIR / "index.html")

# Mount static files (for any additional assets)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

