# Qwen3-1.8B Serving System

Lightweight inference server for custom-trained Qwen3 models.  
**No HuggingFace conversion required** – loads raw PyTorch checkpoints directly.

## 📁 Directory Structure

```
llm_TII/
├── enhanced_training_system/   # Training code & model definitions
│   ├── model_builder.py        # ConfigurableGPT class
│   ├── model_config.py         # ModelArchitectureConfig
│   └── qwen3_tokenizer/        # Tokenizer assets
├── serving_system/             # ← You are here
│   ├── serve_qwen3.py          # FastAPI server
│   ├── requirements.txt
│   └── README.md
```

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
cd /raid/zhf004/llm_TII/serving_system

# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Also need torch and transformers (use existing or install)
pip install torch transformers
```

> **Note:** If you already have a training venv at `/raid/zhf004/llm_TII/venv`, you can reuse it:
> ```bash
> source /raid/zhf004/llm_TII/venv/bin/activate
> pip install fastapi uvicorn pydantic
> ```

### 2. Start the Server

```bash
# Make sure venv is activated
source venv/bin/activate  # or source /raid/zhf004/llm_TII/venv/bin/activate

# Option A: Direct Python
python serve_qwen3.py

# Option B: Uvicorn (recommended for production)
uvicorn serve_qwen3:app --host 0.0.0.0 --port 8000

# Option C: With auto-reload for development
uvicorn serve_qwen3:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | **Chat UI** (ChatGPT-style interface) |
| `/chat` | GET | Alias for Chat UI |
| `/health` | GET | Health check with model info |
| `/generate` | POST | Text generation API |
| `/api` | GET | API info and available endpoints |
| `/docs` | GET | Interactive OpenAPI documentation |

### 🎨 Chat Interface

Open `http://localhost:8000` in your browser for a modern chat UI with:
- Real-time generation with typing indicators
- Adjustable settings (temperature, max tokens, top_p)
- Conversation history
- Suggested prompts to get started

### POST `/generate`

**Request Body:**

```json
{
  "prompt": "string (required)",
  "max_new_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": null,
  "repetition_penalty": 1.0
}
```

**Response:**

```json
{
  "prompt": "The capital of France is",
  "generated_text": " Paris, which is also the largest city...",
  "full_response": "The capital of France is Paris, which is also the largest city...",
  "tokens_generated": 42
}
```

## ⚙️ Configuration

### Environment Variables

All configuration can be done via environment variables (no code changes needed):

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | (all) | Restrict visible GPUs (recommended) |
| `GPU_ID` | `0` | Select GPU index when all are visible |
| `CHECKPOINT` | `ckpt_160000.pt` | Checkpoint filename to load |
| `CHECKPOINT_DIR` | `/raid/zhf004/out-qwen3-1.8b-b200-50h` | Directory containing checkpoints |
| `USE_COMPILE` | `true` | Enable torch.compile (`true`/`false`) |

### GPU Selection Examples

```bash
# Use GPU 0 (default)
uvicorn serve_qwen3:app --port 8000

# Use GPU 2 specifically (hides other GPUs from PyTorch)
CUDA_VISIBLE_DEVICES=2 uvicorn serve_qwen3:app --port 8000

# Use GPU 3 (alternative method, all GPUs still visible)
GPU_ID=3 uvicorn serve_qwen3:app --port 8000

# Use CPU only
CUDA_VISIBLE_DEVICES="" uvicorn serve_qwen3:app --port 8000
```

### Loading Different Checkpoints

```bash
# Serve an earlier checkpoint (e.g., iteration 80000)
CHECKPOINT=ckpt_080000.pt uvicorn serve_qwen3:app --port 8000

# Serve from a different directory
CHECKPOINT_DIR=/path/to/other/checkpoints CHECKPOINT=ckpt.pt uvicorn serve_qwen3:app --port 8000

# Combine: specific GPU + specific checkpoint
CUDA_VISIBLE_DEVICES=1 CHECKPOINT=ckpt_100000.pt uvicorn serve_qwen3:app --port 8000
```

### Disable torch.compile (faster startup, slower inference)

```bash
USE_COMPILE=false uvicorn serve_qwen3:app --port 8000
```

## 📊 Available Checkpoints

From the 50-hour production run:

| Checkpoint | Iteration | Training Tokens |
|------------|-----------|-----------------|
| `ckpt_020000.pt` | 20,000 | ~14.4B |
| `ckpt_040000.pt` | 40,000 | ~28.8B |
| `ckpt_060000.pt` | 60,000 | ~43.2B |
| `ckpt_080000.pt` | 80,000 | ~57.6B |
| `ckpt_100000.pt` | 100,000 | ~72.0B |
| `ckpt_120000.pt` | 120,000 | ~86.4B |
| `ckpt_140000.pt` | 140,000 | ~100.8B |
| `ckpt_160000.pt` | 160,000 | ~115.2B |

## 🔧 Troubleshooting

### CUDA Out of Memory

Reduce batch size or switch to CPU:
```python
DEVICE = "cpu"
DTYPE = torch.float32
USE_COMPILE = False
```

### Missing Keys Warning

Expected for RoPE cache tensors – they are recomputed at runtime.

### Slow First Request

`torch.compile` warmup takes ~30s on first inference. Subsequent requests are fast.

## 🚀 Production Deployment

### Quick: Expose to Internet (ngrok/Cloudflare)

**Option A: ngrok** (easiest, free tier available)
```bash
# Step 1: Sign up at https://dashboard.ngrok.com/signup (free)

# Step 2: Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
#         It looks like: 2abc123DEF456ghi789JKL_0mnoPQRstu1vwxYZ2ab3c

# Step 3: Configure ngrok with your token
ngrok config add-authtoken 35z7kbF7gGXghocSKPWK3j5Njwz_3LZAqTuActXprjC1s7T2q

# Step 4 (Terminal 1): Start server
CUDA_VISIBLE_DEVICES=6 uvicorn serve_qwen3:app --host 0.0.0.0 --port 8000

# Step 5 (Terminal 2): Expose to internet
ngrok http 8000
# → Gives you: https://abc123.ngrok-free.app
```

**Option B: Cloudflare Tunnel** (free, stable, custom domain)
```bash
# Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/
cloudflared tunnel --url http://localhost:8000
```

**Option C: SSH Tunnel** (no install needed)
```bash
ssh -R 80:localhost:8000 serveo.net
# → Gives you: https://xxx.serveo.net
```

### Production: Docker + Nginx

For a proper production setup with SSL and rate limiting:

```bash
cd deploy/

# Build and start
docker-compose up -d

# Or without Docker:
./start_production.sh 0 8000  # GPU 0, port 8000
```

See `deploy/` folder for:
- `Dockerfile` - Container image
- `docker-compose.yml` - Full stack with Nginx
- `nginx.conf` - SSL, rate limiting, CORS
- `start_production.sh` - Simple production script

### Scaling Options

1. **Multiple workers** (CPU-bound tasks only, model stays on 1 GPU):
   ```bash
   uvicorn serve_qwen3:app --workers 4
   ```

2. **Gunicorn** (better process management):
   ```bash
   gunicorn serve_qwen3:app -w 1 -k uvicorn.workers.UvicornWorker --timeout 120
   ```

3. **Load balancer** (multiple GPUs):
   - Run separate instances on different ports/GPUs
   - Use Nginx upstream to load balance

## 📚 Related Documentation

- Training system: `../enhanced_training_system/docs/`
- MFU analysis: `../enhanced_training_system/docs/46_llm_training_incidents.md`
- Checkpoint details: `../enhanced_training_system/docs/49_training_runtime_best_practices.md`

