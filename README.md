# LLM Training Infrastructure & Analysis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Qwen3-1.8B-FF6F00" alt="Qwen3">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A **complete end-to-end infrastructure** for training Large Language Models from scratch. This repository provides everything needed to build, train, evaluate, and serve production-ready LLMs, with **Qwen3-1.8B** as our flagship model architecture.

---

## 🎯 Pipeline Overview: Building an LLM from Scratch

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      LLM TRAINING PIPELINE (End-to-End)                         │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  1. PLANNING     │     │  2. DATA PREP    │     │  3. TRAINING     │
  │                  │     │                  │     │                  │
  │  • Scaling Laws  │ ──▶ │  • SlimPajama    │ ──▶ │  • Qwen3-1.8B    │
  │  • FLOPs Budget  │     │    627B tokens   │     │  • B200 GPUs     │
  │  • Model Size    │     │  • Tokenization  │     │  • MFU Tracking  │
  │  • Token Count   │     │  • Qwen3 Vocab   │     │  • ZeRO-1/FSDP   │
  └──────────────────┘     └──────────────────┘     └──────────────────┘
           │                        │                        │
           │   training_planner/    │   enhanced_training_   │   enhanced_training_
           │                        │   system/data/         │   system/train.py
           │                        │                        │
           ▼                        ▼                        ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                                                                              │
  │    📊 Compute Budget    →    📦 Tokenized Data    →    🧠 Trained Model     │
  │    C = 9.22 ZFLOPs           627B tokens              1.8B parameters        │
  │    D = 64B optimal           Qwen3 tokenizer          115B tokens trained    │
  │                              151,643 vocab                                   │
  │                                                                              │
  └──────────────────────────────────────────────────────────────────────────────┘
           │                        │                        │
           │                        │                        │
           ▼                        ▼                        ▼
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  4. EVALUATION   │     │  5. SERVING      │     │  6. POST-TRAIN   │
  │                  │     │                  │     │                  │
  │  • ARC-E/C       │ ──▶│  • FastAPI       │ ──▶ │  • SFT           │
  │  • OpenBookQA    │     │  • Chat UI       │     │  • DPO           │
  │  • Log-prob      │     │  • REST API      │     │  • Alignment     │
  │  • Generation    │     │  • Production    │     │                  │
  └──────────────────┘     └──────────────────┘     └──────────────────┘
           │                        │                        │
           │   evaluation_system/   │   serving_system/      │   post_training/
           │                        │                        │
           └────────────────────────┴────────────────────────┘
```

---

## 📁 Repository Structure

```
llm_TII/
│
├── 🚀 enhanced_training_system/     # [CORE] Complete LLM training framework
│   ├── train.py                     # Main training script
│   ├── model_builder.py             # Modular model construction
│   ├── model_components.py          # Architecture components (RoPE, SwiGLU, RMSNorm)
│   ├── model_config.py              # Configuration system with presets
│   ├── training_logger.py           # Detailed JSON logging
│   ├── config/                      # Configuration files
│   │   ├── full_qwen3_1.8b_b200_optimal.py  # 🌟 Flagship config
│   │   └── full_llama2_1.36b_b200_optimal.py
│   ├── data/                        # Dataset preparation
│   │   ├── slimpajama_627b_qwen3/   # 🌟 Production dataset (627B tokens)
│   │   └── slimpajama_6b_qwen3/     # Quick testing subset
│   └── docs/                        # Detailed documentation (50+ docs)
│
├── 📊 training_planner/             # [ANALYSIS] FLOPs, Parameters & Scaling Laws
│   ├── analyze.py                   # 🌟 Main analysis tool
│   │   ├── Forward analysis         # Model → FLOPs/params
│   │   └── Backward scaling         # Compute budget → Optimal (N, D)
│   ├── configs/
│   │   ├── models/                  # LLaMA, DeepSeek V3 MoE configs
│   │   └── scaling_laws/            # Chinchilla (Hoffmann), Besiroglu 2024
│   └── docs/                        # Academic formulas & references
│
├── 🎯 post_training/                # SFT & DPO alignment
│   ├── train_sft.py                 # Supervised Fine-Tuning
│   ├── train_dpo.py                 # Direct Preference Optimization
│   ├── data/                        # Dataset preparation scripts
│   └── configs/                     # Training configurations
│
├── 🧪 evaluation_system/            # Model evaluation
│   ├── eval_benchmarks.py           # Benchmark runner (ARC, OpenBookQA)
│   ├── eval_qwen3_official.py       # Official Qwen3 comparison
│   └── docs/                        # Evaluation guides
│
├── 🌐 serving_system/               # Production deployment
│   ├── serve_qwen3.py               # FastAPI server with Chat UI
│   ├── static/index.html            # Modern chat interface
│   └── deploy/                      # Docker, Nginx configs
│
└── 📦 archive/                      # Historical development artifacts
    ├── development_phases/          # nanoGPT → ZeRO-1 → Triton → FSDP
    ├── scaling_law_standalone/      # Early scaling law tool
    ├── mfu_compute_standalone/      # Early MFU tool
    └── legacy_cost_analysis/        # Deprecated cost scripts
```

---

## 🌟 Flagship: Qwen3-1.8B Architecture

Our production model uses the **Qwen3 architecture** with modern optimizations:

| Component | Choice | Benefit |
|-----------|--------|---------|
| **Normalization** | RMSNorm | 25% faster than LayerNorm |
| **Position Encoding** | RoPE (θ=1M) | Better length extrapolation |
| **FFN** | SwiGLU (8/3x) | Better quality per FLOP |
| **Attention** | GQA (16 heads, 8 KV) | 50% KV cache reduction |
| **Activation** | SiLU | Smoother gradients |
| **Vocabulary** | 151,643 (BBPE) | Multilingual support |

### Model Specifications

```python
# Qwen3-1.8B Configuration
n_layer = 24           # Transformer layers
n_head = 16            # Query heads
n_embd = 2048          # Hidden dimension
d_ff = 6144            # FFN intermediate (SwiGLU)
num_key_value_heads = 8  # GQA key-value heads
block_size = 2048      # Context length
vocab_size = 151643    # Qwen3 vocabulary
```

**Total Parameters**: ~1.8B

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd llm_TII/enhanced_training_system
pip install -r requirements.txt
```

### 2. Data Preparation (SlimPajama-627B)

```bash
cd data/slimpajama_627b_qwen3

# Download and tokenize (high-scale workflow)
python build_manifest.py --output manifests/slimpajama_manifest.jsonl
python tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split train \
  --tokenizer ../../qwen3_tokenizer \
  --output-dir tokenized \
  --spawn-workers -1  # Use all CPU cores
```

### 3. Training (8× B200 GPUs)

```bash
# Production training
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py

# Quick test (Shakespeare)
python train.py config/preset_quick_test.py --max_iters=100
```

### 4. Evaluation

```bash
cd ../evaluation_system
python eval_benchmarks.py \
  --checkpoint /path/to/ckpt_160000.pt \
  --tokenizer ../enhanced_training_system/qwen3_tokenizer \
  --mode logprob
```

### 5. Serving

```bash
cd ../serving_system
uvicorn serve_qwen3:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 for Chat UI
```

---

## 📊 Compute Planning with Scaling Laws

The `training_planner/` module provides **detailed academic formulas** (not simplified 6ND) for:

### Forward Analysis: Model → FLOPs

```bash
cd training_planner
python analyze.py --model_config configs/models/llama_7b_config.json
```

Output includes:
- **FLOPs per token** (forward & training)
- **Component breakdown** (Attention vs FFN)
- **Memory requirements** (weights, gradients, optimizer, activations)

### Backward Scaling: Compute Budget → Optimal (N, D)

```bash
python analyze.py --backward_config configs/scaling_laws/hoffmann/backward_scaling_config.jsonc
```

Solves for **optimal training tokens D** given:
- GPU setup (8× H100, 8× B200, etc.)
- Training time (hours/days)
- Expected MFU
- Dataset constraints

```
================================================================================
BACKWARD SCALING LAW: Training Setup → Optimal (N, D)
================================================================================

Step 1: Calculate N from architecture
  Model parameters (N): 6.74B

Step 2: Calculate available compute (C)
  GPU setup: 8× H100 @ 989 TFLOPS
  Training time: 720 hours (30 days)
  Compute budget (C): 9.22e+21 FLOPs

Step 3: Calculate FLOPs per token (detailed formula)
  Training FLOPs/token: 144.00 GFLOPs

Step 4: Solve for D
  D_optimal: 64.03B tokens

Step 5: Predicted loss (Chinchilla)
  L(6.74B, 64.03B) = 2.1590
================================================================================
```

---

## ⚡ MFU (Model FLOPs Utilization)

Architecture-aware MFU calculation (integrated in `training_planner/`):

```python
# Forward pass FLOPs per layer:
attention_flops = 8*H² + 2*a*S²*H    # QKV + scores + output
ffn_flops = 6*H*D_ff                 # SwiGLU: 3 projections

# Training FLOPs = 3× Forward (1 forward + 2 backward)
# MFU = Achieved FLOPs / Hardware Peak FLOPs
```

### Supported Hardware

| GPU | BF16 Peak | Memory | Typical MFU |
|-----|-----------|--------|-------------|
| **B200** | 4,500 TF | 192 GB | 45-55% |
| H200 | 1,979 TF | 141 GB | 40-50% |
| H100 | 989 TF | 80 GB | 40-50% |
| A100 | 312 TF | 40/80 GB | 35-45% |

---

## 🎨 Modular Architecture System

Mix and match components without code changes:

```bash
# Qwen3-style (flagship)
python train.py config/full_qwen3_1.8b_b200_optimal.py

# LLaMA-style
python train.py config/full_llama2_1.36b_b200_optimal.py

# Custom mix
python train.py config/full_custom.py \
  --normalization=rmsnorm \
  --position_encoding=rope \
  --ffn_type=swiglu \
  --attention_backend=flash_attn_2
```

### Component Options

| Component | Options | Default (Qwen3) |
|-----------|---------|-----------------|
| **Normalization** | LayerNorm, RMSNorm | RMSNorm |
| **Position** | Learned, RoPE | RoPE (θ=1M) |
| **FFN** | Standard (4x), SwiGLU (8/3x) | SwiGLU |
| **Attention** | MHA, GQA | GQA |
| **Backend** | SDPA, FlashAttention-2 | FA2 |

---

## 📈 Training Output

```
================================================================================
🚀 TRAINING INITIALIZATION
================================================================================

📊 MODEL ARCHITECTURE:
  Architecture:          Qwen3-1.8B
  Total parameters:      1,831,845,888 (1.83B)
  Layers:                24
  Hidden size:           2048
  Attention heads:       16 (8 KV heads)
  FFN size:              6144 (SwiGLU)
  Sequence length:       2048

⚙️  TRAINING CONFIGURATION:
  Dataset:               SlimPajama-627B (Qwen3 tokenizer)
  Batch size (micro):    64
  Gradient accum:        4 (global: 32)
  Tokens per iteration:  4,194,304
  Total iterations:      25,000
  Total tokens:          ~105B

🖥️  HARDWARE:
  Device:                8× NVIDIA B200
  Peak FLOPs:            36,000 TFLOPS
  Precision:             bfloat16
  Parallelism:           DDP + ZeRO-1

📈 PERFORMANCE:
  Expected MFU:          45-50%
  Expected tokens/s:     ~160,000
================================================================================
```

---

## 🧪 Evaluation Results

| Benchmark | Random | Qwen3-1.8B (ours) | Notes |
|-----------|--------|-------------------|-------|
| **ARC-Easy** | 25% | 45-50% | Grade-school science |
| **ARC-Challenge** | 25% | 28-32% | Harder reasoning |
| **OpenBookQA** | 25% | 30-35% | World knowledge |

*Base model results. Instruction-tuning typically adds +10-20%.*

---

## 📚 Documentation

### Core Guides
- `enhanced_training_system/README.md` - Full training guide
- `enhanced_training_system/docs/` - 50+ detailed docs
- `training_planner/README.md` - Scaling law analysis
- `training_planner/docs/01_academic_formulas.md` - FLOPs formulas

### Quick References
- `post_training/README.md` - SFT & DPO guide
- `evaluation_system/README.md` - Benchmark evaluation
- `serving_system/README.md` - Deployment guide

---

## 📦 Archive

The `archive/` folder contains historical development artifacts preserved for reference:

| Folder | Original | Contents |
|--------|----------|----------|
| `development_phases/` | `system_implementation/` | nanoGPT → ZeRO-1 → Triton → FSDP progression |
| `scaling_law_standalone/` | `scaling_law/` | Early scaling law analysis tool |
| `mfu_compute_standalone/` | `MFU_compute/` | Standalone MFU calculator |
| `legacy_cost_analysis/` | `legacy/` | Deprecated cost scripts |
| `intermediate_sharing/` | `system_branch/` | Cross-repo sharing artifacts |

These modules have been **superseded by `training_planner/`** which integrates all functionality.

---

## 📚 References

1. [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy
2. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) - Alibaba
3. [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) - Hoffmann et al.
4. [LLaMA Paper](https://arxiv.org/abs/2302.13971) - Touvron et al.
5. [SlimPajama Dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B) - Cerebras
6. [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao et al.

---

## 📝 License

MIT License (same as nanoGPT)

## 🙏 Acknowledgments

- Andrej Karpathy for [nanoGPT](https://github.com/karpathy/nanoGPT)
- Alibaba for the Qwen architecture and tokenizer
- Cerebras for SlimPajama-627B dataset
- PyTorch team for distributed training infrastructure

---

*Complete infrastructure for building, training, evaluating, and serving production LLMs from scratch.*
