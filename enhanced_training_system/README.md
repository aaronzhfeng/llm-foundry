# Enhanced GPT Training System

A comprehensive, **fully modular** GPT training implementation with detailed MFU calculation, memory tracking, and gradient monitoring. Built on [nanoGPT](https://github.com/karpathy/nanoGPT) with academic formula-based performance analysis.

## 🚀 Key Features

### **🎨 Fully Modular Architecture System**
**Mix and match components without changing code!**

- **9 Configurable Components**: Normalization, activation, position encoding, FFN type, etc.
- **4 Preset Architectures**: GPT-2, LLaMA, Team model_v1, Hybrid
- **Easy Experimentation**: Change architecture via config file only
- **Extensible**: Add new components via registry pattern

**Supported Components:**
- **Normalization**: LayerNorm, RMSNorm
- **Position Encoding**: Learned Absolute, RoPE
- **FFN**: Standard (4x), SwiGLU (8/3x)
- **Norm Position**: Pre-norm, Post-norm
- **Activation**: GELU, SiLU, ReLU
- **And more!**

### **📊 Architecture-Aware MFU Calculation**
- Academic formula: `FLOPs = 12SBH² + 2aS²BH` per layer (forward pass)
- **Adapts to architecture**: Accounts for SwiGLU, RoPE, RMSNorm overhead
- Hardware support: B200/H200/H100/A100/V100
- Real-time tokens/second and TFLOPS tracking

### **🔍 Comprehensive Monitoring**
- **Memory Stats**: Allocated, peak, and reserved memory per iteration
- **Gradient Health**: Global norm, layer-wise norms, value distribution
- **Performance Metrics**: Tokens/s, FLOPs/token, hardware utilization
- **Training Stability**: Loss variance, learning rate, gradient tracking

### **⚡ Advanced Parallelism**
- **DDP (Data Parallel)**: Standard multi-GPU training
- **ZeRO-1**: Optimizer state sharding (~50% memory reduction)
- **FSDP**: Full parameter/gradient/optimizer sharding (~75-88% memory reduction)

### **Enhanced Terminal Output**
```
================================================================================
🚀 TRAINING INITIALIZATION
================================================================================

📊 MODEL ARCHITECTURE:
  Total parameters:      124,439,808 (124.44M)
  Trainable parameters:  124,439,808 (124.44M)
  Non-embedding params:  123,587,328 (123.59M)
  Layers:                12
  Hidden size:           768
  Attention heads:       12
  Sequence length:       1024
  Vocabulary size:       50304

⚙️  TRAINING CONFIGURATION:
  Batch size (micro):    12
  Gradient accum steps:  40
  Effective batch size:  480
  Tokens per iteration:  491,520
  ...

🖥️  HARDWARE:
  Device:                NVIDIA A100-SXM4-80GB
  GPUs:                  1
  Memory per GPU:        80.0 GB
  Precision:             bfloat16
  Parallelism:           DDP

📈 THEORETICAL PERFORMANCE:
  Hardware peak:         312.0 TFLOPS (A100 bf16)
  FLOPs per token:       28.45 GFLOPs
  Attention/FFN ratio:   0.67
  Expected tokens/s @50% MFU: 5483

================================================================================
🏁 STARTING TRAINING
================================================================================

────────────────────────────────────────────────────────────────────────────────
📍 Iter      5 │ Loss: 10.6456 │ Time: 4298ms │ LR: 1.50e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 32.45% │ Achieved: 101.2 TF │ Peak: 312.0 TF
   Tokens/s: 3,557 │ FLOPs/token: 28.5 GF
💾 Memory: 12.34 GB alloc │ 15.67 GB peak │ 16.00 GB reserved
📊 Gradients: norm=2.3456 │ mean=-1.23e-05 │ std=3.45e-04
```

## 📋 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- (Optional) Triton for custom kernels

### Setup
```bash
# Clone and navigate
git clone <your-repo-url>
cd dsc180_a06

# Install dependencies
pip install torch numpy transformers tiktoken wandb

# Prepare data (Shakespeare for testing)
cd data/shakespeare
python prepare.py
cd ../..
```

## 🎯 Quick Start

### **Test Different Architectures (Single GPU)**
```bash
# GPT-2 architecture (baseline)
python train.py config/full_gpt2_124m.py --max_iters=100

# LLaMA architecture (RoPE + RMSNorm + SwiGLU)
python train.py config/full_llama_124m.py --max_iters=100

# LLaMA 1.36B (production model)
python train.py config/full_llama_1.36b.py --max_iters=100

# Team's model_v1 architecture
python train.py config/full_team_124m.py --max_iters=100

# Custom architecture (experiment!)
python train.py config/full_custom.py --max_iters=100

# Quick test (small model)
python train.py config/preset_quick_test.py --max_iters=100
```

### **Override Architecture On-the-Fly**
```bash
# Take GPT-2 but use RoPE instead of learned positions
python train.py config/full_gpt2_124m.py --position_encoding=rope

# Take LLaMA but use LayerNorm instead of RMSNorm
python train.py config/full_llama_124m.py --normalization=layernorm_nobias

# Mix any components you want!
python train.py config/full_custom.py \
  --normalization=rmsnorm \
  --position_encoding=rope \
  --ffn_type=standard \
  --norm_position=pre
```

### **Multi-GPU Training (4 GPUs)**
```bash
# Standard DDP
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_124m.py

# With ZeRO-1 (50% memory reduction)
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_124m.py --use_zero1=True

# With FSDP (75-88% memory reduction)
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py --use_fsdp=True
```

### **HGX B200 Training** (Production - 1.36B Model)
```bash
# LLaMA 1.36B on 8x B200 with FSDP
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --dataset=slimpajama_627b \
  --use_fsdp=True
```

## ⚙️ Configuration

### Modular Architecture System

**All architectural choices are configurable!** No code changes needed.

#### **Configuration Files:**

**Full Configurations (Architecture + Training):**
- `config/full_gpt2_124m.py` - GPT-2 124M (baseline)
- `config/full_llama_124m.py` - LLaMA 124M (for comparison)
- `config/full_gpt2_1.36b.py` - **GPT-2 1.36B (production - direct comparison)**
- `config/full_llama_1.36b.py` - **LLaMA 1.36B (production model)**
- `config/full_team_124m.py` - Team's model_v1
- `config/full_custom.py` - Experiment with any combination

**Training Presets (Override configs):**
- `config/preset_quick_test.py` - Quick testing (small model)
- `config/preset_gpt2_owt.py` - GPT-2 on OpenWebText

**Documentation:**
- `config/ARCH_GPT2.md` - GPT-2 architecture explained
- `config/ARCH_LLAMA.md` - LLaMA architecture explained
- `config/PARAMETER_FORMULAS.md` - **Parameter counting formulas and design options**

#### **Architectural Options**

| Component | Options | Default (GPT-2) | Default (LLaMA) |
|-----------|---------|-----------------|-----------------|
| **Normalization** | layernorm, layernorm_nobias, rmsnorm | layernorm_nobias | rmsnorm |
| **Activation** | gelu, silu, relu, leaky_relu | gelu | gelu (N/A for SwiGLU) |
| **Position Encoding** | learned_absolute, rope, none | learned_absolute | rope |
| **Attention Backend** | sdpa, manual | sdpa | sdpa |
| **Norm Position** | pre, post | post | pre |
| **FFN Type** | standard, swiglu | standard (4x) | swiglu (8/3x) |
| **Bias** | True, False | False | False |
| **Weight Tying** | True, False | True | False |
| **Dropout** | 0.0-1.0 | 0.0 | 0.0 |

### Command-Line Overrides

```bash
# Override architecture components
python train.py config/full_gpt2_124m.py \
  --normalization=rmsnorm \
  --position_encoding=rope \
  --ffn_type=swiglu

# Override training parameters
python train.py config/full_llama_1.36b.py \
  --batch_size=16 \
  --learning_rate=3e-4 \
  --max_iters=10000 \
  --compile=False

# Use different dataset
python train.py config/full_llama_1.36b.py \
  --dataset=slimpajama_627b
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Architecture** | | |
| `arch_preset` | Preset: 'gpt2', 'llama', 'team', 'custom' | gpt2 |
| `normalization` | Norm type (if arch_preset='custom') | layernorm_nobias |
| `position_encoding` | Position encoding type | learned_absolute |
| `ffn_type` | FFN type ('standard' or 'swiglu') | standard |
| **Model Size** | | |
| `n_layer` | Number of transformer layers | 12 |
| `n_head` | Number of attention heads | 12 |
| `n_embd` | Hidden dimension size | 768 |
| `block_size` | Sequence length | 1024 |
| **Training** | | |
| `batch_size` | Micro-batch size per GPU | 12 |
| `gradient_accumulation_steps` | Gradient accumulation steps | 40 |
| `learning_rate` | Max learning rate | 6e-4 |
| `max_iters` | Training iterations | 600000 |
| **System** | | |
| `use_zero1` | Enable ZeRO-1 optimizer sharding | False |
| `use_fsdp` | Enable FSDP full sharding | False |
| `compile` | Use torch.compile() | True |
| `dtype` | Precision (bfloat16/float16/float32) | bfloat16 |

## 📊 Monitoring & Logging

### JSON Logs
All training runs automatically create detailed JSON logs:

```bash
# View logs
ls out/run_*.json

# Example log structure
{
  "run_name": "run_20250103_143022",
  "start_time": "2025-01-03T14:30:22",
  "config": {...},
  "startup_info": {...},
  "training_iterations": [
    {
      "iter": 100,
      "loss": 3.456,
      "time_ms": 234.5,
      "mfu": {
        "mfu_percent": 32.5,
        "achieved_tflops": 101.2,
        "hardware_peak_tflops": 312.0,
        "tokens_per_sec": 3557,
        "flops_per_token": 28.5e9,
        ...
      },
      "memory": {...},
      "gradients": {...}
    }
  ],
  "eval_steps": [...],
  "checkpoints": [...],
  "summary": {...}
}
```

### WandB Integration
```bash
python train.py --wandb_log=True --wandb_project=my-project
```

## 🧪 Architecture Experiments

### Ablation Studies Made Easy

Test the impact of individual components:

```bash
# Baseline: GPT-2
python train.py config/full_gpt2_124m.py --max_iters=5000

# Ablation 1: GPT-2 + RoPE (test RoPE benefit)
python train.py config/full_gpt2_124m.py --position_encoding=rope --max_iters=5000

# Ablation 2: GPT-2 + RMSNorm (test RMSNorm benefit)
python train.py config/full_gpt2_124m.py --normalization=rmsnorm --max_iters=5000

# Ablation 3: GPT-2 + SwiGLU (test SwiGLU benefit)
python train.py config/full_gpt2_124m.py --ffn_type=swiglu --max_iters=5000

# Full LLaMA (all improvements)
python train.py config/full_llama_124m.py --max_iters=5000

# Compare JSON logs to see which helps most!
```

### Preset Architectures

| Preset | Norm | Position | FFN | Norm Pos | Tying | Description |
|--------|------|----------|-----|----------|-------|-------------|
| **gpt2** | LayerNorm | Learned | Standard (4x) | Post | Yes | Original GPT-2 |
| **llama** | RMSNorm | RoPE | SwiGLU (8/3x) | Pre | No | LLaMA-style |
| **team** | RMSNorm | RoPE | SwiGLU (8/3x) | Pre | No | Team's model_v1 |
| **hybrid** | LayerNorm | RoPE | Standard (4x) | Pre | Yes | Experimental |

### Example Combinations

```python
# In config/full_custom.py:

# Experiment 1: Best of GPT-2 + LLaMA
arch_preset = 'custom'
normalization = 'rmsnorm'           # From LLaMA
position_encoding = 'rope'          # From LLaMA
ffn_type = 'standard'               # From GPT-2
activation = 'gelu'                 # From GPT-2
weight_tying = True                 # From GPT-2
norm_position = 'pre'               # From LLaMA

# Experiment 2: Minimal changes to GPT-2
arch_preset = 'custom'
normalization = 'layernorm_nobias'  # GPT-2
position_encoding = 'rope'          # ONLY CHANGE
ffn_type = 'standard'               # GPT-2
norm_position = 'post'              # GPT-2
weight_tying = True                 # GPT-2
```

## 📈 MFU Calculation Details

### Architecture-Aware Academic Formula

**Automatically adjusts for your architecture!**

```python
# Forward pass FLOPs per layer (base):
attention_flops = 6*S*H² + 2*a*S²*H  # QKV proj + scores + output

# FFN FLOPs (depends on architecture):
if ffn_type == 'swiglu':
    ffn_flops = 3 * (2*S*H*D_ff)     # 3 projections (gate, value, out)
else:
    ffn_flops = 2 * (2*S*H*D_ff)     # 2 projections (up, down)

# Position encoding overhead:
if position_encoding == 'rope':
    rope_flops = 2 * a * S * (H//a) * 2  # Rotation for Q and K

# Normalization:
if normalization == 'rmsnorm':
    norm_flops = 1.5 * S * H         # Faster than LayerNorm
else:
    norm_flops = 2 * S * H           # LayerNorm

# Total:
total_forward = L * (attention_flops + ffn_flops + rope_flops + norm_flops)

# Training FLOPs (PaLM formula: 6N + 12LHQT):
# N = non-embedding params (billions), Q = head_dim, T = seq_len
training_flops_per_token = 6.0 * N_billion + 12.0 * L * H * Q * T / 1e9  # GFLOPs

# MFU calculation:
flops_achieved = training_flops_per_token * tokens_per_sec
mfu = flops_achieved / hardware_peak_flops
```

**Why This Matters:**
- **SwiGLU** has ~50% more FFN FLOPs than standard
- **RoPE** adds position encoding overhead
- **RMSNorm** is ~25% cheaper than LayerNorm
- **Accurate MFU** requires accounting for these differences!

**References:**
- Insu Jang (2022): [Analysis of Transformer Model](https://insujang.github.io/2022-07-30/analysis-of-transformer-model/)
- Epoch AI: [Backward-Forward FLOP Ratio](https://epoch.ai/blog/backward-forward-FLOP-ratio)

### Hardware Specs (Auto-detected)

| GPU | BF16 Peak | FP16 Peak | Memory |
|-----|-----------|-----------|--------|
| **B200** | 4,500 TF | 4,500 TF | 192 GB |
| H200 | 1,979 TF | 1,979 TF | 141 GB |
| H100 | 989 TF | 989 TF | 80 GB |
| A100 | 312 TF | 312 TF | 40/80 GB |
| V100 | 125 TF | 125 TF | 32 GB |

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Solution 1: Reduce batch size
python train.py --batch_size=8

# Solution 2: Enable ZeRO-1 (50% memory reduction)
torchrun --standalone --nproc_per_node=4 train.py --use_zero1=True

# Solution 3: Enable FSDP (75% memory reduction)
torchrun --standalone --nproc_per_node=4 train.py --use_fsdp=True --fsdp_min_num_params=500000
```

### Slow Training
```bash
# Enable compilation
python train.py --compile=True

# Increase batch size (better GPU utilization)
python train.py --batch_size=16 --gradient_accumulation_steps=20
```

### Low MFU (<20%)
- Check if data loading is bottleneck (use profiler)
- Increase batch size
- Enable `torch.compile()`
- Ensure using bfloat16/float16 precision

## 📁 Project Structure

```
enhanced_training_system/
├── model_components.py         # Component registry (norms, FFN, position encodings)
├── model_config.py             # Configuration system with presets
├── model_builder.py            # Modular model builder
├── train.py                    # Enhanced training script
├── training_logger.py          # Detailed JSON logging
├── configurator.py             # CLI configuration
├── model.py                    # Legacy GPT-2 (backward compatibility)
│
├── config/
│   ├── ARCH_GPT2.md           # GPT-2 architecture documentation
│   ├── ARCH_LLAMA.md          # LLaMA architecture documentation
│   ├── full_gpt2_124m.py      # GPT-2 124M config
│   ├── full_llama_124m.py     # LLaMA 124M config
│   ├── full_llama_1.36b.py    # LLaMA 1.36B production config
│   ├── full_team_124m.py      # Team's model_v1 config
│   ├── full_custom.py         # Custom experiment template
│   ├── preset_quick_test.py   # Quick testing preset
│   └── preset_gpt2_owt.py     # GPT-2 OpenWebText preset
│
├── docs/                       # Archived detailed documentation
│   ├── QUICK_START.md         # Quick testing guide
│   ├── START_HERE.md          # System overview
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── EXAMPLE_OUTPUT.md      # Terminal output examples
│   └── ...
│
├── data/                      # Dataset directories
├── compare_architectures.py    # Architecture comparison tool
├── test_imports.py            # Import verification
├── TESTING.md                 # **All testing commands**
├── README.md                  # This file
├── SYSTEM_OVERVIEW.md         # Technical documentation
└── requirements.txt           # Python dependencies
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
cd data/shakespeare
python prepare.py
cd ../..
```

### 3. Run Quick Test
```bash
python test_imports.py  # Verify system works
python train.py config/full_gpt2_124m.py --max_iters=50 --compile=False
```

### 4. See All Testing Commands
```bash
cat TESTING.md  # Complete testing guide
```

## 📚 Documentation

- **TESTING.md** - **All testing commands in one place**
- **SYSTEM_OVERVIEW.md** - Complete implementation details with code references
- **README.md** - This file (usage guide)
- **docs/** - Detailed documentation (archived)
- **JSON Logs** - Detailed per-run metrics in `out/`

## 🎓 References

1. **nanoGPT**: https://github.com/karpathy/nanoGPT
2. **GPT-2 Paper**: https://openai.com/research/better-language-models
3. **PaLM Paper** (Appendix B): https://arxiv.org/abs/2204.02311
4. **Insu Jang Analysis**: https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
5. **Epoch AI Backward/Forward**: https://epoch.ai/blog/backward-forward-FLOP-ratio
6. **PyTorch FSDP**: https://pytorch.org/docs/stable/fsdp.html
7. **ZeRO Paper**: https://arxiv.org/abs/1910.02054

## 📄 License

MIT License (same as nanoGPT)

## 🙏 Acknowledgments

- Andrej Karpathy for [nanoGPT](https://github.com/karpathy/nanoGPT)
- PyTorch team for FSDP and distributed training infrastructure
- Research teams behind academic FLOPs formulas and MFU analysis
