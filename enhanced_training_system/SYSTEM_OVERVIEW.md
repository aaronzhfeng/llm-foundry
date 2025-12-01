# System Overview: Enhanced GPT Training with Detailed MFU Analysis

This document provides a comprehensive technical overview of the enhanced GPT training system, including implementation details, formulas, and code references.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [MFU Calculation (Enhanced)](#mfu-calculation-enhanced)
3. [Memory Tracking](#memory-tracking)
4. [Gradient Monitoring](#gradient-monitoring)
5. [Parallelization Strategies](#parallelization-strategies)
6. [Training Loop Details](#training-loop-details)
7. [Code References](#code-references)
8. [Quick Reference](#quick-reference)

---

## Architecture Overview

### Modular, Configurable Transformer System

This implementation supports **fully configurable architectures** - mix and match components without code changes!

**Supported Architectures:**
- **GPT-2 Standard**: Learned pos + LayerNorm + GELU
- **LLaMA 2**: RoPE + RMSNorm + SwiGLU + MHA
- **LLaMA 3**: RoPE + RMSNorm + SwiGLU + GQA (Grouped Query Attention)
- **Team model_v1**: Same as LLaMA 2
- **Custom Combinations**: Any mix you want!

**Configurable Components:**

| Component | Options | Reference |
|-----------|---------|-----------|
| **Normalization** | layernorm, layernorm_nobias, rmsnorm | `model_components.py:28-73` |
| **Activation** | gelu, silu, relu, leaky_relu | `model_components.py:178-192` |
| **Position Encoding** | learned_absolute, rope, none | `model_components.py:82-162` |
| **Attention Type** | MHA (multi-head), GQA (grouped query) | `model_builder.py:31-104` |
| **Attention Backend** | sdpa (FlashAttention), manual | `model_builder.py:31-104` |
| **Norm Position** | pre, post | `model_builder.py:108-132` |
| **FFN Type** | standard (4x), swiglu (8/3x) | `model_components.py:167-215` |
| **Bias** | True, False | Throughout |
| **Weight Tying** | True, False | `model_builder.py:180-181` |
| **Dropout** | 0.0-1.0 | Throughout |

**Configuration System:**
```python
# Reference: model_config.py:29-118
@dataclass
class ModelArchitectureConfig:
    # Model size
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    num_key_value_heads: Optional[int] = None  # For GQA (None = MHA)
    block_size: int = 1024
    vocab_size: int = 50304
    
    # Architecture choices
    normalization: Literal['layernorm', 'layernorm_nobias', 'rmsnorm']
    activation: Literal['gelu', 'silu', 'relu', 'leaky_relu']
    position_encoding: Literal['learned_absolute', 'rope', 'none']
    norm_position: Literal['pre', 'post']
    ffn_type: Literal['standard', 'swiglu']
    attention_backend: Literal['sdpa', 'manual']
    
    # Component options
    bias: bool = False
    weight_tying: bool = True
    dropout: float = 0.0
```

**Default Preset Architectures:**

**GPT-2** (Reference: `model_config.py:156-177`):
- Learned absolute positions
- LayerNorm (no bias)
- GELU activation, standard FFN (4x)
- Post-norm, weight tying
- Multi-Head Attention (MHA)

**LLaMA 2** (Reference: `model_config.py:180-203`):
- RoPE positions
- RMSNorm
- SwiGLU activation (8/3x expansion)
- Pre-norm, no weight tying
- Multi-Head Attention (MHA)

**LLaMA 3** (Reference: `model_config.py:206-230`):
- RoPE positions (extended: theta=500000)
- RMSNorm
- SwiGLU activation (3.5x expansion)
- Pre-norm, no weight tying
- **Grouped Query Attention (GQA)**: 8 KV heads
- 128K vocabulary

**Usage:**
```python
# In config file:
arch_preset = 'llama3'  # Or 'llama', 'gpt2', 'hybrid', 'team', 'custom'

# Or customize:
arch_preset = 'custom'
normalization = 'rmsnorm'
position_encoding = 'rope'
ffn_type = 'swiglu'
num_key_value_heads = 8  # Enable GQA
...
```

### Component Selection System

**Factory Pattern** (Reference: `model_components.py:218-247`):
```python
# Components selected from registries at runtime
norm_layer = build_norm(config.normalization, ndim, eps)
ffn_layer = build_ffn(config.ffn_type, d_model, d_ff, bias, dropout, activation)
pos_encoding = build_position_encoding(config.position_encoding, max_seq_len, ...)

# Registries:
NORM_REGISTRY = {'layernorm': LayerNormWithBias, 'rmsnorm': RMSNorm, ...}
FFN_REGISTRY = {'standard': StandardFFN, 'swiglu': SwiGLUFFN}
POSITION_ENCODING_REGISTRY = {'learned_absolute': Learned..., 'rope': RoPE..., ...}
```

**Adding New Components:**
1. Create component class in `model_components.py`
2. Add to appropriate registry
3. Use in config files - no other code changes!

**Example - Adding ALiBi:**
```python
# 1. Implement in model_components.py
class ALiBiPositionEncoding(nn.Module):
    def __init__(self, num_heads, max_seq_len):
        ...

# 2. Register
POSITION_ENCODING_REGISTRY['alibi'] = ALiBiPositionEncoding

# 3. Use in config
position_encoding = 'alibi'  # Done!
```

---

## Grouped Query Attention (GQA)

### What is GQA?

**Grouped Query Attention** (introduced in LLaMA 3) reduces memory and computation by **sharing KV heads across multiple Q heads**.

**Multi-Head Attention (MHA)** - Traditional approach:
- N query heads, N key heads, N value heads
- Example: 32 Q, 32 K, 32 V heads
- Large KV cache for inference

**Grouped Query Attention (GQA)** - LLaMA 3 approach:
- N query heads, M key heads, M value heads (M < N)
- Example: 32 Q heads, 8 KV heads (4:1 ratio)
- 4× smaller KV cache for inference
- Minimal quality loss (~1-2% perplexity increase)

### Implementation Details

**Reference:** `model_builder.py:31-104` (ConfigurableAttention)

```python
# GQA configuration
n_head = 32                    # Query heads
num_key_value_heads = 8        # KV heads (shared)
n_rep = n_head // num_key_value_heads  # 4 Q heads per KV head

# Forward pass
q = self.c_q(x)                # [B, T, 32*head_dim]
kv = self.c_kv(x)              # [B, T, 2*8*head_dim]  (smaller!)

# Reshape and repeat KV
k = k.repeat_interleave(n_rep, dim=1)  # Broadcast to 32 heads
v = v.repeat_interleave(n_rep, dim=1)

# Standard attention
attn = (q @ k.transpose(-2, -1)) / sqrt(d_k)
out = attn @ v
```

### Parameter Count Impact

**MHA** (32 heads, head_dim=128, d_model=4096):
```python
Q projection: 4096 × 4096 = 16.8M params
K projection: 4096 × 4096 = 16.8M params
V projection: 4096 × 4096 = 16.8M params
Total: 50.3M params per layer
```

**GQA** (32 Q heads, 8 KV heads, head_dim=128, d_model=4096):
```python
Q projection: 4096 × 4096 = 16.8M params
K projection: 4096 × 1024 = 4.2M params  (75% smaller!)
V projection: 4096 × 1024 = 4.2M params  (75% smaller!)
Total: 25.2M params per layer (50% reduction)
```

### FLOPs Calculation Impact

**Reference:** `training_planner/analyze.py`

GQA reduces FLOPs for QKV projections:
```python
# MHA
qkv_flops = 3 * (2 * seq_len * d_model * d_model)

# GQA
q_flops = 2 * seq_len * d_model * d_model
kv_flops = 2 * seq_len * d_model * (n_kv_heads * head_dim)  # Smaller!
total_flops = q_flops + kv_flops
```

For LLaMA 3.1 8B (32 Q heads, 8 KV heads):
- **Attention projection FLOPs reduced by ~33%**
- **Total model FLOPs reduced by ~8-10%** (attention is ~30% of total)
- **Inference memory reduced by 75%** (KV cache)

### When to Use GQA

**Use GQA when:**
- ✅ Deploying for inference (huge KV cache savings)
- ✅ Training large models (>7B parameters)
- ✅ Following LLaMA 3 architecture
- ✅ Memory constrained

**Use MHA when:**
- ✅ Small models (<1B parameters)
- ✅ Maximum quality is critical
- ✅ Following GPT-2/LLaMA 2 architecture

---

## MFU Calculation (Enhanced & Architecture-Aware)

### Academic Formula Implementation

Unlike the simplified `6N + 12*L*H*Q*T` formula, this implementation uses detailed component-level FLOPs calculation that **adapts to your architecture**:

- **Standard FFN**: 2 linear layers (up, down)
- **SwiGLU FFN**: 3 linear layers (gate, value, output) - ~50% more FLOPs
- **RoPE**: Adds rotation overhead
- **RMSNorm**: Slightly cheaper than LayerNorm

**Reference**: `model_builder.py:222-269` - Architecture-aware FLOPs calculation

#### Forward Pass FLOPs Per Layer

```python
# Reference: model.py:381-399
# Formula: FLOPs = 12SBH² + 2aS²BH per layer

# Attention block:
attention_qkv_flops = 6 * S * H * H          # Q, K, V projections
attention_scores_flops = a * S * S * H        # QK^T computation
attention_output_flops = a * S * S * H        # Attention @ V
attention_proj_flops = 2 * S * H * H          # Output projection
attention_flops = sum(above)

# FFN block:
ffn_up_flops = 2 * S * H * D_ff               # Up projection
ffn_down_flops = 2 * S * D_ff * H             # Down projection
ffn_flops = ffn_up_flops + ffn_down_flops

# Total per layer:
flops_per_layer = attention_flops + ffn_flops + layernorm_flops
```

Where:
- **S** = sequence_length (block_size)
- **H** = hidden_size (n_embd)
- **a** = num_attention_heads (n_head)
- **D_ff** = intermediate_size (4 × H for GPT-2)
- **L** = num_layers (n_layer)

#### Training FLOPs (Forward + Backward)

```python
# PaLM MFU Formula (Appendix B)
# Reference: model_builder.py:383-403
N_billion = non_embedding_params / 1e9
Q = hidden_dim // num_heads
T = sequence_length

non_attn_flops = 6.0 * N_billion  # GFLOPs/token
attn_flops = 12.0 * L * H * Q * T / 1e9  # GFLOPs/token
training_flops_per_token = non_attn_flops + attn_flops  # GFLOPs/token
```

**PaLM Formula (6N + 12LHQT)**: Industry-standard MFU denominator
- Used by PaLM, GPT-3, Gopher, MT-NLG papers
- Accounts for forward + backward in matmul FLOPs
- FMA counted as 2 FLOPs; excludes rematerialization
- Reference: https://epoch.ai/blog/backward-forward-FLOP-ratio

#### MFU Calculation

```python
# Reference: model.py:410-424
tokens_per_iter = S * fwdbwd_per_iter
flops_per_iter = training_flops_per_token * tokens_per_iter
flops_achieved = flops_per_iter / dt  # FLOPs per second
mfu = flops_achieved / hardware_peak_flops
```

#### Hardware Specifications (with B200)

```python
# Reference: model.py:427-437
hardware_specs = {
    'cuda': {
        'B200': {'bf16': 2250e12, 'fp16': 2250e12, 'fp32': 90e12},    # HGX B200 Dense (Sparse: 4500 TFLOPS)
        'H200': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12},
        'H100': {'bf16': 989e12,  'fp16': 989e12,  'fp32': 67e12},
        'A100': {'bf16': 312e12,  'fp16': 312e12,  'fp32': 19.5e12},
                'A6000': {'bf16': 155.0e12, 'fp16': 155.0e12, 'fp32': 38.7e12},  # Dense FP16
        'V100': {'bf16': 125e12,  'fp16': 125e12,  'fp32': 15.7e12},
    }
}
```

#### MFU Breakdown Output

```python
# Reference: model.py:454-472
return {
    'mfu': mfu,                              # MFU as decimal
    'mfu_percent': mfu * 100,                # MFU as percentage
    'flops_achieved': flops_achieved,         # Actual FLOPs/s
    'flops_per_token': training_flops_per_token,  # FLOPs per token
    'tokens_per_sec': tokens_per_sec,         # Throughput
    'hardware_peak_flops': hardware_peak_flops,   # Peak FLOPs
    'hardware_peak_tflops': hardware_peak_flops / 1e12,
    'achieved_tflops': flops_achieved / 1e12,
    'gpu_name': gpu_name,                     # Auto-detected
    'precision': precision_key,               # bf16/fp16/fp32
    'num_gpus': num_gpus,
    'attention_flops_per_layer': attention_flops,
    'ffn_flops_per_layer': ffn_flops,
    'attention_to_ffn_ratio': attention_flops / ffn_flops,
}
```

**Academic References:**
1. Insu Jang (2022): "Analysis of Transformer Model"  
   https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
2. Epoch AI (2024): "Backward-Forward FLOP Ratio"  
   https://epoch.ai/blog/backward-forward-FLOP-ratio

---

## Memory Tracking

### Memory Statistics Per Iteration

```python
# Reference: model.py:474-483
def get_memory_stats(self):
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,      # Currently allocated
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,        # Reserved by allocator
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,  # Peak allocation
        'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,    # Peak reserved
    }
```

### Memory Breakdown (Typical GPT-2 124M on A100)

| Component | Standard DDP | ZeRO-1 | FSDP |
|-----------|--------------|--------|------|
| Model Parameters | 0.5 GB | 0.5 GB | ~0.06 GB (sharded) |
| Gradients | 0.5 GB | 0.5 GB | ~0.06 GB (sharded) |
| Optimizer States | 2.0 GB | ~0.25 GB (sharded) | ~0.25 GB (sharded) |
| Activations | Variable | Variable | Variable |
| **Total** | **~3 GB** | **~1.25 GB** | **~0.37 GB** |

---

## Gradient Monitoring

### Gradient Statistics

```python
# Reference: model.py:485-508
def get_gradient_stats(self):
    return {
        'global_norm': np.sqrt(sum(n**2 for n in grad_norms)),  # L2 norm of all gradients
        'mean_layer_norm': np.mean(grad_norms),                 # Average layer-wise norm
        'max_layer_norm': np.max(grad_norms),                   # Max layer norm
        'min_layer_norm': np.min(grad_norms),                   # Min layer norm
        'grad_mean': float(np.mean(grad_values)),               # Mean gradient value
        'grad_std': float(np.std(grad_values)),                 # Std dev of gradients
        'grad_min': float(np.min(grad_values)),                 # Min gradient
        'grad_max': float(np.max(grad_values)),                 # Max gradient
    }
```

**Usage in Training:**
- Logged every `gradient_log_interval` iterations (default: 10) to avoid overhead
- Used to monitor training stability
- Helps detect gradient explosion/vanishing

---

## Parallelization Strategies

### 1. Standard DDP (Data Parallel)

**Implementation:** `train.py:331`
```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

**Memory Profile:**
- Full model copy per GPU
- Gradient synchronization after backward
- Best for: Models that fit in single GPU memory

### 2. ZeRO-1 (Optimizer State Sharding)

**Implementation:** `train.py:334-355`
```python
from torch.distributed.optim import ZeroRedundancyOptimizer

optimizer = ZeroRedundancyOptimizer(
    optim_groups,
    optimizer_class=torch.optim.AdamW,
    lr=learning_rate,
    betas=(beta1, beta2),
)
```

**Memory Savings:**
- ~50% reduction (4 GPUs)
- ~75% reduction (8 GPUs)
- Only optimizer states are sharded

**Checkpoint Handling:**
```python
# Reference: train.py:555-565
optimizer.consolidate_state_dict(to=0)  # Gather to rank 0 before saving
```

### 3. FSDP (Fully Sharded Data Parallel)

**Implementation:** `train.py:309-329`
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,
    use_orig_params=True,  # Required for torch.compile compatibility
)
```

**Memory Savings:**
- ~75% reduction (4 GPUs)
- ~88% reduction (8 GPUs)
- Parameters, gradients, AND optimizer states all sharded

**Checkpoint Handling:**
```python
# Reference: train.py:535-551
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    save_policy,
    optim_policy,
):
    model_state = model.state_dict()
    optim_state = FSDP.optim_state_dict(model, optimizer)
```

### Comparison Table

| Feature | DDP | ZeRO-1 | FSDP |
|---------|-----|--------|------|
| **Parameter Sharding** | ❌ | ❌ | ✅ |
| **Gradient Sharding** | ❌ | ❌ | ✅ |
| **Optimizer Sharding** | ❌ | ✅ | ✅ |
| **Memory/GPU (8 GPUs)** | 3.0 GB | 1.5 GB | 0.4 GB |
| **Speed vs Baseline** | 100% | 90-95% | 85-90% |
| **Max Model Size** | 1x | 2x | **8x** |
| **Complexity** | Low | Low | High |

---

## Training Loop Details

### Startup Report

**Implementation:** `train.py:460-512`

Displays:
- Model architecture (parameters, layers, dimensions)
- Training configuration (batch sizes, learning rates)
- Hardware information (GPU type, memory, parallelism)
- Theoretical performance (peak FLOPs, expected throughput)

### Per-Iteration Logging

**Implementation:** `train.py:629-673`

**Standard Output:**
```
────────────────────────────────────────────────────────────────────────────────
📍 Iter      5 │ Loss: 10.6456 │ Time: 4298ms │ LR: 1.50e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 32.45% │ Achieved: 101.2 TF │ Peak: 312.0 TF
   Tokens/s: 3,557 │ FLOPs/token: 28.5 GF
💾 Memory: 12.34 GB alloc │ 15.67 GB peak │ 16.00 GB reserved
📊 Gradients: norm=2.3456 │ mean=-1.23e-05 │ std=3.45e-04
```

**JSON Logging:**
```python
# Reference: training_logger.py:81-103
logger.log_iter_detailed(
    iter_num,
    loss,
    dt_ms,
    mfu_breakdown,      # Full MFU breakdown dict
    memory_stats,       # Memory statistics dict
    grad_stats          # Gradient statistics dict (every N iters)
)
```

### Evaluation Steps

**Implementation:** `train.py:508-527`

- Runs every `eval_interval` iterations
- Estimates loss on train and validation sets
- Saves checkpoints if validation loss improves
- Logs to JSON and (optionally) WandB

---

## Code References

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `model_builder.py` | 345 | **Modular model builder** (main model code) |
| `model_config.py` | 250 | **Configuration system** (all arch options) |
| `model_components.py` | 250 | **Component registry** (norms, activations, etc.) |
| `train.py` | 720 | Training loop + startup report |
| `training_logger.py` | 268 | Detailed JSON logging |
| `configurator.py` | 48 | Command-line config system |
| `model.py` | 486 | Legacy GPT-2 (kept for compatibility) |

### Important Functions

#### model_builder.py (Modular System)
- `ConfigurableGPT.__init__()` (line 138): Build model from config
- `ConfigurableGPT.estimate_mfu_detailed()` (line 222): Architecture-aware MFU
- `ConfigurableGPT.get_memory_stats()` (line 280): Memory tracking
- `ConfigurableGPT.get_gradient_stats()` (line 290): Gradient monitoring
- `ConfigurableGPT.configure_optimizers()` (line 319): Optimizer with weight decay
- `TransformerBlock.forward()` (line 124): Pre-norm or post-norm logic

#### model_config.py
- `ModelArchitectureConfig` (line 29): Complete configuration dataclass
- `get_preset_config()` (line 247): Load preset architectures
- `config.get_architecture_name()` (line 123): Generate descriptive name
- `config.get_architecture_summary()` (line 137): Human-readable summary

#### model_components.py
- `build_norm()` (line 218): Norm factory function
- `build_ffn()` (line 230): FFN factory function
- `build_position_encoding()` (line 242): Position encoding factory
- Component registries (lines 66, 165, 217): Extensible registries

#### train.py
- Startup report (line 460-512): Comprehensive initialization info
- Training loop (line 514-687): Main iteration loop with detailed logging
- Gradient accumulation (line 585-604): Micro-batch handling
- Checkpoint saving (line 530-577): FSDP/ZeRO-1 aware checkpointing

#### training_logger.py
- `TrainingLogger.log_iter_detailed()` (line 81): Log with full breakdown
- `TrainingLogger.log_startup_info()` (line 147): Log initialization
- `TrainingLogger.finalize()` (line 175): Generate summary statistics

---

## Quick Reference

### Running Commands

```bash
# Single GPU - Test different architectures
python train.py config/full_gpt2_124m.py        # GPT-2 architecture (MHA)
python train.py config/full_llama_124m.py       # LLaMA 2 architecture (MHA)
python train.py config/full_llama3_8b.py        # LLaMA 3 architecture (GQA)
python train.py config/arch_team.py        # Team's model_v1
python train.py config/full_custom.py      # Your custom mix

# Override architecture on command line
python train.py config/full_gpt2_124m.py --normalization=rmsnorm --position_encoding=rope

# Test GQA (Grouped Query Attention)
python train.py config/full_custom.py --arch_preset=llama3 --n_head=16 --num_key_value_heads=4

# Multi-GPU DDP (4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_124m.py

# Multi-GPU with ZeRO-1 (50% memory reduction)
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_124m.py --use_zero1=True

# Multi-GPU with FSDP (75% memory reduction)
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_124m.py --use_fsdp=True

# LLaMA 3.1 8B with FSDP (8 GPUs)
torchrun --standalone --nproc_per_node=8 train.py config/full_llama3_8b.py --use_fsdp=True

# HGX B200 (8 GPUs, optimal)
torchrun --standalone --nproc_per_node=8 train.py config/full_llama3_8b.py --use_fsdp=True
```

### Key Configuration Flags

```bash
# Architecture (modular system)
--arch_preset=llama3         # 'gpt2', 'llama', 'llama3', 'hybrid', 'team', 'custom'
--normalization=rmsnorm      # 'layernorm', 'layernorm_nobias', 'rmsnorm'
--activation=gelu            # 'gelu', 'silu', 'relu', 'leaky_relu'
--position_encoding=rope     # 'learned_absolute', 'rope', 'none'
--norm_position=pre          # 'pre', 'post'
--ffn_type=swiglu           # 'standard', 'swiglu'
--weight_tying=False        # True/False

# Attention (GQA support)
--n_head=32                  # Number of query heads
--num_key_value_heads=8      # Number of KV heads (for GQA, <n_head)
                            # If not set, defaults to n_head (MHA)

# Training
--batch_size=12              # Micro-batch size per GPU
--gradient_accumulation_steps=40  # Gradient accumulation
--learning_rate=6e-4         # Max learning rate
--max_iters=600000           # Training iterations

# System
--compile=True               # Use torch.compile()
--dtype=bfloat16            # Precision (bfloat16/float16/float32)
--use_zero1=True            # Enable ZeRO-1
--use_fsdp=True             # Enable FSDP
--log_interval=10           # Log every N iterations
--gradient_log_interval=50  # Log gradients every N iterations
```

### Expected Performance (GPT-2 124M on A100)

| Configuration | Tokens/s | MFU | Memory/GPU | Notes |
|---------------|----------|-----|------------|-------|
| 1x A100, compile=True | ~4,000 | ~35% | 12 GB | Baseline |
| 4x A100, DDP | ~15,000 | ~35% | 3 GB | Standard multi-GPU |
| 4x A100, ZeRO-1 | ~14,000 | ~33% | 1.5 GB | -5% speed, -50% mem |
| 4x A100, FSDP | ~13,000 | ~30% | 0.4 GB | -10% speed, -87% mem |

---

## Formula Summary

### MFU Calculation (Complete)

```python
# 1. Forward pass FLOPs per layer
attention_flops = 6*S*H² + 2*a*S²*H
ffn_flops = 4*H*D_ff
flops_per_layer = attention_flops + ffn_flops

# 2. Total forward FLOPs per token
forward_flops_per_token = L * flops_per_layer / S

# 3. Training FLOPs (PaLM formula: 6N + 12LHQT)
training_flops_per_token = 6.0 * N_billion + 12.0 * L * H * Q * T / 1e9  # GFLOPs

# 4. Iteration FLOPs
tokens_per_iter = S * batch_size * gradient_accum_steps
flops_per_iter = training_flops_per_token * tokens_per_iter

# 5. Achieved FLOPs/s
flops_achieved = flops_per_iter / iteration_time

# 6. MFU
mfu = flops_achieved / hardware_peak_flops
```

---

## References

### Academic Papers
1. **GPT-2**: Radford et al. (2019) - Language Models are Unsupervised Multitask Learners
2. **PaLM**: Chowdhery et al. (2022) - PaLM: Scaling Language Modeling with Pathways
3. **ZeRO**: Rajbhandari et al. (2020) - ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
4. **FSDP**: Zhao et al. (2023) - PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

### Technical References
1. **Insu Jang (2022)**: Analysis of Transformer Model  
   https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
2. **Epoch AI (2024)**: Backward-Forward FLOP Ratio  
   https://epoch.ai/blog/backward-forward-FLOP-ratio
3. **nanoGPT**: https://github.com/karpathy/nanoGPT
4. **PyTorch FSDP**: https://pytorch.org/docs/stable/fsdp.html

---

## Adding New Components

The modular system makes it easy to extend with new components.

### Example: Adding ALiBi Position Encoding

**Step 1:** Implement in `model_components.py`
```python
class ALiBiPositionEncoding(nn.Module):
    """ALiBi: Attention with Linear Biases"""
    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        # Generate slopes for each head
        slopes = self._get_slopes(num_heads)
        # ... implementation
    
    def forward(self, token_positions):
        return None  # Applied as attention bias
    
    def get_attention_bias(self, seq_len):
        # Return bias matrix for attention
        ...
```

**Step 2:** Register
```python
POSITION_ENCODING_REGISTRY['alibi'] = ALiBiPositionEncoding
```

**Step 3:** Use in config
```python
# config/full_custom.py
position_encoding = 'alibi'
```

**Done!** No changes needed to `model_builder.py`, `train.py`, or anywhere else.

### Example: Adding GeGLU Activation

**Step 1:** Implement in `model_components.py`
```python
class GeGLUFFN(nn.Module):
    """GeGLU: GELU-based GLU variant"""
    def __init__(self, d_model, d_ff, bias=False, dropout=0.0, **kwargs):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_value = nn.Linear(d_model, d_ff, bias=bias)
        self.w_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gate = F.gelu(self.w_gate(x))  # GELU instead of SiLU
        value = self.w_value(x)
        return self.dropout(self.w_out(gate * value))
```

**Step 2:** Register
```python
FFN_REGISTRY['geglu'] = GeGLUFFN
```

**Step 3:** Use
```python
ffn_type = 'geglu'
```

---

## File Organization

### Core Implementation Files

| Category | Files | Purpose |
|----------|-------|---------|
| **Model** | `model_components.py`, `model_config.py`, `model_builder.py` | Modular architecture system |
| **Training** | `train.py`, `training_logger.py`, `configurator.py` | Training loop and logging |
| **Config** | `config/arch_*.py`, `config/train_*.py` | Architecture and training configs |
| **Tools** | `compare_architectures.py`, `test_imports.py` | Utilities |
| **Docs** | `README.md`, `SYSTEM_OVERVIEW.md`, `TESTING.md` | Documentation |
| **Legacy** | `model.py` | Backward compatibility |

### Documentation Structure

**Main Documentation (Root):**
- `README.md` - Complete usage guide
- `SYSTEM_OVERVIEW.md` - Technical details with code references
- `TESTING.md` - All testing commands

**Detailed Documentation (docs/):**
- Archived detailed guides and examples
- See individual files for specific topics

---

**Last Updated**: 2025-01-03  
**Implementation Status**: ✅ Complete with B200 support and modular architecture system

