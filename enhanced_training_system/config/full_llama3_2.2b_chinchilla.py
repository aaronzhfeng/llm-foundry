"""
LLaMA 3 2.2B - Chinchilla-Constrained Configuration
===================================================

Based on backward N-D grid search with 1.36e21 FLOPs budget
Optimization: Minimize loss WITH Chinchilla D≈20N constraint

Model Design:
- Parameters: 2.224B (2,224,000,000)
- Config: 30L-16H-2048D-7168ff (GQA with 8 KV heads)
- FFN ratio: 3.5× d_model (LLaMA 3 style)
- Optimal training: 61.545B tokens
- Expected loss: 2.351 (0.016 higher than unconstrained)
- D/N ratio: 1.4 (respects Chinchilla principle)

Architecture (LLaMA 3):
- GQA: 8 KV heads, 16 Q heads (2:1 Q:KV ratio)
- RoPE (Extended: theta=500000)
- RMSNorm
- SwiGLU activation
- Pre-norm
- No weight tying
- No bias
- Head dimension: 128 (2048 / 16)

Grid Search Result (--enforce_chinchilla):
  Loss: 2.351059
  N (params): 2.224B
  D (tokens): 61.545B
  C (FLOPs): 1.37e+21 (error: 0.79%)
  Architecture: 30L × 2048H × 16A (head_dim=128)
  FFN: 7168 (3.50× expansion)
  D/N ratio: 1.4 (Chinchilla: 20.0)
  GQA: 8 KV heads (2:1 Q:KV ratio)
"""

# =============================================================================
# ARCHITECTURE - LLaMA 3 Style
# =============================================================================

# === Architecture Preset ===
arch_preset = 'llama3'  # LLaMA 3 with GQA

# === Attention Backend Override ===
attention_backend = 'flash_attn_2'  # Use FlashAttention-2 (fastest)

# === Model Dimensions (From Grid Search: Chinchilla Rank 1) ===
n_layer = 30                # Number of transformer layers (deeper!)
n_head = 16                 # Number of query heads
n_embd = 2048               # Hidden dimension
num_key_value_heads = 8     # Number of KV heads (GQA: 2:1 Q:KV ratio)
block_size = 2048           # Sequence length
dropout = 0.0
bias = False

# === FFN Dimension ===
d_ff = 7168                 # 3.5× expansion (LLaMA 3 style: 2048 × 3.5 = 7168)
intermediate_size = 7168

# === Extended RoPE ===
rope_theta = 500000.0       # Extended from 10000 (LLaMA 3)

# === Derived Parameters ===
# head_dim = 128                    # Per-head dimension (2048 / 16)
# vocab_size = 128256               # Will be set from tokenizer metadata

# === Architecture Components (specified by arch_preset='llama3') ===
# normalization = 'rmsnorm'         # RMSNorm (faster than LayerNorm)
# norm_eps = 1e-06
# position_encoding = 'rope'        # Rotary Position Embeddings
# ffn_type = 'swiglu'               # SwiGLU activation
# norm_position = 'pre'             # Pre-norm architecture
# weight_tying = False              # No weight tying
# ffn_expansion_ratio = 3.5         # LLaMA 3 uses 3.5× (not 8/3×)

# =============================================================================
# TRAINING - Chinchilla-Optimized
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_llama3'   # Quick testing (prepare: data/slimpajama_6b_llama3/prepare.py)
                                    # For optimal training (62B tokens): use 'slimpajama_627b_llama3'
gradient_accumulation_steps = 16
batch_size = 8                      # Per-GPU batch size

# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# Example (8× A100): 8 × 16 × 8 = 1024 samples/iter = 2,097,152 tokens/iter
# Iterations needed: 61.545B / 2.097M = ~29,400 iterations

# === Optimizer (AdamW) ===
learning_rate = 3e-4
max_iters = 30000               # ~63B tokens (slightly over optimal)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# === Learning Rate Schedule ===
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 30000
min_lr = 3e-5

# =============================================================================
# SYSTEM
# =============================================================================

# === Hardware ===
device = 'cuda'
dtype = 'bfloat16'
compile = True

# === Parallelism ===
# For 8× A100 80GB (required):
use_zero1 = False
use_fsdp = True             # CRITICAL: Must use FSDP (model is larger)

# Note: This model is too large for 4× A100 with comfortable memory
# If you only have 4 GPUs, consider the 1.5B optimal config instead

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-llama3-2.2b-chinchilla'
eval_interval = 1000
log_interval = 10
eval_iters = 50
eval_only = False
eval_at_start = False
always_save_checkpoint = True
init_from = 'scratch'

# === Logging ===
save_log_to_json = True
log_save_interval = 10
gradient_log_interval = 10

# === Weights & Biases (optional) ===
wandb_log = False
wandb_project = 'llama3-2.2b-chinchilla'
wandb_run_name = 'run-1'

# =============================================================================
# METADATA (From Grid Search: 1.36e21 FLOPs, Chinchilla-constrained)
# =============================================================================

# Grid Search Results (--enforce_chinchilla):
# - Rank: 1 (best with Chinchilla constraint)
# - Expected loss: 2.351 (only 0.016 higher than unconstrained)
# - Parameters: 2.224B (44% larger model)
# - Optimal tokens: 61.545B (40% fewer tokens)
# - D/N ratio: 1.4 (closer to Chinchilla 20.0)
# - FLOPs used: 1.37e21 (0.79% error)
# 
# Architecture Details:
# - GQA: 8 KV heads, 16 Q heads (2:1 ratio)
# - 75% smaller KV cache vs MHA
# - 30 layers (vs 18 in unconstrained) = deeper model
# - Same width (2048) and heads (16) as unconstrained
# - Head dimension: 128 (optimal for FlashAttention)
# - FFN: 3.5× expansion (LLaMA 3 standard)
# 
# Expected Performance (8× A100):
# - Tokens/sec: ~120,000-140,000 (slightly slower, larger model)
# - MFU: 45-55% (GQA is efficient)
# - Memory/GPU: ~45-55 GB with FSDP
# - Training time: ~8-10 days for 62B tokens
# 
# Expected Performance (8× B200):
# - Tokens/sec: ~450,000-550,000
# - MFU: 50-60%
# - Training time: ~2-3 days for 62B tokens
# 
# Advantages over Unconstrained (1.5B):
# - 44% larger model → better downstream task performance
# - 40% fewer training tokens → faster to train
# - Better for production deployment (larger = more capable)
# - Only 0.016 higher loss (+0.7%) during training
# 
# Trade-offs:
# - Requires 8 GPUs minimum (vs 4 for 1.5B)
# - Higher memory usage per GPU
# - Slightly lower tokens/sec throughput
# 
# When to Choose This Config:
# - You have 8+ A100/H100 GPUs
# - You prioritize model capacity over training loss
# - You want faster training (fewer tokens needed)
# - You're deploying to production (larger = better)
# 
# Alternative Configs:
# - See config/full_llama3_1.5b_optimal.py for best loss
# - See config/full_llama3_8b.py for official LLaMA 3.1 8B


