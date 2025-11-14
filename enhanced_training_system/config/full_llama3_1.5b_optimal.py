"""
LLaMA 3 1.5B - Optimal Configuration (Unconstrained)
====================================================

Based on backward N-D grid search with 1.36e21 FLOPs budget
Optimization: Minimize loss without Chinchilla constraint

Model Design:
- Parameters: 1.545B (1,545,000,000)
- Config: 18L-16H-2048D-7168ff (GQA with 8 KV heads)
- FFN ratio: 3.5× d_model (LLaMA 3 style)
- Optimal training: 101.909B tokens
- Expected loss: 2.335 (best achievable with this budget)

Architecture (LLaMA 3):
- GQA: 8 KV heads, 16 Q heads (2:1 Q:KV ratio)
- RoPE (Extended: theta=500000)
- RMSNorm
- SwiGLU activation
- Pre-norm
- No weight tying
- No bias
- Head dimension: 128 (2048 / 16)

Grid Search Result:
  Loss: 2.335120
  N (params): 1.545B
  D (tokens): 101.909B
  C (FLOPs): 1.36e+21 (error: 0.13%)
  Architecture: 18L × 2048H × 16A (head_dim=128)
  FFN: 7168 (3.50× expansion)
  D/N ratio: 3.3 (Chinchilla: 20.0)
  GQA: 8 KV heads (2:1 Q:KV ratio)
"""

# =============================================================================
# ARCHITECTURE - LLaMA 3 Style
# =============================================================================

# === Architecture Preset ===
arch_preset = 'llama3'  # LLaMA 3 with GQA

# === Attention Backend Override ===
attention_backend = 'flash_attn_2'  # Use FlashAttention-2 (fastest)

# === Model Dimensions (From Grid Search: Rank 1) ===
n_layer = 18                # Number of transformer layers
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
# TRAINING - Optimal Settings
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_llama3'   # Quick testing (prepare: data/slimpajama_6b_llama3/prepare.py)
                                    # For optimal training (102B tokens): use 'slimpajama_627b_llama3'
gradient_accumulation_steps = 16
batch_size = 8                      # Per-GPU batch size

# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# Example (8× A100): 8 × 16 × 8 = 1024 samples/iter = 2,097,152 tokens/iter
# Iterations needed: 101.909B / 2.097M = ~48,600 iterations

# === Optimizer (AdamW) ===
learning_rate = 3e-4
max_iters = 50000               # ~105B tokens (slightly over optimal)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# === Learning Rate Schedule ===
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 50000
min_lr = 3e-5

# =============================================================================
# SYSTEM
# =============================================================================

# === Hardware ===
device = 'cuda'
dtype = 'bfloat16'
compile = True

# === Parallelism ===
# For 8× A100 80GB (recommended):
use_zero1 = False
use_fsdp = True             # Use FSDP for 8+ GPUs

# For 4× A100 80GB (tight):
# use_zero1 = True
# use_fsdp = False
# batch_size = 4
# gradient_accumulation_steps = 32

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-llama3-1.5b-optimal'
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
wandb_project = 'llama3-1.5b-optimal'
wandb_run_name = 'run-1'

# =============================================================================
# METADATA (From Grid Search: 1.36e21 FLOPs)
# =============================================================================

# Grid Search Results:
# - Rank: 1 (best configuration)
# - Expected loss: 2.335
# - Parameters: 1.545B
# - Optimal tokens: 101.909B
# - D/N ratio: 3.3 (prioritizes quality over Chinchilla)
# - FLOPs used: 1.36e21 (0.13% error)
# 
# Architecture Details:
# - GQA: 8 KV heads, 16 Q heads (2:1 ratio)
# - 75% smaller KV cache vs MHA
# - Head dimension: 128 (optimal for FlashAttention)
# - FFN: 3.5× expansion (LLaMA 3 standard)
# 
# Expected Performance (8× A100):
# - Tokens/sec: ~140,000-160,000
# - MFU: 45-55% (GQA is efficient)
# - Memory/GPU: ~35-45 GB with FSDP
# - Training time: ~12-15 days for 102B tokens
# 
# Expected Performance (8× B200):
# - Tokens/sec: ~500,000-600,000
# - MFU: 50-60%
# - Training time: ~3-4 days for 102B tokens
# 
# Comparison to Chinchilla:
# - This config prioritizes absolute best loss
# - Uses more tokens (102B vs 62B for Chinchilla)
# - Smaller model (1.5B vs 2.2B for Chinchilla)
# - Better for: research, loss optimization
# 
# Alternative Configs:
# - See config/full_llama3_2.2b_chinchilla.py for larger model
# - See config/full_llama3_8b.py for official LLaMA 3.1 8B


