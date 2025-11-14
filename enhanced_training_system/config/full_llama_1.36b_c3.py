"""
LLaMA 1.36B Architecture - Production Configuration
===================================================

Based on scaling law optimization from: info/llama_1.36e21_32kV.json

Model Design:
- Target: 1.36B parameters (Chinchilla compute-optimal)
- Actual: 1.358B parameters (18L-18H-2304D-6656ff)
- Design: C-3 (single parameter change) - widest FFN
- Optimal training: ~27B tokens (D ≈ 20N per Chinchilla)
- Target loss: 2.37 (theoretical minimum)

Architecture (LLaMA-style):
- RoPE (Rotary Position Embeddings)
- RMSNorm (faster than LayerNorm)
- SwiGLU activation (d_ff = 2.75 × d_model)
- Pre-norm (better training stability)
- No weight tying
- No bias
- Head dimension: 128 (FlashAttention optimal)

Alternative Designs (all ~1.36B):
- C-1 (current): 24L-16H-2048D-5632ff = 1.364B (deeper, narrower)
- C-2 (minimal change): 19L-18H-2304D-6144ff = 1.358B (classic 8/3 FFN ratio)
- C-3 (single param): 18L-18H-2304D-6656ff = 1.358B (wider FFN only)

Usage:
    # Quick test on 6B tokens
    python train.py config/full_llama_1.36b.py

    # Multi-GPU training (4x A100)
    torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py

    # Production training (8x B200)
    torchrun --standalone --nproc_per_node=8 train.py \\
        config/full_llama_1.36b.py \\
        --dataset=slimpajama_627b \\
        --use_fsdp=True
"""

# =============================================================================
# ARCHITECTURE - What the model IS
# =============================================================================

# === Architecture Preset ===
arch_preset = 'llama'  # Use LLaMA components: RoPE + RMSNorm + SwiGLU + Pre-norm

# === Attention Backend Override ===
attention_backend = 'flash_attn_2'  # Use FlashAttention-2 (fastest, ~2x speedup)

# === Model Dimensions (C-3: Keep Depth, Widen FFN) ===
n_layer = 18                # Number of transformer layers (same as original)
n_head = 18                 # Number of attention heads
n_embd = 2304               # Hidden dimension / embedding dimension
block_size = 2048           # Maximum sequence length / context window
dropout = 0.0               # Dropout rate (LLaMA uses 0.0)
bias = False                # No bias in linear layers (LLaMA standard)

# === FFN Dimension (SwiGLU) ===
# For SwiGLU, d_ff is explicitly set (not auto-calculated)
# Widened to reach 1.36B params: ~2.89× d_model
d_ff = 6656                 # FFN dimension (26×256, kernel-friendly)
intermediate_size = 6656    # Alias for compatibility

# === Derived Parameters ===
# head_dim = 128                    # Per-head dimension (2048 / 16) - FlashAttention optimal
# vocab_size = 32000                # Will be set from tokenizer metadata

# === Alternative Designs (all ~1.36B params) ===
# C-2 (minimal change from old): n_layer=19, n_head=18, n_embd=2304, d_ff=6144
# C-3 (single param change): n_layer=18, n_head=18, n_embd=2304, d_ff=6656

# === Architecture Components (specified by arch_preset='llama') ===
# normalization = 'rmsnorm'         # RMSNorm (faster than LayerNorm)
# norm_eps = 1e-06                  # From JSON: rms_norm_eps
# position_encoding = 'rope'        # Rotary Position Embeddings
# rope_theta = 10000.0              # RoPE base frequency
# ffn_type = 'swiglu'               # SwiGLU activation
# norm_position = 'pre'             # Pre-norm architecture
# weight_tying = False              # No weight tying (better for large models)
# attention_backend = 'flash_attn_2' # FlashAttention-2 if available

# =============================================================================
# TRAINING - How to train it
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_llama'    # Dataset name (data/slimpajama_6b_llama/)
                                    # Uses LLaMA-2 tokenizer (32K vocab)
                                    # Change to 'slimpajama_627b_llama' for production
gradient_accumulation_steps = 16   # Accumulate gradients over N steps
batch_size = 8                     # Micro-batch size per GPU

# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# Example (4x A100): 8 × 16 × 4 = 512 samples/iter = 1,048,576 tokens/iter
# Example (8x B200): 16 × 8 × 8 = 1024 samples/iter = 2,097,152 tokens/iter

# === Optimizer (AdamW) ===
learning_rate = 3e-4               # Peak learning rate (scaled for 1.36B model)
max_iters = 25000                  # Total training iterations
                                    # Example: 25k iters × 262k tokens/iter = 6.5B tokens
                                    # Chinchilla optimal (D≈20N): 27B tokens = ~103k iters
weight_decay = 1e-1                # L2 regularization
beta1 = 0.9                        # Adam beta1
beta2 = 0.95                       # Adam beta2 (LLaMA uses 0.95)
grad_clip = 1.0                    # Gradient clipping threshold

# === Learning Rate Schedule ===
decay_lr = True                    # Enable cosine decay
warmup_iters = 2000                # Linear warmup iterations (~2% of training)
lr_decay_iters = 25000             # Should match max_iters
min_lr = 3e-5                      # Minimum LR (10% of peak)

# =============================================================================
# SYSTEM - Where/how to run
# =============================================================================

# === Hardware ===
device = 'cuda'                    # Device type
dtype = 'bfloat16'                 # Training precision (better than float16)
compile = True                     # Use torch.compile() for speedup

# === Parallelism ===
# For 4x A100 80GB (comfortable):
use_zero1 = True                   # ZeRO-1 optimizer sharding (enabled for testing)
use_fsdp = False                   # Fully Sharded Data Parallel

# For 3x RTX A4500 20GB (tight memory):
# use_zero1 = True                 # CRITICAL: saves ~10GB per GPU
# use_fsdp = False
# batch_size = 2                   # Reduce batch size
# gradient_accumulation_steps = 40 # Increase accumulation

# For 8x B200 128GB (optimal):
# use_zero1 = False
# use_fsdp = True                  # Better scaling for 8+ GPUs
# batch_size = 16
# gradient_accumulation_steps = 8

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-llama-1.36b'        # Output directory for checkpoints
eval_interval = 1000               # Evaluate every N iterations
log_interval = 10                  # Log every N iterations
eval_iters = 50                    # Number of iterations for evaluation (reduced for speed)
eval_only = False                  # If True, run evaluation and exit
eval_at_start = False              # If True, run evaluation before first training iteration
always_save_checkpoint = True      # Save checkpoint after each eval
init_from = 'scratch'              # 'scratch', 'resume', or 'gpt2*'

# === Logging ===
save_log_to_json = True            # Save training logs to JSON
log_save_interval = 10             # Save log every N iterations (increased frequency)
gradient_log_interval = 10         # Log gradient stats every N iterations

# === Weights & Biases (optional) ===
wandb_log = False                  # Enable W&B logging
wandb_project = 'llama-1.36b'      # W&B project name
wandb_run_name = 'run-1'           # W&B run name

# =============================================================================
# METADATA (Compute-optimal design)
# =============================================================================

# Model: 24L-16H-2048D-5632ff = 1.364B params
# Architecture: LLaMA-style (RoPE + RMSNorm + SwiGLU + Pre-norm)
# Head dimension: 128 (FlashAttention optimal)
#
# Chinchilla Scaling Law:
# - Optimal training: D ≈ 20N = 27B tokens
# - Per-token FLOPs (PaLM): 6N + 12LHQT = 8.61 GF/token
# - Target loss: ~2.4-2.5 (with optimal data)
#
# Training Recommendations:
# 1. For Chinchilla optimal (~27B tokens): ~103k iterations @ 262k tokens/iter
# 2. For quick testing (6B tokens): ~23k iterations (6-8 hours on 2× A6000)
# 3. For validation: slimpajama_6b (6B tokens) → expect loss ~4.0-4.5
#
# Expected Performance (2× A6000, ZeRO-1):
# - Tokens/sec: ~18,000
# - MFU: 45-52% (with FlashAttention-2)
# - Memory/GPU: ~28-30 GB
# - Time to 27B tokens: ~420 hours (~17.5 days)
#
# Expected Performance (4× A100, DDP):
# - Tokens/sec: ~50,000-60,000
# - MFU: 40-48%
# - Memory/GPU: ~25-30 GB
# - Time to 27B tokens: ~125-150 hours (~5-6 days)
#
# Expected Performance (8× B200, FSDP):
# - Tokens/sec: ~140,000-180,000
# - MFU: 45-55%
# - Memory/GPU: ~30-40 GB
# - Time to 27B tokens: ~40-55 hours (~2 days)

# =============================================================================
# NOTES & WARNINGS
# =============================================================================

# 1. Tokenizer: This config assumes LLaMA-2 tokenizer (32K vocab)
#    Download before SSH: transformers.LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#
# 2. Dataset: Prepare datasets before training
#    - cd data/slimpajama_6b && python prepare.py
#    - cd data/slimpajama_627b && python prepare.py
#
# 3. Batch size tuning:
#    - Adjust batch_size and gradient_accumulation_steps based on GPU memory
#    - Effective batch should be 512-1024 samples for stable training
#    - Monitor memory usage and reduce batch_size if OOM
#
# 4. Learning rate:
#    - 3e-4 is good starting point for 1.36B model
#    - Increase to 4e-4 if training is too slow to converge
#    - Decrease to 2e-4 if loss is unstable
#
# 5. Early stopping:
#    - Monitor validation loss every eval_interval
#    - Stop when validation loss plateaus (typically after 20-30k iters)
#    - Don't overtrain beyond optimal token count (~85B tokens)

