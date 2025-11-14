"""
LLaMA 3.1 8B - Official Meta Architecture
==========================================

Based on official Meta release: meta-llama/Meta-Llama-3.1-8B

Model Design:
- Parameters: 8.03B (8,030,261,248 exactly)
- Config: 32L-32H-4096D-14336ff-GQA8
- FFN ratio: 3.5× (14336 / 4096) - improved from LLaMA 2's 8/3
- Grouped Query Attention: 8 KV heads (4:1 Q:KV ratio)
- Extended context: 128K tokens (with RoPE scaling)
- Tokenizer: 128K vocabulary (tiktoken-based BPE)

Architecture (LLaMA 3 style):
- RoPE (Rotary Position Embeddings) with extended theta=500000
- RMSNorm (faster than LayerNorm)
- SwiGLU activation (d_ff = 3.5 × d_model = 14336)
- Pre-norm (better training stability)
- No weight tying
- No bias
- Head dimension: 128 (FlashAttention optimal)
- GQA: 8 KV heads across all model sizes

Key differences from LLaMA 2:
- GQA (8 KV heads) instead of MHA → 4× smaller KV cache
- 3.5× FFN expansion instead of 8/3 (~2.67×) → better compute/param ratio
- Extended RoPE (theta=500000) instead of 10000 → supports 128K context
- 128K vocab instead of 32K → better multilingual + efficiency

Usage:
    # Quick test on 6B tokens (for validation)
    python train.py config/full_llama3_8b.py
    
    # Multi-GPU training (4x A100)
    torchrun --standalone --nproc_per_node=4 train.py config/full_llama3_8b.py
    
    # Production training (8x B200) on full dataset
    torchrun --standalone --nproc_per_node=8 train.py \\
        config/full_llama3_8b.py \\
        --dataset=slimpajama_627b_llama3 \\
        --use_fsdp=True

References:
- HuggingFace: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
- Meta Blog: https://ai.meta.com/blog/meta-llama-3-1/
"""

# =============================================================================
# ARCHITECTURE - What the model IS
# =============================================================================

# === Architecture Preset ===
arch_preset = 'llama3'  # Use LLaMA 3 components: RoPE + RMSNorm + SwiGLU + Pre-norm + GQA

# === Attention Backend Override ===
attention_backend = 'flash_attn_2'  # Use FlashAttention-2 (fastest, ~2x speedup)

# === Model Dimensions (Official LLaMA 3.1 8B) ===
n_layer = 32                # Number of transformer layers
n_head = 32                 # Number of attention heads (Q heads)
n_embd = 4096               # Hidden dimension / embedding dimension
block_size = 2048           # Training sequence length (supports up to 128K with RoPE scaling)
dropout = 0.0               # Dropout rate (LLaMA uses 0.0)
bias = False                # No bias in linear layers (LLaMA standard)

# === Grouped Query Attention (GQA) - LLaMA 3 Specific ===
num_key_value_heads = 8     # Number of KV heads (8 for all LLaMA 3 models)
                            # 32 Q heads / 8 KV heads = 4:1 ratio
                            # Benefits: 4× smaller KV cache, similar quality to MHA

# === FFN Dimension (SwiGLU) ===
# LLaMA 3: 3.5× expansion (improved from LLaMA 2's 8/3 ≈ 2.67×)
d_ff = 14336                # FFN dimension (3.5 × 4096 = 14336)
intermediate_size = 14336   # Alias for compatibility

# === Extended RoPE for Long Context ===
rope_theta = 500000.0       # Extended from 10000 (LLaMA 2) for 128K context support
# Note: For sequences > 2048, you may need additional RoPE scaling
# See: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json

# === Derived Parameters ===
# head_dim = 128                    # Per-head dimension (4096 / 32) - FlashAttention optimal
# vocab_size = 128256               # Will be set from tokenizer metadata (128K vocab, rounded up)

# === Architecture Components (specified by arch_preset='llama3') ===
# normalization = 'rmsnorm'         # RMSNorm (faster than LayerNorm)
# norm_eps = 1e-05                  # From Meta's config (slightly larger than LLaMA 2's 1e-06)
# position_encoding = 'rope'        # Rotary Position Embeddings
# ffn_type = 'swiglu'               # SwiGLU activation
# norm_position = 'pre'             # Pre-norm architecture
# weight_tying = False              # No weight tying (better for large models)
# attention_backend = 'flash_attn_2' # FlashAttention-2 if available

# =============================================================================
# TRAINING - How to train it
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_llama'    # Dataset name (data/slimpajama_6b_llama/)
                                    # Uses LLaMA-2 tokenizer (32K vocab) for testing
                                    # For production: use 'slimpajama_627b_llama3' with 128K tokenizer
gradient_accumulation_steps = 8    # Accumulate gradients over N steps
batch_size = 4                     # Micro-batch size per GPU
                                    # Note: LLaMA 3.1 8B fits on 2× A6000 (48GB total) with batch_size=4

# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# Example (2x A6000): 4 × 8 × 2 = 64 samples/iter = 131,072 tokens/iter
# Example (4x A100): 8 × 8 × 4 = 256 samples/iter = 524,288 tokens/iter
# Example (8x B200): 16 × 4 × 8 = 512 samples/iter = 1,048,576 tokens/iter

# === Optimizer (AdamW) ===
learning_rate = 3e-4               # Peak learning rate (Meta uses 3e-4 for 8B)
max_iters = 25000                  # Total training iterations
                                    # Example: 25k iters × 131k tokens/iter = 3.3B tokens (2× A6000)
                                    # Chinchilla optimal (D≈20N): ~161B tokens = ~1.2M iters
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
# For 2x A6000 48GB (tight but works):
use_zero1 = True                   # ZeRO-1 optimizer sharding (saves ~12GB per GPU)
use_fsdp = False                   # Not needed for 8B model on 2 GPUs

# For 4x A100 80GB (comfortable):
# use_zero1 = False                # Optional: can disable if memory is sufficient
# use_fsdp = False
# batch_size = 8
# gradient_accumulation_steps = 8

# For 8x B200 128GB (optimal):
# use_zero1 = False
# use_fsdp = True                  # Better scaling for 8+ GPUs
# batch_size = 16
# gradient_accumulation_steps = 4

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-llama3-8b'          # Output directory for checkpoints
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
wandb_project = 'llama3-8b'        # W&B project name
wandb_run_name = 'run-1'           # W&B run name

# =============================================================================
# METADATA & EXPECTED PERFORMANCE
# =============================================================================

# Model: 32L-32H-4096D-14336ff-GQA8 = 8.03B params
# Architecture: LLaMA 3 style (RoPE + RMSNorm + SwiGLU + Pre-norm + GQA)
# Head dimension: 128 (FlashAttention optimal)
# GQA: 8 KV heads (4:1 Q:KV ratio) → 4× smaller KV cache than MHA
#
# Per-token FLOPs (PaLM formula with GQA):
# - Forward: ~30.6 GF/token (less than MHA due to GQA)
# - Training: ~91.8 GF/token (3× forward)
#
# Chinchilla Optimal:
# - Parameters: 8.03B
# - Optimal tokens: ~161B (20 × 8.03B)
# - Target loss: ~2.2 (estimated from scaling law)
#
# Expected Performance (2× A6000, ZeRO-1, batch_size=4):
# - Tokens/sec: ~8,000-10,000
# - MFU: 25-35% (with FlashAttention-2 and GQA)
# - Memory/GPU: ~38-42 GB (tight but fits)
# - Time to 161B tokens: ~4,500-5,600 hours (~188-233 days)
#
# Expected Performance (4× A100, DDP, batch_size=8):
# - Tokens/sec: ~30,000-35,000
# - MFU: 35-45%
# - Memory/GPU: ~30-35 GB
# - Time to 161B tokens: ~1,300-1,500 hours (~54-63 days)
#
# Expected Performance (8× B200, FSDP, batch_size=16):
# - Tokens/sec: ~90,000-110,000
# - MFU: 42-52%
# - Memory/GPU: ~35-45 GB
# - Time to 161B tokens: ~400-500 hours (~17-21 days)

# =============================================================================
# NOTES & WARNINGS
# =============================================================================

# 1. Tokenizer: This config requires LLaMA 3 tokenizer (128K vocab)
#    The current training system uses LLaMA 2 tokenizer (32K) by default
#    TODO: Add LLaMA 3 tokenizer support to data preparation scripts
#
# 2. Dataset: Prepare datasets before training
#    - cd data/slimpajama_6b && python prepare.py
#    - cd data/slimpajama_627b && python prepare.py
#
# 3. Memory considerations:
#    - 8B model is LARGER than 1.36B (6.2× more parameters)
#    - 2× A6000: batch_size=4 works with ZeRO-1, batch_size=6-8 OOMs
#    - 4× A100: batch_size=8 comfortable, can try 12-16
#    - 8× B200: batch_size=16 comfortable, can try 24-32
#
# 4. GQA benefits:
#    - 4× smaller KV cache → better memory efficiency
#    - Similar quality to MHA (based on Meta's evaluations)
#    - Slightly lower FLOPs than MHA due to smaller K/V projections
#
# 5. Learning rate:
#    - Meta uses 3e-4 for 8B model (same as smaller models)
#    - No need to scale LR with model size for AdamW
#    - Warmup is critical for stability (2000 iters recommended)
#
# 6. Extended RoPE:
#    - rope_theta=500000 supports sequences up to 128K
#    - Training uses 2048 seq_len for efficiency
#    - For longer sequences, may need RoPE scaling (see Meta's config)
#
# 7. Early stopping:
#    - Monitor validation loss every eval_interval
#    - Stop when validation loss plateaus
#    - For 8B model, expect ~500k-1M iters for reasonable convergence


