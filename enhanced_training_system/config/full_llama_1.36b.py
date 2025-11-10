"""
LLaMA 1.36B Architecture - Production Configuration
===================================================

Based on scaling law optimization from: info/llama_1.36e21_32kV.json

Model Design:
- Parameters: 1.36B (1,294,159,104 exactly)
- Optimal training: 84.72B tokens
- Target loss: 2.37 (theoretical minimum)

Architecture (LLaMA-style):
- RoPE (Rotary Position Embeddings)
- RMSNorm (faster than LayerNorm)
- SwiGLU activation (8/3 expansion)
- Pre-norm (better training stability)
- No weight tying
- No bias

Scaled from LLaMA 7B:
- Maintains same architectural proportions
- Same head dimension (128)
- Same FFN expansion ratio (~2.67)
- Same context length (2048)

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

# === Model Dimensions (from scaling law optimization) ===
n_layer = 18                # Number of transformer layers
n_head = 18                 # Number of attention heads
n_embd = 2304               # Hidden dimension / embedding dimension
block_size = 2048           # Maximum sequence length / context window
dropout = 0.0               # Dropout rate (LLaMA uses 0.0)
bias = False                # No bias in linear layers (LLaMA standard)

# === Derived Parameters (auto-calculated by model) ===
# intermediate_size = 6144          # FFN dimension (~8/3 × 2304 for SwiGLU)
# head_dim = 128                    # Per-head dimension (2304 / 18)
# vocab_size = 32000                # Will be set from tokenizer metadata

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
                                    # ~25k iters × 1M tokens/iter = 25B tokens
                                    # For 85B tokens (optimal): ~85k iters
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
use_zero1 = False                  # ZeRO-1 optimizer sharding (not needed on A100)
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
eval_iters = 200                   # Number of iterations for evaluation
eval_only = False                  # If True, run evaluation and exit
always_save_checkpoint = True      # Save checkpoint after each eval
init_from = 'scratch'              # 'scratch', 'resume', or 'gpt2*'

# === Logging ===
save_log_to_json = True            # Save training logs to JSON
log_save_interval = 100            # Save log every N iterations
gradient_log_interval = 50         # Log gradient stats every N iterations

# === Weights & Biases (optional) ===
wandb_log = False                  # Enable W&B logging
wandb_project = 'llama-1.36b'      # W&B project name
wandb_run_name = 'run-1'           # W&B run name

# =============================================================================
# METADATA (from scaling law analysis)
# =============================================================================

# Source: info/llama_1.36e21_32kV.json
#
# Scaling Law Results:
# - Theoretical loss: 2.372087
# - Optimal configuration: 1.294e9 params × 84.72B tokens
# - Validation (62M tokens): loss = 4.712
#
# Training Recommendations:
# 1. For loss ~2.4-2.5: Train on 80-100B tokens (~85k iterations)
# 2. For loss ~4.0-4.5: Use slimpajama_6b (quick testing)
# 3. For production: Use slimpajama_627b with early stopping
#
# Expected Performance (4x A100, DDP):
# - Tokens/sec: ~50,000-60,000
# - MFU: 35-45%
# - Memory/GPU: ~25-30 GB
# - Time to 85B tokens: ~30-50 hours
#
# Expected Performance (8x B200, FSDP):
# - Tokens/sec: ~140,000-180,000
# - MFU: 40-50%
# - Memory/GPU: ~30-40 GB
# - Time to 85B tokens: ~10-15 hours

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

