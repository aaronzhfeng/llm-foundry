"""
GPT-2 1.36B Architecture - For Direct Comparison with LLaMA 1.36B
=================================================================

Architecture (GPT-2-style):
- Learned absolute position embeddings
- LayerNorm (no bias)
- GELU activation
- Standard FFN (4× expansion)
- Post-norm
- Weight tying

Model Dimensions:
- 18 layers (SAME as LLaMA 1.36B)
- 2304 hidden dim (SAME as LLaMA's 2304)
- 18 heads → 128 head_dim
- 2048 context (SAME as LLaMA)
- 50304 vocab (GPT-2 standard)

Parameters: ~1.29B (slightly less than LLaMA 1.36B due to weight tying)

Design Rationale:
- Matching layer count (18) and hidden dim (2304) for fair comparison
- Same head count (18) and head dim (128) for fair attention comparison
- Weight tying reduces params slightly (vs LLaMA's no weight tying)
- Standard FFN (4×) vs LLaMA's SwiGLU (2.67×) is the key difference

Key Differences from LLaMA 1.36B:
┌─────────────────────┬───────────────┬───────────────┬────────────┐
│ Feature             │ GPT-2 1.29B   │ LLaMA 1.36B   │ Impact     │
├─────────────────────┼───────────────┼───────────────┼────────────┤
│ Layers              │ 18            │ 18            │ ✓ Same     │
│ Hidden Size         │ 2304          │ 2304          │ ✓ Same     │
│ Heads               │ 18            │ 18            │ ✓ Same     │
│ Head Dim            │ 128           │ 128           │ ✓ Same     │
│ Context             │ 2048          │ 2048          │ ✓ Same     │
│ FFN Type            │ Standard 4×   │ SwiGLU 2.67×  │ Different  │
│ FFN Size            │ 9216          │ 6144          │ +50%       │
│ Position            │ Learned       │ RoPE          │ Different  │
│ Norm                │ LayerNorm     │ RMSNorm       │ Different  │
│ Norm Position       │ Post          │ Pre           │ Different  │
│ Weight Tying        │ Yes           │ No            │ Different  │
│ Vocab Size          │ 50304         │ 32000         │ +57%       │
│ Total Params        │ ~1.29B        │ ~1.36B        │ Similar    │
└─────────────────────┴───────────────┴───────────────┴────────────┘

Usage:
    # Train GPT-2 1.36B
    python train.py config/full_gpt2_1.36b.py
    
    # Compare with LLaMA 1.36B
    python train.py config/full_gpt2_1.36b.py --max_iters=5000
    python train.py config/full_llama_1.36b.py --max_iters=5000
    
    # Multi-GPU (4x A100)
    torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_1.36b.py
"""

# =============================================================================
# ARCHITECTURE - GPT-2 Style
# =============================================================================

# === Architecture Preset ===
arch_preset = 'gpt2'  # Use GPT-2 components: LayerNorm + Learned Pos + GELU + Standard FFN

# === Model Dimensions ===
n_layer = 18                # Number of transformer layers (SAME as LLaMA)
n_head = 18                 # Number of attention heads (SAME as LLaMA)
n_embd = 2304               # Hidden dimension (SAME as LLaMA's 2304, head_dim=128)
block_size = 2048           # Maximum sequence length (SAME as LLaMA)
dropout = 0.0               # Dropout rate (no dropout for fair comparison)
bias = False                # No bias in linear layers

# === Derived Parameters (must be set to None or correct values when overriding preset) ===
num_key_value_heads = None  # Will auto-set to n_head (18) for MHA
d_ff = None                 # Will auto-calculate as 4×n_embd = 9216

# === Derived Parameters (auto-calculated by model) ===
# intermediate_size = 9216          # FFN dimension (4× 2304 for GPT-2)
# head_dim = 128                    # Per-head dimension (2304 / 18 = 128)
# vocab_size = 50304                # Will be set from tokenizer (GPT-2 standard)

# === Architecture Components (specified by arch_preset='gpt2') ===
# normalization = 'layernorm_nobias'    # LayerNorm without bias
# position_encoding = 'learned_absolute' # Learned position embeddings
# activation = 'gelu'                    # GELU activation
# ffn_type = 'standard'                  # Standard FFN (4× expansion)
# norm_position = 'post'                 # Post-norm architecture
# weight_tying = True                    # Weight tying (saves params)
# attention_backend = 'sdpa'             # PyTorch SDPA (FlashAttention-1)

# =============================================================================
# TRAINING - Same as LLaMA 1.36B for fair comparison
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_gpt2'     # Dataset name (data/slimpajama_6b_gpt2/)
                                    # Uses GPT-2 BPE tokenizer (50K vocab)
                                    # Change to 'slimpajama_627b_gpt2' for production
gradient_accumulation_steps = 16   # Accumulate gradients over N steps
batch_size = 8                     # Micro-batch size per GPU

# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# Example (4x A100): 8 × 16 × 4 = 512 samples/iter = 1,048,576 tokens/iter

# === Optimizer (AdamW) ===
learning_rate = 3e-4               # Peak learning rate (same as LLaMA 1.36B)
max_iters = 25000                  # Total training iterations
                                    # ~25k iters × 1M tokens/iter = 25B tokens
                                    # For 85B tokens (optimal): ~85k iters
weight_decay = 1e-1                # L2 regularization
beta1 = 0.9                        # Adam beta1
beta2 = 0.95                       # Adam beta2
grad_clip = 1.0                    # Gradient clipping threshold

# === Learning Rate Schedule ===
decay_lr = True                    # Enable cosine decay
warmup_iters = 2000                # Linear warmup iterations (~2% of training)
lr_decay_iters = 25000             # Should match max_iters
min_lr = 3e-5                      # Minimum LR (10% of peak)

# =============================================================================
# SYSTEM - Same as LLaMA 1.36B
# =============================================================================

# === Hardware ===
device = 'cuda'                    # Device type
dtype = 'bfloat16'                 # Training precision
compile = True                     # Use torch.compile() for speedup

# === Parallelism ===
# For 4x A100 80GB (comfortable):
use_zero1 = False                  # ZeRO-1 optimizer sharding
use_fsdp = False                   # Fully Sharded Data Parallel

# For 3x RTX A4500 20GB (tight memory):
# use_zero1 = True                 # CRITICAL: saves ~10GB per GPU
# batch_size = 2
# gradient_accumulation_steps = 40

# For 8x B200 128GB (optimal):
# use_fsdp = True                  # Better scaling for 8+ GPUs
# batch_size = 16
# gradient_accumulation_steps = 8

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-gpt2-1.36b'         # Output directory for checkpoints
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
wandb_project = 'gpt2-1.36b'       # W&B project name
wandb_run_name = 'run-1'           # W&B run name

# =============================================================================
# PARAMETER COUNT VERIFICATION
# =============================================================================

# Formula: N = V×H + S×H + L×(4H² + 2H×(4H) + 2H) + H
#
# Where:
# - V = vocab_size = 50304
# - H = hidden_size = 2304
# - S = max_position_embeddings = 2048
# - L = num_layers = 18
# - Weight tying = True (input & output embeddings share parameters)
#
# Breakdown:
# 1. Token embeddings: V×H = 50304×2304 = 115,900,416 (shared with output)
# 2. Position embeddings: S×H = 2048×2304 = 4,718,592
# 3. Per layer (18 layers):
#    - Attention (Q,K,V,O): 4×H² = 4×2304² = 21,233,664
#    - FFN (up, down): 2×H×(4H) = 2×2304×9216 = 42,467,328
#    - LayerNorm: 2×H = 2×2304 = 4,608 (negligible)
#    - Layer total: 63,705,600
#    - All layers: 18×63,705,600 = 1,146,700,800
# 4. Final LayerNorm: H = 2,304
#
# Total: 115,900,416 + 4,718,592 + 1,146,700,800 + 2,304
#      = 1,267,322,112 ≈ 1.27B parameters
#
# Note: Slightly less than LLaMA 1.36B due to:
# - Weight tying (saves ~116M params vs LLaMA's no tying)
# - Position embeddings add ~4.7M (LLaMA uses RoPE with no params)
# - Larger vocab (50304 vs 32000) adds ~50M params
# 
# Net effect: Weight tying savings exceed vocab increase, resulting in ~1.29B total

# =============================================================================
# EXPECTED PERFORMANCE
# =============================================================================

# Computational Cost:
# - FLOPs per token: ~32-35 GFLOPs (forward pass)
#   - Lower than LLaMA (40-45 GF) due to simpler FFN
# - Training speed: ~15-20% faster per token than LLaMA
# - Memory: ~23-25 GB per GPU (position embeddings add memory)
#
# Training Time (4x A100):
# - To 25B tokens: ~15-20 hours
# - To 85B tokens (optimal): ~50-65 hours
# - Slightly faster than LLaMA due to simpler architecture
#
# Expected Loss (85B tokens):
# - Theoretical: ~2.5-2.6 (slightly higher than LLaMA's 2.37)
# - GPT-2 architecture typically 5-10% worse than LLaMA at same param count
# - Still very competitive performance

# =============================================================================
# COMPARISON EXPERIMENT PROTOCOL
# =============================================================================

# To fairly compare GPT-2 vs LLaMA architectures:
#
# 1. Train both to same token count:
#    python train.py config/full_gpt2_1.36b.py --max_iters=25000
#    python train.py config/full_llama_1.36b.py --max_iters=25000
#
# 2. Use identical training settings:
#    - Same dataset (slimpajama_6b or slimpajama_627b)
#    - Same batch size and gradient accumulation
#    - Same learning rate schedule
#    - Same number of training tokens
#
# 3. Compare metrics:
#    - Validation loss (primary metric)
#    - Training speed (tokens/sec)
#    - MFU (model FLOPs utilization)
#    - Memory usage per GPU
#    - Gradient health (norms, stability)
#
# 4. Expected results:
#    - LLaMA should achieve ~5-10% lower loss
#    - GPT-2 should be ~15-20% faster per token
#    - Both should achieve similar MFU (~35-45%)
#    - Memory usage should be similar (~25GB per GPU)

# =============================================================================
# NOTES & WARNINGS
# =============================================================================

# 1. Tokenizer: This config assumes GPT-2 BPE tokenizer (50304 vocab)
#    Different from LLaMA which uses 32K vocab
#    For truly fair comparison, consider using same tokenizer
#
# 2. Hidden dimension: 2432 gives head_dim=135 (not power of 2)
#    This is fine - many successful models use non-power-2 dimensions
#    (e.g., LLaMA 13B uses 5120 = 40×128)
#
# 3. Position embeddings: Add 4.98M parameters vs RoPE (0 params)
#    But RoPE has small computational overhead
#    Trade-off: memory vs compute
#
# 4. Post-norm: May be less stable than pre-norm for 18 layers
#    Monitor gradient norms carefully
#    If unstable, consider reducing learning rate or adding more warmup

# For detailed parameter calculations and formulas, see:
# config/PARAMETER_FORMULAS.md

