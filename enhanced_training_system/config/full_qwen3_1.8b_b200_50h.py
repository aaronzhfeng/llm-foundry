import os

# Qwen3-1.8B B200 Production Configuration (50-Hour Run)
# This configuration targets ~117B tokens over ~50 hours on 8× B200 GPUs.

# Mitigate torch.compile allocator fragmentation when training long runs.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =============================================================================
# ARCHITECTURE
# =============================================================================

arch_preset = 'custom'

normalization = 'rmsnorm'
activation = 'silu'
attention_backend = 'flash_attn_2'
position_encoding = 'rope'
norm_position = 'pre'
ffn_type = 'swiglu'
bias = False
weight_tying = False

n_layer = 24
n_head = 16
n_embd = 2048
num_key_value_heads = 8
block_size = 2048
vocab_size = 151669
dropout = 0.0
d_ff = 6144
intermediate_size = 6144
rope_theta = 1_000_000
norm_eps = 1e-6

# =============================================================================
# TRAINING - 50 Hour Production Target
# =============================================================================

dataset = 'slimpajama_627b_qwen3'

# === Batch schedule ===
# 22 (micro-batch) × 8 (GPUs) × 2 (grad accum) × 2048 (seq len)
# → 720,896 tokens / iteration → ~162k iterations ≈ 117B tokens in 50 hours.
batch_size = 22
gradient_accumulation_steps = 2

# === Optimizer ===
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# === Learning rate schedule ===
max_iters = 162000
lr_decay_iters = 162000
warmup_iters = 2000
min_lr = 3e-5
decay_lr = True

# =============================================================================
# SYSTEM - B200 Specific Optimizations
# =============================================================================

device = 'cuda'
dtype = 'bfloat16'
compile = True
use_cuda_graphs = False

use_dataloader = True
dataloader_num_workers = 16
dataloader_prefetch_factor = 2

# ZeRO-1 is required to stay within memory when running torch.compile + large batches.
use_fsdp = False
use_zero1 = True

# Optional toggle consumed by train.py for MFU denominator selection.
use_sparse_specs = False

# =============================================================================
# I/O & LOGGING
# =============================================================================

out_dir = 'out-qwen3-1.8b-b200-50h'

# Evaluate and checkpoint roughly every 30 minutes to 1 hour.
eval_interval = 20000
eval_iters = 50
always_save_checkpoint = True
keep_all_checkpoints = True

eval_only = False
eval_at_start = False
init_from = 'scratch'

log_interval = 10
save_log_to_json = True
log_save_interval = 10
gradient_log_interval = 10

# =============================================================================
# Instrumentation
# =============================================================================

wandb_log = True
wandb_project = 'qwen3-1.8b-b200'
wandb_run_name = '50h-production-117B'


