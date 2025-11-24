import os

# Qwen3-1.8B B200 Smoke-Test Configuration
# ----------------------------------------
# Exercises the full production pipeline (evaluation, checkpoints, logging)
# while keeping all hyperparameters identical, but only runs a handful of
# iterations so we can validate end-to-end behavior quickly.

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =============================================================================
# ARCHITECTURE  (identical to production)
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
# TRAINING - Smoke Test (same hyperparameters, fewer steps)
# =============================================================================

dataset = 'slimpajama_627b_qwen3'

batch_size = 22
gradient_accumulation_steps = 2

learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

max_iters = 100             # Run only 20 iterations (vs. 162k in prod)
lr_decay_iters = 100
warmup_iters = 20
min_lr = 3e-5
decay_lr = True

# =============================================================================
# SYSTEM (identical to production)
# =============================================================================

device = 'cuda'
dtype = 'bfloat16'
compile = True
use_cuda_graphs = False

use_dataloader = True
dataloader_num_workers = 16
dataloader_prefetch_factor = 2

use_fsdp = False
use_zero1 = True
use_sparse_specs = False

# =============================================================================
# I/O & LOGGING
# =============================================================================

out_dir = 'out-qwen3-1.8b-b200-50h-smoke'

eval_interval = 100          # Eval every 5 iters to hit the eval path quickly
eval_iters = 1
always_save_checkpoint = True
keep_all_checkpoints = True
eval_only = False
eval_at_start = True       # run evaluation before training to test this path
init_from = 'scratch'

log_interval = 2
save_log_to_json = True
log_save_interval = 2
gradient_log_interval = 2

# =============================================================================
# Instrumentation
# =============================================================================

wandb_log = True
wandb_project = 'qwen3-1.8b-b200'
wandb_run_name = '50h-smoke-test'


