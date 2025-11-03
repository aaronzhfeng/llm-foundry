# GPT-2 124M Standard Architecture
# ==================================
# Features:
# - Learned absolute position embeddings
# - LayerNorm (no bias)
# - GELU activation
# - Standard FFN (4x expansion)
# - Post-norm
# - Weight tying

# === ARCHITECTURE ===
arch_preset = 'gpt2'  # Use GPT-2 preset

# === MODEL SIZE ===
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.0
bias = False

# === I/O ===
out_dir = 'out-gpt2'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# === LOGGING ===
save_log_to_json = True
log_save_interval = 100
gradient_log_interval = 50

# === DATA ===
dataset = 'shakespeare'  # Change to 'openwebtext' for production
gradient_accumulation_steps = 40
batch_size = 12

# === OPTIMIZER ===
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# === LR SCHEDULE ===
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# === SYSTEM ===
device = 'cuda'
dtype = 'bfloat16'
compile = True

# === PARALLELISM ===
use_zero1 = False
use_fsdp = False

