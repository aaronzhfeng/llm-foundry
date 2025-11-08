# Custom Architecture - Mix and Match!
# =====================================
# Experiment with different component combinations.
# Change any architectural choice without touching code!

# === ARCHITECTURE ===
arch_preset = 'custom'  # Build from individual components

# Architecture components - mix and match any combination:
normalization = 'rmsnorm'           # Options: 'layernorm', 'layernorm_nobias', 'rmsnorm'
activation = 'gelu'                 # Options: 'gelu', 'silu', 'relu', 'leaky_relu'
attention_backend = 'flash_attn_2'  # Options: 'flash_attn_2' (fastest), 'sdpa' (FA-1), 'manual' (slow)
position_encoding = 'rope'          # Options: 'learned_absolute', 'rope', 'none'
norm_position = 'pre'               # Options: 'pre', 'post'
ffn_type = 'standard'               # Options: 'standard', 'swiglu'
bias = False                        # Options: True, False
weight_tying = True                 # Options: True, False
dropout = 0.0                       # Range: 0.0-1.0
rope_theta = 10000.0                # RoPE base frequency (if using RoPE)

# === MODEL SIZE ===
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024

# === I/O ===
out_dir = 'out-custom'
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

# =====================================
# EXPERIMENT IDEAS:
# =====================================
# 
# 1. GPT-2 with RoPE:
#    position_encoding = 'rope', normalization = 'layernorm_nobias', ffn_type = 'standard'
#
# 2. LLaMA with learned positions:
#    position_encoding = 'learned_absolute', normalization = 'rmsnorm', ffn_type = 'swiglu'
#
# 3. RMSNorm + GELU + RoPE:
#    normalization = 'rmsnorm', activation = 'gelu', position_encoding = 'rope', ffn_type = 'standard'
#
# 4. Best of both:
#    normalization = 'rmsnorm', activation = 'gelu', position_encoding = 'rope', 
#    norm_position = 'pre', ffn_type = 'standard', weight_tying = True
#

