# Train a GPT-2 (124M) on OpenWebText
# Good default settings for 8xA100 40GB

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# logging
save_log_to_json = True
log_save_interval = 100
gradient_log_interval = 50

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 40  # Adjust based on your GPUs
batch_size = 12
block_size = 1024

# model - GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# system
device = 'cuda'
dtype = 'bfloat16' # Use bfloat16 on A100
compile = True

# parallelism - choose one
use_zero1 = False  # Set to True for ZeRO-1
use_fsdp = False   # Set to True for FSDP

