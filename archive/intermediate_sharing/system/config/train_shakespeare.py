# Train on Shakespeare dataset (for quick testing)

# I/O
out_dir = 'out-shakespeare'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# logging
save_log_to_json = True
log_save_interval = 50
gradient_log_interval = 25

# data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# model - smaller for quick testing
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# adamw optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False  # Faster for small models/runs

# parallelism
use_zero1 = False
use_fsdp = False

