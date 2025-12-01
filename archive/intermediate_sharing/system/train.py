"""
Enhanced GPT Training Script with Comprehensive Monitoring
===========================================================

This training script supports:
- Single GPU and multi-GPU training (DDP/FSDP)
- ZeRO-1 optimizer state sharding
- FSDP (Fully Sharded Data Parallel)
- Detailed MFU calculation with academic formulas
- Comprehensive memory and gradient tracking
- Enhanced startup and per-iteration reporting

To run on a single GPU:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with FSDP on 4 GPUs:
$ torchrun --standalone --nproc_per_node=4 train.py --use_fsdp=True

References:
- nanoGPT: https://github.com/karpathy/nanoGPT
- Insu Jang (2022): https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
- Epoch AI backward/forward ratio: https://epoch.ai/blog/backward-forward-FLOP-ratio
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools

from model import GPTConfig, GPT

# Import modular architecture system
try:
    from model_config import ModelArchitectureConfig, get_preset_config
    from model_builder import ConfigurableGPT
    MODULAR_ARCH_AVAILABLE = True
except ImportError:
    MODULAR_ARCH_AVAILABLE = False
    print("WARNING: Modular architecture system not available, using legacy GPT model")

# Try to import training logger (optional)
try:
    from training_logger import TrainingLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# logging
save_log_to_json = True # save training logs to JSON file
log_save_interval = 100 # save log every N iterations
gradient_log_interval = 10 # log gradients every N iterations (more expensive)
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# === ARCHITECTURE CONFIGURATION (modular) ===
arch_preset = 'gpt2'                # Preset: 'gpt2', 'llama', 'hybrid', 'team', or 'custom'
normalization = 'layernorm_nobias'  # 'layernorm', 'layernorm_nobias', 'rmsnorm'
activation = 'gelu'                 # 'gelu', 'silu', 'relu', 'leaky_relu' (for standard FFN)
attention_backend = 'sdpa'          # 'sdpa', 'manual'
position_encoding = 'learned_absolute'  # 'learned_absolute', 'rope', 'none'
norm_position = 'post'              # 'pre', 'post'
ffn_type = 'standard'               # 'standard', 'swiglu'
weight_tying = True                 # True/False - tie token embeddings with lm_head
rope_theta = 10000.0                # RoPE theta parameter (if using RoPE)
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# ZeRO-1 settings
use_zero1 = False # use ZeRO-1 optimizer state sharding
# FSDP settings
use_fsdp = False # use FSDP instead of DDP
fsdp_min_num_params = 1e6 # minimum number of parameters for auto-wrapping (1M default)
fsdp_activation_checkpointing = False # enable activation checkpointing with FSDP
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Build architecture configuration
if MODULAR_ARCH_AVAILABLE and arch_preset != 'legacy':
    # Use modular architecture system
    if arch_preset != 'custom':
        print(f"Loading preset architecture: '{arch_preset}'")
        arch_config = get_preset_config(arch_preset)
        # Override size parameters from CLI
        arch_config.n_layer = n_layer
        arch_config.n_head = n_head
        arch_config.n_embd = n_embd
        arch_config.block_size = block_size
        arch_config.dropout = dropout
        arch_config.bias = bias
        arch_config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    else:
        print("Building custom architecture from config")
        arch_config = ModelArchitectureConfig(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
            normalization=normalization,
            activation=activation,
            attention_backend=attention_backend,
            position_encoding=position_encoding,
            norm_position=norm_position,
            ffn_type=ffn_type,
            bias=bias,
            weight_tying=weight_tying,
            dropout=dropout,
            rope_theta=rope_theta,
        )
    
    # Save architecture config for reproducibility
    model_args = arch_config.to_dict()
else:
    # Legacy: use original GPT model
    print("Using legacy GPT-2 model")
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout)

# model init
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    
    if MODULAR_ARCH_AVAILABLE and arch_preset != 'legacy':
        # Use modular architecture
        model = ConfigurableGPT(arch_config)
    else:
        # Legacy GPT
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    if MODULAR_ARCH_AVAILABLE and arch_preset != 'legacy':
        # Modular architecture: load from checkpoint config
        arch_config = ModelArchitectureConfig.from_dict(checkpoint_model_args)
        model = ConfigurableGPT(arch_config)
        model_args = arch_config.to_dict()
    else:
        # Legacy GPT
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights (legacy only)
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# Convert model to target dtype (important for accurate MFU calculation)
if dtype != 'float32':
    model = model.to(dtype=ptdtype)
    print(f"Model converted to {dtype}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer (will be created after FSDP wrapping if using FSDP)
# For now, store the checkpoint for later optimizer loading
checkpoint_for_resume = checkpoint if init_from == 'resume' else None

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into FSDP or DDP container
if ddp:
    if use_fsdp:
        print("Wrapping model with FSDP...")
        
        # Configure FSDP mixed precision policy
        if dtype == 'bfloat16':
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype == 'float16':
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            mp_policy = None
        
        # Auto-wrap policy: wrap layers with > fsdp_min_num_params parameters
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(fsdp_min_num_params),
        )
        
        # Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,  # Required for torch.compile compatibility
        )
        
        print(f"FSDP enabled with min_params={fsdp_min_num_params}, mixed_precision={dtype}")
        
    else:
        # Standard DDP or ZeRO-1
        model = DDP(model, device_ids=[ddp_local_rank])

# Create optimizer after FSDP/DDP wrapping
if ddp and use_zero1 and not use_fsdp:
    # ZeRO-1: Shard optimizer states across GPUs
    print("Using ZeRO-1 optimizer state sharding")
    # Get parameter groups from model's configure method
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    # Create ZeRO-1 optimizer (note: fused=True may not be compatible with ZeroRedundancyOptimizer)
    optimizer = ZeroRedundancyOptimizer(
        optim_groups,
        optimizer_class=torch.optim.AdamW,
        lr=learning_rate,
        betas=(beta1, beta2),
    )
elif ddp and use_fsdp:
    # FSDP with use_orig_params=True allows direct access to parameters
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    import inspect
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")
else:
    # Standard optimizer (single GPU or DDP without ZeRO)
    raw_model_for_optim = model.module if ddp and not use_fsdp else model
    optimizer = raw_model_for_optim.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Load optimizer state if resuming
if checkpoint_for_resume is not None:
    if ddp and use_fsdp:
        # FSDP optimizer state loading
        optim_state = FSDP.optim_state_dict_to_load(checkpoint_for_resume['optimizer'], model, optimizer)
        optimizer.load_state_dict(optim_state)
    elif ddp and use_zero1:
        # ZeRO-1 checkpoint loading
        optimizer.load_state_dict(checkpoint_for_resume['optimizer'])
    else:
        optimizer.load_state_dict(checkpoint_for_resume['optimizer'])
    print("Loaded optimizer state from checkpoint")

checkpoint_for_resume = None  # free up memory

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Initialize JSON logger
json_logger = None
if save_log_to_json and master_process and LOGGER_AVAILABLE:
    json_logger = TrainingLogger(out_dir=out_dir, config=config)
    json_logger.log_metadata('world_size', ddp_world_size)
    json_logger.log_metadata('device', device)
    json_logger.log_metadata('dtype', dtype)
    json_logger.log_metadata('compile', compile)
    if ddp:
        json_logger.log_metadata('use_zero1', use_zero1)
        json_logger.log_metadata('use_fsdp', use_fsdp)
elif save_log_to_json and master_process and not LOGGER_AVAILABLE:
    print("Warning: training_logger not found, JSON logging disabled")

# unwrap DDP/FSDP container if needed
if ddp and use_fsdp:
    raw_model = model  # FSDP already provides access to original params with use_orig_params=True
elif ddp:
    raw_model = model.module  # DDP
else:
    raw_model = model  # single GPU

# ============================================================================
# STARTUP REPORT
# ============================================================================
if master_process:
    print("\n" + "="*80)
    print("🚀 TRAINING INITIALIZATION")
    print("="*80)
    
    # Model info
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    non_emb_params = raw_model.get_num_params()
    print(f"\n📊 MODEL ARCHITECTURE:")
    
    # Show architecture details if using modular system
    if MODULAR_ARCH_AVAILABLE and arch_preset != 'legacy' and hasattr(raw_model, 'config'):
        arch_summary = raw_model.config.get_architecture_summary()
        print(f"  Architecture Name:     {arch_summary['Architecture Name']}")
        print(f"  Total parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Non-embedding params:  {non_emb_params:,} ({non_emb_params/1e6:.2f}M)")
        print(f"  ")
        print(f"  ├─ Layers:             {n_layer}")
        print(f"  ├─ Hidden size:        {n_embd}")
        print(f"  ├─ Attention heads:    {n_head}")
        print(f"  ├─ Sequence length:    {block_size}")
        print(f"  ├─ Vocabulary size:    {arch_config.vocab_size}")
        print(f"  ")
        print(f"  ├─ Normalization:      {arch_summary['Normalization']}")
        print(f"  ├─ Activation:         {arch_summary['Activation']}")
        print(f"  ├─ Position Encoding:  {arch_summary['Position Encoding']}")
        print(f"  ├─ Attention Backend:  {arch_summary['Attention Backend']}")
        print(f"  ├─ Norm Position:      {arch_summary['Norm Position']}")
        print(f"  ├─ FFN Type:           {arch_summary['FFN Type']}")
        print(f"  ├─ FFN Expansion:      {arch_summary['FFN Expansion']}")
        print(f"  ├─ Bias:               {arch_summary['Bias in Linear']}")
        print(f"  ├─ Weight Tying:       {arch_summary['Weight Tying']}")
        print(f"  └─ Dropout:            {arch_summary['Dropout']}")
    else:
        # Legacy model reporting
        print(f"  Total parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Non-embedding params:  {non_emb_params:,} ({non_emb_params/1e6:.2f}M)")
        print(f"  Layers:                {n_layer}")
        print(f"  Hidden size:           {n_embd}")
        print(f"  Attention heads:       {n_head}")
        print(f"  Sequence length:       {block_size}")
        vocab_size_val = model_args.get('vocab_size', arch_config.vocab_size if MODULAR_ARCH_AVAILABLE and arch_preset != 'legacy' else 50304)
        print(f"  Vocabulary size:       {vocab_size_val}")
    
    # Training config
    effective_batch_size = batch_size * gradient_accumulation_steps * ddp_world_size
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print(f"  Batch size (micro):    {batch_size}")
    print(f"  Gradient accum steps:  {gradient_accumulation_steps}")
    print(f"  Effective batch size:  {effective_batch_size}")
    print(f"  Tokens per iteration:  {tokens_per_iter:,}")
    print(f"  Max iterations:        {max_iters:,}")
    print(f"  Learning rate:         {learning_rate}")
    print(f"  Weight decay:          {weight_decay}")
    print(f"  Gradient clip:         {grad_clip}")
    print(f"  LR warmup iters:       {warmup_iters}")
    print(f"  LR decay iters:        {lr_decay_iters}")
    
    # Hardware info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🖥️  HARDWARE:")
        print(f"  Device:                {gpu_name}")
        print(f"  GPUs:                  {ddp_world_size}")
        print(f"  Memory per GPU:        {gpu_memory:.1f} GB")
        print(f"  Precision:             {dtype}")
        print(f"  TF32 enabled:          True")
        print(f"  Compile:               {compile}")
        if ddp:
            parallelism = 'FSDP' if use_fsdp else ('DDP+ZeRO-1' if use_zero1 else 'DDP')
            print(f"  Parallelism:           {parallelism}")
    
    # Theoretical FLOPs calculation
    mfu_info = raw_model.estimate_mfu_detailed(batch_size * gradient_accumulation_steps, 1.0, 
                                                device_type='cuda', num_gpus=ddp_world_size)
    print(f"\n📈 THEORETICAL PERFORMANCE:")
    print(f"  Hardware peak:         {mfu_info['hardware_peak_tflops']:.1f} TFLOPS ({mfu_info['gpu_name']} {mfu_info['precision']})")
    print(f"  FLOPs per token:       {mfu_info['flops_per_token']/1e9:.2f} GFLOPs")
    print(f"  Attention/FFN ratio:   {mfu_info['attention_to_ffn_ratio']:.2f}")
    print(f"  Expected tokens/s @50% MFU: {(mfu_info['hardware_peak_flops'] * 0.5 / mfu_info['flops_per_token']):.0f}")
    
    # Log startup info
    if json_logger:
        hardware_info = {
            'gpu_name': gpu_name if torch.cuda.is_available() else 'cpu',
            'num_gpus': ddp_world_size,
            'gpu_memory_gb': gpu_memory if torch.cuda.is_available() else 0,
            'precision': dtype,
            'parallelism': 'FSDP' if use_fsdp else ('DDP+ZeRO-1' if use_zero1 else ('DDP' if ddp else 'single')),
        }
        json_logger.log_startup_info(raw_model, optimizer, config, hardware_info)
    
    print("\n" + "="*80)
    print("🏁 STARTING TRAINING")
    print("="*80 + "\n")

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"\n{'━'*80}")
        print(f"📊 EVALUATION │ Step {iter_num:>6d}")
        print(f"{'━'*80}")
        print(f"  Train loss: {losses['train']:.4f} │ Val loss: {losses['val']:.4f} │ LR: {lr:.2e}")
        print(f"{'━'*80}\n")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        
        # Log to JSON
        if json_logger:
            json_logger.log_eval(iter_num, losses['train'], losses['val'], lr=lr)
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                # Save checkpoint with FSDP-aware state dict handling
                if ddp and use_fsdp:
                    # FSDP checkpoint saving
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        save_policy,
                        optim_policy,
                    ):
                        model_state = model.state_dict()
                        optim_state = FSDP.optim_state_dict(model, optimizer)
                    
                    if master_process:
                        checkpoint = {
                            'model': model_state,
                            'optimizer': optim_state,
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"💾 Saving checkpoint to {out_dir}")
                        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                        torch.save(checkpoint, ckpt_path)
                        
                        # Log checkpoint to JSON
                        if json_logger:
                            json_logger.log_checkpoint(iter_num, losses['val'], ckpt_path)
                elif ddp and use_zero1:
                    # ZeRO-1 checkpoint saving
                    optimizer.consolidate_state_dict(to=0)
                    
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict() if ddp_rank == 0 else None,
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"💾 Saving checkpoint to {out_dir}")
                    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                    torch.save(checkpoint, ckpt_path)
                    
                    # Log checkpoint to JSON
                    if json_logger:
                        json_logger.log_checkpoint(iter_num, losses['val'], ckpt_path)
                else:
                    # Standard DDP or single GPU checkpoint
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"💾 Saving checkpoint to {out_dir}")
                    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                    torch.save(checkpoint, ckpt_path)
                    
                    # Log checkpoint to JSON
                    if json_logger:
                        json_logger.log_checkpoint(iter_num, losses['val'], ckpt_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        # Control gradient synchronization
        if ddp and use_fsdp:
            # FSDP: use no_sync() context manager for all but the last micro step
            sync_context = nullcontext() if micro_step == gradient_accumulation_steps - 1 else model.no_sync()
        elif ddp:
            # DDP: toggle require_backward_grad_sync flag
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            sync_context = nullcontext()
        else:
            # Single GPU: no synchronization needed
            sync_context = nullcontext()
        
        with sync_context:
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        
        if local_iter_num >= 5: # let the training loop settle a bit
            # Get detailed MFU breakdown
            mfu_breakdown = raw_model.estimate_mfu_detailed(
                batch_size * gradient_accumulation_steps, 
                dt, 
                device_type=device_type,
                num_gpus=ddp_world_size
            )
            running_mfu = mfu_breakdown['mfu'] if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu_breakdown['mfu']
            
            # Get memory stats
            memory_stats = raw_model.get_memory_stats()
            
            # Get gradient stats (less frequently to avoid overhead)
            grad_stats = None
            if iter_num % gradient_log_interval == 0:
                grad_stats = raw_model.get_gradient_stats()
            
            # Print detailed metrics
            print(f"\n{'─'*80}")
            print(f"📍 Iter {iter_num:>6d} │ Loss: {lossf:.4f} │ Time: {dt*1000:.0f}ms │ LR: {lr:.2e}")
            print(f"{'─'*80}")
            
            # MFU breakdown
            print(f"⚡ MFU: {mfu_breakdown['mfu_percent']:.2f}% │ "
                  f"Achieved: {mfu_breakdown['achieved_tflops']:.1f} TF │ "
                  f"Peak: {mfu_breakdown['hardware_peak_tflops']:.1f} TF")
            print(f"   Tokens/s: {mfu_breakdown['tokens_per_sec']:.0f} │ "
                  f"FLOPs/token: {mfu_breakdown['flops_per_token']/1e9:.1f} GF")
            
            # Memory
            if memory_stats:
                print(f"💾 Memory: {memory_stats['allocated_gb']:.2f} GB alloc │ "
                      f"{memory_stats['max_allocated_gb']:.2f} GB peak │ "
                      f"{memory_stats['reserved_gb']:.2f} GB reserved")
            
            # Gradients (if available)
            if grad_stats:
                print(f"📊 Gradients: norm={grad_stats['global_norm']:.4f} │ "
                      f"mean={grad_stats['grad_mean']:.2e} │ "
                      f"std={grad_stats['grad_std']:.2e}")
            
            # Log to JSON with detailed metrics
            if json_logger:
                json_logger.log_iter_detailed(iter_num, lossf, dt*1000, mfu_breakdown, 
                                             memory_stats, grad_stats)
                # Auto-save periodically
                if iter_num % log_save_interval == 0:
                    json_logger.save()
        else:
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.0f}ms (warming up...)")
            if json_logger:
                json_logger.log_iter(iter_num, lossf, dt*1000, -100.0)
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Finalize and save JSON log
if json_logger and master_process:
    json_logger.finalize(final_iter=iter_num-1, best_val_loss=best_val_loss)
    json_logger.save()
    print(f"\n{'='*80}")
    print(f"✅ Training completed! Log saved to: {json_logger.log_file}")
    print(f"{'='*80}\n")

if ddp:
    destroy_process_group()

