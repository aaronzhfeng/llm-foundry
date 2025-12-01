## NanoGPT Training System Overview

This document summarizes the training system implemented in this `nanoGPT` folder: parallelization, memory/compute strategies, kernels/graph compile choices, core model design, optimization/regularization, data pipeline, and other notable implementation details. Each section cites code snippets from the repository for validation.

---

### Implementation Choices Summary

**Parallelization:**
- ✅ Data Parallel (DDP)
- ❌ Tensor Parallel (TP)
- ❌ Pipeline Parallel (PP)
- ❌ FSDP / ZeRO / DeepSpeed

**Memory & Compute Optimizations:**
- ✅ Mixed Precision (bfloat16/float16 + GradScaler)
- ✅ TF32 on matmul & cuDNN
- ✅ Gradient Accumulation
- ✅ Pinned Memory + Async Transfers
- ✅ Gradient Clipping
- ❌ Activation Checkpointing
- ❌ Optimizer/Gradient Offloading

**Kernels & Graph Compilation:**
- ✅ PyTorch 2.0 `torch.compile`
- ✅ FlashAttention (via SDPA)
- ✅ Fused AdamW Optimizer
- ❌ Custom Triton Kernels
- ❌ Manual Kernel Fusion

**Model Architecture:**
- **Type:** Dense GPT (no MoE)
- **Normalization:** LayerNorm (optional bias, default False)
- **Activation:** GELU
- **Attention:** Multi-head causal self-attention
- **Positional Embedding:** Learned absolute
- **Dropout:** Configurable (default 0.0 for pretraining)
- **Weight Tying:** Token embeddings ↔ output head

**Optimization & Regularization:**
- **Optimizer:** AdamW with grouped weight decay
- **LR Schedule:** Warmup + Cosine Decay
- **Loss:** Cross-Entropy
- **Gradient Clipping:** 1.0

**Data Pipeline:**
- Memory-mapped file streaming (`np.memmap`)
- Random block sampling
- Pinned memory + non-blocking CUDA transfers

---

### Parallelization

- **Data Parallel (DDP)**: Supported via PyTorch DistributedDataParallel. Initialization, device assignment, gradient sync control, and teardown are handled in `train.py`.

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
...
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    ...
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
...
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
...
if ddp:
    destroy_process_group()
```

- **Tensor Parallel (TP)**: Not implemented.
- **Pipeline Parallel (PP)**: Not implemented.
- **FSDP / ZeRO (e.g., DeepSpeed)**: Not implemented. Training uses vanilla DDP with gradient accumulation.

---

### Memory & Compute

- **Mixed Precision + TF32**: Uses autocast for `bfloat16` or `float16` on CUDA, with GradScaler for `float16`. TF32 is enabled on matmul and cuDNN for throughput.

```python
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
...
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
...
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

- **Activation Checkpointing**: Not used. There are no calls to `torch.utils.checkpoint`. Memory savings come from mixed precision and FlashAttention (see below), not from activation recomputation.

- **Gradient Accumulation**: Implemented and scaled per world size to simulate larger batches without increasing memory.

```python
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
...
if ddp:
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
...
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')
    scaler.scale(loss).backward()
```

- **Offloading Strategy**: No optimizer/gradient offloading. Data is streamed from a memory-mapped file and transferred via pinned CPU memory to GPU asynchronously.

```python
# poor man's data loader
if split == 'train':
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
...
if device_type == 'cuda':
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
...
# immediately async prefetch next batch while model is doing the forward pass on the GPU
X, Y = get_batch('train')
```

- **Gradient Clipping & Zeroing**

```python
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
...
optimizer.zero_grad(set_to_none=True)
```

---

### Kernels & Graph (Tiling/Triton/Kernel Fusion)

- **Torch Compile**: Uses `torch.compile` (PyTorch 2.x) to optimize and fuse graphs.

```python
compile = True # use PyTorch 2.0 to compile the model to be faster
...
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)
```

- **FlashAttention (via SDPA)**: Attention uses `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` when available (PyTorch >= 2.0). This dispatches to efficient fused kernels (FlashAttention-style) under the hood.

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
...
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None,
        dropout_p=self.dropout if self.training else 0,
        is_causal=True,
    )
```

- **Fused Optimizer**: AdamW uses PyTorch’s fused implementation on CUDA if available.

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
print(f"using fused AdamW: {use_fused}")
```

- **Custom Triton / Manual Fusion**: None present. The project relies on PyTorch compiler and built-in fused kernels rather than hand-written Triton or custom CUDA kernels.

---

### Model Architecture

- **Core Blocks**: GPT-style transformer blocks with pre-norm and residual connections.

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

- **Normalization**: `LayerNorm` with optional bias (wrapped around `F.layer_norm`). Default training config sets `bias=False` (no bias in Linear and LayerNorm) for speed and slight quality gains.

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
...
# train.py defaults
bias = False # do we use bias inside LayerNorm and Linear layers?
```

- **Activation**: GELU in the MLP with 4× expansion.

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
```

- **Attention**: Multi-head causal self-attention, with dropout on attention and residual outputs, both governed by `config.dropout`.

```python
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
...
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
```

- **Positional Embeddings**: Learned absolute position embeddings.

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
```

- **Dropout**: Global `config.dropout` applied to attention, MLP output, and the top-level embedding dropout. Default `dropout = 0.0` in `train.py` for pretraining.

```python
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
```

- **Weight Tying**: Ties token embeddings and output projection weights.

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.transformer.wte.weight = self.lm_head.weight # weight tying
```

- **MoE or Dense?**: Dense. The MLP is a standard dense feedforward; there is no Mixture-of-Experts.

---

### Optimization & Regularization

- **Optimizer**: AdamW with parameter grouping (decay for 2D tensors like weights/embeddings; no decay for biases/LayerNorms).

```python
# group by dimensionality for weight decay
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0},
]
```

- **Default Hyperparameters (train.py)**

```python
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
```

- **Learning Rate Schedule**: Warmup then cosine decay to `min_lr`.

```python
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

- **Loss**: Cross-entropy over next-token logits, ignoring `-1` targets (for masking).

```python
logits = self.lm_head(x)
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,
)
```

---

### Data Pipeline

- **Dataset & Batching**: Uses `np.memmap` to stream tokenized data directly from disk; random crops to `block_size`. Transfers use pinned memory + non-blocking CUDA copies.

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
if device_type == 'cuda':
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
```

---

### Checkpointing & Resume

- **Saving**: Saves checkpoints after eval (or always, if enabled) with model, optimizer, config, and iteration state.

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

- **Resume**: Restores model and optimizer and synchronizes model args; strips any `_orig_mod.` prefixes.

```python
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
...
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
```

---

### MFU (Model FLOPs Utilization)

- **Runtime Estimation**: Provided helper to estimate MFU based on iteration timing; used in training and `bench.py`.

```python
mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
```

`model.estimate_mfu` follows PaLM Appendix B to approximate FLOPs per iteration:

```python
N = self.get_num_params()
L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
flops_per_token = 6*N + 12*L*H*Q*T
flops_per_fwdbwd = flops_per_token * T
flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
```

---

### What’s Not Implemented (Limitations)

- **Tensor Parallelism (TP):** Not present.
- **Pipeline Parallelism (PP):** Not present.
- **FSDP / ZeRO / DeepSpeed:** Not present.
- **Activation Checkpointing:** Not present.
- **Custom Triton Kernels / Manual Kernel Fusion:** Not present; relies on `torch.compile`, fused AdamW, and SDPA/FlashAttention in PyTorch.
- **MoE:** Not present; the model is a standard dense transformer with 4× MLP.

---

### Quick Defaults Snapshot (from `train.py`)

```python
# data & model
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimization
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# precision & compile
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# parallel & batch scaling
gradient_accumulation_steps = 5 * 8
backend = 'nccl'
```

---

### Bottom Line

- Training runs on single- or multi-GPU with DDP. No TP/PP/FSDP/ZeRO.
- Memory/throughput optimizations: mixed precision (bf16/fp16 + GradScaler), TF32 matmul/cudnn, FlashAttention via SDPA, gradient accumulation, pinned-memory async transfers, and `torch.compile`.
- Model: dense GPT (pre-norm, GELU, learned absolute positions, weight tying), configurable dropout (default 0.0), AdamW with selective weight decay, cosine LR schedule with warmup, cross-entropy loss.

If you’d like, we can extend this with activation checkpointing, FSDP/ZeRO, TP/PP, or custom Triton kernels.
