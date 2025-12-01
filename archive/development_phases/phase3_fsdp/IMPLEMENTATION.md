# Phase 3: FSDP (Fully Sharded Data Parallel)

## Overview

This implementation adds **FSDP (Fully Sharded Data Parallel)** to nanoGPT, which shards model parameters, gradients, AND optimizer states across all GPUs. This is similar to DeepSpeed ZeRO-3 and provides the most aggressive memory savings of all three phases.

## Benefits

- **Maximum Memory Savings**: Shards everything (params + grads + optimizer states)
- **Scale to Larger Models**: Can train models that don't fit on a single GPU
- **Memory Reduction**: ~8x less memory per GPU with 8 GPUs (compared to DDP)
- **Built into PyTorch**: No external dependencies like DeepSpeed
- **torch.compile Compatible**: Works with PyTorch 2.0+ compilation

## Changes from nanoGPT

### Modified Files

#### 1. `train.py`

**Lines 30-45**: Added FSDP imports
```python
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
```

**Lines 88-91**: Added FSDP configuration flags
```python
# FSDP settings
use_fsdp = True  # use FSDP instead of DDP
fsdp_min_num_params = 1e6  # minimum number of parameters for auto-wrapping (1M default)
fsdp_activation_checkpointing = False  # enable activation checkpointing with FSDP (future feature)
```

**Lines 219-221**: Modified optimizer initialization timing
- Moved optimizer creation to after FSDP wrapping
- Stores checkpoint temporarily for later optimizer state loading

**Lines 231-271**: Replaced DDP wrapping with FSDP wrapping
- Configures mixed precision policy based on dtype
- Sets up auto-wrap policy (wraps layers with > 1M params by default)
- Wraps model with FSDP using proper configuration
- Uses `use_orig_params=True` for torch.compile compatibility
- Falls back to DDP if `use_fsdp=False`

**Lines 273-306**: Created optimizer after FSDP wrapping
- For FSDP: manually creates parameter groups and AdamW optimizer
- For DDP/single GPU: uses model's `configure_optimizers()` method
- Handles FSDP-specific optimizer state loading with `optim_state_dict_to_load()`

**Lines 315-320**: Updated raw_model reference
- FSDP uses `model` directly (with `use_orig_params=True`)
- DDP uses `model.module`
- Single GPU uses `model`

**Lines 344-381**: Updated checkpoint saving with FSDP support
- FSDP path: Uses `FSDP.state_dict_type()` context manager
- Saves full state dict with `rank0_only=True`
- Uses `FSDP.optim_state_dict()` for optimizer state
- Only rank 0 saves checkpoint to disk
- Falls back to standard checkpoint for DDP/single GPU

**Lines 387-407**: Updated gradient accumulation for FSDP
- FSDP: Uses `model.no_sync()` context manager for non-final micro-steps
- DDP: Uses `require_backward_grad_sync` flag (original behavior)
- Single GPU: No synchronization needed

### Unchanged Files

- `model.py`: No changes (model architecture unchanged)
- `configurator.py`: No changes
- `config/train_gpt2.py`: No changes

## File Structure

```
phase3_fsdp/
├── model.py              # Unchanged from nanoGPT
├── train.py              # Modified for FSDP
├── configurator.py       # Unchanged
├── config/
│   └── train_gpt2.py     # Unchanged
├── data/                 # Symlink to ../nanoGPT/data/
└── IMPLEMENTATION.md     # This file
```

## Data Preparation

Before running any tests, prepare the dataset:

```bash
# Install tiktoken (required for data preparation)
pip install tiktoken

# Prepare Shakespeare dataset (fast, good for testing)
cd /root/llm_TII/system_implementation/nanoGPT/data/shakespeare
python prepare.py
```

## Test Commands

### Single GPU (baseline, no FSDP)
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
python train.py --dataset=shakespeare --use_fsdp=False --batch_size=12 --max_iters=100 --eval_interval=50 --compile=False
```

### Multi-GPU with FSDP (2 GPUs)
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=2 train.py --dataset=shakespeare --use_fsdp=True --batch_size=12 --max_iters=100
```

### Multi-GPU with FSDP (4 GPUs)
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=4 train.py --dataset=shakespeare --use_fsdp=True --batch_size=12 --max_iters=100
```

### Multi-GPU with FSDP (8 GPUs)
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=8 train.py --dataset=shakespeare --use_fsdp=True --batch_size=12 --max_iters=100
```

### With Custom Wrapping Policy
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=4 train.py \
  --use_fsdp=True \
  --fsdp_min_num_params=500000 \
  --batch_size=12
```

### With torch.compile
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=4 train.py \
  --use_fsdp=True \
  --compile=True \
  --batch_size=12
```

### Resume from Checkpoint
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=4 train.py \
  --use_fsdp=True \
  --init_from=resume \
  --batch_size=12
```

### Multi-Node Example (2 nodes, 4 GPUs each)
**On master node:**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=MASTER_IP --master_port=29500 \
  train.py --use_fsdp=True
```

**On worker node:**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr=MASTER_IP --master_port=29500 \
  train.py --use_fsdp=True
```

## Expected Behavior

### Memory Usage (per GPU, GPT2-124M)

| Configuration | Baseline DDP | FSDP (4 GPUs) | FSDP (8 GPUs) | Memory Saved |
|---------------|--------------|---------------|---------------|--------------|
| Model Params  | ~0.5 GB      | ~0.125 GB     | ~0.06 GB      | **75-88%**   |
| Gradients     | ~0.5 GB      | ~0.125 GB     | ~0.06 GB      | **75-88%**   |
| Optimizer     | ~2.0 GB      | ~0.5 GB       | ~0.25 GB      | **75-88%**   |
| **Total**     | **~3.0 GB**  | **~0.75 GB**  | **~0.37 GB**  | **75-88%**   |

### Scaling to Larger Models

With 8x A100 (80GB each):
- **DDP**: Can train ~20B parameters
- **FSDP**: Can train ~160B parameters (8x larger)

### Performance Impact

| Metric | DDP Baseline | FSDP | Notes |
|--------|--------------|------|-------|
| Throughput | 100% | 85-95% | Communication overhead |
| Memory/GPU | 100% | ~12-25% | Massive savings |
| Scalability | Linear to ~8 GPUs | Linear to 100+ GPUs | Better for large models |
| Setup Time | Fast | +10-30s | FSDP initialization overhead |

### Loss Convergence

- Numerically equivalent to DDP
- Same final loss values
- May have slightly different trajectory due to rounding differences in mixed precision

## Verification Checklist

- [ ] Single GPU works (use_fsdp=False falls back to baseline)
- [ ] Multi-GPU (2, 4, 8) works with FSDP
- [ ] Memory per GPU is significantly reduced (check with `nvidia-smi`)
- [ ] Checkpoint save works (only rank 0 creates checkpoint)
- [ ] Checkpoint load/resume works
- [ ] torch.compile works with FSDP
- [ ] Loss values similar to baseline DDP
- [ ] Training completes without OOM errors
- [ ] Gradient clipping works correctly
- [ ] Mixed precision (bfloat16/float16) works

## Debugging Tips

### 1. Check FSDP is enabled
Look for log message: `"Wrapping model with FSDP..."`

### 2. Monitor memory usage
```bash
watch -n 0.5 nvidia-smi
```

You should see significantly less memory per GPU compared to DDP.

### 3. Verify sharding
```python
# In train.py, after FSDP wrapping:
if master_process:
    print(f"FSDP sharding info:")
    for name, module in model.named_modules():
        if isinstance(module, FSDP):
            print(f"  {name}: {module.numel_padded_per_param}")
```

### 4. Check checkpoint structure
```python
import torch
ckpt = torch.load('out/ckpt.pt')
print(f"Model keys: {len(ckpt['model'].keys())}")
print(f"Optimizer keys: {len(ckpt['optimizer'].keys()) if ckpt['optimizer'] else 'None'}")
```

### 5. Profile communication
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=4 train.py --use_fsdp=True --max_iters=10
```

### 6. Test with smaller model
```bash
torchrun --nproc_per_node=2 train.py \
  --use_fsdp=True \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --max_iters=50
```

## Known Limitations

### 1. Communication Overhead
- FSDP has ~10-15% throughput overhead compared to DDP
- More pronounced on smaller models
- Decreases relative to model size (large models benefit more)

### 2. Initialization Time
- FSDP takes 10-30s to initialize and wrap model
- One-time cost at training start

### 3. torch.compile Compatibility
- Requires `use_orig_params=True` (already set)
- May have issues with PyTorch < 2.1
- Test thoroughly with your PyTorch version

### 4. Checkpoint Size
- Full state dict checkpoint can be large
- Consider sharded checkpoint saving for very large models
- Only rank 0 needs disk space for checkpoint

### 5. Wrapping Policy Tuning
- Default `fsdp_min_num_params=1e6` may not be optimal
- Too small: too many FSDP units, more communication
- Too large: less memory savings
- Experiment with values: 500K - 5M

### 6. CPU Offloading
- Not implemented in this version (future extension)
- Would enable even larger models at cost of speed

## Advanced Configuration

### Custom Wrapping Policy

Instead of size-based, you can wrap specific layers:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import Block

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Block},  # Wrap each transformer block
)
```

### Sharding Strategy

Change from default `FULL_SHARD` to hybrid sharding:

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # Shard within node, replicate across nodes
    # ... other args
)
```

### Activation Checkpointing

Enable gradient checkpointing to save even more memory:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Apply to transformer blocks
def check_fn(submodule):
    return isinstance(submodule, Block)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn,
)
```

## Comparison with Other Phases

| Feature | Phase 1 (ZeRO-1) | Phase 2 (Triton) | Phase 3 (FSDP) |
|---------|------------------|------------------|----------------|
| Memory Savings | ~50% (optimizer only) | ~1-2% | **~75-88%** (all) |
| Speed Impact | -5% to -10% | +5% to +10% | **-10% to -15%** |
| Complexity | Low | Medium | **High** |
| Largest Model | 1x of DDP | 1x of DDP | **8x of DDP** |
| torch.compile | ✅ Compatible | ✅ Compatible | ✅ Compatible |
| Best For | Medium models | Speed optimization | **Huge models** |

## Future Extensions

### 1. CPU Offloading
```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
    # ...
)
```

### 2. Sharded Checkpointing
Instead of full state dict, save sharded checkpoints:
```python
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig

save_policy = ShardedStateDictConfig(offload_to_cpu=True)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    # Each rank saves its shard
```

### 3. Combine with ZeRO-1
Use FSDP for model sharding + ZeRO-1 for additional optimizer sharding within FSDP units.

### 4. Combine with Triton
Use FSDP for memory + Triton kernels for speed.

## References

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [FSDP API Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [ZeRO Paper (Rajbhandari et al.)](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP vs DeepSpeed](https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/)
- [Llama 2 Training with FSDP](https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/)

