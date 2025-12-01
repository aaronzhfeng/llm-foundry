# Phase 1: ZeRO-1 Optimizer State Sharding

## Overview

This implementation adds **ZeRO-1 (Zero Redundancy Optimizer - Stage 1)** to nanoGPT, which shards optimizer states across GPUs while keeping model parameters and gradients replicated.

## Benefits

- **Memory Savings**: Reduces optimizer state memory by ~4x with 4 GPUs, ~8x with 8 GPUs
- **Minimal Code Changes**: Only ~30 lines changed from baseline nanoGPT
- **Compatible**: Works with existing DDP infrastructure
- **Trade-off**: Slight communication overhead during optimizer step

## Changes from nanoGPT

### Modified Files

#### 1. `train.py`

**Line 29**: Added import for ZeroRedundancyOptimizer
```python
from torch.distributed.optim import ZeroRedundancyOptimizer
```

**Lines 72-73**: Added ZeRO-1 configuration flag
```python
# ZeRO-1 settings
use_zero1 = True  # use ZeRO-1 optimizer state sharding
```

**Lines 202-233**: Replaced optimizer initialization with ZeRO-1 aware version
- When `ddp=True` and `use_zero1=True`: Uses `ZeroRedundancyOptimizer` wrapper
- Otherwise: Uses standard optimizer from `model.configure_optimizers()`
- Manually creates parameter groups (decay vs no-decay) for ZeRO-1
- Note: Fused AdamW is not used with ZeRO-1 (compatibility issue)

**Lines 235-236**: Updated checkpoint loading to handle None optimizer state
- ZeRO-1 only saves optimizer state on rank 0

**Lines 312-325**: Updated checkpoint saving with ZeRO-1 state consolidation
- Calls `optimizer.consolidate_state_dict(to=0)` before saving
- Only rank 0 saves optimizer state (others save None)

### Unchanged Files

- `model.py`: No changes (model architecture remains the same)
- `configurator.py`: No changes
- `config/train_gpt2.py`: No changes

## File Structure

```
phase1_zero1/
├── model.py              # Unchanged from nanoGPT
├── train.py              # Modified for ZeRO-1
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

For the full OpenWebText dataset (takes longer):
```bash
cd /root/llm_TII/system_implementation/nanoGPT/data/openwebtext
python prepare.py
```

## Test Commands

### Single GPU (baseline, no ZeRO-1)
```bash
cd /root/llm_TII/system_implementation/phase1_zero1
python train.py --dataset=shakespeare --use_zero1=False --batch_size=12 --max_iters=100 --eval_interval=50 --compile=False
```

### Multi-GPU with ZeRO-1 (2 GPUs)
```bash
cd /root/llm_TII/system_implementation/phase1_zero1
torchrun --standalone --nproc_per_node=2 train.py --dataset=shakespeare --use_zero1=True --batch_size=12 --max_iters=100 --eval_interval=50
```

### Multi-GPU with ZeRO-1 (4 GPUs)
```bash
cd /root/llm_TII/system_implementation/phase1_zero1
torchrun --standalone --nproc_per_node=4 train.py --dataset=shakespeare --use_zero1=True --batch_size=12 --max_iters=100 --eval_interval=50
```

### Multi-GPU with ZeRO-1 (8 GPUs)
```bash
cd /root/llm_TII/system_implementation/phase1_zero1
torchrun --standalone --nproc_per_node=8 train.py --dataset=shakespeare --use_zero1=True --batch_size=12 --max_iters=100 --eval_interval=50
```

### Multi-Node Example (2 nodes, 4 GPUs each)
**On master node (rank 0):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=MASTER_IP --master_port=29500 \
  train.py --use_zero1=True
```

**On worker node (rank 1):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr=MASTER_IP --master_port=29500 \
  train.py --use_zero1=True
```

### Test with Custom Config
```bash
cd /root/llm_TII/system_implementation/phase1_zero1
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py --use_zero1=True
```

## Expected Behavior

### Memory Usage (per GPU, GPT2-124M)

| Configuration | Baseline DDP | ZeRO-1 (4 GPUs) | Memory Saved |
|---------------|--------------|-----------------|--------------|
| Model Params  | ~0.5 GB      | ~0.5 GB         | 0%           |
| Gradients     | ~0.5 GB      | ~0.5 GB         | 0%           |
| Optimizer     | ~2.0 GB      | ~0.5 GB         | **75%**      |
| **Total**     | **~3.0 GB**  | **~1.5 GB**     | **~50%**     |

### Performance Impact

- **Throughput**: ~5-10% slower than baseline DDP (due to optimizer state communication)
- **Scalability**: Overhead decreases with larger models
- **Loss Convergence**: Identical to baseline DDP (numerically equivalent)

## Verification Checklist

- [ ] Single GPU training works (use_zero1=False)
- [ ] Multi-GPU (2, 4, 8) works with ZeRO-1
- [ ] Checkpoint save works (only rank 0 saves optimizer state)
- [ ] Checkpoint load/resume works
- [ ] Memory per GPU is reduced compared to baseline DDP
- [ ] Loss values match baseline DDP training
- [ ] Training completes without errors

## Debugging Tips

1. **Check ZeRO-1 is enabled**: Look for log message `"Using ZeRO-1 optimizer state sharding"`
2. **Monitor memory**: Use `nvidia-smi` during training to verify memory reduction
3. **Checkpoint issues**: Ensure only rank 0 checkpoint contains optimizer state
4. **Performance**: ZeRO-1 should be slightly slower than DDP but use less memory

## Known Limitations

1. **Fused AdamW**: Not compatible with ZeroRedundancyOptimizer (uses standard AdamW)
2. **Communication Overhead**: ~5-10% slower than baseline DDP on small models
3. **Checkpoint Size**: Only rank 0 checkpoint contains optimizer state (this is correct behavior)
4. **Single GPU**: ZeRO-1 is only activated with DDP (multi-GPU); single GPU uses standard optimizer

## References

- [PyTorch ZeroRedundancyOptimizer Docs](https://pytorch.org/docs/stable/distributed.optim.html)
- [ZeRO Paper (Rajbhandari et al.)](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)

