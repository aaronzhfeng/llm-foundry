# OOM Analysis: Why ZeRO-1 Helps (and Other Solutions)

**Date**: 2025-11-21  
**Model**: Qwen 3 1.8B (1.829B parameters)  
**Hardware**: 8× NVIDIA B200 (179 GB usable per GPU)  
**Issue**: OOM during `torch.compile` backward pass (tried to allocate 13.89 GB)

---

## Your Change: `use_zero1 = True`

### What ZeRO-1 Does

ZeRO-1 (Zero Redundancy Optimizer, Stage 1) shards **optimizer states** across GPUs while keeping model parameters and gradients replicated on each GPU.

**Memory Breakdown for DDP vs ZeRO-1**:

| Component | DDP (Per GPU) | ZeRO-1 (Per GPU) | Savings |
|-----------|---------------|------------------|---------|
| Model Params (BF16) | 3.66 GB | 3.66 GB | 0 GB |
| Gradients (BF16) | 3.66 GB | 3.66 GB | 0 GB |
| **Optimizer States** (AdamW) | **14.64 GB** | **1.83 GB** | **12.81 GB** |
| Activations (est.) | 150 GB | 150 GB | 0 GB |
| **Total** | **~172 GB** | **~159 GB** | **~13 GB** |

### Why This Helps Your OOM

Your OOM error:
```
Tried to allocate 13.89 GiB
GPU has 178.36 GiB total, 164.31 GiB in use, 11.54 GiB free
```

**Analysis**:
- You needed **13.89 GB** but only had **11.54 GB** free
- ZeRO-1 saves **~12.8 GB** of optimizer state memory
- This gives you **~24 GB** free (11.54 + 12.8) → **enough headroom!**

### Trade-offs

**Pros**:
- ✅ Saves ~12.8 GB per GPU (7% of B200 memory)
- ✅ Nearly zero performance overhead (<1%)
- ✅ Compatible with `torch.compile` and all optimizations

**Cons**:
- ❌ Slightly slower optimizer step (needs all-gather of params)
- ❌ Overhead becomes noticeable with >64 GPUs
- ❌ Not compatible with `use_zero1=True` + `use_fsdp=True` (mutually exclusive)

### Recommendation

✅ **Keep `use_zero1 = True`** for this run. It's the right solution for your OOM issue.

---

## Alternative Solutions (If ZeRO-1 Doesn't Work)

### Option 1: Memory Allocator Fix (Already Applied)

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

This fixes **fragmentation** where PyTorch can't find a contiguous 13.89 GB block even though total free memory exists.

**Status**: ✅ Already added to your config file

---

### Option 2: Reduce Batch Size

Reduce from `batch_size=24` to `batch_size=20`:

```bash
--batch_size=20 --gradient_accumulation_steps=2
```

**Impact**:
- Saves ~3-4 GB per GPU
- Reduces global batch size from 384 to 320 (17% reduction)
- **MFU impact**: ~2-3% lower (less GPU saturation)

---

### Option 3: Activation Checkpointing (FSDP-specific)

If you switch to FSDP:
```python
use_fsdp = True
fsdp_activation_checkpointing = True
```

**Impact**:
- Saves ~50% of activation memory
- Increases training time by ~30% (recomputation overhead)
- **Not compatible with ZeRO-1**

---

### Option 4: Gradient Checkpointing (Manual)

Wrap transformer blocks with `torch.utils.checkpoint.checkpoint()`:

```python
# In model_builder.py TransformerBlock.forward()
if self.config.use_gradient_checkpointing:
    x = torch.utils.checkpoint.checkpoint(
        self.attn, self.norm1(x), token_positions, use_reentrant=False
    )
```

**Impact**:
- Saves ~60-70 GB of activation memory
- Increases training time by ~20%
- **Compatible with ZeRO-1**

---

## Memory Analysis: Qwen 3 1.8B on B200

### Current Usage (from your logs)

```
max_allocated_gb: 179.779488768  # Peak usage before OOM
reserved_gb: 182.856974336       # Total reserved by PyTorch
```

**Breakdown** (estimated):
- Model params (BF16): 3.66 GB
- Gradients (BF16): 3.66 GB
- Optimizer states (AdamW, FP32): 14.64 GB
- Activations (forward pass): ~150 GB
- Compilation buffers: ~8 GB
- **Total**: ~180 GB (matches your peak!)

### After ZeRO-1

**Breakdown** (estimated):
- Model params (BF16): 3.66 GB
- Gradients (BF16): 3.66 GB
- **Optimizer states (ZeRO-1)**: **1.83 GB** (↓ 12.8 GB)
- Activations (forward pass): ~150 GB
- Compilation buffers: ~8 GB
- **Total**: **~167 GB** (↓ 13 GB, **12 GB free headroom**)

---

## Recommended Configuration

```python
# config/full_qwen3_1.8b_b200_optimal.py

# Memory optimizations
use_zero1 = True                    # ✅ Saves 12.8 GB optimizer state
use_fsdp = False                    # ❌ Incompatible with ZeRO-1

# Batch size (can try increasing after confirming ZeRO-1 works)
batch_size = 24                     # Start here
gradient_accumulation_steps = 2     # Global batch = 384

# Memory allocator fix (already applied)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Compilation (keep enabled)
compile = True                      # Critical for B200 performance
```

---

## Testing Strategy

### Phase 1: Verify ZeRO-1 Fixes OOM (5-10 min)

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=50 \
  --batch_size=24 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False
```

**Expected**:
- ✅ No OOM error
- ✅ Memory usage ~167 GB per GPU (down from 180 GB)
- ✅ MFU ~43-45% (similar to before)

### Phase 2: Try Larger Batch Size (if Phase 1 succeeds)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=50 \
  --batch_size=28 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False
```

**Goal**: Push batch size higher to improve MFU (up to 50-55%)

---

## Why Your Previous Run Succeeded

Looking at your successful run (`run_20251121_102942.json`):

```json
"gpu_memory_gb": 191.514148864,  // Reported 191.5 GB (not 179 GB!)
```

**Hypothesis**:
1. Different measurement (includes system memory pool?)
2. Different driver version (less reserved memory)
3. Run on a different node with more GPU memory
4. ECC was disabled (adds ~10% overhead when enabled)

**Your current B200s**: Only have **179 GB usable** (verified via `nvidia-smi`)

---

## Summary

| Solution | Memory Saved | Performance Impact | Difficulty | Status |
|----------|--------------|--------------------|-----------|---------
| **ZeRO-1** | **12.8 GB** | **~1%** | **Easy** | ✅ **Implemented** |
| Memory allocator | 0 GB (fixes fragmentation) | 0% | Easy | ✅ Implemented |
| Reduce batch size | 3-4 GB | -3% MFU | Easy | ❌ Not needed |
| Gradient checkpointing | 60-70 GB | -20% speed | Medium | ❌ Overkill |

**Recommendation**: Your `use_zero1 = True` change is the correct solution. Run the test now!

