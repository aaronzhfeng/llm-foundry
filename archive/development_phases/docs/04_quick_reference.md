# Quick Reference: Three-Phase Implementation

## 📦 First: Prepare Data

```bash
pip install tiktoken
cd /root/llm_TII/system_implementation/nanoGPT/data/shakespeare
python prepare.py
```

## 📊 New: Automatic JSON Logging

All phases now automatically save training logs to JSON files:

```bash
# After training, view your logs
cd phase1_zero1
python visualize_training.py  # Auto-finds latest log
# Creates: run_TIMESTAMP_analysis.png

# Or specify a log file
python visualize_training.py out/run_20251031_173650.json
```

**Features:**
- ✅ One JSON file per run with timestamp
- ✅ Complete training history (iterations, losses, times, MFU)
- ✅ Eval steps and checkpoints tracked
- ✅ Easy analysis with Python/pandas
- ✅ Automatic visualization script included

See `LOGGING_GUIDE.md` for full documentation.

## 🚀 Fast Testing Commands

### Phase 1: ZeRO-1 (Memory Optimization)
```bash
cd /root/llm_TII/system_implementation/phase1_zero1
torchrun --standalone --nproc_per_node=4 train.py --dataset=shakespeare --use_zero1=True --max_iters=100 --compile=False
```
**Expected**: ~50% memory reduction, "Using ZeRO-1 optimizer state sharding" in logs

---

### Phase 2: Triton (Speed Optimization)
```bash
# First install Triton
pip install triton

cd /root/llm_TII/system_implementation/phase2_triton
python train.py --dataset=shakespeare --max_iters=100 --compile=False
```
**Expected**: ~5-10% speed improvement, "using Triton-accelerated LayerNorm" in logs

---

### Phase 3: FSDP (Maximum Memory Savings)
```bash
cd /root/llm_TII/system_implementation/phase3_fsdp
torchrun --standalone --nproc_per_node=4 train.py --dataset=shakespeare --use_fsdp=True --max_iters=100 --compile=False
```
**Expected**: ~75% memory reduction, "Wrapping model with FSDP..." in logs

---

## 📊 At-a-Glance Comparison

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Memory Saved** | 🟢 50% | 🔵 ~0% | 🟢🟢 75-88% |
| **Speed Change** | 🔴 -5% | 🟢 +5% | 🔴 -10% |
| **Complexity** | 🟢 Low | 🟡 Medium | 🔴 High |
| **Best For** | Medium models | Speed | Huge models |

---

## 🎯 Decision Tree

```
Do you need more memory?
├─ No → Use Phase 2 (Triton) for speed
└─ Yes → How much memory?
    ├─ 50% reduction enough → Use Phase 1 (ZeRO-1)
    └─ Need maximum savings → Use Phase 3 (FSDP)
```

---

## 📝 Files Changed per Phase

**Phase 1**: Only `train.py` (30 lines)  
**Phase 2**: `model.py` + new `kernels/` package  
**Phase 3**: Only `train.py` (150+ lines)

---

## ⚡ Key Log Messages to Look For

### Phase 1 ✅
```
Using ZeRO-1 optimizer state sharding
num decayed parameter tensors: ...
```

### Phase 2 ✅
```
number of parameters: 124.44M
using Triton-accelerated LayerNorm
```

### Phase 3 ✅
```
Wrapping model with FSDP...
FSDP enabled with min_params=1000000.0, mixed_precision=bfloat16
```

---

## 🔧 Common Flags

| Flag | Phase | Description |
|------|-------|-------------|
| `--use_zero1=True/False` | 1 | Enable/disable ZeRO-1 |
| `--use_fsdp=True/False` | 3 | Enable/disable FSDP |
| `--fsdp_min_num_params=1e6` | 3 | FSDP wrapping threshold |
| `--compile=True/False` | All | torch.compile |
| `--dtype=bfloat16` | All | Mixed precision |

---

## 📖 Full Documentation

- **Phase 1**: `phase1_zero1/IMPLEMENTATION.md`
- **Phase 2**: `phase2_triton/IMPLEMENTATION.md`
- **Phase 3**: `phase3_fsdp/IMPLEMENTATION.md`
- **Summary**: `IMPLEMENTATION_SUMMARY.md`

---

## 🐛 Quick Troubleshooting

**"Triton not found"**  
→ `pip install triton` (Linux + CUDA only)

**FSDP OOM**  
→ Reduce `--fsdp_min_num_params=500000`

**ZeRO-1 checkpoint error**  
→ Use checkpoint from rank 0 only

**Slow FSDP**  
→ Tune `fsdp_min_num_params` (500K-5M range)

---

**Ready to test!** 🎉

