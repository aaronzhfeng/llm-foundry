# Quick Start Guide

## 📍 Location
```
/Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system/
```

## 🚀 Quick Test (1 minute)

### Test Different Architectures on Shakespeare Dataset

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# First, prepare the Shakespeare data (if not already done)
cd data/shakespeare
python prepare.py
cd ../..

# Test GPT-2 architecture
python train.py config/arch_gpt2.py --max_iters=100 --compile=False

# Test LLaMA architecture (RoPE + RMSNorm + SwiGLU)
python train.py config/arch_llama.py --max_iters=100 --compile=False

# Test team's model_v1 architecture
python train.py config/arch_team.py --max_iters=100 --compile=False
```

**Expected Output:**
- Detailed startup report showing **your architecture choices**
- Model architecture summary (normalization, position encoding, FFN type, etc.)
- Hardware specs and theoretical performance
- Per-iteration logs with architecture-aware MFU breakdown
- JSON log saved to `out/run_TIMESTAMP.json`

## 📊 What You'll See

### GPT-2 Architecture Output:
```
================================================================================
🚀 TRAINING INITIALIZATION
================================================================================

📊 MODEL ARCHITECTURE:
  Architecture Name:     12L-12H-768D-AbsPos-LN-NB-GELU-PostNorm
  Total parameters:      21,008,448 (21.01M)
  Trainable parameters:  21,008,448 (21.01M)
  Non-embedding params:  20,811,264 (20.81M)
  
  ├─ Layers:             12
  ├─ Hidden size:        768
  ├─ Attention heads:    12
  ├─ Sequence length:    1024
  ├─ Vocabulary size:    50304
  
  ├─ Normalization:      layernorm_nobias
  ├─ Activation:         gelu
  ├─ Position Encoding:  learned_absolute
  ├─ Attention Backend:  sdpa (FlashAttention via PyTorch)
  ├─ Norm Position:      post
  ├─ FFN Type:           standard (4.00x expansion)
  ├─ Bias:               No
  ├─ Weight Tying:       Yes
  └─ Dropout:            0.000
  ...
```

### LLaMA Architecture Output:
```
📊 MODEL ARCHITECTURE:
  Architecture Name:     12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm
  ...
  ├─ Normalization:      rmsnorm
  ├─ Activation:         swiglu (built-in SiLU)
  ├─ Position Encoding:  rope (θ=10000.0)
  ├─ Norm Position:      pre
  ├─ FFN Type:           swiglu (2.67x expansion)
  ├─ Weight Tying:       No
  ...

📈 THEORETICAL PERFORMANCE:
  FLOPs per token:       35.12 GFLOPs (adjusted for SwiGLU)
  Attention/FFN ratio:   0.52 (SwiGLU has more FFN compute)
  ...
```

### Per-Iteration Output:
```
────────────────────────────────────────────────────────────────────────────────
📍 Iter      5 │ Loss: 3.2145 │ Time: 234ms │ LR: 1.50e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 45.23% │ Achieved: 141.1 TF │ Peak: 312.0 TF
   Tokens/s: 21,943 │ FLOPs/token: 6.4 GF
💾 Memory: 2.34 GB alloc │ 3.12 GB peak │ 4.00 GB reserved
📊 Gradients: norm=1.2345 │ mean=-5.67e-06 │ std=2.34e-04
```

## 🧪 Testing Different Architectures

### 1. Compare All Architectures (Single GPU)
```bash
# Test GPT-2 (baseline)
python train.py config/arch_gpt2.py --max_iters=200 --dataset=shakespeare

# Test LLaMA (modern)
python train.py config/arch_llama.py --max_iters=200 --dataset=shakespeare

# Test team's architecture
python train.py config/arch_team.py --max_iters=200 --dataset=shakespeare

# Test custom hybrid
python train.py config/arch_custom.py --max_iters=200 --dataset=shakespeare

# Compare results in JSON logs
ls -lt out/run_*.json | head -4
```

### 2. Ablation Studies
```bash
# Start with GPT-2 baseline
python train.py config/arch_gpt2.py --max_iters=1000

# Test just adding RoPE
python train.py config/arch_gpt2.py --position_encoding=rope --max_iters=1000

# Test just adding RMSNorm
python train.py config/arch_gpt2.py --normalization=rmsnorm --max_iters=1000

# Test just adding Pre-norm
python train.py config/arch_gpt2.py --norm_position=pre --max_iters=1000

# Test all together (LLaMA-like)
python train.py config/arch_llama.py --max_iters=1000
```

### 3. Multi-GPU (4 GPUs) - When Available
```bash
# Standard DDP
torchrun --standalone --nproc_per_node=4 train.py config/arch_llama.py --max_iters=100

# With FSDP (Maximum memory efficiency)
torchrun --standalone --nproc_per_node=4 train.py config/arch_llama.py --max_iters=100 --use_fsdp=True
```

## 📝 Check the Results

### View JSON Log
```bash
# List all logs
ls -lh out/run_*.json

# View the latest log summary
python -c "import json; log = json.load(open(max([f for f in __import__('os').listdir('out') if f.endswith('.json')], key=lambda x: x))); print(json.dumps(log['summary'], indent=2))"
```

### Expected JSON Structure
```json
{
  "run_name": "run_20250103_143022",
  "startup_info": {
    "model": {...},
    "hardware": {...}
  },
  "training_iterations": [
    {
      "iter": 5,
      "loss": 3.2145,
      "time_ms": 234.5,
      "mfu": {
        "mfu_percent": 45.23,
        "achieved_tflops": 141.1,
        "tokens_per_sec": 21943,
        ...
      },
      "memory": {...},
      "gradients": {...}
    }
  ],
  "summary": {
    "avg_mfu": 45.23,
    "avg_time_ms": 234.5,
    ...
  }
}
```

## 🔧 Key Features to Test

### 1. Modular Architecture System
- [ ] Architecture name shows in startup report
- [ ] Different presets work (gpt2, llama, team)
- [ ] Can override components via CLI
- [ ] JSON log saves architecture config

### 2. Architecture-Aware MFU Calculation
- [ ] MFU accounts for SwiGLU overhead (LLaMA has higher FLOPs/token)
- [ ] GPT-2: ~28 GF/token, LLaMA: ~35 GF/token (due to SwiGLU)
- [ ] Verify `achieved_tflops` = `flops_per_token` × `tokens_per_sec`
- [ ] Hardware auto-detection (should show A100 or your GPU)

### 3. Memory Tracking
- [ ] Monitor `allocated_gb`, `peak_gb`, `reserved_gb`
- [ ] Verify memory usage is reasonable for your model size

### 4. Gradient Monitoring
- [ ] Check `global_norm`, `grad_mean`, `grad_std`
- [ ] Ensure gradients are healthy (not NaN, not exploding)

### 5. B200 Support (Future)
- [ ] When on B200, verify it auto-detects: `Hardware peak: 4500.0 TFLOPS (B200 bf16)`

## 🎨 Architecture Comparison Matrix

Quick reference for testing different architectures:

| Architecture | Norm | Position | FFN | Params | FLOPs/token | Expected MFU (A100) |
|--------------|------|----------|-----|--------|-------------|---------------------|
| **GPT-2** (baseline) | LayerNorm | Learned | Standard (4x) | 124M | ~28 GF | 30-35% |
| **LLaMA** | RMSNorm | RoPE | SwiGLU (8/3x) | 124M | ~35 GF | 28-33% |
| **Team** | RMSNorm | RoPE | SwiGLU (8/3x) | 124M | ~35 GF | 28-33% |
| **Hybrid** | LayerNorm | RoPE | Standard (4x) | 124M | ~29 GF | 30-35% |

**Note**: LLaMA has higher FLOPs/token due to SwiGLU (3 projections vs 2)

## 📚 Documentation

- **README.md** - Complete usage guide
- **SYSTEM_OVERVIEW.md** - Technical details with code references
- **EXAMPLE_OUTPUT.md** - Detailed output examples

## ⚡ Performance Expectations

### Shakespeare Dataset (Small Model)
- **Model**: 6 layers, 384 hidden, 6 heads (~21M params)
- **Expected MFU**: 40-50% on A100
- **Iteration time**: ~200-300ms
- **Tokens/s**: ~20,000-25,000

### GPT-2 124M (Full Model)
- **Model**: 12 layers, 768 hidden, 12 heads (~124M params)
- **Expected MFU**: 30-35% on A100
- **Iteration time**: ~4000-5000ms
- **Tokens/s**: ~3,000-4,000

## 🐛 Troubleshooting

### Data Not Found
```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system/data/shakespeare
python prepare.py
```

### Out of Memory
```bash
# Reduce batch size
python train.py config/train_shakespeare.py --batch_size=32

# Or use FSDP (multi-GPU only)
torchrun --standalone --nproc_per_node=4 train.py --use_fsdp=True
```

### Low MFU (<20%)
- Enable compilation: `--compile=True`
- Increase batch size: `--batch_size=64`
- Check data loading isn't bottleneck

## ✅ Validation Checklist

After running, verify:
- [ ] Startup report shows **architecture name** (e.g., "12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm")
- [ ] Startup shows all **component choices** (norm, position, FFN, etc.)
- [ ] Correct GPU detected (A100 or your hardware)
- [ ] MFU is calculated (not -100.0 or NaN)
- [ ] **FLOPs/token** varies by architecture (GPT-2: ~28 GF, LLaMA: ~35 GF)
- [ ] Memory stats are shown
- [ ] Gradients are logged (every 10 iterations by default)
- [ ] JSON log is created in `out/`
- [ ] JSON log contains **architecture config** in metadata
- [ ] Loss is decreasing over iterations
- [ ] No errors or warnings

### Expected Architecture-Specific Metrics

**GPT-2** (`arch_gpt2.py`):
- FLOPs/token: ~28-29 GF
- Attention/FFN ratio: ~0.67
- Architecture name: "12L-12H-768D-AbsPos-LN-NB-GELU-PostNorm"

**LLaMA** (`arch_llama.py`):
- FLOPs/token: ~35-36 GF (higher due to SwiGLU)
- Attention/FFN ratio: ~0.52 (SwiGLU has more FFN compute)
- Architecture name: "12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm"

## 🎯 Next Steps

1. ✅ Test on Shakespeare (quick validation)
2. Test on GPT-2 config (full model)
3. Test multi-GPU with DDP
4. Test FSDP for memory efficiency
5. When ready: Move to team repo (dsc180_a06)
6. Deploy on HGX B200 for production training

---

**Ready to test!** Run the Shakespeare quick test above to validate everything works. 🚀

