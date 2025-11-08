# Testing Guide: Complete Command Reference

All testing commands for the Enhanced GPT Training System in one place.

**Location:** `/Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system/`

---

## ⚡ Quick Start (30 seconds)

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Verify all imports work
python test_imports.py

# Quick 50-iteration test
python train.py config/arch_gpt2.py --max_iters=50 --compile=False
```

**Expected output:**
- ✅ All imports successful
- ✅ Architecture name in startup report
- ✅ MFU calculation with breakdown
- ✅ Memory and gradient stats

---

## 🧪 Test 1: Verify System Works (1 minute)

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Test all imports
python test_imports.py

# Should show:
# ✅ model_components imported successfully
# ✅ model_config imported successfully  
# ✅ model_builder imported successfully
# ✅ training_logger imported successfully
# ✅ model (legacy) imported successfully
```

---

## 🎨 Test 2: Test All Architectures (5 minutes)

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Test GPT-2 architecture
python train.py config/arch_gpt2.py --max_iters=100 --compile=False

# Test LLaMA architecture
python train.py config/arch_llama.py --max_iters=100 --compile=False

# Test team's model_v1 architecture
python train.py config/arch_team.py --max_iters=100 --compile=False

# Test custom architecture
python train.py config/arch_custom.py --max_iters=100 --compile=False
```

**Verify:**
- Architecture name differs for each
- GPT-2: ~28 GF/token
- LLaMA: ~35 GF/token (higher due to SwiGLU)
- JSON logs created in `out/`

---

## 🔬 Test 3: Ablation Studies (15 minutes)

Test impact of individual architectural components:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Baseline: GPT-2
python train.py config/arch_gpt2.py --max_iters=500 --compile=False

# Test 1: GPT-2 + RoPE (test position encoding)
python train.py config/arch_gpt2.py --position_encoding=rope --max_iters=500 --compile=False

# Test 2: GPT-2 + RMSNorm (test normalization)
python train.py config/arch_gpt2.py --normalization=rmsnorm --max_iters=500 --compile=False

# Test 3: GPT-2 + SwiGLU (test FFN type)
python train.py config/arch_gpt2.py --ffn_type=swiglu --max_iters=500 --compile=False

# Test 4: GPT-2 + Pre-norm (test norm position)
python train.py config/arch_gpt2.py --norm_position=pre --max_iters=500 --compile=False

# Test 5: Full LLaMA (all improvements)
python train.py config/arch_llama.py --max_iters=500 --compile=False

# Compare all runs
python compare_architectures.py --latest 6
```

---

## 📊 Test 4: Compare Architectures (10 minutes)

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Run all presets with same training config
python train.py config/arch_gpt2.py --max_iters=1000
python train.py config/arch_llama.py --max_iters=1000
python train.py config/arch_team.py --max_iters=1000

# Compare results
python compare_architectures.py --latest 3

# Expected output:
# ==================================================================================================
# ARCHITECTURE COMPARISON
# ==================================================================================================
# Rank   Run Name                  Architecture                  Val Loss    MFU %     Time/iter   
# --------------------------------------------------------------------------------------------------
# 1      run_20250103_143022      llama(rmsnorm/rope/swiglu)    3.1500      31.20     4512.3      
# 2      run_20250103_142512      gpt2(layernorm_nobias/...)    3.2400      32.50     4298.1      
# ...
```

---

## 🚀 Test 5: Multi-GPU (When Available)

### Standard DDP (4 GPUs)
```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# GPT-2 with DDP
torchrun --standalone --nproc_per_node=4 train.py config/arch_gpt2.py --max_iters=100

# LLaMA with DDP
torchrun --standalone --nproc_per_node=2 train.py config/arch_llama.py --max_iters=100
```

### ZeRO-1 (50% memory reduction)
```bash
torchrun --standalone --nproc_per_node=4 train.py config/arch_gpt2.py --use_zero1=True --max_iters=100
```

### FSDP (75-88% memory reduction)
```bash
torchrun --standalone --nproc_per_node=4 train.py config/arch_llama.py --use_fsdp=True --max_iters=100
```

**Verify:**
- Startup shows world_size = 4
- Parallelism shown (DDP, DDP+ZeRO-1, or FSDP)
- Memory reduced with ZeRO-1/FSDP
- Throughput ~4x higher

---

## 🎯 Test 6: Command-Line Overrides

Test that architecture can be overridden on command line:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Override single component
python train.py config/arch_gpt2.py --normalization=rmsnorm --max_iters=50

# Override multiple components
python train.py config/arch_gpt2.py \
  --normalization=rmsnorm \
  --position_encoding=rope \
  --ffn_type=swiglu \
  --norm_position=pre \
  --max_iters=50

# Override training params
python train.py config/arch_llama.py \
  --batch_size=16 \
  --learning_rate=3e-4 \
  --max_iters=50
```

---

## 📝 Test 7: JSON Logs

Verify logs contain architecture information:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Run a quick test
python train.py config/arch_llama.py --max_iters=50 --compile=False

# Check JSON log
latest_log=$(ls -t out/run_*.json | head -1)
cat $latest_log | python -m json.tool | grep -A 20 '"config"'

# Should show:
# "arch_preset": "llama",
# "normalization": "rmsnorm",
# "position_encoding": "rope",
# "ffn_type": "swiglu",
# ...
```

---

## 🔍 Test 8: Architecture-Aware MFU

Verify MFU calculation differs by architecture:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# GPT-2 (Standard FFN)
python train.py config/arch_gpt2.py --max_iters=50 --compile=False | grep "FLOPs per token"
# Expected: ~28-29 GFLOPs per token

# LLaMA (SwiGLU FFN)
python train.py config/arch_llama.py --max_iters=50 --compile=False | grep "FLOPs per token"
# Expected: ~35-36 GFLOPs per token (higher due to SwiGLU)

# Difference: ~25% more FLOPs for LLaMA
```

---

## 💾 Test 9: Memory Tracking

Verify memory statistics are logged:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Run with memory logging
python train.py config/arch_gpt2.py --max_iters=50 --compile=False | grep "Memory:"

# Should show:
# 💾 Memory: X.XX GB alloc │ Y.YY GB peak │ Z.ZZ GB reserved
```

---

## 📊 Test 10: Gradient Monitoring

Verify gradients are tracked:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Run with gradient logging (every 10 iters by default)
python train.py config/arch_gpt2.py --max_iters=50 --compile=False --gradient_log_interval=10 | grep "Gradients:"

# Should show:
# 📊 Gradients: norm=X.XXXX │ mean=X.XXe-XX │ std=X.XXe-XX
```

---

## 🌐 Test 11: B200 Hardware Support (Future)

When on HGX B200:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Run on B200
python train.py config/arch_llama.py --max_iters=50

# Verify startup shows:
# 🖥️  HARDWARE:
#   Device:                NVIDIA B200
#   ...
# 📈 THEORETICAL PERFORMANCE:
#   Hardware peak:         4500.0 TFLOPS (B200 bf16)
```

---

## 🧬 Test 12: Component Combinations

Test unusual/experimental combinations:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Combination 1: RoPE + LayerNorm + GELU (hybrid)
python train.py config/arch_custom.py \
  --arch_preset=custom \
  --normalization=layernorm_nobias \
  --position_encoding=rope \
  --ffn_type=standard \
  --activation=gelu \
  --max_iters=100

# Combination 2: Learned + RMSNorm + SwiGLU (reverse hybrid)
python train.py config/arch_custom.py \
  --arch_preset=custom \
  --normalization=rmsnorm \
  --position_encoding=learned_absolute \
  --ffn_type=swiglu \
  --max_iters=100

# Combination 3: All modern except position encoding
python train.py config/arch_custom.py \
  --arch_preset=custom \
  --normalization=rmsnorm \
  --position_encoding=learned_absolute \
  --ffn_type=swiglu \
  --norm_position=pre \
  --max_iters=100
```

---

## 📦 Test 13: Checkpoint Save/Resume

Test checkpointing works with modular architectures:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Train for a while
python train.py config/arch_llama.py --max_iters=100 --eval_interval=50

# Resume from checkpoint
python train.py config/arch_llama.py --init_from=resume --max_iters=200

# Verify:
# - Architecture loaded from checkpoint
# - Training continues from iteration 100
# - Loss continues to decrease
```

---

## 🔄 Test 14: Legacy Compatibility

Test that legacy GPT model still works:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Use legacy model
python train.py config/train_shakespeare.py --max_iters=50 --compile=False

# Should work but without modular architecture features
```

---

## 📈 Test 15: Full Comparison Workflow

Complete workflow to compare architectures systematically:

```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system

# Day 1: Baseline
python train.py config/arch_gpt2.py --max_iters=2000 --compile=False

# Day 2: Component ablations
python train.py config/arch_gpt2.py --position_encoding=rope --max_iters=2000 --compile=False
python train.py config/arch_gpt2.py --normalization=rmsnorm --max_iters=2000 --compile=False
python train.py config/arch_gpt2.py --ffn_type=swiglu --max_iters=2000 --compile=False

# Day 3: Full LLaMA
python train.py config/arch_llama.py --max_iters=2000 --compile=False

# Day 4: Compare all
python compare_architectures.py --latest 5

# Day 5: Pick best and run longer
python train.py config/arch_llama.py --max_iters=10000
```

---

## 🎯 Validation Checklist

After running tests, verify:

### ✅ System Functionality
- [ ] `test_imports.py` shows all imports successful
- [ ] Training runs without errors
- [ ] JSON logs created in `out/`
- [ ] Checkpoints saved in `out/ckpt.pt`

### ✅ Modular Architecture
- [ ] Architecture name appears in startup (e.g., "12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm")
- [ ] Component choices listed (normalization, position encoding, FFN type, etc.)
- [ ] Can run all 4 presets (gpt2, llama, team, custom)
- [ ] Can override components via CLI

### ✅ Architecture-Aware MFU
- [ ] FLOPs/token differs by architecture:
  - GPT-2: ~28 GF/token
  - LLaMA: ~35 GF/token
  - Hybrid: ~29 GF/token
- [ ] Attention/FFN ratio differs:
  - GPT-2: ~0.67 (less FFN compute)
  - LLaMA: ~0.52 (SwiGLU has more FFN compute)

### ✅ Monitoring
- [ ] MFU breakdown shown (achieved TF, peak TF, tokens/s)
- [ ] Memory stats shown (allocated, peak, reserved)
- [ ] Gradient stats shown (every N iterations)
- [ ] Hardware auto-detected correctly

### ✅ JSON Logs
- [ ] Architecture config saved in logs
- [ ] MFU breakdown in each iteration
- [ ] Memory stats in each iteration
- [ ] Gradient stats periodically
- [ ] Summary statistics at end

### ✅ Comparison Tool
- [ ] `compare_architectures.py` runs successfully
- [ ] Shows architecture ranking by loss
- [ ] Shows MFU comparison
- [ ] Identifies best architecture

---

## 🚦 Expected Results by Architecture

### GPT-2 (arch_gpt2.py)
```
Architecture Name:     12L-12H-768D-AbsPos-LN-NB-GELU-PostNorm
FLOPs per token:       ~28.45 GFLOPs
Attention/FFN ratio:   ~0.67
Expected MFU (A100):   30-35%
```

### LLaMA (arch_llama.py)
```
Architecture Name:     12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm
FLOPs per token:       ~35.12 GFLOPs (25% higher due to SwiGLU)
Attention/FFN ratio:   ~0.52 (more FFN compute)
Expected MFU (A100):   28-33%
```

### Team (arch_team.py)
```
Architecture Name:     12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm
Same as LLaMA
```

### Hybrid (arch_custom.py - depends on your config)
```
Architecture Name:     Varies based on components chosen
FLOPs per token:       Varies (28-36 GF range)
```

---

## 📋 Quick Command Reference

```bash
# In /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system/

# ===== Basic Tests =====
python test_imports.py                                    # Verify imports
python train.py config/arch_gpt2.py --max_iters=50       # Quick test

# ===== Architecture Tests =====
python train.py config/arch_gpt2.py --max_iters=100      # GPT-2
python train.py config/arch_llama.py --max_iters=100     # LLaMA
python train.py config/arch_team.py --max_iters=100      # Team
python train.py config/arch_custom.py --max_iters=100    # Custom

# ===== Override Tests =====
python train.py config/arch_gpt2.py --position_encoding=rope            # Single override
python train.py config/arch_gpt2.py --normalization=rmsnorm --ffn_type=swiglu  # Multiple

# ===== Comparison =====
python compare_architectures.py                          # All runs
python compare_architectures.py --latest 5               # Latest 5

# ===== Multi-GPU =====
torchrun --standalone --nproc_per_node=4 train.py config/arch_llama.py              # DDP
torchrun --standalone --nproc_per_node=4 train.py config/arch_gpt2.py --use_zero1=True   # ZeRO-1
torchrun --standalone --nproc_per_node=4 train.py config/arch_llama.py --use_fsdp=True   # FSDP

# ===== Utilities =====
python -c "from model_config import list_presets; list_presets()"    # List presets
ls -lh out/run_*.json                                    # View logs
cat out/run_*.json | python -m json.tool | less          # Pretty-print log
```

---

## 🐛 Troubleshooting Tests

### Test Imports Fail
```bash
# Check you're in correct directory
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system
pwd

# Verify files exist
ls model_*.py
```

### Data Not Found
```bash
# Prepare Shakespeare data
cd data/shakespeare
python prepare.py
cd ../..

# Try again
python train.py config/arch_gpt2.py --max_iters=50 --dataset=shakespeare
```

### Architecture Name Not Showing
```bash
# Make sure using new config format (arch_*.py)
python train.py config/arch_gpt2.py  # ✓ Shows architecture
python train.py config/train_gpt2.py # ✗ Legacy format (still works but no arch name)
```

### FLOPs/Token Seems Wrong
```bash
# This is expected! Different architectures have different FLOPs:
# - GPT-2 (standard FFN): ~28 GF/token
# - LLaMA (SwiGLU FFN): ~35 GF/token (25% more)
# - The system is correctly accounting for architectural differences
```

---

## 📊 Performance Benchmarks

Expected performance on A100 (single GPU):

### Small Model (Shakespeare config)
- **Model**: 6L-6H-384D (~21M params)
- **GPT-2**: 40-50% MFU, ~200ms/iter, ~20k tokens/s
- **LLaMA**: 38-48% MFU, ~220ms/iter, ~18k tokens/s

### GPT-2 124M
- **GPT-2**: 30-35% MFU, ~4000ms/iter, ~3.5k tokens/s, 28 GF/token
- **LLaMA**: 28-33% MFU, ~4500ms/iter, ~3.0k tokens/s, 35 GF/token

**Note:** LLaMA has slightly lower MFU % but similar absolute TFLOPS because it does more compute per token!

---

## ✅ Success Criteria

System is working correctly if:

1. **All imports pass** (`test_imports.py`)
2. **Architecture name appears** in startup report
3. **FLOPs/token varies** by architecture:
   - GPT-2: ~28 GF
   - LLaMA: ~35 GF
4. **Can run all 4 presets** without errors
5. **Can override components** via CLI
6. **JSON logs contain architecture config**
7. **Comparison tool works** and ranks architectures
8. **Loss decreases** over iterations
9. **No NaN** in gradients or loss
10. **MFU** is in reasonable range (20-50% on A100)

---

## 🎯 Next Steps After Testing

### If Tests Pass:
1. Run longer training (5k-10k iterations)
2. Compare all architectures systematically
3. Identify best architecture for your task
4. Move to team repo (`dsc180_a06`)
5. Deploy on HGX B200 for production

### If Tests Fail:
1. Check error messages
2. Verify you're in correct directory
3. Check data is prepared (`data/shakespeare/`)
4. Verify PyTorch version (needs 2.0+)
5. Check CUDA is available

---

## 📞 Help & Documentation

| Question | Document | Section |
|----------|----------|---------|
| How do I start? | README.md | Quick Start (line 111) |
| What are all options? | README.md | Configuration (line 162) |
| Technical details? | SYSTEM_OVERVIEW.md | All sections |
| Example outputs? | EXAMPLE_OUTPUT.md | (in docs/) |
| Quick commands? | This file | All sections above |

---

**Ready to test!** Start with Test 1 (verify imports) and work your way through. 🚀

**Recommended first test:**
```bash
cd /Users/aaronfeng/Repo/Hao/llm_TII/enhanced_training_system
python test_imports.py && python train.py config/arch_gpt2.py --max_iters=50 --compile=False
```

