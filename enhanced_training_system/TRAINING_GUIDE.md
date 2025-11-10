# Complete Training Guide: LLaMA 1.36B vs GPT-2 1.36B

## Overview

This guide covers the complete workflow from data preparation to training and evaluation of two 1.36B parameter models on SlimPajama-6B dataset.

**Models:**
- **LLaMA 1.36B**: Modern architecture (RoPE + RMSNorm + SwiGLU + Pre-norm)
- **GPT-2 1.36B**: Traditional architecture (Learned Pos + LayerNorm + GELU + Post-norm)

**Hardware:** 2-4× H20 GPUs (recommended: 4 GPUs for stable training)

**Dataset:** SlimPajama-6B (~6B tokens, ~6GB tokenized)

---

## 🖥️ Hardware Assessment: H20 GPUs

### NVIDIA H20 Specifications

- **Memory**: ~96GB HBM3
- **Performance**: ~296 TFLOPS (BF16) - Export-controlled variant
- **TDP**: ~350W
- **Special**: China-specific version (export control compliant)

### Memory Requirements for 1.36B Models

| Configuration | Memory/GPU | Recommended GPUs |
|---------------|------------|------------------|
| **2× H20 (minimal)** | ~45-50 GB/GPU | Tight but workable with batch_size=4 |
| **4× H20 (recommended)** | ~25-30 GB/GPU | Comfortable with batch_size=8 |
| **8× H20 (optimal)** | ~15-20 GB/GPU | Large batches, use FSDP |

### **Recommendation for Your Case:**

**4× H20 is strongly recommended** for:
- ✅ Comfortable memory usage (~25-30 GB per GPU)
- ✅ Stable training (larger effective batch size)
- ✅ Better distributed efficiency
- ✅ Faster training time

**2× H20 is possible but:**
- ⚠️ Tight memory (~45-50 GB per GPU, close to limit)
- ⚠️ Need smaller batch_size (4 instead of 8)
- ⚠️ Slower training (fewer GPUs)
- ⚠️ Higher risk of OOM (out of memory)

**If you only have 2× H20:**
- Set `batch_size = 4` and `gradient_accumulation_steps = 32`
- Enable `use_zero1 = True` to save ~10GB per GPU
- Monitor memory carefully during first few iterations

---

## 📋 Complete Workflow

### Phase 1: Pre-Training Setup (Local Machine with HuggingFace Access)

**Time:** 30-60 minutes  
**Requirements:** Internet, HuggingFace access

#### Step 1.1: Install Dependencies

```bash
cd /path/to/enhanced_training_system

# Install required packages
pip install torch transformers datasets tiktoken numpy tqdm

# Verify installation
python test_imports.py
```

#### Step 1.2: Download LLaMA-2 Tokenizer

```bash
# Download and save tokenizer locally
python << 'EOF'
from transformers import LlamaTokenizer

print("Downloading LLaMA-2 tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("./llama2_tokenizer")
print("✓ Tokenizer saved to ./llama2_tokenizer/")
print(f"  Vocab size: {tokenizer.vocab_size}")
EOF

# Verify files created
ls -lh llama2_tokenizer/
# Should show: tokenizer.model, tokenizer_config.json, special_tokens_map.json
```

**Note:** You may need to:
1. Accept LLaMA-2 license on HuggingFace
2. Login: `huggingface-cli login`

#### Step 1.3: Prepare SlimPajama-6B with LLaMA-2 Tokenizer

```bash
cd data/slimpajama_6b_llama

# Run preparation (will download dataset + tokenize)
python prepare.py

# Expected output:
#   ✓ Dataset loaded
#   ✓ Tokenization complete: ~6B tokens
#   ✓ train.bin, val.bin, meta.pkl created

# Verify files
ls -lh
# Should show: train.bin (~6GB), val.bin (~30MB), meta.pkl (few KB)

cd ../..
```

#### Step 1.4: Prepare SlimPajama-6B with GPT-2 Tokenizer

```bash
cd data/slimpajama_6b_gpt2

# Run preparation
python prepare.py

# Expected output similar to above
# Verify files
ls -lh

cd ../..
```

#### Step 1.5: Create Upload Package (Optional - if transferring to server)

```bash
# Create tarball for upload (if needed)
cd ..
tar -czf enhanced_training_system_ready.tar.gz \
  enhanced_training_system/train.py \
  enhanced_training_system/model*.py \
  enhanced_training_system/config/ \
  enhanced_training_system/llama2_tokenizer/ \
  enhanced_training_system/data/slimpajama_6b_llama/ \
  enhanced_training_system/data/slimpajama_6b_gpt2/

# Size should be ~15-20 GB (mostly data)
ls -lh enhanced_training_system_ready.tar.gz
```

---

### Phase 2: Training on Server (SSH with H20 GPUs)

**Time:** 4-8 hours (depending on GPU count)  
**Hardware:** 2-4× H20

#### Step 2.1: Transfer Files (if needed)

```bash
# On local machine
scp enhanced_training_system_ready.tar.gz user@server:/path/to/workspace/

# On server
cd /path/to/workspace
tar -xzf enhanced_training_system_ready.tar.gz
cd enhanced_training_system
```

#### Step 2.2: Environment Setup

```bash
# Check CUDA and GPUs
nvidia-smi

# Should show:
#   - 2 or 4× H20 GPUs
#   - CUDA 12.x
#   - ~96GB memory per GPU

# Verify Python environment
python --version  # Should be 3.8+
pip list | grep -E "torch|transformers"
```

#### Step 2.3: Quick Smoke Test

```bash
# Test that everything loads (10 iterations, no actual training)
python train.py config/full_llama_1.36b.py \
  --max_iters=10 \
  --eval_only=False \
  --compile=False

# Should see:
#   ✓ Model builds successfully
#   ✓ Data loads from slimpajama_6b_llama
#   ✓ Vocab size: 32000
#   ✓ 10 training iterations complete
#   ✓ No errors
```

---

### Phase 3: Training (2× H20 Configuration)

#### Configuration for 2× H20 (Tight Memory)

```bash
# Train LLaMA 1.36B on 2× H20
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama_1.36b.py \
  --batch_size=4 \
  --gradient_accumulation_steps=32 \
  --use_zero1=True \
  --max_iters=2000

# Effective batch: 4 × 32 × 2 = 256 samples = 524,288 tokens/iter
# Expected time: ~8-12 hours
# Expected loss: ~4.0-4.5 (undertrained on 6B tokens)
```

```bash
# Train GPT-2 1.36B on 2× H20
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_gpt2_1.36b.py \
  --batch_size=4 \
  --gradient_accumulation_steps=32 \
  --use_zero1=True \
  --max_iters=2000

# Expected time: ~6-10 hours (faster per token than LLaMA)
# Expected loss: ~4.2-4.7 (GPT-2 slightly worse)
```

---

### Phase 3 Alternative: Training (4× H20 Configuration - Recommended)

#### Configuration for 4× H20 (Comfortable Memory)

```bash
# Train LLaMA 1.36B on 4× H20
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_llama_1.36b.py \
  --max_iters=2000

# Uses default: batch_size=8, gradient_accumulation_steps=16
# Effective batch: 8 × 16 × 4 = 512 samples = 1,048,576 tokens/iter
# Expected time: ~4-6 hours
# Expected loss: ~4.0-4.5
```

```bash
# Train GPT-2 1.36B on 4× H20
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_gpt2_1.36b.py \
  --max_iters=2000

# Expected time: ~3-5 hours (faster than LLaMA)
# Expected loss: ~4.2-4.7
```

---

### Phase 4: Monitoring Training

#### Real-Time Monitoring

```bash
# In another terminal, monitor the log file
tail -f out-llama-1.36b/run_*.json

# Or watch the console output
# Key metrics to watch:
#   - Loss (should decrease steadily)
#   - MFU (should be 35-45%)
#   - Tokens/sec (should be consistent)
#   - Memory (should not hit 96GB limit)
```

#### Check Training Progress

```bash
# View latest iteration
python << 'EOF'
import json
import glob

# Find latest log
logs = glob.glob("out-llama-1.36b/run_*.json")
if logs:
    with open(sorted(logs)[-1], 'r') as f:
        data = json.load(f)
    
    iters = data['training_iterations']
    if iters:
        latest = iters[-1]
        print(f"Iteration: {latest['iter']}")
        print(f"Loss: {latest['loss']:.4f}")
        print(f"Tokens/sec: {latest.get('tokens_per_sec', 'N/A')}")
        print(f"MFU: {latest.get('mfu', 'N/A')}")
EOF
```

#### Emergency: Stop Training

```bash
# If you need to stop training
# Press Ctrl+C (will save checkpoint at next eval_interval)

# Or force kill (not recommended, may corrupt checkpoint)
pkill -f "train.py"
```

---

### Phase 5: Evaluation & Comparison

#### Evaluate Final Models

```bash
# Evaluate LLaMA 1.36B
python train.py config/full_llama_1.36b.py \
  --init_from=resume \
  --eval_only=True

# Evaluate GPT-2 1.36B
python train.py config/full_gpt2_1.36b.py \
  --init_from=resume \
  --eval_only=True
```

#### Compare Training Results

```bash
# View final losses
echo "=== LLaMA 1.36B ==="
python << 'EOF'
import json
import glob
logs = glob.glob("out-llama-1.36b/run_*.json")
if logs:
    with open(sorted(logs)[-1]) as f:
        data = json.load(f)
    final = data['training_iterations'][-1]
    print(f"Final Loss: {final['loss']:.4f}")
    print(f"Iterations: {final['iter']}")
    
    # Calculate average MFU
    mfus = [x['mfu'] for x in data['training_iterations'] if x.get('mfu')]
    if mfus:
        print(f"Average MFU: {sum(mfus)/len(mfus):.2f}%")
EOF

echo ""
echo "=== GPT-2 1.36B ==="
python << 'EOF'
import json
import glob
logs = glob.glob("out-gpt2-1.36b/run_*.json")
if logs:
    with open(sorted(logs)[-1]) as f:
        data = json.load(f)
    final = data['training_iterations'][-1]
    print(f"Final Loss: {final['loss']:.4f}")
    print(f"Iterations: {final['iter']}")
    
    mfus = [x['mfu'] for x in data['training_iterations'] if x.get('mfu')]
    if mfus:
        print(f"Average MFU: {sum(mfus)/len(mfus):.2f}%")
EOF
```

#### Generate Comparison Plot (Optional)

```bash
# Create simple comparison script
python << 'EOF'
import json
import glob
import matplotlib.pyplot as plt

# Load logs
llama_logs = glob.glob("out-llama-1.36b/run_*.json")
gpt2_logs = glob.glob("out-gpt2-1.36b/run_*.json")

if llama_logs and gpt2_logs:
    with open(sorted(llama_logs)[-1]) as f:
        llama_data = json.load(f)
    with open(sorted(gpt2_logs)[-1]) as f:
        gpt2_data = json.load(f)
    
    # Extract losses
    llama_iters = [x['iter'] for x in llama_data['training_iterations']]
    llama_losses = [x['loss'] for x in llama_data['training_iterations']]
    gpt2_iters = [x['iter'] for x in gpt2_data['training_iterations']]
    gpt2_losses = [x['loss'] for x in gpt2_data['training_iterations']]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(llama_iters, llama_losses, label='LLaMA 1.36B', linewidth=2)
    plt.plot(gpt2_iters, gpt2_losses, label='GPT-2 1.36B', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('LLaMA vs GPT-2 Training Comparison (1.36B params)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved to comparison_plot.png")
    
    # Print summary
    print(f"\nLLaMA Final Loss: {llama_losses[-1]:.4f}")
    print(f"GPT-2 Final Loss: {gpt2_losses[-1]:.4f}")
    print(f"Difference: {(gpt2_losses[-1] - llama_losses[-1]):.4f}")
else:
    print("Logs not found!")
EOF
```

---

## 🎯 Quick Command Summary

### Complete Training Pipeline (4× H20)

```bash
# =============================================================================
# PHASE 1: Data Preparation (Do once, before training)
# =============================================================================

# Prepare LLaMA dataset
cd data/slimpajama_6b_llama && python prepare.py && cd ../..

# Prepare GPT-2 dataset
cd data/slimpajama_6b_gpt2 && python prepare.py && cd ../..

# =============================================================================
# PHASE 2: Training (Main experiments)
# =============================================================================

# Train LLaMA 1.36B
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py

# Train GPT-2 1.36B (after LLaMA completes)
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_1.36b.py

# =============================================================================
# PHASE 3: Evaluation
# =============================================================================

# Evaluate both models
python train.py config/full_llama_1.36b.py --init_from=resume --eval_only=True
python train.py config/full_gpt2_1.36b.py --init_from=resume --eval_only=True
```

---

## 📊 Expected Results

### Training Metrics (4× H20, 2000 iterations on 6B tokens)

| Metric | LLaMA 1.36B | GPT-2 1.36B |
|--------|-------------|-------------|
| **Final Loss** | ~4.0-4.5 | ~4.2-4.7 |
| **Tokens/sec** | ~50,000-60,000 | ~60,000-75,000 |
| **MFU** | 35-40% | 35-40% |
| **Memory/GPU** | ~25-28 GB | ~23-25 GB |
| **Training Time** | ~4-6 hours | ~3-5 hours |
| **Tokens Trained** | ~1-2B | ~1-2B |

**Note:** Loss ~4.0-4.5 is expected because 6B tokens is only ~7% of optimal training (85B tokens). This is a **testing run** to verify:
- ✅ Model architecture works
- ✅ Data loading works
- ✅ Training is stable
- ✅ No OOM errors
- ✅ Reasonable MFU achieved

**For loss ~2.4 (optimal):** Need to train on 85B tokens (SlimPajama-627B dataset).

---

## 🔧 Configuration Adjustments

### For 2× H20 (Memory Constrained)

Edit both config files:

```python
# Reduce memory usage
batch_size = 4                      # Was: 8
gradient_accumulation_steps = 32   # Was: 16
use_zero1 = True                   # Was: False

# This gives effective batch = 4 × 32 × 2 = 256 samples
# Slightly smaller than recommended, but works
```

### For Faster Testing (Reduce iterations)

```bash
# Quick test (100 iterations, ~5 minutes)
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_llama_1.36b.py \
  --max_iters=100 \
  --eval_interval=50

# Medium test (500 iterations, ~30 minutes)
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_llama_1.36b.py \
  --max_iters=500 \
  --eval_interval=100
```

---

## 🚨 Troubleshooting

### Problem: OOM (Out of Memory)

```bash
# Solution 1: Reduce batch size
--batch_size=2 --gradient_accumulation_steps=64

# Solution 2: Enable ZeRO-1
--use_zero1=True

# Solution 3: Reduce sequence length (if testing only)
--block_size=1024  # Instead of 2048

# Solution 4: Use gradient checkpointing (not implemented, would need code change)
```

### Problem: Dataset not found

```
FileNotFoundError: data/slimpajama_6b_llama/train.bin
```

**Solution:**
```bash
# Verify dataset exists
ls -lh data/slimpajama_6b_llama/

# If missing, run prepare.py
cd data/slimpajama_6b_llama
python prepare.py
```

### Problem: Tokenizer not found (LLaMA)

```
OSError: Can't load tokenizer for 'meta-llama/Llama-2-7b-hf'
```

**Solution:**
```bash
# If HuggingFace is blocked, use local tokenizer
# Edit prepare.py line 19:
# FROM: tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# TO:   tokenizer = LlamaTokenizer.from_pretrained("../../llama2_tokenizer")
```

### Problem: Slow training (<10k tokens/sec)

**Check:**
1. Is `compile=True`? (should be for speed)
2. Is MFU <20%? (something is wrong)
3. Is GPU utilization <50%? (check `nvidia-smi`)

**Solutions:**
- Enable torch.compile: `--compile=True`
- Check for CPU bottlenecks (data loading)
- Increase `num_proc` in prepare.py if data loading is slow

### Problem: Loss not decreasing

**Check:**
1. Learning rate too low? (should be ~3e-4)
2. Gradient clipping too aggressive? (should be 1.0)
3. Warmup too short? (should be ~2000 iterations)

**Solutions:**
- Try `--learning_rate=4e-4` (slightly higher)
- Check gradient norms in logs (should be 0.5-2.0)
- Increase warmup: `--warmup_iters=3000`

---

## 📈 Expected Training Curve

### Loss Trajectory (6B tokens)

```
Iteration    Tokens    Loss (LLaMA)  Loss (GPT-2)
=========    ======    ============  ============
0            0         10.5          10.7         (random init)
100          100M      7.2           7.5          (rapid improvement)
500          500M      5.5           5.8          (learning patterns)
1000         1B        4.8           5.2          (converging)
1500         1.5B      4.4           4.8          (slowing down)
2000         2B        4.2           4.6          (approaching plateau)
```

**Expected final result:**
- LLaMA: ~4.0-4.5 (better)
- GPT-2: ~4.2-4.7 (competitive but slightly worse)

---

## 🎓 Interpretation of Results

### What This Test Shows

✅ **Architecture works** - Models train without errors  
✅ **Data pipeline works** - Tokenization and loading are correct  
✅ **Training is stable** - Loss decreases, gradients are healthy  
✅ **LLaMA outperforms** - ~5-10% better loss at same token count  
✅ **GPT-2 is faster** - ~15-20% more tokens/sec  

### What This Test DOESN'T Show

❌ **Optimal performance** - Need 85B tokens for loss ~2.4  
❌ **Final model quality** - 6B tokens is undertrained  
❌ **Scaling behavior** - Need longer training to see convergence  

### Next Steps After This Test

1. **If results look good:** Proceed to full training (627B dataset)
2. **If issues found:** Debug and fix before scaling up
3. **If loss trajectory is good:** Can extrapolate to full training

---

## 💾 Checkpoint Management

### Checkpoint Files

Training automatically saves checkpoints:

```
out-llama-1.36b/
├── ckpt.pt              # Latest checkpoint
├── run_*.json           # Training logs
└── ckpt_best.pt         # Best validation loss (if enabled)
```

### Resume Training

```bash
# If training was interrupted
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_llama_1.36b.py \
  --init_from=resume

# Will automatically load from out-llama-1.36b/ckpt.pt
```

### Extract Checkpoint Info

```bash
python << 'EOF'
import torch

ckpt = torch.load("out-llama-1.36b/ckpt.pt", map_location='cpu')
print(f"Iteration: {ckpt['iter_num']}")
print(f"Best val loss: {ckpt['best_val_loss']:.4f}")
print(f"Model args: {ckpt['model_args']}")
EOF
```

---

## 📋 Pre-Training Checklist

Before starting training, verify:

```bash
# Run this checklist script
cat > check_ready.sh << 'SCRIPT'
#!/bin/bash
echo "Pre-Training Checklist"
echo "====================="
echo ""

# Check 1: Data files
echo "[1/6] Checking data files..."
if [ -f data/slimpajama_6b_llama/train.bin ] && [ -f data/slimpajama_6b_llama/meta.pkl ]; then
    echo "  ✓ LLaMA dataset ready"
else
    echo "  ❌ LLaMA dataset missing - run data/slimpajama_6b_llama/prepare.py"
fi

if [ -f data/slimpajama_6b_gpt2/train.bin ] && [ -f data/slimpajama_6b_gpt2/meta.pkl ]; then
    echo "  ✓ GPT-2 dataset ready"
else
    echo "  ❌ GPT-2 dataset missing - run data/slimpajama_6b_gpt2/prepare.py"
fi

# Check 2: Config files
echo ""
echo "[2/6] Checking config files..."
if [ -f config/full_llama_1.36b.py ]; then
    echo "  ✓ LLaMA config exists"
else
    echo "  ❌ LLaMA config missing"
fi

if [ -f config/full_gpt2_1.36b.py ]; then
    echo "  ✓ GPT-2 config exists"
else
    echo "  ❌ GPT-2 config missing"
fi

# Check 3: GPUs
echo ""
echo "[3/6] Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4

# Check 4: Python packages
echo ""
echo "[4/6] Checking Python packages..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "  ❌ PyTorch missing"
python -c "import transformers; print(f'  ✓ Transformers installed')" 2>/dev/null || echo "  ❌ Transformers missing"
python -c "import tiktoken; print(f'  ✓ tiktoken installed')" 2>/dev/null || echo "  ❌ tiktoken missing"

# Check 5: Disk space
echo ""
echo "[5/6] Checking disk space..."
df -h . | tail -1

# Check 6: Test imports
echo ""
echo "[6/6] Testing imports..."
python test_imports.py 2>/dev/null && echo "  ✓ All imports work" || echo "  ⚠️  Some imports failed"

echo ""
echo "====================="
echo "Checklist Complete!"
SCRIPT

chmod +x check_ready.sh
./check_ready.sh
```

---

## 🚀 One-Command Training (After Data Prep)

### For 4× H20 GPUs:

```bash
# Train both models sequentially
nohup bash << 'SCRIPT' > training.log 2>&1 &
#!/bin/bash
set -e

echo "Starting LLaMA 1.36B training..."
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py

echo ""
echo "LLaMA training complete! Starting GPT-2 1.36B..."
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_1.36b.py

echo ""
echo "Both trainings complete!"
SCRIPT

# Monitor progress
tail -f training.log
```

### For 2× H20 GPUs:

```bash
# Train with adjusted batch sizes
nohup bash << 'SCRIPT' > training.log 2>&1 &
#!/bin/bash
set -e

echo "Starting LLaMA 1.36B training (2 GPUs)..."
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama_1.36b.py \
  --batch_size=4 \
  --gradient_accumulation_steps=32 \
  --use_zero1=True

echo ""
echo "LLaMA training complete! Starting GPT-2 1.36B..."
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_gpt2_1.36b.py \
  --batch_size=4 \
  --gradient_accumulation_steps=32 \
  --use_zero1=True

echo ""
echo "Both trainings complete!"
SCRIPT

tail -f training.log
```

---

## 🎯 Hardware Recommendation: 2 vs 4 H20 GPUs

### **2× H20 (Minimal Configuration)**

**Pros:**
- ✅ Can run the experiment
- ✅ Lower cost

**Cons:**
- ❌ Tight memory (~45-50 GB per GPU)
- ❌ Risk of OOM errors
- ❌ Smaller effective batch (256 vs 512 samples)
- ❌ 2× slower than 4 GPUs
- ❌ Less stable training (smaller batches)

**Estimated time:** 8-12 hours per model

### **4× H20 (Recommended Configuration)**

**Pros:**
- ✅ Comfortable memory (~25-30 GB per GPU)
- ✅ Larger effective batch (512 samples)
- ✅ 2× faster training
- ✅ More stable (better gradient averaging)
- ✅ Room for experimentation

**Cons:**
- ❌ Higher cost (2× more GPU hours)

**Estimated time:** 4-6 hours per model

### **My Strong Recommendation: Use 4× H20**

**Why:**
1. **Memory safety**: 96GB per GPU → ~25GB used = 75% headroom
2. **Training stability**: Larger effective batch = better gradient estimates
3. **Time efficiency**: 4-6 hours vs 8-12 hours = worth the cost
4. **Debugging**: If issues arise, easier to debug with comfortable memory
5. **Flexibility**: Can experiment with larger batches if needed

**Budget:** If you can only afford 2 GPUs, it will work but expect:
- Slower training (2×)
- Need to carefully tune batch sizes
- Monitor memory closely
- May need ZeRO-1 enabled

---

## 📝 Post-Training Report Template

After training completes, generate a report:

```bash
python << 'EOF'
import json
import glob

print("="*80)
print("TRAINING COMPARISON REPORT")
print("="*80)
print()

# LLaMA results
llama_logs = sorted(glob.glob("out-llama-1.36b/run_*.json"))
if llama_logs:
    with open(llama_logs[-1]) as f:
        llama = json.load(f)
    print("LLaMA 1.36B:")
    print(f"  Config: {llama['config']['dataset']}")
    print(f"  Iterations: {llama['training_iterations'][-1]['iter']}")
    print(f"  Final Loss: {llama['training_iterations'][-1]['loss']:.4f}")
    mfus = [x.get('mfu') for x in llama['training_iterations'] if x.get('mfu')]
    if mfus:
        print(f"  Avg MFU: {sum(mfus)/len(mfus):.2f}%")
    print(f"  Time: {llama.get('total_time', 'N/A')}")

print()

# GPT-2 results
gpt2_logs = sorted(glob.glob("out-gpt2-1.36b/run_*.json"))
if gpt2_logs:
    with open(gpt2_logs[-1]) as f:
        gpt2 = json.load(f)
    print("GPT-2 1.36B:")
    print(f"  Config: {gpt2['config']['dataset']}")
    print(f"  Iterations: {gpt2['training_iterations'][-1]['iter']}")
    print(f"  Final Loss: {gpt2['training_iterations'][-1]['loss']:.4f}")
    mfus = [x.get('mfu') for x in gpt2['training_iterations'] if x.get('mfu')]
    if mfus:
        print(f"  Avg MFU: {sum(mfus)/len(mfus):.2f}%")
    print(f"  Time: {gpt2.get('total_time', 'N/A')}")

if llama_logs and gpt2_logs:
    llama_loss = llama['training_iterations'][-1]['loss']
    gpt2_loss = gpt2['training_iterations'][-1]['loss']
    diff = ((gpt2_loss - llama_loss) / llama_loss) * 100
    
    print()
    print("Comparison:")
    print(f"  LLaMA advantage: {diff:+.2f}%")
    if diff > 0:
        print(f"  LLaMA achieved {diff:.1f}% lower loss")
    else:
        print(f"  GPT-2 achieved {-diff:.1f}% lower loss (unexpected!)")

print()
print("="*80)
EOF
```

---

## 🎉 Success Criteria

Your training is successful if:

- ✅ Loss decreases from ~10 to ~4.0-4.5
- ✅ MFU is 30-45% (reasonable hardware utilization)
- ✅ No OOM errors throughout training
- ✅ Gradients remain healthy (norm < 5.0)
- ✅ LLaMA outperforms GPT-2 by ~5-10%
- ✅ Training completes without crashes

**If all criteria met:** You're ready to scale to SlimPajama-627B! 🚀

---

## 📞 Quick Reference Commands

```bash
# Data prep
cd data/slimpajama_6b_llama && python prepare.py && cd ../..
cd data/slimpajama_6b_gpt2 && python prepare.py && cd ../..

# Train (4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_1.36b.py

# Train (2 GPUs)
torchrun --standalone --nproc_per_node=2 train.py config/full_llama_1.36b.py --batch_size=4 --gradient_accumulation_steps=32 --use_zero1=True

# Monitor
tail -f out-llama-1.36b/run_*.json
nvidia-smi -l 5

# Evaluate
python train.py config/full_llama_1.36b.py --init_from=resume --eval_only=True
```

Good luck with your training! 🚀

