# Scaling Law Analysis - Quick Start

## 📁 Files in This Folder

### Core Script
- **`detailed_cost_analysis.py`** - Main analysis tool

### Example Configurations

**Backward Scaling Configs** (Training Setup → Optimal N, D):
- **`backward_scaling_hoffmann.jsonc`** - Hoffmann et al. (2022) parameters
- **`backward_scaling_besiroglu.jsonc`** - Besiroglu et al. (2024) parameters
- **`backward_scaling_auto.jsonc`** - Auto-detection example (no peak_flops_per_gpu needed!)

**Model Architecture Configs** (Architecture → N, FLOPs):
- **`example_llama_config.jsonc`** - Annotated LLaMA-style example
- **`llama_7b_config.json`** - LLaMA 7B configuration

### Documentation
- **`README.md`** - Complete documentation
- **`QUICK_START.md`** - This file

---

## 🚀 Quick Commands

### Test Backward Scaling Law
```bash
# Using Hoffmann (2022) - Standard Chinchilla
python detailed_cost_analysis.py --backward_config backward_scaling_hoffmann.jsonc

# Using Besiroglu (2024) - Newer Epoch AI reanalysis
python detailed_cost_analysis.py --backward_config backward_scaling_besiroglu.jsonc

# Auto-detection example (peak FLOPs detected from gpu_type + dtype)
python detailed_cost_analysis.py --backward_config backward_scaling_auto.jsonc
```

**What it does:**
- Calculates **N** from your architecture
- Calculates **C** from your GPU setup (type, count, hours, MFU)
- Solves for optimal **D** using detailed formulas
- Predicts **loss** using scaling law
- Checks dataset constraints

### Test Forward Analysis
```bash
# Standard JSON
python detailed_cost_analysis.py --model_config llama_7b_config.json

# JSONC with comments
python detailed_cost_analysis.py --model_config example_llama_config.jsonc
```

**What it does:**
- Calculates **N** (total parameters)
- Calculates **FLOPs** per forward pass
- Calculates **memory** requirements
- Shows component breakdown (attention vs FFN)

### Run Validation
```bash
python detailed_cost_analysis.py --validate
```

**What it does:**
- Tests parameter calculation accuracy
- Verifies FLOPs formulas
- Shows sequence length scaling (S² behavior)

---

## 📊 Example Output

### Backward Scaling (Hoffmann vs Besiroglu)

**Same setup, different scaling law parameters:**

```
Hoffmann (2022):  Loss = 2.2133
Besiroglu (2024): Loss = 2.1957  (0.8% lower)
```

Both use same training setup:
- N = 6.89B parameters
- D = 102.09B tokens
- C = 9.23e+21 FLOPs

---

## 🎯 Key Formulas

### N (Parameters)
```
N = 2VH + L(4H² + 3HD_ff + 2H) + H
```

### C (Compute Budget)
```
C = num_gpus × peak_flops_per_gpu × MFU × hours × 3600
```

### D (Training Tokens)
```
D = C / training_flops_per_token
```

### L (Loss)
```
L(N, D) = E + A·N^(-α) + B·D^(-β)
```

---

## 📝 Editing Configs

All `.jsonc` files support comments:

```jsonc
{
  // This is a comment
  "hidden_size": 4096,  // Hidden dimension
  
  /* Multi-line
     comment */
  "num_hidden_layers": 32
}
```

---

## 🎯 GPU Auto-Detection Feature

The system **automatically detects** peak FLOPs from `gpu_type` and `dtype`:

```jsonc
{
  "training_gear": {
    "gpu_type": "H100",      // Auto-detects: 989 TFLOPS (BF16)
    "dtype": "bfloat16",
    // No need to specify peak_flops_per_gpu!
  }
}
```

**Supported GPUs:**
- B200: 4,500 TFLOPS (BF16)
- H200: 1,979 TFLOPS (BF16)
- H100: 989 TFLOPS (BF16)
- A100: 312 TFLOPS (BF16)
- V100: 125 TFLOPS (FP16)
- RTX4090: 82.6 TFLOPS (FP16)

**Manual override** (for custom GPUs):
```jsonc
{
  "training_gear": {
    "gpu_type": "Custom-GPU",
    "peak_flops_per_gpu": 1500e12  // Specify manually
  }
}
```

---

## ✅ All Tests Passed

- ✅ Forward analysis working
- ✅ JSONC comment support working
- ✅ GPU auto-detection working (H100, A100, B200, V100 tested)
- ✅ Backward scaling (Hoffmann) working
- ✅ Backward scaling (Besiroglu) working
- ✅ Validation tests working

**Ready to use!** 🚀

