# DSC180A Section 06 - Open LLM Training, Inference, and Infrastructure

This repository contains four complementary components for Large Language Model training analysis and implementation.

## 📁 Repository Structure

### 1. `scaling_law_analysis/`
**Backward Scaling Law System** - Calculate optimal training tokens (D) from infrastructure constraints.

**Key Features:**
- ✅ Detailed FLOPs formulas (NOT simplified C=6ND)
- ✅ GPU auto-detection (B200, H200, H100, A100, V100)
- ✅ FP8/BF16/FP16 precision support
- ✅ Flash Attention memory optimization
- ✅ Chinchilla & Besiroglu scaling laws
- ✅ Dataset constraint enforcement

**Quick Start:**
```bash
cd scaling_law_analysis/
python detailed_cost_analysis.py --backward_config backward_scaling_hoffmann.jsonc
```

See [`scaling_law_analysis/QUICK_START.md`](scaling_law_analysis/QUICK_START.md) for details.

---

### 2. `flops_calculation/`
**Simple FLOPs Analysis Tool** - Straightforward parameter and FLOPs counting for LLaMA and DeepSeek models.

**Key Features:**
- ✅ Parameter calculation (embeddings, attention, FFN)
- ✅ FLOPs calculation per token (forward and training)
- ✅ Component breakdown (attention vs FFN)
- ✅ Simple, easy-to-understand implementation
- ✅ Supports LLaMA and DeepSeek V3 architectures

**Quick Start:**
```bash
cd flops_calculation/
python flops_analysis.py llama_7b_config.json
```

---

### 3. `memory_calculation/`
**Memory Estimation Tool** - Lightweight analyzer for peak GPU memory requirements.

**Key Features:**
- ✅ Dense decoder-only models (BF16 precision)
- ✅ Muon optimizer memory (~4 bytes/param, lighter than Adam)
- ✅ Memory breakdown (weights, gradients, optimizer, activations)
- ✅ Per-GPU memory estimation (checks if model fits)
- ✅ Fast iteration for architecture search

**Quick Start:**
```bash
cd memory_calculation/
python memory_calculation.py
```

See [`memory_calculation/README.md`](memory_calculation/README.md) for details.

---

### 4. `system/`
**Enhanced GPT Training System** - Complete training implementation with MFU monitoring.

**Built on [nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy, with significant enhancements:

**Our Enhancements:**
- ✅ **Detailed MFU calculation** using academic formulas (`FLOPs = 12SBH² + 2aS²BH`)
- ✅ **3 Attention backends**: FlashAttention-2, FlashAttention-1 (SDPA), Manual
- ✅ **Comprehensive monitoring**: Memory tracking, gradient health, per-iteration metrics
- ✅ **JSON logging**: Detailed per-run metrics with full configuration tracking
- ✅ **Hardware auto-detection**: B200, H200, H100, A100 GPU specs
- ✅ **Multi-GPU strategies**: DDP, ZeRO-1, FSDP support
- ✅ **Modular architecture**: Configurable components (RMSNorm, SwiGLU, RoPE, etc.)

**Quick Start:**
```bash
cd system/
python train.py config/train_shakespeare.py
```

See [`system/README.md`](system/README.md) for details.

**Original nanoGPT:** https://github.com/karpathy/nanoGPT

---

## 🎯 Key Features by Component

### Scaling Law Analysis (Comprehensive)
1. **Backward calculation**: GPU setup → Optimal (N, D)
2. **Detailed formulas**: Architecture-specific FLOPs (not averaged)
3. **GPU auto-detection**: Automatically detects peak FLOPs from GPU type
4. **FP8 support**: Correct handling of FP8 (2× faster than BF16)
5. **Flash Attention**: Optional memory optimization parameter
6. **Chinchilla & Besiroglu**: Two scaling law bases for comparison

### FLOPs Calculation (Simple)
1. **Straightforward implementation**: Easy to understand and verify
2. **Parameter counting**: Embeddings, attention, FFN breakdown
3. **FLOPs per token**: Forward and training (3×) calculations
4. **Multi-architecture**: LLaMA and DeepSeek V3 support

### Memory Calculation (Lightweight)
1. **Fast estimation**: Optimized for rapid architecture iteration
2. **Muon optimizer**: ~4 bytes/param (lighter than Adam's ~12 bytes)
3. **Dense models**: Focused on standard transformer architectures
4. **Fit check**: Verifies if model fits on available GPU memory

### Training System (Built on nanoGPT)
**Base:** [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy

**Our Enhancements:**
1. **Academic MFU calculation**: Formula-based `FLOPs = 12SBH² + 2aS²BH` (not approximations)
2. **3 Attention backends**: FlashAttention-2 (explicit), FlashAttention-1 (SDPA), Manual
3. **Detailed monitoring**: Component breakdown, memory tracking, gradient health
4. **JSON logging**: Complete per-iteration metrics and configuration tracking
5. **Hardware auto-detection**: B200/H200/H100/A100 peak FLOPs detection
6. **Advanced parallelism**: DDP, ZeRO-1, FSDP with memory optimization
7. **Modular architecture**: Configurable normalization, activation, position encoding

---

## 📚 Documentation

- **[scaling_law_analysis/README.md](scaling_law_analysis/README.md)** - Complete scaling law documentation
- **[scaling_law_analysis/QUICK_START.md](scaling_law_analysis/QUICK_START.md)** - Quick reference guide
- **[memory_calculation/README.md](memory_calculation/README.md)** - Memory estimation guide
- **[system/SYSTEM_OVERVIEW.md](system/SYSTEM_OVERVIEW.md)** - Training system details
- **[system/TESTING.md](system/TESTING.md)** - Testing guide
- **[system/ATTENTION_BACKENDS.md](system/ATTENTION_BACKENDS.md)** - FlashAttention backend options

---

## 🚀 Quick Examples

### Calculate Optimal Training Tokens (Scaling Law)
```bash
cd scaling_law_analysis/
python detailed_cost_analysis.py --backward_config backward_scaling_hoffmann.jsonc
```

### Calculate FLOPs (Simple Tool)
```bash
cd flops_calculation/
python flops_analysis.py llama_7b_config.json
```

### Estimate Memory (Muon Optimizer)
```bash
cd memory_calculation/
python memory_calculation.py
```

### Train a Model
```bash
cd system/
python train.py config/train_gpt2.py
```

### Compare Scaling Laws (Hoffmann vs Besiroglu)
```bash
cd scaling_law_analysis/
python detailed_cost_analysis.py --backward_config backward_scaling_hoffmann.jsonc
python detailed_cost_analysis.py --backward_config backward_scaling_besiroglu.jsonc
```

---

## ✅ Features

**Scaling Law Analysis:**
- [x] Detailed C, N, D formulas
- [x] GPU auto-detection (15+ GPU types)
- [x] FP8/BF16/FP16/FP32 support
- [x] Flash Attention option
- [x] Dataset constraints
- [x] JSONC config support

**FLOPs Calculation:**
- [x] Simple parameter counting
- [x] FLOPs per token calculation
- [x] Component breakdown
- [x] LLaMA & DeepSeek support

**Memory Calculation:**
- [x] BF16 + Muon optimizer
- [x] Dense model focus
- [x] Fast estimation
- [x] Per-GPU fit check

**Training System:**
- [x] Academic MFU calculation
- [x] 3 Attention backends (FA-2, FA-1, manual)
- [x] Multi-GPU parallelism
- [x] JSON logging
- [x] Gradient monitoring
- [x] Memory tracking
- [x] Hardware auto-detection

---

## 📊 Status

**All 4 Components:**
- ✅ Scaling Law Analysis - Fully functional
- ✅ FLOPs Calculation - Fully functional
- ✅ Memory Calculation - Fully functional
- ✅ Training System - Fully functional

**Recent Updates:**
- ✅ Bug fixes applied (training FLOPs calculation)
- ✅ GPU specs corrected (FP8 is 2× faster than BF16)
- ✅ Flash Attention support added (scaling law + training system)
- ✅ 3 Attention backends (FlashAttention-2, FlashAttention-1, Manual)
- ✅ Comprehensive documentation
- ✅ Example configs provided

**Ready for use!** 🎉

---

## 🙏 Acknowledgments

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy - Foundation for our enhanced training system
- **Chinchilla Scaling Laws** (Hoffmann et al., 2022) - Scaling law formulas
- **Epoch AI** (Besiroglu et al., 2024) - Updated scaling law reanalysis
- **FlashAttention** (Dao et al., 2022, 2023) - Memory-efficient attention
- **PyTorch Team** - FSDP and distributed training infrastructure
