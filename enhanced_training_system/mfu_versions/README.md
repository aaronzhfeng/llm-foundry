# MFU Calculation Version History

This folder tracks the evolution of MFU (Model FLOPs Utilization) calculation methods in the enhanced training system.

---

## 📚 Version Overview

| Version | Name | Date | Method | Status |
|---------|------|------|--------|--------|
| **v1** | Legacy nanoGPT | 2023 | PaLM 6N heuristic | 📚 Reference |
| **v2** | Combined Formula | 2025-11 | Component summation (algebraic) | ✅ Working |
| **v3** | Gold Standard | 2025-11 | Component summation (explicit) | ⭐ Current |

---

## Version Details

### 🔵 **Version 1: Legacy nanoGPT** (`model_builder_v1.py`)

**Philosophy**: Simple and pedagogical

**Method**: PaLM Appendix B heuristic
```python
flops_per_token = 6*N + 12*L*H*Q*T
```

**Characteristics**:
- ✅ Simple, clean, educational
- ✅ Accurate for GPT-2 style models
- ❌ Assumes Multi-Head Attention (no GQA)
- ❌ Assumes 4× FFN expansion
- ❌ Hardcoded A100 peak (312 TFLOPS)
- ❌ No component breakdown

**Use Case**: Learning, GPT-2 models, simplicity over accuracy

**Example Output**:
```python
{
    'mfu_percent': 44.0,
    'calculation_method': 'nanogpt_legacy_v1'
}
```

---

### 🟢 **Version 2: Combined Formula** (`model_builder_v2.py`)

**Philosophy**: Mathematically elegant with component awareness

**Method**: Algebraic component summation
```python
# GQA Projections (combined formula)
attention_qkv_flops = 2 * S * H * H * (2 + 2/G)

# Component summation
total = attention + ffn + logit + rope + norms
```

**Characteristics**:
- ✅ GQA-aware (group size G)
- ✅ SwiGLU-aware (dynamic intermediate_size)
- ✅ Logit layer included
- ✅ Hardware auto-detection (B200/H100/A100)
- ✅ Dense/Sparse toggle
- ⚠️ RoPE included (debatable)
- ⚠️ Combined algebraic formula (less explicit)
- ✅ Comprehensive breakdown (25+ metrics)

**Use Case**: Production training with modern architectures

**Example Output**:
```python
{
    'mfu_percent': 44.36,
    'attention_to_ffn_ratio': 2.50,  # Note: This was inflated due to bug
    'gqa_group_size': 2.0,
    'logit_flops': 1270000000
}
```

---

### 🟡 **Version 3: Gold Standard** (`model_builder_v3.py`) ⭐ **CURRENT**

**Philosophy**: Maximum clarity and audit compliance

**Method**: Explicit component breakdown
```python
# Explicit Q/K/V separation
flops_q = 2 * S * H * H
flops_kv = 2 * (2 * S * H * H / G)
flops_proj = 2 * S * H * H

# Pure component summation
total = (Q + K + V + O + scores + context) + ffn + logit + norms
```

**Characteristics**:
- ✅ GQA-aware (explicit Q/K/V breakdown)
- ✅ SwiGLU-aware (hard-coded 3-matrix factor)
- ✅ Logit layer included
- ✅ Hardware auto-detection (B200/H100/A100)
- ✅ Dense/Sparse toggle
- ✅ **RoPE excluded** (strict PaLM compliance)
- ✅ **Explicit variable names** (self-documenting)
- ✅ Comprehensive breakdown (25+ metrics)
- ✅ Calculation method tracking

**Use Case**: Modern production training, academic comparability, auditing

**Example Output**:
```python
{
    'mfu_percent': 44.0,
    'attention_to_ffn_ratio': 0.50,  # Corrected!
    'gqa_group_size': 2.0,
    'logit_flops': 1270000000,
    'calculation_method': 'component_summation_v2025'
}
```

---

## 📊 Comparison Matrix

### Accuracy for Different Architectures

| Architecture | v1 (nanoGPT) | v2 (Combined) | v3 (Gold Standard) |
|--------------|--------------|---------------|---------------------|
| **GPT-2** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Llama 2** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Llama 3 (GQA)** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Qwen 3 (GQA+SwiGLU)** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **MoE Models** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Code Quality

| Aspect | v1 | v2 | v3 |
|--------|----|----|-----|
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Readability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Maintainability** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Auditability** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Future-Proof** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔬 Mathematical Examples (Qwen 3 1.8B)

**Configuration**:
- Layers: 24
- Hidden: 2048
- Heads: 16 (Q), 8 (KV) → G=2
- Sequence: 2048
- FFN: 6144 (SwiGLU)
- Vocab: 151,643

### Version 1 (nanoGPT)

```python
N = 1.829e9
L, H, Q, T = 24, 16, 128, 2048

flops_per_token = 6*N + 12*L*H*Q*T
                = 6*1.829e9 + 12*24*16*128*2048
                = 10.974 GFLOPs + 1.509 GFLOPs
                = 12.48 GFLOPs per token
```

**Issues**:
- ❌ Assumes MHA (not accounting for G=2 savings)
- ❌ Missing logit layer overhead

---

### Version 2 (Combined)

```python
# Attention (combined formula)
attention_qkv = 2 * 2048 * 2048^2 * (2 + 2/2)
              = 2 * 2048 * 2048^2 * 3
              = 51.54 GFLOPs

attention_scores = 2 * 2048^2 * 2048 = 17.18 GFLOPs
attention_output = 2 * 2048^2 * 2048 = 17.18 GFLOPs

# FFN (SwiGLU)
ffn = 6 * 2048 * 2048 * 6144 = 154.62 GFLOPs

# Logit
logit = 2 * 2048 * 2048 * 151643 = 1.27 GFLOPs

# Total forward per token
total = (85.9 per layer * 24) + 1.27 = ~8.4 GFLOPs per token
```

**Note**: Attention calculation was correct, but breakdown ratio was inflated

---

### Version 3 (Gold Standard)

```python
# Attention (explicit)
flops_q = 2 * 2048 * 2048^2 = 17.18 GFLOPs
flops_kv = 2 * (2 * 2048 * 1024^2) = 8.59 GFLOPs  # G=2 reduction
flops_proj = 2 * 2048 * 2048^2 = 17.18 GFLOPs
flops_scores = 2 * 2048^2 * 2048 = 17.18 GFLOPs
flops_context = 2 * 2048^2 * 2048 = 17.18 GFLOPs

attention_total = 77.31 GFLOPs per layer

# FFN (SwiGLU)
ffn = 6 * 2048 * 2048 * 6144 = 154.62 GFLOPs per layer

# Logit
logit = 2 * 2048 * 2048 * 151643 = 1.27 GFLOPs

# Total forward per token
total = ((77.31 + 154.62) * 24 + 1.27) / 2048 = ~2.75 GFLOPs per token
total_training = 3 * 2.75 = 8.25 GFLOPs per token
```

**Attention to FFN Ratio**: 77.31 / 154.62 = **0.50** ✅

---

## 🎯 When to Use Each Version

### Use v1 if:
- Learning GPT-2 architecture
- Teaching/tutorials
- Only training standard GPT-2 models
- Simplicity is paramount

### Use v2 if:
- Production training (pre-Nov 2025)
- Already using and don't want to change
- Algebraic elegance preferred
- Okay with RoPE being counted

### Use v3 if:
- ⭐ Production training (Nov 2025+)
- Training modern architectures (GQA, SwiGLU)
- Need academic comparability
- Want maximum audit transparency
- Future-proofing

---

## 📝 Key Differences Summary

| Feature | v1 | v2 | v3 |
|---------|----|----|-----|
| **GQA Support** | ❌ No | ✅ Yes (algebraic) | ✅ Yes (explicit) |
| **SwiGLU** | ❌ No | ✅ Yes | ✅ Yes |
| **Logit Layer** | ❌ No | ✅ Yes | ✅ Yes |
| **RoPE Handling** | ❌ Ignored | ⚠️ Included | ✅ Excluded (strict) |
| **Hardware Detection** | ❌ Hardcoded A100 | ✅ Auto-detect | ✅ Auto-detect |
| **Code Style** | Simple | Algebraic | Explicit |
| **Calculation Method** | Parameter-count | Component sum | Component sum |

---

## 🔄 Migration Guide

### From v1 → v2 or v3

**Breaking Changes**: None (signature compatible)

**Expected Changes**:
- MFU may appear **lower** (v1 overestimated on modern architectures)
- New metrics available: `gqa_group_size`, `logit_flops`, etc.

### From v2 → v3

**Breaking Changes**: None (fully backward compatible)

**Expected Changes**:
- `attention_to_ffn_ratio`: Will **decrease** (was inflated in some cases)
- MFU percentage: May **increase** slightly (if RoPE was significant)
- New field: `calculation_method: 'component_summation_v2025'`

---

## 📚 References

1. **nanoGPT**: Andrej Karpathy's educational GPT implementation
2. **PaLM Paper**: Chowdhery et al. (2022), Appendix B
3. **GQA Paper**: Ainslie et al. (2023)
4. **MFU Audit**: "Review MFU Computation Logic" Gold Standard (2025)

---

## 🚀 Current Recommendation

**Use Version 3 (Gold Standard)** for all new projects and production training.

It provides the best balance of:
- ✅ Accuracy across all architectures
- ✅ Code clarity and maintainability
- ✅ Academic comparability
- ✅ Future-proofing

---

**Last Updated**: 2025-11-21  
**Current Production Version**: v3 (`model_builder_v3.py`)

