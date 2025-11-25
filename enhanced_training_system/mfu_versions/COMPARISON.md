# MFU Version Comparison Quick Reference

| | v1 (Legacy) | v2 (Combined) | v3 (Gold Standard) ⭐ |
|---|-------------|---------------|----------------------|
| **Lines of Code** | 614 | 730 | 707 |
| **Date** | 2023 (nanoGPT) | Nov 2025 | Nov 2025 |
| **Method** | 6N heuristic | Algebraic components | Explicit components |
| **GQA Support** | ❌ | ✅ | ✅ |
| **SwiGLU** | ❌ | ✅ | ✅ |
| **Logit Layer** | ❌ | ✅ | ✅ |
| **RoPE** | Ignored | Included | Excluded (strict) |
| **B200 Support** | ❌ | ✅ | ✅ |
| **Code Style** | Simple | Elegant | Explicit |
| **Status** | Reference | Previous | **Current** |

---

## Formula Comparison (Attention Projections)

### v1: Parameter-Count Heuristic
```python
# Part of overall 6N + 12LHQT
flops_per_token = 6*N + 12*L*H*Q*T
```
Simple but inflexible.

---

### v2: Combined Algebraic Formula
```python
# GQA-aware, combined formula
attention_qkv_flops = 2 * S * H * H * (2 + 2/G)
# Where G = n_head / n_kv_head
```
Correct but requires mental algebra.

---

### v3: Explicit Component Breakdown
```python
# Q Projection
flops_q = 2 * S * H * H

# K/V Projections (GQA reduced)
flops_kv = 2 * (2 * S * H * H / G)

# Output Projection
flops_proj = 2 * S * H * H

# Total
attention_flops = flops_q + flops_kv + flops_proj + scores + context
```
Crystal clear, self-documenting.

---

## Expected MFU for Qwen 3 1.8B (8× B200)

| Version | Expected MFU | Attn/FFN Ratio | Notes |
|---------|--------------|----------------|-------|
| **v1** | ~52% | N/A | Overestimates (no GQA, no logit) |
| **v2** | ~44% | 2.50 | Correct MFU, inflated ratio |
| **v3** | ~44% | 0.50 | ✅ All metrics correct |

---

## Recommendation

🎯 **Use v3 (Gold Standard)** for all production work.

- Most accurate
- Most maintainable
- Most auditable
- Future-proof

Keep v1 and v2 as reference for understanding the evolution of MFU calculation methods.

