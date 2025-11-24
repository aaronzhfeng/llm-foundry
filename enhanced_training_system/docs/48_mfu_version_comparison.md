# 48. MFU Version Comparison (model_builder v1–v3)

To keep auditability high, we versioned `model_builder.py` inside `mfu_versions/`. This document summarizes the intent, math, and usage notes for each variant.

## Version Index

| Version | Location | Philosophy | When to Use |
|---------|----------|------------|-------------|
| v1 | `mfu_versions/model_builder_v1.py` | nanoGPT legacy (`6N + 12LHQT`) | Baseline parity with old experiments; sanity-check against historical logs |
| v2 | `mfu_versions/model_builder_v2.py` | Hybrid heuristic (Kaplan + manual breakdown) | Transitional runs that still rely on PaLM numerator but want extra stats |
| v3 | `mfu_versions/model_builder_v3.py` (current) | Audit-compliant component summation, GQA-aware, includes logits | Production runs, MFU reporting, compliance reviews |

## Mathematical Snapshot

### Version 1 – Kaplan Heuristic
- Total FLOPs per token: `C ≈ 6N + 12 L H Q T`
- Hardware denominator: single dense peak constant (per GPU type).
- No awareness of GQA, SwiGLU, or vocab/logits.

### Version 2 – Combined Formula
- Still reports core MFU via `6N + 12LHQT`.
- Adds detailed stats but attention projections double-count heads.
- RoPE and logit FLOPs included partially; assumes dense hardware only.

### Version 3 – Component Summation (Audit)
- Calculates per-layer FLOPs explicitly:
  ```text
  flops_attn = 2 S H^2 (2 + 2/G) + 4 S^2 H
  flops_ffn = (6 or 4) S H D_ff   # SwiGLU-aware
  flops_norm = 2 * norm_flops_per_layer
  flops_logits = 2 S H V
  total_forward = L * (attn + ffn + norm) + norm + logits
  training_flops_per_token = 3 * total_forward / S
  ```
- Excludes RoPE, includes logit projection, respects `num_key_value_heads`, `ffn_type`, and `intermediate_size`.
- Supports dense vs sparse peak FLOPs via `use_sparse_specs=True`.

## Version Selection

1. Pick the target file from `mfu_versions/`.
2. Set `MODEL_BUILDER_IMPL` env var or update the import path before launching `train.py`.
3. Document the version in experiment logs (`calculation_method` field is included in the MFU dict).

## Recommended Usage
- **Unit tests / historical parity:** v1.
- **Comparative debugging:** Run all three via `Z_command/B200_QWEN3_MFU_TEST.md` to ensure MFU deltas come from math, not runtime behavior.
- **Production metrics / publications:** v3 only.

## Future Work
- Add FP8/FP4 denominators that inspect mixed precision configs automatically.
- Extend v3 to understand MoE active parameter counts once those configs land.

