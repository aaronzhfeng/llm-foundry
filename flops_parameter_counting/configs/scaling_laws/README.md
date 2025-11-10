# Scaling Law Configs

This directory contains configurations for **backward scaling analysis** (calculating optimal N and D from training infrastructure and compute budget).

## Directory Structure

```
scaling_laws/
├── hoffmann/           # Chinchilla scaling law (Hoffmann et al., 2022)
├── besiroglu/          # Updated scaling law (Besiroglu et al., 2024)
└── custom/             # Custom configs and experiments
```

## Usage

### Backward Analysis (Get N and D from Compute Budget)

```bash
# Using Hoffmann scaling law
python detailed_cost_analysis.py \
  --backward_config configs/scaling_laws/hoffmann/backward_scaling_config.jsonc

# Using Besiroglu scaling law (newer)
python detailed_cost_analysis.py \
  --backward_config configs/scaling_laws/besiroglu/backward_scaling_besiroglu.jsonc

# Verify LLaMA 1.36B
python detailed_cost_analysis.py \
  --backward_config configs/scaling_laws/custom/verify_llama_1.36b.jsonc
```

## Config Format

All configs use JSONC format (JSON with comments):

```jsonc
{
  "architecture": {
    // Model dimensions - defines N
    "hidden_size": 2304,
    "num_hidden_layers": 18,
    ...
  },
  
  "training_gear": {
    // GPU cluster - defines available compute
    "num_gpus": 4,
    "gpu_type": "A100",
    "available_hours": 720,  // Training duration in hours
    ...
  },
  
  "training_efficiency": {
    // Realistic achievable performance
    "expected_mfu": 0.40,  // Model FLOPs Utilization
    "uptime": 0.95
  },
  
  "scaling_law": {
    // Which scaling law to use
    "type": "hoffmann",  // or "besiroglu"
    ...
  }
}
```

## Scaling Law Types

### Hoffmann (Chinchilla, 2022)

**Paper**: Hoffmann et al., "Training Compute-Optimal Large Language Models"

**Formula**:
```
L(N, D) = A/N^α + B/D^β + E
```

**Parameters**:
- A = 406.4, B = 410.7
- α = 0.34, β = 0.28
- E = 1.69

**Use when**: Standard Chinchilla-style scaling, widely adopted baseline

### Besiroglu (Epoch AI, 2024)

**Paper**: Besiroglu et al., "Chinchilla Scaling: A replication attempt"

**Formula**:
```
L(N, D) = A/N^α + B/D^β + E
```

**Parameters** (updated):
- A = 482.0, B = 328.7
- α = 0.35, β = 0.25
- E = 1.82

**Use when**: More recent data, potentially more accurate for modern models

## Files

### Hoffmann (Standard Chinchilla)

- **`hoffmann/backward_scaling_config.jsonc`**
  - Complete annotated example
  - All parameters explained
  - Default Chinchilla parameters

### Besiroglu (Updated)

- **`besiroglu/backward_scaling_besiroglu.jsonc`**
  - Updated scaling law parameters
  - Based on reanalysis of Chinchilla data

### Custom

- **`custom/verify_llama_1.36b.jsonc`**
  - Verification config for your LLaMA 1.36B model
  - Should reproduce: N=1.29B, D=84.72B, L=2.37

- **`custom/backward_scaling_auto.jsonc`**
  - GPU auto-detection example
  - No manual peak_flops specification needed

- **`custom/backward_scaling_flash.jsonc`**
  - Flash Attention memory optimization example
  - Shows memory savings from Flash Attention

## Expected Output

```
Backward Scaling Law: Training Setup → Optimal (N, D)
======================================================

Step 1: Calculate N from architecture
  Architecture: 18L × 2304H
  Model parameters (N): 1.294B

Step 2: Calculate available compute C
  GPUs: 4× A100 (312 TFLOPS each)
  Training time: 720 hours
  MFU: 40%, Uptime: 95%
  Available compute (C): 1.36×10²¹ FLOPs

Step 3: Calculate optimal D
  Training FLOPs per token: 5.16×10¹³
  Optimal tokens (D): 8.472×10¹⁰ (84.72B)

Step 4: Predict loss
  Chinchilla formula: L = 406.4/N^0.34 + 410.7/D^0.28 + 1.69
  Expected loss: 2.372

Verification:
  Dataset size: 627B tokens ✓ (sufficient)
  Training fraction: 13.5% of dataset
  Iterations needed: ~81,000 @ 1M tokens/iter
```

## Creating New Configs

1. Copy an existing config as template
2. Update `architecture` section with your model dimensions
3. Update `training_gear` with your hardware setup
4. Adjust `training_efficiency` based on your measurements
5. Choose scaling law type (`hoffmann` or `besiroglu`)

## Troubleshooting

### D exceeds dataset size

```
WARNING: Optimal D (100B) exceeds dataset size (50B)
```

**Solution**: Either get more data or accept suboptimal training

### C too small for reasonable N

```
WARNING: Very small C leads to N < 100M
```

**Solution**: Increase training time or number of GPUs

### MFU seems too high/low

**Typical MFU ranges**:
- Well-optimized: 35-45%
- Standard: 25-35%
- Poorly optimized: <25%

Check your training logs for actual measured MFU.

## References

1. **Hoffmann et al.**, "Training Compute-Optimal Large Language Models" (2022)
   - https://arxiv.org/abs/2203.15556

2. **Besiroglu et al.**, "Chinchilla Scaling: A replication attempt" (2024)
   - https://arxiv.org/abs/2404.10102

3. **Epoch AI**, Backward-Forward FLOP Ratio
   - https://epoch.ai/blog/backward-forward-FLOP-ratio

