# Scaling Law Analysis for Large Language Models

Implementation of Kaplan (OpenAI 2020) and Chinchilla (DeepMind 2022) scaling laws for optimal compute allocation.

## Overview

Scaling laws predict how model performance varies with model size (N), training tokens (D), and compute budget (C). This implementation provides:

- **Chinchilla scaling law**: Modern optimal allocation (α≈β≈0.5)
- **Kaplan scaling law**: Historical reference (N_opt ∝ C^0.73)
- **Compute budget C**: C ≈ 6·N·D
- **Loss prediction**: L(N,D) for given model and data size
- **Dollar budget conversion**: Map $ to optimal N,D

## Quick Start

```bash
cd Scaling_law/

# Validate against known models
python scaling_law_analysis.py --validate

# Analyze compute budget
python scaling_law_analysis.py --config chinchilla_config.json --compute_budget 1e23

# Compare Kaplan vs Chinchilla
python scaling_law_analysis.py --compare --compute_budget 1e23

# Dollar budget analysis
python scaling_law_analysis.py --config chinchilla_config.json --budget_dollars 10000
```

## Usage

### Command Line Options

```bash
python scaling_law_analysis.py [OPTIONS]

Options:
  --config PATH           chinchilla_config.json or kaplan_config.json
  --compute_budget FLOAT  Compute budget in FLOPs
  --compare               Compare Kaplan vs Chinchilla
  --budget_dollars FLOAT  Dollar budget for training
  --hardware STR          Hardware config (default: 8x_a100)
  --validate              Validate against known models
```

## Examples

### Example 1: Chinchilla Optimal Allocation

```bash
python scaling_law_analysis.py --config chinchilla_config.json --compute_budget 1e23
```

**Output**:
```
Scaling Law Analysis (CHINCHILLA)
============================================================

Compute Budget: 1.00e+23 FLOPs

Optimal Allocation:
  Model size (N): 14,598,306,275 params (14.60B)
  Training tokens (D): 1,141,684,956,624 tokens (1141.68B)
  Predicted loss: 2.0050
  N/D ratio: 0.01

Compute verification:
  C = 6·N·D = 1.00e+23 FLOPs
  Error: 0.0000%
============================================================
```

### Example 2: Kaplan vs Chinchilla Comparison

```bash
python scaling_law_analysis.py --compare --compute_budget 1e23
```

**Output**:
```
Scaling Laws Comparison
============================================================
Compute Budget: 1.00e+23 FLOPs

Method          N (params)           D (tokens)           N/D ratio    Loss    
--------------------------------------------------------------------------------
Chinchilla           14.60B         1141.68B       0.01   2.0050
Kaplan               75.91B          219.57B       0.35   1.8143

Key differences:
  N ratio (Kaplan/Chinchilla): 5.20x
  D ratio (Chinchilla/Kaplan): 5.20x
  Chinchilla uses 5.2x more tokens for same compute
============================================================
```

**Key insight**: Kaplan allocates more to model size, Chinchilla balances N and D equally.

### Example 3: Dollar Budget Analysis

```bash
python scaling_law_analysis.py --config chinchilla_config.json --budget_dollars 10000 --hardware 8x_a100
```

**Output**:
```
Budget Analysis (CHINCHILLA)
============================================================
Dollar budget: $10,000.00
Hardware: 8x_a100
Training time: 625.0 hours (26.0 days)
Compute budget: 5.62e+21 FLOPs

Optimal Model Configuration:
  Model size: 3,976,743,799 params (3.98B)
  Training tokens: 235,368,443,968 tokens (235.37B)
  Predicted loss: 2.1802
============================================================
```

### Example 4: Validation Against Known Models

```bash
python scaling_law_analysis.py --validate
```

**Output**:
```
VALIDATION: Testing scaling laws against known models
================================================================================

Model                N (params)      D (tokens)      C (FLOPs)       Status    
--------------------------------------------------------------------------------
LLaMA 7B             6.70B           1000.00B        4.02e+22        Near-opt  
LLaMA 65B            65.20B          1400.00B        5.48e+23        Over-param
GPT-3 175B           175.00B         300.00B         3.15e+23        Over-param
Chinchilla 70B       70.00B          1400.00B        5.88e+23        Optimal   

Key insights:
  - Chinchilla 70B: Optimal by design
  - LLaMA models: Near-optimal allocation
  - GPT-3 175B: Over-parameterized (under-trained on data)
================================================================================
```

## Configuration Files

### chinchilla_config.json

Contains Chinchilla scaling law parameters:
- E = 1.69 (irreducible loss)
- A = 406.4, α = 0.34 (model size scaling)
- B = 410.7, β = 0.28 (data size scaling)

### kaplan_config.json

Contains Kaplan scaling law parameters:
- N_c = 8.8e13, α_N = 0.076
- D_c = 5.4e13, α_D = 0.095
- optimal_exponent = 0.73

### custom_budget_config.json

Contains hardware specifications for dollar budget conversion.

## Formulas

### Chinchilla Optimal Allocation

```
N_opt = G · (C/6)^(β/(α+β))
D_opt = (1/G) · (C/6)^(α/(α+β))

where G = (αA/βB)^(1/(α+β))
```

**Reference**: Hoffmann et al., 2022, Equation 4

### Chinchilla Loss Prediction

```
L(N,D) = E + A·N^(-α) + B·D^(-β)
```

**Reference**: Hoffmann et al., 2022, Equation 1

### Kaplan Optimal Allocation

```
N_opt ∝ C^0.73
D_opt = C / (6·N_opt)
```

**Reference**: Kaplan et al., 2020, Section 5

### Compute Budget

```
C = 6·N·D
```

where:
- C = total training FLOPs
- N = model parameters
- D = training tokens
- 6 = forward (2) + backward (4) multiplier

## Hardware Configurations

Available in `custom_budget_config.json`:
- `single_a100`: 312 TFLOPS, $2/hour
- `8x_a100`: 2,496 TFLOPS, $16/hour
- `64x_a100`: 19,968 TFLOPS, $128/hour
- `8x_h100`: 7,912 TFLOPS, $24/hour

## References

1. **Kaplan et al.**, "Scaling Laws for Neural Language Models" (2020)
   https://arxiv.org/abs/2001.08361

2. **Hoffmann et al.**, "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
   https://arxiv.org/abs/2203.15556

## Key Insights

### Chinchilla vs Kaplan

| Aspect | Chinchilla | Kaplan |
|--------|-----------|--------|
| **N/D balance** | ~1:1 (equal scaling) | N scales faster (C^0.73) |
| **Recommendation** | More tokens needed | Larger models, less data |
| **Modern usage** | Preferred approach | Historical reference |

### Validation Results

- **LLaMA models**: Near-optimal (follow Chinchilla guidance)
- **GPT-3 175B**: Over-parameterized (175B params on 300B tokens)
- **Chinchilla 70B**: Optimal (70B params on 1.4T tokens)

**Main insight**: Modern LLMs should balance model size and training tokens equally (Chinchilla approach).

