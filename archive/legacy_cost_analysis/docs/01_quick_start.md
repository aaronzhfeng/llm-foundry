# Quick Start Guide

## Installation

No additional dependencies required! Uses only Python standard library and basic modules:
- `argparse`
- `json`
- `math`

## Quick Commands

### 1. Analyze LLaMA 7B
```bash
python llm_cost_analysis.py --model_config llama_7b_config.json
```

**Output**:
```
================================================================================
LLaMA Model Analysis
================================================================================
Total Parameters:        5,295,575,040 (5.30B)
FLOPs per forward pass:  22.81 TFLOPs
Peak Memory (training):  70.53 GB
================================================================================
```

### 2. Analyze DeepSeek V3 MoE
```bash
python llm_cost_analysis.py --model_config deepseek_v3_config.json
```

**Output**:
```
================================================================================
DeepSeek V3 Model Analysis
================================================================================
Total Parameters:        452,260,623,360 (452.26B)
FLOPs per forward pass:  125.14 TFLOPs
Peak Memory (training):  5065.37 GB
================================================================================
```

### 3. Find Optimal Model Size for Budget
```bash
python llm_cost_analysis.py --training_budget 10000
```

**Output**:
```
================================================================================
Optimal Model Sizing (Chinchilla Scaling Laws)
================================================================================
Training Budget:         $10,000.00
Best GPU:                T4
Total Training FLOPs:    6.69e+21 FLOPs
Optimal Model Size (N):  33,381,015,755 parameters (33.38B)
Optimal Training Tokens: 33,381,015,755 tokens (33.38B)
================================================================================
```

### 4. Run Validation Tests
```bash
python llm_cost_analysis.py --validate
```

### 5. Run All Examples
```bash
python example_usage.py
```

This will run 6 comprehensive examples showing:
- LLaMA 7B analysis
- DeepSeek V3 MoE analysis
- Custom model configuration
- Budget optimization
- Scaling analysis
- Sequence length impact

## Using as a Library

```python
from llm_cost_analysis import (
    calculate_llama_parameters,
    calculate_llama_flops,
    calculate_llama_memory
)
import json

# Load config
with open('llama_7b_config.json', 'r') as f:
    config = json.load(f)

# Calculate metrics
params = calculate_llama_parameters(config)
flops = calculate_llama_flops(config, sequence_length=2048)
memory = calculate_llama_memory(config, batch_size=1, sequence_length=2048)

print(f"Parameters: {params/1e9:.2f}B")
print(f"FLOPs: {flops:.2f} TFLOPs")
print(f"Memory: {memory:.2f} GB")
```

## Custom Model Configuration

Create your own model config JSON:

```json
{
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "vocab_size": 50000,
    "tie_word_embeddings": true
}
```

Then analyze:
```bash
python llm_cost_analysis.py --model_config my_custom_config.json
```

## Understanding the Output

### Parameters
- **Total Parameters**: All weights in the model
- For MoE: Includes ALL experts (even if not all activated)

### FLOPs per Forward Pass
- Computational cost for processing one batch
- For MoE: Only counts ACTIVATED experts
- Scales with sequence length (quadratic due to attention)

### Peak Memory
- Memory needed during training
- Includes: model (2 bytes/param) + gradients (2 bytes/param) + optimizer states (8 bytes/param) + activations
- Approximate: ~12× parameter count + activations

## Common Use Cases

### Compare Two Architectures
```bash
# Compare LLaMA vs DeepSeek
python llm_cost_analysis.py --model_config llama_7b_config.json
python llm_cost_analysis.py --model_config deepseek_v3_config.json
```

### Find Optimal Model for Budget
```bash
# Budget in dollars
python llm_cost_analysis.py --training_budget 1000   # $1,000
python llm_cost_analysis.py --training_budget 100000 # $100,000
```

### Test Different Sequence Lengths
```python
# Modify in code or create custom function
for seq_len in [512, 1024, 2048, 4096]:
    flops = calculate_llama_flops(config, sequence_length=seq_len)
    print(f"Seq {seq_len}: {flops:.2f} TFLOPs")
```

## Troubleshooting

### ImportError
Make sure you're in the correct directory:
```bash
cd /path/to/flops_parameter_counting/
python llm_cost_analysis.py --help
```

### Config File Not Found
Ensure JSON files are in the same directory:
```bash
ls *.json
# Should see: deepseek_v3_config.json  llama_7b_config.json
```

### Unexpected Results
Run validation to check implementation:
```bash
python llm_cost_analysis.py --validate
```

## Next Steps

1. Read **README.md** for detailed documentation on special cases
2. Read **IMPLEMENTATION_SUMMARY.md** for comprehensive overview
3. Check **example_usage.py** for more usage examples
4. Modify configs to test your own model architectures

## Key Formulas (Quick Reference)

### Parameters (Standard Transformer)
```
Total = Embeddings + Layers × (Attention + FFN) + Output
      = V×H + L×(4H² + 2H×D) + V×H
```

### FLOPs (per forward pass)
```
Total = L × (8SH² + 4S²H + 4SHD)
```

### Training FLOPs (Chinchilla)
```
Total = 6 × N × D
(N = parameters, D = tokens)
```

### Memory (training)
```
Total ≈ 12×N + Activations
(2 model + 2 grad + 8 optimizer)
```

## Questions?

Refer to:
- **README.md** - Detailed documentation
- **IMPLEMENTATION_SUMMARY.md** - Complete overview
- Source code comments - Inline documentation with paper references

