# Quick Reference - New Organized Structure

## 📁 New Directory Structure (Updated!)

```
flops_parameter_counting/
├── configs/
│   ├── models/              # Forward analysis configs
│   │   ├── llama_1.36b.json      # Your LLaMA 1.36B model ✨ NEW
│   │   ├── gpt2_1.36b.json       # Your GPT-2 1.36B model ✨ NEW
│   │   ├── llama_7b.json         # Reference LLaMA 7B
│   │   ├── deepseek_v3.json      # DeepSeek V3 MoE
│   │   └── README.md             # Model configs documentation
│   │
│   └── scaling_laws/        # Backward analysis configs
│       ├── hoffmann/        # Chinchilla law (2022)
│       │   └── backward_scaling_config.jsonc
│       ├── besiroglu/       # Updated law (2024)
│       │   └── backward_scaling_besiroglu.jsonc
│       ├── custom/          # Custom experiments
│       │   ├── verify_llama_1.36b.jsonc  ✨ NEW
│       │   ├── backward_scaling_auto.jsonc
│       │   └── backward_scaling_flash.jsonc
│       └── README.md        # Scaling laws documentation
│
├── detailed_cost_analysis.py     # Main script (updated)
└── README.md
```

## 🚀 Quick Commands

### Forward Analysis (Calculate N from Architecture)

```bash
# Just specify filename - script searches configs/models/ automatically!
python detailed_cost_analysis.py --model_config llama_1.36b.json
python detailed_cost_analysis.py --model_config gpt2_1.36b.json

# Or specify full path
python detailed_cost_analysis.py --model_config configs/models/llama_7b.json
```

### Backward Analysis (Calculate N and D from Compute Budget)

```bash
# Just specify filename - script searches configs/scaling_laws/ automatically!
python detailed_cost_analysis.py --backward_config verify_llama_1.36b.jsonc

# Or specify subdirectory
python detailed_cost_analysis.py --backward_config hoffmann/backward_scaling_config.jsonc
python detailed_cost_analysis.py --backward_config besiroglu/backward_scaling_besiroglu.jsonc

# Or full path
python detailed_cost_analysis.py --backward_config configs/scaling_laws/custom/verify_llama_1.36b.jsonc
```

## 🎯 Common Use Cases

### Verify Your LLaMA 1.36B Model

```bash
# Forward: Calculate N (parameters)
python detailed_cost_analysis.py --model_config llama_1.36b.json

# Expected output:
#   Model parameters (N): 1.29B
#   FLOPs per token: 42.5 GFLOPs
```

### Verify N-D Pair from Scaling Law

```bash
# Backward: Calculate N and D from compute budget
python detailed_cost_analysis.py --backward_config verify_llama_1.36b.jsonc

# Expected output:
#   N = 1.29B parameters
#   D = 84.72B tokens
#   Expected loss: 2.37
```

### Compare GPT-2 vs LLaMA at Same Scale

```bash
# Analyze both architectures
python detailed_cost_analysis.py --model_config gpt2_1.36b.json
python detailed_cost_analysis.py --model_config llama_1.36b.json

# Compare:
#   GPT-2: ~1.41B params, ~32 GFLOPs/token (faster)
#   LLaMA: ~1.29B params, ~42 GFLOPs/token (better quality)
```

## 📋 Config File Resolution

The script now automatically searches in the new directory structure:

### For `--model_config`:
1. Check if file exists as provided (exact path)
2. Search in `configs/models/`
3. Fall back to current directory (backward compatibility)

### For `--backward_config`:
1. Check if file exists as provided (exact path)
2. Search in `configs/scaling_laws/`
3. Search in `configs/scaling_laws/hoffmann/`
4. Search in `configs/scaling_laws/besiroglu/`
5. Search in `configs/scaling_laws/custom/`
6. Fall back to current directory (backward compatibility)

**This means you can use short filenames!**

```bash
# These all work:
python detailed_cost_analysis.py --model_config llama_1.36b.json
python detailed_cost_analysis.py --model_config configs/models/llama_1.36b.json
python detailed_cost_analysis.py --model_config /full/path/to/llama_1.36b.json
```

## 🆕 What Changed?

### Old Structure (Deprecated)
```
flops_parameter_counting/
├── llama_7b_config.json
├── deepseek_v3_config.json
├── backward_scaling_config.jsonc
└── backward_scaling_besiroglu.jsonc
```

### New Structure (Current)
```
flops_parameter_counting/
└── configs/
    ├── models/           # Model architecture configs
    └── scaling_laws/     # Scaling law configs
        ├── hoffmann/
        ├── besiroglu/
        └── custom/
```

### Migration

All old config files have been moved:
- Model configs → `configs/models/`
- Scaling law configs → `configs/scaling_laws/{hoffmann,besiroglu,custom}/`

**Old commands still work** due to backward compatibility!

## 📝 Adding New Configs

### Add a New Model Config

```bash
# Create file in configs/models/
cat > configs/models/my_model.json << 'EOF'
{
  "hidden_size": 2304,
  "num_hidden_layers": 18,
  "num_attention_heads": 18,
  "vocab_size": 32000,
  ...
}
EOF

# Use it
python detailed_cost_analysis.py --model_config my_model.json
```

### Add a New Scaling Law Config

```bash
# Create file in configs/scaling_laws/custom/
cat > configs/scaling_laws/custom/my_experiment.jsonc << 'EOF'
{
  "architecture": { ... },
  "training_gear": { ... },
  ...
}
EOF

# Use it
python detailed_cost_analysis.py --backward_config my_experiment.jsonc
```

## 📚 Documentation

- **`configs/models/README.md`** - Model config documentation
- **`configs/scaling_laws/README.md`** - Scaling law documentation
- **`README.md`** - Main tool documentation
- **`docs/`** - Detailed academic formulas and comparisons

## 🔍 Help

```bash
# Show usage and examples
python detailed_cost_analysis.py --help

# Run validation tests
python detailed_cost_analysis.py --validate
```

## ✅ Verification Checklist

To verify your LLaMA 1.36B model matches the scaling law optimization:

```bash
# 1. Check N (forward analysis)
python detailed_cost_analysis.py --model_config llama_1.36b.json | grep "Model parameters"
# Expected: 1.29B

# 2. Check N and D (backward analysis)
python detailed_cost_analysis.py --backward_config verify_llama_1.36b.jsonc | grep -E "Model parameters|Optimal tokens"
# Expected: N=1.29B, D=84.72B

# 3. Verify it matches your JSON
cat ../enhanced_training_system/info/llama_1.36e21_32kV.json | grep optimal_n_d
# Should show: [1.294e+09, 8.472e+10]
```

All values should match! ✨

