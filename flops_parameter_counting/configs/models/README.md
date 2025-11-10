# Model Architecture Configs

This directory contains model architecture configurations for **forward analysis** (calculating N, FLOPs, memory from architecture).

## File Format

Standard HuggingFace-style JSON configs with architecture parameters.

## Files

### Production Models

- **`llama_1.36b.json`** - LLaMA 1.36B production model
  - From: `enhanced_training_system/info/llama_1.36e21_32kV.json`
  - Parameters: ~1.29B
  - Use: `python detailed_cost_analysis.py --model_config configs/models/llama_1.36b.json`

- **`gpt2_1.36b.json`** - GPT-2 1.36B for comparison
  - From: `enhanced_training_system/config/full_gpt2_1.36b.py`
  - Parameters: ~1.41B
  - Use: `python detailed_cost_analysis.py --model_config configs/models/gpt2_1.36b.json`

### Reference Models

- **`llama_7b.json`** - Original LLaMA 7B configuration
- **`deepseek_v3.json`** - DeepSeek V3 MoE model
- **`our_moe_config.json`** - Custom MoE configuration

### Examples

- **`example_llama_config.jsonc`** - Annotated LLaMA config with comments
- **`example_model_config.jsonc`** - General model config template

## Usage

### Forward Analysis (Get N from Architecture)

```bash
# Analyze LLaMA 1.36B
python detailed_cost_analysis.py --model_config configs/models/llama_1.36b.json

# Analyze GPT-2 1.36B
python detailed_cost_analysis.py --model_config configs/models/gpt2_1.36b.json

# Compare both
python detailed_cost_analysis.py \
  --model_config configs/models/llama_1.36b.json \
  --model_config configs/models/gpt2_1.36b.json
```

### Expected Output

```
Model Architecture Analysis
===========================

Model: LLaMA 1.36B
  Hidden size: 2304
  Layers: 18
  Heads: 18
  Vocab: 32000
  
Parameters:
  Total: 1.294B
  Embeddings: 147.5M (input + output, no tying)
  Transformer: 1.147B
  Position: 0 (RoPE)
  
FLOPs per Token (forward):
  Total: 42.5 GFLOPs
  Attention: 21.2 GFLOPs (50%)
  FFN: 21.3 GFLOPs (50%)
```

## Adding New Models

Create a new JSON file with required fields:

```json
{
  "hidden_size": 2304,
  "intermediate_size": 6144,
  "num_hidden_layers": 18,
  "num_attention_heads": 18,
  "vocab_size": 32000,
  "max_position_embeddings": 2048,
  "tie_word_embeddings": false
}
```

For detailed parameter explanations, see `example_model_config.jsonc`.

## Related

- **Scaling Laws**: See `../scaling_laws/` for backward analysis (finding optimal N and D)
- **Training Configs**: See `enhanced_training_system/config/` for actual training configurations

