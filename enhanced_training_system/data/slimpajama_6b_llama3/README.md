# SlimPajama-6B Dataset (LLaMA-3 Tokenizer)

## Dataset Source
- **Original Dataset**: [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- **Subset**: First ~12M samples (≈6B tokens)

## Tokenizer
- **Model**: [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- **Vocabulary Size**: 128,256 tokens
- **Type**: Tiktoken-based BPE (GPT-4 style)
- **Extended**: 4× larger vocabulary than LLaMA-2
- **Note**: Requires HuggingFace authentication

## Preparation

### 1. Download tokenizer (first time only)
```bash
cd /root/llm_TII/enhanced_training_system

# Login to HuggingFace (if needed)
huggingface-cli login

# Download tokenizer
python << 'EOF'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.save_pretrained("./llama3_tokenizer")
print(f"✓ Saved to ./llama3_tokenizer/ (vocab={tokenizer.vocab_size})")
EOF
```

### 2. Run preparation script
```bash
cd data/slimpajama_6b_llama3
python prepare.py
```

### 3. Expected output
- `train.bin`: ~6GB tokenized training data
- `val.bin`: ~30MB tokenized validation data
- `meta.pkl`: Metadata (vocab_size=128256, etc.)

### Time: ~30-40 minutes (larger vocab = more processing)

## Usage
This dataset is used by LLaMA-3 model configs:
- `config/full_llama3_1.5b_optimal.py` ⭐ Grid search optimized
- `config/full_llama3_2.2b_chinchilla.py` ⭐ Grid search optimized
- `config/full_llama3_8b.py` (Official Meta architecture)

## Technical Details
- **Tokens per sample**: ~500 (average, but better compression than LLaMA-2)
- **Train/Val split**: 99% / 1%
- **Encoding**: uint32 (vocab > 65536, requires 4 bytes)
- **Format**: Memory-mapped numpy arrays (.bin files)
- **Architecture**: LLaMA-3 style (RoPE extended θ=500K, RMSNorm, SwiGLU 3.5×, GQA)

## Key Differences from LLaMA-2
- **4× larger vocabulary** (128K vs 32K)
- **Better compression** (~20% fewer tokens for same text)
- **Extended RoPE** (theta=500,000 vs 10,000)
- **Grouped Query Attention** (8 KV heads vs MHA)
- **SwiGLU 3.5×** expansion (vs 8/3× ≈ 2.67×)

