# SlimPajama-6B Dataset (LLaMA-2 Tokenizer)

## Dataset Source
- **Original Dataset**: [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- **Subset**: First ~12M samples (≈6B tokens)

## Tokenizer
- **Model**: [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Vocabulary Size**: 32,000 tokens
- **Type**: SentencePiece BPE
- **Note**: Requires HuggingFace authentication

## Preparation

### 1. Download tokenizer (first time only)
```bash
cd /root/llm_TII/enhanced_training_system

# Login to HuggingFace (if needed)
huggingface-cli login

# Download tokenizer
python << 'EOF'
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("./llama2_tokenizer")
print(f"✓ Saved to ./llama2_tokenizer/ (vocab={tokenizer.vocab_size})")
EOF
```

### 2. Run preparation script
```bash
cd data/slimpajama_6b_llama
python prepare.py
```

### 3. Expected output
- `train.bin`: ~6GB tokenized training data
- `val.bin`: ~30MB tokenized validation data
- `meta.pkl`: Metadata (vocab_size=32000, etc.)

### Time: ~25-35 minutes

## Usage
This dataset is used by LLaMA-2 model configs:
- `config/full_llama_1.36b.py`

## Technical Details
- **Tokens per sample**: ~500 (average)
- **Train/Val split**: 99% / 1%
- **Encoding**: uint16 (vocab < 65536)
- **Format**: Memory-mapped numpy arrays (.bin files)
- **Architecture**: LLaMA-2 style (RoPE, RMSNorm, SwiGLU, MHA)

