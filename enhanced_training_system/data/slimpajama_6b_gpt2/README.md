# SlimPajama-6B Dataset (GPT-2 Tokenizer)

## Dataset Source
- **Original Dataset**: [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- **Subset**: First ~12M samples (≈6B tokens)

## Tokenizer
- **Model**: GPT-2 (tiktoken)
- **Vocabulary Size**: 50,257 tokens
- **Source**: OpenAI's GPT-2 tokenizer (BPE)

## Preparation

### 1. No tokenizer download needed
GPT-2 tokenizer is included with tiktoken/transformers.

### 2. Run preparation script
```bash
cd /root/llm_TII/enhanced_training_system/data/slimpajama_6b_gpt2
python prepare.py
```

### 3. Expected output
- `train.bin`: ~6GB tokenized training data
- `val.bin`: ~30MB tokenized validation data
- `meta.pkl`: Metadata (vocab_size, etc.)

### Time: ~20-30 minutes

## Usage
This dataset is used by GPT-2 model configs:
- `config/full_gpt2_1.36b.py`

## Technical Details
- **Tokens per sample**: ~500 (average)
- **Train/Val split**: 99% / 1%
- **Encoding**: uint16 (vocab < 65536)
- **Format**: Memory-mapped numpy arrays (.bin files)

