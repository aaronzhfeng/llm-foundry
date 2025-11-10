# SlimPajama-6B Dataset (GPT-2 Tokenizer)

## Overview

This directory contains the SlimPajama-6B dataset tokenized with **GPT-2 BPE tokenizer (50K vocab)**.

- **Source**: https://huggingface.co/datasets/DKYoon/SlimPajama-6B
- **Tokenizer**: tiktoken GPT-2 BPE (50,304 tokens)
- **Size**: ~6B tokens tokenized, ~6GB binary files
- **Use**: Training GPT-2 1.36B model with correct vocabulary

## Files

After running `prepare.py`, you should have:

```
slimpajama_6b_gpt2/
├── prepare.py           # Preparation script
├── train.bin            # Training data (~6GB)
├── val.bin              # Validation data (~30MB)
├── meta.pkl             # Metadata (vocab_size, tokenizer info)
└── README.md            # This file
```

## Preparation

### Prerequisites

```bash
pip install torch tiktoken datasets numpy tqdm
```

### Run Preparation

```bash
cd data/slimpajama_6b_gpt2
python prepare.py
```

**Time:** 20-40 minutes (depending on internet speed and CPU cores)

**Disk space:** ~12GB total (6GB raw + 6GB tokenized)

## Verification

```bash
# Check files created
ls -lh

# Expected output:
#   train.bin    ~6.0GB
#   val.bin      ~30MB
#   meta.pkl     ~1KB

# Verify vocab size
python << 'EOF'
import pickle
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
print(f"Vocab size: {meta['vocab_size']}")  # Should be 50257 or 50304
print(f"Tokenizer: {meta['tokenizer']}")    # Should be 'gpt2_bpe'
EOF
```

## Usage

### In Training Config

```python
# config/full_gpt2_1.36b.py
dataset = 'slimpajama_6b_gpt2'  # References this directory
```

### Start Training

```bash
# Single GPU
python train.py config/full_gpt2_1.36b.py

# Multi-GPU (4× H20)
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_1.36b.py
```

## Important Notes

1. **Vocab size must match model**: 50,304 (GPT-2 standard, rounded)
2. **Different from LLaMA**: Uses different tokenizer (not interchangeable)
3. **Same text, different tokens**: Same SlimPajama text, tokenized differently
4. **Fair comparison**: Each model uses its designed tokenizer

## Token Count Difference

Due to different tokenizers, the same text produces different token counts:

| Tokenizer | Vocab Size | Tokens (estimate) | Efficiency |
|-----------|------------|-------------------|------------|
| LLaMA-2 | 32,000 | ~6B | Baseline |
| GPT-2 BPE | 50,304 | ~7-8B | Less efficient |

**Note:** GPT-2 may produce ~15-20% more tokens for the same text (less compression). This is expected and accounted for in training.

## Troubleshooting

### tiktoken not installed

**Problem:**
```
ModuleNotFoundError: No module named 'tiktoken'
```

**Solution:**
```bash
pip install tiktoken
```

### HuggingFace dataset download fails

**Problem:**
```
ConnectionError: Couldn't reach HuggingFace
```

**Solutions:**
1. Check internet connection
2. If on SSH with blocked HF:
   - Download dataset locally first
   - Upload prepared .bin files to server
   - Or use HF mirror (if available in China)

### Memory error during preparation

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Edit prepare.py, reduce num_proc:
num_proc = 4  # Instead of 8
```

## Related

- **LLaMA version**: See `../slimpajama_6b_llama/` for LLaMA tokenized version
- **Full dataset**: SlimPajama-627B (~895GB) for optimal training
- **Training guide**: See `../../TRAINING_GUIDE.md` for complete workflow

