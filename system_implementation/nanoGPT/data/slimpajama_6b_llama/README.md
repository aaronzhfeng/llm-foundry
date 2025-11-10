# SlimPajama-6B Dataset (LLaMA-2 Tokenizer)

## Overview

This directory contains the SlimPajama-6B dataset tokenized with **LLaMA-2 tokenizer (32K vocab)**.

- **Source**: https://huggingface.co/datasets/DKYoon/SlimPajama-6B
- **Tokenizer**: meta-llama/Llama-2-7b-hf (32,000 tokens)
- **Size**: ~6B tokens tokenized, ~6GB binary files
- **Use**: Training LLaMA 1.36B model with correct vocabulary

## Files

After running `prepare.py`, you should have:

```
slimpajama_6b_llama/
├── prepare.py           # Preparation script
├── train.bin            # Training data (~6GB)
├── val.bin              # Validation data (~30MB)
├── meta.pkl             # Metadata (vocab_size, tokenizer info)
└── README.md            # This file
```

## Preparation

### Prerequisites

```bash
pip install torch transformers datasets numpy tqdm
huggingface-cli login  # May be required for LLaMA-2 tokenizer
```

### Run Preparation

```bash
cd data/slimpajama_6b_llama
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
print(f"Vocab size: {meta['vocab_size']}")  # Should be 32000
print(f"Tokenizer: {meta['tokenizer']}")    # Should be 'llama2'
EOF
```

## Usage

### In Training Config

```python
# config/full_llama_1.36b.py
dataset = 'slimpajama_6b_llama'  # References this directory
```

### Start Training

```bash
# Single GPU
python train.py config/full_llama_1.36b.py

# Multi-GPU (4× H20)
torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py
```

## Important Notes

1. **Vocab size must match model**: 32,000 (LLaMA-2 standard)
2. **Different from GPT-2**: Uses different tokenizer (not interchangeable)
3. **Same text, different tokens**: Same SlimPajama text, tokenized differently
4. **Fair comparison**: Each model uses its designed tokenizer

## Troubleshooting

### LLaMA-2 tokenizer access denied

**Problem:**
```
HfHubHTTPError: 401 Unauthorized
```

**Solution:**
1. Go to https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Accept the license agreement
3. Login: `huggingface-cli login`
4. Run prepare.py again

### HuggingFace blocked (e.g., on SSH server)

**Solution:** Download tokenizer locally first:

```python
# On local machine with HF access:
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("../../llama2_tokenizer")

# Then edit prepare.py line 19:
# FROM: tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# TO:   tokenizer = LlamaTokenizer.from_pretrained("../../llama2_tokenizer")
```

## Related

- **GPT-2 version**: See `../slimpajama_6b_gpt2/` for GPT-2 tokenized version
- **Full dataset**: SlimPajama-627B (~895GB) for optimal training
- **Training guide**: See `../../TRAINING_GUIDE.md` for complete workflow

