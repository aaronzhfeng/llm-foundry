"""
Prepare SlimPajama-6B dataset with GPT-2 tokenizer (50K vocab).
Downloads from HuggingFace and tokenizes with tiktoken GPT-2 BPE.
Creates train.bin and val.bin for fast memory-mapped training.

Dataset: https://huggingface.co/datasets/DKYoon/SlimPajama-6B
Tokenizer: tiktoken GPT-2 BPE (50304 vocab)

Time estimate: 20-40 minutes on 8 cores
Disk space: ~6GB for raw data + ~6GB for tokenized data
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Configuration
num_proc = 8  # Number of CPU cores for parallel processing
num_proc_load_dataset = num_proc

# Initialize GPT-2 tokenizer
print("Loading tiktoken GPT-2 tokenizer...")
try:
    enc = tiktoken.get_encoding("gpt2")
    print(f"✓ GPT-2 tokenizer loaded (vocab_size = {enc.n_vocab})")
except Exception as e:
    print(f"❌ Error loading tiktoken: {e}")
    print("\nInstall tiktoken: pip install tiktoken")
    exit(1)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("SlimPajama-6B Dataset Preparation (GPT-2 Tokenizer)")
    print("="*80)
    
    # Load the dataset
    print("\n[1/4] Loading SlimPajama-6B from HuggingFace...")
    print("      This will download ~6GB and may take 5-15 minutes...")
    try:
        dataset = load_dataset("DKYoon/SlimPajama-6B", num_proc=num_proc_load_dataset)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install datasets: pip install datasets")
        print("3. For SSH blocked HF: download locally first, then upload")
        exit(1)
    
    print(f"✓ Dataset loaded: {dataset}")
    
    # Check dataset structure and create train/val split
    print("\n[2/4] Creating train/validation split...")
    if 'train' in dataset and 'validation' not in dataset:
        print("      Splitting train set (99.5% train, 0.5% validation)...")
        split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
    elif 'train' in dataset and 'validation' in dataset:
        print("      Using existing splits...")
        split_dataset = {
            'train': dataset['train'],
            'val': dataset['validation']
        }
    else:
        print("      Dataset structure unknown, creating 99.5/0.5 split...")
        split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
    
    print(f"✓ Split complete:")
    for split_name, split_data in split_dataset.items():
        print(f"    {split_name}: {len(split_data):,} examples")
    
    # Tokenization function
    def process(example):
        """Tokenize a single example with GPT-2 BPE."""
        text = example['text']
        ids = enc.encode_ordinary(text)  # encode_ordinary ignores special tokens
        ids.append(enc.eot_token)  # Add end-of-text token (50256 for GPT-2)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # Tokenize the dataset
    print("\n[3/4] Tokenizing with GPT-2 BPE (50K vocab)...")
    print("      This may take 10-20 minutes...")
    tokenized = split_dataset.map(
        process,
        remove_columns=list(split_dataset['train'].features.keys()),
        desc="Tokenizing",
        num_proc=num_proc,
    )
    
    # Calculate total tokens
    total_train_tokens = sum(tokenized['train']['len'])
    total_val_tokens = sum(tokenized['val']['len'])
    print(f"✓ Tokenization complete:")
    print(f"    Train: {total_train_tokens:,} tokens")
    print(f"    Val: {total_val_tokens:,} tokens")
    
    # Write to binary files
    print("\n[4/4] Writing binary files...")
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # GPT-2 vocab (50256) < 2^16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"  Writing {split}.bin ({arr_len:,} tokens)...")
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'  Progress'):
            # Batch samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"  ✓ {split}.bin written ({os.path.getsize(filename) / 1e9:.2f} GB)")
    
    # Save metadata
    print("\n  Writing meta.pkl...")
    meta = {
        'vocab_size': enc.n_vocab,  # Should be 50257 or 50304
        'tokenizer': 'gpt2_bpe',
        'eot_token': enc.eot_token,
    }
    import pickle
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"  ✓ meta.pkl written (vocab_size = {meta['vocab_size']})")
    
    print("\n" + "="*80)
    print("✅ Dataset preparation complete!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - train.bin: {total_train_tokens:,} tokens")
    print(f"  - val.bin: {total_val_tokens:,} tokens")
    print(f"  - meta.pkl: vocab_size = {meta['vocab_size']}")
    print(f"\nNext steps:")
    print(f"  1. Verify files exist: ls -lh {os.path.dirname(__file__)}/*.bin")
    print(f"  2. Update config: dataset = 'slimpajama_6b_gpt2'")
    print(f"  3. Start training: python train.py config/full_gpt2_1.36b.py")

