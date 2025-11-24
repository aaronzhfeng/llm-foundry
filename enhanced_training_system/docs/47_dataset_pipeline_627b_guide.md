# 47. SlimPajama-627B Data Pipeline Guide

This document consolidates the staged workflow we adopted to process the full 627B-token SlimPajama dump with Qwen3 tokenization. The emphasis is on parallelism, resumability, and observability.

## Goals
- Separate network-bound mirroring from CPU-bound tokenization.
- Provide deterministic manifests so multiple workers can resume without coordination.
- Surface validation hooks (`meta.pkl`, eye tests, disk usage) that catch corrupt shards early.

## Stage 0 – Prerequisites
- HF cache resides under `/raid/zhf004/huggingface_cache`.
- Qwen3 tokenizer assets available locally (see `data/slimpajama_627b_qwen3/README.md`).
- `tokenize_from_manifest.py` dependencies installed (`transformers`, `zstandard`, etc.).

## Stage 1 – Mirror
1. Use `huggingface-cli` or `aria2c` to mirror `cerebras/SlimPajama-627B` into the cache.
2. Verify presence of `snapshots/<commit>/.../*.jsonl.zst`.
3. Record disk usage:
   ```bash
   du -sh /raid/zhf004/huggingface_cache/hub/datasets--cerebras--SlimPajama-627B
   ```
   Expect multiple terabytes; keep at least 20% free headroom for tokenized bins.

## Stage 2 – Build Manifest
`build_manifest.py` walks the mirrored snapshots and writes:
- `manifest.jsonl` – one entry per `.jsonl.zst` shard with split, chunk index, size, absolute path.
- `manifest_summary.json` – aggregate counts per split for quick spot checks.

Example command:
```bash
python data/slimpajama_627b_qwen3/build_manifest.py \
  --dataset-dir /raid/zhf004/huggingface_cache/hub/datasets--cerebras--SlimPajama-627B \
  --output-dir data/slimpajama_627b_qwen3
```

**Validation:**
- Ensure manifest count matches HF metadata (`wc -l manifest.jsonl`).
- Spot-check a few entries to confirm absolute paths resolve.

## Stage 3 – Tokenize from Manifest
`tokenize_from_manifest.py` assigns shards to workers and writes `<split>/worker_<id>_<shard>.bin` incrementally (uint32 IDs).

### Auto-spawned mode
```bash
python data/slimpajama_627b_qwen3/tokenize_from_manifest.py \
  --manifest data/slimpajama_627b_qwen3/manifest.jsonl \
  --output-root /raid/zhf004/slimpajama_627b_qwen3_bins \
  --spawn-workers auto \
  --max-workers $(nproc) \
  --tokens-per-write 131072
```

### Manual MPI-style mode
Launch one process per node:
```bash
python tokenize_from_manifest.py ... --process-count 8 --process-index 0
# repeat for index 1..7
```

**Key behaviors:**
- Every worker writes progress metadata and flushes `.bin` chunks frequently to avoid losing work.
- Zstd reader is wrapped in `io.TextIOWrapper`, so each line is decoded lazily.
- Errors skip the shard but log the offending file for manual retry.

## Stage 4 – Combine and Split
After all worker bins finish:
1. Concatenate per split into `train.bin`, `val.bin`, `test.bin` using `cat` or Python to preserve ordering.
2. Write `meta.pkl`:
   ```python
   meta = {
       'tokenizer': 'qwen3',
       'vocab_size': 151669,
       'eos_token_id': 151645,
       'dtype': 'uint32',
   }
   ```
3. Run `du -h --max-depth=1` inside the dataset directory to confirm expected sizes (~2.3 TB train).

## Stage 5 – Validation Hooks
- **Eye test:** Decode random offsets (both near 0 and deep into the file) with the Qwen tokenizer to ensure entropy.
- **Alignment test:** Confirm memmap slicing obeys the `shift-by-one` label rule:
  ```python
  x = data[start:start+block]
  y = data[start+1:start+1+block]
  assert (x[1:] == y[:-1]).all()
  ```
- **Sampler dry-run:** Use the smoke config to confirm the DataLoader sees random samples and loss starts near 9–11.

## Stage 6 – Monitoring and Resume
- Use `mpstat`/`htop` plus script logs to ensure CPU saturation.
- If a worker crashes, relaunch with the same `--process-index`; completed shards are skipped via manifest bookkeeping.
- Track ETA via provided `@python3` progress helper (tokens processed vs total).

## FAQ
- **Does tokenization consume idle CPUs?** Workers only run on processes you spawn; they do not steal from unrelated jobs.
- **How to monitor `.bin` growth?** `watch -n 60 'du -h --max-depth=1 /raid/.../train'`.
- **Validation/test splits?** Run the same pipeline but point `--splits val,test` or filter the manifest entries before tokenization.

For full CLI options and examples, see `data/slimpajama_627b_qwen3/README.md`.

