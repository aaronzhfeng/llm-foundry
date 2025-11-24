# 46. LLM Training Incident Log (Qwen3-1.8B on B200)

This playbook captures the engineering incidents we hit while building the 627B-token training stack. Each entry summarizes the trigger, how we detected it, and the fix that stuck. Use it to shortcut future investigations.

## Incident Timeline

| # | Date (approx) | Symptom | Root Cause | Fix/Reference |
|---|---------------|---------|------------|---------------|
| 1 | 2025-11-18 | Loss collapses to ~0 instantly (“Phantom Zero”) | `uint16` dataset despite 150k vocab, high bits truncated | Regenerated bins as `uint32`, wrote `meta.pkl` with dtype |
| 2 | 2025-11-19 | Loss 2.8 → 1.5 in <300 iters | DataLoader sliding window (no shuffle) | Added `RandomIndexSampler` chunked sampler |
| 3 | 2025-11-19 | CUDA OOM during `torch.compile` | Fragmented allocator and optimizer redundancy | Enabled ZeRO-1, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 4 | 2025-11-20 | `ModuleNotFoundError: wandb` in smoke test | Environment missing optional dep | Documented `pip install wandb` or `wandb_log=False` |
| 5 | 2025-11-20 | `CUDA error: device-side assert triggered` | Model built with GPT-2 vocab (50304) while data used 151669 | Forced configs + dataset metadata to share vocab |
| 6 | 2025-11-21 | `io.UnsupportedOperation` in tokenizer workers | zstd stream reader lacked text wrapper | Wrapped with `io.TextIOWrapper`, verified resumability |
| 7 | 2025-11-21 | Manifest builder missing snapshots | Defaulted to wrong HF cache root | Added explicit `--dataset-dir` guidance |

## Detailed Notes

### 1. Phantom Zero (`uint16` bins)
- **Detection:** `np.memmap` slices showed many tokens >65535 becoming 0; decoded text looked truncated.
- **Fix:** Re-tokenized SlimPajama shards to `uint32`; added `meta.pkl` with `dtype`, `vocab_size`, and `tokenizer` so `train.py` auto-selects the correct dtype.
- **Lesson:** Never assume GPT-2 vocab size—derive from tokenizer or metadata.

### 2. Sliding Window Leakage
- **Detection:** Eye test of consecutive samples revealed 2047-token overlap; loss curve dropped unrealistically fast.
- **Fix:** Replaced `shuffle=False` with a chunked `RandomIndexSampler` that samples 64k-sized windows to avoid `torch.randperm` OOM.
- **Lesson:** “Preface sampling” is as dangerous as bad labels; randomness must be enforced at file offsets, not just within tensors.

### 3. Torch Compile OOM
- **Detection:** Backward pass tried to allocate ~14 GiB although 11 GiB was free; `torch.compile` + BF16 spikes fragmentation.
- **Fix:** Set `os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'` before CUDA init and enabled ZeRO-1 in configs where optimizer state > GPU headroom.
- **Lesson:** Always document allocator knobs next to config toggles; without both, B200 memory reports look inconsistent.

### 4. `wandb` Missing
- **Detection:** Smoke test aborted during import.
- **Fix:** Documented two sanctioned paths: install `wandb` globally or set `wandb_log = False` in config. Added reminder to smoke-test after environment rebuilds.
- **Lesson:** Optional integrations should fail closed with config flag guidance.

### 5. Vocab Mismatch
- **Detection:** Device-side assert complaining `tmp4 < 50304`; dataset logs printed `vocab_size=151669`.
- **Fix:** Configs now hardcode `vocab_size = 151669`, and `train.py` trusts `meta.pkl` when present. Added logging to show dtype/vocab detection at startup.
- **Lesson:** Small models with huge vocabs are sensitive; log the effective vocab before constructing embeddings.

### 6. Tokenizer Worker Crash
- **Detection:** Parallel tokenization emitted `io.UnsupportedOperation` from `zstd.ZstdDecompressor.stream_reader`.
- **Fix:** Wrap the stream reader with `io.TextIOWrapper` so it behaves like a file object; verified each worker writes `.bin` incrementally to avoid all-or-nothing failures.
- **Lesson:** For multi-day tokenization, ensure every worker emits partial output frequently to survive crashes.

### 7. Manifest Path Confusion
- **Detection:** `FileNotFoundError` pointing to relative HF cache path.
- **Fix:** CLI now accepts `--dataset-dir` and README examples use absolute `/raid/zhf004/huggingface_cache/...` paths.
- **Lesson:** Large mirrors rarely live inside repo roots; always allow overrides.

## Reuse Checklist
1. Verify `meta.pkl` (dtype, vocab, eos) before launching any job.
2. Run the “eye test” script on bytes [0:10MB] of `train.bin` before trusting a loss curve.
3. Run the smoke config after any loader/tokenizer change; it exercises eval + checkpoint + W&B.
4. Enable ZeRO-1 plus expandable segments on B200 unless explicitly profiled otherwise.
5. Keep this file updated—append new incidents, do not overwrite history.

