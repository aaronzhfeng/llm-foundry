# 50. Troubleshooting Quick Reference

Use this sheet when something looks wrong during data prep or training. It links symptoms to the fastest checks and known fixes.

| Symptom | Likely Cause | Quick Check | Fix |
|---------|--------------|-------------|-----|
| Loss < 2 within first 500 iterations | Sliding window / easy “preface” data | Decode consecutive batches, inspect offsets near 0 GB | Ensure `RandomIndexSampler` active; reshuffle bins if needed |
| Loss ≈ 0 instantly | `uint16` truncation (“Phantom Zero”) | `np.memmap` sample contains many zeros; `meta.pkl` missing dtype | Regenerate bins as `uint32`, write metadata |
| `torch.OutOfMemoryError` during backward with free memory reported | CUDA allocator fragmentation | `env | grep PYTORCH_CUDA_ALLOC_CONF` | Export `expandable_segments:True`; enable ZeRO-1 |
| `ModuleNotFoundError: wandb` | Env missing dependency | `pip show wandb` | Install `wandb` or set `wandb_log=False` |
| `CUDA error: device-side assert triggered` mentioning vocab | Model embedding smaller than tokenizer IDs | Print `vocab_size` from config vs `meta.pkl` | Set config vocab to 151669; ensure dtype uint32 |
| Tokenizer worker `io.UnsupportedOperation` | zstd reader not wrapped | Check stack trace for `stream_reader` | Upgrade script (already wrapped) or re-run latest version |
| Manifest builder `FileNotFoundError` | Wrong HF cache path | Confirm `--dataset-dir` | Pass absolute cache path `/raid/zhf004/...` |
| Throughput < 400k tokens/s | Dataloader or NCCL issue | Watch CPU utilization; check `nvidia-smi dmon` | Increase workers, verify pinning, restart NCCL job |
| Checkpoints not rotating | `keep_all_checkpoints` disabled | Inspect `out_dir` contents | Set `keep_all_checkpoints=True`; confirm symlink creation |
| Serving assets incomplete | Missing tokenizer/config alongside checkpoint | Ensure `meta.pkl`, tokenizer files, config commit saved | Package checkpoint + config + tokenizer for deployment |

## Quick Commands
- **Eye test:** `python scripts/decode_sample.py --offset 0 --count 256`
- **Disk usage:** `du -h --max-depth=1 data/slimpajama_627b_qwen3`
- **CPU stats:** `mpstat 5`
- **Sampler sanity:** Run smoke config and inspect logs for random offsets.

Keep this table updated whenever a new failure mode is diagnosed. Link each row back to the full incident entry in `docs/46_llm_training_incidents.md`.

