# 49. Training Runtime Best Practices (Qwen3-1.8B B200)

This checklist aggregates the hard lessons from smoke tests, MFU audits, and production rehearsals. Follow it before every long run.

## 1. Pre-Flight Checklist

- **Dataset Sanity**
  - `data/slimpajama_627b_qwen3/meta.pkl` exists with `dtype='uint32'`, `vocab_size=151669`.
  - `train.bin`, `val.bin`, `test.bin` sizes match manifest totals; run `du -h --max-depth=1`.
  - Run the “eye test” decode script on offsets near 0 GB and near the end.

- **Loader Configuration**
  - `RandomIndexSampler` enabled (or equivalent) to avoid sliding windows.
  - `token_dtype` determined from metadata, not config guesses.
  - `dataloader_num_workers >= 16`, `pin_memory=True`, `persistent_workers=True`.

- **Model/Optimizer**
  - Config sets `use_zero1=True` for 1.8B on B200 when gradient accumulation >1.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` exported before any CUDA call.
  - Checkpoint cadence defined (`keep_all_checkpoints=True` if retention required).

- **Environment**
  - `wandb` installed or `wandb_log=False`.
  - Enough disk for checkpoints: `eval_interval * checkpoint_size * duration`.
  - Verify `out_dir` is empty or intentionally continuing from `init_from`.

- **Smoke Test**
  - Run `config/full_qwen3_1.8b_b200_50h_smoke.py`.
  - Confirm it completes: evaluation, checkpoint write, logging, W&B init.

## 2. Launch Guidelines

- **Command Template**
  ```bash
  torchrun --standalone --nproc_per_node=8 train.py \
    config/full_qwen3_1.8b_b200_50h.py \
    --device=cuda --dtype=bfloat16 --compile=True
  ```
- **Logging**
  - `log_interval=10` gives per-minute visibility.
  - Redirect stdout to both terminal and file (`tee`) for postmortems.
  - Snapshot `git rev-parse HEAD` before launch.

- **Initial Monitoring**
  - Expect loss ~9–11 at iter 0; anything <5 requires immediate eye test.
  - Tokens/sec target: ~600–650k global on 8×B200; significant drops imply dataloader or NCCL stalls.
  - GPU memory: 170–175 GB usage with ZeRO-1; fragmentation creeping upward signals checkpoint accumulation issues.

## 3. Runtime Safety Nets

- **Checkpointing**
  - `keep_all_checkpoints=True` now writes `ckpt_<iter>.pt` plus `ckpt.pt` symlink.
  - Estimate disk needs (e.g., 1.5 GB * number of eval intervals).

- **Evaluation Cadence**
  - Use `num_datapoints / (batch_size * ddp_world_size)` heuristic if you want one eval per data sweep; for 627B/ (24*8)=~3.3M steps, adjust as needed.
  - For 50h run we target `eval_interval=20000`.

- **Alerts**
  - Monitor W&B for loss spikes; set alert if loss <1.5 before 10k iterations (likely data leak).
  - If throughput drops >20% for >5 minutes, capture profiler trace before killing job.

## 4. Post-Run

- Validate final checkpoint by running the smoke config in `eval_only` mode.
- Archive `run_YYYYMMDD_HHMMSS.json` logs alongside config and git SHA.
- Summarize MFU version used (`calculation_method` field) in experiment tracker.

## 5. Quick Commands Reference

- **Eye test script:** See `docs/47_dataset_pipeline_627b_guide.md`.
- **Random sampler snippet:** Located in `train.py` around the DataLoader definition.
- **Allocator export:**
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

Adhering to this runbook catches the issues we already hit (dtype mismatch, data leakage, wandb missing) before they torpedo a 50-hour run.

