# Training Logging Guide

## Overview

All three phases now include automatic JSON logging that saves complete training history to structured JSON files. One file is created per training run with a unique timestamp.

## Features

✅ **Automatic logging** of all training metrics  
✅ **One JSON file per run** with timestamp  
✅ **Complete history** preserved (iterations, evaluations, checkpoints)  
✅ **Easy analysis** with standard JSON tools  
✅ **Optional and non-intrusive** (enabled by default)

## File Structure

After training, you'll find JSON logs in your output directory:

```
out/
├── ckpt.pt
├── run_20231031_162345.json  ← Training log
└── run_20231031_171523.json  ← Another run
```

## JSON Log Contents

Each log file contains:

```json
{
  "run_name": "run_20231031_162345",
  "start_time": "2023-10-31T16:23:45.123456",
  "end_time": "2023-10-31T16:45:12.789012",
  
  "config": {
    "learning_rate": 0.0006,
    "batch_size": 12,
    "max_iters": 100,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    ...
  },
  
  "training_iterations": [
    {"iter": 0, "loss": 11.0049, "time_ms": 19130.8, "mfu": -100.0},
    {"iter": 1, "loss": 10.9733, "time_ms": 3978.64, "mfu": -100.0},
    ...
  ],
  
  "eval_steps": [
    {"iter": 0, "train_loss": 11.004, "val_loss": 10.9977, "lr": 0.0006, "timestamp": "..."},
    {"iter": 50, "train_loss": 8.358, "val_loss": 8.2892, "lr": 0.0006, "timestamp": "..."},
    ...
  ],
  
  "checkpoints": [
    {"iter": 50, "val_loss": 8.2892, "path": "out/ckpt.pt", "timestamp": "..."},
    {"iter": 100, "val_loss": 7.1189, "path": "out/ckpt.pt", "timestamp": "..."}
  ],
  
  "metadata": {
    "world_size": 1,
    "device": "cuda",
    "dtype": "bfloat16",
    "compile": false,
    "use_zero1": false
  },
  
  "summary": {
    "total_iterations": 101,
    "final_iter": 100,
    "final_train_loss": 6.9127,
    "best_val_loss": 7.1189,
    "avg_time_ms": 4523.45,
    "avg_mfu": 32.87,
    "total_eval_steps": 3,
    "total_checkpoints": 2
  }
}
```

## Usage

### Enable/Disable Logging

Logging is **enabled by default**. To disable:

```bash
python train.py --save_log_to_json=False
```

### Custom Run Name

```bash
python train.py --wandb_run_name=my_experiment
# Will create: out/my_experiment_20231031_162345.json
```

### Change Save Interval

By default, logs are auto-saved every 100 iterations:

```bash
python train.py --log_save_interval=50  # Save every 50 iterations
```

## Analyzing Logs

### Load and View in Python

```python
import json

# Load a log file
with open('out/run_20231031_162345.json', 'r') as f:
    log = json.load(f)

# View summary
print(log['summary'])

# Plot loss curve
import matplotlib.pyplot as plt

iters = [x['iter'] for x in log['training_iterations']]
losses = [x['loss'] for x in log['training_iterations']]

plt.plot(iters, losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
```

### Compare Multiple Runs

```python
from training_logger import compare_runs

# Compare multiple runs
comparison = compare_runs([
    'out/run_20231031_162345.json',
    'out/run_20231031_171523.json'
])

for run in comparison['runs']:
    print(f"{run['run_name']}: final_loss={run['final_loss']:.4f}, "
          f"best_val={run['best_val_loss']:.4f}, mfu={run['avg_mfu']:.2f}%")
```

### Extract Specific Metrics

```python
import json
import pandas as pd

# Load log
with open('out/run_20231031_162345.json', 'r') as f:
    log = json.load(f)

# Convert to DataFrame for easy analysis
df = pd.DataFrame(log['training_iterations'])

# Statistics
print(f"Mean loss: {df['loss'].mean():.4f}")
print(f"Min loss: {df['loss'].min():.4f}")
print(f"Mean time: {df['time_ms'].mean():.2f} ms")
print(f"Mean MFU: {df[df['mfu'] > 0]['mfu'].mean():.2f}%")

# Export to CSV
df.to_csv('training_metrics.csv', index=False)
```

## Quick Examples

### Example 1: Plot Training Progress

```python
import json
import matplotlib.pyplot as plt

with open('out/run_20231031_162345.json', 'r') as f:
    log = json.load(f)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Loss over time
iters = [x['iter'] for x in log['training_iterations']]
losses = [x['loss'] for x in log['training_iterations']]
ax1.plot(iters, losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

# Time per iteration
times = [x['time_ms'] for x in log['training_iterations'][5:]]  # Skip warmup
ax2.hist(times, bins=30)
ax2.set_title('Time Distribution')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Frequency')

# MFU over time
mfus = [x['mfu'] for x in log['training_iterations'] if x['mfu'] > 0]
ax3.plot(mfus)
ax3.set_title('Model FLOPs Utilization')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('MFU (%)')

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=150)
print("Saved to training_analysis.png")
```

### Example 2: Compare Training and Validation Loss

```python
import json
import matplotlib.pyplot as plt

with open('out/run_20231031_162345.json', 'r') as f:
    log = json.load(f)

eval_iters = [x['iter'] for x in log['eval_steps']]
train_losses = [x['train_loss'] for x in log['eval_steps']]
val_losses = [x['val_loss'] for x in log['eval_steps']]

plt.figure(figsize=(10, 6))
plt.plot(eval_iters, train_losses, label='Train Loss', marker='o')
plt.plot(eval_iters, val_losses, label='Val Loss', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('train_val_comparison.png', dpi=150)
```

### Example 3: Export Summary Table

```python
import json
import pandas as pd

# Load multiple logs
logs = [
    'out/run_baseline.json',
    'out/run_zero1.json',
    'out/run_fsdp.json'
]

summaries = []
for log_file in logs:
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    summaries.append({
        'Run': data['run_name'],
        'Final Train Loss': data['summary']['final_train_loss'],
        'Best Val Loss': data['summary']['best_val_loss'],
        'Avg Time (ms)': data['summary']['avg_time_ms'],
        'Avg MFU (%)': data['summary']['avg_mfu'],
        'Total Iters': data['summary']['total_iterations']
    })

df = pd.DataFrame(summaries)
print(df.to_markdown(index=False))

# Save to CSV
df.to_csv('run_comparison.csv', index=False)
```

## Command Line Tips

### View Log Without Python

```bash
# Pretty-print JSON
cat out/run_20231031_162345.json | jq '.'

# View summary only
cat out/run_20231031_162345.json | jq '.summary'

# Extract final loss
cat out/run_20231031_162345.json | jq '.summary.final_train_loss'

# Count iterations
cat out/run_20231031_162345.json | jq '.training_iterations | length'
```

### List All Logs

```bash
ls -lht out/*.json
```

### Find Best Run

```bash
for f in out/*.json; do
    loss=$(jq -r '.summary.best_val_loss' "$f")
    echo "$f: $loss"
done | sort -t: -k2 -n | head -1
```

## Integration with Other Tools

### TensorBoard (Future)

The JSON logs can be converted to TensorBoard format:

```python
from torch.utils.tensorboard import SummaryWriter
import json

with open('out/run_20231031_162345.json', 'r') as f:
    log = json.load(f)

writer = SummaryWriter('runs/experiment_1')

for data in log['training_iterations']:
    writer.add_scalar('Loss/train', data['loss'], data['iter'])
    if data['mfu'] > 0:
        writer.add_scalar('MFU', data['mfu'], data['iter'])

for data in log['eval_steps']:
    writer.add_scalar('Loss/val', data['val_loss'], data['iter'])

writer.close()
```

### Weights & Biases

Logs can supplement W&B logging:

```bash
# Both JSON logging and W&B work simultaneously
python train.py --wandb_log=True --save_log_to_json=True
```

## Troubleshooting

### Log file not created?

Check:
1. `save_log_to_json=True` in config
2. `training_logger.py` exists in the directory
3. You're the master process (rank 0 in multi-GPU)

### Large log files?

For very long runs, log files can get large. Reduce save frequency:

```bash
python train.py --log_save_interval=1000  # Save every 1000 iters
```

Or disable iteration logging and keep only eval logs by modifying `train.py`.

## Benefits

1. **Reproducibility**: Complete training history saved
2. **Debugging**: Trace exact loss/time progression
3. **Comparison**: Easy to compare different runs
4. **Analysis**: Use standard JSON/Python tools
5. **Sharing**: Share results without checkpoints

## Next Steps

- Create visualization scripts
- Build experiment tracking dashboard
- Automate hyperparameter tuning with logs
- Archive logs for long-term experiments

---

**All phases now include JSON logging by default!** 📊

