# Training Analysis Scripts

This folder contains analysis scripts and notebooks for the LLM training project.

## Scripts

### `compute_training_times.py`

Analyzes training JSON logs to compute actual training time by summing `time_ms` for each iteration.

**Usage:**
```bash
cd /root/llm_TII/enhanced_training_system/analysis
python compute_training_times.py
```

**Note:** This script filters out evaluation iterations (which have very large `time_ms` values) to compute pure training time. If JSON logs are incomplete, results will reflect only the logged iterations.

## Results Summary (from 199 logged iterations)

| Model | Batch Size<br>(per GPU) | Gradient<br>Accumulation | GPUs | Effective<br>Batch Size | Tokens per<br>Iteration | Training<br>Time |
|-------|:-----------------------:|:------------------------:|:----:|:-----------------------:|:-----------------------:|:----------------:|
| **Qwen3 1.8B Optimal** | 2 | 96 | 2 | **384** | **786,432** | **1h 33m** (199 iters) |
| **GPT-2 1.29B** | 4 | 32 | 2 | **256** | **524,288** | **0h 39m** (199 iters) |
| **LLaMA 3 2.2B Chinchilla** | 2 | 64 | 2 | **256** | **524,288** | **1h 24m** (199 iters) |
| **LLaMA 2 1.36B** | 5 | 16 | 2 | **160** | **327,680** | **0h 25m** (199 iters) |

**Note:** These are partial times based on logged iterations. Full 2000-iteration training times from terminal output are:
- **Qwen3**: ~15h 39m
- **GPT-2**: ~12-13h (estimated)
- **LLaMA 3**: ~14h 16m
- **LLaMA 2**: ~9-10h (estimated)

## Key Findings

**Training time is NOT proportional to model size alone!** It depends on:

1. **Gradient accumulation steps**: Higher = more time
   - Qwen3: 96 steps (slowest per iteration)
   - LLaMA3: 64 steps
   - GPT-2: 32 steps
   - LLaMA2: 16 steps (fastest per iteration)

2. **Vocabulary size**: Larger vocab = larger output layer = slower
   - Qwen3: 152K vocab (largest)
   - LLaMA3: 128K vocab
   - GPT-2/LLaMA2: 32-50K vocab

3. **Model size**: More parameters = more compute
   - LLaMA3: 2.22B (largest)
   - Qwen3: 1.83B
   - LLaMA2: 1.36B
   - GPT-2: 1.29B (smallest)

4. **Efficiency (MFU)**: Higher MFU = better hardware utilization
   - GPT-2: 30.57% (highest)
   - LLaMA2: 30.19%
   - Qwen3: 27.62%
   - LLaMA3: 24.61% (lowest, due to large model size)

## Tokens Processed per Hour

Based on the logged iterations:
- GPT-2: **158.1M tokens/hour** (fastest)
- LLaMA2: **151.6M tokens/hour**
- Qwen3: **100.3M tokens/hour**
- LLaMA3: **74.0M tokens/hour** (slowest, but largest model)

