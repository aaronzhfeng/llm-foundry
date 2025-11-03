# Example Terminal Output

This document shows what you'll see when running the enhanced training system.

## Startup Report Example

```
found vocab_size = 50304 (inside data/shakespeare/meta.pkl)
Initializing a new model from scratch
defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
number of parameters: 124.44M
num decayed parameter tensors: 74, with 124,354,560 parameters
num non-decayed parameter tensors: 25, with 19,200 parameters
using fused AdamW: True
compiling the model... (takes a ~minute)
Training logger initialized: out/run_20250103_143022.json

================================================================================
🚀 TRAINING INITIALIZATION
================================================================================

📊 MODEL ARCHITECTURE:
  Total parameters:      124,439,808 (124.44M)
  Trainable parameters:  124,439,808 (124.44M)
  Non-embedding params:  123,587,328 (123.59M)
  Layers:                12
  Hidden size:           768
  Attention heads:       12
  Sequence length:       1024
  Vocabulary size:       50304

⚙️  TRAINING CONFIGURATION:
  Batch size (micro):    12
  Gradient accum steps:  40
  Effective batch size:  480
  Tokens per iteration:  491,520
  Max iterations:        600,000
  Learning rate:         0.0006
  Weight decay:          0.1
  Gradient clip:         1.0
  LR warmup iters:       2,000
  LR decay iters:        600,000

🖥️  HARDWARE:
  Device:                NVIDIA A100-SXM4-80GB
  GPUs:                  1
  Memory per GPU:        80.0 GB
  Precision:             bfloat16
  TF32 enabled:          True
  Compile:               True
  Parallelism:           single

📈 THEORETICAL PERFORMANCE:
  Hardware peak:         312.0 TFLOPS (A100 bf16)
  FLOPs per token:       28.45 GFLOPs
  Attention/FFN ratio:   0.67
  Expected tokens/s @50% MFU: 5483

================================================================================
🏁 STARTING TRAINING
================================================================================

Iter 0: loss 11.0182, time 18880ms (warming up...)
Iter 1: loss 10.9234, time 4320ms (warming up...)
Iter 2: loss 10.8567, time 4298ms (warming up...)
Iter 3: loss 10.7891, time 4305ms (warming up...)
Iter 4: loss 10.7123, time 4312ms (warming up...)

────────────────────────────────────────────────────────────────────────────────
📍 Iter      5 │ Loss: 10.6456 │ Time: 4298ms │ LR: 1.50e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 32.45% │ Achieved: 101.2 TF │ Peak: 312.0 TF
   Tokens/s: 3,557 │ FLOPs/token: 28.5 GF
💾 Memory: 12.34 GB alloc │ 15.67 GB peak │ 16.00 GB reserved

────────────────────────────────────────────────────────────────────────────────
📍 Iter     10 │ Loss: 9.9834 │ Time: 4305ms │ LR: 3.00e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 33.12% │ Achieved: 103.3 TF │ Peak: 312.0 TF
   Tokens/s: 3,632 │ FLOPs/token: 28.5 GF
💾 Memory: 12.45 GB alloc │ 15.89 GB peak │ 16.00 GB reserved
📊 Gradients: norm=2.3456 │ mean=-1.23e-05 │ std=3.45e-04

────────────────────────────────────────────────────────────────────────────────
📍 Iter     15 │ Loss: 9.4523 │ Time: 4298ms │ LR: 4.50e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 33.45% │ Achieved: 104.4 TF │ Peak: 312.0 TF
   Tokens/s: 3,667 │ FLOPs/token: 28.5 GF
💾 Memory: 12.56 GB alloc │ 16.01 GB peak │ 16.50 GB reserved

────────────────────────────────────────────────────────────────────────────────
📍 Iter     20 │ Loss: 8.9876 │ Time: 4301ms │ LR: 6.00e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 33.67% │ Achieved: 105.0 TF │ Peak: 312.0 TF
   Tokens/s: 3,688 │ FLOPs/token: 28.5 GF
💾 Memory: 12.67 GB alloc │ 16.12 GB peak │ 16.50 GB reserved
📊 Gradients: norm=1.8923 │ mean=-8.34e-06 │ std=2.98e-04

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 EVALUATION │ Step   2000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Train loss: 3.2145 │ Val loss: 3.3421 │ LR: 6.00e-04
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 Saving checkpoint to out

────────────────────────────────────────────────────────────────────────────────
📍 Iter   2001 │ Loss: 3.2089 │ Time: 4295ms │ LR: 6.00e-04
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 34.56% │ Achieved: 107.8 TF │ Peak: 312.0 TF
   Tokens/s: 3,789 │ FLOPs/token: 28.5 GF
💾 Memory: 13.12 GB alloc │ 16.45 GB peak │ 17.00 GB reserved

...
```

## Multi-GPU Output Example (4 GPUs with FSDP)

```bash
$ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py --use_fsdp=True
```

```
[Rank 0] Initializing process group...
[Rank 1] Initializing process group...
[Rank 2] Initializing process group...
[Rank 3] Initializing process group...
tokens per iteration will be: 1,966,080
Initializing a new model from scratch
number of parameters: 124.44M
Wrapping model with FSDP...
FSDP enabled with min_params=1000000.0, mixed_precision=bfloat16
num decayed parameter tensors: 74, with 124,354,560 parameters
num non-decayed parameter tensors: 25, with 19,200 parameters
using fused AdamW: True
compiling the model... (takes a ~minute)
Training logger initialized: out/run_20250103_145632.json

================================================================================
🚀 TRAINING INITIALIZATION
================================================================================

📊 MODEL ARCHITECTURE:
  Total parameters:      124,439,808 (124.44M)
  Trainable parameters:  124,439,808 (124.44M)
  Non-embedding params:  123,587,328 (123.59M)
  Layers:                12
  Hidden size:           768
  Attention heads:       12
  Sequence length:       1024
  Vocabulary size:       50304

⚙️  TRAINING CONFIGURATION:
  Batch size (micro):    12
  Gradient accum steps:  10
  Effective batch size:  480
  Tokens per iteration:  1,966,080
  Max iterations:        600,000
  Learning rate:         0.0006
  Weight decay:          0.1
  Gradient clip:         1.0
  LR warmup iters:       2,000
  LR decay iters:        600,000

🖥️  HARDWARE:
  Device:                NVIDIA A100-SXM4-80GB
  GPUs:                  4
  Memory per GPU:        80.0 GB
  Precision:             bfloat16
  TF32 enabled:          True
  Compile:               True
  Parallelism:           FSDP

📈 THEORETICAL PERFORMANCE:
  Hardware peak:         1248.0 TFLOPS (A100 bf16)
  FLOPs per token:       28.45 GFLOPs
  Attention/FFN ratio:   0.67
  Expected tokens/s @50% MFU: 21932

================================================================================
🏁 STARTING TRAINING
================================================================================

Iter 0: loss 11.0234, time 22145ms (warming up...)
Iter 1: loss 10.9156, time 4567ms (warming up...)
Iter 2: loss 10.8423, time 4512ms (warming up...)
Iter 3: loss 10.7645, time 4534ms (warming up...)
Iter 4: loss 10.6892, time 4521ms (warming up...)

────────────────────────────────────────────────────────────────────────────────
📍 Iter      5 │ Loss: 10.6123 │ Time: 4529ms │ LR: 1.50e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 29.87% │ Achieved: 372.8 TF │ Peak: 1248.0 TF
   Tokens/s: 13,109 │ FLOPs/token: 28.5 GF
💾 Memory: 3.12 GB alloc │ 4.23 GB peak │ 4.50 GB reserved

────────────────────────────────────────────────────────────────────────────────
📍 Iter     10 │ Loss: 9.8765 │ Time: 4515ms │ LR: 3.00e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 30.45% │ Achieved: 380.0 TF │ Peak: 1248.0 TF
   Tokens/s: 13,359 │ FLOPs/token: 28.5 GF
💾 Memory: 3.15 GB alloc │ 4.28 GB peak │ 4.50 GB reserved
📊 Gradients: norm=2.1234 │ mean=-9.87e-06 │ std=3.21e-04

...
```

## JSON Log Example

```json
{
  "run_name": "run_20250103_143022",
  "start_time": "2025-01-03T14:30:22.123456",
  "config": {
    "batch_size": 12,
    "gradient_accumulation_steps": 40,
    "learning_rate": 0.0006,
    "max_iters": 600000,
    "n_layer": 12,
    "n_embd": 768,
    "use_fsdp": false,
    ...
  },
  "startup_info": {
    "timestamp": "2025-01-03T14:30:25.789012",
    "model": {
      "total_params": 124439808,
      "trainable_params": 124439808,
      "non_embedding_params": 123587328
    },
    "optimizer": {
      "type": "AdamW",
      "param_groups": 2
    },
    "hardware": {
      "gpu_name": "NVIDIA A100-SXM4-80GB",
      "num_gpus": 1,
      "gpu_memory_gb": 80.0,
      "precision": "bfloat16",
      "parallelism": "single"
    }
  },
  "training_iterations": [
    {
      "iter": 5,
      "loss": 10.6456,
      "time_ms": 4298.0,
      "mfu": {
        "mfu": 0.3245,
        "mfu_percent": 32.45,
        "flops_achieved": 1.012e14,
        "flops_per_token": 28450000000.0,
        "tokens_per_sec": 3557.0,
        "hardware_peak_flops": 3.12e14,
        "hardware_peak_tflops": 312.0,
        "achieved_tflops": 101.2,
        "gpu_name": "A100",
        "precision": "bf16",
        "num_gpus": 1,
        "attention_flops_per_layer": 73728000,
        "ffn_flops_per_layer": 115343360,
        "attention_to_ffn_ratio": 0.639
      },
      "memory": {
        "allocated_gb": 12.34,
        "reserved_gb": 16.0,
        "max_allocated_gb": 15.67,
        "max_reserved_gb": 16.0
      }
    },
    {
      "iter": 10,
      "loss": 9.9834,
      "time_ms": 4305.0,
      "mfu": {
        "mfu": 0.3312,
        "mfu_percent": 33.12,
        ...
      },
      "memory": {
        "allocated_gb": 12.45,
        "reserved_gb": 16.0,
        "max_allocated_gb": 15.89,
        "max_reserved_gb": 16.0
      },
      "gradients": {
        "global_norm": 2.3456,
        "mean_layer_norm": 0.0234,
        "max_layer_norm": 0.1234,
        "min_layer_norm": 0.0012,
        "grad_mean": -1.23e-05,
        "grad_std": 3.45e-04,
        "grad_min": -0.0023,
        "grad_max": 0.0019
      }
    },
    ...
  ],
  "eval_steps": [
    {
      "iter": 0,
      "train_loss": 11.0034,
      "val_loss": 10.9976,
      "timestamp": "2025-01-03T14:30:28.456789",
      "lr": 2.9985e-07
    },
    {
      "iter": 2000,
      "train_loss": 3.2145,
      "val_loss": 3.3421,
      "timestamp": "2025-01-03T16:45:12.123456",
      "lr": 0.0006
    },
    ...
  ],
  "checkpoints": [
    {
      "iter": 2000,
      "val_loss": 3.3421,
      "path": "out/ckpt.pt",
      "timestamp": "2025-01-03T16:45:15.789012"
    },
    ...
  ],
  "metadata": {
    "world_size": 1,
    "device": "cuda",
    "dtype": "bfloat16",
    "compile": true,
    "use_zero1": false,
    "use_fsdp": false
  },
  "end_time": "2025-01-04T02:15:33.456789",
  "summary": {
    "total_iterations": 10000,
    "final_iter": 9999,
    "final_train_loss": 2.8956,
    "best_val_loss": 3.1234,
    "avg_time_ms": 4312.5,
    "avg_mfu": 33.78,
    "total_eval_steps": 5,
    "total_checkpoints": 5
  }
}
```

## Key Differences from Original nanoGPT

### Original nanoGPT Output:
```
iter 100: loss 3.2145, time 234.56ms, mfu 28.34%
```

### Enhanced System Output:
```
────────────────────────────────────────────────────────────────────────────────
📍 Iter    100 │ Loss: 3.2145 │ Time: 4298ms │ LR: 6.00e-04
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 33.45% │ Achieved: 104.4 TF │ Peak: 312.0 TF
   Tokens/s: 3,667 │ FLOPs/token: 28.5 GF
💾 Memory: 12.56 GB alloc │ 16.01 GB peak │ 16.50 GB reserved
📊 Gradients: norm=1.8923 │ mean=-8.34e-06 │ std=2.98e-04
```

## Formula Breakdown in Output

The enhanced system shows you exactly how MFU is calculated:

- **MFU: 33.45%** = (Achieved TF / Peak TF) × 100 = (104.4 / 312.0) × 100
- **Achieved: 104.4 TF** = FLOPs/token × Tokens/s = 28.5 GF × 3,667 = 104.5 TF
- **Tokens/s: 3,667** = Tokens per iteration / Time = 491,520 / 4.298s = 114,354 tokens/s (per-GPU: 3,667)
- **FLOPs/token: 28.5 GF** = Calculated using academic formula: `3 * (12*S*H² + 2*a*S²*H) * L / S`

This transparency helps you understand:
1. Where your performance bottlenecks are
2. How close you are to theoretical peak
3. Whether data loading or computation is the limiting factor

