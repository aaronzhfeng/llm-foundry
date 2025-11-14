#!/usr/bin/env python3
"""
Training Time Analysis
======================
Computes actual total training time for each model by summing time_ms from JSON logs.

Models:
- Qwen3 1.8B Optimal (24L-16H-2048D)
- GPT-2 1.29B (18L-18H-2304D)
- LLaMA 3 2.2B Chinchilla (30L-16H-2048D)
- LLaMA 2 1.36B (18L-18H-2304D)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Model information mapping
MODEL_INFO = {
    "run_20251113_225009": {
        "name": "Qwen3_1.8B_Optimal",
        "full_name": "Qwen3 1.8B Optimal",
        "arch": "24L-16H-2048D",
        "params": "1.83B"
    },
    "run_20251113_095419": {
        "name": "GPT2_1.29B",
        "full_name": "GPT-2 1.29B",
        "arch": "18L-18H-2304D",
        "params": "1.29B"
    },
    "run_20251112_122613": {
        "name": "LLaMA3_2.2B_Chinchilla",
        "full_name": "LLaMA 3 2.2B Chinchilla",
        "arch": "30L-16H-2048D",
        "params": "2.22B"
    },
    "run_20251110_174040": {
        "name": "LLaMA2_1.36B",
        "full_name": "LLaMA 2 1.36B",
        "arch": "18L-18H-2304D",
        "params": "1.36B"
    }
}


def load_training_data(json_path):
    """Load training data from JSON log file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_total_time(data):
    """Compute total training time from iteration time_ms.
    
    Filters out evaluation iterations (which have very large time_ms values
    due to the eval loop) to get accurate training-only time.
    """
    times_ms = []
    eval_count = 0
    
    for entry in data['training_iterations']:
        time_ms = entry['time_ms']
        
        # Evaluation iterations have very large time (>100 seconds = 100,000 ms)
        # Skip these to get pure training time
        if time_ms > 100000:
            eval_count += 1
            continue
        
        times_ms.append(time_ms)
    
    total_ms = sum(times_ms)
    total_seconds = total_ms / 1000
    total_hours = total_seconds / 3600
    total_minutes = (total_seconds % 3600) / 60
    
    return {
        'total_ms': total_ms,
        'total_seconds': total_seconds,
        'total_hours': total_hours,
        'total_time_str': f"{int(total_hours)}h {int(total_minutes)}m",
        'avg_time_ms': np.mean(times_ms),
        'training_iters': len(times_ms),
        'eval_iters_skipped': eval_count
    }


def extract_batch_config(data):
    """Extract batch configuration from training data."""
    config = data['config']
    metadata = data.get('metadata', {})
    
    batch_size = config.get('batch_size', 'N/A')
    grad_accum = config.get('gradient_accumulation_steps', 'N/A')
    num_gpus = metadata.get('world_size', 1)
    block_size = config.get('block_size', 2048)
    
    effective_batch = batch_size * grad_accum * num_gpus if batch_size != 'N/A' else 'N/A'
    tokens_per_iter = effective_batch * block_size if effective_batch != 'N/A' else 'N/A'
    
    return {
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'num_gpus': num_gpus,
        'effective_batch': effective_batch,
        'tokens_per_iter': tokens_per_iter
    }


def main():
    # Process all runs
    saves_dir = Path("../saves")
    results = []

    for run_name, model_info in MODEL_INFO.items():
        json_path = saves_dir / f"{run_name}.json"
        
        if not json_path.exists():
            print(f"⚠️  Skipping {run_name} (file not found)")
            continue
        
        print(f"Processing: {model_info['full_name']}...")
        
        # Load data
        data = load_training_data(json_path)
        
        # Compute time
        time_stats = compute_total_time(data)
        
        # Extract batch config
        batch_config = extract_batch_config(data)
        
        # Extract MFU stats
        mfu_values = []
        for entry in data['training_iterations']:
            if entry.get('mfu') and isinstance(entry['mfu'], dict):
                mfu_val = entry['mfu'].get('mfu_percent')
                if mfu_val is not None and entry['time_ms'] < 100000:  # Skip eval iters
                    mfu_values.append(mfu_val)
        
        avg_mfu = np.mean(mfu_values) if mfu_values else None
        
        # Combine results
        results.append({
            'Model': model_info['full_name'],
            'Architecture': model_info['arch'],
            'Parameters': model_info['params'],
            'Batch Size': batch_config['batch_size'],
            'Grad Accum': batch_config['grad_accum'],
            'GPUs': batch_config['num_gpus'],
            'Effective Batch': batch_config['effective_batch'],
            'Tokens/Iter': f"{batch_config['tokens_per_iter']:,}" if batch_config['tokens_per_iter'] != 'N/A' else 'N/A',
            'Training Time': time_stats['total_time_str'],
            'Total Hours': f"{time_stats['total_hours']:.2f}",
            'Avg Time/Iter (s)': f"{time_stats['avg_time_ms']/1000:.2f}",
            'Training Iters': time_stats['training_iters'],
            'Avg MFU (%)': f"{avg_mfu:.2f}" if avg_mfu else 'N/A'
        })
        
        print(f"  ✓ Total time: {time_stats['total_time_str']} ({time_stats['training_iters']} iterations)\n")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("TRAINING TIME ANALYSIS COMPLETE")
    print("="*80)
    
    # Display full results
    print("\n" + "="*80)
    print("Full Training Configuration:")
    print("="*80)
    print(df.to_string(index=False))
    
    # Summary table
    print("\n" + "="*80)
    print("Training Batch Configuration Summary:")
    print("="*80)
    summary_df = df[['Model', 'Batch Size', 'Grad Accum', 'GPUs', 'Effective Batch', 'Tokens/Iter', 'Training Time']].copy()
    print(summary_df.to_string(index=False))
    
    # Training time ranking
    print("\n" + "="*80)
    print("Training Time Ranking (Fastest to Slowest):")
    print("="*80)
    time_ranking = df[['Model', 'Training Time', 'Total Hours', 'Avg Time/Iter (s)', 'Avg MFU (%)']].copy()
    time_ranking['Total Hours (float)'] = time_ranking['Total Hours'].astype(float)
    time_ranking = time_ranking.sort_values('Total Hours (float)')
    time_ranking['Rank'] = range(1, len(time_ranking) + 1)
    print(time_ranking[['Rank', 'Model', 'Training Time', 'Total Hours', 'Avg Time/Iter (s)', 'Avg MFU (%)']].to_string(index=False))
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: What Affects Training Time?")
    print("="*80)
    
    print("\n1. Gradient Accumulation Steps (Higher = Slower):")
    for _, row in df.iterrows():
        print(f"   {row['Model']:30s}: {row['Grad Accum']:3d} steps → {row['Training Time']:8s}")
    
    print("\n2. Model Size (Larger = More Compute):")
    for _, row in df.iterrows():
        print(f"   {row['Model']:30s}: {row['Parameters']:6s} params → {row['Avg Time/Iter (s)']:6s} s/iter")
    
    print("\n3. Efficiency (MFU):")
    for _, row in df.iterrows():
        print(f"   {row['Model']:30s}: {row['Avg MFU (%)']:6s}% MFU")
    
    print("\n4. Tokens Processed per Hour:")
    for _, row in df.iterrows():
        if row['Tokens/Iter'] != 'N/A':
            tokens_per_iter = int(row['Tokens/Iter'].replace(',', ''))
            total_hours = float(row['Total Hours'])
            training_iters = int(row['Training Iters'])
            tokens_per_hour = (tokens_per_iter * training_iters) / total_hours
            print(f"   {row['Model']:30s}: {tokens_per_hour/1e6:.1f}M tokens/hour")
    
    # Generate markdown table
    print("\n" + "="*80)
    print("Markdown Table for Documentation:")
    print("="*80)
    print("\n```markdown")
    print("| Model | Batch Size<br>(per GPU) | Gradient<br>Accumulation | GPUs | Effective<br>Batch Size | Tokens per<br>Iteration | Training<br>Time |")
    print("|-------|:-----------------------:|:------------------------:|:----:|:-----------------------:|:-----------------------:|:----------------:|")
    
    for _, row in df.iterrows():
        print(f"| **{row['Model']}** | {row['Batch Size']} | {row['Grad Accum']} | {row['GPUs']} | **{row['Effective Batch']}** | **{row['Tokens/Iter']}** | **{row['Training Time']}** |")
    
    print("```\n")
    
    print("="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("""
Training time is NOT proportional to model size alone! It depends on:

1. **Gradient accumulation steps**: Higher = more time
   - More gradient accumulation steps means more forward/backward passes per optimizer step
   - This is the dominant factor in training time differences

2. **Vocabulary size**: Larger vocab = larger output layer = slower
   - Qwen3: 152K vocab (largest)
   - LLaMA3: 128K vocab
   - GPT-2/LLaMA2: 32-50K vocab

3. **Model size**: More parameters = more compute
   - But can be offset by better batch configuration

4. **MFU (Model FLOPs Utilization)**: Higher MFU = better hardware efficiency
   - GPT-2 achieves highest MFU (~30.6%) despite being smallest
   - Larger models tend to have lower MFU due to memory bottlenecks
""")
    print("="*80)


if __name__ == '__main__':
    main()

