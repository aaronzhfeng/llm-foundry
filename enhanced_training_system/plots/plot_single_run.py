#!/usr/bin/env python3
"""
Plot comprehensive training analysis for a single run.
Shows loss, perplexity, MFU, throughput, learning rate, and memory usage.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_run(json_path):
    """Load training log JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_training_data(data):
    """Extract all training iteration data."""
    iterations = []
    losses = []
    mfus = []
    times_ms = []
    tokens_per_sec = []
    memory_alloc = []
    memory_peak = []
    learning_rates = []
    
    for entry in data['training_iterations']:
        iter_num = entry['iter']
        if iter_num == 0:  # Skip first iteration (initialization)
            continue
        
        # Check if this is an evaluation run (time > 100s indicates eval)
        is_eval_iter = entry['time_ms'] > 100000
            
        iterations.append(iter_num)
        losses.append(entry['loss'])
        
        # Only add time-based metrics if not evaluation
        if not is_eval_iter:
            times_ms.append(entry['time_ms'])
        else:
            times_ms.append(None)  # Mark as None to skip in plots
        
        # Extract MFU
        if isinstance(entry.get('mfu'), dict):
            # Skip MFU from eval iters (it's artificially low)
            if not is_eval_iter:
                mfus.append(entry['mfu']['mfu_percent'])
                tokens_per_sec.append(entry['mfu'].get('tokens_per_sec', 0))
            else:
                mfus.append(None)
                tokens_per_sec.append(None)
        else:
            mfus.append(None)
            tokens_per_sec.append(None)
        
        # Extract memory
        if 'memory' in entry:
            memory_alloc.append(entry['memory']['allocated_gb'])
            memory_peak.append(entry['memory']['max_allocated_gb'])
        else:
            memory_alloc.append(None)
            memory_peak.append(None)
    
    return {
        'iterations': iterations,
        'losses': losses,
        'mfus': mfus,
        'times_ms': times_ms,
        'tokens_per_sec': tokens_per_sec,
        'memory_alloc': memory_alloc,
        'memory_peak': memory_peak,
    }

def extract_eval_data(data):
    """Extract evaluation data."""
    if not data.get('eval_steps'):
        return None
    
    eval_iters = []
    train_losses = []
    val_losses = []
    lrs = []
    
    for entry in data['eval_steps']:
        eval_iters.append(entry['iter'])
        train_losses.append(entry['train_loss'])
        val_losses.append(entry['val_loss'])
        if 'lr' in entry:
            lrs.append(entry['lr'])
    
    return {
        'iters': eval_iters,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'lrs': lrs
    }

def compute_learning_rate(iter_num, config):
    """Compute learning rate at given iteration."""
    lr = config.get('learning_rate', 3e-4)
    warmup = config.get('warmup_iters', 2000)
    decay_iters = config.get('lr_decay_iters', 25000)
    min_lr = config.get('min_lr', 3e-5)
    decay_lr = config.get('decay_lr', True)
    
    if not decay_lr:
        return lr
    
    # Warmup
    if iter_num < warmup:
        return lr * (iter_num + 1) / (warmup + 1)
    
    # Decay
    if iter_num > decay_iters:
        return min_lr
    
    # Cosine decay
    decay_ratio = (iter_num - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def main():
    # Determine input file
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = Path("../saves/run_20251110_174040.json")
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print(f"Usage: {sys.argv[0]} [path/to/run.json]")
        sys.exit(1)
    
    print(f"Loading: {json_path}")
    data = load_run(json_path)
    
    # Extract data
    train_data = extract_training_data(data)
    eval_data = extract_eval_data(data)
    config = data.get('config', {})
    startup = data.get('startup_info', {})
    
    # Get model info
    model_info = startup.get('model', {})
    total_params = model_info.get('total_params', 0)
    
    # Compute derived metrics
    iters = np.array(train_data['iterations'])
    losses = np.array(train_data['losses'])
    perplexities = np.exp(losses)
    
    # Compute learning rates for all iterations
    lrs = [compute_learning_rate(i, config) for i in iters]
    
    # Compute cumulative tokens (assuming constant tokens per iter)
    block_size = config.get('block_size', 2048)
    batch_size = config.get('batch_size', 8)
    grad_accum = config.get('gradient_accumulation_steps', 16)
    num_gpus = data.get('metadata', {}).get('world_size', 1)
    tokens_per_iter = block_size * batch_size * grad_accum * num_gpus
    cumulative_tokens = iters * tokens_per_iter / 1e9  # In billions
    
    # Filter out None values for MFU and memory
    mfu_iters = [iters[i] for i, m in enumerate(train_data['mfus']) if m is not None]
    mfu_values = [m for m in train_data['mfus'] if m is not None]
    
    tokens_iters = [iters[i] for i, t in enumerate(train_data['tokens_per_sec']) if t is not None and t > 0]
    tokens_values = [t for t in train_data['tokens_per_sec'] if t is not None and t > 0]
    
    mem_iters = [iters[i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    mem_alloc_values = [m for m in train_data['memory_alloc'] if m is not None]
    mem_peak_values = [train_data['memory_peak'][i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'LLaMA 1.29B Training Analysis - {len(iters)} Iterations\n'
                 f'Model: {total_params/1e6:.0f}M params | Hardware: {num_gpus}× A6000 | '
                 f'Total Tokens: {cumulative_tokens[-1]:.2f}B',
                 fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== Plot 1: Loss & Perplexity (Dual Y-axis) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    # Loss on left axis
    line1 = ax1.plot(iters, losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=11, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Mark eval point
    if eval_data and eval_data['iters']:
        ax1.scatter(eval_data['iters'], eval_data['val_loss'], 
                   color='red', s=100, zorder=5, marker='*', 
                   label=f"Val Loss: {eval_data['val_loss'][0]:.3f}")
    
    # Perplexity on right axis
    line2 = ax1_twin.plot(iters, perplexities, 'g-', linewidth=1.5, 
                          label='Perplexity', alpha=0.6, linestyle='--')
    ax1_twin.set_ylabel('Perplexity = exp(loss)', fontsize=11, color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1_twin.set_yscale('log')
    
    ax1.set_title('Loss & Perplexity', fontsize=12, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if eval_data and eval_data['iters']:
        labels.insert(1, f"Val @ {eval_data['iters'][0]}")
    ax1.legend(lines, labels, loc='upper right', fontsize=9)
    
    # ========== Plot 2: Learning Rate Schedule ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iters, lrs, 'purple', linewidth=2)
    
    # Mark warmup end
    warmup = config.get('warmup_iters', 2000)
    if max(iters) > warmup:
        ax2.axvline(x=warmup, color='red', linestyle='--', alpha=0.5, label=f'Warmup End: {warmup}')
        ax2.legend(fontsize=9)
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # ========== Plot 3: Cumulative Tokens ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iters, cumulative_tokens, 'teal', linewidth=2, marker='o', markersize=1)
    
    # Don't show 27B reference (makes plot too compressed)
    # User will report progress separately
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Cumulative Tokens (Billions)', fontsize=11)
    ax3.set_title(f'Tokens Processed: {cumulative_tokens[-1]:.3f}B', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add text annotation showing progress
    progress_pct = (cumulative_tokens[-1] / 27.0) * 100
    ax3.text(0.98, 0.02, f'{progress_pct:.1f}% of Chinchilla optimal (27B)', 
            transform=ax3.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== Plot 4: MFU Over Time ==========
    ax4 = fig.add_subplot(gs[1, 0])
    if mfu_values:
        ax4.plot(mfu_iters, mfu_values, 'orange', linewidth=2, marker='o', markersize=2)
        
        # Average MFU
        avg_mfu = np.mean(mfu_values)
        ax4.axhline(y=avg_mfu, color='blue', linestyle='--', alpha=0.6, 
                   label=f'Average: {avg_mfu:.2f}%')
        
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('MFU (%)', fontsize=11)
        ax4.set_title(f'Model FLOPs Utilization (Avg: {avg_mfu:.1f}%)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Truncate y-axis to actual data range (with small margin)
        mfu_min = min(mfu_values)
        mfu_max = max(mfu_values)
        mfu_range = mfu_max - mfu_min
        margin = max(2, mfu_range * 0.2)  # At least 2% margin or 20% of range
        ax4.set_ylim([max(0, mfu_min - margin), mfu_max + margin])
    
    # ========== Plot 5: Throughput (Tokens/sec) ==========
    ax5 = fig.add_subplot(gs[1, 1])
    if tokens_values:
        ax5.plot(tokens_iters, tokens_values, 'brown', linewidth=2, marker='o', markersize=2)
        
        # Average throughput
        avg_tokens = np.mean(tokens_values)
        ax5.axhline(y=avg_tokens, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_tokens:.0f} tokens/s')
        
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Tokens per Second', fontsize=11)
        ax5.set_title(f'Training Throughput (Avg: {avg_tokens:.0f} tokens/s)', 
                     fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # ========== Plot 6: Memory Usage ==========
    ax6 = fig.add_subplot(gs[1, 2])
    if mem_alloc_values:
        ax6.plot(mem_iters, mem_alloc_values, 'steelblue', linewidth=2, 
                label='Allocated', marker='o', markersize=2)
        ax6.plot(mem_iters, mem_peak_values, 'darkred', linewidth=2, 
                label='Peak', marker='s', markersize=2)
        
        # GPU capacity
        gpu_capacity = 48  # A6000 has 48 GB
        ax6.axhline(y=gpu_capacity, color='red', linestyle='--', alpha=0.4, 
                   label=f'GPU Capacity: {gpu_capacity} GB')
        
        ax6.set_xlabel('Iteration', fontsize=11)
        ax6.set_ylabel('Memory (GB)', fontsize=11)
        ax6.set_title(f'GPU Memory Usage (Peak: {max(mem_peak_values):.1f} GB / {gpu_capacity} GB)', 
                     fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, gpu_capacity * 1.1])
    
    # Save figure
    output_path = json_path.parent / f"{json_path.stem}_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    # Model info
    print(f"\n📊 MODEL:")
    print(f"   Total params:       {total_params/1e9:.3f}B ({total_params/1e6:.0f}M)")
    arch_name = startup.get('architecture_name', 'Unknown')
    print(f"   Architecture:       {arch_name}")
    print(f"   Hardware:           {num_gpus}× A6000")
    parallelism = data.get('metadata', {}).get('parallelism', 
                  startup.get('hardware', {}).get('parallelism', 'Unknown'))
    print(f"   Parallelism:        {parallelism}")
    
    # Training progress
    print(f"\n📈 TRAINING PROGRESS:")
    print(f"   Iterations:         {len(iters)} ({iters[0]} → {iters[-1]})")
    print(f"   Tokens processed:   {cumulative_tokens[-1]:.2f}B")
    print(f"   Chinchilla optimal: 27B tokens ({cumulative_tokens[-1]/27*100:.1f}% complete)")
    
    # Loss metrics
    print(f"\n📉 LOSS:")
    print(f"   Initial:            {losses[0]:.4f}")
    print(f"   Final:              {losses[-1]:.4f}")
    print(f"   Reduction:          {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    if eval_data and eval_data['val_loss']:
        print(f"   Val @ iter {eval_data['iters'][0]}:   {eval_data['val_loss'][0]:.4f}")
    
    # Perplexity
    print(f"\n🎯 PERPLEXITY:")
    print(f"   Initial:            {perplexities[0]:.1f}")
    print(f"   Final:              {perplexities[-1]:.1f}")
    if eval_data and eval_data['val_loss']:
        print(f"   Val @ iter {eval_data['iters'][0]}:   {np.exp(eval_data['val_loss'][0]):.1f}")
    
    # Performance
    if mfu_values:
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Avg MFU:            {np.mean(mfu_values):.2f}% (expected: 40-50%)")
        print(f"   Avg Throughput:     {np.mean(tokens_values):.0f} tokens/s")
        # Filter out None values from times
        valid_times = [t for t in train_data['times_ms'] if t is not None]
        print(f"   Avg Time/Iter:      {np.mean(valid_times)/1000:.2f}s")
    
    # Memory
    if mem_peak_values:
        print(f"\n💾 MEMORY:")
        print(f"   Peak allocated:     {max(mem_peak_values):.2f} GB / 48 GB ({max(mem_peak_values)/48*100:.1f}%)")
        print(f"   Avg allocated:      {np.mean(mem_alloc_values):.2f} GB")
    
    # Training efficiency
    if len(iters) > 1:
        valid_times = [t for t in train_data['times_ms'] if t is not None]
        total_time_hours = sum(valid_times) / 1000 / 3600
        print(f"\n⏱️  TIME:")
        print(f"   Total training:     {total_time_hours:.2f} hours")
        if tokens_values:
            tokens_per_hour = np.mean(tokens_values) * 3600
            time_to_27B = (27e9 - cumulative_tokens[-1] * 1e9) / tokens_per_hour
            print(f"   Est. to 27B tokens: {time_to_27B:.1f} hours (~{time_to_27B/24:.1f} days)")
    
    print("\n" + "="*70 + "\n")
    
    # Show plot (if running interactively)
    # plt.show()

if __name__ == '__main__':
    main()

