#!/usr/bin/env python3
"""
Plot all training runs with proper model names.
Generates one comprehensive plot per run.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Model name mapping
MODEL_INFO = {
    "run_20251113_225009": {
        "name": "Qwen3_1.8B_Optimal",
        "full_name": "Qwen3 1.8B Optimal (24L-16H-2048D)",
        "params": "1.83B",
        "optimal_tokens": "81.7B"
    },
    "run_20251113_095419": {
        "name": "GPT2_1.29B",
        "full_name": "GPT-2 1.29B (18L-18H-2304D)",
        "params": "1.29B",
        "optimal_tokens": "27B"
    },
    "run_20251112_122613": {
        "name": "LLaMA3_2.2B_Chinchilla",
        "full_name": "LLaMA 3 2.2B Chinchilla (30L-16H-2048D)",
        "params": "2.22B",
        "optimal_tokens": "61.5B"
    },
    "run_20251110_174040": {
        "name": "LLaMA2_1.36B",
        "full_name": "LLaMA 2 1.36B (18L-18H-2304D)",
        "params": "1.36B",
        "optimal_tokens": "27B"
    }
}

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
    
    for entry in data['training_iterations']:
        iter_num = entry['iter']
        if iter_num == 0:
            continue
        
        is_eval_iter = entry['time_ms'] > 100000
            
        iterations.append(iter_num)
        losses.append(entry['loss'])
        
        if not is_eval_iter:
            times_ms.append(entry['time_ms'])
        else:
            times_ms.append(None)
        
        # Extract MFU
        if isinstance(entry.get('mfu'), dict):
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
    
    if iter_num < warmup:
        return lr * (iter_num + 1) / (warmup + 1)
    
    if iter_num > decay_iters:
        return min_lr
    
    decay_ratio = (iter_num - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def plot_single_run(json_path, model_info):
    """Plot comprehensive analysis for a single run."""
    print(f"\nProcessing: {json_path.name}")
    print(f"Model: {model_info['full_name']}")
    
    data = load_run(json_path)
    
    # Extract data
    train_data = extract_training_data(data)
    eval_data = extract_eval_data(data)
    config = data.get('config', {})
    startup = data.get('startup_info', {})
    
    # Get model info
    model_params = startup.get('model', {})
    total_params = model_params.get('total_params', 0)
    
    # Compute derived metrics
    iters = np.array(train_data['iterations'])
    losses = np.array(train_data['losses'])
    perplexities = np.exp(losses)
    
    # Compute learning rates
    lrs = [compute_learning_rate(i, config) for i in iters]
    
    # Compute cumulative tokens
    block_size = config.get('block_size', 2048)
    batch_size = config.get('batch_size', 8)
    grad_accum = config.get('gradient_accumulation_steps', 16)
    num_gpus = data.get('metadata', {}).get('world_size', 1)
    tokens_per_iter = block_size * batch_size * grad_accum * num_gpus
    cumulative_tokens = iters * tokens_per_iter / 1e9
    
    # Filter None values
    mfu_iters = [iters[i] for i, m in enumerate(train_data['mfus']) if m is not None]
    mfu_values = [m for m in train_data['mfus'] if m is not None]
    
    tokens_iters = [iters[i] for i, t in enumerate(train_data['tokens_per_sec']) if t is not None and t > 0]
    tokens_values = [t for t in train_data['tokens_per_sec'] if t is not None and t > 0]
    
    mem_iters = [iters[i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    mem_alloc_values = [m for m in train_data['memory_alloc'] if m is not None]
    mem_peak_values = [train_data['memory_peak'][i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'{model_info["full_name"]} - {len(iters)} Iterations\n'
                 f'{model_info["params"]} params | Hardware: {num_gpus}× A6000 | '
                 f'Total Tokens: {cumulative_tokens[-1]:.2f}B / {model_info["optimal_tokens"]}',
                 fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Loss & Perplexity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(iters, losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=11, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    if eval_data and eval_data['iters']:
        ax1.scatter(eval_data['iters'], eval_data['val_loss'], 
                   color='red', s=100, zorder=5, marker='*', 
                   label=f"Val Loss: {eval_data['val_loss'][-1]:.3f}")
    
    line2 = ax1_twin.plot(iters, perplexities, 'g-', linewidth=1.5, 
                          label='Perplexity', alpha=0.6, linestyle='--')
    ax1_twin.set_ylabel('Perplexity = exp(loss)', fontsize=11, color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1_twin.set_yscale('log')
    
    ax1.set_title('Loss & Perplexity', fontsize=12, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9)
    
    # Plot 2: Learning Rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iters, lrs, 'purple', linewidth=2)
    
    warmup = config.get('warmup_iters', 2000)
    if max(iters) > warmup:
        ax2.axvline(x=warmup, color='red', linestyle='--', alpha=0.5, label=f'Warmup End: {warmup}')
        ax2.legend(fontsize=9)
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Cumulative Tokens
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iters, cumulative_tokens, 'teal', linewidth=2, marker='o', markersize=1)
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Cumulative Tokens (Billions)', fontsize=11)
    ax3.set_title(f'Tokens Processed: {cumulative_tokens[-1]:.3f}B', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add progress annotation
    optimal_tokens_num = float(model_info["optimal_tokens"].replace('B', ''))
    progress_pct = (cumulative_tokens[-1] / optimal_tokens_num) * 100
    ax3.text(0.98, 0.02, f'{progress_pct:.1f}% of optimal ({model_info["optimal_tokens"]})', 
            transform=ax3.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: MFU
    ax4 = fig.add_subplot(gs[1, 0])
    if mfu_values:
        ax4.plot(mfu_iters, mfu_values, 'orange', linewidth=2, marker='o', markersize=2)
        
        avg_mfu = np.mean(mfu_values)
        ax4.axhline(y=avg_mfu, color='blue', linestyle='--', alpha=0.6, 
                   label=f'Average: {avg_mfu:.2f}%')
        
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('MFU (%)', fontsize=11)
        ax4.set_title(f'Model FLOPs Utilization (Avg: {avg_mfu:.1f}%)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        mfu_min = min(mfu_values)
        mfu_max = max(mfu_values)
        mfu_range = mfu_max - mfu_min
        margin = max(2, mfu_range * 0.2)
        ax4.set_ylim([max(0, mfu_min - margin), mfu_max + margin])
    
    # Plot 5: Throughput
    ax5 = fig.add_subplot(gs[1, 1])
    if tokens_values:
        ax5.plot(tokens_iters, tokens_values, 'brown', linewidth=2, marker='o', markersize=2)
        
        avg_tokens = np.mean(tokens_values)
        ax5.axhline(y=avg_tokens, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_tokens:.0f} tokens/s')
        
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Tokens per Second', fontsize=11)
        ax5.set_title(f'Training Throughput (Avg: {avg_tokens:.0f} tokens/s)', 
                     fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Memory
    ax6 = fig.add_subplot(gs[1, 2])
    if mem_alloc_values:
        ax6.plot(mem_iters, mem_alloc_values, 'steelblue', linewidth=2, 
                label='Allocated', marker='o', markersize=2)
        ax6.plot(mem_iters, mem_peak_values, 'darkred', linewidth=2, 
                label='Peak', marker='s', markersize=2)
        
        gpu_capacity = 48
        ax6.axhline(y=gpu_capacity, color='red', linestyle='--', alpha=0.4, 
                   label=f'GPU Capacity: {gpu_capacity} GB')
        
        ax6.set_xlabel('Iteration', fontsize=11)
        ax6.set_ylabel('Memory (GB)', fontsize=11)
        ax6.set_title(f'GPU Memory Usage (Peak: {max(mem_peak_values):.1f} GB / {gpu_capacity} GB)', 
                     fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, gpu_capacity * 1.1])
    
    # Save with model name
    output_path = json_path.parent / f"{model_info['name']}_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()
    
    # Print summary
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Avg MFU: {np.mean(mfu_values):.2f}%" if mfu_values else "   MFU: N/A")
    print(f"   Avg Throughput: {np.mean(tokens_values):.0f} tokens/s" if tokens_values else "   Throughput: N/A")

def main():
    saves_dir = Path("../saves")
    
    print("="*70)
    print("PLOTTING ALL TRAINING RUNS")
    print("="*70)
    
    # Plot each run
    plotted_count = 0
    for run_name, model_info in MODEL_INFO.items():
        json_path = saves_dir / f"{run_name}.json"
        
        if not json_path.exists():
            print(f"\n⚠️  Skipping {run_name} (file not found)")
            continue
        
        try:
            plot_single_run(json_path, model_info)
            plotted_count += 1
        except Exception as e:
            print(f"❌ Error plotting {run_name}: {e}")
    
    print("\n" + "="*70)
    print(f"✅ GENERATED {plotted_count} PLOTS")
    print("="*70)
    print(f"\nOutput location: {saves_dir.resolve()}")
    print("\nGenerated files:")
    for model_info in MODEL_INFO.values():
        plot_name = f"{model_info['name']}_analysis.png"
        plot_path = saves_dir / plot_name
        if plot_path.exists():
            print(f"  ✓ {plot_name}")

if __name__ == '__main__':
    main()

