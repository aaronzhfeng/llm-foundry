#!/usr/bin/env python3
"""
Visualization script for training logs.
Reads JSON logs and generates plots.
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will not be generated")


def load_log(log_file):
    """Load a training log from JSON file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def print_summary(log_data):
    """Print a summary of the training run."""
    print(f"\n{'='*60}")
    print(f"Training Run Summary: {log_data['run_name']}")
    print(f"{'='*60}")
    
    # Config
    print(f"\nConfiguration:")
    print(f"  Dataset: {log_data['config'].get('dataset', 'N/A')}")
    print(f"  Model: {log_data['config']['n_layer']}L / {log_data['config']['n_head']}H / {log_data['config']['n_embd']}D")
    print(f"  Batch Size: {log_data['config']['batch_size']}")
    print(f"  Learning Rate: {log_data['config']['learning_rate']}")
    print(f"  Dtype: {log_data['config']['dtype']}")
    print(f"  Compile: {log_data['config']['compile']}")
    
    if 'use_zero1' in log_data['config']:
        print(f"  ZeRO-1: {log_data['config']['use_zero1']}")
    
    # Summary stats
    if 'summary' in log_data:
        summary = log_data['summary']
        print(f"\nTraining Summary:")
        print(f"  Total Iterations: {summary['total_iterations']}")
        print(f"  Final Train Loss: {summary['final_train_loss']:.4f}")
        print(f"  Best Val Loss: {summary['best_val_loss']:.4f}")
        print(f"  Avg Time/Iter: {summary['avg_time_ms']:.2f} ms")
        print(f"  Avg MFU: {summary['avg_mfu']:.2f}%")
        print(f"  Checkpoints: {summary['total_checkpoints']}")
    
    print(f"\n{'='*60}\n")


def plot_training(log_data, output_file='training_analysis.png'):
    """Generate training plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Analysis: {log_data['run_name']}", fontsize=14, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    train_iters = [x['iter'] for x in log_data['training_iterations']]
    train_losses = [x['loss'] for x in log_data['training_iterations']]
    ax1.plot(train_iters, train_losses, linewidth=2, color='#1f77b4', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. Train vs Val Loss
    ax2 = axes[0, 1]
    if log_data['eval_steps']:
        eval_iters = [x['iter'] for x in log_data['eval_steps']]
        train_eval_losses = [x['train_loss'] for x in log_data['eval_steps']]
        val_losses = [x['val_loss'] for x in log_data['eval_steps']]
        
        ax2.plot(eval_iters, train_eval_losses, marker='o', label='Train', linewidth=2)
        ax2.plot(eval_iters, val_losses, marker='s', label='Val', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Train vs Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No eval data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Train vs Validation Loss')
    
    # 3. Time per Iteration
    ax3 = axes[1, 0]
    times = [x['time_ms'] for x in log_data['training_iterations']]
    ax3.plot(train_iters, times, linewidth=1, color='#ff7f0e', alpha=0.6)
    ax3.axhline(y=sum(times)/len(times), color='r', linestyle='--', label=f'Avg: {sum(times)/len(times):.1f}ms')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Time per Iteration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MFU (Model FLOPs Utilization)
    ax4 = axes[1, 1]
    mfus = [(x['iter'], x['mfu']) for x in log_data['training_iterations'] if x['mfu'] > 0]
    if mfus:
        mfu_iters = [x[0] for x in mfus]
        mfu_values = [x[1] for x in mfus]
        ax4.plot(mfu_iters, mfu_values, linewidth=2, color='#2ca02c', alpha=0.8)
        ax4.axhline(y=sum(mfu_values)/len(mfu_values), color='r', linestyle='--', 
                    label=f'Avg: {sum(mfu_values)/len(mfu_values):.2f}%')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('MFU (%)')
        ax4.set_title('Model FLOPs Utilization')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No MFU data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model FLOPs Utilization')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        # Find the latest log file
        log_files = sorted(Path('out').glob('run_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("No log files found in out/ directory")
            print("Usage: python visualize_training.py <log_file.json>")
            return
        log_file = log_files[0]
        print(f"Using latest log file: {log_file}")
    else:
        log_file = sys.argv[1]
    
    # Load and display
    log_data = load_log(log_file)
    print_summary(log_data)
    
    # Generate plots
    output_file = Path(log_file).stem + '_analysis.png'
    plot_training(log_data, output_file)


if __name__ == '__main__':
    main()

