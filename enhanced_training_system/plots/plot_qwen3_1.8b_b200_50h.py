#!/usr/bin/env python3
"""
Plot comprehensive training analysis for Qwen3 1.8B on 8× B200 (50h production run).
Shows loss, perplexity, MFU, throughput, learning rate, memory, and evaluation checkpoints.
Optimized for 162K iteration full training analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

def load_run(json_path):
    """Load training log JSON."""
    print(f"Loading {json_path} (this may take a moment for large files)...")
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_training_data(data, sample_rate=10):
    """
    Extract training iteration data with optional sampling for large runs.
    sample_rate: Only keep every Nth point for plotting efficiency.
    """
    iterations = []
    losses = []
    mfus = []
    times_ms = []
    tokens_per_sec = []
    achieved_tflops = []
    memory_alloc = []
    memory_peak = []
    memory_reserved = []
    
    for idx, entry in enumerate(data['training_iterations']):
        iter_num = entry['iter']
        if iter_num == 0:  # Skip first iteration (initialization)
            continue
        
        # Sample data for large runs (keep every Nth point)
        if idx % sample_rate != 0:
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
                achieved_tflops.append(entry['mfu'].get('achieved_tflops', 0))
            else:
                mfus.append(None)
                tokens_per_sec.append(None)
                achieved_tflops.append(None)
        else:
            mfus.append(None)
            tokens_per_sec.append(None)
            achieved_tflops.append(None)
        
        # Extract memory
        if 'memory' in entry:
            memory_alloc.append(entry['memory']['allocated_gb'])
            memory_peak.append(entry['memory']['max_allocated_gb'])
            memory_reserved.append(entry['memory'].get('reserved_gb', 0))
        else:
            memory_alloc.append(None)
            memory_peak.append(None)
            memory_reserved.append(None)
    
    return {
        'iterations': iterations,
        'losses': losses,
        'mfus': mfus,
        'times_ms': times_ms,
        'tokens_per_sec': tokens_per_sec,
        'achieved_tflops': achieved_tflops,
        'memory_alloc': memory_alloc,
        'memory_peak': memory_peak,
        'memory_reserved': memory_reserved,
    }

def extract_eval_data(data):
    """Extract evaluation checkpoint data."""
    if not data.get('eval_steps'):
        return None
    
    eval_iters = []
    train_losses = []
    val_losses = []
    lrs = []
    timestamps = []
    
    for entry in data['eval_steps']:
        eval_iters.append(entry['iter'])
        train_losses.append(entry.get('train_loss'))
        val_losses.append(entry['val_loss'])
        if 'lr' in entry:
            lrs.append(entry['lr'])
        if 'timestamp' in entry:
            timestamps.append(entry['timestamp'])
    
    return {
        'iters': eval_iters,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'lrs': lrs,
        'timestamps': timestamps
    }

def compute_learning_rate(iter_num, config):
    """Compute learning rate at given iteration using cosine schedule."""
    lr = config.get('learning_rate', 3e-4)
    warmup = config.get('warmup_iters', 2000)
    decay_iters = config.get('lr_decay_iters', 162000)
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

def compute_training_time(data):
    """Compute total training time from timestamps."""
    try:
        start_str = data['start_time']
        end_str = data['end_time']
        start_time = datetime.fromisoformat(start_str)
        end_time = datetime.fromisoformat(end_str)
        return (end_time - start_time).total_seconds()
    except:
        return None

def main():
    # Determine input file
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Default to the 50h production run
        json_path = Path(__file__).parent.parent / "saves" / "run_20251122_114701.json"
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print(f"Usage: {sys.argv[0]} [path/to/run.json]")
        sys.exit(1)
    
    data = load_run(json_path)
    
    # Extract data (sample every 10th point for plotting efficiency)
    train_data = extract_training_data(data, sample_rate=10)
    eval_data = extract_eval_data(data)
    config = data.get('config', {})
    startup = data.get('startup_info', {})
    summary = data.get('summary', {})
    
    # Get model info
    model_info = startup.get('model', {})
    total_params = model_info.get('total_params', 0)
    
    # Get hardware info
    hardware = startup.get('hardware', {})
    num_gpus = hardware.get('num_gpus', 8)
    gpu_name = hardware.get('gpu_name', 'B200')
    gpu_memory_gb = hardware.get('gpu_memory_gb', 192)
    
    # Compute derived metrics
    iters = np.array(train_data['iterations'])
    losses = np.array(train_data['losses'])
    perplexities = np.exp(losses)
    
    # Compute learning rates for all iterations
    lrs = [compute_learning_rate(i, config) for i in iters]
    
    # Compute cumulative tokens
    block_size = config.get('block_size', 2048)
    batch_size = config.get('batch_size', 22)
    grad_accum_global = config.get('gradient_accumulation_steps_global', 16)
    tokens_per_iter = block_size * batch_size * grad_accum_global
    cumulative_tokens = iters * tokens_per_iter / 1e9  # In billions
    
    # Target tokens (117B from run name)
    target_tokens = 117  # 117B tokens for 50h production
    
    # Filter out None values for MFU and memory
    mfu_iters = [iters[i] for i, m in enumerate(train_data['mfus']) if m is not None]
    mfu_values = [m for m in train_data['mfus'] if m is not None]
    
    tokens_iters = [iters[i] for i, t in enumerate(train_data['tokens_per_sec']) if t is not None and t > 0]
    tokens_values = [t for t in train_data['tokens_per_sec'] if t is not None and t > 0]
    
    tflops_iters = [iters[i] for i, t in enumerate(train_data['achieved_tflops']) if t is not None and t > 0]
    tflops_values = [t for t in train_data['achieved_tflops'] if t is not None and t > 0]
    
    mem_iters = [iters[i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    mem_alloc_values = [m for m in train_data['memory_alloc'] if m is not None]
    mem_peak_values = [train_data['memory_peak'][i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    mem_reserved_values = [train_data['memory_reserved'][i] for i, m in enumerate(train_data['memory_reserved']) if m is not None]
    
    # Valid iteration times
    valid_times = [(iters[i], t/1000) for i, t in enumerate(train_data['times_ms']) if t is not None]
    
    # Compute total training time
    total_time_sec = compute_training_time(data)
    total_time_hours = total_time_sec / 3600 if total_time_sec else summary.get('avg_time_ms', 0) * len(iters) / 1000 / 3600
    
    # ======================== Create Figure ========================
    fig = plt.figure(figsize=(22, 14))
    
    # Model name and configuration
    model_name = "Qwen3 1.8B"
    attention_backend = config.get('attention_backend', 'flash_attn_2')
    compile_status = "Compiled" if config.get('compile', False) else "No Compile"
    precision = config.get('dtype', 'bfloat16')
    parallelism = hardware.get('parallelism', 'DDP+ZeRO-1')
    
    fig.suptitle(f'{model_name} Training on {num_gpus}× {gpu_name} — 162K Iterations (50h Production Run)\n'
                 f'{total_params/1e9:.2f}B params | {attention_backend} | {compile_status} | {precision} | {parallelism} | '
                 f'Total Tokens: {cumulative_tokens[-1]:.1f}B / {target_tokens}B ({cumulative_tokens[-1]/target_tokens*100:.1f}%)',
                 fontsize=15, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, top=0.91, bottom=0.05, left=0.05, right=0.97)
    
    # Color scheme
    loss_color = '#2E86AB'      # Blue
    perp_color = '#28A745'      # Green
    lr_color = '#6F42C1'        # Purple
    mfu_color = '#FD7E14'       # Orange
    tflops_color = '#E83E8C'    # Pink
    throughput_color = '#795548' # Brown
    mem_alloc_color = '#4682B4' # Steel blue
    mem_peak_color = '#8B0000'  # Dark red
    mem_reserved_color = '#FFA500' # Orange
    time_color = '#2E7D32'      # Dark green
    val_color = '#DC3545'       # Red for validation
    
    # ========== Plot 1: Loss & Perplexity (Dual Y-axis) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    # Training loss on left axis
    line1 = ax1.plot(iters, losses, color=loss_color, linewidth=1.5, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=10, color=loss_color)
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.grid(True, alpha=0.3)
    
    # Mark eval checkpoints
    if eval_data and eval_data['iters']:
        ax1.scatter(eval_data['iters'], eval_data['val_loss'], 
                   color=val_color, s=80, zorder=5, marker='*', 
                   label=f"Val Loss (best: {min(eval_data['val_loss']):.3f})")
    
    # Perplexity on right axis (log scale)
    line2 = ax1_twin.plot(iters, perplexities, color=perp_color, linewidth=1.2, 
                          label='Perplexity', alpha=0.6, linestyle='--')
    ax1_twin.set_ylabel('Perplexity = exp(loss)', fontsize=10, color=perp_color)
    ax1_twin.tick_params(axis='y', labelcolor=perp_color)
    ax1_twin.set_yscale('log')
    
    ax1.set_title('Loss & Perplexity', fontsize=11, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=8)
    
    # Add final loss annotation
    ax1.text(0.98, 0.02, f"Final: {losses[-1]:.3f} (PPL: {perplexities[-1]:.1f})",
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== Plot 2: Learning Rate Schedule ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iters, lrs, color=lr_color, linewidth=2)
    
    # Mark warmup end
    warmup = config.get('warmup_iters', 2000)
    if max(iters) > warmup:
        ax2.axvline(x=warmup, color='red', linestyle='--', alpha=0.5, label=f'Warmup End: {warmup}')
    
    # Mark decay end
    decay_iters = config.get('lr_decay_iters', 162000)
    ax2.axvline(x=decay_iters, color='green', linestyle='--', alpha=0.5, label=f'Decay End: {decay_iters}')
    
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Learning Rate', fontsize=10)
    ax2.set_title('Learning Rate Schedule (Cosine Decay)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # ========== Plot 3: Cumulative Tokens ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iters, cumulative_tokens, color='teal', linewidth=2)
    
    # Target line
    ax3.axhline(y=target_tokens, color='red', linestyle='--', alpha=0.6, 
                label=f'Target: {target_tokens}B tokens')
    
    # Mark eval checkpoints
    if eval_data and eval_data['iters']:
        eval_tokens = [i * tokens_per_iter / 1e9 for i in eval_data['iters']]
        ax3.scatter(eval_data['iters'], eval_tokens, color=val_color, s=60, zorder=5, marker='o', 
                   label='Checkpoints')
    
    ax3.set_xlabel('Iteration', fontsize=10)
    ax3.set_ylabel('Cumulative Tokens (Billions)', fontsize=10)
    ax3.set_title(f'Tokens Processed: {cumulative_tokens[-1]:.1f}B / {target_tokens}B', 
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add progress annotation
    progress_pct = cumulative_tokens[-1] / target_tokens * 100
    ax3.text(0.02, 0.98, f'{progress_pct:.1f}% complete', 
            transform=ax3.transAxes, ha='left', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== Plot 4: MFU Over Time ==========
    ax4 = fig.add_subplot(gs[1, 0])
    if mfu_values:
        ax4.plot(mfu_iters, mfu_values, color=mfu_color, linewidth=1.5, alpha=0.8)
        
        # Average MFU
        avg_mfu = np.mean(mfu_values)
        ax4.axhline(y=avg_mfu, color='blue', linestyle='--', alpha=0.6, 
                   label=f'Average: {avg_mfu:.2f}%')
        
        # B200 target (optimistic)
        target_mfu = 40
        ax4.axhline(y=target_mfu, color='green', linestyle=':', alpha=0.4,
                   label=f'Target: {target_mfu}%')
        
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.set_ylabel('MFU (%)', fontsize=10)
        ax4.set_title(f'Model FLOPs Utilization (Avg: {avg_mfu:.2f}%)', 
                     fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        mfu_min = min(mfu_values)
        mfu_max = max(mfu_values)
        mfu_range = mfu_max - mfu_min
        margin = max(5, mfu_range * 0.3)
        ax4.set_ylim([max(0, mfu_min - margin), min(100, max(mfu_max + margin, target_mfu + 10))])
    
    # ========== Plot 5: Throughput (Tokens/sec) ==========
    ax5 = fig.add_subplot(gs[1, 1])
    if tokens_values:
        ax5.plot(tokens_iters, [t/1000 for t in tokens_values], color=throughput_color, 
                linewidth=1.5, alpha=0.8)
        
        # Average throughput
        avg_tokens = np.mean(tokens_values)
        ax5.axhline(y=avg_tokens/1000, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_tokens/1000:.0f}K tokens/s')
        
        ax5.set_xlabel('Iteration', fontsize=10)
        ax5.set_ylabel('Tokens per Second (Thousands)', fontsize=10)
        ax5.set_title(f'Training Throughput (Avg: {avg_tokens/1000:.1f}K tokens/s)', 
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # ========== Plot 6: Achieved TFLOPS ==========
    ax6 = fig.add_subplot(gs[1, 2])
    if tflops_values:
        ax6.plot(tflops_iters, tflops_values, color=tflops_color, linewidth=1.5, alpha=0.8)
        
        # Average TFLOPS
        avg_tflops = np.mean(tflops_values)
        ax6.axhline(y=avg_tflops, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_tflops:.0f} TFLOPS')
        
        # B200 peak (bf16 tensor cores) - 8× B200 = 8 × 2250 TFLOPS = 18000 TFLOPS
        peak_tflops = 18000
        ax6.axhline(y=peak_tflops, color='red', linestyle=':', alpha=0.4,
                   label=f'Peak: {peak_tflops/1000:.0f}K TFLOPS')
        
        ax6.set_xlabel('Iteration', fontsize=10)
        ax6.set_ylabel('Achieved TFLOPS', fontsize=10)
        ax6.set_title(f'Compute Performance (Avg: {avg_tflops:.0f} TFLOPS)', 
                     fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, peak_tflops * 1.05])  # Start from 0
    
    # ========== Plot 7: Memory Usage ==========
    ax7 = fig.add_subplot(gs[2, 0])
    if mem_alloc_values:
        ax7.plot(mem_iters, mem_alloc_values, color=mem_alloc_color, linewidth=1.5, 
                label='Allocated', alpha=0.8)
        ax7.plot(mem_iters, mem_peak_values, color=mem_peak_color, linewidth=1.5, 
                label='Peak', alpha=0.8)
        ax7.plot(mem_iters, mem_reserved_values, color=mem_reserved_color, linewidth=1.2, 
                label='Reserved', alpha=0.5)
        
        # GPU capacity
        ax7.axhline(y=gpu_memory_gb, color='red', linestyle='--', alpha=0.4, 
                   label=f'GPU Capacity: {gpu_memory_gb:.0f} GB')
        
        ax7.set_xlabel('Iteration', fontsize=10)
        ax7.set_ylabel('Memory (GB)', fontsize=10)
        ax7.set_title(f'GPU Memory Usage (Peak: {max(mem_peak_values):.1f} / {gpu_memory_gb:.0f} GB = {max(mem_peak_values)/gpu_memory_gb*100:.1f}%)', 
                     fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8, loc='right')
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, gpu_memory_gb * 1.05])
    
    # ========== Plot 8: Iteration Time ==========
    ax8 = fig.add_subplot(gs[2, 1])
    if valid_times:
        time_iters, time_values = zip(*valid_times)
        ax8.plot(time_iters, time_values, color=time_color, linewidth=1.5, alpha=0.8)
        
        avg_time = np.mean(time_values)
        ax8.axhline(y=avg_time, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_time:.2f}s')
        
        ax8.set_xlabel('Iteration', fontsize=10)
        ax8.set_ylabel('Time (seconds)', fontsize=10)
        ax8.set_title(f'Time per Iteration (Avg: {avg_time:.2f}s)', 
                     fontsize=11, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    
    # ========== Plot 9: Validation Loss Progression ==========
    ax9 = fig.add_subplot(gs[2, 2])
    
    if eval_data and eval_data['val_loss']:
        val_iters = eval_data['iters']
        val_losses = eval_data['val_loss']
        
        ax9.plot(val_iters, val_losses, color=val_color, linewidth=2.5, 
                marker='o', markersize=8, label='Validation Loss')
        
        # Add training loss at eval points if available
        if eval_data['train_loss'] and eval_data['train_loss'][0] is not None:
            train_losses_at_eval = [t for t in eval_data['train_loss'] if t is not None]
            train_iters_at_eval = [val_iters[i] for i, t in enumerate(eval_data['train_loss']) if t is not None]
            ax9.plot(train_iters_at_eval, train_losses_at_eval, color=loss_color, linewidth=2.5, 
                    marker='s', markersize=8, label='Train Loss @ Eval', alpha=0.7)
        
        ax9.set_xlabel('Iteration', fontsize=10)
        ax9.set_ylabel('Loss', fontsize=10)
        ax9.set_title(f'Checkpoint Losses (Best Val: {min(val_losses):.4f})', 
                     fontsize=11, fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)
        
        # Annotate best checkpoint
        best_idx = np.argmin(val_losses)
        ax9.annotate(f'Best: {val_losses[best_idx]:.4f}\n@ iter {val_iters[best_idx]}',
                    xy=(val_iters[best_idx], val_losses[best_idx]),
                    xytext=(val_iters[best_idx] - 20000, val_losses[best_idx] + 0.05),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax9.axis('off')
        # Create summary text box instead
        summary_lines = []
        summary_lines.append(f"🏗️  MODEL SUMMARY")
        summary_lines.append(f"  • Params: {total_params/1e9:.3f}B ({total_params/1e6:.0f}M)")
        summary_lines.append(f"  • Layers: {config.get('n_layer', '?')} × {config.get('n_embd', '?')}D")
        summary_lines.append(f"  • Heads: {config.get('n_head', '?')}")
        summary_lines.append(f"  • Context: {config.get('block_size', '?')} tokens")
        summary_lines.append(f"")
        summary_lines.append(f"⚡ PERFORMANCE")
        if mfu_values:
            summary_lines.append(f"  • MFU: {np.mean(mfu_values):.2f}%")
        if tokens_values:
            summary_lines.append(f"  • Throughput: {np.mean(tokens_values)/1000:.1f}K tok/s")
        summary_lines.append(f"")
        summary_lines.append(f"📊 TRAINING")
        summary_lines.append(f"  • Iterations: {len(iters)}")
        summary_lines.append(f"  • Tokens: {cumulative_tokens[-1]:.1f}B / {target_tokens}B")
        summary_lines.append(f"  • Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
        summary_lines.append(f"  • Time: {total_time_hours:.1f} hours")
        
        summary_text = '\n'.join(summary_lines)
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # ======================== Save Figure ========================
    output_path = json_path.parent / f"{json_path.stem}_qwen3_b200_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved: {output_path}")
    
    # ======================== Print Summary ========================
    print("\n" + "="*85)
    print("QWEN3 1.8B × 8× B200 — 50H PRODUCTION RUN ANALYSIS")
    print("="*85)
    
    # Model info
    print(f"\n🏗️  MODEL:")
    print(f"   Total params:       {total_params/1e9:.3f}B ({total_params/1e6:.0f}M)")
    print(f"   Architecture:       {config.get('n_layer', '?')}L-{config.get('n_head', '?')}H-{config.get('n_embd', '?')}D")
    print(f"   FFN type:           {config.get('ffn_type', '?')} (intermediate: {config.get('intermediate_size', '?')})")
    print(f"   Hardware:           {num_gpus}× {gpu_name} ({gpu_memory_gb:.0f} GB/GPU)")
    print(f"   Parallelism:        {parallelism}")
    
    # Optimization settings
    print(f"\n🔧 OPTIMIZATION:")
    print(f"   Attention:          {attention_backend}")
    print(f"   torch.compile():    {config.get('compile', False)}")
    print(f"   Precision:          {precision}")
    print(f"   Batch size:         {batch_size} (per GPU)")
    print(f"   Grad accum:         {config.get('gradient_accumulation_steps', '?')} (global: {grad_accum_global})")
    print(f"   Effective batch:    {tokens_per_iter:,} tokens/iter")
    
    # Training progress
    print(f"\n📈 TRAINING PROGRESS:")
    print(f"   Iterations:         {summary.get('total_iterations', len(iters))} ({iters[0]} → {summary.get('final_iter', iters[-1])})")
    print(f"   Tokens processed:   {cumulative_tokens[-1]:.2f}B / {target_tokens}B ({cumulative_tokens[-1]/target_tokens*100:.1f}%)")
    print(f"   Checkpoints:        {summary.get('total_checkpoints', len(eval_data['iters']) if eval_data else 0)}")
    
    # Loss metrics
    print(f"\n📉 LOSS & PERPLEXITY:")
    print(f"   Initial:            {losses[0]:.4f} (PPL: {perplexities[0]:.1f})")
    print(f"   Final (train):      {summary.get('final_train_loss', losses[-1]):.4f} (PPL: {np.exp(summary.get('final_train_loss', losses[-1])):.1f})")
    print(f"   Best (val):         {summary.get('best_val_loss', min(eval_data['val_loss']) if eval_data else 'N/A'):.4f}")
    print(f"   Loss reduction:     {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    # Performance
    if mfu_values:
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Avg MFU:            {summary.get('avg_mfu', np.mean(mfu_values)):.2f}% (target: 35-50%)")
        print(f"   Min/Max MFU:        {min(mfu_values):.2f}% / {max(mfu_values):.2f}%")
    if tokens_values:
        print(f"   Avg Throughput:     {np.mean(tokens_values):,.0f} tokens/s ({np.mean(tokens_values)/1000:.1f}K)")
    if tflops_values:
        print(f"   Avg TFLOPS:         {np.mean(tflops_values):,.0f} / {peak_tflops:,} ({np.mean(tflops_values)/peak_tflops*100:.2f}%)")
    if valid_times:
        avg_time_s = summary.get('avg_time_ms', np.mean([t for _, t in valid_times]) * 1000) / 1000
        print(f"   Avg Time/Iter:      {avg_time_s:.2f}s")
    
    # Memory
    if mem_peak_values:
        print(f"\n💾 MEMORY (per GPU):")
        print(f"   Peak allocated:     {max(mem_peak_values):.2f} GB / {gpu_memory_gb:.0f} GB ({max(mem_peak_values)/gpu_memory_gb*100:.1f}%)")
        print(f"   Avg allocated:      {np.mean(mem_alloc_values):.2f} GB")
        if mem_reserved_values:
            print(f"   Reserved (cached):  {max(mem_reserved_values):.2f} GB")
    
    # Training time
    print(f"\n⏱️  TIME:")
    print(f"   Total training:     {total_time_hours:.1f} hours ({total_time_hours/24:.1f} days)")
    if tokens_values:
        tokens_per_hour = np.mean(tokens_values) * 3600
        remaining_tokens = (target_tokens - cumulative_tokens[-1]) * 1e9
        if remaining_tokens > 0:
            time_remaining = remaining_tokens / tokens_per_hour
            print(f"   Est. remaining:     {time_remaining:.1f} hours ({time_remaining/24:.1f} days)")
    
    # Eval checkpoints
    if eval_data and eval_data['iters']:
        print(f"\n📊 CHECKPOINTS:")
        for i, (it, vl) in enumerate(zip(eval_data['iters'], eval_data['val_loss'])):
            marker = " ← BEST" if vl == min(eval_data['val_loss']) else ""
            print(f"   Iter {it:>6}: val_loss = {vl:.4f}{marker}")
    
    print("\n" + "="*85 + "\n")

if __name__ == '__main__':
    main()

