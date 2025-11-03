#!/usr/bin/env python3
"""
One-time plotting script to compare GPT-2 and LLaMA training runs.
Compares first 200 iterations only.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the JSON files
gpt2_path = Path("../out-gpt2/run_20251104_042513.json")
llama_path = Path("../out-llama/run_20251104_060817.json")

with open(gpt2_path, 'r') as f:
    gpt2_data = json.load(f)

with open(llama_path, 'r') as f:
    llama_data = json.load(f)

# Extract data for first 200 iterations
def extract_data(data, max_iter=200):
    iterations = []
    losses = []
    mfus = []
    times = []
    memory_allocated = []
    
    for entry in data['training_iterations']:
        iter_num = entry['iter']
        if iter_num > max_iter:
            break
        if iter_num == 0:  # Skip first iteration (initialization)
            continue
            
        iterations.append(iter_num)
        losses.append(entry['loss'])
        times.append(entry['time_ms'] / 1000.0)  # Convert to seconds
        
        # Extract MFU
        if isinstance(entry.get('mfu'), dict):
            mfus.append(entry['mfu']['mfu_percent'])
        else:
            mfus.append(None)
        
        # Extract memory
        if 'memory' in entry:
            memory_allocated.append(entry['memory']['allocated_gb'])
        else:
            memory_allocated.append(None)
    
    return {
        'iterations': iterations,
        'losses': losses,
        'mfus': mfus,
        'times': times,
        'memory': memory_allocated
    }

gpt2 = extract_data(gpt2_data)
llama = extract_data(llama_data)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GPT-2 vs LLaMA Architecture Training Comparison\n(First 200 Iterations on Shakespeare Dataset)', 
             fontsize=14, fontweight='bold')

# Plot 1: Loss over iterations
ax1 = axes[0, 0]
ax1.plot(gpt2['iterations'], gpt2['losses'], label='GPT-2 (1 GPU)', marker='o', markersize=3, linewidth=1.5)
ax1.plot(llama['iterations'], llama['losses'], label='LLaMA (2 GPUs, DDP)', marker='s', markersize=3, linewidth=1.5)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: MFU over iterations
ax2 = axes[0, 1]
gpt2_mfus = [m for m in gpt2['mfus'] if m is not None]
llama_mfus = [m for m in llama['mfus'] if m is not None]
gpt2_iters = [gpt2['iterations'][i] for i, m in enumerate(gpt2['mfus']) if m is not None]
llama_iters = [llama['iterations'][i] for i, m in enumerate(llama['mfus']) if m is not None]

ax2.plot(gpt2_iters, gpt2_mfus, label='GPT-2 (1 GPU)', marker='o', markersize=3, linewidth=1.5)
ax2.plot(llama_iters, llama_mfus, label='LLaMA (2 GPUs, DDP)', marker='s', markersize=3, linewidth=1.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MFU (%)')
ax2.set_title('Model FLOPs Utilization (MFU)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add average MFU as horizontal lines
if gpt2_mfus:
    avg_gpt2_mfu = np.mean(gpt2_mfus)
    ax2.axhline(y=avg_gpt2_mfu, color='C0', linestyle='--', alpha=0.5, 
                label=f'GPT-2 Avg: {avg_gpt2_mfu:.2f}%')
if llama_mfus:
    avg_llama_mfu = np.mean(llama_mfus)
    ax2.axhline(y=avg_llama_mfu, color='C1', linestyle='--', alpha=0.5,
                label=f'LLaMA Avg: {avg_llama_mfu:.2f}%')
ax2.legend()

# Plot 3: Time per iteration
ax3 = axes[1, 0]
ax3.plot(gpt2['iterations'], gpt2['times'], label='GPT-2 (1 GPU)', marker='o', markersize=3, linewidth=1.5)
ax3.plot(llama['iterations'], llama['times'], label='LLaMA (2 GPUs, DDP)', marker='s', markersize=3, linewidth=1.5)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Time (seconds)')
ax3.set_title('Time per Iteration')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Memory usage
ax4 = axes[1, 1]
gpt2_mem = [m for m in gpt2['memory'] if m is not None]
llama_mem = [m for m in llama['memory'] if m is not None]
gpt2_mem_iters = [gpt2['iterations'][i] for i, m in enumerate(gpt2['memory']) if m is not None]
llama_mem_iters = [llama['iterations'][i] for i, m in enumerate(llama['memory']) if m is not None]

ax4.plot(gpt2_mem_iters, gpt2_mem, label='GPT-2 (1 GPU)', marker='o', markersize=3, linewidth=1.5)
ax4.plot(llama_mem_iters, llama_mem, label='LLaMA (2 GPUs, DDP)', marker='s', markersize=3, linewidth=1.5)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Allocated Memory (GB)')
ax4.set_title('GPU Memory Usage (Allocated)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Plot saved as 'training_comparison.png'")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS (First 200 Iterations)")
print("="*60)

print("\n📊 GPT-2 Architecture (1 GPU):")
print(f"   Final Loss:     {gpt2['losses'][-1]:.4f}")
print(f"   Avg MFU:        {np.mean(gpt2_mfus):.2f}%")
print(f"   Avg Time/Iter:  {np.mean(gpt2['times']):.2f}s")
print(f"   Avg Memory:     {np.mean(gpt2_mem):.2f} GB")

print("\n📊 LLaMA Architecture (2 GPUs, DDP):")
print(f"   Final Loss:     {llama['losses'][-1]:.4f}")
print(f"   Avg MFU:        {np.mean(llama_mfus):.2f}%")
print(f"   Avg Time/Iter:  {np.mean(llama['times']):.2f}s")
print(f"   Avg Memory:     {np.mean(llama_mem):.2f} GB")

print("\n" + "="*60)


