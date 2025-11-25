#!/usr/bin/env python3
"""
Plot comparison between custom-trained Qwen3-1.8B and official Qwen2.5-1.5B.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results data
benchmarks = ['OpenBookQA', 'ARC-Challenge', 'ARC-Easy']

# Your custom-trained model (160k iterations)
custom_qwen3_1_8b = [26.40, 27.22, 52.57]

# Official Qwen2.5-1.5B (pretrained baseline)
official_qwen2_5_1_5b = [40.20, 43.69, 73.57]

# Random baseline (4-choice = 25%)
random_baseline = [25.0, 25.0, 25.0]

x = np.arange(len(benchmarks))
width = 0.28

fig, ax = plt.subplots(figsize=(12, 9))

# Create bars
bars1 = ax.bar(x - width, custom_qwen3_1_8b, width, label='Custom Qwen3-1.8B (160k iter, ~26B tokens)', 
               color='#4ECDC4', edgecolor='#2C3E50', linewidth=1.5)
bars2 = ax.bar(x, official_qwen2_5_1_5b, width, label='Official Qwen2.5-1.5B (Full Pretrain)', 
               color='#FF6B6B', edgecolor='#2C3E50', linewidth=1.5)
bars3 = ax.bar(x + width, random_baseline, width, label='Random Baseline (25%)', 
               color='#95A5A6', edgecolor='#2C3E50', linewidth=1.5, alpha=0.6)

# Customize
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Benchmark', fontsize=14, fontweight='bold')
ax.set_title('Model Comparison: Custom Training vs Official Pretrained\n(Log-Probability Evaluation)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 85)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Add training info annotation
training_info = """
Training Details:
• Custom: 160k iterations (~26B tokens)
• Official: Full pretrain (~18T tokens)
• Gap: Expected due to ~700x less training data
"""
ax.text(0.02, 0.98, training_info, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/raid/zhf004/llm_TII/evaluation_system/benchmark_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('/raid/zhf004/llm_TII/evaluation_system/benchmark_comparison.pdf', bbox_inches='tight')
print("✅ Saved: benchmark_comparison.png and benchmark_comparison.pdf")

# Also print summary table
print("\n" + "="*70)
print("BENCHMARK COMPARISON SUMMARY")
print("="*70)
print(f"{'Benchmark':<20} {'Custom (160k)':<18} {'Official':<18} {'Gap':<12}")
print("-"*70)
for i, bench in enumerate(benchmarks):
    gap = official_qwen2_5_1_5b[i] - custom_qwen3_1_8b[i]
    print(f"{bench:<20} {custom_qwen3_1_8b[i]:.2f}%{'':<12} {official_qwen2_5_1_5b[i]:.2f}%{'':<12} -{gap:.2f}%")
print("-"*70)
avg_custom = np.mean(custom_qwen3_1_8b)
avg_official = np.mean(official_qwen2_5_1_5b)
print(f"{'Average':<20} {avg_custom:.2f}%{'':<12} {avg_official:.2f}%{'':<12} -{avg_official-avg_custom:.2f}%")
print("="*70)

# Performance relative to random
print("\n📊 Performance vs Random Baseline (25%):")
for i, bench in enumerate(benchmarks):
    custom_lift = custom_qwen3_1_8b[i] - 25.0
    official_lift = official_qwen2_5_1_5b[i] - 25.0
    relative_perf = (custom_lift / official_lift) * 100 if official_lift > 0 else 0
    print(f"  {bench}: Custom achieves {relative_perf:.1f}% of Official's lift above random")

