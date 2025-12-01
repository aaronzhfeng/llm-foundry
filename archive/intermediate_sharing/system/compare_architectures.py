#!/usr/bin/env python3
"""
Architecture Comparison Tool
=============================

Compare training runs with different architectures.
Analyzes JSON logs to compare MFU, loss, and performance metrics.

Usage:
    python compare_architectures.py
    python compare_architectures.py out/run_*.json
    python compare_architectures.py --latest 5
"""

import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_log(log_path: str) -> Dict[str, Any]:
    """Load a training log JSON file"""
    with open(log_path, 'r') as f:
        return json.load(f)


def extract_key_metrics(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a log"""
    summary = log_data.get('summary', {})
    config = log_data.get('config', {})
    startup = log_data.get('startup_info', {})
    
    # Get architecture info
    arch_name = "Unknown"
    arch_details = {}
    
    if 'arch_preset' in config:
        arch_preset = config.get('arch_preset', 'unknown')
        norm = config.get('normalization', 'unknown')
        pos = config.get('position_encoding', 'unknown')
        ffn = config.get('ffn_type', 'unknown')
        arch_details = {
            'preset': arch_preset,
            'normalization': norm,
            'position_encoding': pos,
            'ffn_type': ffn,
        }
        arch_name = f"{arch_preset}({norm}/{pos}/{ffn})"
    
    return {
        'run_name': log_data.get('run_name', 'unknown'),
        'architecture': arch_name,
        'arch_details': arch_details,
        'final_loss': summary.get('final_train_loss', float('nan')),
        'best_val_loss': summary.get('best_val_loss', float('nan')),
        'avg_mfu': summary.get('avg_mfu', 0.0),
        'avg_time_ms': summary.get('avg_time_ms', 0.0),
        'total_iters': summary.get('total_iterations', 0),
        'num_gpus': startup.get('hardware', {}).get('num_gpus', 1) if startup else 1,
    }


def compare_runs(log_paths: List[str]):
    """Compare multiple training runs"""
    
    runs = []
    for path in log_paths:
        try:
            log_data = load_log(path)
            metrics = extract_key_metrics(log_data)
            runs.append(metrics)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    if not runs:
        print("No valid logs found!")
        return
    
    # Sort by validation loss (best first)
    runs.sort(key=lambda x: x['best_val_loss'])
    
    # Print comparison table
    print("\n" + "="*100)
    print("ARCHITECTURE COMPARISON")
    print("="*100)
    print(f"{'Rank':<6} {'Run Name':<25} {'Architecture':<30} {'Val Loss':<12} {'MFU %':<10} {'Time/iter':<12}")
    print("-"*100)
    
    for i, run in enumerate(runs, 1):
        print(f"{i:<6} {run['run_name']:<25} {run['architecture']:<30} "
              f"{run['best_val_loss']:<12.4f} {run['avg_mfu']:<10.2f} {run['avg_time_ms']:<12.1f}")
    
    print("="*100)
    
    # Best architecture
    best = runs[0]
    print(f"\n🏆 Best Architecture: {best['architecture']}")
    print(f"   Validation Loss: {best['best_val_loss']:.4f}")
    print(f"   Average MFU: {best['avg_mfu']:.2f}%")
    print(f"   Run: {best['run_name']}")
    
    # Detailed comparison
    if len(runs) > 1:
        print(f"\n📊 Detailed Comparison:")
        print(f"\n{'Architecture':<30} {'Val Loss':<12} {'Improvement':<15} {'MFU %':<10}")
        print("-"*70)
        
        baseline_loss = runs[-1]['best_val_loss']  # Worst as baseline
        for run in runs:
            improvement = ((baseline_loss - run['best_val_loss']) / baseline_loss) * 100
            print(f"{run['architecture']:<30} {run['best_val_loss']:<12.4f} "
                  f"{improvement:>+6.2f}%{' ':<8} {run['avg_mfu']:<10.2f}")
    
    # Architecture-specific insights
    print(f"\n💡 Insights:")
    
    # Group by architecture components
    arch_groups = {}
    for run in runs:
        details = run['arch_details']
        if details:
            preset = details.get('preset', 'unknown')
            if preset not in arch_groups:
                arch_groups[preset] = []
            arch_groups[preset].append(run)
    
    for preset, preset_runs in arch_groups.items():
        if len(preset_runs) > 0:
            avg_loss = sum(r['best_val_loss'] for r in preset_runs) / len(preset_runs)
            avg_mfu = sum(r['avg_mfu'] for r in preset_runs) / len(preset_runs)
            print(f"   {preset:<15}: Avg Val Loss = {avg_loss:.4f}, Avg MFU = {avg_mfu:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare training runs with different architectures")
    parser.add_argument('logs', nargs='*', help='Log files to compare (default: out/run_*.json)')
    parser.add_argument('--latest', type=int, help='Compare latest N runs')
    parser.add_argument('--out_dir', default='out', help='Output directory')
    
    args = parser.parse_args()
    
    # Get log files
    if args.logs:
        log_paths = args.logs
    else:
        # Find all logs in out directory
        log_paths = sorted(glob.glob(f"{args.out_dir}/run_*.json"))
        
        if args.latest:
            log_paths = log_paths[-args.latest:]
    
    if not log_paths:
        print(f"No log files found in {args.out_dir}/")
        print(f"Run some training first!")
        return
    
    print(f"Comparing {len(log_paths)} training runs...")
    compare_runs(log_paths)


if __name__ == '__main__':
    main()

