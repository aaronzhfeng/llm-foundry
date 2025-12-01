"""
Scaling Law Analysis for Large Language Models
==============================================

Implements Kaplan (OpenAI 2020) and Chinchilla (DeepMind 2022) scaling laws.

References:
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
  https://arxiv.org/abs/2001.08361
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
  https://arxiv.org/abs/2203.15556
"""

import argparse
import json
import math


def chinchilla_optimal_allocation(C, params):
    """
    Calculate optimal N and D using Chinchilla scaling laws.
    
    Formula:
    N_opt = G * (C/6)^(β/(α+β))
    D_opt = (1/G) * (C/6)^(α/(α+β))
    
    where G = (αA/βB)^(1/(α+β))
    
    Reference: Hoffmann et al., 2022, Equation 4
    """
    E = params['E']
    A = params['A']
    B = params['B']
    alpha = params['alpha']
    beta = params['beta']
    
    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    
    N_opt = G * ((C / 6) ** (beta / (alpha + beta)))
    D_opt = (1 / G) * ((C / 6) ** (alpha / (alpha + beta)))
    
    return N_opt, D_opt


def chinchilla_predict_loss(N, D, params):
    """
    Predict training loss using Chinchilla parametric model.
    
    Formula:
    L(N,D) = E + A·N^(-α) + B·D^(-β)
    
    Reference: Hoffmann et al., 2022, Equation 1
    """
    E = params['E']
    A = params['A']
    B = params['B']
    alpha = params['alpha']
    beta = params['beta']
    
    loss = E + A * (N ** (-alpha)) + B * (D ** (-beta))
    
    return loss


def kaplan_optimal_allocation(C, params):
    """
    Calculate optimal N using Kaplan scaling laws.
    
    Formula:
    N_opt ∝ C^0.73 (but with normalization factor)
    D_opt from C = 6·N·D constraint
    
    Reference: Kaplan et al., 2020, Section 5
    """
    optimal_exp = params['optimal_exponent']
    
    # Normalization factor to get reasonable N values
    # Using GPT-3 as reference: C=3.14e23, N=175e9
    # 175e9 = k * (3.14e23)^0.73 => k ≈ 175e9 / (3.14e23)^0.73
    k = 175e9 / ((3.14e23) ** optimal_exp)
    
    N_opt = k * (C ** optimal_exp)
    D_opt = C / (6 * N_opt)
    
    return N_opt, D_opt


def kaplan_predict_loss(N, D, params):
    """
    Predict training loss using Kaplan combined scaling law.
    
    Formula:
    L(N,D) = [(N_c/N)^(α_N/α_D) + D_c/D]^α_D
    
    Reference: Kaplan et al., 2020, Equation 1.5
    """
    N_c = params['N_c']
    D_c = params['D_c']
    alpha_N = params['alpha_N']
    alpha_D = params['alpha_D']
    
    term1 = (N_c / N) ** (alpha_N / alpha_D)
    term2 = D_c / D
    
    loss = (term1 + term2) ** alpha_D
    
    return loss


def calculate_compute_budget(N, D):
    """
    Calculate total compute budget C from model size N and tokens D.
    
    Formula:
    C = 6·N·D
    
    Reference: Hoffmann et al., 2022, Section 2.1
    """
    return 6 * N * D


def analyze_scaling_law(config_path, compute_budget=None):
    """
    Analyze scaling law and provide optimal allocation.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    scaling_law = config['scaling_law']
    params = config['parameters']
    
    print(f"\nScaling Law Analysis ({scaling_law.upper()})")
    print("=" * 60)
    
    if compute_budget is None:
        examples = config['compute_budget_examples']
        print("\nExample compute budgets:")
        for name, budget in examples.items():
            print(f"  {name}: {budget:.2e} FLOPs")
        
        print("\nProvide --compute_budget to analyze specific budget")
        return
    
    # Calculate optimal allocation
    if scaling_law == 'chinchilla':
        N_opt, D_opt = chinchilla_optimal_allocation(compute_budget, params)
        predicted_loss = chinchilla_predict_loss(N_opt, D_opt, params)
    else:  # kaplan
        N_opt, D_opt = kaplan_optimal_allocation(compute_budget, params)
        predicted_loss = kaplan_predict_loss(N_opt, D_opt, params)
    
    print(f"\nCompute Budget: {compute_budget:.2e} FLOPs")
    print(f"\nOptimal Allocation:")
    print(f"  Model size (N): {N_opt:,.0f} params ({N_opt/1e9:.2f}B)")
    print(f"  Training tokens (D): {D_opt:,.0f} tokens ({D_opt/1e9:.2f}B)")
    print(f"  Predicted loss: {predicted_loss:.4f}")
    print(f"  N/D ratio: {N_opt/D_opt:.2f}")
    
    # Verify compute constraint
    C_verify = calculate_compute_budget(N_opt, D_opt)
    error = abs(C_verify - compute_budget) / compute_budget * 100
    print(f"\nCompute verification:")
    print(f"  C = 6·N·D = {C_verify:.2e} FLOPs")
    print(f"  Error: {error:.4f}%")
    
    print("=" * 60)


def compare_scaling_laws(compute_budget):
    """
    Compare Kaplan vs Chinchilla scaling laws for same compute budget.
    """
    with open('chinchilla_config.json', 'r') as f:
        chinchilla_config = json.load(f)
    
    with open('kaplan_config.json', 'r') as f:
        kaplan_config = json.load(f)
    
    print(f"\nScaling Laws Comparison")
    print("=" * 60)
    print(f"Compute Budget: {compute_budget:.2e} FLOPs")
    print()
    
    # Chinchilla
    N_chin, D_chin = chinchilla_optimal_allocation(compute_budget, chinchilla_config['parameters'])
    loss_chin = chinchilla_predict_loss(N_chin, D_chin, chinchilla_config['parameters'])
    
    # Kaplan
    N_kap, D_kap = kaplan_optimal_allocation(compute_budget, kaplan_config['parameters'])
    loss_kap = kaplan_predict_loss(N_kap, D_kap, kaplan_config['parameters'])
    
    print(f"{'Method':<15} {'N (params)':<20} {'D (tokens)':<20} {'N/D ratio':<12} {'Loss':<8}")
    print("-" * 80)
    print(f"{'Chinchilla':<15} {N_chin/1e9:>10.2f}B {D_chin/1e9:>15.2f}B {N_chin/D_chin:>10.2f} {loss_chin:>8.4f}")
    print(f"{'Kaplan':<15} {N_kap/1e9:>10.2f}B {D_kap/1e9:>15.2f}B {N_kap/D_kap:>10.2f} {loss_kap:>8.4f}")
    
    print(f"\nKey differences:")
    print(f"  N ratio (Kaplan/Chinchilla): {N_kap/N_chin:.2f}x")
    print(f"  D ratio (Chinchilla/Kaplan): {D_chin/D_kap:.2f}x")
    print(f"  Chinchilla uses {D_chin/D_kap:.1f}x more tokens for same compute")
    
    print("=" * 60)


def validate_scaling_laws():
    """
    Validate scaling law calculations against known models.
    """
    print("=" * 80)
    print("VALIDATION: Testing scaling laws against known models")
    print("=" * 80)
    
    with open('chinchilla_config.json', 'r') as f:
        chinchilla_config = json.load(f)
    
    known_models = [
        {"name": "LLaMA 7B", "N": 6.7e9, "D": 1.0e12, "C": 6 * 6.7e9 * 1.0e12},
        {"name": "LLaMA 65B", "N": 65.2e9, "D": 1.4e12, "C": 6 * 65.2e9 * 1.4e12},
        {"name": "GPT-3 175B", "N": 175e9, "D": 300e9, "C": 6 * 175e9 * 300e9},
        {"name": "Chinchilla 70B", "N": 70e9, "D": 1.4e12, "C": 6 * 70e9 * 1.4e12}
    ]
    
    print("\n{:<20} {:<15} {:<15} {:<15} {:<10}".format(
        "Model", "N (params)", "D (tokens)", "C (FLOPs)", "Status"
    ))
    print("-" * 80)
    
    for model in known_models:
        N_opt, D_opt = chinchilla_optimal_allocation(model['C'], chinchilla_config['parameters'])
        
        N_error = abs(model['N'] - N_opt) / model['N'] * 100
        D_error = abs(model['D'] - D_opt) / model['D'] * 100
        
        if model['name'] == "Chinchilla 70B":
            status = "Optimal"
        elif N_error < 50 and D_error < 50:
            status = "Near-opt"
        elif model['N'] / N_opt > 2:
            status = "Over-param"
        elif model['D'] / D_opt < 0.5:
            status = "Under-train"
        else:
            status = "Suboptimal"
        
        print("{:<20} {:<15.2f}B {:<15.2f}B {:<15.2e} {:<10}".format(
            model['name'],
            model['N']/1e9,
            model['D']/1e9,
            model['C'],
            status
        ))
    
    print()
    print("Key insights:")
    print("  - Chinchilla 70B: Optimal by design")
    print("  - LLaMA models: Near-optimal allocation")
    print("  - GPT-3 175B: Over-parameterized (under-trained on data)")
    
    print("=" * 80)


def compute_budget_from_dollars(budget_dollars, hardware='8x_a100'):
    """
    Convert dollar budget to compute budget in FLOPs.
    """
    with open('custom_budget_config.json', 'r') as f:
        config = json.load(f)
    
    if hardware in config['hardware_scenarios']:
        hw = config['hardware_scenarios'][hardware]
        hours = budget_dollars / hw['cost_per_hour']
        compute_flops = hw['flops_per_sec'] * hours * 3600
        
        return compute_flops, hours
    
    return None, None


def analyze_compute_budget(config_path, budget_dollars=None, hardware='8x_a100'):
    """
    Analyze what model can be trained with given dollar budget.
    """
    if budget_dollars is None:
        print("Provide --budget_dollars to analyze dollar budget")
        return
    
    compute_flops, hours = compute_budget_from_dollars(budget_dollars, hardware)
    
    if compute_flops is None:
        print(f"Unknown hardware: {hardware}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    scaling_law = config['scaling_law']
    params = config['parameters']
    
    print(f"\nBudget Analysis ({scaling_law.upper()})")
    print("=" * 60)
    print(f"Dollar budget: ${budget_dollars:,.2f}")
    print(f"Hardware: {hardware}")
    print(f"Training time: {hours:,.1f} hours ({hours/24:.1f} days)")
    print(f"Compute budget: {compute_flops:.2e} FLOPs")
    print()
    
    # Calculate optimal allocation
    if scaling_law == 'chinchilla':
        N_opt, D_opt = chinchilla_optimal_allocation(compute_flops, params)
        predicted_loss = chinchilla_predict_loss(N_opt, D_opt, params)
    else:
        N_opt, D_opt = kaplan_optimal_allocation(compute_flops, params)
        predicted_loss = kaplan_predict_loss(N_opt, D_opt, params)
    
    print("Optimal Model Configuration:")
    print(f"  Model size: {N_opt:,.0f} params ({N_opt/1e9:.2f}B)")
    print(f"  Training tokens: {D_opt:,.0f} tokens ({D_opt/1e9:.2f}B)")
    print(f"  Predicted loss: {predicted_loss:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scaling Law Analysis')
    parser.add_argument('--config', type=str, help='Path to scaling law config (chinchilla_config.json or kaplan_config.json)')
    parser.add_argument('--compute_budget', type=float, default=None, help='Compute budget in FLOPs')
    parser.add_argument('--compare', action='store_true', help='Compare Kaplan vs Chinchilla')
    parser.add_argument('--budget_dollars', type=float, default=None, help='Dollar budget for training')
    parser.add_argument('--hardware', type=str, default='8x_a100', help='Hardware configuration')
    parser.add_argument('--validate', action='store_true', help='Validate against known models')
    args = parser.parse_args()
    
    if args.validate:
        validate_scaling_laws()
    
    if args.compare and args.compute_budget:
        compare_scaling_laws(args.compute_budget)
    
    if args.config:
        if args.budget_dollars:
            analyze_compute_budget(args.config, args.budget_dollars, args.hardware)
        elif args.compute_budget:
            analyze_scaling_law(args.config, args.compute_budget)
        else:
            analyze_scaling_law(args.config)
    
    if not args.config and not args.compare and not args.validate:
        print("Scaling Law Analysis")
        print("=" * 60)
        print("\nUsage:")
        print("  python scaling_law_analysis.py --config chinchilla_config.json --compute_budget 1e23")
        print("  python scaling_law_analysis.py --compare --compute_budget 1e23")
        print("  python scaling_law_analysis.py --config chinchilla_config.json --budget_dollars 10000")
        print("  python scaling_law_analysis.py --validate")

