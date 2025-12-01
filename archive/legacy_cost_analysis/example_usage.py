"""
Example Usage of LLM Cost Analysis Tool
========================================

This script demonstrates how to use the llm_cost_analysis module to:
1. Calculate parameters and FLOPs for different model architectures
2. Estimate memory requirements
3. Find optimal model sizes for a given budget
"""

from llm_cost_analysis import (
    calculate_llama_parameters,
    calculate_llama_flops,
    calculate_llama_memory,
    calculate_deepseek_parameters,
    calculate_deepseek_flops,
    calculate_deepseek_memory,
    get_optimal_N_D_from_cost
)
import json


def example_1_llama_analysis():
    """Example 1: Analyze a LLaMA-style model"""
    print("=" * 80)
    print("Example 1: LLaMA 7B Analysis")
    print("=" * 80)
    
    # Load config
    with open('llama_7b_config.json', 'r') as f:
        config = json.load(f)
    
    # Calculate metrics
    params = calculate_llama_parameters(config)
    flops = calculate_llama_flops(config, sequence_length=2048)
    memory = calculate_llama_memory(config, batch_size=1, sequence_length=2048)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size:        {config['hidden_size']}")
    print(f"  Layers:             {config['num_hidden_layers']}")
    print(f"  Attention heads:    {config['num_attention_heads']}")
    print(f"  FFN dimension:      {config['intermediate_size']}")
    print(f"  Vocabulary:         {config['vocab_size']}")
    
    print(f"\nComputed Metrics:")
    print(f"  Total parameters:   {params:,} ({params/1e9:.2f}B)")
    print(f"  FLOPs per token:    {flops:.2f} TFLOPs")
    print(f"  Training memory:    {memory:.2f} GB")
    print()


def example_2_deepseek_analysis():
    """Example 2: Analyze a DeepSeek V3 MoE model"""
    print("=" * 80)
    print("Example 2: DeepSeek V3 MoE Analysis")
    print("=" * 80)
    
    # Load config
    with open('deepseek_v3_config.json', 'r') as f:
        config = json.load(f)
    
    # Calculate metrics
    params = calculate_deepseek_parameters(config)
    flops = calculate_deepseek_flops(config, sequence_length=2048)
    memory = calculate_deepseek_memory(config, batch_size=1, sequence_length=2048)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size:        {config['hidden_size']}")
    print(f"  Layers:             {config['num_hidden_layers']}")
    print(f"  Dense layers:       {config.get('first_k_dense_replace', 0)}")
    print(f"  MoE layers:         {config['num_hidden_layers'] - config.get('first_k_dense_replace', 0)}")
    print(f"  Total experts:      {config['n_routed_experts']}")
    print(f"  Experts per token:  {config['num_experts_per_tok']}")
    
    print(f"\nComputed Metrics:")
    print(f"  Total parameters:   {params:,} ({params/1e9:.2f}B)")
    print(f"  Active params/tok:  ~{params * config['num_experts_per_tok'] / config['n_routed_experts'] / 1e9:.2f}B")
    print(f"  FLOPs per token:    {flops:.2f} TFLOPs")
    print(f"  Training memory:    {memory:.2f} GB")
    print()


def example_3_custom_model():
    """Example 3: Analyze a custom model configuration"""
    print("=" * 80)
    print("Example 3: Custom Model Configuration")
    print("=" * 80)
    
    # Create custom config
    custom_config = {
        'hidden_size': 2048,
        'intermediate_size': 8192,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'vocab_size': 50000,
        'tie_word_embeddings': True
    }
    
    # Calculate metrics
    params = calculate_llama_parameters(custom_config)
    flops = calculate_llama_flops(custom_config, sequence_length=1024)
    memory = calculate_llama_memory(custom_config, batch_size=1, sequence_length=1024)
    
    print(f"\nCustom Configuration:")
    print(f"  Hidden size:        {custom_config['hidden_size']}")
    print(f"  Layers:             {custom_config['num_hidden_layers']}")
    print(f"  Attention heads:    {custom_config['num_attention_heads']}")
    print(f"  FFN dimension:      {custom_config['intermediate_size']}")
    print(f"  Tied embeddings:    {custom_config['tie_word_embeddings']}")
    
    print(f"\nComputed Metrics:")
    print(f"  Total parameters:   {params:,} ({params/1e9:.2f}B)")
    print(f"  FLOPs per token:    {flops:.2f} TFLOPs")
    print(f"  Training memory:    {memory:.2f} GB")
    print()


def example_4_budget_optimization():
    """Example 4: Find optimal model size for different budgets"""
    print("=" * 80)
    print("Example 4: Optimal Model Sizing for Different Budgets")
    print("=" * 80)
    
    budgets = [100, 1000, 10000, 100000]
    
    print("\n{:<15} {:<10} {:<15} {:<15} {:<20}".format(
        "Budget", "Best GPU", "Model Size", "Training Tokens", "Training FLOPs"
    ))
    print("-" * 80)
    
    for budget in budgets:
        N, D, flops, gpu = get_optimal_N_D_from_cost(budget)
        print("{:<15} {:<10} {:<15} {:<15} {:<20}".format(
            f"${budget:,}",
            gpu,
            f"{N/1e9:.2f}B",
            f"{D/1e9:.2f}B",
            f"{flops:.2e}"
        ))
    print()


def example_5_scaling_analysis():
    """Example 5: Analyze how metrics scale with model size"""
    print("=" * 80)
    print("Example 5: Scaling Analysis")
    print("=" * 80)
    
    # Test different model sizes
    sizes = [
        {'name': 'Small', 'hidden': 768, 'layers': 12, 'ffn': 3072},
        {'name': 'Medium', 'hidden': 1024, 'layers': 24, 'ffn': 4096},
        {'name': 'Large', 'hidden': 2048, 'layers': 32, 'ffn': 8192},
        {'name': 'XLarge', 'hidden': 4096, 'layers': 48, 'ffn': 16384},
    ]
    
    print("\n{:<10} {:<15} {:<15} {:<15}".format(
        "Size", "Parameters", "FLOPs/token", "Memory (GB)"
    ))
    print("-" * 80)
    
    for size in sizes:
        config = {
            'hidden_size': size['hidden'],
            'intermediate_size': size['ffn'],
            'num_hidden_layers': size['layers'],
            'num_attention_heads': size['hidden'] // 64,  # 64 dim per head
            'vocab_size': 32000,
            'tie_word_embeddings': False
        }
        
        params = calculate_llama_parameters(config)
        flops = calculate_llama_flops(config, sequence_length=2048)
        memory = calculate_llama_memory(config, batch_size=1, sequence_length=2048)
        
        print("{:<10} {:<15} {:<15} {:<15}".format(
            size['name'],
            f"{params/1e9:.2f}B",
            f"{flops:.2f} TF",
            f"{memory:.2f}"
        ))
    print()


def example_6_sequence_length_impact():
    """Example 6: Show impact of sequence length on FLOPs"""
    print("=" * 80)
    print("Example 6: Impact of Sequence Length on FLOPs")
    print("=" * 80)
    
    with open('llama_7b_config.json', 'r') as f:
        config = json.load(f)
    
    sequence_lengths = [512, 1024, 2048, 4096, 8192]
    
    print("\n{:<15} {:<15} {:<20}".format(
        "Sequence Len", "FLOPs (TF)", "FLOPs/Token (GF)"
    ))
    print("-" * 80)
    
    for seq_len in sequence_lengths:
        flops = calculate_llama_flops(config, sequence_length=seq_len)
        flops_per_token = (flops * 1e12) / seq_len / 1e9  # Convert to GFLOPs per token
        
        print("{:<15} {:<15} {:<20}".format(
            seq_len,
            f"{flops:.2f}",
            f"{flops_per_token:.2f}"
        ))
    
    print("\nNote: FLOPs scale quadratically with sequence length due to attention mechanism")
    print()


if __name__ == "__main__":
    # Run all examples
    example_1_llama_analysis()
    example_2_deepseek_analysis()
    example_3_custom_model()
    example_4_budget_optimization()
    example_5_scaling_analysis()
    example_6_sequence_length_impact()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)

