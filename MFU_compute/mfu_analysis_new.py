#!/usr/bin/env python3
"""
Simple MFU (Model Flops Utilization) Analysis
============================================

A simplified implementation focusing on the core MFU calculation
without complex formatting issues.

This demonstrates the MFU calculation methodology using detailed
academic formulas, following the same pattern as flops_parameter_counting.
"""

import argparse
import json
import math


def calculate_llama_parameters(config):
    """Calculate total parameters for LLaMA-style model."""
    H = config['hidden_size']
    D_ff = config['intermediate_size']
    L = config['num_hidden_layers']
    V = config['vocab_size']

    num_kv_heads = config.get('num_key_value_heads', config['num_attention_heads'])
    num_q_heads = config['num_attention_heads']

    embedding_params = V * H

    if num_kv_heads == num_q_heads:
        attention_params = 4 * H * H
    else:
        head_dim = H // num_q_heads
        attention_params = H * H + 2 * H * (num_kv_heads * head_dim) + H * H

    ffn_params = H * D_ff + D_ff * H
    layernorm_params = 2 * H
    params_per_layer = attention_params + ffn_params + layernorm_params

    transformer_params = L * params_per_layer

    tie_word_embeddings = config.get('tie_word_embeddings', False)
    if tie_word_embeddings:
        output_params = 0
    else:
        output_params = H * V

    final_norm_params = H
    total_params = embedding_params + transformer_params + output_params + final_norm_params

    return total_params


def calculate_deepseek_parameters(config):
    """Calculate total parameters for DeepSeek V3 MoE model."""
    H = config['hidden_size']
    L = config['num_hidden_layers']
    V = config['vocab_size']

    q_lora_rank = config.get('q_lora_rank', H)
    kv_lora_rank = config.get('kv_lora_rank', H)

    num_q_heads = config['num_attention_heads']
    num_kv_heads = config.get('num_key_value_heads', num_q_heads)

    qk_nope_head_dim = config.get('qk_nope_head_dim', 128)
    qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
    v_head_dim = config.get('v_head_dim', 128)

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    k_head_dim = qk_nope_head_dim + qk_rope_head_dim

    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    intermediate_size = config['intermediate_size']

    first_k_dense = config.get('first_k_dense_replace', 0)
    num_dense_layers = first_k_dense
    num_moe_layers = L - first_k_dense

    embedding_params = V * H

    q_proj_params = H * q_lora_rank + q_lora_rank * (num_q_heads * q_head_dim)
    k_proj_params = H * kv_lora_rank + kv_lora_rank * (num_kv_heads * k_head_dim)
    v_proj_params = H * kv_lora_rank + kv_lora_rank * (num_kv_heads * v_head_dim)
    o_proj_params = num_q_heads * v_head_dim * H

    attention_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params

    dense_ffn_params = 2 * H * intermediate_size
    shared_expert_params = n_shared_experts * (2 * H * moe_intermediate_size)
    routed_expert_params = n_routed_experts * (2 * H * moe_intermediate_size)
    router_params = H * n_routed_experts

    moe_ffn_params = shared_expert_params + routed_expert_params + router_params
    layernorm_params = 2 * H

    dense_layer_params = attention_params_per_layer + dense_ffn_params + layernorm_params
    moe_layer_params = attention_params_per_layer + moe_ffn_params + layernorm_params

    transformer_params = (num_dense_layers * dense_layer_params + num_moe_layers * moe_layer_params)

    tie_word_embeddings = config.get('tie_word_embeddings', False)
    if tie_word_embeddings:
        output_params = 0
    else:
        output_params = H * V

    final_norm_params = H
    total_params = embedding_params + transformer_params + output_params + final_norm_params

    return total_params


def calculate_llama_flops_per_token_detailed(config, sequence_length=2048):
    """Calculate FLOPs per token for LLaMA using detailed academic formula."""
    H = config['hidden_size']
    D_ff = config['intermediate_size']
    L = config['num_hidden_layers']
    S = sequence_length
    a = config['num_attention_heads']

    attention_qkv_flops = 6 * H * H
    attention_scores_flops = a * S * H
    attention_output_flops = a * S * H
    attention_proj_flops = 2 * H * H
    attention_flops = attention_qkv_flops + attention_scores_flops + attention_output_flops + attention_proj_flops

    ffn_up_flops = 2 * H * D_ff
    ffn_down_flops = 2 * D_ff * H
    ffn_flops = ffn_up_flops + ffn_down_flops

    flops_per_layer = attention_flops + ffn_flops
    total_flops_per_token = L * flops_per_layer
    embedding_flops = H

    return total_flops_per_token + embedding_flops


def calculate_deepseek_flops_per_token_detailed(config, sequence_length=2048):
    """Calculate FLOPs per token for DeepSeek V3 using detailed academic formula."""
    H = config['hidden_size']
    L = config['num_hidden_layers']
    S = sequence_length

    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 1)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])

    first_k_dense = config.get('first_k_dense_replace', 0)
    num_dense_layers = first_k_dense
    num_moe_layers = L - first_k_dense

    attention_flops_per_layer = 8 * H * H + 4 * S * H
    dense_ffn_flops = 4 * H * config['intermediate_size']

    active_experts = num_experts_per_tok + n_shared_experts
    moe_ffn_flops = 4 * H * moe_intermediate_size * active_experts
    router_flops = H * n_routed_experts

    dense_layer_total_flops = attention_flops_per_layer + dense_ffn_flops
    moe_layer_total_flops = attention_flops_per_layer + moe_ffn_flops + router_flops

    total_flops_per_token = (num_dense_layers * dense_layer_total_flops + num_moe_layers * moe_layer_total_flops)

    return total_flops_per_token


def calculate_mfu(config_path, achieved_tokens_per_sec=None):
    """
    Calculate MFU for a given configuration.

    Returns:
        mfu_percent: MFU percentage
        flops_per_token: FLOPs per token
        hardware_peak_flops: Hardware peak FLOPs
        achieved_flops: Achieved FLOPs per second
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['model_config']
    hardware_specs = config['hardware_specs']
    training_config = config['training_config']
    performance = config['performance_measurements']

    if achieved_tokens_per_sec is None:
        achieved_tokens_per_sec = performance.get('tokens_per_second_achieved', 0)

    if achieved_tokens_per_sec == 0:
        return 0, 0, 0, 0

    sequence_length = training_config['sequence_length']
    if config['model_name'].startswith('llama'):
        flops_per_token = calculate_llama_flops_per_token_detailed(model_config, sequence_length)
    else:
        flops_per_token = calculate_deepseek_flops_per_token_detailed(model_config, sequence_length)

    precision = config['precision']
    precision_key = f'peak_tflops_{precision.lower()}'
    if precision_key in hardware_specs:
        peak_tflops_per_gpu = hardware_specs[precision_key]
    else:
        peak_tflops_per_gpu = hardware_specs['peak_tflops_fp16']

    gpus = config['gpus']
    hardware_peak_flops = gpus * peak_tflops_per_gpu * 1e12
    achieved_flops = flops_per_token * achieved_tokens_per_sec
    mfu_percent = (achieved_flops / hardware_peak_flops) * 100

    return mfu_percent, flops_per_token, hardware_peak_flops, achieved_flops


def analyze_mfu_config(config_path, achieved_tokens_per_sec=None):
    """Analyze MFU configuration and provide comprehensive report."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name = config['model_name']
    hardware = config['hardware']
    gpus = config['gpus']
    precision = config['precision']

    print(f"\nMFU Analysis: {model_name} on {hardware}")
    print("=" * 50)

    # Basic configuration
    print(f"GPUs: {gpus} x {hardware}")
    print(f"Precision: {precision}")
    print(f"Batch size: {config['training_config']['batch_size']}")
    print(f"Sequence length: {config['training_config']['sequence_length']}")
    print()

    # Hardware specifications
    hw_specs = config['hardware_specs']
    precision_key = f'peak_tflops_{precision.lower()}'
    if precision_key in hw_specs:
        peak_tflops_per_gpu = hw_specs[precision_key]
    else:
        peak_tflops_per_gpu = hw_specs['peak_tflops_fp16']

    total_peak_flops = gpus * peak_tflops_per_gpu * 1e12

    print("Hardware Specifications:")
    print(f"  Peak FLOPs per GPU: {peak_tflops_per_gpu:,} TFLOPS ({precision})")
    print(f"  Total peak FLOPs: {total_peak_flops / 1e15:.2f} PFLOPS")
    print(f"  Memory per GPU: {hw_specs['memory_gb']} GB")
    print()

    # Model analysis
    model_config = config['model_config']
    if model_name.startswith('llama'):
        total_params = calculate_llama_parameters(model_config)
        flops_per_token = calculate_llama_flops_per_token_detailed(model_config, config['training_config']['sequence_length'])
    else:
        total_params = calculate_deepseek_parameters(model_config)
        flops_per_token = calculate_deepseek_flops_per_token_detailed(model_config, config['training_config']['sequence_length'])

    print("Model Specifications:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  FLOPs per token: {flops_per_token / 1e9:.2f} GFLOPs")

    if model_name.startswith('llama'):
        kv_heads = model_config.get('num_key_value_heads', model_config['num_attention_heads'])
        if kv_heads != model_config['num_attention_heads']:
            print(f"  KV heads (GQA): {kv_heads}")
        print(f"  FFN dimension: {model_config['intermediate_size']}")
    else:
        print(f"  Dense layers: {model_config.get('first_k_dense_replace', 0)}")
        print(f"  MoE layers: {model_config['num_hidden_layers'] - model_config.get('first_k_dense_replace', 0)}")
        print(f"  Total experts: {model_config.get('n_routed_experts', 0)}")
        print(f"  Experts per token: {model_config.get('num_experts_per_tok', 1)}")
        print(f"  MLA compression: Q={model_config.get('q_lora_rank', 'full')}, KV={model_config.get('kv_lora_rank', 'full')}")

    print()

    # MFU analysis
    if config['performance_measurements']['tokens_per_second_achieved'] > 0 or achieved_tokens_per_sec:
        # Real performance measurement available
        tokens_per_sec = achieved_tokens_per_sec if achieved_tokens_per_sec else config['performance_measurements']['tokens_per_second_achieved']

        mfu_percent, flops_per_token, hardware_peak_flops, achieved_flops = calculate_mfu(
            config_path, achieved_tokens_per_sec=tokens_per_sec
        )

        print("Performance Analysis (Measured):")
        print(f"  Achieved throughput: {tokens_per_sec:,} tokens/sec")
        print(f"  Achieved FLOPs: {achieved_flops / 1e15:.2f} PFLOPS")
        print(f"  MFU achieved: {mfu_percent:.1f}%")

        # Targets
        targets = config['mfu_targets']
        if mfu_percent >= targets['target_mfu_percent']:
            print(f"  Status: EXCELLENT - Exceeds target MFU ({targets['target_mfu_percent']}%)")
        elif mfu_percent >= targets['realistic_mfu_percent']:
            print(f"  Status: GOOD - Meets realistic MFU ({targets['realistic_mfu_percent']}%)")
        elif mfu_percent >= targets['minimum_acceptable_mfu_percent']:
            print(f"  Status: OK - Above minimum MFU ({targets['minimum_acceptable_mfu_percent']}%)")
        else:
            print(f"  Status: POOR - Below minimum MFU ({targets['minimum_acceptable_mfu_percent']}%)")
        
        # Calculate training time if training objectives are specified
        if 'training_objectives' in config:
            total_training_tokens = config['training_objectives'].get('total_training_tokens', 0)
            if total_training_tokens > 0:
                # Time = total_tokens / tokens_per_second
                training_time_seconds = total_training_tokens / tokens_per_sec
                training_time_hours = training_time_seconds / 3600
                training_time_days = training_time_hours / 24
                
                print(f"\n  Training Time Estimate (at current throughput):")
                print(f"    Total training tokens: {total_training_tokens:,.0f} ({total_training_tokens/1e12:.2f}T)")
                print(f"    Training time: {training_time_hours:,.1f} hours ({training_time_days:.1f} days)")
                
                # Cost estimate if available
                if 'cost_per_hour' in config.get('hardware_specs', {}):
                    cost_per_hour = config['hardware_specs']['cost_per_hour']
                    total_cost = training_time_hours * cost_per_hour
                    print(f"    Estimated cost: ${total_cost:,.2f}")

    else:
        # No measurement available, show theoretical analysis
        print("Performance Analysis (Theoretical):")
        print("  No throughput measurements provided in config.")
        print("  Set 'tokens_per_second_achieved' in performance_measurements to get MFU.")

        # Calculate optimal throughput for target MFU
        target_mfu = config['mfu_targets']['realistic_mfu_percent'] / 100
        if flops_per_token > 0:
            max_throughput = (target_mfu * total_peak_flops) / flops_per_token

            print(f"  Target MFU: {config['mfu_targets']['realistic_mfu_percent']}%")
            print(f"  Required throughput: {max_throughput:,.0f} tokens/sec")
            print(f"  Current batch size: {config['training_config']['batch_size']}")
            print(f"  Effective tokens per iteration: {config['training_config']['batch_size'] * config['training_config']['sequence_length']:,}")
            
            # Calculate training time if training objectives are specified
            if 'training_objectives' in config:
                total_training_tokens = config['training_objectives'].get('total_training_tokens', 0)
                if total_training_tokens > 0:
                    # Time = total_tokens / tokens_per_second
                    training_time_seconds = total_training_tokens / max_throughput
                    training_time_hours = training_time_seconds / 3600
                    training_time_days = training_time_hours / 24
                    
                    print(f"\n  Training Time Estimate (at target MFU):")
                    print(f"    Total training tokens: {total_training_tokens:,.0f} ({total_training_tokens/1e12:.2f}T)")
                    print(f"    Training time: {training_time_hours:,.1f} hours ({training_time_days:.1f} days)")
                    
                    # Cost estimate if available
                    if 'cost_per_hour' in config.get('hardware_specs', {}):
                        cost_per_hour = config['hardware_specs']['cost_per_hour']
                        total_cost = training_time_hours * cost_per_hour
                        print(f"    Estimated cost: ${total_cost:,.2f}")
        else:
            print("  Cannot calculate - no FLOPs per token available")

    # Memory analysis
    memory_usage = {
        'model_weights_gb': total_params * 2 / (1024**3),  # FP16
        'gradients_gb': total_params * 2 / (1024**3),      # FP16
        'optimizer_gb': total_params * 8 / (1024**3),       # Adam (2×FP32)
        'total_memory_gb': (total_params * 12 / (1024**3)) + 10  # + activations estimate
    }

    print(f"\nMemory Requirements (per GPU):")
    print(f"  Model weights: {memory_usage['model_weights_gb'] / gpus:.1f} GB")
    print(f"  Gradients: {memory_usage['gradients_gb'] / gpus:.1f} GB")
    print(f"  Optimizer states: {memory_usage['optimizer_gb'] / gpus:.1f} GB")
    print(f"  Total: {memory_usage['total_memory_gb'] / gpus:.1f} GB")

    if memory_usage['total_memory_gb'] / gpus > hw_specs['memory_gb'] * 0.95:
        print(f"  WARNING: Memory usage ({memory_usage['total_memory_gb'] / gpus:.1f}GB) exceeds GPU memory ({hw_specs['memory_gb']}GB)")
    else:
        print(f"  Memory usage within GPU limits")

    print("=" * 50)


def validate_mfu_calculations():
    """Validate MFU calculations against known benchmarks."""
    print("=" * 80)
    print("VALIDATION: Testing MFU calculations against known benchmarks")
    print("=" * 80)

    # Test LLaMA 7B on A100 with typical performance
    config = {
        "model_name": "llama_7b",
        "gpus": 8,
        "precision": "FP16",
        "model_config": {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        },
        "hardware_specs": {
            "peak_tflops_fp16": 312
        },
        "training_config": {
            "sequence_length": 2048
        }
    }

    flops_per_token_detailed = calculate_llama_flops_per_token_detailed(config['model_config'])
    flops_per_token_nanoGPT = 6 * calculate_llama_parameters(config['model_config']) + 12 * 32 * 32 * (4096//32) * 2048

    print()
    print("LLaMA 7B FLOPs per token calculation:")
    print(f"  Detailed academic: {flops_per_token_detailed / 1e9:.2f} GFLOPs")
    print(f"  nanoGPT simplified: {flops_per_token_nanoGPT / 1e9:.2f} GFLOPs")
    print(f"  Ratio (detailed/nanoGPT): {flops_per_token_detailed / flops_per_token_nanoGPT:.2f}")

    hardware_peak = 8 * 312e12
    typical_throughput = 45_000

    mfu_detailed = (flops_per_token_detailed * typical_throughput) / hardware_peak * 100
    mfu_nanoGPT = (flops_per_token_nanoGPT * typical_throughput) / hardware_peak * 100

    print()
    print("Typical LLaMA 7B training performance:")
    print(f"  Hardware peak: {hardware_peak / 1e15:.1f} PFLOPS")
    print(f"  Typical throughput: {typical_throughput:,} tokens/sec")
    print(f"  MFU (detailed): {mfu_detailed:.1f}%")
    print(f"  MFU (nanoGPT): {mfu_nanoGPT:.1f}%")

    expected_mfu_range = (40, 55)
    if expected_mfu_range[0] <= mfu_detailed <= expected_mfu_range[1]:
        print(f"  PASS - MFU {mfu_detailed:.1f}% in expected range {expected_mfu_range[0]}-{expected_mfu_range[1]}%")
    else:
        print(f"  WARNING - MFU {mfu_detailed:.1f}% outside expected range")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple MFU Analysis')
    parser.add_argument('--config', type=str, help='Path to MFU configuration JSON file')
    parser.add_argument('--throughput', type=float, default=None, help='Achieved tokens per second')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    args = parser.parse_args()

    if args.validate:
        validate_mfu_calculations()

    if args.config:
        analyze_mfu_config(args.config, args.throughput)

    if not args.config and not args.validate:
        print("Please provide a configuration file with --config")
        print("Or run validation with --validate")
        print("\nExample usage:")
        print("  python simple_mfu_analysis.py --config llama_7b_a100_config.json")
        print("  python simple_mfu_analysis.py --config deepseek_v3_h100_config.json --throughput 25000")
        print("  python simple_mfu_analysis.py --validate")

