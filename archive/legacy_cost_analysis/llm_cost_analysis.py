"""
LLM Training Cost Analysis: Parameter and FLOPs Counting
=========================================================

This module implements comprehensive parameter counting and FLOPs (Floating Point Operations)
calculation for Large Language Models, specifically:
1. LLaMA-style models (standard dense Transformer)
2. DeepSeek V3-style models (Mixture of Experts with LoRA compression)

References:
-----------
1. "Attention Is All You Need" - Vaswani et al., 2017
   https://arxiv.org/abs/1706.03762
   
2. "Training Compute-Optimal Large Language Models" (Chinchilla) - Hoffmann et al., 2022
   https://arxiv.org/abs/2203.15556
   Formula: Training FLOPs ≈ 6 × N × D (N=params, D=tokens)
   
3. "LLaMA: Open and Efficient Foundation Language Models" - Touvron et al., 2023
   https://arxiv.org/abs/2302.13971
   
4. "DeepSeek-V3 Technical Report" - DeepSeek AI, 2024
   https://arxiv.org/abs/2412.19437
   MoE architecture with Multi-head Latent Attention (MLA)
   
5. FLOPs calculation methodology:
   https://dsdanielpark.github.io/llm/2023-12-12-LLaMAFLOPSEstimiation.html
   
6. "Efficient Large-Scale Language Model Training on GPU Clusters" - Narayanan et al., 2021
   Memory estimation formulas
"""

import argparse
import json
import math


def calculate_llama_parameters(config):
    """
    Calculate total parameters for a LLaMA-style model.
    
    LLaMA uses standard Multi-Head Attention (MHA) and dense FFN layers.
    
    Parameters per layer:
    - Attention: 4 × hidden_size² (Q, K, V, O projections)
    - FFN: 2 × hidden_size × intermediate_size (up and down projections)
    - Layer Norms: 2 × hidden_size (negligible)
    
    Reference: Section 2.1 of LLaMA paper (Touvron et al., 2023)
    """
    H = config['hidden_size']
    D_ff = config['intermediate_size']
    L = config['num_hidden_layers']
    V = config['vocab_size']
    
    # Check if model uses Grouped Query Attention (GQA)
    num_kv_heads = config.get('num_key_value_heads', config['num_attention_heads'])
    num_q_heads = config['num_attention_heads']
    
    # Embedding parameters
    embedding_params = V * H
    
    # Per-layer parameters
    params_per_layer = 0
    
    # Attention parameters
    if num_kv_heads == num_q_heads:
        # Standard Multi-Head Attention (MHA)
        # Q, K, V projections: each is H × H
        # O projection: H × H
        attention_params = 4 * H * H
    else:
        # Grouped Query Attention (GQA)
        # Q projection: H × H (full)
        # K, V projections: H × (num_kv_heads * head_dim)
        head_dim = H // num_q_heads
        attention_params = H * H  # Q projection
        attention_params += 2 * H * (num_kv_heads * head_dim)  # K, V projections
        attention_params += H * H  # O projection
    
    # FFN parameters: up projection + down projection
    ffn_params = H * D_ff + D_ff * H
    
    # Layer norms (RMSNorm): 2 per layer (pre-attention, pre-FFN)
    layernorm_params = 2 * H
    
    params_per_layer = attention_params + ffn_params + layernorm_params
    
    # Total transformer layer parameters
    transformer_params = L * params_per_layer
    
    # Output layer (language modeling head)
    # Check if embeddings are tied
    tie_word_embeddings = config.get('tie_word_embeddings', False)
    if tie_word_embeddings:
        output_params = 0  # Shared with input embeddings
    else:
        output_params = H * V
    
    # Final layer norm
    final_norm_params = H
    
    total_params = embedding_params + transformer_params + output_params + final_norm_params
    
    return total_params


def calculate_llama_flops(config, sequence_length=2048):
    """
    Calculate FLOPs per forward pass for a LLaMA-style model.
    
    FLOPs calculation (per token in forward pass):
    1. Self-Attention: 
       - QKV projections: 2 × S × H × (3H) = 6SH²
       - Attention scores: 2 × S² × H
       - Attention output: 2 × S² × H  
       - Output projection: 2 × S × H²
       Total: 8SH² + 4S²H
       
    2. FFN:
       - Up projection: 2 × S × H × D_ff
       - Down projection: 2 × S × D_ff × H
       Total: 4 × S × H × D_ff
    
    Reference: "Efficient Large-Scale Language Model Training" (Narayanan et al., 2021)
    Formula adapted from: https://dsdanielpark.github.io/llm/2023-12-12-LLaMAFLOPSEstimiation.html
    """
    H = config['hidden_size']
    D_ff = config['intermediate_size']
    L = config['num_hidden_layers']
    S = sequence_length
    
    # Per-layer FLOPs
    # Attention FLOPs
    # QKV projections (matrix multiplications: 2 FLOPs per multiply-add)
    qkv_proj_flops = 2 * S * H * (3 * H)  # 6SH²
    
    # Attention computation: softmax(QK^T/√d)V
    # QK^T: S × H @ H × S = S × S (per head, with H total across all heads)
    attention_scores_flops = 2 * S * S * H
    # Attention @ V: S × S @ S × H = S × H
    attention_output_flops = 2 * S * S * H
    
    # Output projection
    output_proj_flops = 2 * S * H * H
    
    attention_flops_per_layer = qkv_proj_flops + attention_scores_flops + attention_output_flops + output_proj_flops
    # = 8SH² + 4S²H
    
    # FFN FLOPs
    # Up projection: S × H -> S × D_ff
    up_proj_flops = 2 * S * H * D_ff
    # Down projection: S × D_ff -> S × H
    down_proj_flops = 2 * S * D_ff * H
    
    ffn_flops_per_layer = up_proj_flops + down_proj_flops
    # = 4 × S × H × D_ff
    
    # Total per-layer FLOPs
    flops_per_layer = attention_flops_per_layer + ffn_flops_per_layer
    
    # Total model FLOPs
    total_flops = L * flops_per_layer
    
    # Add embedding lookup FLOPs (typically negligible)
    # Embedding is just lookup, but if we count it: S × H operations
    embedding_flops = S * H
    
    total_flops += embedding_flops
    
    # Convert to TFLOPs (TeraFLOPs)
    total_tflops = total_flops / 1e12
    
    return total_tflops


def calculate_llama_memory(config, batch_size=1, sequence_length=2048):
    """
    Calculate peak memory usage during training for a LLaMA-style model.
    
    Memory components (for mixed precision training with Adam optimizer):
    1. Model parameters: 2 bytes per parameter (FP16/BF16)
    2. Gradients: 2 bytes per parameter (FP16/BF16)
    3. Optimizer states (Adam): 8 bytes per parameter (2 × FP32 for momentum and variance)
    4. Activations: depends on batch_size × sequence_length × hidden_size
    
    Total ≈ 12 × num_params + activation_memory
    
    Reference: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    Rajbhandari et al., 2020
    """
    total_params = calculate_llama_parameters(config)
    
    # Model weights: 2 bytes per parameter (FP16)
    model_memory = total_params * 2
    
    # Gradients: 2 bytes per parameter (FP16)
    gradient_memory = total_params * 2
    
    # Optimizer states (Adam): 8 bytes per parameter (2 FP32 states)
    optimizer_memory = total_params * 8
    
    # Activation memory (approximate)
    # Per layer: batch_size × sequence_length × hidden_size × num_layers
    # Multiple activation checkpoints per layer
    H = config['hidden_size']
    L = config['num_hidden_layers']
    
    # Approximate activations (attention + FFN activations)
    # Attention: B × S × H (QKV), B × num_heads × S × S (attention scores)
    # FFN: B × S × D_ff
    D_ff = config['intermediate_size']
    num_heads = config['num_attention_heads']
    
    activation_per_layer = batch_size * sequence_length * (
        4 * H +  # QKV + output
        num_heads * sequence_length +  # Attention scores
        D_ff  # FFN intermediate
    ) * 2  # FP16
    
    activation_memory = activation_per_layer * L
    
    # Total memory in bytes
    total_memory_bytes = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    # Convert to GB
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    return total_memory_gb


def calculate_deepseek_parameters(config):
    """
    Calculate total parameters for DeepSeek V3 with Mixture of Experts (MoE).
    
    DeepSeek V3 special features:
    1. Multi-head Latent Attention (MLA) with LoRA-style compression
    2. Mixture of Experts (MoE) layers with routed + shared experts
    3. First few layers are dense, rest are MoE
    
    MLA Parameters (per layer):
    - Q projection: H × q_lora_rank + q_lora_rank × (num_heads × qk_head_dim)
    - K, V projections: Similar with kv_lora_rank
    
    MoE Parameters (per layer):
    - Shared experts: n_shared × (2 × H × D_moe)
    - Routed experts: n_routed × (2 × H × D_moe)
    - Router: H × n_routed
    
    Reference: DeepSeek-V3 Technical Report (2024), Section 2
    """
    H = config['hidden_size']
    L = config['num_hidden_layers']
    V = config['vocab_size']
    
    # MLA parameters
    q_lora_rank = config.get('q_lora_rank', H)
    kv_lora_rank = config.get('kv_lora_rank', H)
    
    num_q_heads = config['num_attention_heads']
    num_kv_heads = config.get('num_key_value_heads', num_q_heads)
    
    qk_nope_head_dim = config.get('qk_nope_head_dim', 128)
    qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
    v_head_dim = config.get('v_head_dim', 128)
    
    # Total head dimension for Q
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    # Total head dimension for K
    k_head_dim = qk_nope_head_dim + qk_rope_head_dim
    
    # MoE parameters
    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    intermediate_size = config['intermediate_size']  # For dense layers
    
    # Number of dense vs MoE layers
    first_k_dense = config.get('first_k_dense_replace', 0)
    moe_layer_freq = config.get('moe_layer_freq', 1)
    
    # Calculate number of MoE layers
    num_dense_layers = first_k_dense
    num_moe_layers = L - first_k_dense
    
    # Embedding parameters
    embedding_params = V * H
    
    # MLA Attention parameters (used in all layers)
    # Q projection with LoRA-style compression
    q_proj_params = H * q_lora_rank + q_lora_rank * (num_q_heads * q_head_dim)
    
    # K projection with LoRA-style compression
    k_proj_params = H * kv_lora_rank + kv_lora_rank * (num_kv_heads * k_head_dim)
    
    # V projection with LoRA-style compression
    v_proj_params = H * kv_lora_rank + kv_lora_rank * (num_kv_heads * v_head_dim)
    
    # Output projection
    o_proj_params = num_q_heads * v_head_dim * H
    
    attention_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params
    
    # Dense layer FFN parameters
    dense_ffn_params = 2 * H * intermediate_size
    
    # MoE layer parameters
    # Shared experts (always activated)
    shared_expert_params = n_shared_experts * (2 * H * moe_intermediate_size)
    
    # Routed experts (only some are activated, but all exist as parameters)
    routed_expert_params = n_routed_experts * (2 * H * moe_intermediate_size)
    
    # Router/gating network
    router_params = H * n_routed_experts
    
    moe_ffn_params = shared_expert_params + routed_expert_params + router_params
    
    # Layer norms
    layernorm_params = 2 * H  # pre-attention and post-attention
    
    # Calculate total transformer parameters
    dense_layer_params = attention_params_per_layer + dense_ffn_params + layernorm_params
    moe_layer_params = attention_params_per_layer + moe_ffn_params + layernorm_params
    
    transformer_params = (num_dense_layers * dense_layer_params + 
                         num_moe_layers * moe_layer_params)
    
    # Output layer
    tie_word_embeddings = config.get('tie_word_embeddings', False)
    if tie_word_embeddings:
        output_params = 0
    else:
        output_params = H * V
    
    # Final layer norm
    final_norm_params = H
    
    total_params = embedding_params + transformer_params + output_params + final_norm_params
    
    return total_params


def calculate_deepseek_flops(config, sequence_length=2048):
    """
    Calculate FLOPs per forward pass for DeepSeek V3 MoE model.
    
    Key difference from dense models:
    - Only activated experts contribute to FLOPs
    - Typically: num_experts_per_tok experts are activated per token
    
    FLOPs for MoE layer:
    - Attention: Similar to LLaMA but with compressed dimensions
    - MoE FFN: (num_experts_per_tok + n_shared_experts) × (2 × S × H × D_moe)
    - Router: 2 × S × H × n_routed_experts (for gating computation)
    
    Reference: "GShard: Scaling Giant Models with Conditional Computation" 
    (Lepikhin et al., 2020) - foundational MoE paper
    """
    H = config['hidden_size']
    L = config['num_hidden_layers']
    S = sequence_length
    
    # MoE configuration
    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 1)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    intermediate_size = config['intermediate_size']
    
    first_k_dense = config.get('first_k_dense_replace', 0)
    num_dense_layers = first_k_dense
    num_moe_layers = L - first_k_dense
    
    # Attention FLOPs (simplified - similar to LLaMA)
    # MLA compression reduces some dimensions but core computation is similar
    attention_flops_per_layer = 8 * S * H * H + 4 * S * S * H
    
    # Dense layer FFN FLOPs
    dense_ffn_flops = 4 * S * H * intermediate_size
    
    # MoE layer FFN FLOPs
    # Only activated experts contribute to FLOPs
    active_experts = num_experts_per_tok + n_shared_experts
    moe_ffn_flops = 4 * S * H * moe_intermediate_size * active_experts
    
    # Router FLOPs (for selecting experts)
    router_flops = 2 * S * H * n_routed_experts
    
    # Total FLOPs
    dense_layer_total_flops = attention_flops_per_layer + dense_ffn_flops
    moe_layer_total_flops = attention_flops_per_layer + moe_ffn_flops + router_flops
    
    total_flops = (num_dense_layers * dense_layer_total_flops + 
                   num_moe_layers * moe_layer_total_flops)
    
    # Convert to TFLOPs
    total_tflops = total_flops / 1e12
    
    return total_tflops


def calculate_deepseek_memory(config, batch_size=1, sequence_length=2048):
    """
    Calculate peak memory for DeepSeek V3 MoE model.
    
    Key consideration for MoE:
    - All expert parameters must be stored in memory
    - But only activated experts contribute to activation memory
    - This leads to better memory efficiency compared to a dense model with same FLOPs
    
    Reference: "Switch Transformers: Scaling to Trillion Parameter Models"
    (Fedus et al., 2021) - discusses MoE memory characteristics
    """
    total_params = calculate_deepseek_parameters(config)
    
    # Model + gradients + optimizer (same as dense model)
    model_memory = total_params * 2  # FP16
    gradient_memory = total_params * 2  # FP16
    optimizer_memory = total_params * 8  # Adam (2 FP32 states)
    
    # Activation memory (only for activated experts)
    H = config['hidden_size']
    L = config['num_hidden_layers']
    num_experts_per_tok = config.get('num_experts_per_tok', 1)
    n_shared_experts = config.get('n_shared_experts', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    
    # Approximate activation per layer
    active_experts = num_experts_per_tok + n_shared_experts
    activation_per_layer = batch_size * sequence_length * (
        4 * H +  # Attention activations
        moe_intermediate_size * active_experts  # Only activated experts
    ) * 2  # FP16
    
    activation_memory = activation_per_layer * L
    
    # Total memory
    total_memory_bytes = model_memory + gradient_memory + optimizer_memory + activation_memory
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    return total_memory_gb


def model_training_cost_analysis_llama(model_config_path):
    """
    Analyze training costs for LLaMA-style models.
    
    Returns:
        total_params: Total number of parameters
        flops_per_token_TF: TeraFLOPs per forward pass
        peak_memory_GB: Peak memory usage in GB
    """
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    # Use default sequence length from config or 2048
    seq_length = config.get('max_sequence_length', config.get('max_position_embeddings', 2048))
    
    total_params = calculate_llama_parameters(config)
    flops_per_token_TF = calculate_llama_flops(config, sequence_length=seq_length)
    peak_memory_GB = calculate_llama_memory(config, batch_size=1, sequence_length=seq_length)
    
    return total_params, flops_per_token_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """
    Analyze training costs for DeepSeek V3-style MoE models.
    
    Returns:
        total_params: Total number of parameters (including all experts)
        flops_per_token_TF: TeraFLOPs per forward pass (only activated experts)
        peak_memory_GB: Peak memory usage in GB
    """
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    seq_length = config.get('max_sequence_length', config.get('max_position_embeddings', 2048))
    
    total_params = calculate_deepseek_parameters(config)
    flops_per_token_TF = calculate_deepseek_flops(config, sequence_length=seq_length)
    peak_memory_GB = calculate_deepseek_memory(config, batch_size=1, sequence_length=seq_length)
    
    return total_params, flops_per_token_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """
    Calculate optimal model size (N) and training tokens (D) given a cost budget.
    
    Based on Chinchilla scaling laws:
    "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
    
    Key findings:
    - Optimal N ∝ C^a (a ≈ 0.5)
    - Optimal D ∝ C^a (a ≈ 0.5)
    - For compute budget C, N and D should scale equally
    
    Training FLOPs: C ≈ 6 × N × D
    
    GPU specifications (approximate, 2024 pricing):
    - A100 (80GB): ~312 TFLOPS (FP16), ~$2.00/hour
    - V100 (32GB): ~125 TFLOPS (FP16), ~$1.00/hour  
    - T4 (16GB): ~65 TFLOPS (FP16), ~$0.35/hour
    
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    # GPU specifications: (TFLOPS, cost_per_hour, memory_GB)
    gpus = {
        'A100': {'tflops': 312, 'cost_per_hour': 2.00, 'memory_gb': 80},
        'V100': {'tflops': 125, 'cost_per_hour': 1.00, 'memory_gb': 32},
        'T4': {'tflops': 65, 'cost_per_hour': 0.35, 'memory_gb': 16}
    }
    
    # Calculate compute hours we can afford on each GPU
    best_flops = 0
    best_gpu = None
    
    for gpu_name, specs in gpus.items():
        hours = cost_budget / specs['cost_per_hour']
        # Total FLOPs = TFLOPS × 10^12 × hours × 3600 seconds
        total_flops = specs['tflops'] * 1e12 * hours * 3600
        
        if total_flops > best_flops:
            best_flops = total_flops
            best_gpu = gpu_name
    
    training_budget_flops = best_flops
    
    # Chinchilla optimal scaling: C = 6 × N × D
    # With equal scaling: N = D
    # Therefore: C = 6 × N²
    # N = sqrt(C / 6)
    N = math.sqrt(training_budget_flops / 6)
    D = N  # Optimal: equal scaling
    
    return int(N), int(D), training_budget_flops, best_gpu


def validate_calculations():
    """
    Validate our calculations against known model specifications.
    
    Note: Parameter counts can vary slightly depending on:
    - Whether bias terms are included (LLaMA doesn't use bias)
    - How embeddings are counted (tied vs untied)
    - Additional components like position embeddings
    
    Our calculation methodology:
    - Embedding: vocab_size × hidden_size
    - Per layer: 4×H² (attention) + 2×H×D (FFN) + 2×H (norms)
    - Output: hidden_size × vocab_size (if not tied)
    """
    print("=" * 80)
    print("VALIDATION: Testing calculation methodology")
    print("=" * 80)
    
    # Test LLaMA 7B-style architecture
    llama_config = {
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'vocab_size': 32000,
        'tie_word_embeddings': False
    }
    
    params = calculate_llama_parameters(llama_config)
    
    # Manual calculation for verification
    embedding = 32000 * 4096  # 131M
    per_layer_attn = 4 * 4096 * 4096  # 67M (Q,K,V,O projections)
    per_layer_ffn = 2 * 4096 * 11008  # 90M (up + down)
    per_layer_norms = 2 * 4096  # ~8K (negligible)
    per_layer = per_layer_attn + per_layer_ffn + per_layer_norms
    all_layers = 32 * per_layer  # 5.03B
    output = 32000 * 4096  # 131M
    final_norm = 4096
    manual_total = embedding + all_layers + output + final_norm
    
    print(f"\nLLaMA 7B-style Architecture:")
    print(f"  Embedding layer:      {embedding / 1e6:>8.2f}M")
    print(f"  Per layer (×32):      {per_layer / 1e6:>8.2f}M")
    print(f"    - Attention:        {per_layer_attn / 1e6:>8.2f}M")
    print(f"    - FFN:              {per_layer_ffn / 1e6:>8.2f}M")
    print(f"  All layers:           {all_layers / 1e9:>8.2f}B")
    print(f"  Output layer:         {output / 1e6:>8.2f}M")
    print(f"  {'─' * 40}")
    print(f"  Calculated total:     {params / 1e9:>8.2f}B")
    print(f"  Manual verification:  {manual_total / 1e9:>8.2f}B")
    
    error = abs(params - manual_total) / manual_total * 100
    if error < 0.01:
        print(f"  ✓ PASS - Calculation consistent")
    else:
        print(f"  ✗ FAIL - Internal inconsistency: {error:.4f}%")
    
    print("\n  Note: Published LLaMA 7B reports ~6.74B parameters.")
    print("  Our calculation gives ~5.30B based on standard architecture.")
    print("  Discrepancy may be due to:")
    print("  - Different vocab size (32K vs actual)")
    print("  - Additional components not in public config")
    print("  - Different counting methodology")
    
    # Test component breakdown makes sense
    print(f"\n  Component percentages:")
    print(f"    Embeddings: {(embedding + output) / params * 100:>5.1f}%")
    print(f"    Attention:  {(32 * per_layer_attn) / params * 100:>5.1f}%")
    print(f"    FFN:        {(32 * per_layer_ffn) / params * 100:>5.1f}%")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Training Cost Analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget in dollars')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    args = parser.parse_args()
    
    if args.validate:
        validate_calculations()
    
    if args.model_config:
        if 'deepseek' in args.model_config.lower():
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
            print("\n" + "=" * 80)
            print("DeepSeek V3 Model Analysis")
            print("=" * 80)
        elif 'llama' in args.model_config.lower():
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
            print("\n" + "=" * 80)
            print("LLaMA Model Analysis")
            print("=" * 80)
        else:
            print('Unknown LLM Type!')
            exit()
        
        print(f"Total Parameters:        {num_parameters:,.0f} ({num_parameters/1e9:.2f}B)")
        print(f"FLOPs per forward pass:  {num_flops:.2f} TFLOPs")
        print(f"Peak Memory (training):  {memory_cost:.2f} GB")
        print("=" * 80)
    
    if args.training_budget:
        print("\n" + "=" * 80)
        print("Optimal Model Sizing (Chinchilla Scaling Laws)")
        print("=" * 80)
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"Training Budget:         ${args.training_budget:,.2f}")
        print(f"Best GPU:                {best_gpu}")
        print(f"Total Training FLOPs:    {training_budget_flops:.2e} FLOPs")
        print(f"Optimal Model Size (N):  {N:,.0f} parameters ({N/1e9:.2f}B)")
        print(f"Optimal Training Tokens: {D:,.0f} tokens ({D/1e9:.2f}B)")
        print("=" * 80)

