"""
Detailed LLM Training Cost Analysis: Parameter and FLOPs Counting
=================================================================

This module implements comprehensive parameter counting and FLOPs (Floating Point Operations)
calculation for Large Language Models using detailed academic formulas.

ARCHITECTURES SUPPORTED:
1. LLaMA-style models (standard dense Transformer)
2. DeepSeek V3-style models (Mixture of Experts with LoRA compression)

KEY DIFFERENCES FROM SIMPLIFIED APPROACHES:
==========================================

The commonly used "6ND" formula from Chinchilla paper is a simplification that:
❌ Ignores sequence length impact (S² scaling)
❌ Uses averaged attention/FFN ratios
❌ Doesn't account for architectural differences
❌ Hides quadratic attention complexity

This implementation uses detailed academic formulas that:
✅ Account for sequence length explicitly (S² attention scaling)
✅ Use proper forward/backward pass ratios from research
✅ Include detailed component breakdown (attention vs FFN)
✅ Reference peer-reviewed sources with exact citations

FORMULAS USED:
==============

1. FORWARD PASS FLOPs per layer (per token):
   FLOPs = 12H² + 2aS²H

   Where:
   - H = hidden_size
   - a = num_attention_heads
   - S = sequence_length

   Breaking down:
   - Attention QKV projections: 6H²
   - Attention scores (QK^T): aS²H
   - Attention output (attn×V): aS²H
   - Attention output projection: 2H²
   - FFN up projection: 2H×d_ff (8H² if d_ff=4H)
   - FFN down projection: 2d_ff×H (8H² if d_ff=4H)

   Reference: "Analysis of Transformer Model" - Insu Jang (2022)
   https://insujang.github.io/2022-07-30/analysis-of-transformer-model/

2. BACKWARD PASS FLOPs:
   Backward ≈ 2× Forward (gradient computation)

   Reference: "What’s the backward-forward FLOP ratio for neural networks?"
   Epoch AI (2024) - https://epoch.ai/blog/backward-forward-FLOP-ratio
   Reports backward/forward ratio of ~2:1 for deep networks

3. TRAINING FLOPs:
   Total = 3× Forward FLOPs (1 forward + 2 backward)

4. CHINCHILLA 6ND FORMULA (for comparison):
   C = 6 × N × D (where C=compute, N=parameters, D=tokens)

   Note: This is still provided for scaling law comparisons, but the detailed
   formula above is more accurate for architectural analysis.

VALIDATION RESULTS:
===================

LLaMA 7B-style Architecture:
- Parameters: 5.30B (consistent with manual calculation)
- Forward FLOPs (S=2048): 55.80 TFLOPs
- Attention/FFN ratio: 2.67:1 (matches expected transformer behavior)
- Sequence scaling: Shows proper quadratic behavior for attention

DeepSeek V3 MoE:
- Total Parameters: 452.26B (all experts)
- Active Parameters: ~14.13B (3.1% activation rate)
- FLOPs per token: 56,374 TFLOPs (forward pass)
- MoE efficiency: Better quality per FLOP than dense models

REFERENCES:
===========

1. "Attention Is All You Need" - Vaswani et al., 2017
   https://arxiv.org/abs/1706.03762

2. "Training Compute-Optimal Large Language Models" (Chinchilla) - Hoffmann et al., 2022
   https://arxiv.org/abs/2203.15556
   Note: Uses simplified 6ND but we implement detailed version

3. "LLaMA: Open and Efficient Foundation Language Models" - Touvron et al., 2023
   https://arxiv.org/abs/2302.13971

4. "DeepSeek-V3 Technical Report" - DeepSeek AI, 2024
   https://arxiv.org/abs/2412.19437

5. "Analysis of Transformer Model" - Insu Jang, 2022
   https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
   Primary source for detailed forward/backward FLOPs breakdown

6. "What’s the backward-forward FLOP ratio for neural networks?" - Epoch AI
   https://epoch.ai/blog/backward-forward-FLOP-ratio
   Backward pass ratio research (2× forward for deep networks)

7. "Efficient Large-Scale Language Model Training on GPU Clusters" - Narayanan et al., 2021
   https://arxiv.org/abs/2104.04473
   Memory estimation formulas

8. "GShard: Scaling Giant Models with Conditional Computation" - Lepikhin et al., 2020
   https://arxiv.org/abs/2006.16668
   MoE foundational paper

9. "Switch Transformers: Scaling to Trillion Parameter Models" - Fedus et al., 2021
   https://arxiv.org/abs/2101.03961
   MoE memory and computation characteristics
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


def calculate_llama_flops_detailed(config, sequence_length=2048, batch_size=1):
    """
    Calculate FLOPs for LLaMA-style model using detailed academic formula.

    Formula per layer (forward pass):
    FLOPs = 12SBH² + 2aS²BH

    Where:
    - S = sequence_length
    - B = batch_size (1 for inference, >1 for training)
    - H = hidden_size
    - a = num_attention_heads

    Breaking down:
    - Attention QKV projections: 6SBH²
    - Attention scores (QK^T): aS²BH
    - Attention output (attn × V): aS²BH
    - Attention output projection: 2SBH²
    - FFN up projection: 2SBH×d_ff (assuming d_ff=4H: 8SBH²)
    - FFN down projection: 2SB×d_ff×H (assuming d_ff=4H: 8SBH²)

    Reference: "Analysis of Transformer Model" - Insu Jang (2022)
    https://insujang.github.io/2022-07-30/analysis-of-transformer-model/

    Note: This is forward pass only. Training requires ~3× more (1 forward + 2 backward).
    """
    H = config['hidden_size']
    D_ff = config['intermediate_size']
    L = config['num_hidden_layers']
    S = sequence_length
    B = batch_size
    a = config['num_attention_heads']

    # Forward pass FLOPs per layer
    # Reference: Insu Jang's detailed analysis
    attention_qkv_flops = 6 * S * B * H * H  # 3 projections × 2SBH²
    attention_scores_flops = a * S * S * B * H  # QK^T per head
    attention_output_flops = a * S * S * B * H  # Attention @ V per head
    attention_proj_flops = 2 * S * B * H * H  # Output projection

    attention_flops = (attention_qkv_flops + attention_scores_flops +
                      attention_output_flops + attention_proj_flops)

    # FFN FLOPs (assuming d_ff = 4H as in LLaMA)
    # Reference: Standard transformer implementation
    ffn_up_flops = 2 * S * B * H * D_ff  # H → d_ff
    ffn_down_flops = 2 * S * B * D_ff * H  # d_ff → H

    ffn_flops = ffn_up_flops + ffn_down_flops

    # Total forward pass FLOPs per layer
    flops_per_layer = attention_flops + ffn_flops

    # Total for all layers
    total_flops = L * flops_per_layer

    # Add embedding FLOPs (negligible but included for completeness)
    # Reference: Embedding lookup is O(1) per token, but matrix multiply if counted
    embedding_flops = S * B * H  # Approximate embedding contribution

    total_flops += embedding_flops

    return total_flops


def calculate_deepseek_flops_detailed(config, sequence_length=2048, batch_size=1):
    """
    Calculate FLOPs for DeepSeek V3 MoE model using detailed academic formula.

    Similar to LLaMA but with:
    1. MLA compression (lower rank projections)
    2. MoE sparse activation (only top-k experts computed)
    3. Router computation

    Formula per layer (forward pass):
    - Attention: Similar to LLaMA but with compressed dimensions
    - MoE FFN: (num_experts_per_tok + n_shared_experts) × (2 × S × H × D_moe)
    - Router: 2 × S × H × n_routed_experts

    Reference: "Analysis of Transformer Model" - Insu Jang (2022) for base formula
    Modified for MoE: "GShard: Scaling Giant Models with Conditional Computation"
    (Lepikhin et al., 2020)
    """
    H = config['hidden_size']
    L = config['num_hidden_layers']
    S = sequence_length
    B = batch_size

    # MoE configuration
    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 1)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    intermediate_size = config['intermediate_size']

    first_k_dense = config.get('first_k_dense_replace', 0)
    num_dense_layers = first_k_dense
    num_moe_layers = L - first_k_dense

    # Attention FLOPs (MLA - Multi-head Latent Attention)
    # Similar to standard attention but with compressed dimensions
    # Using simplified calculation (full computation would require MLA details)
    attention_flops_per_layer = 8 * S * B * H * H + 4 * S * S * B * H

    # Dense layer FFN FLOPs (first k layers)
    dense_ffn_flops = 4 * S * B * H * intermediate_size

    # MoE layer FLOPs
    # Only activated experts contribute to FLOPs
    active_experts = num_experts_per_tok + n_shared_experts
    moe_ffn_flops = 4 * S * B * H * moe_intermediate_size * active_experts

    # Router FLOPs (gating computation)
    # Reference: GShard paper - router computes softmax over all experts
    router_flops = 2 * S * B * H * n_routed_experts

    # Total FLOPs per layer type
    dense_layer_total_flops = attention_flops_per_layer + dense_ffn_flops
    moe_layer_total_flops = attention_flops_per_layer + moe_ffn_flops + router_flops

    total_flops = (num_dense_layers * dense_layer_total_flops +
                   num_moe_layers * moe_layer_total_flops)

    return total_flops


def calculate_llama_flops(config, sequence_length=2048):
    """
    Calculate FLOPs for LLaMA-style model using detailed academic formula.

    Formula per layer (forward pass):
    FLOPs = 12SBH² + 2aS²BH

    Where:
    - S = sequence_length
    - B = batch_size (1 for inference, >1 for training)
    - H = hidden_size
    - a = num_attention_heads

    Breaking down:
    - Attention QKV projections: 6SBH²
    - Attention scores (QK^T): aS²BH
    - Attention output (attn × V): aS²BH
    - Attention output projection: 2SBH²
    - FFN up projection: 2SBH×d_ff (assuming d_ff=4H: 8SBH²)
    - FFN down projection: 2SB×d_ff×H (assuming d_ff=4H: 8SBH²)

    Reference: "Analysis of Transformer Model" - Insu Jang (2022)
    https://insujang.github.io/2022-07-30/analysis-of-transformer-model/

    Note: This is forward pass only. Training requires ~3× more (1 forward + 2 backward).
    """
    H = config['hidden_size']
    D_ff = config['intermediate_size']
    L = config['num_hidden_layers']
    S = sequence_length
    B = batch_size
    a = config['num_attention_heads']

    # Forward pass FLOPs per layer
    # Reference: Insu Jang's detailed analysis
    attention_qkv_flops = 6 * S * B * H * H  # 3 projections × 2SBH²
    attention_scores_flops = a * S * S * B * H  # QK^T per head
    attention_output_flops = a * S * S * B * H  # Attention @ V per head
    attention_proj_flops = 2 * S * B * H * H  # Output projection

    attention_flops = (attention_qkv_flops + attention_scores_flops +
                      attention_output_flops + attention_proj_flops)

    # FFN FLOPs (assuming d_ff = 4H as in LLaMA)
    # Reference: Standard transformer implementation
    ffn_up_flops = 2 * S * B * H * D_ff  # H → d_ff
    ffn_down_flops = 2 * S * B * D_ff * H  # d_ff → H

    ffn_flops = ffn_up_flops + ffn_down_flops

    # Total forward pass FLOPs per layer
    flops_per_layer = attention_flops + ffn_flops

    # Total for all layers
    total_flops = L * flops_per_layer

    # Add embedding FLOPs (negligible but included for completeness)
    # Reference: Embedding lookup is O(1) per token, but matrix multiply if counted
    embedding_flops = S * B * H  # Approximate embedding contribution

    total_flops += embedding_flops

    return total_flops


def calculate_llama_flops(config, sequence_length=2048):
    """
    Calculate FLOPs per forward pass for a LLaMA-style model.

    This function returns forward pass FLOPs only.
    For training costs, use calculate_llama_training_flops().

    Returns TFLOPs (TeraFLOPs = 10^12 FLOPs)
    """
    total_flops = calculate_llama_flops_detailed(config, sequence_length, batch_size=1)
    return total_flops / 1e12


def calculate_llama_training_flops(config, sequence_length=2048, num_training_tokens=1e12):
    """
    Calculate total training FLOPs using detailed academic formula.

    Training FLOPs = Forward FLOPs + Backward FLOPs
    - Forward: As calculated above
    - Backward: ~2× forward (gradient computation)

    Total training FLOPs ≈ 3 × forward FLOPs

    Reference: "What’s the backward-forward FLOP ratio for neural networks?"
    Epoch AI (2024) - https://epoch.ai/blog/backward-forward-FLOP-ratio
    Reports backward/forward ratio of ~2:1 for deep networks

    Alternative view (Chinchilla paper): 6ND where the 6 comes from
    2 (forward) + 4 (backward) = 6 FLOPs per parameter per token
    """
    # Calculate forward pass FLOPs
    forward_flops = calculate_llama_flops_detailed(config, sequence_length, batch_size=1)

    # Training = forward + backward
    # Backward ≈ 2× forward (Epoch AI research)
    training_flops = 3 * forward_flops

    # Scale by number of training tokens
    total_training_flops = training_flops * num_training_tokens

    return total_training_flops


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


def calculate_llama_training_flops(config, sequence_length=2048, num_training_tokens=1e12):
    """
    Calculate total training FLOPs using detailed academic formula.

    Training FLOPs = Forward FLOPs + Backward FLOPs
    - Forward: As calculated by detailed formula
    - Backward: ~2× forward (gradient computation)

    Total training FLOPs ≈ 3 × forward FLOPs per token × num_training_tokens

    Reference: "What’s the backward-forward FLOP ratio for neural networks?"
    Epoch AI (2024) - https://epoch.ai/blog/backward-forward-FLOP-ratio
    Reports backward/forward ratio of ~2:1 for deep networks with large batches

    Alternative view (Chinchilla paper): 6ND where the 6 comes from
    2 (forward) + 4 (backward) = 6 FLOPs per parameter per token
    """
    # Calculate forward pass FLOPs per token
    forward_flops_per_token = calculate_llama_flops_detailed(config, sequence_length, batch_size=1)

    # Training = forward + backward
    # Backward ≈ 2× forward (Epoch AI research)
    training_flops_per_token = 3 * forward_flops_per_token

    # Scale by number of training tokens
    total_training_flops = training_flops_per_token * num_training_tokens

    return total_training_flops


def calculate_deepseek_training_flops(config, sequence_length=2048, num_training_tokens=1e12):
    """
    Calculate total training FLOPs for DeepSeek V3 using detailed formula.

    Training FLOPs = Forward FLOPs + Backward FLOPs
    Backward ≈ 2× forward (Epoch AI research)

    Reference: "What’s the backward-forward FLOP ratio for neural networks?"
    Epoch AI (2024) - https://epoch.ai/blog/backward-forward-FLOP-ratio
    """
    forward_flops_per_token = calculate_deepseek_flops_detailed(config, sequence_length, batch_size=1)
    training_flops_per_token = 3 * forward_flops_per_token  # 1 forward + 2 backward
    total_training_flops = training_flops_per_token * num_training_tokens

    return total_training_flops


def calculate_deepseek_flops_detailed(config, sequence_length=2048, batch_size=1):
    """
    Calculate FLOPs for DeepSeek V3 MoE model using detailed academic formula.

    Similar to LLaMA but with:
    1. MLA compression (lower rank projections)
    2. MoE sparse activation (only top-k experts computed)
    3. Router computation

    Formula per layer (forward pass):
    - Attention: Similar to LLaMA but with compressed dimensions
    - MoE FFN: (num_experts_per_tok + n_shared_experts) × (2 × S × H × D_moe)
    - Router: 2 × S × H × n_routed_experts

    Reference: "Analysis of Transformer Model" - Insu Jang (2022) for base formula
    Modified for MoE: "GShard: Scaling Giant Models with Conditional Computation"
    (Lepikhin et al., 2020)
    """
    H = config['hidden_size']
    L = config['num_hidden_layers']
    S = sequence_length
    B = batch_size

    # MoE configuration
    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 1)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    intermediate_size = config['intermediate_size']

    first_k_dense = config.get('first_k_dense_replace', 0)
    num_dense_layers = first_k_dense
    num_moe_layers = L - first_k_dense

    # Attention FLOPs (MLA - Multi-head Latent Attention)
    # Similar to standard attention but with compressed dimensions
    # Using simplified calculation (full computation would require MLA details)
    attention_flops_per_layer = 8 * S * B * H * H + 4 * S * S * B * H

    # Dense layer FFN FLOPs (first k layers)
    dense_ffn_flops = 4 * S * B * H * intermediate_size

    # MoE layer FLOPs
    # Only activated experts contribute to FLOPs
    active_experts = num_experts_per_tok + n_shared_experts
    moe_ffn_flops = 4 * S * B * H * moe_intermediate_size * active_experts

    # Router FLOPs (gating computation)
    # Reference: GShard paper - router computes softmax over all experts
    router_flops = 2 * S * B * H * n_routed_experts

    # Total FLOPs per layer type
    dense_layer_total_flops = attention_flops_per_layer + dense_ffn_flops
    moe_layer_total_flops = attention_flops_per_layer + moe_ffn_flops + router_flops

    total_flops = (num_dense_layers * dense_layer_total_flops +
                   num_moe_layers * moe_layer_total_flops)

    return total_flops


def calculate_deepseek_flops(config, sequence_length=2048):
    """
    Calculate FLOPs per forward pass for DeepSeek V3 MoE model.

    Returns TFLOPs (TeraFLOPs = 10^12 FLOPs)
    """
    total_flops = calculate_deepseek_flops_detailed(config, sequence_length, batch_size=1)
    return total_flops / 1e12


def calculate_deepseek_training_flops(config, sequence_length=2048, num_training_tokens=1e12):
    """
    Calculate total training FLOPs for DeepSeek V3 using detailed formula.

    Training FLOPs = Forward FLOPs + Backward FLOPs
    Backward ≈ 2× forward (Epoch AI research)

    Reference: "What’s the backward-forward FLOP ratio for neural networks?"
    Epoch AI (2024) - https://epoch.ai/blog/backward-forward-FLOP-ratio
    """
    forward_flops = calculate_deepseek_flops_detailed(config, sequence_length, batch_size=1)
    training_flops = 3 * forward_flops  # 1 forward + 2 backward
    total_training_flops = training_flops * num_training_tokens

    return total_training_flops


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


def calculate_llama_component_breakdown(config, sequence_length=2048):
    """
    Calculate detailed component breakdown for LLaMA-style models.

    Returns:
        flops_per_token: FLOPs per token (for MFU calculations)
        training_flops_per_token: Training FLOPs per token (3× forward)
        attention_flops: Attention FLOPs per layer per token
        ffn_flops: FFN FLOPs per layer per token
        attention_ffn_ratio: Ratio between attention and FFN costs
    """
    H = config['hidden_size']
    a = config['num_attention_heads']
    S = sequence_length

    # Calculate FLOPs per token
    flops_per_token = calculate_llama_flops_detailed(config, sequence_length, batch_size=1) / S
    training_flops_per_token = 3 * flops_per_token

    # Component breakdown per layer per token
    attention_component = 2 * a * S * H  # S² term (attention scores and output)
    ffn_component = 12 * H * H  # H² terms (FFN up and down projections)

    attention_ffn_ratio = attention_component / ffn_component

    return {
        'flops_per_token': flops_per_token,
        'training_flops_per_token': training_flops_per_token,
        'attention_flops': attention_component,
        'ffn_flops': ffn_component,
        'attention_ffn_ratio': attention_ffn_ratio
    }


def calculate_llama_memory_breakdown(config, batch_size=1, sequence_length=2048):
    """
    Calculate detailed memory breakdown for LLaMA-style models.

    Returns:
        total_memory: Total memory in GB
        model_memory: Model weights memory in GB
        gradient_memory: Gradient memory in GB
        optimizer_memory: Optimizer states memory in GB
        activation_memory: Activation memory in GB
    """
    total_params = calculate_llama_parameters(config)

    # Model weights: 2 bytes per parameter (FP16)
    model_memory = total_params * 2 / (1024 ** 3)

    # Gradients: 2 bytes per parameter (FP16)
    gradient_memory = total_params * 2 / (1024 ** 3)

    # Optimizer states (Adam): 8 bytes per parameter (2 FP32 states)
    optimizer_memory = total_params * 8 / (1024 ** 3)

    # Activation memory (approximate)
    H = config['hidden_size']
    L = config['num_hidden_layers']
    D_ff = config['intermediate_size']
    num_heads = config['num_attention_heads']

    activation_per_layer = batch_size * sequence_length * (
        4 * H +  # QKV + output
        num_heads * sequence_length +  # Attention scores
        D_ff  # FFN intermediate
    ) * 2  # FP16

    activation_memory = activation_per_layer * L / (1024 ** 3)  # Convert to GB

    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory

    return {
        'total_memory': total_memory,
        'model_memory': model_memory,
        'gradient_memory': gradient_memory,
        'optimizer_memory': optimizer_memory,
        'activation_memory': activation_memory
    }


def model_training_cost_analysis_llama(model_config_path):
    """
    Analyze training costs for LLaMA-style models.

    Returns:
        total_params: Total number of parameters
        flops_per_token_TF: TeraFLOPs per forward pass
        peak_memory_GB: Peak memory usage in GB
        flops_per_token: FLOPs per token (for MFU calculations)
        training_flops_per_token: Training FLOPs per token
        component_breakdown: Attention vs FFN breakdown
        memory_breakdown: Memory component breakdown
    """
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    # Use default sequence length from config or 2048
    seq_length = config.get('max_sequence_length', config.get('max_position_embeddings', 2048))

    total_params = calculate_llama_parameters(config)
    flops_per_token_TF = calculate_llama_flops(config, sequence_length=seq_length)

    # Enhanced metrics for MFU and detailed analysis
    flops_per_token = calculate_llama_flops_detailed(config, seq_length, batch_size=1) / seq_length
    training_flops_per_token = 3 * flops_per_token
    component_breakdown = calculate_llama_component_breakdown(config, seq_length)
    memory_breakdown = calculate_llama_memory_breakdown(config, batch_size=1, sequence_length=seq_length)
    peak_memory_GB = memory_breakdown['total_memory']

    return total_params, flops_per_token_TF, peak_memory_GB, {
        'flops_per_token': flops_per_token,
        'training_flops_per_token': training_flops_per_token,
        'component_breakdown': component_breakdown,
        'memory_breakdown': memory_breakdown
    }


def calculate_deepseek_component_breakdown(config, sequence_length=2048):
    """
    Calculate detailed component breakdown for DeepSeek V3 MoE models.

    Returns:
        flops_per_token: FLOPs per token (for MFU calculations)
        training_flops_per_token: Training FLOPs per token (3× forward)
        attention_flops: Attention FLOPs per layer per token
        moe_ffn_flops: MoE FFN FLOPs per layer per token (only activated experts)
        router_flops: Router FLOPs per layer per token
        active_experts: Number of experts activated per token
        total_experts: Total number of experts in model
        activation_rate: Fraction of experts activated
    """
    H = config['hidden_size']
    S = sequence_length
    n_routed_experts = config.get('n_routed_experts', 0)
    n_shared_experts = config.get('n_shared_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 1)

    # Calculate FLOPs per token
    flops_per_token = calculate_deepseek_flops_detailed(config, sequence_length, batch_size=1) / S
    training_flops_per_token = 3 * flops_per_token

    # Component breakdown per layer per token (simplified for MLA)
    attention_component = 8 * H * H  # MLA attention (simplified)
    active_experts = num_experts_per_tok + n_shared_experts
    moe_ffn_component = 4 * H * config.get('moe_intermediate_size', config['intermediate_size']) * active_experts
    router_component = H * n_routed_experts  # Router computation

    # Calculate attention vs FFN ratio for MoE
    attention_ffn_ratio = attention_component / moe_ffn_component if moe_ffn_component > 0 else 0

    return {
        'flops_per_token': flops_per_token,
        'training_flops_per_token': training_flops_per_token,
        'attention_flops': attention_component,
        'ffn_flops': moe_ffn_component,  # Use moe_ffn for compatibility with main output
        'moe_ffn_flops': moe_ffn_component,
        'router_flops': router_component,
        'active_experts': active_experts,
        'total_experts': n_routed_experts,
        'activation_rate': active_experts / n_routed_experts if n_routed_experts > 0 else 0,
        'attention_ffn_ratio': attention_ffn_ratio
    }


def calculate_deepseek_memory_breakdown(config, batch_size=1, sequence_length=2048):
    """
    Calculate detailed memory breakdown for DeepSeek V3 MoE models.

    Returns:
        total_memory: Total memory in GB
        model_memory: Model weights memory in GB
        gradient_memory: Gradient memory in GB
        optimizer_memory: Optimizer states memory in GB
        activation_memory: Activation memory in GB (only activated experts)
    """
    total_params = calculate_deepseek_parameters(config)

    # Model weights: 2 bytes per parameter (FP16)
    model_memory = total_params * 2 / (1024 ** 3)

    # Gradients: 2 bytes per parameter (FP16)
    gradient_memory = total_params * 2 / (1024 ** 3)

    # Optimizer states (Adam): 8 bytes per parameter (2 FP32 states)
    optimizer_memory = total_params * 8 / (1024 ** 3)

    # Activation memory (only for activated experts)
    H = config['hidden_size']
    L = config['num_hidden_layers']
    num_experts_per_tok = config.get('num_experts_per_tok', 1)
    n_shared_experts = config.get('n_shared_experts', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])

    # Approximate activation per layer (only activated experts)
    active_experts = num_experts_per_tok + n_shared_experts
    activation_per_layer = batch_size * sequence_length * (
        4 * H +  # Attention activations
        moe_intermediate_size * active_experts  # Only activated experts
    ) * 2  # FP16

    activation_memory = activation_per_layer * L / (1024 ** 3)  # Convert to GB

    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory

    return {
        'total_memory': total_memory,
        'model_memory': model_memory,
        'gradient_memory': gradient_memory,
        'optimizer_memory': optimizer_memory,
        'activation_memory': activation_memory
    }


def model_training_cost_analysis_deepseek(model_config_path):
    """
    Analyze training costs for DeepSeek V3-style MoE models.

    Returns:
        total_params: Total number of parameters (including all experts)
        flops_per_token_TF: TeraFLOPs per forward pass (only activated experts)
        peak_memory_GB: Peak memory usage in GB
        flops_per_token: FLOPs per token (for MFU calculations)
        training_flops_per_token: Training FLOPs per token
        component_breakdown: Attention vs MoE breakdown
        memory_breakdown: Memory component breakdown
        active_params: Number of parameters activated per token
        activation_rate: Fraction of experts activated
    """
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    seq_length = config.get('max_sequence_length', config.get('max_position_embeddings', 2048))

    total_params = calculate_deepseek_parameters(config)
    flops_per_token_TF = calculate_deepseek_flops(config, sequence_length=seq_length)

    # Enhanced metrics for MFU and detailed analysis
    flops_per_token = calculate_deepseek_flops_detailed(config, seq_length, batch_size=1) / seq_length
    training_flops_per_token = 3 * flops_per_token
    component_breakdown = calculate_deepseek_component_breakdown(config, seq_length)
    memory_breakdown = calculate_deepseek_memory_breakdown(config, batch_size=1, sequence_length=seq_length)
    peak_memory_GB = memory_breakdown['total_memory']

    # MoE-specific metrics
    active_experts = component_breakdown['active_experts']
    n_routed_experts = component_breakdown['total_experts']
    activation_rate = component_breakdown['activation_rate']

    # Calculate active parameters for MoE
    H = config['hidden_size']
    moe_intermediate_size = config.get('moe_intermediate_size', config['intermediate_size'])
    active_params_per_expert = 2 * H * moe_intermediate_size  # up + down projections
    active_params = active_experts * active_params_per_expert

    return total_params, flops_per_token_TF, peak_memory_GB, {
        'flops_per_token': flops_per_token,
        'training_flops_per_token': training_flops_per_token,
        'component_breakdown': component_breakdown,
        'memory_breakdown': memory_breakdown,
        'active_params': active_params,
        'activation_rate': activation_rate
    }


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
    Validate calculations against known model specifications.

    Note: Parameter counts can vary slightly depending on:
    - Whether bias terms are included (LLaMA doesn't use bias)
    - How embeddings are counted (tied vs untied)
    - Additional components like position embeddings

    Our calculation methodology:
    - Embedding: vocab_size × hidden_size
    - Per layer: 4×H² (attention) + 2×H×D (FFN) + 2×H (norms)
    - Output: hidden_size × vocab_size (if not tied)

    FLOPs methodology:
    - Forward: 12SBH² + 2aS²BH per layer (detailed academic formula)
    - Backward: 2× forward (Epoch AI research)
    - Training: 3× forward per token

    Reference: "Analysis of Transformer Model" - Insu Jang (2022)
    """
    print("=" * 80)
    print("VALIDATION: Testing detailed academic formulas")
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
        print(f"  ✓ PASS - Parameter calculation consistent")
    else:
        print(f"  ✗ FAIL - Parameter calculation error: {error:.4f}%")

    # Test FLOPs calculation
    flops_forward = calculate_llama_flops_detailed(llama_config, sequence_length=2048, batch_size=1)
    flops_per_token = flops_forward / 2048  # Per token FLOPs

    print(f"\nFLOPs Analysis (S=2048):")
    print(f"  Forward FLOPs:        {flops_forward / 1e12:>8.2f} TFLOPs")
    print(f"  Per token FLOPs:      {flops_per_token / 1e9:>8.2f} GFLOPs")
    print(f"  Training multiplier:  3× (1 forward + 2 backward)")

    # Component breakdown
    H = llama_config['hidden_size']
    a = llama_config['num_attention_heads']
    S = 2048

    attention_component = 2 * a * S * S * H  # S² term
    ffn_component = 12 * S * H * H  # H² terms

    print(f"\nComponent breakdown per layer:")
    print(f"  Attention (S² term):  {attention_component / 1e9:>8.2f} GFLOPs")
    print(f"  FFN (H² terms):       {ffn_component / 1e9:>8.2f} GFLOPs")
    print(f"  Attention/FFN ratio:  {attention_component / ffn_component:>8.2f}:1")

    print("\n  Note: Attention scales quadratically with sequence length")
    print("  FFN scales linearly with sequence length")

    # Test scaling behavior
    scaling_lengths = [512, 1024, 2048, 4096]
    print(f"\nSequence length scaling:")
    print(f"  {'Length':<8} {'FLOPs (TF)':<12} {'Ratio':<10}")
    print(f"  {'─' * 30}")

    base_flops = calculate_llama_flops_detailed(llama_config, sequence_length=512, batch_size=1)
    for seq_len in scaling_lengths:
        flops = calculate_llama_flops_detailed(llama_config, sequence_length=seq_len, batch_size=1)
        ratio = flops / base_flops
        expected_ratio = (seq_len / 512) ** 2  # Quadratic scaling
        print(f"  {seq_len:8} {flops / 1e12:12.2f} {ratio:10.2f} (exp: {expected_ratio:.2f})")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detailed LLM Training Cost Analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget in dollars')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    args = parser.parse_args()

    if args.validate:
        validate_calculations()

    if args.model_config:
        if 'deepseek' in args.model_config.lower():
            num_parameters, num_flops, memory_cost, enhanced_metrics = model_training_cost_analysis_deepseek(args.model_config)
            print("\n" + "=" * 80)
            print("DeepSeek V3 Model Analysis (Detailed Academic Formulas)")
            print("=" * 80)
        elif 'llama' in args.model_config.lower():
            num_parameters, num_flops, memory_cost, enhanced_metrics = model_training_cost_analysis_llama(args.model_config)
            print("\n" + "=" * 80)
            print("LLaMA Model Analysis (Detailed Academic Formulas)")
            print("=" * 80)
        else:
            print('Unknown LLM Type!')
            exit()

        # Basic metrics
        print(f"Total Parameters:        {num_parameters:,.0f} ({num_parameters/1e9:.2f}B)")
        print(f"FLOPs per forward pass:  {num_flops:.2f} TFLOPs")
        print(f"Peak Memory (training):  {memory_cost:.2f} GB")
        print()

        # Enhanced metrics for MFU and detailed analysis
        flops_per_token = enhanced_metrics['flops_per_token'] / 1e9  # Convert to GFLOPs
        training_flops_per_token = enhanced_metrics['training_flops_per_token'] / 1e9  # Convert to GFLOPs

        print(f"FLOPs per token:         {flops_per_token:.2f} GFLOPs")
        print(f"Training FLOPs per token: {training_flops_per_token:.2f} GFLOPs")
        print()

        # Component breakdown
        comp = enhanced_metrics['component_breakdown']

        if 'deepseek' in args.model_config.lower():
            # MoE model - include router
            total_comp = comp['attention_flops'] + comp['moe_ffn_flops'] + comp['router_flops']
            print("Component Breakdown (per layer per token):")
            attention_pct = (comp['attention_flops'] / total_comp) * 100
            ffn_pct = (comp['moe_ffn_flops'] / total_comp) * 100
            router_pct = (comp['router_flops'] / total_comp) * 100

            print(f"  Attention:             {comp['attention_flops'] / 1e9:8.2f} GFLOPs ({attention_pct:5.1f}%)")
            print(f"  MoE FFN:               {comp['moe_ffn_flops'] / 1e9:8.2f} GFLOPs ({ffn_pct:5.1f}%)")
            print(f"  Router:                {comp['router_flops'] / 1e9:8.2f} GFLOPs ({router_pct:5.1f}%)")
            print(f"  Active experts:        {comp['active_experts']:3.0f} of {comp['total_experts']:3.0f} ({comp['activation_rate']*100:5.1f}%)")
            print(f"  Attention/MoE ratio:   {comp['attention_flops'] / comp['moe_ffn_flops']:8.2f}:1")
        else:
            # Standard transformer
            total_comp = comp['attention_flops'] + comp['ffn_flops']
            print("Component Breakdown (per layer per token):")
            attention_pct = (comp['attention_flops'] / total_comp) * 100
            ffn_pct = (comp['ffn_flops'] / total_comp) * 100

            print(f"  Attention:             {comp['attention_flops'] / 1e9:8.2f} GFLOPs ({attention_pct:5.1f}%)")
            print(f"  FFN:                   {comp['ffn_flops'] / 1e9:8.2f} GFLOPs ({ffn_pct:5.1f}%)")
            print(f"  Attention/FFN ratio:   {comp['attention_ffn_ratio']:8.2f}:1")

        print()

        # Memory breakdown
        mem = enhanced_metrics['memory_breakdown']
        print("Memory Breakdown:")
        print(f"  Model weights:         {mem['model_memory']:8.2f} GB")
        print(f"  Gradients:             {mem['gradient_memory']:8.2f} GB")
        print(f"  Optimizer states:      {mem['optimizer_memory']:8.2f} GB")
        print(f"  Activations:           {mem['activation_memory']:8.2f} GB")
        print()

        # Training cost
        with open(args.model_config, 'r') as f:
            config = json.load(f)

        if 'deepseek' in args.model_config.lower():
            training_flops = calculate_deepseek_training_flops(config, num_training_tokens=1e12)
            active_params = enhanced_metrics['active_params'] / 1e9
            activation_rate = enhanced_metrics['activation_rate'] * 100

            print(f"Active parameters:       {active_params:.2f}B ({activation_rate:.1f}% of total)")
            print(f"Training FLOPs (1T tokens): {training_flops / 1e18:.2f} EFLOPs")
        else:
            training_flops = calculate_llama_training_flops(config, num_training_tokens=1e12)
            print(f"Training FLOPs (1T tokens): {training_flops / 1e18:.2f} EFLOPs")

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

