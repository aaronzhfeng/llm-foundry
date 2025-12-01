# LLM Parameter and FLOPs Counting Implementation

A comprehensive implementation for calculating parameters, FLOPs, and memory requirements for Large Language Models (LLMs), with support for both standard dense Transformers (LLaMA) and Mixture of Experts models (DeepSeek V3).

## Overview

This implementation provides accurate parameter counting and FLOPs calculation for:
1. **LLaMA-style models**: Standard dense Transformer architecture
2. **DeepSeek V3-style models**: Mixture of Experts (MoE) with Multi-head Latent Attention (MLA)

## Usage

### Basic Usage

```bash
# Analyze LLaMA 7B model
python llm_cost_analysis.py --model_config llama_7b_config.json

# Analyze DeepSeek V3 model
python llm_cost_analysis.py --model_config deepseek_v3_config.json

# Calculate optimal model size for a given budget
python llm_cost_analysis.py --training_budget 10000

# Run validation tests
python llm_cost_analysis.py --validate
```

### Example Output

```
================================================================================
LLaMA Model Analysis
================================================================================
Total Parameters:        6,738,415,616 (6.74B)
FLOPs per forward pass:  173.95 TFLOPs
Peak Memory (training):  87.34 GB
================================================================================
```

## Special Cases and Implementation Details

### 1. Grouped Query Attention (GQA)

**What it is**: GQA reduces the number of key-value heads compared to query heads to save memory and computation while maintaining most of the model quality.

**Implementation**:
```python
# Check if model uses GQA
num_kv_heads = config.get('num_key_value_heads', config['num_attention_heads'])
num_q_heads = config['num_attention_heads']

if num_kv_heads == num_q_heads:
    # Standard Multi-Head Attention (MHA)
    attention_params = 4 * H * H
else:
    # Grouped Query Attention (GQA)
    head_dim = H // num_q_heads
    attention_params = H * H  # Q projection (full)
    attention_params += 2 * H * (num_kv_heads * head_dim)  # K, V projections (compressed)
    attention_params += H * H  # O projection
```

**Example**: LLaMA 2 70B uses GQA with 64 query heads but only 8 KV heads, reducing memory by 8×.

**Reference**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

---

### 2. Multi-head Latent Attention (MLA) with LoRA Compression

**What it is**: DeepSeek V3's innovation that uses low-rank decomposition to compress Q, K, V projections, significantly reducing parameters while maintaining performance.

**Implementation**:
```python
# MLA with LoRA-style compression
q_lora_rank = config.get('q_lora_rank', H)  # e.g., 1536 instead of 7168
kv_lora_rank = config.get('kv_lora_rank', H)  # e.g., 512

# Q projection: H × rank + rank × (num_heads × head_dim)
q_proj_params = H * q_lora_rank + q_lora_rank * (num_q_heads * q_head_dim)

# Similar for K, V
k_proj_params = H * kv_lora_rank + kv_lora_rank * (num_kv_heads * k_head_dim)
v_proj_params = H * kv_lora_rank + kv_lora_rank * (num_kv_heads * v_head_dim)
```

**Why it matters**: 
- Standard attention: H² parameters per projection
- MLA: ~H × rank parameters (much smaller when rank << H)
- For DeepSeek V3: Saves millions of parameters per layer

**Reference**: DeepSeek-V3 Technical Report (2024), Section 2.1

---

### 3. Mixture of Experts (MoE)

**What it is**: Instead of one large FFN per layer, MoE uses multiple "expert" networks, but only activates a few per token (sparse activation).

**Key Concepts**:
- **Total Parameters**: All experts exist as parameters
- **Activated Parameters**: Only top-k experts are computed per token
- **Shared Experts**: Always activated (provide stable knowledge)
- **Routed Experts**: Selected by a gating mechanism

**Implementation**:
```python
# Total parameters (all experts stored)
shared_expert_params = n_shared_experts * (2 * H * moe_intermediate_size)
routed_expert_params = n_routed_experts * (2 * H * moe_intermediate_size)
router_params = H * n_routed_experts

total_moe_params = shared_expert_params + routed_expert_params + router_params

# But for FLOPs, only activated experts count
active_experts = num_experts_per_tok + n_shared_experts  # e.g., 8 + 1 = 9
moe_ffn_flops = 4 * S * H * moe_intermediate_size * active_experts
```

**Example (DeepSeek V3)**:
- 256 routed experts + 1 shared expert
- Only 8 routed + 1 shared = 9 experts activated per token
- Total params: 256 experts stored, but FLOPs: only 9 computed

**Why it matters**:
- **Parameter efficiency**: Can have 671B total parameters
- **Compute efficiency**: But only use ~37B activated parameters
- **Result**: Better quality per FLOPs than dense models

**Reference**: 
- "GShard: Scaling Giant Models with Conditional Computation" (Lepikhin et al., 2020)
- "Switch Transformers" (Fedus et al., 2021)

---

### 4. Dense vs. MoE Layer Distribution

**What it is**: DeepSeek V3 uses dense FFN layers for the first few layers, then switches to MoE.

**Implementation**:
```python
first_k_dense = config.get('first_k_dense_replace', 0)  # e.g., 3
num_dense_layers = first_k_dense
num_moe_layers = L - first_k_dense  # e.g., 61 - 3 = 58

# Calculate separately
dense_layer_params = attention_params + dense_ffn_params + layernorm_params
moe_layer_params = attention_params + moe_ffn_params + layernorm_params

total_params = (num_dense_layers * dense_layer_params + 
               num_moe_layers * moe_layer_params)
```

**Why it matters**: Early layers need dense computation for basic feature extraction; later layers can use sparse experts for specialized knowledge.

---

### 5. Tied Embeddings

**What it is**: Input embeddings and output projection can share the same weight matrix.

**Implementation**:
```python
tie_word_embeddings = config.get('tie_word_embeddings', False)

embedding_params = V * H  # Input embeddings

if tie_word_embeddings:
    output_params = 0  # Reuse input embeddings
else:
    output_params = H * V  # Separate output projection
```

**Why it matters**: Saves V × H parameters (e.g., 32000 × 4096 = 131M for LLaMA 7B)

---

## Validation Strategy

### 1. Known Model Validation

Compare against published model sizes:

```python
def validate_calculations():
    # LLaMA 7B: Known to have ~6.74B parameters
    llama_config = {...}
    params = calculate_llama_parameters(llama_config)
    expected = 6.74e9
    error = abs(params - expected) / expected * 100
    assert error < 5%, f"Error too large: {error:.2f}%"
```

**Known benchmarks**:
- LLaMA 7B: 6.74B parameters
- LLaMA 13B: 13.02B parameters
- LLaMA 70B: 65.2B parameters (with GQA)
- DeepSeek V3: ~671B total parameters, ~37B activated

### 2. Cross-Reference with Tools

Compare with existing tools:
- `transformers.modeling_utils.num_parameters()`
- `calflops` library: https://pypi.org/project/calflops/
- Manual counting from model architecture diagrams

### 3. Sanity Checks

```python
# Check 1: FLOPs should scale roughly with parameters
# Rule of thumb: Dense model with N params should have ~6N FLOPs per token
flops_per_param = total_flops / total_params
assert 4 < flops_per_param < 8, "FLOPs/param ratio out of expected range"

# Check 2: Memory should be dominated by model + gradients + optimizer
# For Adam: ~12 bytes per parameter (2 + 2 + 8)
memory_per_param = total_memory_bytes / total_params
assert 10 < memory_per_param < 20, "Memory/param ratio unexpected"

# Check 3: MoE should have lower FLOPs/param than dense
# Because only some experts are activated
if is_moe:
    activated_ratio = num_experts_per_tok / n_routed_experts
    assert activated_ratio < 0.1, "Too many experts activated"
```

### 4. Component-wise Validation

Break down and verify each component:

```python
# Verify attention parameters
expected_attn = 4 * H * H  # Q, K, V, O projections
calculated_attn = calculate_attention_params(config)
assert abs(expected_attn - calculated_attn) / expected_attn < 0.01

# Verify FFN parameters
expected_ffn = 2 * H * D_ff
calculated_ffn = calculate_ffn_params(config)
assert abs(expected_ffn - calculated_ffn) / expected_ffn < 0.01
```

### 5. Scaling Law Validation

Verify against Chinchilla scaling laws:

```python
# For compute budget C, optimal N and D should satisfy:
# C ≈ 6 × N × D
# And N ≈ D (equal scaling)

C = 1e21  # Example: 1 ZettaFLOP
N, D, _, _ = get_optimal_N_D_from_cost(C)

# Verify
computed_C = 6 * N * D
error = abs(computed_C - C) / C
assert error < 0.01, "Chinchilla formula not satisfied"

ratio = N / D
assert 0.9 < ratio < 1.1, "N and D should scale equally"
```

---

## Formulas and References

### Core FLOPs Formula (Forward Pass)

**For a single token in a single layer**:

```
Attention FLOPs:
  - QKV projections: 6 × S × H²
  - Attention scores: 2 × S² × H
  - Attention output: 2 × S² × H
  - Output projection: 2 × S × H²
  Total: 8SH² + 4S²H

FFN FLOPs:
  - Up projection: 2 × S × H × D
  - Down projection: 2 × S × D × H
  Total: 4 × S × H × D

Total per layer: 8SH² + 4S²H + 4SHD
Total model: L × (8SH² + 4S²H + 4SHD)
```

Where:
- S = sequence length
- H = hidden dimension
- D = FFN intermediate dimension
- L = number of layers

**Reference**: 
- "Efficient Large-Scale Language Model Training" (Narayanan et al., 2021)
- https://dsdanielpark.github.io/llm/2023-12-12-LLaMAFLOPSEstimiation.html

### Training FLOPs (Chinchilla)

```
Total Training FLOPs ≈ 6 × N × D
```

Where:
- N = total model parameters
- D = number of training tokens
- Factor of 6 accounts for: 2 (forward) + 4 (backward)

**Reference**: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
https://arxiv.org/abs/2203.15556

### Memory Formula

```
Training Memory (bytes) = 
    Model Parameters × 2 (FP16) +
    Gradients × 2 (FP16) +
    Optimizer States × 8 (Adam: 2 × FP32) +
    Activations (depends on batch size and sequence length)

Simplified: Memory ≈ 12 × N + activation_memory
```

**Reference**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)

---

## Key Papers Referenced

1. **Attention Mechanism**
   - Vaswani et al., "Attention Is All You Need" (2017)
   - https://arxiv.org/abs/1706.03762

2. **LLaMA Architecture**
   - Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
   - https://arxiv.org/abs/2302.13971

3. **Chinchilla Scaling Laws**
   - Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
   - https://arxiv.org/abs/2203.15556

4. **Grouped Query Attention**
   - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer" (2023)
   - https://arxiv.org/abs/2305.13245

5. **Mixture of Experts**
   - Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (2020)
   - https://arxiv.org/abs/2006.16668
   - Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (2021)
   - https://arxiv.org/abs/2101.03961

6. **DeepSeek V3**
   - DeepSeek AI, "DeepSeek-V3 Technical Report" (2024)
   - https://arxiv.org/abs/2412.19437

7. **Memory Optimization**
   - Rajbhandari et al., "ZeRO: Memory Optimizations" (2020)
   - https://arxiv.org/abs/1910.02054

8. **Training at Scale**
   - Narayanan et al., "Efficient Large-Scale Language Model Training" (2021)
   - https://arxiv.org/abs/2104.04473

---

## Testing

Run the validation suite:

```bash
python llm_cost_analysis.py --validate
```

Expected output:
```
================================================================================
VALIDATION: Testing against known model specifications
================================================================================

LLaMA 7B:
  Calculated: 6.74B parameters
  Expected:   6.74B parameters
  Error:      0.12%
  ✓ PASS
================================================================================
```

---

## Notes on Approximations

1. **Layer Norms**: We include RMSNorm parameters (H per norm) but they're negligible compared to attention and FFN.

2. **Activation Memory**: Our calculation is approximate. Actual memory depends on:
   - Gradient checkpointing strategy
   - Activation recomputation
   - Mixed precision training details

3. **FLOPs**: We count multiply-add as 2 FLOPs (standard convention). Some tools count it as 1 FLOP.

4. **MoE Router FLOPs**: Router computation is relatively small but included for completeness.

5. **Expert Load Balancing**: We assume perfect load balancing. Real systems may activate slightly more experts for load balancing.

---

## Future Extensions

Potential improvements:
1. Support for more architectures (GPT, T5, etc.)
2. Gradient checkpointing impact on memory
3. Pipeline parallelism and model parallelism effects
4. Flash Attention memory optimization
5. Sparse attention patterns (e.g., local + global attention)
6. Quantization effects on memory and computation

---

## Contact & Contributions

For questions or improvements, please refer to the original assignment materials or contact the course instructors.

