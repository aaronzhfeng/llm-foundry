# LLaMA Architecture Configuration

## Quick Reference

### Architecture Components
- **Preset**: `llama`
- **Normalization**: RMSNorm
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Activation**: SiLU (within SwiGLU)
- **FFN Type**: SwiGLU (2.67× expansion, ~8/3)
- **Norm Position**: Pre-norm
- **Attention Backend**: FlashAttention-2 (or SDPA fallback)
- **Weight Tying**: No
- **Bias**: False
- **Dropout**: 0.0

### Model Dimensions (LLaMA 124M - for comparison)
- **Layers**: 12
- **Hidden size**: 768
- **Attention heads**: 12
- **Head dimension**: 64 (768 / 12)
- **FFN dimension**: 2048 (2.67× expansion for SwiGLU)
- **Context length**: 1024
- **Vocabulary**: 50304 (for comparison) or 32000 (with LLaMA tokenizer)

### Model Dimensions (LLaMA 1.36B - production)
- **Layers**: 18
- **Hidden size**: 2304
- **Attention heads**: 18
- **Head dimension**: 128 (2304 / 18)
- **FFN dimension**: 6144 (2.67× expansion for SwiGLU)
- **Context length**: 2048
- **Vocabulary**: 32000 (LLaMA tokenizer)

### Total Parameters (1.36B model)
- **1.36B** parameters
  - Token embeddings: 73.7M
  - Output projection: 73.7M (not tied)
  - Position embeddings: 0 (RoPE has no parameters)
  - Attention layers: 383M (18 layers)
  - SwiGLU FFN layers: 765M (18 layers)
  - Normalization: ~0.1M

---

## Reference & Explanations

### Architecture Origins

**Primary Source:**
> **LLaMA: Open and Efficient Foundation Language Models**  
> Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. (Meta AI)  
> arXiv:2302.13971, February 2023  
> https://arxiv.org/abs/2302.13971

**Component Papers:**

**RoPE:**
> **RoFormer: Enhanced Transformer with Rotary Position Embedding**  
> Jianlin Su, Yu Lu, Shengfeng Pan, et al.  
> arXiv:2104.09864, April 2021  
> https://arxiv.org/abs/2104.09864

**RMSNorm:**
> **Root Mean Square Layer Normalization**  
> Biao Zhang, Rico Sennrich  
> arXiv:1910.07467, October 2019  
> https://arxiv.org/abs/1910.07467

**SwiGLU:**
> **GLU Variants Improve Transformer**  
> Noam Shazeer (Google Brain)  
> arXiv:2002.05202, February 2020  
> https://arxiv.org/abs/2002.05202

---

## Component Details

### 1. RoPE (Rotary Position Embeddings)

**What it is:**
- Applies rotation to Query and Key vectors based on position
- No learned parameters (computed from position indices)
- Naturally encodes relative position information

**Formula:**
```python
# Frequency for dimension i:
freq_i = θ^(-2i/d) where θ = 10000

# For position m, rotate by angle m × freq_i:
q_rot = rotate(q, m × freq)
k_rot = rotate(k, m × freq)
```

**Rotation Matrix (per dimension pair):**
```
[cos(mθ)  -sin(mθ)]   [q_even]
[sin(mθ)   cos(mθ)] × [q_odd ]
```

**Pros:**
- ✅ Excellent length extrapolation (can handle >training length)
- ✅ Encodes relative positions naturally
- ✅ No extra parameters
- ✅ Works well for long-range dependencies

**Cons:**
- ❌ Slightly more complex implementation
- ❌ Small computational overhead (~2-3%)

**Key Insight:** Attention naturally sees relative positions through dot product of rotated Q and K.

---

### 2. RMSNorm (Root Mean Square Normalization)

**What it is:**
- Simplified normalization that only scales (no centering)
- ~15-20% faster than LayerNorm with similar performance

**Formula:**
```python
RMS(x) = sqrt(mean(x²) + ε)
RMSNorm(x) = (x / RMS(x)) * γ
```

**Comparison with LayerNorm:**
```python
# LayerNorm (2 steps):
x_norm = (x - mean(x)) / sqrt(var(x) + ε)  # Center + scale
output = γ * x_norm + β                     # Affine transform

# RMSNorm (1 step):
x_norm = x / sqrt(mean(x²) + ε)            # Just scale
output = γ * x_norm                         # No bias
```

**Settings:**
- `eps = 1e-6` (LLaMA 1) or `1e-5` (LLaMA 2)
- No bias parameter
- Only learnable scale (γ)

**Pros:**
- ✅ ~15% faster than LayerNorm
- ✅ Simpler (fewer operations)
- ✅ Similar or better performance

**Cons:**
- ❌ Less common (fewer library optimizations)

---

### 3. SwiGLU FFN

**What it is:**
- Gated activation mechanism with Swish/SiLU
- Uses 3 linear projections instead of 2
- Expansion ratio: ~8/3 ≈ 2.67× (vs 4× for standard FFN)

**Formula:**
```python
SwiGLU(x) = (Swish(xW_gate) ⊙ xW_value) W_out

# Where:
Swish(x) = x × sigmoid(x)  # Also called SiLU
⊙ = element-wise multiplication
```

**Structure:**
```python
gate = Linear(x)      # d_model → d_ff
value = Linear(x)     # d_model → d_ff
hidden = SiLU(gate) * value
output = Linear(hidden)  # d_ff → d_model
```

**Parameters per layer:**
- 3 × d_model × d_ff (vs 2× for standard FFN)
- Example (1.36B model): 3 × 2304 × 6144 = 42.5M per layer

**Pros:**
- ✅ Better performance than GELU/ReLU
- ✅ Gating mechanism = better information flow
- ✅ Used in all modern LLMs (PaLM, LLaMA, etc.)

**Cons:**
- ❌ 50% more parameters than standard FFN
- ❌ Slightly more computation per token

**Why 8/3 expansion?**
- For same parameter count as standard 4× FFN:
- Standard: 2 × d × 4d = 8d² parameters
- SwiGLU: 3 × d × (8/3)d = 8d² parameters

---

### 4. Pre-Norm Architecture

**What it is:**
- Apply normalization BEFORE sub-layer (attention/FFN)
- Better gradient flow for deep networks

**Structure:**
```python
# Pre-norm (LLaMA):
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))

# vs Post-norm (GPT-2):
x = RMSNorm(x + Attention(x))
x = RMSNorm(x + FFN(x))
```

**Pros:**
- ✅ Better gradient flow (critical for 18+ layers)
- ✅ More stable training
- ✅ Can use higher learning rates
- ✅ Standard for modern deep transformers

**Cons:**
- ❌ Slightly different final representation

**When it matters:**
- Essential for models with >16 layers
- Less critical for shallow models (6-12 layers)

---

### 5. No Weight Tying

**What it is:**
- Input embeddings and output projection are separate
- Allows each to specialize for different tasks

**Parameters:**
```python
# Input: token → representation
token_emb = nn.Embedding(32000, 2304)  # 73.7M params

# Output: representation → logits
output_proj = nn.Linear(2304, 32000)   # 73.7M params

# Total: 147.4M parameters (not shared)
```

**Pros:**
- ✅ Better for large models (>1B params)
- ✅ Each layer can specialize
- ✅ Improved performance

**Cons:**
- ❌ ~2× more embedding parameters
- ❌ Higher memory usage

**Rule of thumb:**
- Models <500M: Weight tying is fine
- Models >1B: No tying is better

---

## Usage Examples

### Basic Training

```bash
# Train LLaMA 124M (for comparison with GPT-2)
python train.py config/full_llama_124m.py

# Train LLaMA 1.36B (production model)
python train.py config/full_llama_1.36b.py
```

### Multi-GPU Training

```bash
# Test run on 4x A100 (6B tokens)
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_llama_1.36b.py \
  --dataset=slimpajama_6b \
  --max_iters=2000

# Production run on 8x B200 (627B tokens)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --dataset=slimpajama_627b \
  --use_fsdp=True \
  --max_iters=25000
```

### Override Parameters

```bash
# Change learning rate
python train.py config/full_llama_1.36b.py --learning_rate=4e-4

# Increase batch size
python train.py config/full_llama_1.36b.py --batch_size=16 --gradient_accumulation_steps=8

# Enable FSDP for better multi-GPU scaling
torchrun --nproc_per_node=8 train.py config/full_llama_1.36b.py --use_fsdp=True
```

---

## Performance Characteristics

### Computational Cost (1.36B model)
- **FLOPs per token**: ~40-45 GFLOPs (forward pass)
  - Higher than GPT-2 due to SwiGLU (3 projections vs 2)
- **Attention/FFN ratio**: 0.50 (50% attention, 50% FFN)
- **Parameters**: 1.36B total, 1.21B non-embedding

### Training Speed (4x A100 80GB, DDP)
- **Tokens/second**: ~50,000-60,000
- **MFU**: 35-45% (model FLOPs utilization)
- **Memory per GPU**: ~25-30 GB (batch_size=8, block_size=2048)
- **Iterations/second**: ~0.3-0.4 (effective batch ~512 samples)

### Expected Loss (85B tokens - optimal)
- **Theoretical minimum**: 2.37 (from scaling law)
- **Practical target**: 2.4-2.5 (achievable)
- **6B tokens (test)**: ~4.0-4.5 (underfitting)

---

## Comparison with GPT-2

| Feature | GPT-2 124M | LLaMA 124M | LLaMA 1.36B |
|---------|-----------|------------|-------------|
| **Parameters** | 124M | 124M | 1,360M |
| **Position** | Learned | RoPE | RoPE |
| **Norm** | LayerNorm | RMSNorm | RMSNorm |
| **Norm Position** | Post | Pre | Pre |
| **FFN** | Standard 4× | SwiGLU 2.67× | SwiGLU 2.67× |
| **Weight Tying** | Yes | Yes | No |
| **Context** | 1024 | 1024 | 2048 |
| **FLOPs/token** | 28.5 GF | 35 GF | 42 GF |
| **Speed** | Faster | ~Same | ~2× slower |
| **Performance** | Baseline | +5-10% | +15-30% |

**Key Takeaways:**
1. LLaMA consistently outperforms GPT-2 at same parameter count
2. SwiGLU adds ~25% more FLOPs but improves quality
3. Pre-norm + RMSNorm enable stable training for deeper models
4. RoPE enables better length extrapolation

---

## Configuration Files

- **Comparison model**: `config/full_llama_124m.py` (same size as GPT-2 124M)
- **Production model**: `config/full_llama_1.36b.py` (your 1.36B model from scaling law)
- **Custom experiments**: `config/full_custom.py` (mix and match)

For direct GPT-2 vs LLaMA comparison at same scale:
- **`config/full_gpt2_1.36b.py`** - GPT-2 1.36B (matches LLaMA 1.36B)
- **`config/full_llama_1.36b.py`** - LLaMA 1.36B

For detailed parameter counting formulas, see: **`config/PARAMETER_FORMULAS.md`**

### Key Parameters to Adjust (1.36B model)

**For different hardware:**
```python
# 4x A100 80GB (comfortable)
batch_size = 8
gradient_accumulation_steps = 16
use_fsdp = False

# 3x RTX A4500 20GB (tight)
batch_size = 2
gradient_accumulation_steps = 40
use_zero1 = True  # Critical for memory!

# 8x B200 128GB (optimal)
batch_size = 16
gradient_accumulation_steps = 8
use_fsdp = True
```

**For different training budgets:**
```python
# Quick test (6B tokens)
dataset = 'slimpajama_6b'
max_iters = 2000
learning_rate = 3e-4

# Optimal training (85B tokens)
dataset = 'slimpajama_627b'
max_iters = 25000
learning_rate = 3e-4

# Extended training (200B+ tokens)
dataset = 'slimpajama_627b'
max_iters = 60000
learning_rate = 2e-4  # Lower for stability
```

---

## Scaling Law Context (1.36B Model)

From `info/llama_1.36e21_32kV.json`:

```json
{
  "theoretical_loss": 2.372087,
  "optimal_n_d": [1.294e+09, 8.472e+10],
  "validation_n_d": [1294159104, 62294983],
  "validation_loss": 4.7124714543356045
}
```

**Interpretation:**
- **Optimal**: 1.29B params × 84.7B tokens → loss 2.37
- **Current**: 1.29B params × 62M tokens → loss 4.71
- **Your model**: 1.36B params (close to optimal size!)

**Training Recommendation:**
- Train on ~85-100B tokens for loss ~2.4
- With SlimPajama-627B, stop around iteration 25,000
- Use early stopping when validation loss plateaus

---

## Common Pitfalls

### 1. Context Length Mismatch
```python
# Wrong: Training on 2048 but config says 1024
block_size = 1024  # ❌
# Fix:
block_size = 2048  # ✅ Match your JSON
```

### 2. Vocabulary Size Mismatch
```python
# If using LLaMA tokenizer (32K):
vocab_size = 50304  # ❌ Wrong tokenizer!
# Fix:
vocab_size = 32000  # ✅ Matches tokenizer
```

### 3. Learning Rate Too High
```python
# For 1.36B model:
learning_rate = 6e-4  # ❌ Too high (good for 124M)
# Fix:
learning_rate = 3e-4  # ✅ Scaled for size
```

### 4. Insufficient Warmup
```python
# For 1.36B model with large batch:
warmup_iters = 100  # ❌ Too short
# Fix:
warmup_iters = 2000  # ✅ ~2% of training
```

---

## Next Steps

1. **Quick validation**: Run `config/preset_quick_test.py` to verify setup
2. **Small test**: Train on 6B tokens to check for bugs
3. **Production**: Train on 627B tokens for optimal performance
4. **Monitor**: Watch MFU (should be 35-45%), loss trajectory, gradient norms

**Happy training! 🚀**

