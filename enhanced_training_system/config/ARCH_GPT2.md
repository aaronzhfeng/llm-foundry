# GPT-2 Architecture Configuration

## Quick Reference

### Architecture Components
- **Preset**: `gpt2`
- **Normalization**: LayerNorm (no bias)
- **Position Encoding**: Learned Absolute
- **Activation**: GELU
- **FFN Type**: Standard (4× expansion)
- **Norm Position**: Post-norm
- **Attention Backend**: SDPA (FlashAttention-1)
- **Weight Tying**: Yes
- **Bias**: False (optimized version)
- **Dropout**: 0.0

### Model Dimensions (GPT-2 124M)
- **Layers**: 12
- **Hidden size**: 768
- **Attention heads**: 12
- **Head dimension**: 64 (768 / 12)
- **FFN dimension**: 3072 (4× expansion)
- **Context length**: 1024
- **Vocabulary**: 50304 (50257 rounded up for efficiency)

### Total Parameters
- **124.44M** parameters
  - Token embeddings: 38.6M (shared with output)
  - Position embeddings: 0.8M
  - Attention layers: 28.3M (12 layers)
  - FFN layers: 56.6M (12 layers)
  - Normalization: ~0.1M

---

## Reference & Explanations

### Architecture Origins

**Primary Source:**
> **Language Models are Unsupervised Multitask Learners**  
> Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI)  
> 2019  
> https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

**Implementation Base:**
> **nanoGPT** - Karpathy's minimal GPT-2 implementation  
> https://github.com/karpathy/nanoGPT

---

## Component Details

### 1. Learned Absolute Position Embeddings

**What it is:**
- Separate learned embedding table for each position (0 to 1023)
- Added to token embeddings before transformer layers

**Formula:**
```python
pos_emb = nn.Embedding(1024, 768)
x = token_emb + pos_emb(position_ids)
```

**Pros:**
- ✅ Simple, proven approach
- ✅ Fast (just lookup + add)

**Cons:**
- ❌ Fixed maximum length (1024 tokens)
- ❌ Poor extrapolation to longer sequences
- ❌ Doesn't encode relative positions explicitly

---

### 2. LayerNorm (No Bias)

**What it is:**
- Normalizes activations by mean and variance
- Optimized version removes bias parameter

**Formula:**
```python
mean = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True)
x_norm = (x - mean) / sqrt(var + eps)
output = gamma * x_norm  # Only scale, no bias
```

**Settings:**
- `eps = 1e-5` (numerical stability)
- `bias = False` (optimization)

---

### 3. GELU Activation

**What it is:**
- Gaussian Error Linear Unit
- Smooth approximation of ReLU

**Formula:**
```python
GELU(x) = x * Φ(x)
# where Φ(x) is cumulative distribution function of standard normal
```

**Why:**
- Better than ReLU for language modeling
- Smooth gradients (no dead neurons)

---

### 4. Standard FFN (4× Expansion)

**What it is:**
- Two-layer MLP with GELU activation
- Expands dimension by 4×, then projects back

**Structure:**
```python
FFN(x):
    h = GELU(Linear_1(x))    # 768 → 3072
    output = Linear_2(h)      # 3072 → 768
```

**Parameters per layer:**
- 2 × 768 × 3072 = 4.7M parameters

---

### 5. Post-Norm Architecture

**What it is:**
- Apply normalization AFTER residual addition

**Structure:**
```python
x = Norm(x + Attention(x))
x = Norm(x + FFN(x))
```

**Note:**
- Original Transformer design
- Can be less stable for very deep networks (>24 layers)

---

### 6. Weight Tying

**What it is:**
- Input token embeddings and output projection share weights
- Reduces parameters by ~38.6M (vocab_size × hidden_dim)

**Implementation:**
```python
token_emb = nn.Embedding(50304, 768)
# Output layer reuses embedding weights
logits = F.linear(hidden_states, token_emb.weight)
```

**Why:**
- Saves memory
- Works well for smaller models (<1B params)

---

## Usage Examples

### Basic Training

```bash
# Train GPT-2 124M from scratch
python train.py config/full_gpt2_124m.py
```

### Multi-GPU Training

```bash
# DDP on 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py config/full_gpt2_124m.py

# With FSDP (for larger models)
torchrun --standalone --nproc_per_node=8 train.py config/full_gpt2_124m.py --use_fsdp=True
```

### Override Parameters

```bash
# Change learning rate
python train.py config/full_gpt2_124m.py --learning_rate=3e-4

# Use different dataset
python train.py config/full_gpt2_124m.py --dataset=openwebtext

# Quick test with smaller model
python train.py config/preset_quick_test.py
```

---

## Performance Characteristics

### Computational Cost
- **FLOPs per token**: ~28.5 GFLOPs (forward pass)
- **Attention/FFN ratio**: 0.33 (33% attention, 67% FFN)
- **Parameters**: 124.44M total, 123.59M non-embedding

### Training Speed (Single A100 80GB)
- **Tokens/second**: ~18,000 (with compile=True)
- **MFU**: 38-48% (model FLOPs utilization)
- **Memory**: ~5-8 GB (batch_size=12, block_size=1024)

### Expected Loss (OpenWebText, 10B tokens)
- **Final validation loss**: ~3.0-3.2
- **Perplexity**: ~20-25

---

## Comparison with Other Architectures

| Feature | GPT-2 | LLaMA | Impact |
|---------|-------|-------|--------|
| Position | Learned | RoPE | LLaMA better for long sequences |
| Norm | LayerNorm | RMSNorm | LLaMA ~15% faster |
| Norm Position | Post | Pre | LLaMA more stable for deep nets |
| FFN | Standard (4×) | SwiGLU (2.67×) | LLaMA better performance |
| Weight Tying | Yes | No | GPT-2 saves memory |

**When to use GPT-2:**
- ✅ Smaller models (<500M params)
- ✅ Educational/reference implementation
- ✅ Standard benchmarking
- ✅ Memory-constrained environments

**When to use LLaMA:**
- ✅ Larger models (>1B params)
- ✅ Better performance per parameter
- ✅ Longer context windows
- ✅ Modern production systems

---

## Configuration Files

- **`config/full_gpt2_124m.py`** - GPT-2 124M (baseline comparison model)
- **`config/full_gpt2_1.36b.py`** - GPT-2 1.36B (matches LLaMA 1.36B for direct comparison)

For detailed parameter counting formulas, see: **`config/PARAMETER_FORMULAS.md`**

Key parameters to adjust:
- `learning_rate`: 6e-4 (default), increase for smaller models
- `batch_size`: Adjust based on GPU memory
- `gradient_accumulation_steps`: Simulate larger batches
- `max_iters`: Depends on dataset size (600k for full OpenWebText)
- `compile`: Set to True for ~30% speedup (requires PyTorch 2.0+)

