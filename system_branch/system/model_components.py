"""
Modular Model Components Registry
==================================

All architectural components in one place. Easy to extend!

To add a new component type:
1. Create the component class
2. Add it to the appropriate registry
3. Use it in config files - no other code changes needed!

Component Categories:
1. Normalization (LayerNorm, RMSNorm)
2. Position Encoding (Learned, RoPE)
3. FFN (Standard, SwiGLU)
4. Attention (Configurable with multiple backends)

References:
- GPT-2: https://github.com/openai/gpt-2
- LLaMA: https://arxiv.org/abs/2302.13971
- RoPE: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. NORMALIZATION COMPONENTS
# ============================================================================

class LayerNormWithBias(nn.Module):
    """Standard LayerNorm with bias"""
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
        self.eps = eps
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class LayerNormNoBias(nn.Module):
    """LayerNorm without bias (GPT-2 default)"""
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, self.eps)


class RMSNorm(nn.Module):
    """
    RMSNorm (LLaMA style) - faster, no mean centering.
    RMS(x) = sqrt(mean(x²))
    RMSNorm(x) = x / RMS(x) * weight
    """
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
    
    def forward(self, x):
        # Manual RMSNorm implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


NORM_REGISTRY = {
    'layernorm': LayerNormWithBias,
    'layernorm_nobias': LayerNormNoBias,
    'rmsnorm': RMSNorm,
}


# ============================================================================
# 2. POSITION ENCODING COMPONENTS
# ============================================================================

class LearnedAbsolutePositionEncoding(nn.Module):
    """Learned absolute position embeddings (GPT-2 style)"""
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
    
    def forward(self, token_positions):
        """
        Args:
            token_positions: [B, L] - token position indices (0, 1, 2, ...)
        Returns:
            pos_emb: [B, L, d_model] - position embeddings to add to token embeddings
        """
        return self.wpe(token_positions)
    
    def apply_to_qk(self, q, k, token_positions):
        """No-op for learned absolute (applied as additive embedding)"""
        return q, k


class RoPEPositionEncoding(nn.Module):
    """
    Rotary Position Encoding (RoPE) - LLaMA style.
    Applies rotation to Q and K based on position.
    
    Reference: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    """
    def __init__(self, d_k, max_seq_len, theta=10000.0):
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be even for RoPE, got {d_k}"
        
        half_dim = d_k // 2
        # Compute frequency bands: θ^(-2i/d) for i = 0, 1, ..., d/2-1
        freq_exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        inv_freq = theta ** (-freq_exponents)
        
        # Precompute sin/cos for all positions
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)  # [max_seq_len, half_dim]
        
        self.register_buffer("cos_cached", torch.cos(angles))
        self.register_buffer("sin_cached", torch.sin(angles))
        self.d_k = d_k
        self.max_seq_len = max_seq_len
    
    def forward(self, token_positions):
        """Returns None for RoPE (applied directly to Q/K in attention)"""
        return None
    
    def apply_to_qk(self, q, k, token_positions):
        """
        Apply RoPE rotation to Q and K tensors.
        
        Args:
            q, k: [B, H, L, d_k] - query/key after head split
            token_positions: [B, L] - position indices
        
        Returns:
            q_rot, k_rot: [B, H, L, d_k] - rotated queries and keys
        """
        B, H, L, D = q.shape
        
        # Get cos/sin for these positions
        cos = self.cos_cached[token_positions].to(dtype=q.dtype)  # [B, L, d_k/2]
        sin = self.sin_cached[token_positions].to(dtype=q.dtype)
        
        # Reshape to broadcast over heads: [B, 1, L, d_k/2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Apply rotation to Q and K
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        
        return q_rot, k_rot
    
    def _rotate(self, x, cos, sin):
        """
        Apply RoPE rotation: [cos θ, -sin θ; sin θ, cos θ] to pairs
        
        Args:
            x: [B, H, L, d_k]
            cos, sin: [B, 1, L, d_k/2]
        """
        # Split into even and odd indices
        x_even = x[..., 0::2]  # [B, H, L, d_k/2]
        x_odd = x[..., 1::2]   # [B, H, L, d_k/2]
        
        # Rotation: [cos*even - sin*odd, sin*even + cos*odd]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_odd * cos + x_even * sin
        
        # Interleave back: [even0, odd0, even1, odd1, ...]
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


class NoPositionEncoding(nn.Module):
    """No position encoding (for experiments)"""
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, token_positions):
        return None
    
    def apply_to_qk(self, q, k, token_positions):
        return q, k


POSITION_ENCODING_REGISTRY = {
    'learned_absolute': LearnedAbsolutePositionEncoding,
    'rope': RoPEPositionEncoding,
    'none': NoPositionEncoding,
}


# ============================================================================
# 3. FFN COMPONENTS
# ============================================================================

class StandardFFN(nn.Module):
    """
    Standard FFN with configurable activation (GPT-2 style).
    Two linear layers with activation in between: x -> fc1 -> act -> fc2
    """
    def __init__(self, d_model, d_ff, bias=True, dropout=0.0, activation='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        
        # Select activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'gelu', 'silu', 'relu', or 'leaky_relu'")
        
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN (LLaMA style): y = W_out(swish(xW_gate) ⊙ xW_value)
    
    Uses 3 linear layers instead of 2:
    - gate projection: d_model -> d_ff
    - value projection: d_model -> d_ff  
    - output projection: d_ff -> d_model
    
    Reference: GLU Variants Improve Transformer (Shazeer, 2020)
    https://arxiv.org/abs/2002.05202
    """
    def __init__(self, d_model, d_ff, bias=False, dropout=0.0, **kwargs):
        super().__init__()
        # SwiGLU uses 3 projections (gate, value, output)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_value = nn.Linear(d_model, d_ff, bias=bias)
        self.w_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # swish activation
        value = self.w_value(x)
        x = self.w_out(gate * value)   # element-wise product
        return self.dropout(x)


FFN_REGISTRY = {
    'standard': StandardFFN,
    'swiglu': SwiGLUFFN,
}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def build_norm(norm_type: str, ndim: int, eps: float = 1e-5):
    """
    Factory function to build normalization layer.
    
    Args:
        norm_type: 'layernorm', 'layernorm_nobias', or 'rmsnorm'
        ndim: Dimension to normalize
        eps: Epsilon for numerical stability
    
    Returns:
        Normalization module
    """
    if norm_type not in NORM_REGISTRY:
        raise ValueError(f"Unknown norm type: {norm_type}. Available: {list(NORM_REGISTRY.keys())}")
    return NORM_REGISTRY[norm_type](ndim, eps=eps)


def build_ffn(ffn_type: str, d_model: int, d_ff: int, bias: bool, 
              dropout: float, activation: str = 'gelu'):
    """
    Factory function to build FFN layer.
    
    Args:
        ffn_type: 'standard' or 'swiglu'
        d_model: Model dimension
        d_ff: FFN hidden dimension
        bias: Use bias in linear layers
        dropout: Dropout rate
        activation: Activation function (for standard FFN)
    
    Returns:
        FFN module
    """
    if ffn_type not in FFN_REGISTRY:
        raise ValueError(f"Unknown FFN type: {ffn_type}. Available: {list(FFN_REGISTRY.keys())}")
    return FFN_REGISTRY[ffn_type](d_model, d_ff, bias, dropout, activation=activation)


def build_position_encoding(pos_type: str, max_seq_len: int, d_model: int = None, 
                            d_k: int = None, theta: float = 10000.0):
    """
    Factory function to build position encoding.
    
    Args:
        pos_type: 'learned_absolute', 'rope', or 'none'
        max_seq_len: Maximum sequence length
        d_model: Model dimension (for learned_absolute)
        d_k: Head dimension (for rope)
        theta: RoPE base frequency
    
    Returns:
        Position encoding module
    """
    if pos_type not in POSITION_ENCODING_REGISTRY:
        raise ValueError(f"Unknown position encoding: {pos_type}. Available: {list(POSITION_ENCODING_REGISTRY.keys())}")
    
    if pos_type == 'learned_absolute':
        assert d_model is not None, "d_model required for learned_absolute"
        return POSITION_ENCODING_REGISTRY[pos_type](max_seq_len, d_model)
    elif pos_type == 'rope':
        assert d_k is not None, "d_k (head_dim) required for rope"
        return POSITION_ENCODING_REGISTRY[pos_type](d_k, max_seq_len, theta)
    else:
        return POSITION_ENCODING_REGISTRY[pos_type]()

