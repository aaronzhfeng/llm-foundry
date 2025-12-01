"""
Triton kernels for optimized operations in GPT training.
"""

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from .layer_norm import triton_layer_norm, TritonLayerNorm

__all__ = ['triton_layer_norm', 'TritonLayerNorm', 'TRITON_AVAILABLE']

