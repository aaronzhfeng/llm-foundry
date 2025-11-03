"""
Model Configuration System
===========================

Comprehensive configuration for model architecture and system settings.
All design choices in one place - easily create new architectures!

Usage:
    # Use preset
    config = get_preset_config('llama')
    
    # Or customize
    config = ModelArchitectureConfig(
        normalization='rmsnorm',
        position_encoding='rope',
        ffn_type='swiglu',
        ...
    )
    
    # Save/load
    config.save_json('my_config.json')
    config = ModelArchitectureConfig.load_json('my_config.json')
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any
import json


@dataclass
class ModelArchitectureConfig:
    """
    Complete architectural configuration.
    All design choices for the transformer model.
    """
    
    # ========== MODEL SIZE ==========
    block_size: int = 1024              # Sequence length / context window
    vocab_size: int = 50304             # Vocabulary size (GPT-2: 50257 -> 50304 for efficiency)
    n_layer: int = 12                   # Number of transformer layers
    n_head: int = 12                    # Number of attention heads
    n_embd: int = 768                   # Embedding/hidden dimension
    
    # ========== NORMALIZATION ==========
    normalization: Literal['layernorm', 'layernorm_nobias', 'rmsnorm'] = 'layernorm_nobias'
    norm_eps: float = 1e-5              # Epsilon for numerical stability
    
    # ========== ACTIVATION ==========
    activation: Literal['gelu', 'silu', 'relu', 'leaky_relu'] = 'gelu'
    
    # ========== ATTENTION ==========
    attention_backend: Literal['sdpa', 'manual'] = 'sdpa'  # SDPA uses FlashAttention when available
    
    # ========== POSITION ENCODING ==========
    position_encoding: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute'
    rope_theta: float = 10000.0         # RoPE base frequency (used if position_encoding='rope')
    
    # ========== ARCHITECTURE STYLE ==========
    norm_position: Literal['pre', 'post'] = 'post'  # Pre-norm (LLaMA) or post-norm (GPT-2)
    ffn_type: Literal['standard', 'swiglu'] = 'standard'
    
    # ========== COMPONENT OPTIONS ==========
    bias: bool = False                  # Use bias in Linear layers
    dropout: float = 0.0                # Dropout rate (0.0 for pretraining)
    weight_tying: bool = True           # Tie token embeddings with lm_head weights
    
    # ========== FFN CONFIGURATION ==========
    d_ff: Optional[int] = None          # FFN hidden dim (None = auto-calculate)
    ffn_expansion_ratio: Optional[float] = None  # Alternative to d_ff (e.g., 4.0 or 8/3)
    
    # ========== ADVANCED OPTIONS ==========
    attention_dropout: Optional[float] = None    # Separate attention dropout (None = use dropout)
    residual_dropout: Optional[float] = None     # Separate residual dropout (None = use dropout)
    
    def __post_init__(self):
        """Auto-configure and validate after initialization"""
        
        # Auto-calculate d_ff if not specified
        if self.d_ff is None:
            if self.ffn_expansion_ratio is not None:
                self.d_ff = int(self.n_embd * self.ffn_expansion_ratio)
            elif self.ffn_type == 'swiglu':
                # LLaMA uses 8/3 expansion for SwiGLU
                self.d_ff = int(8 * self.n_embd / 3)
            else:
                # GPT-2 uses 4x expansion for standard FFN
                self.d_ff = 4 * self.n_embd
        
        # Auto-set dropout variants
        if self.attention_dropout is None:
            self.attention_dropout = self.dropout
        if self.residual_dropout is None:
            self.residual_dropout = self.dropout
        
        # Validation
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.norm_position in ['pre', 'post'], "norm_position must be 'pre' or 'post'"
        assert self.block_size > 0, "block_size must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        
        # Informational warnings about unusual combinations
        if self.position_encoding == 'learned_absolute' and self.ffn_type == 'swiglu':
            print("INFO: Using learned positions (GPT-2) with SwiGLU (LLaMA). Hybrid architecture.")
        
        if self.normalization == 'layernorm' and self.ffn_type == 'swiglu':
            print("INFO: Using LayerNorm with SwiGLU. LLaMA typically uses RMSNorm.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to: {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Load configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def load_json(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        print(f"Configuration loaded from: {path}")
        return cls.from_dict(config_dict)
    
    def get_architecture_name(self) -> str:
        """
        Generate descriptive architecture name.
        Example: '12L-12H-768D-RoPE-RMS-SwiGLU-PreNorm'
        """
        components = []
        
        # Size: Layers-Heads-Dimension
        components.append(f"{self.n_layer}L-{self.n_head}H-{self.n_embd}D")
        
        # Position encoding
        if self.position_encoding == 'rope':
            components.append("RoPE")
        elif self.position_encoding == 'learned_absolute':
            components.append("AbsPos")
        elif self.position_encoding == 'none':
            components.append("NoPos")
        
        # Normalization
        if self.normalization == 'rmsnorm':
            components.append("RMS")
        elif self.normalization == 'layernorm':
            components.append("LN")
        elif self.normalization == 'layernorm_nobias':
            components.append("LN-NB")
        
        # FFN/Activation
        if self.ffn_type == 'swiglu':
            components.append("SwiGLU")
        else:
            components.append(self.activation.upper())
        
        # Norm position
        if self.norm_position == 'pre':
            components.append("PreNorm")
        else:
            components.append("PostNorm")
        
        return "-".join(components)
    
    def get_architecture_summary(self) -> Dict[str, str]:
        """
        Return human-readable architecture summary.
        Used in startup report.
        """
        return {
            'Architecture Name': self.get_architecture_name(),
            'Model Size': f"{self.n_layer}L × {self.n_head}H × {self.n_embd}D",
            'Parameters (est)': f"~{self._estimate_params()/1e6:.1f}M",
            'Normalization': self.normalization,
            'Activation': self.activation if self.ffn_type != 'swiglu' else 'swiglu (built-in SiLU)',
            'Position Encoding': self.position_encoding + (f' (θ={self.rope_theta})' if self.position_encoding == 'rope' else ''),
            'Attention Backend': self.attention_backend,
            'Norm Position': self.norm_position,
            'FFN Type': self.ffn_type,
            'FFN Expansion': f'{self.d_ff}/{self.n_embd} = {self.d_ff/self.n_embd:.2f}x',
            'Bias in Linear': 'Yes' if self.bias else 'No',
            'Weight Tying': 'Yes' if self.weight_tying else 'No',
            'Dropout': f'{self.dropout:.3f}',
        }
    
    def _estimate_params(self):
        """Rough parameter estimate for reporting"""
        # Token embeddings
        params = self.vocab_size * self.n_embd
        
        # Position embeddings (if learned)
        if self.position_encoding == 'learned_absolute':
            params += self.block_size * self.n_embd
        
        # Per-layer parameters
        # Attention: Q, K, V, O projections (4 * n_embd * n_embd)
        attn_params = 4 * self.n_embd * self.n_embd
        
        # FFN parameters
        if self.ffn_type == 'swiglu':
            # 3 projections: gate, value, output
            ffn_params = 3 * self.n_embd * self.d_ff
        else:
            # 2 projections: up, down
            ffn_params = 2 * self.n_embd * self.d_ff
        
        # Normalization: 2 per layer + 1 final (weight only, ~negligible)
        norm_params = 3 * self.n_embd
        
        layer_params = attn_params + ffn_params + norm_params
        params += self.n_layer * layer_params
        
        # Output head (if not weight-tied)
        if not self.weight_tying:
            params += self.vocab_size * self.n_embd
        
        # Final norm
        params += self.n_embd
        
        return params


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_gpt2_config() -> ModelArchitectureConfig:
    """
    GPT-2 124M standard architecture.
    
    Features:
    - Learned absolute position embeddings
    - LayerNorm (no bias)
    - GELU activation
    - Standard FFN (4x expansion)
    - Post-norm
    - Weight tying
    
    Reference: https://github.com/openai/gpt-2
    """
    return ModelArchitectureConfig(
        # Size (GPT-2 124M)
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        
        # Architecture
        normalization='layernorm_nobias',
        activation='gelu',
        position_encoding='learned_absolute',
        norm_position='post',
        ffn_type='standard',
        attention_backend='sdpa',
        
        # Options
        bias=False,
        weight_tying=True,
        dropout=0.0,
    )


def get_llama_style_config() -> ModelArchitectureConfig:
    """
    LLaMA-style architecture.
    
    Features:
    - RoPE (rotary position embeddings)
    - RMSNorm
    - SwiGLU activation (8/3 expansion)
    - Pre-norm
    - No weight tying
    
    Reference: LLaMA paper (Touvron et al., 2023)
    https://arxiv.org/abs/2302.13971
    """
    return ModelArchitectureConfig(
        # Size (same as GPT-2 124M for comparison)
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        
        # Architecture (LLaMA style)
        normalization='rmsnorm',
        activation='gelu',  # Not used with SwiGLU
        position_encoding='rope',
        norm_position='pre',
        ffn_type='swiglu',
        attention_backend='sdpa',
        
        # Options
        bias=False,
        weight_tying=False,  # LLaMA doesn't tie weights
        dropout=0.0,
        rope_theta=10000.0,
    )


def get_hybrid_config() -> ModelArchitectureConfig:
    """
    Experimental hybrid: RoPE + LayerNorm + GELU.
    
    Combines:
    - RoPE from LLaMA (better position encoding)
    - LayerNorm from GPT-2 (simpler)
    - Standard FFN with GELU
    - Pre-norm for training stability
    """
    return ModelArchitectureConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        
        normalization='layernorm_nobias',
        activation='gelu',
        position_encoding='rope',
        norm_position='pre',
        ffn_type='standard',
        attention_backend='sdpa',
        
        bias=False,
        weight_tying=True,
        dropout=0.0,
    )


def get_team_config() -> ModelArchitectureConfig:
    """
    Team's model_v1 architecture.
    
    Features:
    - RoPE
    - RMSNorm  
    - SwiGLU
    - Pre-norm
    - FlashAttention-2 support
    """
    return ModelArchitectureConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        
        normalization='rmsnorm',
        activation='gelu',  # Not used with SwiGLU
        position_encoding='rope',
        norm_position='pre',
        ffn_type='swiglu',
        attention_backend='sdpa',
        
        bias=False,
        weight_tying=False,
        dropout=0.0,
    )


PRESET_CONFIGS = {
    'gpt2': get_gpt2_config,
    'llama': get_llama_style_config,
    'hybrid': get_hybrid_config,
    'team': get_team_config,
}


def get_preset_config(name: str) -> ModelArchitectureConfig:
    """
    Get a preset configuration by name.
    
    Args:
        name: Preset name ('gpt2', 'llama', 'hybrid', 'team')
    
    Returns:
        ModelArchitectureConfig instance
    
    Available presets:
    - 'gpt2': Standard GPT-2 124M architecture
    - 'llama': LLaMA-style (RoPE + RMSNorm + SwiGLU)
    - 'hybrid': Experimental (RoPE + LayerNorm + GELU)
    - 'team': Team's model_v1 architecture
    """
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]()


def list_presets():
    """List all available preset configurations"""
    print("Available preset configurations:")
    for name in PRESET_CONFIGS.keys():
        config = get_preset_config(name)
        print(f"\n  {name}:")
        print(f"    Architecture: {config.get_architecture_name()}")
        print(f"    Norm: {config.normalization}, Pos: {config.position_encoding}, FFN: {config.ffn_type}")


if __name__ == '__main__':
    # Demo: Show all presets
    list_presets()
    
    # Demo: Create and save custom config
    custom = ModelArchitectureConfig(
        normalization='rmsnorm',
        position_encoding='rope',
        ffn_type='standard',
        norm_position='pre',
    )
    print(f"\nCustom config: {custom.get_architecture_name()}")
    print(f"Summary: {json.dumps(custom.get_architecture_summary(), indent=2)}")

