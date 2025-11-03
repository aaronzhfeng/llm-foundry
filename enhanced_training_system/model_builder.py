"""
Modular Model Builder
=====================

Builds transformer models from configuration without hardcoded architecture.
Add new components to registries - this code adapts automatically!

Key Design:
- All architecture choices from ModelArchitectureConfig
- Components selected from registries
- Enhanced with MFU tracking, memory stats, gradient monitoring
- Compatible with DDP/FSDP/ZeRO-1

References:
- Component registries: model_components.py
- Configuration: model_config.py
- MFU formulas: Insu Jang (2022), Epoch AI (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

from model_config import ModelArchitectureConfig
from model_components import (
    build_norm, build_ffn, build_position_encoding,
    POSITION_ENCODING_REGISTRY, RoPEPositionEncoding
)


class ConfigurableAttention(nn.Module):
    """
    Fully configurable causal self-attention.
    Supports: SDPA (FlashAttention), manual attention, RoPE integration
    """
    
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head
        self.dropout = config.attention_dropout
        self.attention_backend = config.attention_backend
        self.position_encoding_type = config.position_encoding
        
        # Q, K, V projections (combined for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        # Check SDPA availability
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and config.attention_backend == 'sdpa':
            print("WARNING: PyTorch SDPA not available (requires PyTorch 2.0+), using manual attention")
            self.attention_backend = 'manual'
        
        # Register causal mask for manual attention
        if self.attention_backend == 'manual':
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
        
        # Create RoPE if needed (applied to Q/K in forward)
        if config.position_encoding == 'rope':
            self.rope = RoPEPositionEncoding(self.d_k, config.block_size, config.rope_theta)
        else:
            self.rope = None
    
    def forward(self, x, token_positions=None):
        """
        Args:
            x: [B, T, d_model]
            token_positions: [B, T] - position indices (required for RoPE)
        
        Returns:
            out: [B, T, d_model]
        """
        B, T, C = x.size()
        
        # QKV projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to multi-head format: [B, H, T, d_k]
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Apply RoPE if configured
        if self.rope is not None and token_positions is not None:
            q, k = self.rope.apply_to_qk(q, k, token_positions)
        
        # Attention computation
        if self.attention_backend == 'sdpa' and self.flash:
            # Use PyTorch SDPA (dispatches to FlashAttention when available)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention (fallback)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    """
    Configurable transformer block.
    Supports pre-norm/post-norm, any FFN/activation/norm type.
    """
    
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Normalization layers
        self.norm1 = build_norm(config.normalization, config.n_embd, config.norm_eps)
        self.norm2 = build_norm(config.normalization, config.n_embd, config.norm_eps)
        
        # Attention
        self.attn = ConfigurableAttention(config)
        
        # FFN
        self.ffn = build_ffn(
            config.ffn_type,
            config.n_embd,
            config.d_ff,
            config.bias,
            config.dropout,
            config.activation
        )
    
    def forward(self, x, token_positions=None):
        """
        Args:
            x: [B, T, d_model]
            token_positions: [B, T] - for RoPE
        
        Returns:
            x: [B, T, d_model]
        """
        if self.config.norm_position == 'pre':
            # Pre-norm (LLaMA style): Norm -> Sublayer -> Residual
            x = x + self.attn(self.norm1(x), token_positions)
            x = x + self.ffn(self.norm2(x))
        else:
            # Post-norm (GPT-2 style): Sublayer -> Residual -> Norm
            x = self.norm1(x + self.attn(x, token_positions))
            x = self.norm2(x + self.ffn(x))
        
        return x


class ConfigurableGPT(nn.Module):
    """
    Fully configurable GPT model.
    Architecture determined entirely by ModelArchitectureConfig!
    
    Enhanced with:
    - Detailed MFU calculation (academic formulas)
    - Memory tracking
    - Gradient monitoring
    - Compatible with DDP/FSDP/ZeRO-1
    """
    
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position encoding (if not RoPE - RoPE is applied in attention)
        if config.position_encoding == 'learned_absolute':
            self.pos_encoding = build_position_encoding(
                config.position_encoding,
                config.block_size,
                d_model=config.n_embd
            )
        else:
            self.pos_encoding = None
        
        # Embedding dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final normalization
        self.final_norm = build_norm(config.normalization, config.n_embd, config.norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        if config.weight_tying:
            self.token_embeddings.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report model info
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"Architecture: {config.get_architecture_name()}")
    
    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_encoding is not None:
            if hasattr(self.pos_encoding, 'wpe'):
                n_params -= self.pos_encoding.wpe.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None):
        """
        Forward pass compatible with training system.
        
        Args:
            idx: [B, T] - token indices
            targets: [B, T] - target tokens (optional, for training)
        
        Returns:
            logits: [B, T, vocab_size] (or [B, 1, vocab_size] for inference)
            loss: scalar (if targets provided)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence {T} > block_size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.token_embeddings(idx)  # [B, T, n_embd]
        
        # Position encoding
        token_positions = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        
        if self.config.position_encoding == 'learned_absolute':
            # Add learned position embeddings
            pos_emb = self.pos_encoding(token_positions)
            x = self.drop(tok_emb + pos_emb)
        else:
            # RoPE or no position encoding (RoPE applied in attention)
            x = self.drop(tok_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, token_positions)
        
        # Final normalization
        x = self.final_norm(x)
        
        # LM head and loss
        if targets is not None:
            # Training: compute logits for all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference: only compute last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    # ========================================================================
    # ENHANCED MFU CALCULATION (Architecture-aware)
    # ========================================================================
    
    def estimate_mfu_detailed(self, fwdbwd_per_iter, dt, device_type='cuda', num_gpus=1):
        """
        Detailed MFU calculation accounting for actual architecture.
        
        Adjusts FLOPs calculation based on:
        - FFN type (standard vs SwiGLU)
        - Position encoding (RoPE overhead)
        - Normalization (RMSNorm vs LayerNorm)
        
        Returns dict with complete MFU breakdown.
        """
        cfg = self.config
        H = cfg.n_embd
        L = cfg.n_layer
        a = cfg.n_head
        S = cfg.block_size
        D_ff = cfg.d_ff
        
        # ===== ATTENTION FLOPs (same for all architectures) =====
        # QKV projections: 3 × (2 × S × H × H) = 6SH²
        attention_qkv_flops = 6 * S * H * H
        
        # Attention scores: a × S × S × H (QK^T per head)
        attention_scores_flops = a * S * S * H
        
        # Attention output: a × S × S × H (softmax(QK^T) @ V per head)
        attention_output_flops = a * S * S * H
        
        # Output projection: 2 × S × H × H
        attention_proj_flops = 2 * S * H * H
        
        attention_flops = (attention_qkv_flops + attention_scores_flops + 
                          attention_output_flops + attention_proj_flops)
        
        # ===== FFN FLOPs (depends on type) =====
        if cfg.ffn_type == 'swiglu':
            # SwiGLU: 3 linear layers (gate, value, output)
            # gate: 2×S×H×D_ff, value: 2×S×H×D_ff, output: 2×S×D_ff×H
            ffn_flops = 3 * (2 * S * H * D_ff)
        else:
            # Standard: 2 linear layers (up, down)
            # up: 2×S×H×D_ff, down: 2×S×D_ff×H
            ffn_flops = 2 * (2 * S * H * D_ff)
        
        # ===== POSITION ENCODING FLOPs =====
        rope_flops = 0
        if cfg.position_encoding == 'rope':
            # RoPE rotation operations (approximate)
            # Per head: rotation of d_k elements for Q and K
            rope_flops = 2 * a * S * (H // a) * 2  # 2 for Q and K
        
        # ===== NORMALIZATION FLOPs =====
        if cfg.normalization == 'rmsnorm':
            # RMSNorm: variance computation + rsqrt + multiply
            norm_flops_per_layer = 1.5 * S * H
        else:
            # LayerNorm: mean + variance + normalize
            norm_flops_per_layer = 2 * S * H
        
        # 2 norms per block + 1 final norm
        total_norm_flops = (2 * L + 1) * norm_flops_per_layer
        
        # ===== TOTAL FORWARD PASS FLOPs =====
        flops_per_layer = attention_flops + ffn_flops + rope_flops + 2 * norm_flops_per_layer
        total_forward_flops = L * flops_per_layer + norm_flops_per_layer  # +1 for final norm
        
        # Per-token FLOPs
        forward_flops_per_token = total_forward_flops / S
        
        # ===== TRAINING FLOPs (Forward + Backward) =====
        # Backward ≈ 2× forward (from Epoch AI research)
        # Reference: https://epoch.ai/blog/backward-forward-FLOP-ratio
        training_flops_per_token = 3 * forward_flops_per_token
        
        # Total FLOPs for this iteration
        tokens_per_iter = S * fwdbwd_per_iter
        flops_per_iter = training_flops_per_token * tokens_per_iter
        
        # Achieved throughput
        flops_achieved = flops_per_iter / dt
        tokens_per_sec = tokens_per_iter / dt
        
        # ===== HARDWARE SPECS (with B200) =====
        hardware_specs = {
            'cuda': {
                'B200': {'bf16': 4500e12, 'fp16': 4500e12, 'fp32': 90e12},
                'H200': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12},
                'H100': {'bf16': 989e12, 'fp16': 989e12, 'fp32': 67e12},
                'A100': {'bf16': 312e12, 'fp16': 312e12, 'fp32': 19.5e12},
                'V100': {'bf16': 125e12, 'fp16': 125e12, 'fp32': 15.7e12},
            }
        }
        
        # Auto-detect GPU
        gpu_name = 'A100'  # Default
        if torch.cuda.is_available():
            gpu_name_full = torch.cuda.get_device_name(0)
            for name in ['B200', 'H200', 'H100', 'A100', 'V100']:
                if name in gpu_name_full:
                    gpu_name = name
                    break
        
        # Get precision
        dtype = str(self.token_embeddings.weight.dtype).split('.')[-1]
        precision_key = 'bf16' if 'bfloat16' in dtype else 'fp16' if 'float16' in dtype else 'fp32'
        
        hardware_peak_flops_per_gpu = hardware_specs.get(device_type, {}).get(gpu_name, {}).get(precision_key, 312e12)
        hardware_peak_flops = hardware_peak_flops_per_gpu * num_gpus
        
        # ===== MFU CALCULATION =====
        mfu = flops_achieved / hardware_peak_flops
        
        # Return detailed breakdown
        return {
            'mfu': mfu,
            'mfu_percent': mfu * 100,
            'flops_achieved': flops_achieved,
            'flops_per_token': training_flops_per_token,
            'tokens_per_sec': tokens_per_sec,
            'hardware_peak_flops': hardware_peak_flops,
            'hardware_peak_tflops': hardware_peak_flops / 1e12,
            'achieved_tflops': flops_achieved / 1e12,
            'gpu_name': gpu_name,
            'precision': precision_key,
            'num_gpus': num_gpus,
            'attention_flops_per_layer': attention_flops,
            'ffn_flops_per_layer': ffn_flops,
            'attention_to_ffn_ratio': attention_flops / ffn_flops if ffn_flops > 0 else 0,
            'architecture': cfg.get_architecture_name(),
        }
    
    # ========================================================================
    # MEMORY & GRADIENT TRACKING
    # ========================================================================
    
    def get_memory_stats(self):
        """Get detailed memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
        }
    
    def get_gradient_stats(self):
        """Get gradient statistics for monitoring training health"""
        grad_norms = []
        grad_values = []
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                # Sample gradient values (don't collect all to avoid memory issues)
                grad_values.extend(param.grad.flatten().cpu().numpy().tolist()[:1000])
        
        if not grad_values:
            return {}
        
        import numpy as np
        grad_values = np.array(grad_values)
        
        return {
            'global_norm': np.sqrt(sum(n**2 for n in grad_norms)),
            'mean_layer_norm': np.mean(grad_norms) if grad_norms else 0,
            'max_layer_norm': np.max(grad_norms) if grad_norms else 0,
            'min_layer_norm': np.min(grad_norms) if grad_norms else 0,
            'grad_mean': float(np.mean(grad_values)),
            'grad_std': float(np.std(grad_values)),
            'grad_min': float(np.min(grad_values)),
            'grad_max': float(np.max(grad_values)),
        }
    
    # ========================================================================
    # OPTIMIZER CONFIGURATION
    # ========================================================================
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay groups.
        All 2D parameters (weights) get decay, 1D parameters (biases, norms) don't.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate parameters by dimension for weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer
    
    # ========================================================================
    # TEXT GENERATION
    # ========================================================================
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Text generation (compatible with nanoGPT).
        
        Args:
            idx: [B, T] - conditioning sequence
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (None = no filtering)
        
        Returns:
            idx: [B, T + max_new_tokens] - completed sequence
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

