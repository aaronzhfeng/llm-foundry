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

# Try importing FlashAttention-2 and FlashAttention-3
try:
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_3_func
    import flash_attn
    from packaging import version
    HAS_FLASH_ATTN_2 = True
    HAS_FLASH_ATTN_3 = version.parse(flash_attn.__version__) >= version.parse("2.5.0")
except ImportError:
    HAS_FLASH_ATTN_2 = False
    HAS_FLASH_ATTN_3 = False


class ConfigurableAttention(nn.Module):
    """
    Fully configurable causal self-attention.
    Supports: SDPA (FlashAttention), manual attention, RoPE integration, GQA
    
    Features:
    - Multi-Head Attention (MHA): num_key_value_heads == n_head
    - Grouped Query Attention (GQA): num_key_value_heads < n_head (LLaMA 3)
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
        
        # GQA configuration
        self.n_kv_head = config.num_key_value_heads
        self.use_gqa = self.n_kv_head < self.n_head
        self.n_rep = self.n_head // self.n_kv_head  # How many Q heads per KV head
        
        if self.use_gqa:
            # Grouped Query Attention: separate Q and KV projections
            self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_kv = nn.Linear(config.n_embd, 2 * self.n_kv_head * self.d_k, bias=config.bias)
        else:
            # Multi-Head Attention: combined QKV projection
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        # Check backend availability
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.has_flash_attn_2 = HAS_FLASH_ATTN_2
        self.has_flash_attn_3 = HAS_FLASH_ATTN_3
        
        # Validate attention backend choice with fallback chain
        if config.attention_backend == 'flash_attn_3':
            if self.has_flash_attn_3:
                self.attention_backend = 'flash_attn_3'
            elif self.has_flash_attn_2:
                print("WARNING: flash_attn_3 requested but flash-attn >= 2.5.0 not available. Falling back to flash_attn_2")
                self.attention_backend = 'flash_attn_2'
            elif self.flash:
                print("WARNING: flash_attn_3 not available. Falling back to sdpa")
                self.attention_backend = 'sdpa'
            else:
                print("WARNING: No optimized attention available. Falling back to manual")
                self.attention_backend = 'manual'
        elif config.attention_backend == 'flash_attn_2':
            if self.has_flash_attn_2:
                self.attention_backend = 'flash_attn_2'
            elif self.flash:
                print("WARNING: flash_attn_2 not available. Falling back to sdpa")
                self.attention_backend = 'sdpa'
            else:
                print("WARNING: flash_attn_2 not available. Falling back to manual")
                self.attention_backend = 'manual'
        elif config.attention_backend == 'sdpa':
            if self.flash:
                self.attention_backend = 'sdpa'
            else:
                print("WARNING: sdpa requested but PyTorch < 2.0. Falling back to manual.")
                self.attention_backend = 'manual'
        else:
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
        
        if self.use_gqa:
            # Grouped Query Attention: separate Q and KV projections
            q = self.c_q(x)  # [B, T, n_embd]
            kv = self.c_kv(x)  # [B, T, 2 * n_kv_head * d_k]
            
            # Reshape Q: [B, T, n_head, d_k] -> [B, n_head, T, d_k]
            q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
            
            # Reshape KV: split into K and V
            k, v = kv.split(self.n_kv_head * self.d_k, dim=2)
            k = k.view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)  # [B, n_kv_head, T, d_k]
            v = v.view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)  # [B, n_kv_head, T, d_k]
            
            # Repeat K and V for each group of Q heads
            # From [B, n_kv_head, T, d_k] to [B, n_head, T, d_k]
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        else:
            # Multi-Head Attention: combined QKV projection
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            
            # Reshape to multi-head format: [B, H, T, d_k]
            q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Apply RoPE if configured
        if self.rope is not None and token_positions is not None:
            q, k = self.rope.apply_to_qk(q, k, token_positions)
        
        # Attention computation
        if self.attention_backend == 'flash_attn_3':
            # FlashAttention-3: Hopper/Blackwell optimized
            q = q.transpose(1, 2).contiguous()  # (B, T, H, D)
            k = k.transpose(1, 2).contiguous()  # (B, T, H, D)
            v = v.transpose(1, 2).contiguous()  # (B, T, H, D)
            y = flash_attn_3_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True)
            # y is already (B, T, H, D)
            y = y.reshape(B, T, C).contiguous()
            # Skip the reassemble step since y is already in the right format
            y = self.resid_dropout(self.c_proj(y))
            return y
            
        elif self.attention_backend == 'flash_attn_2':
            # FlashAttention-2: Explicit implementation
            q = q.transpose(1, 2).contiguous()  # (B, T, H, D)
            k = k.transpose(1, 2).contiguous()  # (B, T, H, D)
            v = v.transpose(1, 2).contiguous()  # (B, T, H, D)
            y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True)
            # y is already (B, T, H, D)
            y = y.reshape(B, T, C).contiguous()
            # Skip the reassemble step since y is already in the right format
            y = self.resid_dropout(self.c_proj(y))
            return y
            
        elif self.attention_backend == 'sdpa' and self.flash:
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
    
    def estimate_mfu_detailed(self, fwdbwd_per_iter, dt, device_type='cuda', num_gpus=1, use_sparse_specs=False):
        """
        Audit-compliant MFU calculation for modern Transformer architectures (2025).
        Refactored to align with "Review MFU Computation Logic" Gold Standard.
        
        This implementation uses pure component summation (not parameter-count heuristics):
        
        1. Grouped Query Attention (GQA) with explicit Q/K/V separation
        2. SwiGLU FFN with explicit 3-matrix calculation
        3. Vocabulary/Logit layer overhead (critical for Qwen)
        4. Dense vs Sparse hardware peaks (B200 Blackwell support)
        5. RoPE excluded from MFU numerator (per PaLM definition)
        
        Args:
            fwdbwd_per_iter: Global number of sequences processed per iteration
                             (micro_batch_size × grad_accum_per_gpu × world_size).
            dt: Wall-clock time for the iteration (seconds).
            device_type: Device type ('cuda', etc.)
            num_gpus: Number of GPUs
            use_sparse_specs: If True, use sparse tensor core specs (2:4 sparsity).
                             Default False uses dense specs for honest MFU reporting.
        
        Returns:
            dict: Complete MFU breakdown with audit-compliant metrics
        """
        cfg = self.config
        H = cfg.n_embd
        L = cfg.n_layer
        a = cfg.n_head
        S = cfg.block_size
        D_ff = cfg.d_ff
        V = cfg.vocab_size
        
        # GQA (Grouped Query Attention) parameters
        H_kv = getattr(cfg, 'num_key_value_heads', a)  # Default to MHA if not specified
        G = a / H_kv  # Group size
        
        # ===== 1. ATTENTION FLOPs (GQA-aware, per audit Section 4.1) =====
        # Q Projection: 2 * S * H * H
        flops_q = 2 * S * H * H
        
        # K/V Projections: Reduced by Group Size G
        # K: 2 * S * H * (H/G) | V: 2 * S * H * (H/G)
        flops_kv = 2 * (2 * S * H * (H / G))
        
        # Output Projection: 2 * S * H * H
        flops_proj = 2 * S * H * H
        
        # Attention Scores (Quadratic): 2 * S * S * H (QK^T)
        flops_scores = 2 * S * S * H
        
        # Attention Context (Quadratic): 2 * S * S * H (Softmax @ V)
        flops_context = 2 * S * S * H
        
        # Total Attention FLOPs per layer
        attention_flops = flops_q + flops_kv + flops_proj + flops_scores + flops_context
        
        # ===== 2. FFN FLOPs (SwiGLU-aware, per audit Section 3.2) =====
        if cfg.ffn_type == 'swiglu':
            # SwiGLU: 3 linear layers (Gate, Value, Out)
            # 3 matrices * (2 * S * H * D_ff)
            ffn_flops = 6 * S * H * D_ff
        else:
            # Standard GeLU: 2 linear layers (Up, Down)
            # 2 matrices * (2 * S * H * D_ff)
            ffn_flops = 4 * S * H * D_ff
        
        # ===== 3. ROPE & NORM FLOPs =====
        # RoPE is strictly excluded from MFU per PaLM definition
        # (non-GEMM operation, not Tensor Core saturating)
        rope_flops = 0
        
        # Norms: 2 per block + 1 final (RMSNorm ≈ 1.5·SH, LayerNorm = 2·SH)
        if cfg.normalization == 'rmsnorm':
            norm_flops_per_layer = 1.5 * S * H
        else:
            norm_flops_per_layer = 2 * S * H
        
        # ===== 4. LOGIT FLOPs (Critical for Qwen, per audit Section 5.1) =====
        # Vocabulary Projection: 2 * S * H * V
        logit_flops = 2 * S * H * V
        
        # ===== 5. TOTAL COMPUTE SUMMATION =====
        # Forward pass per layer
        flops_per_layer = attention_flops + ffn_flops + rope_flops + 2 * norm_flops_per_layer
        
        # Total Forward Model FLOPs (All Layers + Final Norm + Logits)
        total_forward_flops = (L * flops_per_layer) + norm_flops_per_layer + logit_flops
        
        # Total Training FLOPs (Forward + Backward)
        # Standard approximation: Backward = 2 * Forward → Total = 3 * Forward
        # Strictly excludes Activation Recomputation (HFU vs MFU distinction)
        model_flops_per_token = 3 * (total_forward_flops / S)
        
        # ===== 6. MFU METRICS =====
        # Total FLOPs for this iteration (Global sequences)
        tokens_per_iter = S * fwdbwd_per_iter
        flops_per_iter = model_flops_per_token * tokens_per_iter
        
        # Achieved throughput
        flops_achieved = flops_per_iter / dt
        flops_achieved_per_gpu = flops_achieved / max(num_gpus, 1)
        tokens_per_sec = tokens_per_iter / dt
        tokens_per_sec_per_gpu = tokens_per_sec / max(num_gpus, 1)
        
        # ===== 7. HARDWARE SPECS (Dense vs Sparse, per audit Section 5.1) =====
        # Dense specs: standard dense training (realistic for PyTorch)
        # Sparse specs: with 2:4 structured sparsity (requires special setup)
        hardware_specs_dense = {
            'cuda': {
                'B200': {'bf16': 2250e12, 'fp16': 2250e12, 'fp32': 90e12},
                'H200': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12},
                'H100': {'bf16': 989e12, 'fp16': 989e12, 'fp32': 67e12},
                'A100': {'bf16': 312e12, 'fp16': 312e12, 'fp32': 19.5e12},
                'A6000': {'bf16': 155.0e12, 'fp16': 155.0e12, 'fp32': 38.7e12},
                'V100': {'bf16': 125e12, 'fp16': 125e12, 'fp32': 15.7e12},
            }
        }
        
        hardware_specs_sparse = {
            'cuda': {
                'B200': {'bf16': 4500e12, 'fp16': 4500e12, 'fp32': 90e12},  # 2:4 structured sparsity
                'H200': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12},
                'H100': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12},  # 2× dense
                'A100': {'bf16': 312e12, 'fp16': 312e12, 'fp32': 19.5e12},
                'A6000': {'bf16': 309.7e12, 'fp16': 309.7e12, 'fp32': 38.7e12},  # 2:4 sparsity
                'V100': {'bf16': 125e12, 'fp16': 125e12, 'fp32': 15.7e12},
            }
        }
        
        # Select hardware specs based on mode
        hardware_specs = hardware_specs_sparse if use_sparse_specs else hardware_specs_dense
        
        # Auto-detect GPU
        gpu_name = 'A100'  # Default fallback
        if torch.cuda.is_available():
            gpu_name_full = torch.cuda.get_device_name(0)
            for name in ['B200', 'H200', 'H100', 'A100', 'A6000', 'V100']:
                if name in gpu_name_full:
                    gpu_name = name
                    break
        
        # Get precision from model dtype
        dtype = str(self.token_embeddings.weight.dtype).split('.')[-1]
        precision_key = 'bf16' if 'bfloat16' in dtype else 'fp16' if 'float16' in dtype else 'fp32'
        
        hardware_peak_flops_per_gpu = hardware_specs.get(device_type, {}).get(gpu_name, {}).get(precision_key, 312e12)
        hardware_peak_flops = hardware_peak_flops_per_gpu * num_gpus
        
        # ===== 8. MFU CALCULATION =====
        mfu = flops_achieved / hardware_peak_flops
        
        # Return comprehensive breakdown
        return {
            'mfu': mfu,
            'mfu_percent': mfu * 100,
            'flops_achieved': flops_achieved,
            'flops_achieved_per_gpu': flops_achieved_per_gpu,
            'flops_per_token': model_flops_per_token,
            'tokens_per_sec': tokens_per_sec,
            'tokens_per_sec_per_gpu': tokens_per_sec_per_gpu,
            'tokens_per_iter': tokens_per_iter,
            'hardware_peak_flops': hardware_peak_flops,
            'hardware_peak_tflops': hardware_peak_flops / 1e12,
            'achieved_tflops': flops_achieved / 1e12,
            'gpu_name': gpu_name,
            'precision': precision_key,
            'num_gpus': num_gpus,
            'sparse_mode': use_sparse_specs,
            'gqa_group_size': G,
            'attention_flops_per_layer': attention_flops,
            'ffn_flops_per_layer': ffn_flops,
            'logit_flops': logit_flops,
            'attention_to_ffn_ratio': attention_flops / ffn_flops if ffn_flops > 0 else 0,
            'architecture': cfg.get_architecture_name(),
            'calculation_method': 'component_summation_v2025'  # Audit-compliant method
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
