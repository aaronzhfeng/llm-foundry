"""
Full definition of a GPT Language Model, all of it in this single file.

Enhanced with detailed MFU calculation using academic formulas from:
- flops_parameter_counting (detailed FLOPs breakdown)
- MFU_compute (hardware-aware MFU calculation)

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) Insu Jang (2022) - "Analysis of Transformer Model":
https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Try importing FlashAttention-2
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN_2 = True
except ImportError:
    HAS_FLASH_ATTN_2 = False

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Attention backend configuration
        self.attention_backend = getattr(config, 'attention_backend', 'sdpa')
        
        # Check PyTorch SDPA availability (FlashAttention-1 backend)
        self.has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # Check FlashAttention-2 availability
        self.has_flash_attn_2 = HAS_FLASH_ATTN_2
        
        # Validate attention backend choice
        if self.attention_backend == 'flash_attn_2' and not self.has_flash_attn_2:
            print(f"WARNING: flash_attn_2 requested but flash-attn package not installed.")
            print(f"         Falling back to 'sdpa' (FlashAttention-1 via PyTorch)")
            self.attention_backend = 'sdpa'
        
        if self.attention_backend == 'sdpa' and not self.has_sdpa:
            print(f"WARNING: sdpa requested but PyTorch < 2.0. Falling back to 'manual' attention.")
            self.attention_backend = 'manual'
        
        # For manual attention, need causal mask
        if self.attention_backend == 'manual':
            print("INFO: Using manual attention (stores full O(S²) attention matrix)")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
        # Print which backend is actually being used
        backend_names = {
            'flash_attn_2': 'FlashAttention-2 (explicit, fastest)',
            'sdpa': 'PyTorch SDPA (FlashAttention-1, standard)',
            'manual': 'Manual attention (slow, for debugging)'
        }
        print(f"INFO: Attention backend: {backend_names.get(self.attention_backend, self.attention_backend)}")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Attention computation - 3 explicit backends
        if self.attention_backend == 'flash_attn_2':
            # FlashAttention-2: Explicit, fastest (~2× faster than FA-1)
            # Requires: pip install flash-attn
            # q, k, v should be (B, T, nh, hs) for flash_attn_func
            q = q.transpose(1, 2)  # (B, T, nh, hs)
            k = k.transpose(1, 2)  # (B, T, nh, hs)
            v = v.transpose(1, 2)  # (B, T, nh, hs)
            y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True)
            y = y.contiguous().view(B, T, C)  # Already in (B, T, nh, hs), just reshape
            
        elif self.attention_backend == 'sdpa':
            # PyTorch SDPA: FlashAttention-1 (standard, good compatibility)
            # Requires: PyTorch >= 2.0
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
            
        else:  # 'manual'
            # Manual attention: Stores full O(S²) attention matrix
            # Slow but works everywhere
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu_detailed(self, fwdbwd_per_iter, dt, device_type='cuda', num_gpus=1):
        """
        Estimate model flops utilization (MFU) with detailed breakdown.
        Uses academic formula: FLOPs = 12SBH² + 2aS²BH per layer (forward pass)
        
        Formula Reference:
        - Insu Jang (2022): https://insujang.github.io/2022-07-30/analysis-of-transformer-model/
        - Epoch AI backward/forward ratio: https://epoch.ai/blog/backward-forward-FLOP-ratio
        
        Returns dict with all MFU components for transparency.
        """
        cfg = self.config
        H = cfg.n_embd
        L = cfg.n_layer
        a = cfg.n_head
        S = cfg.block_size
        D_ff = 4 * H  # Standard GPT ratio
        
        # Forward pass FLOPs per layer (academic formula)
        # Attention: QKV projections (6SH²) + scores (aS²H) + output (aS²H) + proj (2SH²)
        attention_qkv_flops = 6 * S * H * H
        attention_scores_flops = a * S * S * H
        attention_output_flops = a * S * S * H
        attention_proj_flops = 2 * S * H * H
        attention_flops = attention_qkv_flops + attention_scores_flops + attention_output_flops + attention_proj_flops
        
        # FFN: up (2SH×D_ff) + down (2S×D_ff×H)
        ffn_up_flops = 2 * S * H * D_ff
        ffn_down_flops = 2 * S * D_ff * H
        ffn_flops = ffn_up_flops + ffn_down_flops
        
        # Layer norm (negligible but included for completeness)
        layernorm_flops = 2 * S * H  # 2 layer norms per layer
        
        flops_per_layer = attention_flops + ffn_flops + layernorm_flops
        
        # Total forward pass FLOPs
        forward_flops_per_token = L * flops_per_layer / S  # Per token
        
        # Training FLOPs (forward + backward)
        # Backward ≈ 2× forward (from Epoch AI research)
        training_flops_per_token = 3 * forward_flops_per_token
        
        # Total FLOPs for this iteration
        tokens_per_iter = S * fwdbwd_per_iter
        flops_per_iter = training_flops_per_token * tokens_per_iter
        
        # Achieved throughput
        flops_achieved = flops_per_iter / dt  # FLOPs per second
        tokens_per_sec = tokens_per_iter / dt
        
        # Hardware peak FLOPs (configurable with B200 support)
        hardware_specs = {
            'cuda': {
                'A100': {'bf16': 312e12, 'fp16': 312e12, 'fp32': 19.5e12},
                'H100': {'bf16': 989e12, 'fp16': 989e12, 'fp32': 67e12},
                'H200': {'bf16': 1979e12, 'fp16': 1979e12, 'fp32': 67e12},
                'B200': {'bf16': 4500e12, 'fp16': 4500e12, 'fp32': 90e12},  # B200 specs
                'V100': {'bf16': 125e12, 'fp16': 125e12, 'fp32': 15.7e12},
            }
        }
        
        # Auto-detect GPU if possible
        gpu_name = 'A100'  # Default
        if torch.cuda.is_available():
            gpu_name_full = torch.cuda.get_device_name(0)
            if 'B200' in gpu_name_full:
                gpu_name = 'B200'
            elif 'H200' in gpu_name_full:
                gpu_name = 'H200'
            elif 'H100' in gpu_name_full:
                gpu_name = 'H100'
            elif 'V100' in gpu_name_full:
                gpu_name = 'V100'
            elif 'A100' in gpu_name_full:
                gpu_name = 'A100'
        
        # Get precision
        dtype = str(self.transformer.wte.weight.dtype).split('.')[-1]
        precision_key = 'bf16' if 'bfloat16' in dtype else 'fp16' if 'float16' in dtype else 'fp32'
        
        hardware_peak_flops_per_gpu = hardware_specs.get(device_type, {}).get(gpu_name, {}).get(precision_key, 312e12)
        hardware_peak_flops = hardware_peak_flops_per_gpu * num_gpus
        
        # Calculate MFU
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
        }

    def get_memory_stats(self):
        """Get detailed memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
        }

    def get_gradient_stats(self):
        """Get gradient statistics for monitoring training health."""
        grad_norms = []
        grad_values = []
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                # Sample some gradient values (don't collect all to avoid memory issues)
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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Legacy method for compatibility. 
        Uses simplified PaLM formula: 6N + 12*L*H*Q*T
        For detailed MFU, use estimate_mfu_detailed() instead.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

