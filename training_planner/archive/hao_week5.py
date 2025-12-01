"""
Lean memory/parameter analyzer for our decoder-only MoE model (BF16, Muon).
- Assumes PreNorm + RMSNorm, SwiGLU, no biases, RoPE, FlashAttn kernels.
- Focuses on DP + Expert Parallel (no ZeRO/offload/checkpointing/accum).
- Uses Muon-style optimizer memory (~4 bytes per parameter).
Run: `python detailed_cost_analysis.py` (auto prints summary).
"""

from __future__ import annotations
import json, math, os, sys
from dataclasses import dataclass

BF16_BYTES = 2
MUON_OPT_BYTES_PER_PARAM = 4   # conservative: 1 FP32-like state per param

@dataclass
class ModelCfg:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    v_head_dim: int
    qk_nope_head_dim: int
    vocab_size: int
    max_position_embeddings: int
    intermediate_size: int
    # MoE
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    first_k_dense_replace: int
    # misc
    torch_dtype: str = "bfloat16"
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

def round_up(x, m): return ((x + m - 1) // m) * m

def load_cfg(path: str) -> ModelCfg:
    with open(path, "r") as f:
        c = json.load(f)
    return ModelCfg(
        hidden_size = c["hidden_size"],
        num_hidden_layers = c["num_hidden_layers"],
        num_attention_heads = c["num_attention_heads"],
        v_head_dim = c.get("v_head_dim", c.get("head_dim", 128)),
        qk_nope_head_dim = c.get("qk_nope_head_dim", 128),
        vocab_size = c["vocab_size"],
        max_position_embeddings = c.get("max_position_embeddings", 4096),
        intermediate_size = c["intermediate_size"],
        n_routed_experts = c.get("n_routed_experts", 0),
        num_experts_per_tok = c.get("num_experts_per_tok", 0),
        n_shared_experts = c.get("n_shared_experts", 0),
        first_k_dense_replace = c.get("first_k_dense_replace", 0),
    )

def attn_params_per_layer(H: int) -> int:
    # q, k, v, o projections; no biases; treat as dense HxH each
    return 4 * H * H

def mlp_params_dense(H: int, d_ff: int) -> int:
    # SwiGLU: [H -> 2*d_ff] + [d_ff -> H]; no bias
    return H * (2 * d_ff) + d_ff * H

def mlp_params_moe(H: int, d_ff: int, n_experts: int) -> int:
    # Each expert has its own SwiGLU MLP weights; include small router (H * n_experts)
    return n_experts * mlp_params_dense(H, d_ff) + (H * n_experts)

def rmsnorm_params(H: int) -> int:
    # One scale per hidden dim; two norms per block
    return 2 * H

def params_total(cfg: ModelCfg) -> int:
    H, L, d_ff = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    # Embedding + LM head (untied)
    embed = cfg.vocab_size * H
    lm_head = cfg.vocab_size * H  # untied
    blocks = 0
    for l in range(L):
        blocks += attn_params_per_layer(H)
        if l < cfg.first_k_dense_replace:
            blocks += mlp_params_dense(H, d_ff)
        else:
            blocks += mlp_params_moe(H, d_ff, cfg.n_routed_experts + cfg.n_shared_experts)
        blocks += rmsnorm_params(H)
    return embed + lm_head + blocks

def memory_breakdown_bytes(n_params: int, seq_len: int, layers: int, hidden: int) -> dict:
    model = n_params * BF16_BYTES
    grads = n_params * BF16_BYTES
    optim = n_params * MUON_OPT_BYTES_PER_PARAM

    # Activation model: scale linearly with sequence length.
    # Calibrated to ~15.21875 GB at 4096 seq for our sizes.
    base_gb_4096 = 15.21875
    act = base_gb_4096 * (seq_len / 4096.0) * (hidden / 3584.0) ** 1.0 * (layers / 32.0) ** 1.0
    activations = int(act * (1024**3))

    total = model + grads + optim + activations
    return {
        "model_bytes": model,
        "gradient_bytes": grads,
        "optimizer_bytes": optim,
        "activation_bytes": activations,
        "total_bytes": total,
    }

def to_gb(b): return b / (1024**3)

def pretty(breakdown: dict) -> dict:
    return {k.replace("_bytes",""): round(to_gb(v), 3) for k,v in breakdown.items()}

def summarize(cfg_path: str, num_gpus: int = 8, gpu_mem_gb: int = 192) -> dict:
    cfg = load_cfg(cfg_path)
    n_params = params_total(cfg)
    mb = memory_breakdown_bytes(n_params, cfg.max_position_embeddings, cfg.num_hidden_layers, cfg.hidden_size)
    per_replica_gb = to_gb(mb["total_bytes"])
    return {
        "config_path": cfg_path,
        "total_params": n_params,
        "total_params_B": n_params / 1e9,
        "per_replica_peak_GB": per_replica_gb,
        "memory_breakdown_GB": pretty(mb),
        "fits_per_gpu": per_replica_gb <= gpu_mem_gb,
        "gpu_memory_GB": gpu_mem_gb,
        "gpus": num_gpus,
        "notes": "BF16 weights/grads; Muon optimizer; DP+EP; no ZeRO/offload; 4k ctx baseline scaled linearly for activations."
    }

def main():
    cfg_candidates = [
        "our_moe_config.json",
    ]
    for p in cfg_candidates:
        if os.path.exists(p):
            s = summarize(p)
            print("="*72)
            print(f"Config: {s['config_path']}")
            print(f"Total params: {s['total_params_B']:.3f} B  ({int(s['total_params']):,} params)")
            print(f"Per-replica peak memory (GB): {s['per_replica_peak_GB']:.2f}")
            print(f"  Breakdown (GB): {s['memory_breakdown_GB']}")
            print(f"GPU Memory (GB): {s['gpu_memory_GB']}  | Fits per GPU: {s['fits_per_gpu']}")
            print(f"Assumptions: {s['notes']}")
    print("="*72)

if __name__ == "__main__":
    main()