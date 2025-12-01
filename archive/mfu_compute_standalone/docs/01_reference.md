# MFU-related code and definitions

# --- 1. nanoGPT: MFU estimation for GPT-style models
# Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
def estimate_mfu(self, fwdbwd_per_iter: int, dt: float):
    """
    Estimate model flops utilization (MFU) in units of the A100 peak FLOPs.
    Arguments:
      - fwdbwd_per_iter: number of forward/backward passes per iteration (e.g. 1)
      - dt: time per iteration in seconds (wall‑clock)
    Steps:
      1. Estimate FLOPs per token:
         flops_per_token = 6*N + 12*num_layers*num_heads*head_dim*context_length
         # The '6*N' term accounts for forward/backward passes through the model weights.
         # The second term estimates attention projection and softmax costs.
      2. Multiply by sequence length and fwdbwd_per_iter to get FLOPs per iteration.
      3. Divide by dt to get FLOPs per second (achieved).
      4. Divide by flops_promised (peak A100 TFLOPS) to get MFU.
    """
    if dt == 0:
        return 0.0
    # compute approximate flops per token
    flops_per_token = 6 * self.num_parameters + 12 * self.n_layer * self.n_head * self.head_dim * self.block_size
    # forward/backward per iteration
    flops_per_fwdbwd = flops_per_token * self.block_size
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter / dt
    flops_promised = 312e12  # A100 peak FP16 FLOPs/sec
    mfu = flops_achieved / flops_promised
    return mfu

# Usage (in bench.py):
# mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
# print(f" achieved MFU = {mfu*100:.2f}%")
# The code multiplies batch_size * num_steps to get fwdbwd_per_iter, measures dt, and prints MFU.

# --- 2. Detailed FLOP formulas (Chinchilla / Transformer FLOPs)
# The simplified 6*N*D model ignores embeddings, softmax, and de-embedding; more accurate counting is available.
# For example, the Transformer FLOPs breakdown (from Adam Casson’s article) sums per-operation FLOPs:
#   - Embeddings: 2 * n_ctx * vocab_size * d_model
#   - QKV projections: 2 * n_ctx * 3 * d_model * d_key * n_heads
#   - Attention dot products: 2 * n_ctx^2 * d_key * n_heads
#   - Softmax and reductions: O(n_ctx^2 * n_heads)
#   - Feedforward layers: 4 * n_ctx * d_model * d_ff
#   - Projection back: 2 * n_ctx * (d_key * n_heads) * d_model
#   - De-embedding (logits): 2 * n_ctx * d_model * vocab_size
# DeepMind’s Chinchilla paper tabulates similar formulas.  The total forward-pass FLOPs are summed and then doubled
# for the backward pass to get ~6*N*D with context-dependent corrections:contentReference[oaicite:0]{index=0}.

# --- 3. Chinchilla toolkit: compute-optimal training functions
# Source: https://github.com/kyo-takano/chinchilla/blob/master/chinchilla/core.py
class Chinchilla:
    def adjust_D_to_N(self, N: float) -> float:
        """
        Given model size N and scaling-law parameters (A,B,alpha,beta),
        adjust the number of data samples D so that the loss predictor is balanced.
        Uses: D = G^(-(1 + b/a)) * N^(b/a), where a=beta/(alpha+beta), b=alpha/(alpha+beta).
        """:contentReference[oaicite:1]{index=1}

    @classmethod
    def allocate_compute(cls, C: float, params: dict) -> tuple[float, float]:
        """
        Allocate a compute budget C optimally between model size (N) and data size (D).
        Returns N_opt = G * (C/6)^(beta/(alpha+beta)), D_opt = (1/G) * (C/6)^(alpha/(alpha+beta)).
        Requires scaling-law parameters (alpha, beta, A, B).
        """:contentReference[oaicite:2]{index=2}

    @classmethod
    def predict_loss(cls, N: float, D: float, params: dict) -> float:
        """
        Predicts the training loss using the scaling-law model:
        L(N,D) = E + A * N^-alpha + B * D^-beta.
        """:contentReference[oaicite:3]{index=3}

# These functions are not directly used to compute MFU, but they provide the compute law C≈6ND,
# which underlies MFU calculations.  They can be used to choose N and D given a compute budget.

# --- 4. Notes on MFU definitions and usage
# MFU measures the ratio of actual FLOPs per second to the peak FLOPs of the hardware.
# In other words:
#   MFU = (FLOPs_per_token * tokens_per_second) / FLOPs_peak
# where
#   FLOPs_per_token: theoretical FLOPs needed per token (model-dependent)
#   tokens_per_second: throughput (batch_size * seq_len / runtime)
#   FLOPs_peak: theoretical peak FLOP/s of the GPU (or cluster)
# Different formulations exist (e.g., counting per sequence vs per token),
# but the key idea is that MFU accounts for how well we saturate the hardware’s capability.



### Proposed MFU formula

A general formula that captures nanoGPT’s intent but allows more detail is:

[
\text{MFU} = \frac{\text{FLOPs}*\text{per–token} \times \text{tokens}*\text{per–second}}{\text{FLOPs}_\text{peak}}
]

Where:

* **FLOPs_per–token** – the theoretical number of floating‑point operations required to process one token (forward plus backward).  NanoGPT uses (6N) plus an attention‑cost term.  For greater accuracy, sum the FLOPs of embeddings, QKV projections, dot‑products, softmax, feed‑forward layers, and de‑embedding.

* **tokens_per–second** – how many tokens you actually process per second.  This equals `(batch_size × sequence_length × forward_backward_passes) ÷ runtime`.

* **FLOPs_peak** – the theoretical peak FLOPs/s of your hardware.  Multiply the per‑GPU peak by the number of GPUs used.

### Additional information needed and suggested defaults

To compute MFU accurately, you need more than just parameter counts and FLOPs:

1. **Detailed FLOP formula**:

   * Approximate: (6N) for decoder‑only models (as nanoGPT does).
   * More precise: include terms for embeddings, QKV, feed‑forward, attention softmax, etc., as outlined above.

2. **Measured throughput** (`tokens_per_second`):

   * You must measure how many tokens your training loop processes per second.  This requires timing your training code and knowing the batch size and sequence length.

3. **Hardware peak FLOPs** (`FLOPs_peak`):

   * The peak FLOP/s of your GPU or cluster.  For an **NVIDIA B200**, published figures indicate roughly **2 petaFLOPS (2×10^15) of FP8 peak performance per GPU** (and around **1 petaFLOPS of FP16 performance**).  If you have **8 B200 GPUs**, a reasonable default for FP8 training would be
     [
     \text{FLOPs}_\text{peak} \approx 8 \times 2\times10^{15} = 1.6\times10^{16}\ \text{FLOPs/s}.
     ]
     For FP16 training you might assume ~1 PFLOPS per GPU, giving (8\times10^{15}) FLOPs/s.  Exact figures should be taken from NVIDIA’s specifications for your chosen precision.

4. **Model parameters** (`N`, number of layers, heads, etc.):

   * Required to compute theoretical FLOPs.  These come from your model configuration (e.g. hidden size, number of layers).

5. **Context length and sequence length** (`seq_len`):

   * Used in the attention‑cost calculation and to compute tokens/sec.

### Summary of other related information

* **MFU origin:** Proposed in Google’s PaLM paper (2022) as a hardware‑agnostic efficiency metric.  It uses the compute law (C ≈ 6ND) and measures the ratio of achieved FLOPs/s to the theoretical peak.

* **Simplifications vs detailed models:**  NanoGPT uses a simplified estimate (6 N plus a term for attention), which is sufficient for relative benchmarking but ignores embeddings and de‑embedding.  More detailed calculators (e.g. Chinchilla or Adam Casson’s FLOPs breakdown) count each transformer component.

* **Scaling law connection:**  The Chinchilla scaling law (Hoffmann et al., 2022) provides formulas for optimal model size and data size under a compute budget.  While not directly used for MFU, it relies on the same compute law and can help determine training configurations that maximize efficiency.

By combining detailed FLOP calculations, measured throughput, and the appropriate peak FLOPs for your 8 × B200 cluster, you can compute MFU more accurately than nanoGPT’s default implementation.
