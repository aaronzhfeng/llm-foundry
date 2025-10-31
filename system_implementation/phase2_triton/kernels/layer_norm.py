"""
Triton implementation of LayerNorm with optional bias.
Optimized for GPT-style models with pre-norm architecture.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:
    @triton.jit
    def _layer_norm_fwd_kernel(
        X_ptr,  # pointer to input
        Y_ptr,  # pointer to output
        W_ptr,  # pointer to weight
        B_ptr,  # pointer to bias (can be null)
        Mean_ptr,  # pointer to mean (for backward)
        Rstd_ptr,  # pointer to rstd (for backward)
        stride_x_row,  # stride for rows in X
        stride_y_row,  # stride for rows in Y
        N,  # number of columns (feature dimension)
        eps,  # epsilon for numerical stability
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused LayerNorm kernel.
        Each program instance normalizes one row.
        """
        # Get row index
        row_idx = tl.program_id(0)
        
        # Compute row offsets
        X_row_ptr = X_ptr + row_idx * stride_x_row
        Y_row_ptr = Y_ptr + row_idx * stride_y_row
        
        # Compute mean
        mean = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            mean += tl.sum(x)
        mean = mean / N
        
        # Compute variance
        variance = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            variance += tl.sum(diff * diff)
        variance = variance / N
        rstd = 1.0 / tl.sqrt(variance + eps)
        
        # Normalize and apply affine transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            
            # Load input
            x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            
            # Normalize
            x_hat = (x - mean) * rstd
            
            # Apply weight
            w = tl.load(W_ptr + cols, mask=mask, other=1.0)
            y = x_hat * w
            
            # Apply bias if present
            if HAS_BIAS:
                b = tl.load(B_ptr + cols, mask=mask, other=0.0)
                y = y + b
            
            # Store output
            tl.store(Y_row_ptr + cols, y, mask=mask)
        
        # Store mean and rstd for backward pass
        tl.store(Mean_ptr + row_idx, mean)
        tl.store(Rstd_ptr + row_idx, rstd)


    @triton.jit
    def _layer_norm_bwd_kernel(
        DY_ptr,  # gradient w.r.t. output
        X_ptr,   # input
        W_ptr,   # weight
        Mean_ptr,  # mean from forward
        Rstd_ptr,  # rstd from forward
        DX_ptr,  # gradient w.r.t. input
        DW_ptr,  # gradient w.r.t. weight
        DB_ptr,  # gradient w.r.t. bias (can be null)
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        LayerNorm backward kernel.
        """
        row_idx = tl.program_id(0)
        
        DY_row_ptr = DY_ptr + row_idx * stride_dy_row
        X_row_ptr = X_ptr + row_idx * stride_x_row
        DX_row_ptr = DX_ptr + row_idx * stride_dx_row
        
        mean = tl.load(Mean_ptr + row_idx)
        rstd = tl.load(Rstd_ptr + row_idx)
        
        # First pass: compute intermediate sums
        sum1 = 0.0
        sum2 = 0.0
        
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            
            x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=1.0)
            
            x_hat = (x - mean) * rstd
            wdy = w * dy
            
            sum1 += tl.sum(wdy)
            sum2 += tl.sum(wdy * x_hat)
        
        # Second pass: compute gradients
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            
            x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=1.0)
            
            x_hat = (x - mean) * rstd
            wdy = w * dy
            
            # Gradient w.r.t. input
            dx = (wdy - (sum1 + x_hat * sum2) / N) * rstd
            tl.store(DX_row_ptr + cols, dx, mask=mask)
            
            # Accumulate gradients for weight and bias
            dw = dy * x_hat
            tl.atomic_add(DW_ptr + cols, dw, mask=mask)
            
            if HAS_BIAS:
                tl.atomic_add(DB_ptr + cols, dy, mask=mask)


class TritonLayerNorm(torch.autograd.Function):
    """
    Triton-accelerated LayerNorm with optional bias.
    Falls back to PyTorch implementation if Triton is not available.
    """
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch
            return torch.nn.functional.layer_norm(
                x, weight.shape, weight, bias, eps
            )
        
        # Ensure contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Get dimensions
        *batch_dims, N = x.shape
        M = x.numel() // N  # total number of rows
        
        # Flatten to 2D
        x_2d = x.view(M, N)
        
        # Allocate outputs
        y = torch.empty_like(x)
        y_2d = y.view(M, N)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        
        # Determine block size
        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE > 4096:
            BLOCK_SIZE = 4096
        
        # Launch kernel
        grid = (M,)
        _layer_norm_fwd_kernel[grid](
            x_2d, y_2d, weight, bias, mean, rstd,
            x_2d.stride(0), y_2d.stride(0),
            N, eps,
            HAS_BIAS=bias is not None,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.N = N
        
        return y
    
    @staticmethod
    def backward(ctx, dy):
        if not TRITON_AVAILABLE:
            # This shouldn't happen if forward used Triton
            raise RuntimeError("Triton backward called but Triton not available")
        
        x, weight, bias, mean, rstd = ctx.saved_tensors
        N = ctx.N
        BLOCK_SIZE = ctx.BLOCK_SIZE
        
        # Get dimensions
        *batch_dims, _ = x.shape
        M = x.numel() // N
        
        # Flatten
        dy_2d = dy.contiguous().view(M, N)
        x_2d = x.view(M, N)
        
        # Allocate gradient outputs
        dx = torch.empty_like(x)
        dx_2d = dx.view(M, N)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(bias) if bias is not None else None
        
        # Launch kernel
        grid = (M,)
        _layer_norm_bwd_kernel[grid](
            dy_2d, x_2d, weight, mean, rstd,
            dx_2d, dw, db,
            dy_2d.stride(0), x_2d.stride(0), dx_2d.stride(0),
            N,
            HAS_BIAS=bias is not None,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return dx, dw, db, None


def triton_layer_norm(x, weight, bias, eps=1e-5):
    """
    Functional interface for Triton LayerNorm.
    
    Args:
        x: Input tensor of shape (*batch_dims, normalized_shape)
        weight: Weight tensor of shape (normalized_shape,)
        bias: Bias tensor of shape (normalized_shape,) or None
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as x
    """
    return TritonLayerNorm.apply(x, weight, bias, eps)

