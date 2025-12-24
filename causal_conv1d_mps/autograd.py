"""
PyTorch Autograd Functions for CausalConv1D MPS Implementation

This module provides autograd-enabled functions that support backward pass
for causal convolution operations on Apple Silicon devices using Metal.
"""

import torch
from typing import Optional, Tuple, Any
from . import _C


class CausalConv1dMPSFunction(torch.autograd.Function):
    """
    PyTorch autograd function for causal 1D convolution with MPS backend.
    
    This function wraps the MPS implementation and provides automatic 
    differentiation support by implementing both forward and backward passes.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
        return_final_states: bool = False,
        final_states_out: Optional[torch.Tensor] = None,
        silu_activation: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for causal convolution.
        
        Args:
            ctx: Autograd context for saving tensors
            x: Input tensor of shape (batch, dim, seqlen)
            weight: Weight tensor of shape (dim, width)
            bias: Optional bias tensor of shape (dim,)
            seq_idx: Optional sequence indices (not implemented in MPS version)
            initial_states: Optional initial states (not implemented in MPS version)
            return_final_states: Whether to return final states
            final_states_out: Optional output tensor for final states
            silu_activation: Whether to apply SiLU activation
            
        Returns:
            Output tensor of shape (batch, dim, seqlen)
        """
        # Validate inputs
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this device")

        if x.device.type != "mps":
            raise ValueError("Input tensor must be on MPS device")

        if weight.device.type != "mps":
            raise ValueError("Weight tensor must be on MPS device")

        if bias is not None and bias.device.type != "mps":
            raise ValueError("Bias tensor must be on MPS device")

        # For now, ignore advanced features not implemented in MPS version
        if seq_idx is not None:
            raise NotImplementedError("seq_idx not supported in MPS version")
        
        if initial_states is not None:
            raise NotImplementedError("initial_states not supported in MPS version")
            
        if return_final_states:
            raise NotImplementedError("final states not supported in MPS version")

        # Validate tensor shapes
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, dim, seqlen), got {x.dim()}D")

        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight tensor (dim, width), got {weight.dim()}D")

        batch, dim, seqlen = x.shape
        weight_dim, width = weight.shape

        if dim != weight_dim:
            raise ValueError(f"Input dim {dim} does not match weight dim {weight_dim}")

        if bias is not None:
            if bias.dim() != 1:
                raise ValueError(f"Expected 1D bias tensor, got {bias.dim()}D")
            if bias.shape[0] != dim:
                raise ValueError(f"Bias dim {bias.shape[0]} does not match input dim {dim}")

        # Align dtypes
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)

        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        else:
            # Create empty tensor for C++ interface
            bias = torch.tensor([], device=x.device, dtype=x.dtype)

        # Save tensors for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.silu_activation = silu_activation
        
        # Call forward kernel
        output = _C.causal_conv1d_fwd(x, weight, bias, silu_activation)
        
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for causal convolution.
        
        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient w.r.t. output tensor
            
        Returns:
            Tuple of gradients w.r.t. inputs: (dx, dweight, dbias, None, None, None, None, None)
        """
        x, weight, bias = ctx.saved_tensors
        silu_activation = ctx.silu_activation
        
        # Ensure grad_output is contiguous
        if grad_output.stride(2) != 1 and grad_output.stride(1) != 1:
            grad_output = grad_output.contiguous()
        
        # Call backward kernel
        dx, dweight, dbias = _C.causal_conv1d_bwd(
            x, weight, bias, grad_output, silu_activation
        )
        
        # Return gradients in the same order as forward arguments
        # (x, weight, bias, seq_idx, initial_states, return_final_states, final_states_out, silu_activation)
        return (
            dx,
            dweight,
            dbias if (bias.numel() > 0) else None,
            None,  # seq_idx
            None,  # initial_states
            None,  # return_final_states
            None,  # final_states_out
            None,  # silu_activation
        )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Causal 1D convolution with automatic differentiation support.
    
    Args:
        x: Input tensor of shape (batch, dim, seqlen)
        weight: Weight tensor of shape (dim, width)  
        bias: Optional bias tensor of shape (dim,)
        seq_idx: Optional sequence indices (not supported in MPS version)
        initial_states: Optional initial states (not supported in MPS version)
        return_final_states: Whether to return final states (not supported)
        final_states_out: Optional output for final states (not supported)
        activation: Activation function ("silu", "swish", or None)
        
    Returns:
        Output tensor of shape (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    silu_activation = activation in ["silu", "swish"]
    
    return CausalConv1dMPSFunction.apply(
        x, weight, bias, seq_idx, initial_states, 
        return_final_states, final_states_out, silu_activation
    )

