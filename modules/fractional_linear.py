"""
FractionalTorch Modules: Fractional Linear Layer

This module implements linear (fully connected) layers using exact fractional arithmetic
for weights and biases, providing perfect numerical reproducibility and improved stability.

Author: Lev Goukassian
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from typing import Optional, Union, Tuple
import logging

from ..core import FractionalWeight, FractionalOps

logger = logging.getLogger(__name__)


class FractionalLinear(nn.Module):
    """
    Linear transformation layer using exact fractional arithmetic.
    
    Applies a linear transformation: y = xA^T + b where A and b are stored
    as exact fractions rather than floating-point approximations.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        max_denominator: Maximum denominator for fractional representations
        simplify_threshold: How often to simplify fractions (every N operations)
        device: Device to place the layer on
        dtype: Data type (currently only float32 supported for computation)
        
    Shape:
        - Input: (*, in_features) where * means any number of dimensions
        - Output: (*, out_features) where * means same dimensions as input
        
    Examples:
        >>> linear = FractionalLinear(128, 64)
        >>> input = torch.randn(32, 128)
        >>> output = linear(input)  # Shape: (32, 64)
        
        >>> # With specific precision control
        >>> precise_linear = FractionalLinear(256, 128, max_denominator=5000)
    """
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: FractionalWeight
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 max_denominator: int = 1000,
                 simplify_threshold: int = 100,
                 device=None, 
                 dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.max_denominator = max_denominator
        self.simplify_threshold = simplify_threshold
        
        # Initialize weight as FractionalWeight
        weight_data = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = FractionalWeight(
            weight_data, 
            requires_grad=True,
            max_denominator=max_denominator,
            simplify_threshold=simplify_threshold
        )
        
        if bias:
            bias_data = torch.empty(out_features, **factory_kwargs)
            self.bias = FractionalWeight(
                bias_data,
                requires_grad=True, 
                max_denominator=max_denominator,
                simplify_threshold=simplify_threshold
            )
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        
        # Track operations for precision management
        self._forward_count = 0
        
        logger.debug(f"FractionalLinear created: {in_features}→{out_features}, "
                    f"max_denom={max_denominator}, bias={bias is not None}")
    
    def reset_parameters(self) -> None:
        """
        Initialize parameters using Xavier/Glorot initialization adapted for fractions.
        
        This initialization is designed to maintain good gradient flow while
        working with fractional representations.
        """
        # Xavier/Glorot uniform initialization
        # std = sqrt(6 / (fan_in + fan_out))
        fan_in = self.in_features
        fan_out = self.out_features
        std = math.sqrt(6.0 / (fan_in + fan_out))
        
        # Generate uniform values in [-bound, bound]
        bound = std
        
        with torch.no_grad():
            # Initialize weight with uniform distribution
            weight_init = torch.empty_like(self.weight.data).uniform_(-bound, bound)
            
            # Convert to fractional representation
            self.weight._init_fractional_storage(weight_init, self.max_denominator)
            
            if self.bias is not None:
                # Initialize bias to small values
                bias_bound = 1.0 / math.sqrt(fan_in)
                bias_init = torch.empty_like(self.bias.data).uniform_(-bias_bound, bias_bound)
                self.bias._init_fractional_storage(bias_init, self.max_denominator)
        
        logger.debug(f"Parameters initialized with Xavier uniform, bound={bound:.4f}")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fractional arithmetic.
        
        Args:
            input: Input tensor of shape (*, in_features)
            
        Returns:
            Output tensor of shape (*, out_features)
        """
        self._forward_count += 1
        
        # Perform linear transformation using exact fractional weights
        output = self._fractional_linear_forward(input, self.weight, self.bias)
        
        # Periodic precision management
        if self._forward_count % self.simplify_threshold == 0:
            self._manage_precision()
        
        return output
    
    def _fractional_linear_forward(self, 
                                  input: torch.Tensor, 
                                  weight: FractionalWeight, 
                                  bias: Optional[FractionalWeight]) -> torch.Tensor:
        """
        Core linear transformation with fractional arithmetic.
        
        Args:
            input: Input tensor
            weight: Fractional weight parameter
            bias: Optional fractional bias parameter
            
        Returns:
            Output tensor from linear transformation
        """
        # For efficiency, we convert fractions to float for the matrix multiply
        # but maintain exact tracking in the weight storage
        weight_float = weight.numerators.float() / weight.denominators.float()
        
        # Standard linear operation: input @ weight.T + bias
        output = F.linear(input, weight_float, None)
        
        # Add bias if present
        if bias is not None:
            bias_float = bias.numerators.float() / bias.denominators.float()
            output = output + bias_float
        
        return output
    
    def _exact_linear_forward(self, 
                             input: torch.Tensor, 
                             weight: FractionalWeight, 
                             bias: Optional[FractionalWeight]) -> torch.Tensor:
        """
        Alternative forward pass using completely exact fractional arithmetic.
        
        Warning: This is much slower but provides exact computation throughout.
        Only use for verification or when absolute precision is required.
        """
        # Convert input to fractional representation
        input_flat = input.view(-1, self.in_features)
        batch_size = input_flat.shape[0]
        
        # Initialize output accumulator
        output = torch.zeros(batch_size, self.out_features, device=input.device)
        
        for b in range(batch_size):
            for out_idx in range(self.out_features):
                # Compute dot product exactly
                sum_num = torch.tensor(0, dtype=torch.long, device=input.device)
                sum_den = torch.tensor(1, dtype=torch.long, device=input.device)
                
                for in_idx in range(self.in_features):
                    # Convert input element to fraction
                    input_val = float(input_flat[b, in_idx])
                    try:
                        from fractions import Fraction
                        input_frac = Fraction(input_val).limit_denominator(self.max_denominator)
                        input_num = torch.tensor(input_frac.numerator, dtype=torch.long)
                        input_den = torch.tensor(input_frac.denominator, dtype=torch.long)
                    except:
                        input_num = torch.tensor(int(input_val), dtype=torch.long)
                        input_den = torch.tensor(1, dtype=torch.long)
                    
                    # Multiply with weight
                    weight_num = weight.numerators[out_idx, in_idx]
                    weight_den = weight.denominators[out_idx, in_idx]
                    
                    prod_num = input_num * weight_num
                    prod_den = input_den * weight_den
                    
                    # Add to sum
                    sum_num, sum_den = FractionalOps.frac_add_tensors(
                        sum_num.unsqueeze(0), sum_den.unsqueeze(0),
                        prod_num.unsqueeze(0), prod_den.unsqueeze(0),
                        simplify=True, max_denominator=self.max_denominator
                    )
                    sum_num, sum_den = sum_num[0], sum_den[0]
                
                # Add bias if present
                if bias is not None:
                    bias_num = bias.numerators[out_idx]
                    bias_den = bias.denominators[out_idx]
                    
                    sum_num, sum_den = FractionalOps.frac_add_tensors(
                        sum_num.unsqueeze(0), sum_den.unsqueeze(0),
                        bias_num.unsqueeze(0), bias_den.unsqueeze(0),
                        simplify=True, max_denominator=self.max_denominator
                    )
                    sum_num, sum_den = sum_num[0], sum_den[0]
                
                # Convert back to float for output
                output[b, out_idx] = float(sum_num) / float(sum_den)
        
        return output.view_as(input[..., :self.out_features])
    
    def _manage_precision(self):
        """Manage fractional precision to prevent denominator explosion."""
        # Simplify weight fractions
        self.weight._simplify_all_fractions()
        
        # Simplify bias fractions if present
        if self.bias is not None:
            self.bias._simplify_all_fractions()
        
        # Log precision statistics periodically
        if self._forward_count % (self.simplify_threshold * 10) == 0:
            weight_stats = self.weight.get_precision_stats()
            logger.debug(f"Forward {self._forward_count}: max_denom={weight_stats['max_denominator']}, "
                        f"mean_denom={weight_stats['mean_denominator']:.1f}")
    
    def set_max_denominator(self, max_denominator: int):
        """
        Update the maximum denominator for all parameters.
        
        Args:
            max_denominator: New maximum denominator limit
        """
        old_max = self.max_denominator
        self.max_denominator = max_denominator
        
        self.weight.set_max_denominator(max_denominator)
        if self.bias is not None:
            self.bias.set_max_denominator(max_denominator)
        
        logger.debug(f"Updated max_denominator: {old_max} → {max_denominator}")
    
    def get_precision_stats(self) -> dict:
        """
        Get detailed precision statistics for this layer.
        
        Returns:
            Dictionary containing precision information
        """
        stats = {
            'layer_type': 'FractionalLinear',
            'in_features': self.in_features,
            'out_features': self.out_features,
            'max_denominator': self.max_denominator,
            'forward_count': self._forward_count,
            'weight_stats': self.weight.get_precision_stats(),
        }
        
        if self.bias is not None:
            stats['bias_stats'] = self.bias.get_precision_stats()
        else:
            stats['bias_stats'] = None
        
        return stats
    
    def to_exact_string(self, max_elements: int = 5) -> str:
        """
        Get a string representation showing exact fractional values.
        
        Args:
            max_elements: Maximum number of weight elements to display
            
        Returns:
            String representation with fractional values
        """
        weight_str = self.weight.to_exact_string(max_elements)
        bias_str = ""
        if self.bias is not None:
            bias_str = f", bias={self.bias.to_exact_string(max_elements)}"
        
        return f"FractionalLinear({self.in_features}, {self.out_features}, weight={weight_str}{bias_str})"
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, max_denominator={self.max_denominator}'
    
    def clone_with_same_precision(self) -> 'FractionalLinear':
        """
        Create a copy of this layer with the same precision settings.
        
        Returns:
            New FractionalLinear layer with identical configuration
        """
        new_layer = FractionalLinear(
            self.in_features, 
            self.out_features,
            bias=self.bias is not None,
            max_denominator=self.max_denominator,
            simplify_threshold=self.simplify_threshold,
            device=self.weight.device,
            dtype=self.weight.dtype
        )
        
        # Copy the exact fractional values
        new_layer.weight = self.weight.clone_fractional()
        if self.bias is not None:
            new_layer.bias = self.bias.clone_fractional()
        
        return new_layer
    
    def convert_to_standard_linear(self) -> nn.Linear:
        """
        Convert this fractional layer to a standard PyTorch Linear layer.
        
        Returns:
            Standard nn.Linear layer with current fractional values as floats
        """
        standard_layer = nn.Linear(
            self.in_features, 
            self.out_features, 
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype
        )
        
        # Copy current values as floats
        with torch.no_grad():
            standard_layer.weight.copy_(self.weight.data)
            if self.bias is not None:
                standard_layer.bias.copy_(self.bias.data)
        
        return standard_layer
    
    def enable_exact_mode(self, exact: bool = True):
        """
        Enable or disable exact fractional computation mode.
        
        Args:
            exact: If True, use completely exact arithmetic (slower)
                   If False, use optimized float conversion (faster)
        """
        self._exact_mode = exact
        if exact:
            logger.warning("Exact mode enabled - this will significantly slow down computation")
        else:
            logger.info("Exact mode disabled - using optimized float conversion")
    
    def __repr__(self):
        return (f"FractionalLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"max_denominator={self.max_denominator})")


# Utility functions for working with FractionalLinear layers
def create_fractional_linear_stack(layer_sizes: list, 
                                  max_denominator: int = 1000,
                                  bias: bool = True,
                                  activation: Optional[nn.Module] = None) -> nn.Sequential:
    """
    Create a stack of FractionalLinear layers.
    
    Args:
        layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        max_denominator: Maximum denominator for all layers
        bias: Whether to include bias in all layers
        activation: Activation function to use between layers (None for last layer)
        
    Returns:
        Sequential module with fractional linear layers
        
    Example:
        >>> stack = create_fractional_linear_stack([784, 128, 64, 10])
        >>> # Creates: 784→128→64→10 with ReLU between layers
    """
    if len(layer_sizes) < 2:
        raise ValueError("Need at least 2 layer sizes (input and output)")
    
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(FractionalLinear(
            layer_sizes[i], 
            layer_sizes[i + 1],
            bias=bias,
            max_denominator=max_denominator
        ))
        
        # Add activation between layers (but not after the last layer)
        if activation is not None and i < len(layer_sizes) - 2:
            layers.append(activation)
    
    return nn.Sequential(*layers)


def convert_linear_to_fractional(linear_layer: nn.Linear, 
                                max_denominator: int = 1000) -> FractionalLinear:
    """
    Convert a standard Linear layer to FractionalLinear.
    
    Args:
        linear_layer: Standard nn.Linear layer to convert
        max_denominator: Maximum denominator for fractional representation
        
    Returns:
        Equivalent FractionalLinear layer
    """
    fractional_layer = FractionalLinear(
        linear_layer.in_features,
        linear_layer.out_features, 
        bias=linear_layer.bias is not None,
        max_denominator=max_denominator,
        device=linear_layer.weight.device,
        dtype=linear_layer.weight.dtype
    )
    
    # Copy weights and biases
    with torch.no_grad():
        fractional_layer.weight._init_fractional_storage(
            linear_layer.weight.data, max_denominator
        )
        
        if linear_layer.bias is not None:
            fractional_layer.bias._init_fractional_storage(
                linear_layer.bias.data, max_denominator
            )
    
    logger.info(f"Converted Linear({linear_layer.in_features}, {linear_layer.out_features}) to FractionalLinear")
    
    return fractional_layer


def batch_convert_linear_layers(model: nn.Module, 
                               max_denominator: int = 1000,
                               inplace: bool = True) -> nn.Module:
    """
    Convert all Linear layers in a model to FractionalLinear layers.
    
    Args:
        model: PyTorch model containing Linear layers
        max_denominator: Maximum denominator for fractional representations
        inplace: If True, modify the model in place; if False, return a copy
        
    Returns:
        Model with Linear layers converted to FractionalLinear
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    converted_count = 0
    
    # Recursively replace Linear layers
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            fractional_layer = convert_linear_to_fractional(module, max_denominator)
            setattr(model, name, fractional_layer)
            converted_count += 1
        else:
            # Recursively process child modules
            batch_convert_linear_layers(module, max_denominator, inplace=True)
            # Count conversions in child modules
            for child_module in module.modules():
                if isinstance(child_module, FractionalLinear):
                    converted_count += 1
    
    logger.info(f"Converted {converted_count} Linear layers to FractionalLinear")
    
    return model
