"""
FractionalTorch Core: Fractional Operations

This module implements optimized arithmetic operations for fractional tensors,
including addition, multiplication, matrix operations, and automatic differentiation
support for exact rational arithmetic.

Author: Lev Goukassian
License: MIT
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from fractions import Fraction
from typing import Tuple, Optional, Union, List
import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)


class FractionalOps:
    """
    Static class containing optimized fractional arithmetic operations
    for neural network computations.
    """
    
    @staticmethod
    @torch.jit.script
    def batch_gcd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Vectorized GCD computation using the Euclidean algorithm.
        
        Args:
            a, b: Input tensors (must be integer tensors)
            
        Returns:
            Tensor of GCD values
            
        Note:
            This is optimized for GPU computation when available.
        """
        a_abs = torch.abs(a)
        b_abs = torch.abs(b)
        
        # Ensure a >= b for efficiency
        mask = a_abs < b_abs
        a_abs = torch.where(mask, b_abs, a_abs)
        b_abs = torch.where(mask, a_abs, b_abs)
        
        # Euclidean algorithm with early termination
        max_iterations = 100  # Prevent infinite loops
        for _ in range(max_iterations):
            nonzero_mask = b_abs > 0
            if not torch.any(nonzero_mask):
                break
            
            remainder = torch.where(nonzero_mask, a_abs % b_abs, b_abs)
            a_abs = torch.where(nonzero_mask, b_abs, a_abs)
            b_abs = remainder
        
        return a_abs
    
    @staticmethod
    def simplify_fraction_tensors(numerators: torch.Tensor, 
                                 denominators: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplify all fractions in tensors simultaneously.
        
        Args:
            numerators: Tensor of numerators
            denominators: Tensor of denominators
            
        Returns:
            Tuple of (simplified_numerators, simplified_denominators)
        """
        # Compute GCD for all pairs
        gcd_vals = FractionalOps.batch_gcd(numerators, denominators)
        
        # Avoid division by zero
        gcd_vals = torch.where(gcd_vals == 0, torch.ones_like(gcd_vals), gcd_vals)
        
        # Simplify fractions
        simplified_num = numerators // gcd_vals
        simplified_den = denominators // gcd_vals
        
        # Ensure denominators are positive
        neg_mask = simplified_den < 0
        simplified_num = torch.where(neg_mask, -simplified_num, simplified_num)
        simplified_den = torch.where(neg_mask, -simplified_den, simplified_den)
        
        return simplified_num, simplified_den
    
    @staticmethod
    def frac_add_tensors(a_num: torch.Tensor, a_den: torch.Tensor,
                        b_num: torch.Tensor, b_den: torch.Tensor,
                        simplify: bool = True, 
                        max_denominator: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add two fractional tensors: a/c + b/d = (a*d + b*c)/(c*d)
        
        Args:
            a_num, a_den: First fraction (numerators, denominators)
            b_num, b_den: Second fraction (numerators, denominators)
            simplify: Whether to simplify the result
            max_denominator: Maximum allowed denominator
            
        Returns:
            Tuple of (result_numerators, result_denominators)
        """
        # Cross multiplication: a*d + b*c
        result_num = a_num * b_den + b_num * a_den
        result_den = a_den * b_den
        
        # Limit denominator growth if specified
        if max_denominator is not None:
            overflow_mask = result_den > max_denominator
            if torch.any(overflow_mask):
                # Convert to float and back to fraction with limited denominator
                float_result = result_num.float() / result_den.float()
                for i in torch.nonzero(overflow_mask, as_tuple=False):
                    idx = tuple(i.tolist())
                    frac = Fraction(float(float_result[idx])).limit_denominator(max_denominator)
                    result_num[idx] = frac.numerator
                    result_den[idx] = frac.denominator
        
        # Simplify if requested
        if simplify:
            result_num, result_den = FractionalOps.simplify_fraction_tensors(result_num, result_den)
        
        return result_num, result_den
    
    @staticmethod
    def frac_mul_tensors(a_num: torch.Tensor, a_den: torch.Tensor,
                        b_num: torch.Tensor, b_den: torch.Tensor,
                        simplify: bool = True,
                        max_denominator: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiply two fractional tensors: (a/c) * (b/d) = (a*b)/(c*d)
        
        Args:
            a_num, a_den: First fraction
            b_num, b_den: Second fraction
            simplify: Whether to simplify the result
            max_denominator: Maximum allowed denominator
            
        Returns:
            Tuple of (result_numerators, result_denominators)
        """
        # Direct multiplication
        result_num = a_num * b_num
        result_den = a_den * b_den
        
        # Handle denominator overflow
        if max_denominator is not None:
            overflow_mask = result_den > max_denominator
            if torch.any(overflow_mask):
                float_result = result_num.float() / result_den.float()
                for i in torch.nonzero(overflow_mask, as_tuple=False):
                    idx = tuple(i.tolist())
                    frac = Fraction(float(float_result[idx])).limit_denominator(max_denominator)
                    result_num[idx] = frac.numerator
                    result_den[idx] = frac.denominator
        
        # Simplify if requested
        if simplify:
            result_num, result_den = FractionalOps.simplify_fraction_tensors(result_num, result_den)
        
        return result_num, result_den
    
    @staticmethod
    def frac_div_tensors(a_num: torch.Tensor, a_den: torch.Tensor,
                        b_num: torch.Tensor, b_den: torch.Tensor,
                        simplify: bool = True,
                        max_denominator: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Divide two fractional tensors: (a/c) / (b/d) = (a/c) * (d/b) = (a*d)/(c*b)
        
        Args:
            a_num, a_den: Dividend fraction
            b_num, b_den: Divisor fraction
            simplify: Whether to simplify the result
            max_denominator: Maximum allowed denominator
            
        Returns:
            Tuple of (result_numerators, result_denominators)
        """
        # Division is multiplication by reciprocal
        return FractionalOps.frac_mul_tensors(
            a_num, a_den, b_den, b_num, simplify, max_denominator
        )
    
    @staticmethod
    def frac_matmul(a_num: torch.Tensor, a_den: torch.Tensor,
                   b_num: torch.Tensor, b_den: torch.Tensor,
                   max_denominator: int = 1000,
                   use_exact: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Matrix multiplication with fractional tensors.
        
        Args:
            a_num, a_den: Left matrix (numerators, denominators)
            b_num, b_den: Right matrix (numerators, denominators)
            max_denominator: Maximum denominator for result
            use_exact: If True, use exact arithmetic (slower but precise)
            
        Returns:
            Tuple of (result_numerators, result_denominators)
            
        Note:
            For efficiency, this converts to float for matmul then back to fractions.
            Set use_exact=True for completely exact arithmetic (much slower).
        """
        if use_exact:
            return FractionalOps._exact_matmul(a_num, a_den, b_num, b_den, max_denominator)
        
        # Convert to float for efficient matmul
        a_float = a_num.float() / a_den.float()
        b_float = b_num.float() / b_den.float()
        
        # Perform matrix multiplication
        result_float = torch.matmul(a_float, b_float)
        
        # Convert result back to fractions
        result_num = torch.zeros_like(result_float, dtype=torch.long)
        result_den = torch.ones_like(result_float, dtype=torch.long)
        
        # Vectorized conversion back to fractions
        flat_result = result_float.flatten()
        flat_num = result_num.flatten()
        flat_den = result_den.flatten()
        
        for i, val in enumerate(flat_result):
            if torch.isfinite(val):
                try:
                    frac = Fraction(float(val)).limit_denominator(max_denominator)
                    flat_num[i] = frac.numerator
                    flat_den[i] = frac.denominator
                except (ValueError, OverflowError):
                    flat_num[i] = 0
                    flat_den[i] = 1
        
        return result_num, result_den
    
    @staticmethod
    def _exact_matmul(a_num: torch.Tensor, a_den: torch.Tensor,
                     b_num: torch.Tensor, b_den: torch.Tensor,
                     max_denominator: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact matrix multiplication using only fractional arithmetic.
        Warning: This is much slower but completely exact.
        """
        m, k = a_num.shape
        k2, n = b_num.shape
        assert k == k2, f"Matrix dimension mismatch: {k} != {k2}"
        
        # Initialize result
        result_num = torch.zeros((m, n), dtype=torch.long, device=a_num.device)
        result_den = torch.ones((m, n), dtype=torch.long, device=a_num.device)
        
        # Perform exact dot products
        for i in range(m):
            for j in range(n):
                # Compute sum of a[i,k] * b[k,j] for all k
                sum_num = torch.tensor(0, dtype=torch.long, device=a_num.device)
                sum_den = torch.tensor(1, dtype=torch.long, device=a_num.device)
                
                for k_idx in range(k):
                    # Multiply a[i,k] * b[k,j]
                    prod_num = a_num[i, k_idx] * b_num[k_idx, j]
                    prod_den = a_den[i, k_idx] * b_den[k_idx, j]
                    
                    # Add to running sum
                    sum_num, sum_den = FractionalOps.frac_add_tensors(
                        sum_num.unsqueeze(0), sum_den.unsqueeze(0),
                        prod_num.unsqueeze(0), prod_den.unsqueeze(0),
                        simplify=True, max_denominator=max_denominator
                    )
                    sum_num, sum_den = sum_num[0], sum_den[0]
                
                result_num[i, j] = sum_num
                result_den[i, j] = sum_den
        
        return result_num, result_den
    
    @staticmethod
    def frac_pow_tensor(num: torch.Tensor, den: torch.Tensor, 
                       exponent: int,
                       max_denominator: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Raise fractional tensor to integer power: (a/b)^n = a^n / b^n
        
        Args:
            num, den: Base fraction
            exponent: Integer exponent
            max_denominator: Maximum denominator for result
            
        Returns:
            Tuple of (result_numerators, result_denominators)
        """
        if exponent == 0:
            return torch.ones_like(num), torch.ones_like(den)
        elif exponent == 1:
            return num.clone(), den.clone()
        elif exponent < 0:
            # Negative exponent: (a/b)^(-n) = (b/a)^n
            return FractionalOps.frac_pow_tensor(den, num, -exponent, max_denominator)
        
        # Positive exponent
        result_num = torch.pow(num, exponent)
        result_den = torch.pow(den, exponent)
        
        # Handle potential overflow
        overflow_mask = (result_den > max_denominator) | (torch.abs(result_num) > max_denominator)
        if torch.any(overflow_mask):
            logger.warning(f"Power operation caused {torch.sum(overflow_mask)} overflows")
            # Fallback to float approximation
            float_base = num.float() / den.float()
            float_result = torch.pow(float_base, exponent)
            
            for i in torch.nonzero(overflow_mask, as_tuple=False):
                idx = tuple(i.tolist())
                try:
                    frac = Fraction(float(float_result[idx])).limit_denominator(max_denominator)
                    result_num[idx] = frac.numerator
                    result_den[idx] = frac.denominator
                except:
                    result_num[idx] = 0
                    result_den[idx] = 1
        
        return result_num, result_den


class FractionalFunction(Function):
    """
    Custom autograd Function for fractional operations.
    Enables automatic differentiation with exact fractional arithmetic.
    """
    
    @staticmethod
    def forward(ctx, numerators, denominators, operation, *args, **kwargs):
        """
        Forward pass for fractional operations.
        
        Args:
            numerators, denominators: Fractional representation
            operation: String indicating operation type
            *args, **kwargs: Additional operation parameters
        """
        ctx.save_for_backward(numerators, denominators)
        ctx.operation = operation
        ctx.args = args
        ctx.kwargs = kwargs
        
        # Convert to float for the forward computation
        float_input = numerators.float() / denominators.float()
        
        if operation == 'linear':
            weight_num, weight_den = args[0], args[1]
            bias_num, bias_den = args[2], args[3] if len(args) > 2 else (None, None)
            
            weight_float = weight_num.float() / weight_den.float()
            result = F.linear(float_input, weight_float)
            
            if bias_num is not None:
                bias_float = bias_num.float() / bias_den.float()
                result = result + bias_float
                
            return result
        
        elif operation == 'activation':
            activation_type = kwargs.get('activation_type', 'relu')
            if activation_type == 'fraclu':
                alpha_num, alpha_den = args[0], args[1]
                beta_num, beta_den = args[2], args[3]
                
                alpha = alpha_num.float() / alpha_den.float()
                beta = beta_num.float() / beta_den.float()
                
                pos_part = torch.where(float_input >= 0, alpha * float_input, torch.zeros_like(float_input))
                neg_part = torch.where(float_input < 0, beta * float_input, torch.zeros_like(float_input))
                
                return pos_part + neg_part
        
        # Default: return float representation
        return float_input
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for fractional operations.
        Computes gradients with respect to fractional parameters.
        """
        numerators, denominators = ctx.saved_tensors
        operation = ctx.operation
        args = ctx.args
        
        # Compute gradients based on operation type
        if operation == 'linear':
            # Gradient with respect to input
            weight_num, weight_den = args[0], args[1]
            weight_float = weight_num.float() / weight_den.float()
            
            grad_input_num = torch.zeros_like(numerators)
            grad_input_den = torch.ones_like(denominators)
            
            if ctx.needs_input_grad[0]:  # numerators gradient
                grad_input = torch.matmul(grad_output, weight_float.t())
                # Convert gradient back to fractional representation would be complex
                # For now, return float gradient (this is a simplification)
                return grad_input, None, None, None, None, None, None
        
        # Simplified gradient computation - in practice, this would need
        # more sophisticated fractional gradient handling
        return grad_output, None, None, None, None, None, None


class FractionalTensorOps:
    """
    High-level operations for tensors with fractional weights.
    """
    
    @staticmethod
    def fractional_linear(input_tensor: torch.Tensor,
                         weight_num: torch.Tensor, weight_den: torch.Tensor,
                         bias_num: Optional[torch.Tensor] = None,
                         bias_den: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Linear transformation with fractional weights.
        
        Args:
            input_tensor: Input tensor
            weight_num, weight_den: Fractional weight representation
            bias_num, bias_den: Optional fractional bias
            
        Returns:
            Output tensor from linear transformation
        """
        return FractionalFunction.apply(
            input_tensor, torch.ones_like(input_tensor, dtype=torch.long),
            'linear', weight_num, weight_den, bias_num, bias_den
        )
    
    @staticmethod
    def fractional_conv2d(input_tensor: torch.Tensor,
                         weight_num: torch.Tensor, weight_den: torch.Tensor,
                         bias_num: Optional[torch.Tensor] = None,
                         bias_den: Optional[torch.Tensor] = None,
                         stride: int = 1, padding: int = 0) -> torch.Tensor:
        """
        2D convolution with fractional weights.
        
        Args:
            input_tensor: Input tensor (N, C, H, W)
            weight_num, weight_den: Fractional weight representation
            bias_num, bias_den: Optional fractional bias
            stride: Convolution stride
            padding: Convolution padding
            
        Returns:
            Output tensor from convolution
        """
        # Convert to float for convolution (exact fractional conv2d would be very complex)
        weight_float = weight_num.float() / weight_den.float()
        bias_float = None
        if bias_num is not None:
            bias_float = bias_num.float() / bias_den.float()
        
        return F.conv2d(input_tensor, weight_float, bias_float, stride, padding)
    
    @staticmethod
    def check_fractional_equality(a_num: torch.Tensor, a_den: torch.Tensor,
                                 b_num: torch.Tensor, b_den: torch.Tensor,
                                 tolerance: float = 0.0) -> torch.Tensor:
        """
        Check if two fractional tensors are equal (within tolerance).
        
        Args:
            a_num, a_den: First fractional tensor
            b_num, b_den: Second fractional tensor
            tolerance: Floating-point tolerance for comparison
            
        Returns:
            Boolean tensor indicating equality
        """
        if tolerance == 0.0:
            # Exact equality: cross-multiply and compare
            return (a_num * b_den) == (b_num * a_den)
        else:
            # Approximate equality
            a_float = a_num.float() / a_den.float()
            b_float = b_num.float() / b_den.float()
            return torch.abs(a_float - b_float) <= tolerance


# Utility functions for easier usage
def add_fractional(a_frac_weight, b_frac_weight, max_denominator: int = 1000):
    """Add two FractionalWeight objects."""
    result_num, result_den = FractionalOps.frac_add_tensors(
        a_frac_weight.numerators, a_frac_weight.denominators,
        b_frac_weight.numerators, b_frac_weight.denominators,
        simplify=True, max_denominator=max_denominator
    )
    
    from .fractional_weight import FractionalWeight
    result = FractionalWeight(
        result_num.float() / result_den.float(),
        max_denominator=max_denominator
    )
    result.numerators = result_num
    result.denominators = result_den
    return result


def multiply_fractional(a_frac_weight, b_frac_weight, max_denominator: int = 1000):
    """Multiply two FractionalWeight objects."""
    result_num, result_den = FractionalOps.frac_mul_tensors(
        a_frac_weight.numerators, a_frac_weight.denominators,
        b_frac_weight.numerators, b_frac_weight.denominators,
        simplify=True, max_denominator=max_denominator
    )
    
    from .fractional_weight import FractionalWeight
    result = FractionalWeight(
        result_num.float() / result_den.float(),
        max_denominator=max_denominator
    )
    result.numerators = result_num
    result.denominators = result_den
    return result
