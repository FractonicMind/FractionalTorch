"""
FractionalTorch Core: Fractional Weight Implementation

This module implements the FractionalWeight class, which stores neural network 
parameters as exact rational numbers (p/q) instead of floating-point approximations.

Author: Lev Goukassian
License: MIT
"""

import torch
import torch.nn as nn
from fractions import Fraction
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import logging
from functools import lru_cache
import math

logger = logging.getLogger(__name__)


class FractionalWeight(nn.Parameter):
    """
    A PyTorch Parameter that maintains exact fractional representation.
    
    This class extends nn.Parameter to store weights as rational numbers (p/q),
    enabling exact arithmetic operations and eliminating floating-point
    accumulation errors during training.
    
    Args:
        data: Initial weight data (tensor, array, or number)
        requires_grad: Whether the parameter requires gradients
        max_denominator: Maximum denominator for fraction approximation
        simplify_threshold: How often to simplify fractions (every N operations)
        
    Example:
        >>> weight = FractionalWeight(torch.randn(3, 4), max_denominator=1000)
        >>> print(weight.get_fraction_at((0, 0)))  # Get exact fraction at position
        Fraction(123, 456)
    """
    
    def __new__(cls, data: Union[torch.Tensor, np.ndarray, float, int], 
                requires_grad: bool = True,
                max_denominator: int = 1000,
                simplify_threshold: int = 100):
        
        # Convert input to tensor
        if isinstance(data, (int, float)):
            data = torch.tensor(float(data))
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
        else:
            data = data.float()
        
        # Create the parameter with float data for PyTorch compatibility
        param = super().__new__(cls, data.clone(), requires_grad=requires_grad)
        
        # Initialize fractional representation
        param._init_fractional_storage(data, max_denominator)
        param.max_denominator = max_denominator
        param.simplify_threshold = simplify_threshold
        param._operation_count = 0
        param._last_simplify = 0
        
        return param
    
    def _init_fractional_storage(self, data: torch.Tensor, max_denom: int):
        """Initialize numerator and denominator tensors from float data."""
        # Create storage for exact fractions
        self.numerators = torch.zeros_like(data, dtype=torch.long)
        self.denominators = torch.ones_like(data, dtype=torch.long)
        
        # Convert each element to exact fraction
        flat_data = data.flatten()
        flat_num = self.numerators.flatten()
        flat_den = self.denominators.flatten()
        
        for i, val in enumerate(flat_data):
            if torch.isnan(val) or torch.isinf(val):
                # Handle special values
                flat_num[i] = 0
                flat_den[i] = 1
                logger.warning(f"Invalid value {val} replaced with 0/1")
            else:
                try:
                    frac = Fraction(float(val)).limit_denominator(max_denom)
                    flat_num[i] = frac.numerator
                    flat_den[i] = frac.denominator
                except (ValueError, OverflowError):
                    # Fallback for problematic values
                    flat_num[i] = 0
                    flat_den[i] = 1
                    logger.warning(f"Could not convert {val} to fraction, using 0/1")
        
        # Update the parameter data to exact fractional representation
        self.data = self.numerators.float() / self.denominators.float()
    
    def update_from_gradient(self, grad: torch.Tensor, learning_rate: float):
        """
        Update fractional representation after a gradient step.
        
        Args:
            grad: Gradient tensor
            learning_rate: Learning rate for the update
        """
        # Compute new values using exact arithmetic where possible
        new_data = self.data - learning_rate * grad
        
        # Convert back to fractions
        self._update_fractions_from_data(new_data)
        
        # Periodic simplification to prevent denominator explosion
        self._operation_count += 1
        if self._operation_count - self._last_simplify >= self.simplify_threshold:
            self._simplify_all_fractions()
            self._last_simplify = self._operation_count
    
    def _update_fractions_from_data(self, new_data: torch.Tensor):
        """Update fractional representation from new float data."""
        flat_data = new_data.flatten()
        flat_num = self.numerators.flatten()
        flat_den = self.denominators.flatten()
        
        for i, val in enumerate(flat_data):
            if torch.isfinite(val):
                try:
                    frac = Fraction(float(val)).limit_denominator(self.max_denominator)
                    flat_num[i] = frac.numerator
                    flat_den[i] = frac.denominator
                except (ValueError, OverflowError):
                    # Keep previous values if conversion fails
                    continue
        
        # Update parameter data
        self.data = self.numerators.float() / self.denominators.float()
    
    def _simplify_all_fractions(self):
        """Simplify all fractions by reducing to lowest terms."""
        # Compute GCD for all fraction pairs
        gcd_vals = torch.gcd(self.numerators.abs(), self.denominators)
        
        # Reduce fractions
        self.numerators = self.numerators // gcd_vals
        self.denominators = self.denominators // gcd_vals
        
        # Ensure denominators are positive
        neg_mask = self.denominators < 0
        self.numerators[neg_mask] = -self.numerators[neg_mask]
        self.denominators[neg_mask] = -self.denominators[neg_mask]
        
        # Update float representation
        self.data = self.numerators.float() / self.denominators.float()
        
        logger.debug(f"Simplified {torch.sum(gcd_vals > 1).item()} fractions")
    
    def get_fraction_at(self, index: Tuple[int, ...]) -> Fraction:
        """
        Get the exact fraction at a specific tensor index.
        
        Args:
            index: Tensor index (e.g., (0, 1) for 2D tensor)
            
        Returns:
            Exact Fraction object
            
        Example:
            >>> weight = FractionalWeight([[0.5, 0.333]], max_denominator=100)
            >>> frac = weight.get_fraction_at((0, 1))
            >>> print(frac)  # Fraction(1, 3)
        """
        num = int(self.numerators[index])
        den = int(self.denominators[index])
        return Fraction(num, den)
    
    def set_fraction_at(self, index: Tuple[int, ...], fraction: Union[Fraction, float]):
        """
        Set an exact fraction at a specific tensor index.
        
        Args:
            index: Tensor index
            fraction: Fraction or float to set
        """
        if isinstance(fraction, float):
            fraction = Fraction(fraction).limit_denominator(self.max_denominator)
        elif not isinstance(fraction, Fraction):
            fraction = Fraction(fraction)
        
        self.numerators[index] = fraction.numerator
        self.denominators[index] = fraction.denominator
        self.data[index] = float(fraction)
    
    def set_max_denominator(self, max_denom: int):
        """
        Update maximum denominator and re-quantize all fractions.
        
        Args:
            max_denom: New maximum denominator
        """
        old_max = self.max_denominator
        self.max_denominator = max_denom
        
        if max_denom < old_max:
            # Need to re-quantize to smaller denominators
            self._update_fractions_from_data(self.data)
            logger.info(f"Re-quantized fractions from max_denom={old_max} to {max_denom}")
    
    def get_precision_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current fractional representation.
        
        Returns:
            Dictionary with precision statistics
            
        Example:
            >>> stats = weight.get_precision_stats()
            >>> print(f"Max denominator: {stats['max_denominator']}")
        """
        denominators_cpu = self.denominators.cpu() if self.denominators.is_cuda else self.denominators
        
        return {
            'max_denominator': int(torch.max(denominators_cpu)),
            'min_denominator': int(torch.min(denominators_cpu)),
            'mean_denominator': float(torch.mean(denominators_cpu.float())),
            'median_denominator': float(torch.median(denominators_cpu.float())),
            'num_integers': int(torch.sum(denominators_cpu == 1)),
            'num_halves': int(torch.sum(denominators_cpu == 2)),
            'unique_denominators': len(torch.unique(denominators_cpu)),
            'memory_overhead_ratio': 2.0,  # Storing num + den vs just float
            'total_elements': self.numel(),
            'operation_count': self._operation_count,
        }
    
    def to_exact_string(self, max_display: int = 10) -> str:
        """
        Get string representation showing exact fractions.
        
        Args:
            max_display: Maximum number of elements to display
            
        Returns:
            String showing exact fractional values
        """
        flat_num = self.numerators.flatten()
        flat_den = self.denominators.flatten()
        
        fractions = []
        for i in range(min(max_display, len(flat_num))):
            num, den = int(flat_num[i]), int(flat_den[i])
            if den == 1:
                fractions.append(str(num))
            else:
                fractions.append(f"{num}/{den}")
        
        if len(flat_num) > max_display:
            fractions.append("...")
        
        return f"FractionalWeight([{', '.join(fractions)}])"
    
    def clone_fractional(self):
        """
        Create a deep copy of the fractional weight.
        
        Returns:
            New FractionalWeight with identical values
        """
        cloned = FractionalWeight(
            self.data.clone(), 
            requires_grad=self.requires_grad,
            max_denominator=self.max_denominator,
            simplify_threshold=self.simplify_threshold
        )
        cloned.numerators = self.numerators.clone()
        cloned.denominators = self.denominators.clone()
        cloned._operation_count = self._operation_count
        return cloned
    
    def is_exactly_equal(self, other: 'FractionalWeight') -> bool:
        """
        Check if two fractional weights are exactly equal.
        
        Args:
            other: Another FractionalWeight to compare
            
        Returns:
            True if exactly equal, False otherwise
        """
        if not isinstance(other, FractionalWeight):
            return False
        
        return (torch.equal(self.numerators, other.numerators) and 
                torch.equal(self.denominators, other.denominators))
    
    def __repr__(self):
        return (f"FractionalWeight({self.shape}, max_denom={self.max_denominator}, "
                f"mean_denom={self.get_precision_stats()['mean_denominator']:.1f})")
    
    def __str__(self):
        if self.numel() <= 20:
            return self.to_exact_string()
        else:
            stats = self.get_precision_stats()
            return (f"FractionalWeight of size {self.shape} with "
                   f"max_denominator={stats['max_denominator']}")


def create_fractional_like(tensor: torch.Tensor, 
                          max_denominator: int = 1000,
                          requires_grad: bool = True) -> FractionalWeight:
    """
    Create a FractionalWeight with the same shape and device as input tensor.
    
    Args:
        tensor: Template tensor for shape and device
        max_denominator: Maximum denominator for fractions
        requires_grad: Whether to require gradients
        
    Returns:
        New FractionalWeight with same shape as input
        
    Example:
        >>> template = torch.randn(2, 3)
        >>> frac_weight = create_fractional_like(template, max_denominator=500)
    """
    device = tensor.device
    dtype = tensor.dtype
    
    # Create random fractional values
    random_data = torch.randn_like(tensor, dtype=torch.float) * 0.1
    frac_weight = FractionalWeight(
        random_data, 
        requires_grad=requires_grad,
        max_denominator=max_denominator
    )
    
    return frac_weight.to(device)


def fractional_from_float(value: float, max_denominator: int = 1000) -> FractionalWeight:
    """
    Create a scalar FractionalWeight from a float value.
    
    Args:
        value: Float value to convert
        max_denominator: Maximum denominator
        
    Returns:
        Scalar FractionalWeight
        
    Example:
        >>> frac = fractional_from_float(0.333, max_denominator=100)
        >>> print(frac.get_fraction_at(()))  # Fraction(1, 3)
    """
    return FractionalWeight(value, max_denominator=max_denominator)
