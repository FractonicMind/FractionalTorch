"""
FractionalTorch Modules: Fractional Linear Unit (FracLU)

This module implements the FracLU activation function with learnable fractional slopes,
providing adaptive activation behavior while maintaining exact arithmetic properties.

Author: Lev Goukassian
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from typing import Optional, Union, Tuple, List
import logging
from fractions import Fraction

from ..core import FractionalWeight

logger = logging.getLogger(__name__)


class FracLU(nn.Module):
    """
    Fractional Linear Unit activation function with learnable rational slopes.
    
    FracLU generalizes ReLU by introducing learnable fractional slopes for both
    positive and negative input regions:
    
    FracLU(x) = max(α₁/β₁ * x, α₂/β₂ * x)
    
    where α₁, β₁, α₂, β₂ are learnable integer parameters that form exact
    rational slopes during training.
    
    Args:
        num_features: Number of features (channels) for the activation
        init_pos_slope: Initial positive slope (default: 1.0)
        init_neg_slope: Initial negative slope (default: 0.01, like LeakyReLU)
        max_denominator: Maximum denominator for fractional slopes
        learnable: Whether slopes should be learnable parameters
        clamp_denominators: Whether to clamp denominators to stay positive
        slope_init_strategy: Strategy for initializing slopes ('fixed', 'random', 'adaptive')
        
    Shape:
        - Input: (N, num_features) or (N, num_features, *) where * means any number of dimensions
        - Output: Same shape as input
        
    Examples:
        >>> # Basic FracLU with learnable slopes
        >>> fraclu = FracLU(128)
        >>> output = fraclu(input_tensor)
        
        >>> # Fixed slopes (non-learnable)
        >>> fixed_fraclu = FracLU(64, init_pos_slope=1.0, init_neg_slope=0.1, learnable=False)
        
        >>> # Custom initialization
        >>> custom_fraclu = FracLU(256, init_pos_slope=0.8, init_neg_slope=0.2)
    """
    
    def __init__(self, 
                 num_features: int,
                 init_pos_slope: float = 1.0,
                 init_neg_slope: float = 0.01,
                 max_denominator: int = 1000,
                 learnable: bool = True,
                 clamp_denominators: bool = True,
                 slope_init_strategy: str = 'fixed',
                 channel_wise: bool = True):
        
        super(FracLU, self).__init__()
        
        self.num_features = num_features
        self.max_denominator = max_denominator
        self.learnable = learnable
        self.clamp_denominators = clamp_denominators
        self.slope_init_strategy = slope_init_strategy
        self.channel_wise = channel_wise
        
        # Convert initial slopes to fractions
        try:
            pos_frac = Fraction(init_pos_slope).limit_denominator(max_denominator)
            neg_frac = Fraction(init_neg_slope).limit_denominator(max_denominator)
        except (ValueError, OverflowError):
            logger.warning(f"Could not convert slopes to fractions, using defaults")
            pos_frac = Fraction(1, 1)
            neg_frac = Fraction(1, 100)
        
        # Determine parameter shape
        if channel_wise:
            param_shape = (num_features,)
        else:
            param_shape = (1,)
        
        # Initialize fractional slope parameters
        if learnable:
            self.pos_alpha = Parameter(torch.full(param_shape, float(pos_frac.numerator)))
            self.pos_beta = Parameter(torch.full(param_shape, float(pos_frac.denominator)))
            self.neg_alpha = Parameter(torch.full(param_shape, float(neg_frac.numerator)))
            self.neg_beta = Parameter(torch.full(param_shape, float(neg_frac.denominator)))
        else:
            self.register_buffer('pos_alpha', torch.full(param_shape, float(pos_frac.numerator)))
            self.register_buffer('pos_beta', torch.full(param_shape, float(pos_frac.denominator)))
            self.register_buffer('neg_alpha', torch.full(param_shape, float(neg_frac.numerator)))
            self.register_buffer('neg_beta', torch.full(param_shape, float(neg_frac.denominator)))
        
        # Initialize parameters based on strategy
        self._initialize_slopes()
        
        # Ensure denominators start positive
        if clamp_denominators:
            with torch.no_grad():
                self.pos_beta.clamp_(min=1.0)
                self.neg_beta.clamp_(min=1.0)
        
        # Track activations for analysis
        self._forward_count = 0
        self._activation_stats = {
            'positive_activations': 0,
            'negative_activations': 0,
            'zero_activations': 0
        }
        
        logger.debug(f"FracLU created: {num_features} features, "
                    f"learnable={learnable}, channel_wise={channel_wise}")
    
    def _initialize_slopes(self):
        """Initialize slope parameters based on the chosen strategy."""
        if not self.learnable:
            return  # Skip initialization for non-learnable parameters
        
        with torch.no_grad():
            if self.slope_init_strategy == 'fixed':
                # Keep initial values (already set in __init__)
                pass
            
            elif self.slope_init_strategy == 'random':
                # Random initialization around default values
                pos_noise = torch.randn_like(self.pos_alpha) * 0.1
                neg_noise = torch.randn_like(self.neg_alpha) * 0.01
                
                self.pos_alpha.add_(pos_noise)
                self.neg_alpha.add_(neg_noise)
                
                # Ensure reasonable bounds
                self.pos_alpha.clamp_(0.1, 10.0)
                self.neg_alpha.clamp_(-2.0, 2.0)
            
            elif self.slope_init_strategy == 'adaptive':
                # Initialize based on feature index for diversity
                if self.channel_wise:
                    feature_indices = torch.arange(self.num_features, dtype=torch.float)
                    
                    # Positive slopes: slight variation around 1.0
                    pos_variation = 0.1 * torch.sin(feature_indices * 0.1)
                    self.pos_alpha.copy_(1.0 + pos_variation)
                    
                    # Negative slopes: variation in leakiness
                    neg_variation = 0.05 * torch.cos(feature_indices * 0.15) + 0.01
                    self.neg_alpha.copy_(neg_variation)
                    
                    # Keep denominators at 1 initially for simplicity
                    self.pos_beta.fill_(1.0)
                    self.neg_beta.fill_(1.0)
            
            else:
                raise ValueError(f"Unknown slope initialization strategy: {self.slope_init_strategy}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FracLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor with same shape as input
        """
        self._forward_count += 1
        
        # Ensure denominators stay positive
        if self.clamp_denominators:
            pos_beta_clamped = torch.clamp(self.pos_beta, min=1.0)
            neg_beta_clamped = torch.clamp(self.neg_beta, min=1.0)
        else:
            pos_beta_clamped = self.pos_beta
            neg_beta_clamped = self.neg_beta
        
        # Compute fractional slopes
        pos_slope = self.pos_alpha / pos_beta_clamped
        neg_slope = self.neg_alpha / neg_beta_clamped
        
        # Apply fractional linear activation
        # Reshape slopes for broadcasting if needed
        if self.channel_wise and x.dim() > 2:
            # For inputs like (N, C, H, W), expand slopes to (1, C, 1, 1)
            shape = [1] * x.dim()
            shape[1] = self.num_features  # Assume channel dimension is 1
            pos_slope = pos_slope.view(shape)
            neg_slope = neg_slope.view(shape)
        elif self.channel_wise and x.dim() == 2:
            # For inputs like (N, C), slopes are already correct shape
            pass
        
        # Compute positive and negative parts
        pos_part = torch.where(x >= 0, pos_slope * x, torch.zeros_like(x))
        neg_part = torch.where(x < 0, neg_slope * x, torch.zeros_like(x))
        
        output = pos_part + neg_part
        
        # Update activation statistics
        if self._forward_count % 100 == 0:  # Sample statistics periodically
            self._update_activation_stats(x)
        
        return output
    
    def _update_activation_stats(self, x: torch.Tensor):
        """Update statistics about activation patterns."""
        with torch.no_grad():
            positive_count = torch.sum(x > 0).item()
            negative_count = torch.sum(x < 0).item()
            zero_count = torch.sum(x == 0).item()
            
            self._activation_stats['positive_activations'] += positive_count
            self._activation_stats['negative_activations'] += negative_count
            self._activation_stats['zero_activations'] += zero_count
    
    def get_current_slopes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current positive and negative slopes as float tensors.
        
        Returns:
            Tuple of (positive_slopes, negative_slopes)
        """
        with torch.no_grad():
            pos_beta_safe = torch.clamp(self.pos_beta, min=1.0) if self.clamp_denominators else self.pos_beta
            neg_beta_safe = torch.clamp(self.neg_beta, min=1.0) if self.clamp_denominators else self.neg_beta
            
            pos_slopes = self.pos_alpha / pos_beta_safe
            neg_slopes = self.neg_alpha / neg_beta_safe
            
            return pos_slopes.clone(), neg_slopes.clone()
    
    def get_fractional_slopes(self) -> List[Tuple[Fraction, Fraction]]:
        """
        Get current slopes as exact Fraction objects.
        
        Returns:
            List of (positive_slope_fraction, negative_slope_fraction) for each feature
        """
        slopes = []
        
        for i in range(self.num_features if self.channel_wise else 1):
            try:
                pos_frac = Fraction(int(self.pos_alpha[i]), int(self.pos_beta[i]))
                neg_frac = Fraction(int(self.neg_alpha[i]), int(self.neg_beta[i]))
                slopes.append((pos_frac, neg_frac))
            except (ValueError, ZeroDivisionError):
                # Fallback for problematic values
                slopes.append((Fraction(1, 1), Fraction(1, 100)))
        
        return slopes
    
    def set_slopes(self, pos_slope: float, neg_slope: float, feature_idx: Optional[int] = None):
        """
        Set slopes for specific features or all features.
        
        Args:
            pos_slope: Positive slope value
            neg_slope: Negative slope value  
            feature_idx: Feature index to set (None for all features)
        """
        # Convert to fractions
        try:
            pos_frac = Fraction(pos_slope).limit_denominator(self.max_denominator)
            neg_frac = Fraction(neg_slope).limit_denominator(self.max_denominator)
        except (ValueError, OverflowError):
            logger.error(f"Could not convert slopes to fractions: {pos_slope}, {neg_slope}")
            return
        
        with torch.no_grad():
            if feature_idx is None:
                # Set all features
                self.pos_alpha.fill_(float(pos_frac.numerator))
                self.pos_beta.fill_(float(pos_frac.denominator))
                self.neg_alpha.fill_(float(neg_frac.numerator))
                self.neg_beta.fill_(float(neg_frac.denominator))
            else:
                # Set specific feature
                if 0 <= feature_idx < self.num_features:
                    self.pos_alpha[feature_idx] = float(pos_frac.numerator)
                    self.pos_beta[feature_idx] = float(pos_frac.denominator)
                    self.neg_alpha[feature_idx] = float(neg_frac.numerator)
                    self.neg_beta[feature_idx] = float(neg_frac.denominator)
                else:
                    logger.error(f"Invalid feature index: {feature_idx}")
    
    def get_activation_stats(self) -> dict:
        """
        Get statistics about activation patterns.
        
        Returns:
            Dictionary with activation statistics
        """
        total_activations = sum(self._activation_stats.values())
        
        if total_activations == 0:
            return self._activation_stats.copy()
        
        stats = self._activation_stats.copy()
        stats['positive_percentage'] = stats['positive_activations'] / total_activations * 100
        stats['negative_percentage'] = stats['negative_activations'] / total_activations * 100
        stats['zero_percentage'] = stats['zero_activations'] / total_activations * 100
        stats['total_activations'] = total_activations
        stats['forward_count'] = self._forward_count
        
        return stats
    
    def reset_stats(self):
        """Reset activation statistics."""
        self._activation_stats = {
            'positive_activations': 0,
            'negative_activations': 0, 
            'zero_activations': 0
        }
        self._forward_count = 0
    
    def get_parameter_info(self) -> dict:
        """
        Get detailed information about the fractional parameters.
        
        Returns:
            Dictionary with parameter information
        """
        pos_slopes, neg_slopes = self.get_current_slopes()
        fractional_slopes = self.get_fractional_slopes()
        
        return {
            'num_features': self.num_features,
            'learnable': self.learnable,
            'channel_wise': self.channel_wise,
            'max_denominator': self.max_denominator,
            'current_pos_slopes': pos_slopes.tolist(),
            'current_neg_slopes': neg_slopes.tolist(),
            'fractional_slopes': [(str(pos), str(neg)) for pos, neg in fractional_slopes],
            'pos_alpha_range': (float(torch.min(self.pos_alpha)), float(torch.max(self.pos_alpha))),
            'pos_beta_range': (float(torch.min(self.pos_beta)), float(torch.max(self.pos_beta))),
            'neg_alpha_range': (float(torch.min(self.neg_alpha)), float(torch.max(self.neg_alpha))),
            'neg_beta_range': (float(torch.min(self.neg_beta)), float(torch.max(self.neg_beta))),
        }
    
    def simplify_slopes(self):
        """Simplify all fractional slopes to their lowest terms."""
        if not self.learnable:
            return
        
        with torch.no_grad():
            for i in range(self.num_features if self.channel_wise else 1):
                # Simplify positive slope
                try:
                    pos_frac = Fraction(int(self.pos_alpha[i]), int(self.pos_beta[i]))
                    self.pos_alpha[i] = float(pos_frac.numerator)
                    self.pos_beta[i] = float(pos_frac.denominator)
                except (ValueError, ZeroDivisionError):
                    continue
                
                # Simplify negative slope
                try:
                    neg_frac = Fraction(int(self.neg_alpha[i]), int(self.neg_beta[i]))
                    self.neg_alpha[i] = float(neg_frac.numerator)
                    self.neg_beta[i] = float(neg_frac.denominator)
                except (ValueError, ZeroDivisionError):
                    continue
        
        logger.debug("Simplified all fractional slopes")
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (f'num_features={self.num_features}, learnable={self.learnable}, '
                f'channel_wise={self.channel_wise}, max_denominator={self.max_denominator}')
    
    def __repr__(self):
        return f"FracLU({self.num_features})"


class AdaptiveFracLU(FracLU):
    """
    Adaptive FracLU that automatically adjusts slopes based on activation statistics.
    
    This variant monitors activation patterns and adapts slopes to maintain
    a balanced activation distribution between positive and negative regions.
    
    Args:
        num_features: Number of features for the activation
        target_pos_ratio: Target ratio of positive activations (0.0 to 1.0)
        adaptation_rate: Rate at which slopes adapt (smaller = slower)
        adaptation_interval: How often to adapt slopes (in forward passes)
        **kwargs: Additional arguments passed to FracLU
    """
    
    def __init__(self,
                 num_features: int,
                 target_pos_ratio: float = 0.6,
                 adaptation_rate: float = 0.01,
                 adaptation_interval: int = 1000,
                 **kwargs):
        
        super().__init__(num_features, **kwargs)
        
        self.target_pos_ratio = target_pos_ratio
        self.adaptation_rate = adaptation_rate
        self.adaptation_interval = adaptation_interval
        
        # Track recent activation ratios
        self._recent_pos_ratios = []
        self._adaptation_count = 0
        
        logger.debug(f"AdaptiveFracLU created: target_ratio={target_pos_ratio}, "
                    f"adaptation_rate={adaptation_rate}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive slope adjustment."""
        output = super().forward(x)
        
        # Adapt slopes periodically
        if self._forward_count % self.adaptation_interval == 0 and self.learnable:
            self._adapt_slopes(x)
        
        return output
    
    def _adapt_slopes(self, x: torch.Tensor):
        """Adapt slopes based on current activation patterns."""
        with torch.no_grad():
            # Calculate current positive activation ratio
            total_elements = x.numel()
            positive_elements = torch.sum(x > 0).item()
            current_pos_ratio = positive_elements / total_elements if total_elements > 0 else 0.5
            
            self._recent_pos_ratios.append(current_pos_ratio)
            if len(self._recent_pos_ratios) > 10:
                self._recent_pos_ratios.pop(0)
            
            # Calculate average recent ratio
            avg_pos_ratio = np.mean(self._recent_pos_ratios)
            
            # Adjust negative slope to influence positive/negative balance
            ratio_error = self.target_pos_ratio - avg_pos_ratio
            
            if abs(ratio_error) > 0.05:  # Only adapt if error is significant
                # Increase negative slope magnitude if too few negative activations
                # Decrease negative slope magnitude if too many negative activations
                adjustment = self.adaptation_rate * ratio_error
                
                # Apply adjustment to negative slopes
                self.neg_alpha.add_(adjustment * self.neg_alpha.abs())
                self.neg_alpha.clamp_(-5.0, 5.0)  # Prevent extreme values
                
                self._adaptation_count += 1
                
                if self._adaptation_count % 10 == 0:
                    logger.debug(f"Adapted slopes: pos_ratio={avg_pos_ratio:.3f}, "
                               f"target={self.target_pos_ratio:.3f}, "
                               f"adjustment={adjustment:.4f}")


def create_fraclu_stack(num_features_list: List[int], 
                       init_pos_slope: float = 1.0,
                       init_neg_slope: float = 0.01,
                       **kwargs) -> nn.ModuleList:
    """
    Create a stack of FracLU activations for different feature dimensions.
    
    Args:
        num_features_list: List of feature dimensions
        init_pos_slope: Initial positive slope for all layers
        init_neg_slope: Initial negative slope for all layers
        **kwargs: Additional arguments for FracLU
        
    Returns:
        ModuleList containing FracLU activations
    """
    activations = nn.ModuleList()
    
    for num_features in num_features_list:
        activation = FracLU(
            num_features=num_features,
            init_pos_slope=init_pos_slope,
            init_neg_slope=init_neg_slope,
            **kwargs
        )
        activations.append(activation)
    
    return activations


def convert_relu_to_fraclu(module: nn.Module, 
                          num_features: int,
                          inplace: bool = True,
                          **fraclu_kwargs) -> nn.Module:
    """
    Convert ReLU activations in a module to FracLU.
    
    Args:
        module: Module containing ReLU activations
        num_features: Number of features for FracLU
        inplace: Whether to modify the module in place
        **fraclu_kwargs: Additional arguments for FracLU
        
    Returns:
        Module with ReLU replaced by FracLU
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)
    
    converted_count = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.ReLU, nn.LeakyReLU)):
            # Determine initial negative slope
            if isinstance(child, nn.LeakyReLU):
                init_neg_slope = child.negative_slope
            else:
                init_neg_slope = 0.0  # ReLU has no negative slope
            
            fraclu = FracLU(
                num_features=num_features,
                init_pos_slope=1.0,
                init_neg_slope=init_neg_slope,
                **fraclu_kwargs
            )
            
            setattr(module, name, fraclu)
            converted_count += 1
        else:
            # Recursively process child modules
            convert_relu_to_fraclu(child, num_features, inplace=True, **fraclu_kwargs)
    
    logger.info(f"Converted {converted_count} ReLU/LeakyReLU activations to FracLU")
    
    return module
