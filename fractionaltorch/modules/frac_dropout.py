"""
FractionalTorch Modules: Fractional Dropout

This module implements fractional dropout with learnable rational dropout rates,
providing adaptive regularization while maintaining exact arithmetic properties.

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


class FracDropout(nn.Module):
    """
    Fractional Dropout layer with learnable rational dropout rates.
    
    Unlike standard dropout with fixed rates, FracDropout learns optimal
    fractional dropout rates α/β during training, enabling more precise
    regularization control and adaptive behavior based on training dynamics.
    
    The dropout rate is represented as an exact fraction p = α/β where α and β
    are learnable integer parameters, ensuring exact reproducibility.
    
    Args:
        initial_rate: Initial dropout rate (0.0 to 1.0)
        learnable: Whether the dropout rate should be learnable
        max_denominator: Maximum denominator for fractional representation
        channel_wise: Whether to have different rates per channel
        num_features: Number of features (required if channel_wise=True)
        rate_bounds: Tuple of (min_rate, max_rate) to constrain learning
        adaptation_strategy: Strategy for rate adaptation ('none', 'loss_based', 'gradient_based')
        
    Shape:
        - Input: (N, *) where * means any number of dimensions
        - Output: Same shape as input
        
    Examples:
        >>> # Basic learnable fractional dropout
        >>> frac_dropout = FracDropout(0.5, learnable=True)
        >>> output = frac_dropout(input_tensor)
        
        >>> # Channel-wise dropout rates
        >>> channel_dropout = FracDropout(0.3, learnable=True, channel_wise=True, num_features=128)
        
        >>> # Fixed fractional rate (non-learnable)
        >>> fixed_dropout = FracDropout(0.25, learnable=False)
    """
    
    def __init__(self,
                 initial_rate: float = 0.5,
                 learnable: bool = True,
                 max_denominator: int = 1000,
                 channel_wise: bool = False,
                 num_features: Optional[int] = None,
                 rate_bounds: Tuple[float, float] = (0.0, 0.95),
                 adaptation_strategy: str = 'none',
                 inplace: bool = False):
        
        super(FracDropout, self).__init__()
        
        # Validate inputs
        if not 0.0 <= initial_rate <= 1.0:
            raise ValueError(f"Initial rate must be between 0.0 and 1.0, got {initial_rate}")
        
        if channel_wise and num_features is None:
            raise ValueError("num_features must be specified when channel_wise=True")
        
        if not 0.0 <= rate_bounds[0] <= rate_bounds[1] <= 1.0:
            raise ValueError(f"Invalid rate bounds: {rate_bounds}")
        
        self.learnable = learnable
        self.max_denominator = max_denominator
        self.channel_wise = channel_wise
        self.num_features = num_features or 1
        self.rate_bounds = rate_bounds
        self.adaptation_strategy = adaptation_strategy
        self.inplace = inplace
        
        # Convert initial rate to fraction
        try:
            initial_frac = Fraction(initial_rate).limit_denominator(max_denominator)
        except (ValueError, OverflowError):
            logger.warning(f"Could not convert initial rate to fraction, using 1/2")
            initial_frac = Fraction(1, 2)
        
        # Determine parameter shape
        if channel_wise:
            param_shape = (num_features,)
        else:
            param_shape = (1,)
        
        # Initialize fractional parameters
        if learnable:
            self.alpha = Parameter(torch.full(param_shape, float(initial_frac.numerator)))
            self.beta = Parameter(torch.full(param_shape, float(initial_frac.denominator)))
        else:
            self.register_buffer('alpha', torch.full(param_shape, float(initial_frac.numerator)))
            self.register_buffer('beta', torch.full(param_shape, float(initial_frac.denominator)))
        
        # Ensure beta starts positive and non-zero
        with torch.no_grad():
            self.beta.clamp_(min=1.0)
        
        # Initialize adaptation strategy
        self._setup_adaptation_strategy()
        
        # Statistics tracking
        self._forward_count = 0
        self._dropout_stats = {
            'total_elements': 0,
            'dropped_elements': 0,
            'effective_rates': []
        }
        
        logger.debug(f"FracDropout created: rate={initial_rate}, learnable={learnable}, "
                    f"channel_wise={channel_wise}, strategy={adaptation_strategy}")
    
    def _setup_adaptation_strategy(self):
        """Setup adaptation strategy-specific components."""
        if self.adaptation_strategy == 'loss_based':
            self.register_buffer('loss_history', torch.tensor([]))
            self.register_buffer('rate_history', torch.tensor([]))
            self.adaptation_window = 100
            self.adaptation_threshold = 0.01
            
        elif self.adaptation_strategy == 'gradient_based':
            self.register_buffer('grad_norm_history', torch.tensor([]))
            self.adaptation_target_grad_norm = 1.0
            self.adaptation_sensitivity = 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fractional dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with dropout applied (if training)
        """
        if not self.training:
            return x
        
        self._forward_count += 1
        
        # Compute current dropout rates
        dropout_rates = self.get_current_rates()
        
        # Apply dropout
        if self.channel_wise:
            output = self._apply_channel_wise_dropout(x, dropout_rates)
        else:
            output = self._apply_global_dropout(x, dropout_rates[0])
        
        # Update statistics
        self._update_stats(x, output, dropout_rates)
        
        # Apply adaptation strategy
        if self.learnable and self._forward_count % 100 == 0:
            self._apply_adaptation_strategy()
        
        return output
    
    def _apply_global_dropout(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        """Apply dropout with a single rate across all elements."""
        if rate <= 0.0:
            return x
        elif rate >= 1.0:
            return torch.zeros_like(x)
        
        # Generate dropout mask
        keep_prob = 1.0 - rate
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        
        # Apply mask with proper scaling
        if self.inplace:
            x.mul_(mask).div_(keep_prob)
            return x
        else:
            return x * mask / keep_prob
    
    def _apply_channel_wise_dropout(self, x: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        """Apply different dropout rates per channel."""
        if x.dim() < 2:
            raise ValueError("Channel-wise dropout requires at least 2D input")
        
        # Assume channel dimension is 1 (N, C, ...)
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        
        if num_channels != len(rates):
            raise ValueError(f"Number of channels ({num_channels}) doesn't match rates ({len(rates)})")
        
        output = x.clone() if not self.inplace else x
        
        for c in range(num_channels):
            rate = float(rates[c])
            
            if rate <= 0.0:
                continue
            elif rate >= 1.0:
                output[:, c] = 0.0
                continue
            
            # Generate mask for this channel
            keep_prob = 1.0 - rate
            channel_shape = output[:, c].shape
            mask = torch.bernoulli(torch.full(channel_shape, keep_prob, device=x.device))
            
            # Apply mask
            output[:, c] = output[:, c] * mask / keep_prob
        
        return output
    
    def get_current_rates(self) -> torch.Tensor:
        """
        Get current dropout rates as float tensor.
        
        Returns:
            Tensor of current dropout rates
        """
        with torch.no_grad():
            # Ensure beta > 0
            beta_safe = torch.clamp(self.beta, min=1.0)
            
            # Compute rates and clamp to bounds
            rates = self.alpha / beta_safe
            rates = torch.clamp(rates, min=self.rate_bounds[0], max=self.rate_bounds[1])
            
            return rates
    
    def get_fractional_rates(self) -> List[Fraction]:
        """
        Get current dropout rates as exact Fraction objects.
        
        Returns:
            List of Fraction objects representing exact dropout rates
        """
        rates = []
        
        num_rates = self.num_features if self.channel_wise else 1
        
        for i in range(num_rates):
            try:
                # Get integer values
                alpha_val = int(self.alpha[i])
                beta_val = max(1, int(self.beta[i]))  # Ensure beta >= 1
                
                # Create and simplify fraction
                frac = Fraction(alpha_val, beta_val)
                
                # Clamp to bounds
                frac_float = float(frac)
                if frac_float < self.rate_bounds[0]:
                    frac = Fraction(self.rate_bounds[0]).limit_denominator(self.max_denominator)
                elif frac_float > self.rate_bounds[1]:
                    frac = Fraction(self.rate_bounds[1]).limit_denominator(self.max_denominator)
                
                rates.append(frac)
                
            except (ValueError, ZeroDivisionError):
                # Fallback for problematic values
                rates.append(Fraction(1, 2))
        
        return rates
    
    def set_rate(self, rate: float, channel_idx: Optional[int] = None):
        """
        Set dropout rate for specific channel or all channels.
        
        Args:
            rate: New dropout rate (0.0 to 1.0)
            channel_idx: Channel index to set (None for all channels)
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Rate must be between 0.0 and 1.0, got {rate}")
        
        # Convert to fraction
        try:
            frac = Fraction(rate).limit_denominator(self.max_denominator)
        except (ValueError, OverflowError):
            logger.error(f"Could not convert rate to fraction: {rate}")
            return
        
        with torch.no_grad():
            if channel_idx is None:
                # Set all channels
                self.alpha.fill_(float(frac.numerator))
                self.beta.fill_(float(frac.denominator))
            else:
                # Set specific channel
                if self.channel_wise and 0 <= channel_idx < self.num_features:
                    self.alpha[channel_idx] = float(frac.numerator)
                    self.beta[channel_idx] = float(frac.denominator)
                else:
                    logger.error(f"Invalid channel index: {channel_idx}")
    
    def _update_stats(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, rates: torch.Tensor):
        """Update dropout statistics."""
        if not self.training:
            return
        
        with torch.no_grad():
            total_elements = input_tensor.numel()
            
            # Estimate dropped elements (this is approximate due to randomness)
            if self.channel_wise:
                avg_rate = torch.mean(rates).item()
            else:
                avg_rate = rates[0].item()
            
            estimated_dropped = int(total_elements * avg_rate)
            
            self._dropout_stats['total_elements'] += total_elements
            self._dropout_stats['dropped_elements'] += estimated_dropped
            self._dropout_stats['effective_rates'].append(avg_rate)
            
            # Keep only recent rates
            if len(self._dropout_stats['effective_rates']) > 1000:
                self._dropout_stats['effective_rates'] = self._dropout_stats['effective_rates'][-1000:]
    
    def _apply_adaptation_strategy(self):
        """Apply the selected adaptation strategy."""
        if not self.learnable:
            return
        
        if self.adaptation_strategy == 'loss_based':
            self._adapt_based_on_loss()
        elif self.adaptation_strategy == 'gradient_based':
            self._adapt_based_on_gradients()
    
    def _adapt_based_on_loss(self):
        """Adapt dropout rates based on loss trends."""
        # This would typically require access to current loss
        # For now, we implement a placeholder that could be extended
        current_rates = self.get_current_rates()
        
        # Simple adaptation: if rates are too extreme, pull toward center
        with torch.no_grad():
            center_rate = 0.5
            adaptation_strength = 0.01
            
            rate_adjustment = (center_rate - current_rates) * adaptation_strength
            
            # Convert adjustment back to fractional parameters
            for i in range(len(self.alpha)):
                current_rate = current_rates[i].item()
                new_rate = current_rate + rate_adjustment[i].item()
                new_rate = np.clip(new_rate, self.rate_bounds[0], self.rate_bounds[1])
                
                try:
                    new_frac = Fraction(new_rate).limit_denominator(self.max_denominator)
                    self.alpha[i] = float(new_frac.numerator)
                    self.beta[i] = float(new_frac.denominator)
                except:
                    continue
    
    def _adapt_based_on_gradients(self):
        """Adapt dropout rates based on gradient norms."""
        # This is a placeholder for gradient-based adaptation
        # In practice, this would monitor gradient norms and adjust rates accordingly
        pass
    
    def get_dropout_stats(self) -> dict:
        """
        Get comprehensive dropout statistics.
        
        Returns:
            Dictionary with dropout statistics
        """
        total_elements = self._dropout_stats['total_elements']
        dropped_elements = self._dropout_stats['dropped_elements']
        
        stats = {
            'total_elements': total_elements,
            'dropped_elements': dropped_elements,
            'overall_drop_rate': dropped_elements / total_elements if total_elements > 0 else 0.0,
            'forward_count': self._forward_count,
            'current_rates': self.get_current_rates().tolist(),
            'fractional_rates': [str(f) for f in self.get_fractional_rates()],
            'learnable': self.learnable,
            'channel_wise': self.channel_wise,
            'adaptation_strategy': self.adaptation_strategy
        }
        
        if self._dropout_stats['effective_rates']:
            rates = self._dropout_stats['effective_rates']
            stats['rate_statistics'] = {
                'mean_rate': np.mean(rates),
                'std_rate': np.std(rates),
                'min_rate': np.min(rates),
                'max_rate': np.max(rates),
                'recent_trend': np.polyfit(range(len(rates[-100:])), rates[-100:], 1)[0] if len(rates) >= 100 else 0.0
            }
        
        return stats
    
    def reset_stats(self):
        """Reset all dropout statistics."""
        self._dropout_stats = {
            'total_elements': 0,
            'dropped_elements': 0,
            'effective_rates': []
        }
        self._forward_count = 0
    
    def simplify_rates(self):
        """Simplify all fractional rates to their lowest terms."""
        if not self.learnable:
            return
        
        with torch.no_grad():
            num_rates = self.num_features if self.channel_wise else 1
            
            for i in range(num_rates):
                try:
                    # Get current values as integers
                    alpha_val = int(self.alpha[i])
                    beta_val = max(1, int(self.beta[i]))
                    
                    # Simplify fraction
                    frac = Fraction(alpha_val, beta_val)
                    
                    # Update parameters
                    self.alpha[i] = float(frac.numerator)
                    self.beta[i] = float(frac.denominator)
                    
                except (ValueError, ZeroDivisionError):
                    continue
        
        logger.debug("Simplified all fractional dropout rates")
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        current_rates = self.get_current_rates()
        if len(current_rates) == 1:
            rate_str = f"{current_rates[0]:.4f}"
        else:
            rate_str = f"[{current_rates[0]:.4f}...] (channel-wise)"
        
        return (f'rate={rate_str}, learnable={self.learnable}, '
                f'channel_wise={self.channel_wise}, '
                f'max_denominator={self.max_denominator}')
    
    def __repr__(self):
        current_rates = self.get_current_rates()
        if len(current_rates) == 1:
            return f"FracDropout(rate={current_rates[0]:.4f})"
        else:
            return f"FracDropout(channel_wise={self.num_features})"


class AdaptiveFracDropout(FracDropout):
    """
    Adaptive Fractional Dropout that automatically adjusts rates based on training dynamics.
    
    This variant monitors training progress and adapts dropout rates to maintain
    optimal regularization throughout training.
    
    Args:
        initial_rate: Initial dropout rate
        target_overfitting_ratio: Target ratio of train/val performance for adaptation
        adaptation_sensitivity: How quickly to adapt rates
        adaptation_interval: How often to check and adapt rates
        **kwargs: Additional arguments passed to FracDropout
    """
    
    def __init__(self,
                 initial_rate: float = 0.5,
                 target_overfitting_ratio: float = 1.05,
                 adaptation_sensitivity: float = 0.02,
                 adaptation_interval: int = 500,
                 **kwargs):
        
        super().__init__(initial_rate, **kwargs)
        
        self.target_overfitting_ratio = target_overfitting_ratio
        self.adaptation_sensitivity = adaptation_sensitivity
        self.adaptation_interval = adaptation_interval
        
        # Track training metrics
        self.register_buffer('train_losses', torch.tensor([]))
        self.register_buffer('val_losses', torch.tensor([]))
        self._adaptation_count = 0
        
        logger.debug(f"AdaptiveFracDropout created: target_ratio={target_overfitting_ratio}")
    
    def update_losses(self, train_loss: float, val_loss: Optional[float] = None):
        """
        Update loss tracking for adaptation.
        
        Args:
            train_loss: Current training loss
            val_loss: Current validation loss (optional)
        """
        self.train_losses = torch.cat([self.train_losses, torch.tensor([train_loss])])
        
        if val_loss is not None:
            self.val_losses = torch.cat([self.val_losses, torch.tensor([val_loss])])
        
        # Keep only recent history
        if len(self.train_losses) > 1000:
            self.train_losses = self.train_losses[-1000:]
        if len(self.val_losses) > 1000:
            self.val_losses = self.val_losses[-1000:]
        
        # Check if adaptation is needed
        if (len(self.train_losses) > 10 and len(self.val_losses) > 10 and 
            self._forward_count % self.adaptation_interval == 0):
            self._adapt_to_overfitting()
    
    def _adapt_to_overfitting(self):
        """Adapt dropout rates based on overfitting indicators."""
        if not self.learnable or len(self.val_losses) < 10:
            return
        
        with torch.no_grad():
            # Calculate recent performance ratio
            recent_train = torch.mean(self.train_losses[-10:]).item()
            recent_val = torch.mean(self.val_losses[-10:]).item()
            
            if recent_train > 0:
                overfitting_ratio = recent_val / recent_train
            else:
                return
            
            # Determine adaptation direction
            if overfitting_ratio > self.target_overfitting_ratio:
                # Overfitting detected - increase dropout
                rate_adjustment = self.adaptation_sensitivity
            else:
                # Underfitting possible - decrease dropout
                rate_adjustment = -self.adaptation_sensitivity
            
            # Apply adjustment
            current_rates = self.get_current_rates()
            for i in range(len(self.alpha)):
                current_rate = current_rates[i].item()
                new_rate = current_rate + rate_adjustment
                new_rate = np.clip(new_rate, self.rate_bounds[0], self.rate_bounds[1])
                
                try:
                    new_frac = Fraction(new_rate).limit_denominator(self.max_denominator)
                    self.alpha[i] = float(new_frac.numerator)
                    self.beta[i] = float(new_frac.denominator)
                except:
                    continue
            
            self._adaptation_count += 1
            
            if self._adaptation_count % 10 == 0:
                logger.debug(f"Adapted dropout: overfitting_ratio={overfitting_ratio:.3f}, "
                           f"adjustment={rate_adjustment:.4f}")


def create_frac_dropout_schedule(dropout_rates: List[float],
                                max_denominator: int = 1000,
                                **kwargs) -> nn.ModuleList:
    """
    Create a schedule of FracDropout layers with different rates.
    
    Args:
        dropout_rates: List of dropout rates for different layers
        max_denominator: Maximum denominator for all layers
        **kwargs: Additional arguments for FracDropout
        
    Returns:
        ModuleList containing FracDropout layers
    """
    dropouts = nn.ModuleList()
    
    for rate in dropout_rates:
        dropout = FracDropout(
            initial_rate=rate,
            max_denominator=max_denominator,
            **kwargs
        )
        dropouts.append(dropout)
    
    return dropouts


def convert_dropout_to_fractional(module: nn.Module,
                                 inplace: bool = True,
                                 **frac_dropout_kwargs) -> nn.Module:
    """
    Convert standard Dropout layers in a module to FracDropout.
    
    Args:
        module: Module containing Dropout layers
        inplace: Whether to modify the module in place
        **frac_dropout_kwargs: Additional arguments for FracDropout
        
    Returns:
        Module with Dropout layers converted to FracDropout
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)
    
    converted_count = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            # Extract current dropout rate
            current_rate = child.p
            
            frac_dropout = FracDropout(
                initial_rate=current_rate,
                **frac_dropout_kwargs
            )
            
            setattr(module, name, frac_dropout)
            converted_count += 1
        else:
            # Recursively process child modules
            convert_dropout_to_fractional(child, inplace=True, **frac_dropout_kwargs)
    
    logger.info(f"Converted {converted_count} Dropout layers to FracDropout")
    
    return module
