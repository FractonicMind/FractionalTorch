"""
FractionalTorch Modules: Fractional Multi-Head Attention

This module implements multi-head attention mechanisms using exact fractional arithmetic
for scaling factors, providing numerically stable and reproducible attention computations.

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
from .fractional_linear import FractionalLinear
from .frac_dropout import FracDropout

logger = logging.getLogger(__name__)


class FracAttention(nn.Module):
    """
    Multi-head attention with fractional scaling factors and exact arithmetic.
    
    This module replaces the standard 1/√d_k scaling with learnable fractional
    scaling factors α/β for each attention head, enabling adaptive attention
    behavior while maintaining exact reproducibility.
    
    Args:
        d_model: Model dimensionality
        n_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        bias: Whether to use bias in linear projections
        max_denominator: Maximum denominator for fractional scaling
        learnable_scales: Whether scaling factors should be learnable
        head_specific_scales: Whether each head has its own scaling factor
        scale_init_strategy: Strategy for initializing scales ('sqrt', 'uniform', 'adaptive')
        use_fractional_projections: Whether to use fractional linear layers for Q,K,V,O
        
    Shape:
        - query: (batch_size, seq_len, d_model)
        - key: (batch_size, seq_len, d_model)  
        - value: (batch_size, seq_len, d_model)
        - output: (batch_size, seq_len, d_model)
        
    Examples:
        >>> # Basic fractional attention
        >>> attention = FracAttention(512, 8)
        >>> output = attention(query, key, value)
        
        >>> # With learnable head-specific scales
        >>> custom_attention = FracAttention(256, 4, learnable_scales=True, head_specific_scales=True)
        
        >>> # Using fractional projections throughout
        >>> full_frac_attention = FracAttention(512, 8, use_fractional_projections=True)
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True,
                 max_denominator: int = 1000,
                 learnable_scales: bool = True,
                 head_specific_scales: bool = True,
                 scale_init_strategy: str = 'sqrt',
                 use_fractional_projections: bool = False,
                 use_fractional_dropout: bool = True):
        
        super(FracAttention, self).__init__()
        
        # Validate inputs
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_denominator = max_denominator
        self.learnable_scales = learnable_scales
        self.head_specific_scales = head_specific_scales
        self.scale_init_strategy = scale_init_strategy
        self.use_fractional_projections = use_fractional_projections
        
        # Create linear projections
        if use_fractional_projections:
            self.w_q = FractionalLinear(d_model, d_model, bias=bias, max_denominator=max_denominator)
            self.w_k = FractionalLinear(d_model, d_model, bias=bias, max_denominator=max_denominator)
            self.w_v = FractionalLinear(d_model, d_model, bias=bias, max_denominator=max_denominator)
            self.w_o = FractionalLinear(d_model, d_model, bias=bias, max_denominator=max_denominator)
        else:
            self.w_q = nn.Linear(d_model, d_model, bias=bias)
            self.w_k = nn.Linear(d_model, d_model, bias=bias)
            self.w_v = nn.Linear(d_model, d_model, bias=bias)
            self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize fractional scaling parameters
        self._initialize_scaling_factors()
        
        # Dropout layer
        if use_fractional_dropout:
            self.dropout = FracDropout(dropout, learnable=True)
        else:
            self.dropout = nn.Dropout(dropout)
        
        # Statistics tracking
        self._forward_count = 0
        self._attention_stats = {
            'avg_attention_entropy': [],
            'max_attention_weights': [],
            'attention_sparsity': []
        }
        
        logger.debug(f"FracAttention created: d_model={d_model}, n_heads={n_heads}, "
                    f"learnable_scales={learnable_scales}, fractional_proj={use_fractional_projections}")
    
    def _initialize_scaling_factors(self):
        """Initialize fractional scaling factors based on strategy."""
        # Determine parameter shape
        if self.head_specific_scales:
            param_shape = (self.n_heads,)
        else:
            param_shape = (1,)
        
        # Calculate initial scaling based on strategy
        if self.scale_init_strategy == 'sqrt':
            # Standard 1/√d_k scaling
            initial_scale = 1.0 / math.sqrt(self.d_k)
        elif self.scale_init_strategy == 'uniform':
            # Uniform scaling across heads
            initial_scale = 1.0
        elif self.scale_init_strategy == 'adaptive':
            # Will be set per head below
            initial_scale = 1.0
        else:
            raise ValueError(f"Unknown scale initialization strategy: {self.scale_init_strategy}")
        
        # Convert to fraction
        try:
            initial_frac = Fraction(initial_scale).limit_denominator(self.max_denominator)
        except (ValueError, OverflowError):
            logger.warning("Could not convert initial scale to fraction, using 1/1")
            initial_frac = Fraction(1, 1)
        
        # Create parameters
        if self.learnable_scales:
            self.scale_alpha = Parameter(torch.full(param_shape, float(initial_frac.numerator)))
            self.scale_beta = Parameter(torch.full(param_shape, float(initial_frac.denominator)))
        else:
            self.register_buffer('scale_alpha', torch.full(param_shape, float(initial_frac.numerator)))
            self.register_buffer('scale_beta', torch.full(param_shape, float(initial_frac.denominator)))
        
        # Apply adaptive initialization if requested
        if self.scale_init_strategy == 'adaptive' and self.head_specific_scales:
            self._apply_adaptive_initialization()
        
        # Ensure denominators start positive
        with torch.no_grad():
            self.scale_beta.clamp_(min=1.0)
    
    def _apply_adaptive_initialization(self):
        """Apply adaptive initialization for head-specific scales."""
        with torch.no_grad():
            for i in range(self.n_heads):
                # Vary scaling factors across heads
                head_factor = 1.0 + 0.1 * math.sin(i * math.pi / self.n_heads)
                base_scale = head_factor / math.sqrt(self.d_k)
                
                try:
                    scale_frac = Fraction(base_scale).limit_denominator(self.max_denominator)
                    self.scale_alpha[i] = float(scale_frac.numerator)
                    self.scale_beta[i] = float(scale_frac.denominator)
                except:
                    continue
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of fractional multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor, optionally with attention weights
        """
        batch_size, seq_len, d_model = query.shape
        self._forward_count += 1
        
        # Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)    # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        
        # Apply fractional scaled dot-product attention
        attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final output projection
        output = self.w_o(attention_output)
        
        # Update statistics
        if self._forward_count % 100 == 0:
            self._update_attention_stats(attention_weights)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def _scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention with fractional scaling.
        
        Args:
            Q, K, V: Query, Key, Value tensors (batch_size, n_heads, seq_len, d_k)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, n_heads, seq_len, seq_len)
        
        # Apply fractional scaling
        fractional_scales = self.get_current_scales()
        
        # Reshape scales for broadcasting
        if self.head_specific_scales:
            # scales: (n_heads,) -> (1, n_heads, 1, 1)
            scales = fractional_scales.view(1, self.n_heads, 1, 1)
        else:
            # scales: (1,) -> (1, 1, 1, 1)
            scales = fractional_scales.view(1, 1, 1, 1)
        
        scores = scores * scales
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct shape for broadcasting
            if mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            elif mask.dim() == 2:  # (seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def get_current_scales(self) -> torch.Tensor:
        """
        Get current fractional scaling factors as float tensor.
        
        Returns:
            Tensor of current scaling factors
        """
        with torch.no_grad():
            # Ensure denominators are positive
            beta_safe = torch.clamp(self.scale_beta, min=1.0)
            
            # Compute fractional scales
            scales = self.scale_alpha / beta_safe
            
            return scales
    
    def get_fractional_scales(self) -> List[Fraction]:
        """
        Get current scaling factors as exact Fraction objects.
        
        Returns:
            List of Fraction objects representing exact scaling factors
        """
        scales = []
        
        num_scales = self.n_heads if self.head_specific_scales else 1
        
        for i in range(num_scales):
            try:
                alpha_val = int(self.scale_alpha[i])
                beta_val = max(1, int(self.scale_beta[i]))
                
                frac = Fraction(alpha_val, beta_val)
                scales.append(frac)
                
            except (ValueError, ZeroDivisionError):
                # Fallback for problematic values
                scales.append(Fraction(1, int(math.sqrt(self.d_k))))
        
        return scales
    
    def set_scale(self, scale: float, head_idx: Optional[int] = None):
        """
        Set scaling factor for specific head or all heads.
        
        Args:
            scale: New scaling factor
            head_idx: Head index to set (None for all heads)
        """
        # Convert to fraction
        try:
            frac = Fraction(scale).limit_denominator(self.max_denominator)
        except (ValueError, OverflowError):
            logger.error(f"Could not convert scale to fraction: {scale}")
            return
        
        with torch.no_grad():
            if head_idx is None:
                # Set all heads
                self.scale_alpha.fill_(float(frac.numerator))
                self.scale_beta.fill_(float(frac.denominator))
            else:
                # Set specific head
                if self.head_specific_scales and 0 <= head_idx < self.n_heads:
                    self.scale_alpha[head_idx] = float(frac.numerator)
                    self.scale_beta[head_idx] = float(frac.denominator)
                else:
                    logger.error(f"Invalid head index or not head-specific: {head_idx}")
    
    def _update_attention_stats(self, attention_weights: torch.Tensor):
        """Update attention statistics for analysis."""
        with torch.no_grad():
            # attention_weights: (batch_size, n_heads, seq_len, seq_len)
            
            # Compute attention entropy (measure of attention spread)
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
            avg_entropy = torch.mean(entropy).item()
            self._attention_stats['avg_attention_entropy'].append(avg_entropy)
            
            # Maximum attention weight (measure of attention concentration)
            max_weight = torch.max(attention_weights).item()
            self._attention_stats['max_attention_weights'].append(max_weight)
            
            # Attention sparsity (percentage of weights below threshold)
            threshold = 0.01
            sparse_count = torch.sum(attention_weights < threshold).item()
            total_count = attention_weights.numel()
            sparsity = sparse_count / total_count
            self._attention_stats['attention_sparsity'].append(sparsity)
            
            # Keep only recent statistics
            for key in self._attention_stats:
                if len(self._attention_stats[key]) > 1000:
                    self._attention_stats[key] = self._attention_stats[key][-1000:]
    
    def get_attention_stats(self) -> dict:
        """
        Get comprehensive attention statistics.
        
        Returns:
            Dictionary with attention statistics
        """
        stats = {
            'forward_count': self._forward_count,
            'current_scales': self.get_current_scales().tolist(),
            'fractional_scales': [str(f) for f in self.get_fractional_scales()],
            'learnable_scales': self.learnable_scales,
            'head_specific_scales': self.head_specific_scales,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_k': self.d_k
        }
        
        # Add statistical summaries if data exists
        for stat_name, values in self._attention_stats.items():
            if values:
                stats[stat_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'recent_trend': np.polyfit(range(len(values[-100:])), values[-100:], 1)[0] if len(values) >= 100 else 0.0
                }
        
        return stats
    
    def reset_stats(self):
        """Reset all attention statistics."""
        self._attention_stats = {
            'avg_attention_entropy': [],
            'max_attention_weights': [],
            'attention_sparsity': []
        }
        self._forward_count = 0
    
    def simplify_scales(self):
        """Simplify all fractional scaling factors to their lowest terms."""
        if not self.learnable_scales:
            return
        
        with torch.no_grad():
            num_scales = self.n_heads if self.head_specific_scales else 1
            
            for i in range(num_scales):
                try:
                    alpha_val = int(self.scale_alpha[i])
                    beta_val = max(1, int(self.scale_beta[i]))
                    
                    frac = Fraction(alpha_val, beta_val)
                    
                    self.scale_alpha[i] = float(frac.numerator)
                    self.scale_beta[i] = float(frac.denominator)
                    
                except (ValueError, ZeroDivisionError):
                    continue
        
        logger.debug("Simplified all fractional attention scales")
    
    def get_head_analysis(self) -> dict:
        """
        Analyze individual attention head characteristics.
        
        Returns:
            Dictionary with per-head analysis
        """
        current_scales = self.get_current_scales()
        fractional_scales = self.get_fractional_scales()
        
        analysis = {
            'head_count': self.n_heads,
            'heads': []
        }
        
        for i in range(self.n_heads):
            if self.head_specific_scales:
                scale_val = current_scales[i].item()
                frac_str = str(fractional_scales[i])
            else:
                scale_val = current_scales[0].item()
                frac_str = str(fractional_scales[0])
            
            head_info = {
                'head_idx': i,
                'scale_value': scale_val,
                'fractional_scale': frac_str,
                'relative_scale': scale_val / (1.0 / math.sqrt(self.d_k)),  # Relative to standard scale
                'effective_temperature': 1.0 / scale_val if scale_val > 0 else float('inf')
            }
            
            analysis['heads'].append(head_info)
        
        return analysis
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        current_scales = self.get_current_scales()
        if len(current_scales) == 1:
            scale_str = f"{current_scales[0]:.6f}"
        else:
            scale_str = f"[{current_scales[0]:.6f}...] (head-specific)"
        
        return (f'd_model={self.d_model}, n_heads={self.n_heads}, '
                f'scales={scale_str}, learnable={self.learnable_scales}, '
                f'fractional_proj={self.use_fractional_projections}')
    
    def __repr__(self):
        return (f"FracAttention(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"learnable_scales={self.learnable_scales})")


class FracCausalAttention(FracAttention):
    """
    Causal (masked) version of FracAttention for autoregressive models.
    
    This variant applies a causal mask to prevent attention to future positions,
    making it suitable for language modeling and other autoregressive tasks.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Register causal mask buffer (will be created dynamically)
        self.register_buffer('causal_mask', None)
        self._max_seq_len = 0
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal mask for given sequence length."""
        if self.causal_mask is None or seq_len > self._max_seq_len:
            # Create new causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            self.causal_mask = mask
            self._max_seq_len = seq_len
            return mask
        else:
            # Return subset of existing mask
            return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with causal masking."""
        seq_len = query.shape[1]
        
        # Get causal mask
        causal_mask = self._get_causal_mask(seq_len, query.device)
        
        # Combine with user-provided mask if any
        if mask is not None:
            combined_mask = mask * causal_mask
        else:
            combined_mask = causal_mask
        
        return super().forward(query, key, value, combined_mask, return_attention)


def create_fractional_transformer_block(d_model: int,
                                       n_heads: int,
                                       d_ff: int,
                                       dropout: float = 0.1,
                                       max_denominator: int = 1000,
                                       causal: bool = False) -> nn.Module:
    """
    Create a complete transformer block using fractional components.
    
    Args:
        d_model: Model dimensionality
        n_heads: Number of attention heads
        d_ff: Feed-forward network dimensionality
        dropout: Dropout probability
        max_denominator: Maximum denominator for fractional components
        causal: Whether to use causal attention
        
    Returns:
        Complete transformer block with fractional components
    """
    from .fraclu import FracLU
    
    class FractionalTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Attention layer
            if causal:
                self.attention = FracCausalAttention(
                    d_model, n_heads, dropout, 
                    max_denominator=max_denominator,
                    use_fractional_projections=True,
                    use_fractional_dropout=True
                )
            else:
                self.attention = FracAttention(
                    d_model, n_heads, dropout,
                    max_denominator=max_denominator,
                    use_fractional_projections=True,
                    use_fractional_dropout=True
                )
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Feed-forward network with fractional components
            self.ffn = nn.Sequential(
                FractionalLinear(d_model, d_ff, max_denominator=max_denominator),
                FracLU(d_ff, learnable=True, max_denominator=max_denominator),
                FracDropout(dropout, learnable=True, max_denominator=max_denominator),
                FractionalLinear(d_ff, d_model, max_denominator=max_denominator),
            )
            
            # Dropout for residual connections
            self.dropout = FracDropout(dropout, learnable=True, max_denominator=max_denominator)
        
        def forward(self, x, mask=None):
            # Self-attention with residual connection
            attn_output = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with residual connection
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
            
            return x
    
    return FractionalTransformerBlock()


def convert_attention_to_fractional(module: nn.Module,
                                   max_denominator: int = 1000,
                                   inplace: bool = True) -> nn.Module:
    """
    Convert MultiheadAttention layers in a module to FracAttention.
    
    Args:
        module: Module containing MultiheadAttention layers
        max_denominator: Maximum denominator for fractional components
        inplace: Whether to modify the module in place
        
    Returns:
        Module with attention layers converted to fractional
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)
    
    converted_count = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            # Extract parameters from existing attention
            d_model = child.embed_dim
            n_heads = child.num_heads
            dropout = child.dropout
            bias = child.in_proj_bias is not None
            
            # Create fractional attention
            frac_attention = FracAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
                max_denominator=max_denominator,
                use_fractional_projections=True,
                use_fractional_dropout=True
            )
            
            # Copy weights if possible
            with torch.no_grad():
                if hasattr(child, 'in_proj_weight') and child.in_proj_weight is not None:
                    # Split combined QKV weights
                    qkv_weight = child.in_proj_weight
                    q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
                    
                    if hasattr(frac_attention.w_q, 'weight'):
                        frac_attention.w_q.weight.data.copy_(q_weight)
                        frac_attention.w_k.weight.data.copy_(k_weight)
                        frac_attention.w_v.weight.data.copy_(v_weight)
                
                if hasattr(child, 'out_proj') and hasattr(frac_attention.w_o, 'weight'):
                    frac_attention.w_o.weight.data.copy_(child.out_proj.weight)
            
            setattr(module, name, frac_attention)
            converted_count += 1
        else:
            # Recursively process child modules
            convert_attention_to_fractional(child, max_denominator, inplace=True)
    
    logger.info(f"Converted {converted_count} MultiheadAttention layers to FracAttention")
    
    return module
