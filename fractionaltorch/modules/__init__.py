"""
FractionalTorch Modules

Neural network layers and components using exact fractional arithmetic.

Author: Lev Goukassian
License: MIT
"""

__version__ = "0.1.0"

# Import the modules that actually exist
try:
    from .fractional_linear import FractionalLinear
except ImportError:
    pass

try:
    from .fraclu import FracLU
except ImportError:
    pass

try:
    from .frac_dropout import FracDropout
except ImportError:
    pass

try:
    from .frac_attention import FracAttention
except ImportError:
    pass

# Export what we have
__all__ = []
if 'FractionalLinear' in globals():
    __all__.append('FractionalLinear')
if 'FracLU' in globals():
    __all__.append('FracLU')
if 'FracDropout' in globals():
    __all__.append('FracDropout')
if 'FracAttention' in globals():
    __all__.append('FracAttention')
