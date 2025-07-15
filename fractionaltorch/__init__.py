"""
FractionalTorch: Exact Rational Arithmetic for Neural Networks

Author: Lev Goukassian
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Lev Goukassian"

# Try to import core components
try:
    from .core.fractional_weight import FractionalWeight
except ImportError:
    pass

try:
    from .core.fractional_ops import FractionalOps
except ImportError:
    pass

try:
    from .modules.fractional_linear import FractionalLinear
except ImportError:
    pass

try:
    from .modules.fraclu import FracLU
except ImportError:
    pass

try:
    from .modules.frac_dropout import FracDropout
except ImportError:
    pass

try:
    from .modules.frac_attention import FracAttention
except ImportError:
    pass

print("✅ FractionalTorch loaded successfully!")
