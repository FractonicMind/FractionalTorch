"""
FractionalTorch: Exact Rational Arithmetic for Numerically Stable Neural Network Training

A PyTorch extension that replaces traditional floating-point arithmetic with exact 
rational number representations, achieving perfect numerical reproducibility and 
improved training stability.

Author: Lev Goukassian
License: MIT
GitHub: https://github.com/FractonicMind/FractionalTorch
"""

__version__ = "0.1.0"
__author__ = "[Your Name]"
__email__ = "[your.email@example.com]"
__license__ = "MIT"
__url__ = "https://github.com/FractonicMind/FractionalTorch"

# Import core functionality
from .core.fractional_weight import FractionalWeight
from .core.fractional_ops import FractionalOps
from .core.denominator_scheduler import DenominatorScheduler

# Import specialized modules
from .modules.fractional_linear import FractionalLinear
from .modules.fraclu import FracLU
from .modules.frac_dropout import FracDropout
from .modules.frac_attention import FracAttention

# Import utility functions
from .utils.conversion import convert_to_fractional, convert_from_fractional
from .utils.precision import get_precision_stats, optimize_denominators

# Import benchmark utilities
from .benchmarks.stability import stability_benchmark
from .benchmarks.reproducibility import reproducibility_test
from .benchmarks.performance import performance_benchmark

# Version information
version_info = tuple(map(int, __version__.split('.')))

# Define what gets imported with "from fractionaltorch import *"
__all__ = [
    # Core classes
    'FractionalWeight',
    'FractionalOps', 
    'DenominatorScheduler',
    
    # Neural network modules
    'FractionalLinear',
    'FracLU',
    'FracDropout', 
    'FracAttention',
    
    # Utility functions
    'convert_to_fractional',
    'convert_from_fractional',
    'get_precision_stats',
    'optimize_denominators',
    
    # Benchmark functions
    'stability_benchmark',
    'reproducibility_test', 
    'performance_benchmark',
    
    # Version info
    '__version__',
    'version_info',
]

# Package-level configuration
import logging

# Set up logging for the package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Configuration defaults
DEFAULT_MAX_DENOMINATOR = 1000
DEFAULT_SIMPLIFY_THRESHOLD = 100
DEFAULT_SCHEDULER_STRATEGY = 'adaptive'

# Compatibility checks
def _check_dependencies():
    """Check if required dependencies are available."""
    try:
        import torch
        if tuple(map(int, torch.__version__.split('.')[:2])) < (2, 0):
            logger.warning(
                f"FractionalTorch requires PyTorch 2.0+, found {torch.__version__}. "
                "Some features may not work correctly."
            )
    except ImportError:
        raise ImportError(
            "PyTorch is required for FractionalTorch. "
            "Install it with: pip install torch>=2.0.0"
        )
    
    try:
        import numpy as np
        if tuple(map(int, np.__version__.split('.')[:2])) < (1, 21):
            logger.warning(
                f"FractionalTorch requires NumPy 1.21+, found {np.__version__}. "
                "Some features may not work correctly."
            )
    except ImportError:
        raise ImportError(
            "NumPy is required for FractionalTorch. "
            "Install it with: pip install numpy>=1.21.0"
        )

# Run dependency checks on import
_check_dependencies()

# Package initialization message
logger.info(f"FractionalTorch {__version__} initialized successfully")

# Convenience functions for quick start
def quick_convert(model):
    """
    Quickly convert a PyTorch model to use fractional arithmetic.
    
    Args:
        model: PyTorch model to convert
        
    Returns:
        Model with fractional components
        
    Example:
        >>> import torchvision.models as models
        >>> import fractionaltorch as ft
        >>> 
        >>> standard_model = models.resnet18()
        >>> fractional_model = ft.quick_convert(standard_model)
    """
    return convert_to_fractional(model)

def set_global_precision(max_denominator):
    """
    Set the global maximum denominator for all fractional operations.
    
    Args:
        max_denominator (int): Maximum denominator to use
        
    Example:
        >>> import fractionaltorch as ft
        >>> ft.set_global_precision(500)  # Use denominators up to 500
    """
    global DEFAULT_MAX_DENOMINATOR
    DEFAULT_MAX_DENOMINATOR = max_denominator
    logger.info(f"Global precision set to max denominator: {max_denominator}")

def get_global_precision():
    """
    Get the current global maximum denominator setting.
    
    Returns:
        int: Current maximum denominator
        
    Example:
        >>> import fractionaltorch as ft
        >>> precision = ft.get_global_precision()
        >>> print(f"Current precision: {precision}")
    """
    return DEFAULT_MAX_DENOMINATOR

# Advanced configuration
class Config:
    """Global configuration for FractionalTorch."""
    
    def __init__(self):
        self.max_denominator = DEFAULT_MAX_DENOMINATOR
        self.simplify_threshold = DEFAULT_SIMPLIFY_THRESHOLD
        self.scheduler_strategy = DEFAULT_SCHEDULER_STRATEGY
        self.enable_logging = True
        self.enable_warnings = True
    
    def __repr__(self):
        return (
            f"FractionalTorch.Config(\n"
            f"  max_denominator={self.max_denominator},\n"
            f"  simplify_threshold={self.simplify_threshold},\n"
            f"  scheduler_strategy='{self.scheduler_strategy}',\n"
            f"  enable_logging={self.enable_logging},\n"
            f"  enable_warnings={self.enable_warnings}\n"
            f")"
        )

# Global configuration instance
config = Config()

# Helpful aliases for common use cases
FracLinear = FractionalLinear  # Shorter alias
FracScheduler = DenominatorScheduler  # Shorter alias

# Export the config for user access
__all__.extend(['config', 'quick_convert', 'set_global_precision', 'get_global_precision'])

# Package metadata for programmatic access
__package_info__ = {
    'name': 'fractionaltorch',
    'version': __version__,
    'description': 'Exact rational arithmetic for numerically stable neural network training',
    'author': __author__,
    'license': __license__,
    'url': __url__,
    'requires_python': '>=3.8',
    'requires': ['torch>=2.0.0', 'numpy>=1.21.0'],
}
