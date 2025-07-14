"""
FractionalTorch Modules

This module contains neural network layers and components that use exact fractional arithmetic,
including linear layers, activation functions, attention mechanisms, and regularization techniques.

Author: Lev Goukassian
License: MIT
"""

# Version information
__version__ = "0.1.0"

# Import neural network modules
from .fractional_linear import FractionalLinear
from .fraclu import FracLU
from .frac_dropout import FracDropout
from .frac_attention import FracAttention

# Import utility functions for module conversion
from .conversion import (
    convert_linear_to_fractional,
    convert_activation_to_fractional,
    convert_module_to_fractional,
    replace_modules_recursively
)

# Import module factories
from .factories import (
    create_fractional_mlp,
    create_fractional_transformer_block,
    create_fractional_cnn_block
)

# All public exports
__all__ = [
    # Core neural network modules
    'FractionalLinear',
    'FracLU',
    'FracDropout',
    'FracAttention',
    
    # Conversion utilities
    'convert_linear_to_fractional',
    'convert_activation_to_fractional', 
    'convert_module_to_fractional',
    'replace_modules_recursively',
    
    # Factory functions
    'create_fractional_mlp',
    'create_fractional_transformer_block',
    'create_fractional_cnn_block',
    
    # Version info
    '__version__',
]

# Module-level imports and setup
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Type, Any, Optional
import warnings

# Set up logging for modules
logger = logging.getLogger(__name__)

# Default configuration for modules
DEFAULT_MODULE_CONFIG = {
    'max_denominator': 1000,
    'simplify_threshold': 100,
    'enable_learnable_rates': True,
    'enable_adaptive_precision': True,
    'verbose_initialization': True
}

def _check_module_dependencies():
    """Check that required dependencies are available for modules."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            warnings.warn(
                f"FractionalTorch modules require PyTorch 2.0+, found {torch.__version__}. "
                "Some modules may not work correctly.",
                RuntimeWarning
            )
    except ImportError:
        raise ImportError(
            "PyTorch is required for FractionalTorch modules. "
            "Install it with: pip install torch>=2.0.0"
        )

# Run dependency checks
_check_module_dependencies()

# Module registry for tracking available fractional modules
_FRACTIONAL_MODULE_REGISTRY = {}

def register_fractional_module(name: str, module_class: Type[nn.Module]):
    """
    Register a fractional module for dynamic discovery.
    
    Args:
        name: Name to register the module under
        module_class: The module class to register
    """
    _FRACTIONAL_MODULE_REGISTRY[name] = module_class
    logger.debug(f"Registered fractional module: {name}")

def get_registered_modules() -> Dict[str, Type[nn.Module]]:
    """Get all registered fractional modules."""
    return _FRACTIONAL_MODULE_REGISTRY.copy()

def list_available_modules() -> List[str]:
    """
    List all available fractional module types.
    
    Returns:
        List of available module names
    """
    return list(_FRACTIONAL_MODULE_REGISTRY.keys())

# Register core modules
register_fractional_module('linear', FractionalLinear)
register_fractional_module('fraclu', FracLU)
register_fractional_module('dropout', FracDropout)
register_fractional_module('attention', FracAttention)

# Module configuration management
class ModuleConfig:
    """Configuration manager for fractional modules."""
    
    def __init__(self):
        self.max_denominator = DEFAULT_MODULE_CONFIG['max_denominator']
        self.simplify_threshold = DEFAULT_MODULE_CONFIG['simplify_threshold']
        self.enable_learnable_rates = DEFAULT_MODULE_CONFIG['enable_learnable_rates']
        self.enable_adaptive_precision = DEFAULT_MODULE_CONFIG['enable_adaptive_precision']
        self.verbose_initialization = DEFAULT_MODULE_CONFIG['verbose_initialization']
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated module config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        for key, value in DEFAULT_MODULE_CONFIG.items():
            setattr(self, key, value)
        logger.info("Module configuration reset to defaults")
    
    def __repr__(self):
        return (
            f"ModuleConfig(\n"
            f"  max_denominator={self.max_denominator},\n"
            f"  simplify_threshold={self.simplify_threshold},\n"
            f"  enable_learnable_rates={self.enable_learnable_rates},\n"
            f"  enable_adaptive_precision={self.enable_adaptive_precision},\n"
            f"  verbose_initialization={self.verbose_initialization}\n"
            f")"
        )

# Global module configuration
config = ModuleConfig()

# Utility functions for module introspection
def analyze_module_composition(module: nn.Module) -> Dict[str, Any]:
    """
    Analyze the composition of fractional vs standard modules in a network.
    
    Args:
        module: PyTorch module to analyze
        
    Returns:
        Dictionary with composition analysis
    """
    total_modules = 0
    fractional_modules = 0
    module_counts = {}
    fractional_params = 0
    total_params = 0
    
    for name, submodule in module.named_modules():
        if name == '':  # Skip the root module
            continue
            
        total_modules += 1
        module_type = type(submodule).__name__
        
        # Count module types
        if module_type not in module_counts:
            module_counts[module_type] = {'total': 0, 'fractional': 0}
        module_counts[module_type]['total'] += 1
        
        # Check if it's a fractional module
        is_fractional = any(
            isinstance(submodule, cls) for cls in _FRACTIONAL_MODULE_REGISTRY.values()
        )
        
        if is_fractional:
            fractional_modules += 1
            module_counts[module_type]['fractional'] += 1
        
        # Count parameters
        for param in submodule.parameters(recurse=False):
            total_params += param.numel()
            if hasattr(param, 'numerators'):  # FractionalWeight check
                fractional_params += param.numel()
    
    return {
        'total_modules': total_modules,
        'fractional_modules': fractional_modules,
        'fractional_module_percentage': (fractional_modules / total_modules * 100) if total_modules > 0 else 0,
        'module_type_breakdown': module_counts,
        'total_parameters': total_params,
        'fractional_parameters': fractional_params,
        'fractional_param_percentage': (fractional_params / total_params * 100) if total_params > 0 else 0,
        'is_fully_fractional': fractional_modules == total_modules,
        'has_fractional_components': fractional_modules > 0
    }

def get_module_precision_stats(module: nn.Module) -> Dict[str, Any]:
    """
    Get precision statistics for all fractional modules in a network.
    
    Args:
        module: PyTorch module to analyze
        
    Returns:
        Dictionary with precision statistics
    """
    from ..core import FractionalWeight
    
    stats = {
        'modules': [],
        'global_stats': {
            'max_denominator': 0,
            'min_denominator': float('inf'),
            'mean_denominator': 0,
            'total_fractional_params': 0
        }
    }
    
    all_denominators = []
    
    for name, submodule in module.named_modules():
        module_stats = {'name': name, 'type': type(submodule).__name__, 'fractional_params': []}
        
        for param_name, param in submodule.named_parameters(recurse=False):
            if isinstance(param, FractionalWeight):
                param_stats = param.get_precision_stats()
                param_stats['param_name'] = param_name
                module_stats['fractional_params'].append(param_stats)
                
                # Collect denominators for global stats
                if param.denominators is not None:
                    denominators = param.denominators.cpu().flatten().tolist()
                    all_denominators.extend(denominators)
        
        if module_stats['fractional_params']:
            stats['modules'].append(module_stats)
    
    # Calculate global statistics
    if all_denominators:
        stats['global_stats']['max_denominator'] = max(all_denominators)
        stats['global_stats']['min_denominator'] = min(all_denominators)
        stats['global_stats']['mean_denominator'] = sum(all_denominators) / len(all_denominators)
        stats['global_stats']['total_fractional_params'] = len(all_denominators)
    
    return stats

def create_module_summary(module: nn.Module) -> str:
    """
    Create a human-readable summary of fractional modules in a network.
    
    Args:
        module: PyTorch module to summarize
        
    Returns:
        Formatted string summary
    """
    composition = analyze_module_composition(module)
    precision_stats = get_module_precision_stats(module)
    
    summary_lines = [
        "=== FractionalTorch Module Summary ===",
        "",
        f"Total modules: {composition['total_modules']}",
        f"Fractional modules: {composition['fractional_modules']} ({composition['fractional_module_percentage']:.1f}%)",
        f"Total parameters: {composition['total_parameters']:,}",
        f"Fractional parameters: {composition['fractional_parameters']:,} ({composition['fractional_param_percentage']:.1f}%)",
        "",
        "Module Type Breakdown:"
    ]
    
    for module_type, counts in composition['module_type_breakdown'].items():
        frac_pct = (counts['fractional'] / counts['total'] * 100) if counts['total'] > 0 else 0
        summary_lines.append(f"  {module_type}: {counts['fractional']}/{counts['total']} ({frac_pct:.1f}%) fractional")
    
    if precision_stats['global_stats']['total_fractional_params'] > 0:
        gs = precision_stats['global_stats']
        summary_lines.extend([
            "",
            "Precision Statistics:",
            f"  Max denominator: {gs['max_denominator']}",
            f"  Min denominator: {gs['min_denominator']}",
            f"  Mean denominator: {gs['mean_denominator']:.1f}",
        ])
    
    summary_lines.append("=" * 35)
    
    return "\n".join(summary_lines)

# Module validation and testing
def validate_module_functionality():
    """Validate that all fractional modules can be instantiated and used."""
    validation_results = {}
    
    try:
        # Test FractionalLinear
        linear = FractionalLinear(10, 5)
        test_input = torch.randn(2, 10)
        output = linear(test_input)
        validation_results['FractionalLinear'] = {'status': 'pass', 'output_shape': output.shape}
    except Exception as e:
        validation_results['FractionalLinear'] = {'status': 'fail', 'error': str(e)}
    
    try:
        # Test FracLU
        fraclu = FracLU(5)
        test_input = torch.randn(2, 5)
        output = fraclu(test_input)
        validation_results['FracLU'] = {'status': 'pass', 'output_shape': output.shape}
    except Exception as e:
        validation_results['FracLU'] = {'status': 'fail', 'error': str(e)}
    
    try:
        # Test FracDropout
        dropout = FracDropout(0.5, learnable=True)
        test_input = torch.randn(2, 5)
        output = dropout(test_input)
        validation_results['FracDropout'] = {'status': 'pass', 'output_shape': output.shape}
    except Exception as e:
        validation_results['FracDropout'] = {'status': 'fail', 'error': str(e)}
    
    try:
        # Test FracAttention
        attention = FracAttention(64, 8)
        test_input = torch.randn(2, 10, 64)
        output = attention(test_input, test_input, test_input)
        validation_results['FracAttention'] = {'status': 'pass', 'output_shape': output.shape}
    except Exception as e:
        validation_results['FracAttention'] = {'status': 'fail', 'error': str(e)}
    
    # Log validation results
    passed = sum(1 for r in validation_results.values() if r['status'] == 'pass')
    total = len(validation_results)
    
    if passed == total:
        logger.info(f"Module validation passed: {passed}/{total} modules working correctly")
    else:
        logger.warning(f"Module validation partial: {passed}/{total} modules working correctly")
        for module_name, result in validation_results.items():
            if result['status'] == 'fail':
                logger.error(f"{module_name} validation failed: {result['error']}")
    
    return validation_results

# Add utility functions to __all__
__all__.extend([
    'config',
    'ModuleConfig',
    'register_fractional_module',
    'get_registered_modules',
    'list_available_modules',
    'analyze_module_composition',
    'get_module_precision_stats',
    'create_module_summary',
    'validate_module_functionality'
])

# Module initialization
logger.info(f"FractionalTorch modules v{__version__} loaded")

# Run validation if requested
if config.verbose_initialization:
    try:
        validation_results = validate_module_functionality()
    except Exception as e:
        logger.warning(f"Module validation skipped due to missing dependencies: {e}")

# Export version
__version__ = __version__
