"""
FractionalTorch Core Module

This module contains the core functionality for exact rational arithmetic in neural networks,
including fractional weight representations, arithmetic operations, and adaptive precision scheduling.

Author: Lev Goukassian
License: MIT
"""

# Version information
__version__ = "0.1.0"

# Import core classes and functions
from .fractional_weight import (
    FractionalWeight,
    create_fractional_like,
    fractional_from_float
)

from .fractional_ops import (
    FractionalOps,
    FractionalFunction,
    FractionalTensorOps,
    add_fractional,
    multiply_fractional
)

from .denominator_scheduler import (
    BaseDenominatorScheduler,
    ExponentialScheduler,
    LinearScheduler,
    AdaptiveScheduler,
    CosineScheduler,
    StepScheduler,
    CustomScheduler,
    DenominatorScheduler,  # Alias for AdaptiveScheduler
    SchedulerManager,
    create_scheduler,
    SchedulerState
)

# Import utility types and data classes
from .denominator_scheduler import SchedulerState

# Core functionality groupings
__all__ = [
    # Core weight representation
    'FractionalWeight',
    'create_fractional_like',
    'fractional_from_float',
    
    # Arithmetic operations
    'FractionalOps',
    'FractionalFunction', 
    'FractionalTensorOps',
    'add_fractional',
    'multiply_fractional',
    
    # Scheduler classes
    'BaseDenominatorScheduler',
    'ExponentialScheduler',
    'LinearScheduler', 
    'AdaptiveScheduler',
    'CosineScheduler',
    'StepScheduler',
    'CustomScheduler',
    'DenominatorScheduler',
    'SchedulerManager',
    'SchedulerState',
    
    # Factory functions
    'create_scheduler',
    
    # Version info
    '__version__',
]

# Module-level configuration
import logging
import torch
import warnings

# Set up logging for the core module
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_DENOMINATOR = 1000
DEFAULT_SIMPLIFY_THRESHOLD = 100
DEFAULT_SCHEDULER_STRATEGY = 'adaptive'

# Core module initialization
def _check_core_dependencies():
    """Check that core dependencies are available and compatible."""
    try:
        import torch
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            warnings.warn(
                f"FractionalTorch core requires PyTorch 2.0+, found {torch.__version__}. "
                "Some features may not work correctly.",
                RuntimeWarning
            )
    except ImportError:
        raise ImportError(
            "PyTorch is required for FractionalTorch core functionality. "
            "Install it with: pip install torch>=2.0.0"
        )
    
    try:
        import numpy as np
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        if numpy_version < (1, 21):
            warnings.warn(
                f"FractionalTorch core requires NumPy 1.21+, found {np.__version__}. "
                "Some features may not work correctly.",
                RuntimeWarning
            )
    except ImportError:
        raise ImportError(
            "NumPy is required for FractionalTorch core functionality. "
            "Install it with: pip install numpy>=1.21.0"
        )

# Run dependency checks
_check_core_dependencies()

# Core utility functions
def get_core_version():
    """Get the version of the core module."""
    return __version__

def list_available_schedulers():
    """
    List all available denominator scheduler types.
    
    Returns:
        List of scheduler names
    """
    return [
        'exponential',
        'linear', 
        'adaptive',
        'cosine',
        'step',
        'custom'
    ]

def create_default_scheduler(total_steps: int = 10000, strategy: str = 'adaptive'):
    """
    Create a scheduler with sensible defaults for common use cases.
    
    Args:
        total_steps: Total number of training steps
        strategy: Scheduling strategy ('adaptive', 'exponential', 'linear')
        
    Returns:
        Configured scheduler instance
        
    Example:
        >>> scheduler = create_default_scheduler(5000, 'adaptive')
        >>> max_denom = scheduler.step(loss=0.1)
    """
    if strategy == 'adaptive':
        return AdaptiveScheduler(
            initial_max_denom=10,
            final_max_denom=1000,
            patience=max(50, total_steps // 100),
            factor=1.5,
            threshold=1e-6
        )
    elif strategy == 'exponential':
        return ExponentialScheduler(
            initial_max_denom=10,
            final_max_denom=1000,
            total_steps=total_steps
        )
    elif strategy == 'linear':
        return LinearScheduler(
            initial_max_denom=10,
            final_max_denom=1000,
            total_steps=total_steps,
            warmup_steps=max(100, total_steps // 20)
        )
    else:
        raise ValueError(f"Unknown default strategy: {strategy}")

def setup_fractional_training(model: torch.nn.Module, 
                             scheduler_strategy: str = 'adaptive',
                             total_steps: int = 10000,
                             **scheduler_kwargs):
    """
    Convenience function to set up fractional training for a model.
    
    Args:
        model: PyTorch model with FractionalWeight parameters
        scheduler_strategy: Type of scheduler to use
        total_steps: Total training steps
        **scheduler_kwargs: Additional arguments for scheduler
        
    Returns:
        SchedulerManager instance ready for training
        
    Example:
        >>> manager = setup_fractional_training(model, 'adaptive', 5000)
        >>> # During training:
        >>> new_max_denom = manager.step(loss=current_loss)
    """
    # Create scheduler
    if scheduler_kwargs:
        scheduler = create_scheduler(scheduler_strategy, **scheduler_kwargs)
    else:
        scheduler = create_default_scheduler(total_steps, scheduler_strategy)
    
    # Create manager
    manager = SchedulerManager(model, scheduler)
    
    logger.info(f"Fractional training setup complete: {scheduler_strategy} scheduler, "
                f"{len(manager.fractional_params)} fractional parameters")
    
    return manager

# Core diagnostic functions
def diagnose_fractional_model(model: torch.nn.Module) -> dict:
    """
    Analyze a model for fractional components and provide diagnostic information.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with diagnostic information
    """
    total_params = 0
    fractional_params = 0
    fractional_elements = 0
    param_info = []
    
    for name, param in model.named_parameters():
        total_params += 1
        
        if isinstance(param, FractionalWeight):
            fractional_params += 1
            fractional_elements += param.numel()
            
            stats = param.get_precision_stats()
            param_info.append({
                'name': name,
                'shape': list(param.shape),
                'elements': param.numel(),
                'max_denominator': stats['max_denominator'],
                'mean_denominator': stats['mean_denominator'],
                'num_integers': stats['num_integers']
            })
    
    # Calculate memory overhead
    standard_memory = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    fractional_memory = fractional_elements * 8  # 8 bytes per fraction (2 int32s)
    memory_overhead = (fractional_memory / standard_memory) if standard_memory > 0 else 0
    
    return {
        'total_parameters': total_params,
        'fractional_parameters': fractional_params,
        'fractional_elements': fractional_elements,
        'fractional_percentage': (fractional_params / total_params * 100) if total_params > 0 else 0,
        'memory_overhead_factor': 1 + memory_overhead,
        'parameter_details': param_info,
        'has_fractional_weights': fractional_params > 0,
        'recommendations': _generate_recommendations(fractional_params, total_params, fractional_elements)
    }

def _generate_recommendations(frac_params: int, total_params: int, frac_elements: int) -> list:
    """Generate recommendations based on model analysis."""
    recommendations = []
    
    if frac_params == 0:
        recommendations.append("No fractional parameters found. Use convert_to_fractional() to add fractional arithmetic.")
    
    if frac_params > 0 and frac_params < total_params:
        recommendations.append("Partially fractional model. Consider converting all parameters for maximum benefit.")
    
    if frac_elements > 1000000:  # 1M elements
        recommendations.append("Large number of fractional elements. Consider using adaptive scheduler to manage computational overhead.")
    
    if frac_params / total_params > 0.8:
        recommendations.append("Highly fractional model. Monitor training speed and consider performance optimizations.")
    
    return recommendations

# Performance monitoring
class CorePerformanceMonitor:
    """Monitor performance of core fractional operations."""
    
    def __init__(self):
        self.operation_counts = {}
        self.operation_times = {}
        self.enabled = False
    
    def enable(self):
        """Enable performance monitoring."""
        self.enabled = True
        logger.info("Core performance monitoring enabled")
    
    def disable(self):
        """Disable performance monitoring."""
        self.enabled = False
        logger.info("Core performance monitoring disabled")
    
    def record_operation(self, operation: str, duration: float):
        """Record an operation and its duration."""
        if not self.enabled:
            return
        
        if operation not in self.operation_counts:
            self.operation_counts[operation] = 0
            self.operation_times[operation] = 0.0
        
        self.operation_counts[operation] += 1
        self.operation_times[operation] += duration
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for op in self.operation_counts:
            count = self.operation_counts[op]
            total_time = self.operation_times[op]
            stats[op] = {
                'count': count,
                'total_time': total_time,
                'average_time': total_time / count if count > 0 else 0.0
            }
        return stats
    
    def reset(self):
        """Reset all performance counters."""
        self.operation_counts.clear()
        self.operation_times.clear()

# Global performance monitor instance
performance_monitor = CorePerformanceMonitor()

# Core module configuration
class CoreConfig:
    """Configuration class for core module settings."""
    
    def __init__(self):
        self.default_max_denominator = DEFAULT_MAX_DENOMINATOR
        self.default_simplify_threshold = DEFAULT_SIMPLIFY_THRESHOLD
        self.default_scheduler_strategy = DEFAULT_SCHEDULER_STRATEGY
        self.enable_warnings = True
        self.enable_performance_monitoring = False
        self.log_level = logging.INFO
    
    def __repr__(self):
        return (
            f"CoreConfig(\n"
            f"  default_max_denominator={self.default_max_denominator},\n"
            f"  default_simplify_threshold={self.default_simplify_threshold},\n"
            f"  default_scheduler_strategy='{self.default_scheduler_strategy}',\n"
            f"  enable_warnings={self.enable_warnings},\n"
            f"  enable_performance_monitoring={self.enable_performance_monitoring},\n"
            f"  log_level={self.log_level}\n"
            f")"
        )

# Global configuration instance
config = CoreConfig()

# Expose config and monitor in __all__
__all__.extend(['config', 'performance_monitor', 'CoreConfig', 'CorePerformanceMonitor'])

# Additional utility functions in __all__
__all__.extend([
    'get_core_version',
    'list_available_schedulers', 
    'create_default_scheduler',
    'setup_fractional_training',
    'diagnose_fractional_model'
])

# Module initialization message
logger.info(f"FractionalTorch core module v{__version__} loaded successfully")

# Validate core functionality on import
def _validate_core_functionality():
    """Quick validation that core functionality works."""
    try:
        # Test basic fractional weight creation
        test_weight = FractionalWeight([0.5, 0.25], max_denominator=100)
        
        # Test basic arithmetic
        result_num, result_den = FractionalOps.frac_add_tensors(
            test_weight.numerators, test_weight.denominators,
            test_weight.numerators, test_weight.denominators
        )
        
        # Test scheduler creation
        test_scheduler = create_scheduler('adaptive', patience=10)
        
        logger.debug("Core functionality validation passed")
        
    except Exception as e:
        logger.error(f"Core functionality validation failed: {e}")
        warnings.warn(
            "FractionalTorch core validation failed. Some features may not work correctly.",
            RuntimeWarning
        )

# Run validation
_validate_core_functionality()

# Export version for compatibility
__version__ = __version__
