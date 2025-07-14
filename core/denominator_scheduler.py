"""
FractionalTorch Core: Denominator Scheduler

This module implements adaptive scheduling strategies for controlling the maximum
denominator in fractional representations during neural network training. The
scheduler balances computational efficiency with numerical precision.

Author: Lev Goukassian
License: MIT
"""

import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SchedulerState:
    """State information for the denominator scheduler."""
    step: int = 0
    epoch: int = 0
    current_max_denom: int = 10
    loss_history: List[float] = None
    best_loss: float = float('inf')
    plateau_count: int = 0
    last_improvement_step: int = 0
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []


class BaseDenominatorScheduler(ABC):
    """
    Abstract base class for denominator scheduling strategies.
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 verbose: bool = True):
        self.initial_max_denom = initial_max_denom
        self.final_max_denom = final_max_denom
        self.verbose = verbose
        self.state = SchedulerState(current_max_denom=initial_max_denom)
        
        if initial_max_denom >= final_max_denom:
            warnings.warn(
                f"initial_max_denom ({initial_max_denom}) >= final_max_denom ({final_max_denom}). "
                "Consider using a smaller initial value."
            )
    
    @abstractmethod
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """
        Compute the next maximum denominator value.
        
        Args:
            loss: Current training loss (optional, depending on strategy)
            epoch: Current epoch (optional, depending on strategy)
            
        Returns:
            Maximum denominator for current step
        """
        pass
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.state = SchedulerState(current_max_denom=self.initial_max_denom)
        if self.verbose:
            logger.info("Denominator scheduler reset to initial state")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for saving/loading."""
        return {
            'initial_max_denom': self.initial_max_denom,
            'final_max_denom': self.final_max_denom,
            'state': self.state.__dict__.copy()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from dictionary."""
        self.initial_max_denom = state_dict['initial_max_denom']
        self.final_max_denom = state_dict['final_max_denom']
        state_data = state_dict['state']
        self.state = SchedulerState(**state_data)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"initial={self.initial_max_denom}, "
                f"final={self.final_max_denom}, "
                f"current={self.state.current_max_denom})")


class ExponentialScheduler(BaseDenominatorScheduler):
    """
    Exponential growth scheduler: max_denom = initial * (final/initial)^(step/total_steps)
    
    Args:
        initial_max_denom: Starting maximum denominator
        final_max_denom: Final maximum denominator
        total_steps: Total number of training steps
        gamma: Exponential growth rate (default: computed from initial/final ratio)
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 total_steps: int = 10000,
                 gamma: Optional[float] = None,
                 verbose: bool = True):
        super().__init__(initial_max_denom, final_max_denom, verbose)
        self.total_steps = total_steps
        
        if gamma is None:
            # Compute gamma to reach final_max_denom at total_steps
            self.gamma = (final_max_denom / initial_max_denom) ** (1.0 / total_steps)
        else:
            self.gamma = gamma
        
        if self.verbose:
            logger.info(f"ExponentialScheduler initialized: gamma={self.gamma:.6f}")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """Compute exponential schedule step."""
        self.state.step += 1
        
        if self.state.step >= self.total_steps:
            self.state.current_max_denom = self.final_max_denom
        else:
            progress = self.state.step / self.total_steps
            self.state.current_max_denom = int(
                self.initial_max_denom * (self.final_max_denom / self.initial_max_denom) ** progress
            )
        
        # Ensure we don't exceed bounds
        self.state.current_max_denom = max(self.initial_max_denom, 
                                          min(self.state.current_max_denom, self.final_max_denom))
        
        if self.verbose and self.state.step % 1000 == 0:
            logger.info(f"Step {self.state.step}: max_denominator = {self.state.current_max_denom}")
        
        return self.state.current_max_denom


class LinearScheduler(BaseDenominatorScheduler):
    """
    Linear growth scheduler: max_denom = initial + (final - initial) * (step / total_steps)
    
    Args:
        initial_max_denom: Starting maximum denominator
        final_max_denom: Final maximum denominator
        total_steps: Total number of training steps
        warmup_steps: Number of steps to keep at initial value
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 total_steps: int = 10000,
                 warmup_steps: int = 0,
                 verbose: bool = True):
        super().__init__(initial_max_denom, final_max_denom, verbose)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
        if self.verbose:
            logger.info(f"LinearScheduler initialized: {warmup_steps} warmup steps")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """Compute linear schedule step."""
        self.state.step += 1
        
        if self.state.step <= self.warmup_steps:
            # Warmup phase: keep initial denominator
            self.state.current_max_denom = self.initial_max_denom
        elif self.state.step >= self.total_steps:
            # Final phase: use final denominator
            self.state.current_max_denom = self.final_max_denom
        else:
            # Linear interpolation
            effective_step = self.state.step - self.warmup_steps
            effective_total = self.total_steps - self.warmup_steps
            progress = effective_step / effective_total
            
            self.state.current_max_denom = int(
                self.initial_max_denom + 
                (self.final_max_denom - self.initial_max_denom) * progress
            )
        
        if self.verbose and self.state.step % 1000 == 0:
            logger.info(f"Step {self.state.step}: max_denominator = {self.state.current_max_denom}")
        
        return self.state.current_max_denom


class AdaptiveScheduler(BaseDenominatorScheduler):
    """
    Adaptive scheduler that increases precision based on training progress.
    Monitors loss plateaus and adjusts denominator accordingly.
    
    Args:
        initial_max_denom: Starting maximum denominator
        final_max_denom: Final maximum denominator
        patience: Steps to wait before considering a plateau
        factor: Multiplication factor for denominator increase
        threshold: Minimum loss improvement to avoid plateau detection
        cooldown: Steps to wait after increasing denominator
        min_delta: Minimum absolute improvement in loss
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 patience: int = 100,
                 factor: float = 1.5,
                 threshold: float = 1e-6,
                 cooldown: int = 50,
                 min_delta: float = 1e-8,
                 verbose: bool = True):
        super().__init__(initial_max_denom, final_max_denom, verbose)
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_delta = min_delta
        self.last_increase_step = 0
        
        if self.verbose:
            logger.info(f"AdaptiveScheduler initialized: patience={patience}, factor={factor}")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """Compute adaptive schedule step."""
        self.state.step += 1
        
        if loss is not None:
            self.state.loss_history.append(loss)
            
            # Check for improvement
            if loss < self.state.best_loss - self.min_delta:
                self.state.best_loss = loss
                self.state.plateau_count = 0
                self.state.last_improvement_step = self.state.step
            else:
                self.state.plateau_count += 1
            
            # Check if we should increase precision
            should_increase = (
                self.state.plateau_count >= self.patience and
                self.state.step - self.last_increase_step >= self.cooldown and
                self.state.current_max_denom < self.final_max_denom
            )
            
            if should_increase:
                old_denom = self.state.current_max_denom
                self.state.current_max_denom = min(
                    int(self.state.current_max_denom * self.factor),
                    self.final_max_denom
                )
                self.last_increase_step = self.state.step
                self.state.plateau_count = 0
                
                if self.verbose:
                    logger.info(
                        f"Step {self.state.step}: Plateau detected, "
                        f"increasing max_denominator: {old_denom} → {self.state.current_max_denom}"
                    )
        
        return self.state.current_max_denom
    
    def get_loss_variance(self, window: int = 10) -> float:
        """Get variance of recent losses."""
        if len(self.state.loss_history) < window:
            return float('inf')
        
        recent_losses = self.state.loss_history[-window:]
        return float(np.var(recent_losses))


class CosineScheduler(BaseDenominatorScheduler):
    """
    Cosine annealing scheduler for denominator precision.
    Follows a cosine curve from initial to final denominator.
    
    Args:
        initial_max_denom: Starting maximum denominator
        final_max_denom: Final maximum denominator
        total_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default: 0.5 for half cycle)
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 total_steps: int = 10000,
                 num_cycles: float = 0.5,
                 verbose: bool = True):
        super().__init__(initial_max_denom, final_max_denom, verbose)
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        
        if self.verbose:
            logger.info(f"CosineScheduler initialized: {num_cycles} cycles over {total_steps} steps")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """Compute cosine schedule step."""
        self.state.step += 1
        
        if self.state.step >= self.total_steps:
            self.state.current_max_denom = self.final_max_denom
        else:
            progress = self.state.step / self.total_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * self.num_cycles * progress))
            
            # Invert cosine so we start low and end high
            cosine_factor = 1.0 - cosine_factor
            
            range_size = self.final_max_denom - self.initial_max_denom
            self.state.current_max_denom = int(
                self.initial_max_denom + range_size * cosine_factor
            )
        
        if self.verbose and self.state.step % 1000 == 0:
            logger.info(f"Step {self.state.step}: max_denominator = {self.state.current_max_denom}")
        
        return self.state.current_max_denom


class StepScheduler(BaseDenominatorScheduler):
    """
    Step-wise scheduler that increases denominator at specific milestones.
    
    Args:
        initial_max_denom: Starting maximum denominator
        final_max_denom: Final maximum denominator
        milestones: List of steps at which to increase denominator
        gamma: Multiplication factor at each milestone
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 milestones: List[int] = None,
                 gamma: float = 2.0,
                 verbose: bool = True):
        super().__init__(initial_max_denom, final_max_denom, verbose)
        
        if milestones is None:
            # Default milestones: exponentially spaced
            milestones = [1000, 3000, 7000, 15000]
        
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.milestone_index = 0
        
        if self.verbose:
            logger.info(f"StepScheduler initialized: milestones={milestones}, gamma={gamma}")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """Compute step schedule step."""
        self.state.step += 1
        
        # Check if we've reached a milestone
        while (self.milestone_index < len(self.milestones) and 
               self.state.step >= self.milestones[self.milestone_index]):
            
            old_denom = self.state.current_max_denom
            self.state.current_max_denom = min(
                int(self.state.current_max_denom * self.gamma),
                self.final_max_denom
            )
            
            if self.verbose:
                logger.info(
                    f"Step {self.state.step}: Milestone reached, "
                    f"max_denominator: {old_denom} → {self.state.current_max_denom}"
                )
            
            self.milestone_index += 1
        
        return self.state.current_max_denom


class CustomScheduler(BaseDenominatorScheduler):
    """
    Custom scheduler that accepts a user-defined function.
    
    Args:
        initial_max_denom: Starting maximum denominator
        final_max_denom: Final maximum denominator
        schedule_func: Function that takes (step, loss, epoch) and returns max_denominator
    """
    
    def __init__(self, 
                 initial_max_denom: int = 10,
                 final_max_denom: int = 1000,
                 schedule_func: Callable[[int, Optional[float], Optional[int]], int] = None,
                 verbose: bool = True):
        super().__init__(initial_max_denom, final_max_denom, verbose)
        
        if schedule_func is None:
            # Default to linear schedule
            def default_func(step, loss, epoch):
                return min(initial_max_denom + step // 100, final_max_denom)
            self.schedule_func = default_func
        else:
            self.schedule_func = schedule_func
        
        if self.verbose:
            logger.info("CustomScheduler initialized with user-defined function")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """Compute custom schedule step."""
        self.state.step += 1
        
        try:
            self.state.current_max_denom = self.schedule_func(self.state.step, loss, epoch)
            # Ensure bounds
            self.state.current_max_denom = max(self.initial_max_denom,
                                              min(self.state.current_max_denom, self.final_max_denom))
        except Exception as e:
            logger.error(f"Custom schedule function failed: {e}")
            # Fallback to current value
            pass
        
        return self.state.current_max_denom


# Convenience factory function
def create_scheduler(strategy: str, **kwargs) -> BaseDenominatorScheduler:
    """
    Factory function to create denominator schedulers.
    
    Args:
        strategy: Scheduler type ('exponential', 'linear', 'adaptive', 'cosine', 'step')
        **kwargs: Additional arguments for the specific scheduler
        
    Returns:
        Configured scheduler instance
        
    Example:
        >>> scheduler = create_scheduler('adaptive', patience=50, factor=2.0)
        >>> max_denom = scheduler.step(loss=0.1)
    """
    strategy_map = {
        'exponential': ExponentialScheduler,
        'linear': LinearScheduler,
        'adaptive': AdaptiveScheduler,
        'cosine': CosineScheduler,
        'step': StepScheduler,
        'custom': CustomScheduler,
    }
    
    if strategy not in strategy_map:
        available = ', '.join(strategy_map.keys())
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")
    
    return strategy_map[strategy](**kwargs)


# Alias for backward compatibility
DenominatorScheduler = AdaptiveScheduler


class SchedulerManager:
    """
    Manager class for handling multiple schedulers and model updates.
    """
    
    def __init__(self, model: torch.nn.Module, scheduler: BaseDenominatorScheduler):
        self.model = model
        self.scheduler = scheduler
        self.fractional_params = []
        
        # Find all fractional parameters
        self._discover_fractional_params()
    
    def _discover_fractional_params(self):
        """Find all FractionalWeight parameters in the model."""
        from .fractional_weight import FractionalWeight
        
        for name, param in self.model.named_parameters():
            if isinstance(param, FractionalWeight):
                self.fractional_params.append((name, param))
        
        logger.info(f"Found {len(self.fractional_params)} fractional parameters")
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> int:
        """
        Update scheduler and apply new denominator to all fractional parameters.
        
        Args:
            loss: Current training loss
            epoch: Current epoch
            
        Returns:
            New maximum denominator value
        """
        new_max_denom = self.scheduler.step(loss, epoch)
        
        # Update all fractional parameters
        for name, param in self.fractional_params:
            param.set_max_denominator(new_max_denom)
        
        return new_max_denom
    
    def get_precision_summary(self) -> Dict[str, Any]:
        """Get summary of precision across all fractional parameters."""
        if not self.fractional_params:
            return {'num_fractional_params': 0}
        
        all_stats = []
        for name, param in self.fractional_params:
            stats = param.get_precision_stats()
            stats['param_name'] = name
            all_stats.append(stats)
        
        # Aggregate statistics
        total_elements = sum(s['total_elements'] for s in all_stats)
        max_denominators = [s['max_denominator'] for s in all_stats]
        
        return {
            'num_fractional_params': len(self.fractional_params),
            'total_fractional_elements': total_elements,
            'global_max_denominator': max(max_denominators),
            'mean_max_denominator': np.mean(max_denominators),
            'scheduler_state': self.scheduler.state.__dict__.copy(),
            'per_param_stats': all_stats
        }
