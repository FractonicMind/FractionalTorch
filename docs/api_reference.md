# FractionalTorch API Reference

## Core Modules

### fractionaltorch.modules

#### FractionalLinear
```python
class FractionalLinear(nn.Module):
    """Exact fractional linear transformation layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: Size of input features
            out_features: Size of output features  
            bias: Whether to include bias term
        """
```

**Key Features:**
- Exact fractional weight representation
- Perfect numerical reproducibility
- Compatible with standard PyTorch training

#### FracLU (Fractional Linear Unit)
```python
class FracLU(nn.Module):
    """Learnable fractional activation function."""
    
    def __init__(self, num_features: int, learnable: bool = True):
        """
        Args:
            num_features: Number of features
            learnable: Whether slopes are learnable parameters
        """
```

**Activation Function:**
```
FracLU(x) = max(α₁/β₁ * x, α₂/β₂ * x)
```

#### FracDropout
```python
class FracDropout(nn.Module):
    """Fractional dropout with learnable rates."""
    
    def __init__(self, p: float = 0.5, learnable: bool = False):
        """
        Args:
            p: Initial dropout probability
            learnable: Whether dropout rate is learnable
        """
```

### fractionaltorch.core

#### FractionalWeight
```python
class FractionalWeight:
    """Exact fractional weight representation."""
    
    def __init__(self, numerator: int, denominator: int):
        """
        Args:
            numerator: Fraction numerator
            denominator: Fraction denominator
        """
```

#### DenominatorScheduler
```python
class DenominatorScheduler:
    """Adaptive precision scheduling during training."""
    
    def __init__(self, initial_max_denom: int = 10, 
                 final_max_denom: int = 1000,
                 strategy: str = 'adaptive'):
        """
        Args:
            initial_max_denom: Starting precision level
            final_max_denom: Final precision level
            strategy: Scheduling strategy ('adaptive', 'linear', 'exponential')
        """
```

### fractionaltorch.auth

#### Authentication Functions
```python
def verify_authentic() -> bool:
    """Verify authentic FractionalTorch by Lev Goukassian."""

def get_author_signature() -> str:
    """Get authentic author signature."""

def get_authentication_info() -> dict:
    """Get complete authentication information."""
```

## Usage Examples

### Basic Model Creation
```python
import torch
import torch.nn as nn
from fractionaltorch.modules import FractionalLinear, FracLU, FracDropout

# Create a simple fractional network
model = nn.Sequential(
    FractionalLinear(784, 256),
    FracLU(256),
    FracDropout(0.3),
    FractionalLinear(256, 128),
    FracLU(128),
    FractionalLinear(128, 10)
)
```

### Converting Existing Models
```python
# Convert standard PyTorch layers to fractional equivalents
def convert_to_fractional(standard_model):
    """Convert standard model to use fractional arithmetic."""
    for name, module in standard_model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with FractionalLinear
            fractional_layer = FractionalLinear(
                module.in_features, 
                module.out_features, 
                module.bias is not None
            )
            # Copy weights
            fractional_layer.weight.data = module.weight.data
            if module.bias is not None:
                fractional_layer.bias.data = module.bias.data
```

### Precision Control
```python
# Set global precision
import fractionaltorch
fractionaltorch.set_global_max_denominator(100)

# Layer-specific precision
layer = FractionalLinear(784, 256)
layer.set_max_denominator(50)

# Get precision statistics
stats = model.get_precision_stats()
print(f"Max denominator: {stats['max_denominator']}")
print(f"Memory overhead: {stats['memory_overhead']:.1f}×")
```

### Authentication Usage
```python
from fractionaltorch.auth import verify_authentic, get_authentication_info

# Verify authenticity
assert verify_authentic(), "Not authentic FractionalTorch!"

# Get authentication details
auth_info = get_authentication_info()
print(f"Author: {auth_info['author']}")
print(f"Framework: {auth_info['framework']}")
print(f"Version: {auth_info['version']}")
```

## Complete Examples

### MNIST Classification
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fractionaltorch.modules import FractionalLinear, FracLU

class FractionalMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            FractionalLinear(784, 128),
            FracLU(128),
            FractionalLinear(128, 64),
            FracLU(64),
            FractionalLinear(64, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# Perfect reproducibility guaranteed!
model = FractionalMNISTNet()
```

### Training Loop
```python
# Standard PyTorch training with perfect reproducibility
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Error Handling

### Common Issues
```python
# Check for authentication failures
try:
    import fractionaltorch
except ImportError as e:
    if "MALICIOUS FORK DETECTED" in str(e):
        print("⚠️ Security Warning: Unauthorized FractionalTorch detected!")
        print("Use only official: pip install fractionaltorch")
```

### Memory Management
```python
# Monitor memory usage
import torch
import gc

def check_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Clear cache periodically
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Integration with PyTorch Ecosystem

### Compatibility
- ✅ **PyTorch 2.0+**: Full compatibility
- ✅ **TorchVision**: Works with standard transforms
- ✅ **Lightning**: Compatible with PyTorch Lightning
- ✅ **Optimizers**: All standard PyTorch optimizers
- ✅ **Loss Functions**: All standard loss functions

### Performance Considerations
- **Memory**: ~80% overhead for exact arithmetic
- **Speed**: ~24% slower training for perfect reproducibility
- **Precision**: Infinite precision vs. floating-point approximations

## Troubleshooting

### Import Issues
```python
# Verify installation
try:
    import fractionaltorch
    print("✅ FractionalTorch imported successfully")
    print(f"Author: {fractionaltorch.get_author_signature()}")
except ImportError:
    print("❌ FractionalTorch not found. Install with: pip install fractionaltorch")
```

### Performance Optimization
```python
# Enable optimizations
fractionaltorch.set_memory_efficient(True)
fractionaltorch.set_precision_mode('balanced')  # 'fast', 'balanced', 'high'
```

For more detailed examples, see the `examples/` directory with comprehensive tutorials and benchmarks.
