# Getting Started with FractionalTorch

## Installation

### Simple Installation
```bash
pip install fractionaltorch
git clone https://github.com/FractonicMind/FractionalTorch
cd FractionalTorch
pip install -e ".[dev]"import torch
import torch.nn as nn
from fractionaltorch.modules import FractionalLinear, FracLU, FracDropout

# Build exact arithmetic neural network
model = nn.Sequential(
    FractionalLinear(784, 256),
    FracLU(256),
    FracDropout(0.3),
    FractionalLinear(256, 10)
)

# Perfect reproducibility guaranteed!
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)from fractionaltorch.auth import verify_authentic, get_author_signature

print(f"Authentic: {verify_authentic()}")
print(f"Author: {get_author_signature()}")
