# FractionalTorch

FractionalTorch is an experimental PyTorch module that implements **NeuroFraction-of-1.5**, a creative approach to generative AI that embraces glitch, artifacts, and creative noise.

## Why NeuroFraction-of-1.5?
Perfect AI is boring. NeuroFraction-of-1.5 introduces fractional dropout to create **controlled chaos** in real-time video generation—transforming mistakes into features.

## Quick Example
```python
import torch
from fractionaldropout import FractionalDropout

model = MyModel()
model.add_module('fractional_dropout', FractionalDropout(p=0.33))
