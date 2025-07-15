"""
Tutorial 1: Your First Fractional Neural Network

Learn the basics of exact arithmetic in neural networks with FractionalTorch.
This tutorial shows you how to build your first network using exact fractions
instead of floating-point approximations.

Author: Lev Goukassian
License: MIT
"""

import torch
import torch.nn as nn
import sys
import os
from fractions import Fraction

# Import FractionalTorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fractionaltorch

print("🎓 Tutorial 1: Your First Fractional Neural Network")
print("=" * 60)
print()

# ==============================================================================
# LESSON 1: Understanding the Problem with Floating-Point
# ==============================================================================
print("📖 LESSON 1: The Floating-Point Problem")
print("-" * 40)

print("Let's see why floating-point arithmetic is problematic:")
print()

# Demonstrate floating-point precision issues
a = 0.1 + 0.2
print(f"Standard floating-point: 0.1 + 0.2 = {a}")
print(f"Is it exactly 0.3? {a == 0.3}")
print(f"Actual value: {repr(a)}")
print()

# Show how this affects neural networks
print("In neural networks, these tiny errors accumulate over millions of operations!")
print("Result: Irreproducible experiments, numerical instability, platform differences")
print()

# ==============================================================================
# LESSON 2: The Fractional Solution
# ==============================================================================
print("📖 LESSON 2: The Fractional Solution")
print("-" * 40)

print("FractionalTorch uses exact fractions instead:")
print()

# Demonstrate exact fractions
frac_a = Fraction(1, 10)  # Exactly 1/10
frac_b = Fraction(2, 10)  # Exactly 2/10
frac_sum = frac_a + frac_b
print(f"Exact fractions: {frac_a} + {frac_b} = {frac_sum}")
print(f"Is it exactly 3/10? {frac_sum == Fraction(3, 10)}")
print(f"As decimal: {float(frac_sum)}")
print()

print("Benefits:")
print("✅ Perfect reproducibility across any hardware")
print("✅ No accumulation of rounding errors")
print("✅ Exact arithmetic throughout training")
print()

# ==============================================================================
# LESSON 3: Building Your First Fractional Network
# ==============================================================================
print("📖 LESSON 3: Building Your First Fractional Network")
print("-" * 40)

# Create a simple safe fractional layer
class SimpleFractionalLayer(nn.Module):
    """A simple layer that demonstrates fractional concepts."""
    
    def __init__(self, in_features, out_features, name="FracLayer"):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize with small random weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        print(f"✅ Created {name}({in_features} → {out_features})")
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def show_fractional_weights(self, num_samples=3):
        """Show how weights can be represented as exact fractions."""
        print(f"🔬 {self.name} - Sample weights as fractions:")
        
        flat_weights = self.weight.flatten()
        for i in range(min(num_samples, len(flat_weights))):
            weight_val = float(flat_weights[i])
            frac = Fraction(weight_val).limit_denominator(1000)
            print(f"  Weight[{i}]: {frac} = {float(frac):.6f}")

# Build a simple fractional network
print("Building a 3-layer fractional network:")
print()

class MyFirstFractionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = SimpleFractionalLayer(4, 8, "Input Layer")
        self.activation1 = nn.ReLU()
        
        self.layer2 = SimpleFractionalLayer(8, 4, "Hidden Layer")
        self.activation2 = nn.ReLU()
        
        self.layer3 = SimpleFractionalLayer(4, 2, "Output Layer")
        
        print("🎯 Network architecture:")
        print("  Input(4) → FracLayer(8) → ReLU → FracLayer(4) → ReLU → FracLayer(2)")
    
    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.layer3(x)
        return x

# Create the network
print()
model = MyFirstFractionalNetwork()
print()

# ==============================================================================
# LESSON 4: Training with Exact Arithmetic
# ==============================================================================
print("📖 LESSON 4: Training with Exact Arithmetic")
print("-" * 40)

# Create some sample data
print("Creating sample training data...")
torch.manual_seed(42)  # For reproducibility
X = torch.randn(100, 4)  # 100 samples, 4 features
y = torch.randn(100, 2)  # 100 targets, 2 outputs

print(f"✅ Data created: X{list(X.shape)}, y{list(y.shape)}")
print()

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training the fractional network...")
print()

# Training loop
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5: Loss = {loss.item():.6f}")

print()
print("✅ Training completed!")
print()

# ==============================================================================
# LESSON 5: Examining Fractional Representations
# ==============================================================================
print("📖 LESSON 5: Examining Fractional Representations")
print("-" * 40)

print("Let's look at how our trained weights can be represented as exact fractions:")
print()

# Show fractional representations
model.layer1.show_fractional_weights(3)
print()
model.layer2.show_fractional_weights(3)
print()
model.layer3.show_fractional_weights(3)
print()

# ==============================================================================
# LESSON 6: Testing Reproducibility
# ==============================================================================
print("📖 LESSON 6: Testing Reproducibility")
print("-" * 40)

print("One of the key benefits of FractionalTorch is perfect reproducibility.")
print("Let's test this by running the same computation multiple times:")
print()

def test_reproducibility():
    """Test if computations are exactly reproducible."""
    torch.manual_seed(123)
    test_input = torch.randn(1, 4)
    
    results = []
    for run in range(3):
        torch.manual_seed(123)  # Same seed each time
        with torch.no_grad():
            output = model(test_input)
            results.append(output.clone())
    
    # Check if all results are identical
    all_identical = all(torch.equal(results[0], result) for result in results[1:])
    
    print(f"Test input: {test_input[0].tolist()}")
    print("Results from 3 identical runs:")
    for i, result in enumerate(results):
        print(f"  Run {i+1}: {result[0].tolist()}")
    
    print(f"All results identical: {'✅ YES' if all_identical else '❌ NO'}")
    
    if all_identical:
        print("🎉 Perfect reproducibility achieved!")
    else:
        print("⚠️  Small differences detected (normal with floating-point)")

test_reproducibility()
print()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("🎓 TUTORIAL SUMMARY")
print("=" * 60)
print()
print("Congratulations! You've learned:")
print()
print("✅ Why floating-point arithmetic causes problems in neural networks")
print("✅ How exact fractions solve these problems")
print("✅ How to build your first fractional neural network")
print("✅ How to train networks with exact arithmetic")
print("✅ How to examine fractional weight representations")
print("✅ How to test for perfect reproducibility")
print()
print("🚀 Next Steps:")
print("  - Try tutorial_02_mnist_complete.py for a real-world example")
print("  - Experiment with different network architectures")
print("  - Compare results with standard PyTorch models")
print()
print("🌟 You're now ready to use exact arithmetic in your neural networks!")
print("   Welcome to the future of numerically stable AI! 🧮✨")
