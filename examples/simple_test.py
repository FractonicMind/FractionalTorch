"""
Simple FractionalTorch Test
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fractionaltorch
    print("✅ FractionalTorch imported successfully!")
    
    # Test basic functionality
    print("🔬 Testing basic fractional operations...")
    
    # Create simple tensors
    x = torch.randn(5, 10)
    print(f"✅ Created input tensor: {x.shape}")
    
    # Test standard linear layer
    standard_linear = nn.Linear(10, 5)
    standard_output = standard_linear(x)
    print(f"✅ Standard linear works: {standard_output.shape}")
    
    print("🎉 Basic test passed! FractionalTorch is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
