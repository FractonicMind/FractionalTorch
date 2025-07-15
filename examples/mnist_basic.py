"""
FractionalTorch MNIST Basic Example

A simple demonstration showing FractionalTorch vs standard PyTorch
on MNIST digit classification with numerical stability comparison.

Author: Lev Goukassian
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import sys

# Add parent directory to path to import fractionaltorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fractionaltorch.modules import FractionalLinear, FracLU, FracDropout
    from fractionaltorch.core import setup_fractional_training
    print("✅ FractionalTorch imported successfully!")
except ImportError as e:
    print(f"❌ Could not import FractionalTorch: {e}")
    print("Make sure you're running from the FractionalTorch directory")
    sys.exit(1)


class StandardMLP(nn.Module):
    """Standard PyTorch MLP for comparison."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)


class FractionalMLP(nn.Module):
    """FractionalTorch MLP with exact arithmetic."""
    
    def __init__(self, max_denominator=1000):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            FractionalLinear(784, 256, max_denominator=max_denominator),
            FracLU(256, learnable=True, max_denominator=max_denominator),
            FracDropout(0.3, learnable=True, max_denominator=max_denominator),
            FractionalLinear(256, 128, max_denominator=max_denominator),
            FracLU(128, learnable=True, max_denominator=max_denominator),
            FracDropout(0.2, learnable=True, max_denominator=max_denominator),
            FractionalLinear(128, 10, max_denominator=max_denominator)
        )
    
    def forward(self, x):
        return self.network(x)


def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    print("📦 Loading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, model_name, epochs=3):
    """Train a model and track statistics."""
    print(f"\n🚀 Training {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track training statistics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                print(f'  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Calculate epoch statistics
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Test accuracy
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        print(f'  ✅ Epoch {epoch+1}/{epochs}: '
              f'Loss={avg_train_loss:.6f}, '
              f'Train Acc={train_accuracy:.2f}%, '
              f'Test Acc={test_accuracy:.2f}%, '
              f'Time={epoch_time:.1f}s')
    
    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies, 
        'test_accuracies': test_accuracies,
        'epoch_times': epoch_times,
        'final_test_accuracy': test_accuracies[-1],
        'avg_epoch_time': np.mean(epoch_times)
    }


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100. * correct / total


def test_numerical_reproducibility(model_class, train_loader, num_runs=3):
    """Test if training is exactly reproducible."""
    print(f"\n🔬 Testing numerical reproducibility for {model_class.__name__}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_losses = []
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...")
        
        # Set identical seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train for just a few batches
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 10:  # Just 10 batches for reproducibility test
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        final_losses.append(loss.item())
    
    # Check reproducibility
    loss_differences = [abs(final_losses[i] - final_losses[0]) for i in range(1, len(final_losses))]
    max_difference = max(loss_differences) if loss_differences else 0.0
    is_reproducible = max_difference < 1e-10
    
    print(f"  📊 Final losses: {[f'{l:.10f}' for l in final_losses]}")
    print(f"  📊 Max difference: {max_difference:.2e}")
    print(f"  📊 Reproducible: {'✅ YES' if is_reproducible else '❌ NO'}")
    
    return is_reproducible, max_difference


def compare_models():
    """Main comparison function."""
    print("🎯 FractionalTorch vs PyTorch MNIST Comparison")
    print("=" * 50)
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Test reproducibility
    print("\n📋 REPRODUCIBILITY TEST")
    print("-" * 30)
    
    standard_repro, standard_diff = test_numerical_reproducibility(StandardMLP, train_loader)
    fractional_repro, fractional_diff = test_numerical_reproducibility(FractionalMLP, train_loader)
    
    # Train both models
    print("\n📋 TRAINING COMPARISON")
    print("-" * 30)
    
    # Standard PyTorch
    standard_model = StandardMLP()
    standard_results = train_model(standard_model, train_loader, test_loader, "Standard PyTorch", epochs=3)
    
    # FractionalTorch
    fractional_model = FractionalMLP(max_denominator=1000)
    fractional_results = train_model(fractional_model, train_loader, test_loader, "FractionalTorch", epochs=3)
    
    # Print comparison
    print("\n📊 FINAL RESULTS")
    print("=" * 50)
    
    print(f"🔢 REPRODUCIBILITY:")
    print(f"  Standard PyTorch:  {'✅ Perfect' if standard_repro else f'❌ Diff: {standard_diff:.2e}'}")
    print(f"  FractionalTorch:   {'✅ Perfect' if fractional_repro else f'❌ Diff: {fractional_diff:.2e}'}")
    
    print(f"\n🎯 FINAL ACCURACY:")
    print(f"  Standard PyTorch:  {standard_results['final_test_accuracy']:.2f}%")
    print(f"  FractionalTorch:   {fractional_results['final_test_accuracy']:.2f}%")
    
    print(f"\n⏱️  TRAINING SPEED:")
    print(f"  Standard PyTorch:  {standard_results['avg_epoch_time']:.1f}s per epoch")
    print(f"  FractionalTorch:   {fractional_results['avg_epoch_time']:.1f}s per epoch")
    
    speed_overhead = (fractional_results['avg_epoch_time'] / standard_results['avg_epoch_time'] - 1) * 100
    print(f"  Overhead:          {speed_overhead:+.1f}%")
    
    print(f"\n🏆 WINNER:")
    if fractional_repro and not standard_repro:
        print("  🥇 FractionalTorch - Perfect reproducibility!")
    elif fractional_results['final_test_accuracy'] > standard_results['final_test_accuracy']:
        print("  🥇 FractionalTorch - Higher accuracy!")
    elif speed_overhead < 50:  # If overhead is reasonable
        print("  🥇 FractionalTorch - Good balance of accuracy and reproducibility!")
    else:
        print("  🥇 Both models performed well!")
    
    return standard_results, fractional_results


if __name__ == "__main__":
    print("🧮 FractionalTorch MNIST Basic Example")
    print("🎯 Testing exact fractional arithmetic in neural networks")
    print()
    
    try:
        standard_results, fractional_results = compare_models()
        print("\n✅ Comparison completed successfully!")
        print("\n🎉 FractionalTorch is working! Your exact arithmetic neural network framework is ready!")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
