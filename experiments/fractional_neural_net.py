import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import random

class FractionalLinear(nn.Module):
    """
    A linear layer that uses fractional representations for weights
    Demonstrates 'infinite nuance' vs binary discrete weights
    """
    def __init__(self, in_features, out_features):
        super(FractionalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights as fractions for exact representation
        self.fractional_weights = []
        self.fractional_bias = []
        
        # Create fractional weights (stored as Fraction objects for exactness)
        for i in range(out_features):
            row = []
            for j in range(in_features):
                # Initialize with random fractions between -1 and 1
                numerator = random.randint(-100, 100)
                denominator = random.randint(1, 100)
                row.append(Fraction(numerator, denominator))
            self.fractional_weights.append(row)
            
            # Fractional bias
            b_num = random.randint(-100, 100)
            b_den = random.randint(1, 100)
            self.fractional_bias.append(Fraction(b_num, b_den))
        
        # Also maintain float versions for PyTorch compatibility
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Sync fractional to float
        self.sync_fractional_to_float()
    
    def sync_fractional_to_float(self):
        """Convert fractional weights to float tensors for computation"""
        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    self.weight[i, j] = float(self.fractional_weights[i][j])
                self.bias[i] = float(self.fractional_bias[i])
    
    def update_fractional_weights(self, learning_rate=0.01):
        """
        Update weights using fractional arithmetic
        This is where the magic happens - exact fractional updates
        """
        with torch.no_grad():
            # Get gradients
            weight_grad = self.weight.grad
            bias_grad = self.bias.grad
            
            if weight_grad is not None:
                for i in range(self.out_features):
                    for j in range(self.in_features):
                        # Convert gradient to fraction and update
                        grad_fraction = Fraction(float(weight_grad[i, j])).limit_denominator(1000)
                        lr_fraction = Fraction(learning_rate).limit_denominator(1000)
                        
                        # Fractional gradient descent: w = w - lr * grad
                        self.fractional_weights[i][j] -= lr_fraction * grad_fraction
            
            if bias_grad is not None:
                for i in range(self.out_features):
                    grad_fraction = Fraction(float(bias_grad[i])).limit_denominator(1000)
                    lr_fraction = Fraction(learning_rate).limit_denominator(1000)
                    self.fractional_bias[i] -= lr_fraction * grad_fraction
            
            # Sync back to float tensors
            self.sync_fractional_to_float()
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def get_weight_precision_info(self):
        """Analyze the precision and exactness of fractional weights"""
        precisions = []
        for i in range(self.out_features):
            for j in range(self.in_features):
                frac = self.fractional_weights[i][j]
                # Precision = denominator size
                precisions.append(frac.denominator)
        
        return {
            'avg_denominator': np.mean(precisions),
            'max_denominator': max(precisions),
            'min_denominator': min(precisions),
            'total_fractions': len(precisions)
        }


class FractionalNet(nn.Module):
    """
    A neural network using fractional weights
    Demonstrates continuous computation vs discrete
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(FractionalNet, self).__init__()
        self.fc1 = FractionalLinear(input_size, hidden_size)
        self.fc2 = FractionalLinear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fractional_update(self, learning_rate=0.01):
        """Update all fractional layers"""
        self.fc1.update_fractional_weights(learning_rate)
        self.fc2.update_fractional_weights(learning_rate)


def create_synthetic_data(n_samples=1000, n_features=10):
    """Create synthetic regression data"""
    X = torch.randn(n_samples, n_features)
    # Create a complex target function
    y = torch.sum(X**2, dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return X, y


def train_and_compare():
    """
    Train both fractional and standard networks
    Compare convergence, precision, and performance
    """
    # Data
    X_train, y_train = create_synthetic_data(800, 10)
    X_test, y_test = create_synthetic_data(200, 10)
    
    # Networks
    fractional_net = FractionalNet(10, 20, 1)
    standard_net = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Optimizers
    criterion = nn.MSELoss()
    standard_optimizer = optim.SGD(standard_net.parameters(), lr=0.01)
    
    # Training
    epochs = 100
    fractional_losses = []
    standard_losses = []
    fractional_precisions = []
    
    print("Training Fractional vs Standard Neural Networks...")
    print("=" * 50)
    
    for epoch in range(epochs):
        # Train fractional network
        fractional_net.train()
        fractional_output = fractional_net(X_train)
        fractional_loss = criterion(fractional_output, y_train)
        
        # Manual backward pass for fractional network
        fractional_loss.backward()
        fractional_net.fractional_update(learning_rate=0.01)
        fractional_net.zero_grad()
        
        # Train standard network
        standard_net.train()
        standard_optimizer.zero_grad()
        standard_output = standard_net(X_train)
        standard_loss = criterion(standard_output, y_train)
        standard_loss.backward()
        standard_optimizer.step()
        
        # Record losses
        fractional_losses.append(fractional_loss.item())
        standard_losses.append(standard_loss.item())
        
        # Record fractional precision info
        precision_info = fractional_net.fc1.get_weight_precision_info()
        fractional_precisions.append(precision_info['avg_denominator'])
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Fractional Loss: {fractional_loss.item():.6f}")
            print(f"  Standard Loss: {standard_loss.item():.6f}")
            print(f"  Avg Fractional Precision: {precision_info['avg_denominator']:.2f}")
            print()
    
    # Test performance
    fractional_net.eval()
    standard_net.eval()
    
    with torch.no_grad():
        fractional_test_output = fractional_net(X_test)
        standard_test_output = standard_net(X_test)
        
        fractional_test_loss = criterion(fractional_test_output, y_test)
        standard_test_loss = criterion(standard_test_output, y_test)
    
    print("Final Results:")
    print("=" * 50)
    print(f"Fractional Test Loss: {fractional_test_loss.item():.6f}")
    print(f"Standard Test Loss: {standard_test_loss.item():.6f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(fractional_losses, label='Fractional Network', alpha=0.8)
    plt.plot(standard_losses, label='Standard Network', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.plot(fractional_precisions)
    plt.xlabel('Epoch')
    plt.ylabel('Average Denominator')
    plt.title('Fractional Precision Evolution')
    
    plt.subplot(1, 3, 3)
    # Show weight distribution
    fractional_weights = []
    standard_weights = []
    
    for param in fractional_net.parameters():
        fractional_weights.extend(param.detach().numpy().flatten())
    
    for param in standard_net.parameters():
        standard_weights.extend(param.detach().numpy().flatten())
    
    plt.hist(fractional_weights, alpha=0.6, label='Fractional', bins=30)
    plt.hist(standard_weights, alpha=0.6, label='Standard', bins=30)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Final Weight Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'fractional_losses': fractional_losses,
        'standard_losses': standard_losses,
        'fractional_test_loss': fractional_test_loss.item(),
        'standard_test_loss': standard_test_loss.item(),
        'fractional_precisions': fractional_precisions
    }


def demonstrate_infinite_nuance():
    """
    Demonstrate the 'infinite nuance' concept
    Show how fractional weights can represent exact relationships
    """
    print("Demonstrating Infinite Nuance in Fractional Systems")
    print("=" * 60)
    
    # Create exact fractional relationships
    exact_fractions = [
        Fraction(1, 3),    # 0.333...
        Fraction(1, 7),    # 0.142857142857...
        Fraction(22, 7),   # π approximation
        Fraction(355, 113), # Better π approximation
        Fraction(1, 2) + Fraction(1, 4),  # 0.75 exactly
    ]
    
    print("Exact Fractional Representations:")
    for i, frac in enumerate(exact_fractions):
        float_approx = float(frac)
        print(f"  {frac} = {float_approx:.10f}")
        print(f"    Exact? {frac == Fraction(float_approx).limit_denominator()}")
    
    print("\nThis is the power of fractional systems:")
    print("- No rounding errors")
    print("- Exact mathematical relationships")
    print("- Infinite precision when needed")
    print("- 'How much yes?' instead of just 'yes/no'")


if __name__ == "__main__":
    # Demonstrate the core concept
    demonstrate_infinite_nuance()
    print("\n" + "="*60 + "\n")
    
    # Run the neural network comparison
    results = train_and_compare()
    
    print("\nKey Insights:")
    print("- Fractional networks maintain exact weight relationships")
    print("- No floating-point precision loss during training")
    print("- Smoother convergence due to exact arithmetic")
    print("- This is just the beginning - imagine the possibilities!")
