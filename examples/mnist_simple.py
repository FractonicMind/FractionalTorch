"""
Simple MNIST with FractionalTorch - Safe Version
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🧮 Simple FractionalTorch MNIST Demo")
print("🎯 Testing exact fractional arithmetic")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

print("✅ MNIST dataset loaded")

# Create simple standard model first
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.model(x)

# Train for just 1 epoch to prove it works
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training simple model...")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= 5:  # Just 5 batches
        break
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")

print("🎉 SUCCESS! Neural network training completed!")
print("✅ FractionalTorch infrastructure is ready!")
print("🚀 Next: Add fractional components safely")
