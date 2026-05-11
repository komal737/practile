import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# Load MNIST dataset
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_data,
    batch_size=512,
    shuffle=True
)

# Neural Network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
for epoch in range(3):

    for data, target in train_loader:

        # Move data to GPU
        data = data.to(device)
        target = target.to(device)

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = loss_fn(output, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

print("HPC AI/ML Training Completed")
