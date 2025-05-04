#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

if torch.cuda.is_available():
    print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
    DEVICE = "cuda"
else:
    print("CUDA is not available")
    DEVICE = "cpu"

torch.manual_seed(42)

# random noise as input, oh boy...
X = torch.randn(2000, 64)

#print(X.transpose(-1, -2))
print(X)
y = torch.randint(0, 10, (2000,))

model = nn.Sequential(
    nn.Linear(64, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
epochs = 30
dataset = TensorDataset(X,y)

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

for epoch in range(epochs):
    model.train(True)
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(test_loader.dataset)
    val_accuracy = correct / total

    # --- GPU Memory Monitor ---
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2 if torch.cuda.is_available() else 0

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_accuracy*100:.2f}% | "
          f"GPU Mem: {gpu_mem:.2f} MB")
