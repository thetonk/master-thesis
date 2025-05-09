import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    DEVICE = "cuda"
else:
    print("No GPU found, using CPU instead")
    DEVICE = "cpu"

torch.manual_seed(42)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 10)                  
        )
    def forward(self, x):
        return self.layers(x)

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()                               # count number of correct ones

def train(data_loader: DataLoader, model: nn.Module, criterion, optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        
        # Do a forward pass
        output = model(data)
        
        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")

def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(DEVICE)
            target = target.to(DEVICE)
        
            # Do a forward pass
            output = model(data)
        
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")


model = SimpleMLP().to(DEVICE)
print(model)
for parameter in model.parameters():
    print(parameter.size())

batch_size = 64

train_data = datasets.MNIST("../datasets/MNIST",train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST("../datasets/MNIST",train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

for epoch in range(epochs):
    # --- GPU Memory Monitor ---
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2 if torch.cuda.is_available() else 0
    print(f"Epoch: {epoch+1}, GPU memory usage: {gpu_mem:.2f} MiB")
    train(train_loader, model, criterion, optimizer)

test(test_loader, model, criterion)