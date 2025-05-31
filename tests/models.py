import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# CODE TO CHECK ON TRAINER, FOR TRAINING SPEED BOOST. WORKS ON AMPERE GPU DEVICES OR NEWER
# scaler = torch.cuda.amp.GradScaler()

# for data, label in train_loader:
#     data = data.to(DEVICE)
#     label = label.to(DEVICE)
#     optimizer.zero_grad()

#     with torch.cuda.amp.autocast():
#         output = model(data)
#         loss = loss_function(output, label)

#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()


class EncoderTransformer(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim:int = 128, num_heads:int = 8, dropout:float = 0.0, epsilon=1e-6, ff_neurons=256):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(normalized_shape=embedding_dim, eps=epsilon)
        self.ff_net = nn.Sequential(
            nn.Linear(embedding_dim, ff_neurons),
            nn.ReLU(),
            nn.Linear(ff_neurons, embedding_dim)
        )
    
    def forward(self, x):
        input = self.input_projection(x)
        x = self.normalization(input)
        x, _ = self.multihead_attention(x, x, x)
        res = x + input
        x = self.dropout(res)
        x = self.normalization(x)
        x = self.ff_net(x)
        res = x + res
        res = self.normalization(res)
        return res


class MLPClassifier(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, dropout:float = 0.1, hidden_neurons=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, n_classes)
        )

    def forward(self, x):
        return self.layers(x)
    

class MyModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_encoders:int = 1, num_mlps:int = 1):
        super().__init__()
        layer_list = []
        for i in range(num_encoders):
            if i == 0:
                layer_list.append(EncoderTransformer(num_features))
            else:
                layer_list.append(EncoderTransformer(128))
        for i in range(num_mlps):
            if i == num_mlps - 1:
                layer_list.append(MLPClassifier(128, num_classes, hidden_neurons=512))
            else:
                layer_list.append(MLPClassifier(128, 128, hidden_neurons=512))
        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.layers(x)


def _correct(output: torch.Tensor, target: torch.Tensor):
    predicted = output.argmax(dim=1)
    return (predicted == target).sum().item()


def train_model(model: nn.Module, train_loader: DataLoader, loss_function = nn.CrossEntropyLoss(), epochs: int = 30, learning_rate: int = 1e-3, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    save_model = False
    start_time = int(time.time())
    for epoch in range(epochs):
        save_model = False
        model.train()
        batch_number = len(train_loader)
        samples = len(train_loader.dataset)
        total_loss = 0
        total_correct = 0
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_function(output, label)
            total_loss += loss.item()
            total_correct += _correct(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss/batch_number
        train_accuracy = total_correct/samples
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            save_model = True
        print("="*50)
        print(f"Epoch {epoch+1}/{epochs}. Average accuracy: {train_accuracy*100:.3f}%, average loss: {train_loss:.5f}, current loss: {loss:.5f}.")
        if save_model:
            print("Saved model!")
            torch.save(model.state_dict(), "trained_models/best_model.pt")
    stop_time = int(time.time())
    print(f"Training took {stop_time - start_time} seconds.")


def test_model(model: nn.Module, test_loader: DataLoader, metrics, loss_function = nn.CrossEntropyLoss(), device="cuda") -> list[np.ndarray]:
    model.eval()
    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)
    test_loss = 0
    total_correct = 0
    metric_results = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_function(output, label)
            test_loss = loss.item()
            total_correct += _correct(output, label)
            for metric in metrics:
                metric.update(output, label)
    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Testset accuracy: {100*accuracy:.3f}%, average loss: {test_loss}")
    for metric in metrics:
        metric_results.append(metric.compute().cpu().numpy())
    return metric_results

