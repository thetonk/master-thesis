import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import random_split, DataLoader, TensorDataset

import dataset_utils

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


torch.manual_seed(42)

if torch.cuda.is_available():
    print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
    DEVICE = "cuda"
else:
    print("CUDA is not available")
    DEVICE = "cpu"


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
        for _ in range(num_encoders):
            layer_list.append(EncoderTransformer(num_features))
        for _ in range(num_mlps):
            layer_list.append(MLPClassifier(128, num_classes))
        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.layers(x)


def _correct(output: torch.Tensor, target: torch.Tensor):
    predicted = output.argmax(dim=1)
    return (predicted == target).sum().item()


def train_model(model: nn.Module, train_loader: DataLoader, loss_function = nn.CrossEntropyLoss(), epochs: int = 30, learning_rate: int = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        batch_number = len(train_loader)
        samples = len(train_loader.dataset)
        total_loss = 0
        total_correct = 0
        for data, label in train_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            output = model(data)
            loss = loss_function(output, label)
            total_loss += loss.item()
            total_correct += _correct(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss/batch_number
        train_accuracy = total_correct/samples
        print("="*50)
        print(f"Epoch {epoch+1}/{epochs}. Average accuracy: {train_accuracy*100:.3f}%, average loss: {train_loss:.5f}, current loss: {loss:.5f}.")


def test_model(model: nn.Module, test_loader: DataLoader, loss_function = nn.CrossEntropyLoss()):
    model.eval()
    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            output = model(data)
            loss = loss_function(output, label)
            test_loss = loss.item()
            total_correct += _correct(output, label)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Testset accuracy: {100*accuracy:.3f}%, average loss: {test_loss}")


if __name__ == "__main__":
    X, y = dataset_utils.dataset_to_tensor(dataset_utils.MERGED_DATASET_PATH, "Label")
    # replace nans and infs with the mean of corresponding column
    invalid_values_mask = torch.isnan(X) | torch.isinf(X)
    X[invalid_values_mask] = torch.nan
    col_means = torch.nanmean(X, dim=0)
    X[invalid_values_mask] = col_means.unsqueeze(0).expand(X.shape[0], X.shape[1])[invalid_values_mask]
    dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    batch_size = 2048
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)
    num_features = X.shape[1]
    num_classes = len(y.unique())
    print(f"# of features: {num_features}, # of classes: {num_classes}, datatype: {X.dtype}")
    model = MyModel(num_features, num_classes).to(DEVICE)
    torchinfo.summary(model, (batch_size, num_features))
    print("STARTING TRAINING SESSION!!!")
    train_model(model, train_loader, epochs=5)
    test_model(model, test_loader)