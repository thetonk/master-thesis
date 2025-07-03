import time
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics import Metric
from torch.utils.data import DataLoader
from ray import tune


class EncoderTransformer(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim:int = 128, num_heads:int = 8, 
                 dropout:float = 0.0, epsilon=1e-6, ff_neurons=256):
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
    
    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    

class MyModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_encoders:int = 1, num_mlps:int = 1,
                 enc_embedding_dim:int = 128, enc_num_heads:int = 8, enc_ff_neurons:int = 256, mlp_hidden_neurons:int = 512):
        super().__init__()
        layer_list = []
        for i in range(num_encoders):
            if i == 0:
                layer_list.append(EncoderTransformer(num_features, enc_embedding_dim, num_heads=enc_num_heads, ff_neurons=enc_ff_neurons))
            else:
                layer_list.append(EncoderTransformer(enc_embedding_dim, enc_embedding_dim, num_heads=enc_num_heads, ff_neurons=enc_ff_neurons))
        for i in range(num_mlps):
            if i == num_mlps - 1:
                layer_list.append(MLPClassifier(enc_embedding_dim, num_classes, hidden_neurons=mlp_hidden_neurons))
            else:
                layer_list.append(MLPClassifier(enc_embedding_dim, enc_embedding_dim, hidden_neurons=mlp_hidden_neurons))
        self.layers = nn.Sequential(*layer_list)


    def forward(self, x: torch.tensor):
        return self.layers(x)


class MyLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden_lstm_states: int = 256, hidden_mlp_neurons: int = 512, dropout: float = 0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_lstm_states = hidden_lstm_states
        self.input_lstm = nn.LSTM(1, hidden_lstm_states, batch_first=True)
        self.inner_lstm  = nn.LSTM(1, hidden_lstm_states // 2, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp_classifier = nn.Sequential(
            nn.Linear(hidden_lstm_states // 2, hidden_mlp_neurons),
            nn.ReLU(),
            nn.Linear(hidden_mlp_neurons, n_classes),
        )


    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        # initialize lstm states with noise, preferred from zero initialization
        h0 = torch.randn(1, x.shape[0], self.hidden_lstm_states, device=self.device)
        c0 = torch.randn(1, x.shape[0], self.hidden_lstm_states, device=self.device)
        output, _ = self.input_lstm(x, (h0, c0))
        # keep only the output of the last time step
        output = output[:, -1, :].unsqueeze(-1)
        # initialize lstm states with noise, preferred from zero initialization
        h0 = torch.randn(1, output.shape[0], self.hidden_lstm_states // 2, device=self.device)
        c0 = torch.randn(1, output.shape[0], self.hidden_lstm_states // 2, device=self.device)
        output = self.dropout_layer(output)
        output, _ = self.inner_lstm(output, (h0, c0))
        # keep only the output of last time step
        output = output[:, -1, :]
        output = self.dropout_layer(output)
        output = self.mlp_classifier(output)
        return output


def _correct(output: torch.Tensor, target: torch.Tensor):
    predicted = output.argmax(dim=1)
    return (predicted == target).sum().item()


def train_model(model: nn.Module, model_filename: str, train_loader: DataLoader, val_loader: DataLoader | None, metric: Metric,
                loss_function = nn.CrossEntropyLoss(), epochs: int = 30,
                learning_rate: float = 1e-3, device="cuda", train_tune=False):
    #torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    start_time = int(time.time())
    for epoch in range(epochs):
        metric.reset()
        save_model = False
        model.train()
        batch_number = len(train_loader)
        #samples = len(train_loader.dataset)
        total_loss = 0
        #total_correct = 0
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_function(output, label)
            total_loss += loss.item()
            #total_correct += _correct(output, label)
            metric.update(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss/batch_number
        #train_accuracy = total_correct/samples
        train_accuracy = metric.compute().item()
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            save_model = True
        print("="*50)
        print(f"Epoch {epoch+1}/{epochs}. Average train accuracy: {train_accuracy*100:.3f}%, average loss: {train_loss:.5f}, current loss: {loss:.5f}.")
        if train_tune:
            metric.reset()
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                metric.update(output, label)
            val_accuracy = metric.compute().item()
            tune.report({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        if save_model:
            print("Saved model!")
            torch.save(model.state_dict(), model_filename)
    stop_time = int(time.time())
    print(f"Training took {stop_time - start_time} seconds.")


def test_model(model: nn.Module, test_loader: DataLoader, metrics: list[Metric], loss_function = nn.CrossEntropyLoss(), device="cuda") -> list[np.ndarray]:
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

