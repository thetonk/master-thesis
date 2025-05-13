import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import random_split, DataLoader, TensorDataset

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

if __name__ == "__main__":
    batch = 64
    torchinfo.summary(EncoderTransformer(80), (batch, 1, 80))