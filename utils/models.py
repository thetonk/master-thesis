# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import torch
import torch.nn as nn
import enum

class ModelTypes(enum.StrEnum):
    CNN = enum.auto()
    TRANSFORMER = enum.auto()
    LSTM = enum.auto()


class EncoderTransformer(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim:int = 128, num_heads:int = 8, 
                 attn_dropout:float = 0.0, ff_dropout:float = 0.0, epsilon=1e-6, ff_neurons=256):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(ff_dropout)
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
    

class MyTransformerModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_encoders:int = 1, num_mlps:int = 1,
                 enc_embedding_dim:int = 128, enc_num_heads:int = 8, enc_ff_neurons:int = 256,
                 enc_ff_dropout:float = 0, enc_attn_dropout:float = 0,
                 mlp_hidden_neurons:int = 512, mlp_dropout:float = 0.1):
        super().__init__()
        transformer_list = []
        mlp_list = []
        for i in range(num_encoders):
            if i == 0:
                transformer_list.append(EncoderTransformer(num_features, enc_embedding_dim, num_heads=enc_num_heads, ff_neurons=enc_ff_neurons,
                                                     ff_dropout=enc_ff_dropout, attn_dropout=enc_attn_dropout))
            else:
                transformer_list.append(EncoderTransformer(enc_embedding_dim, enc_embedding_dim, num_heads=enc_num_heads, ff_neurons=enc_ff_neurons,
                                                     ff_dropout=enc_ff_dropout, attn_dropout=enc_ff_dropout))
        for i in range(num_mlps):
            if i == num_mlps - 1:
                mlp_list.append(MLPClassifier(enc_embedding_dim, num_classes, hidden_neurons=mlp_hidden_neurons, dropout=mlp_dropout))
            else:
                mlp_list.append(MLPClassifier(enc_embedding_dim, enc_embedding_dim, hidden_neurons=mlp_hidden_neurons, dropout=mlp_dropout))
        self.transformer_layers = nn.Sequential(*transformer_list)
        self.mlp_layers = nn.Sequential(*mlp_list)


    def forward(self, x: torch.tensor):
        x = self.transformer_layers(x)
        x = self.mlp_layers(x)
        return x


class MyLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden_lstm_states: int = 256, hidden_mlp_neurons: int = 512,
                 mlp_dropout: float = 0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_lstm_states = hidden_lstm_states
        self.input_lstm = nn.LSTM(1, hidden_lstm_states, batch_first=True)
        self.inner_lstm  = nn.LSTM(1, hidden_lstm_states // 2, batch_first=True)
        self.dropout_layer = nn.Dropout(mlp_dropout)
        self.mlp_classifier = nn.Sequential(
            nn.Linear(hidden_lstm_states // 2, hidden_mlp_neurons),
            nn.ReLU(),
            nn.Linear(hidden_mlp_neurons, n_classes),
        )

    
    def lstm_layers(self, x: torch.Tensor):
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
        return output


    def forward(self, x: torch.Tensor):
        output = self.lstm_layers(x)
        output = self.dropout_layer(output)
        output = self.mlp_classifier(output)
        return output


class MyCNNModel(nn.Module):
    def __init__(self, n_classes, kernel_size, padding, hidden_mlp_neurons, mlp_dropout=0.1, conv_layers=10):
        super().__init__()
        cnn_layers = []
        for _ in range(conv_layers):
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding),
                nn.GroupNorm(1, 1),
                nn.LeakyReLU()
            ))
        self.cnn_model = nn.Sequential(*cnn_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp_classifier = nn.Sequential(
            nn.LazyLinear(hidden_mlp_neurons),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            #nn.Linear(mlp_hidden_neurons, mlp_hidden_neurons),
            #nn.ReLU(),
            #nn.Dropout(mlp_dropout),
            nn.Linear(hidden_mlp_neurons, n_classes)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn_model(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.mlp_classifier(x)
        return x
