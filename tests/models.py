import torch
import torch.nn as nn
#import torchinfo
from torcheval.metrics import MulticlassAccuracy
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
import shap
from shap.plots import bar
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
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

SEED = 42
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
    best_accuracy = 0
    save_model = False
    for epoch in range(epochs):
        save_model = False
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
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            save_model = True
        print("="*50)
        print(f"Epoch {epoch+1}/{epochs}. Average accuracy: {train_accuracy*100:.3f}%, average loss: {train_loss:.5f}, current loss: {loss:.5f}.")
        if save_model:
            print("Saved model!")
            torch.save(model.state_dict(), "models/best_model.pt")


def test_model(model: nn.Module, test_loader: DataLoader, metric, loss_function = nn.CrossEntropyLoss()) -> torch.Tensor:
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
            metric.update(output, label)
    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Testset accuracy: {100*accuracy:.3f}%, average loss: {test_loss}")
    return metric.compute()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    csv_dataset = dataset_utils.CSVDataset(dataset_utils.MERGED_DATASET_PATH, "Cat", chunk_size=3e+6)
    csv_dataset.load()
    X, y, category_map, feature_names = csv_dataset.X, csv_dataset.y, csv_dataset.categories, csv_dataset.features
    dataset = TensorDataset(X, y)
    #train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    batch_size = 1500
    folds = 2
    epochs = 5
    num_features = X.shape[1]
    num_classes = len(category_map)
    metric = MulticlassAccuracy(average=None, num_classes=num_classes, device=DEVICE)
    print(f"# of rows: {X.shape[0]}, # of features: {num_features}, # of classes: {num_classes}, datatype: {X.dtype}")
    #model = MyModel(num_features, num_classes).to(DEVICE)
    #torchinfo.summary(model, (batch_size, num_features))
    strat_kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    final_model = None
    for fold, (train_index, test_index) in enumerate(strat_kfold.split(X, y)):
        model = MyModel(num_features, num_classes).to(DEVICE)
        print("-"*50)
        print(f"Fold {fold+1}/{folds}")
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=10)
        test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=10)
        print("STARTING TRAINING SESSION!!!")
        train_model(model, train_loader, epochs=epochs)
        print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
        multiclass_accuracy = test_model(model, test_loader, metric)
        print("Accuracy per class:")
        for i, class_accuracy in enumerate(multiclass_accuracy):
            print(f"{category_map[i]}: {class_accuracy*100} %")
        print("-"*50)
        final_model = model
        break
        
    final_model = final_model.to("cpu")
    shap_batch_loader = DataLoader(dataset, 110, shuffle=True)
    features, _ = next(iter(shap_batch_loader))
    features = features.detach().cpu()
    final_model.eval()
    background = features[:100]
    test_values = features[100:]
    with torch.no_grad():
        base_values = final_model(background).mean(dim=0).numpy()
    explainer = shap.GradientExplainer(final_model, background)
    shap_values = explainer.shap_values(test_values)
    print("base values", base_values)
    print("features", features.shape[1])
    print("test values shape", test_values.shape)
    charts_per_row = 3
    rows = num_classes // charts_per_row + ((num_classes % charts_per_row) != 0)
    fig, axes = plt.subplots(rows, charts_per_row, dpi=300, figsize=(charts_per_row*4, rows*3), constrained_layout=True)
    axes = axes.ravel()
    for i in range(num_classes):
        # Create Explanation object for class 0 (you can loop for others)
        shap_explanation = shap.Explanation(
            values=shap_values[i].T,                           # SHAP values for class i
            base_values=base_values[i],                      # base value for class i
            data=features[100:].numpy(),                  # input data
            feature_names=feature_names
        )
        print(f"shap {i} value shape", shap_values[i].shape)
        bar(shap_explanation, max_display=10, ax=axes[i], show=False)
        # Explicitly set font size for axis labels
        axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=3)
        axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=3)
        # Set tick label font sizes
        for tick in axes[i].get_xticklabels():
            tick.set_fontsize(4)
        for tick in axes[i].get_yticklabels():
            tick.set_fontsize(4)
        for child in axes[i].get_children():
            if isinstance(child, plt.Text):
                # This filters the number labels — skip titles, labels etc.
                if child.get_position()[0] > 0:
                    child.set_fontsize(3)
        # Set title font size
        axes[i].set_title(category_map[i], fontsize=5, pad=4)

    plt.tight_layout(pad=0.8)
    plt.show()
