# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import Metric
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall
import numpy as np
import matplotlib.pyplot as plt
from ray import tune
import shap
from shap.plots import bar


def _correct(output: torch.Tensor, target: torch.Tensor):
    predicted = output.argmax(dim=1)
    return (predicted == target).sum().item()


class EarlyStopping():
    def __init__(self, patience: int, delta: float) -> None:
        self._stop_counter = 0
        self._best_loss = None
        self.patience = patience
        self.delta = delta


    def check_stop(self, current_loss: float) -> bool:
        if self._best_loss is None or current_loss <= self._best_loss + self.delta:
            self._best_loss = current_loss
        else:
            self._stop_counter += 1
            if self._stop_counter >= self.patience:
                print("Early stopping training!")
                return True
        return False


def train_model(model: nn.Module, model_filename: str, train_loader: DataLoader, val_loader: DataLoader | None, metric: Metric,
                loss_function = nn.CrossEntropyLoss(), epochs: int = 30,
                learning_rate: float = 1e-3, device=torch.device("cuda"), train_tune=False, early_stopper: EarlyStopping | None =None):
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
        loss = 0
        #total_correct = 0
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_function(output, label)
            total_loss += loss.item()
            #total_correct += _correct(output, label)
            output = output.argmax(dim=1)
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
        if val_loader is not None:
            with torch.no_grad():
                metric.reset()
                batch_number = len(val_loader)
                total_loss = 0
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    output = model(data)
                    loss = loss_function(output, label)
                    total_loss += loss.item()
                    output = output.argmax(dim=1)
                    metric.update(output, label)
                val_loss = total_loss / batch_number
                val_accuracy = metric.compute().item()
                print(f"Validation accuracy: {val_accuracy*100:.3f}%, validation loss: {val_loss:.5f}")
                if train_tune:
                    tune.report({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
                # check for early stopping
                if early_stopper is not None:
                    if early_stopper.check_stop(val_loss):
                        break
        elif train_tune and val_loader is None:
            print("Enabled tuning, but no validation loader is set!")
            tune.report({"train_accuracy": train_accuracy})
        if save_model:
            print("Saved model!")
            torch.save(model.state_dict(), model_filename)
    stop_time = int(time.time())
    print(f"Training took {stop_time - start_time} seconds.")


def test_model(model: nn.Module, test_loader: DataLoader, metrics: list[Metric],
               loss_function = nn.CrossEntropyLoss(), device: torch.device = torch.device("cuda")) -> list[np.ndarray]:
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
            output = output.argmax(dim=1)
            for metric in metrics:
                metric.update(output, label)
    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Testset accuracy: {100*accuracy:.3f}%, average loss: {test_loss}")
    for metric in metrics:
        metric_results.append(metric.compute().cpu().numpy())
    return metric_results


def plot_confusion_matrix(confusion_matrix, category_map, plot_filename):
    # Plot last confusion matrix
    fig, axes = plt.subplots(dpi=500)
    mat = axes.matshow(confusion_matrix, cmap=plt.cm.Blues)
    n_classes = confusion_matrix.shape[1]
    if n_classes > 3:
        rotation = 90
    else:
        rotation = 0
    axes.set_title("Confusion matrix")
    axes.set_xticks(range(n_classes), labels=category_map.values(), rotation=rotation)
    axes.set_yticks(range(n_classes), labels=category_map.values())
    axes.set_ylabel("Actual")
    axes.set_xlabel("Predicted")
    axes.xaxis.set_ticks_position("bottom")
    for i in range(n_classes):
        for j in range(n_classes):
            axes.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center")
    plt.colorbar(mat)
    fig.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


def plot_shap_values(model, dataset, category_map, num_classes, feature_names, plot_filename):
    # Prepare and plot last SHAP values
    model = model.to("cpu")
    model.device = "cpu"
    shap_batch_loader = DataLoader(dataset, 110, shuffle=True)
    features, _ = next(iter(shap_batch_loader))
    features = features.cpu()
    model.eval()
    background = features[:100]
    test_values = features[100:]
    with torch.no_grad():
        base_values = model(background).mean(dim=0).numpy()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_values)
    print("base values", base_values)
    print("features", features.shape[1])
    print("test values shape", test_values.shape)
    charts_per_row = 2
    rows = num_classes // charts_per_row + ((num_classes % charts_per_row) != 0)
    fig, axes = plt.subplots(rows, charts_per_row, dpi=500, figsize=(charts_per_row * 4, rows * 3))
    axes = axes.ravel()
    shap_values_per_class = []
    for i in range(num_classes):
        shap_values_per_class.append(np.mean(np.abs(shap_values[i]), axis=1))
        # Create Explanation object for class 0 (you can loop for others)
        shap_explanation = shap.Explanation(
            values=shap_values[i].T,  # SHAP values for class i
            base_values=base_values[i],  # base value for class i
            data=features[100:].numpy(),  # input data
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

    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=0.8)
    plt.savefig(plot_filename)
    plt.close(fig)
    return shap_values_per_class


def prepare_test_metrics(num_classes: int, device: torch.device = torch.device("cuda"), binary_class=False, confusion_matrix=True) -> list[Metric]:
    if binary_class:
        accuracy_metric = BinaryAccuracy(device=device)
        f1_metric = BinaryF1Score(device=device)
        confusion_matrix_metric = BinaryConfusionMatrix(device=device)
        precision_metric = BinaryPrecision(device=device)
        recall_metric = BinaryRecall(device=device)
    else:
        accuracy_metric = MulticlassAccuracy(average=None, num_classes=num_classes, device=device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, device=device, average=None)
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes, device=device)
        precision_metric = MulticlassPrecision(num_classes=num_classes, average=None, device=device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average=None, device=device)
    if confusion_matrix:
        metrics = [accuracy_metric, precision_metric, recall_metric, f1_metric, confusion_matrix_metric]
    else:
        metrics = [accuracy_metric, precision_metric, recall_metric, f1_metric]
    return metrics


def get_patience(epochs: int) -> int:
    patience = None
    if epochs > 50:
        patience = 10
    elif epochs > 15:
        patience = 5
    elif epochs > 5:
        patience = 3
    else:
        patience = 1
    return patience


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
    return device


def get_all_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            print("CUDA available! GPU device name is:", torch.cuda.get_device_name(i))
            yield i
    else:
        print("CUDA is not available")
        yield None
