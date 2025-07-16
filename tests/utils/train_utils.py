import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import Metric
import numpy as np
from ray import tune

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
        loss = 0
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
        if train_tune and val_loader is not None:
            metric.reset()
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                metric.update(output, label)
            val_accuracy = metric.compute().item()
            tune.report({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        elif train_tune and val_loader is None:
            print("Enabled tuning, but no validation loader is set!")
            tune.report({"train_accuracy": train_accuracy})
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
