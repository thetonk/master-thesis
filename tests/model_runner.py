import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import torchinfo
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import shap
from shap.plots import bar
import dataset_utils
from models import MyModel, MyLSTMClassifier, train_model, test_model


#SEED = 42

def plot_confusion_matrix(confusion_matrix, plot_filename):
    # Plot last confusion matrix
    fig, axes = plt.subplots(dpi=500)
    mat = axes.matshow(confusion_matrix, cmap=plt.cm.Blues)
    n_classes = confusion_matrix.shape[1]
    axes.set_title("Confusion matrix")
    axes.set_xticks(range(n_classes), labels=category_map.values())
    axes.set_yticks(range(n_classes), labels=category_map.values())
    axes.set_ylabel("Actual")
    axes.set_xlabel("Predicted")
    axes.xaxis.set_ticks_position("bottom")
    for i in range(n_classes):
        for j in range(n_classes):
            axes.text(j, i, confusion_matrix[i, j], ha="center", va="center")
    plt.colorbar(mat)
    fig.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


def plot_shap_values(model, dataset, plot_filename):
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
    explainer = shap.GradientExplainer(final_model, background)
    shap_values = explainer.shap_values(test_values)
    print("base values", base_values)
    print("features", features.shape[1])
    print("test values shape", test_values.shape)
    charts_per_row = 2
    rows = num_classes // charts_per_row + ((num_classes % charts_per_row) != 0)
    fig, axes = plt.subplots(rows, charts_per_row, dpi=500, figsize=(charts_per_row * 4, rows * 3))
    axes = axes.ravel()
    for i in range(num_classes):
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


if __name__ == "__main__":
    #torch.manual_seed(SEED)
    if torch.cuda.is_available():
        print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
        DEVICE = "cuda"
    else:
        print("CUDA is not available")
        DEVICE = "cpu"

    HELPTEXT = f"Usage: {sys.argv[0]} MODEL DATASET_PATH LABEL_COLUMN N_RUNS N_FOLDS N_EPOCHS\nAvailable options for MODEL are: 'lstm', 'transformer'."
    use_transformer = True
    if len(sys.argv) < 7:
        print("Error! You must specify label column name, number of folds and number of epochs. Exiting!", file=sys.stderr)
        print(HELPTEXT)
        sys.exit(1)
    else:
        try:
            model_name = sys.argv[1]
            raytune_results_dir = os.path.join("tests", "results", "raytune")
            if model_name.lower() == "lstm":
                use_transformer = False
                config_file = os.path.join(raytune_results_dir, "test_raytune_lstm", "best_config.json")
            else:
                config_file = os.path.join(raytune_results_dir, "test_raytune_transformer", "best_config.json")
            with open(config_file, "r") as file:
                json_data = json.load(file)
                config = json_data["config"]
            learning_rate = config.pop("lr")
            batch_size = config.pop("batch_size")
            model_hyperparameters = config
            dataset_path = sys.argv[2]
            label_column = sys.argv[3]
            num_runs = int(sys.argv[4])
            folds = int(sys.argv[5])
            epochs = int(sys.argv[6])
            dataset_name = os.path.basename(dataset_path).split(".")[0]
            if folds < 0 or epochs < 1:
                raise ValueError
            results_dir = os.path.join("tests", "results")
            trained_models_dir = os.path.join(results_dir, "trained_models")
            images_dir = os.path.join(results_dir, "images")
            os.makedirs(trained_models_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            model_filename = os.path.join("trained_models", f"best_model_{model_name}_{dataset_name}.pt")
        except ValueError:
            print("Please specify valid number of folds and epochs", file=sys.stderr)
            print(HELPTEXT)
            sys.exit(1)

    csv_dataset = dataset_utils.CSVDataset(dataset_path, label_column, chunk_size=3e+6)
    csv_dataset.load(balance_classes=True, rows_limit=250e+3)
    X, y, category_map, feature_names = csv_dataset.X, csv_dataset.y, csv_dataset.categories, csv_dataset.features
    num_classes = len(category_map)
    class_frequencies = y.bincount(minlength=num_classes)
    class_percentages = class_frequencies.float() / y.shape[0]
    print("Class percentages:", class_percentages)
    dataset = TensorDataset(X, y)
    #batch_size = 256
    num_features = X.shape[1]
    print(f"# of rows: {X.shape[0]}, # of features: {num_features}, # of classes: {num_classes}, datatype: {X.dtype}")
    metric_names = ["Run #","Fold #","Class", "Accuracy", "Precision", "Recall", "F1 Score"]
    df_list = []
    training_metric = MulticlassAccuracy(average='macro', num_classes=num_classes, device=DEVICE)
    for i in range(num_runs):
        print("#"*50,f"RUN {i}", "#"*50)
        if folds == 0:
            print("Training and testing model with a random split of 80% train and 20% test!")
            multilclass_accuracy_metric = MulticlassAccuracy(average=None, num_classes=num_classes, device=DEVICE)
            multiclass_f1_metric = MulticlassF1Score(num_classes=num_classes, device=DEVICE, average=None)
            multiclass_confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes, device=DEVICE)
            multiclass_precision_metric = MulticlassPrecision(num_classes=num_classes, average=None, device=DEVICE)
            multiclass_recall_metric = MulticlassRecall(num_classes=num_classes, average=None, device=DEVICE)
            metrics = [multilclass_accuracy_metric, multiclass_precision_metric, multiclass_recall_metric, multiclass_f1_metric, multiclass_confusion_matrix_metric]
            if use_transformer:
                model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
            else:
                model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
            torchinfo.summary(model, input_size=(batch_size, num_features))
            train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=6)
            test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=6)
            print("STARTING TRAINING SESSION!!!")
            train_model(model, model_filename, train_loader, None, training_metric, epochs=epochs, device=DEVICE, learning_rate=learning_rate)
            print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
            if use_transformer:
                final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
            else:
                final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
            final_model.load_state_dict(torch.load(model_filename, weights_only=False))
            multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_confusion_matrix = test_model(final_model, test_loader, metrics, device=DEVICE)
            metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1]*num_classes, [1]*num_classes, category_map.values(), multiclass_accuracy, 
                                                                        multiclass_precision, multiclass_recall, multiclass_f1_score])))
            del train_loader, test_loader
            df_list.append(metrics_df)
            print("Metrics:\n", metrics_df, sep='')

        else:
            strat_kfold = StratifiedKFold(n_splits=folds, shuffle=True)
            final_model = None
            show_summary = True
            multiclass_confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.uint64)
            for fold, (train_index, test_index) in enumerate(strat_kfold.split(X, y)):
                multilclass_accuracy_metric = MulticlassAccuracy(average=None, num_classes=num_classes, device=DEVICE)
                multiclass_f1_metric = MulticlassF1Score(num_classes=num_classes, device=DEVICE, average=None)
                multiclass_confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes, device=DEVICE)
                multiclass_precision_metric = MulticlassPrecision(num_classes=num_classes, average=None, device=DEVICE)
                multiclass_recall_metric = MulticlassRecall(num_classes=num_classes, average=None, device=DEVICE)
                metrics = [multilclass_accuracy_metric, multiclass_precision_metric, multiclass_recall_metric, multiclass_f1_metric, multiclass_confusion_matrix_metric]
                if use_transformer:
                    model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                if show_summary:
                    torchinfo.summary(model, input_size=(batch_size, num_features))
                    show_summary = False
                print("-"*50)
                print(f"Fold {fold+1}/{folds}")
                train_dataset = Subset(dataset, train_index)
                test_dataset = Subset(dataset, test_index)
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=6)
                test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=6)
                print("STARTING TRAINING SESSION!!!")
                train_model(model, model_filename, train_loader, None, metric=training_metric, epochs=epochs, device=DEVICE, learning_rate=learning_rate)
                print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                if use_transformer:
                    final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                final_model.load_state_dict(torch.load(model_filename, weights_only=False))
                multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, fold_confusion_matrix = test_model(final_model, test_loader, metrics, device=DEVICE)
                multiclass_confusion_matrix += fold_confusion_matrix.astype(np.uint64)
                metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1]*num_classes, [fold+1]*num_classes, category_map.values(), multiclass_accuracy, 
                                                                            multiclass_precision, multiclass_recall, multiclass_f1_score])))
                del train_loader, test_loader
                df_list.append(metrics_df)
                print("Metrics:\n", metrics_df, sep='')
                print("-"*50)
        print(f"[Run {i+1}] Multiclass confusion matrix:\n", multiclass_confusion_matrix, sep='')
        plot_confusion_matrix(multiclass_confusion_matrix, os.path.join(images_dir, f"confusion_matrix_{model_name}_{dataset_name}_{i+1}.png"))
        plot_shap_values(final_model, dataset, os.path.join(images_dir, f"shap_values_{model_name}_{dataset_name}_{i+1}.png"))
    results_df = pd.concat(df_list)
    results_df.to_csv(os.path.join(results_dir, f"results_{model_name}_{dataset_name}.csv"))
