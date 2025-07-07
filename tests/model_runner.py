import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torcheval.metrics
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import torchinfo
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
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
            axes.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center")
    plt.colorbar(mat)
    fig.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


def plot_shap_values(model, dataset, num_classes, feature_names, plot_filename):
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


def prepare_test_metrics(num_classes: int) -> list[torcheval.metrics.Metric]:
    multilclass_accuracy_metric = MulticlassAccuracy(average=None, num_classes=num_classes, device=DEVICE)
    multiclass_f1_metric = MulticlassF1Score(num_classes=num_classes, device=DEVICE, average=None)
    multiclass_confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes, device=DEVICE)
    multiclass_precision_metric = MulticlassPrecision(num_classes=num_classes, average=None, device=DEVICE)
    multiclass_recall_metric = MulticlassRecall(num_classes=num_classes, average=None, device=DEVICE)
    metrics = [multilclass_accuracy_metric, multiclass_precision_metric, multiclass_recall_metric, multiclass_f1_metric,
               multiclass_confusion_matrix_metric]
    return metrics


if __name__ == "__main__":
    #torch.manual_seed(SEED)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("model", choices=("lstm", "transformer"), help="The model type")
    parser.add_argument("label_column", type=str, help="The name of the column that will be used as class.", default="Label")
    parser.add_argument("runs", type=int, help="Number of runs. Must not be 0", default=1)
    parser.add_argument("folds", type=int, help="Number of folds. Must not be negative. Ignored if used with zero-shot.",default=0)
    parser.add_argument("epochs", type=int, help="Number of traininig epochs. Must be larger than 0", default=10)
    parser.add_argument("-z", "--zero-shot", action="store_true", help="Run zero-shot transfer learning")
    dataset_args = parser.add_mutually_exclusive_group(required=True)
    dataset_args.add_argument("-f", "--file", type=str, help="Dataset CSV file", dest="dataset_file")
    dataset_args.add_argument("-d", "--directory", type=str, help="Dataset directory containing CSV files", dest="dataset_folder")
    args = parser.parse_args()
    use_transformer = True
    load_directory = False
    zero_shot = False
    N_WORKERS = 6

    if torch.cuda.is_available():
        print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
        DEVICE = torch.device("cuda")
    else:
        print("CUDA is not available")
        DEVICE = torch.device("cpu")

    try:
        model_name = args.model
        raytune_results_dir = os.path.join("tests", "results", "raytune")
        if model_name == "lstm":
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
        if args.zero_shot and args.dataset_folder is None:
            raise ValueError("Zero shot transfer learning requires a directory that contains datasets!")
        if args.dataset_folder is None:
            dataset_file = args.dataset_file
        else:
            dataset_folder = args.dataset_folder
            load_directory = True
        label_column = args.label_column
        num_runs = args.runs
        folds = args.folds
        epochs = args.epochs
        zero_shot = args.zero_shot
        if folds < 0 or epochs < 1 or num_runs < 1:
            raise ValueError("Please specify valid number of folds and epochs")
    except ValueError as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    results_dir = os.path.join("tests", "results")
    trained_models_dir = os.path.join(results_dir, "trained_models")
    images_dir = os.path.join(results_dir, "images")
    os.makedirs(trained_models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    if not load_directory:
        dataset_name = os.path.basename(dataset_file).split(".")[0]
        model_filename = os.path.join("trained_models", f"best_model_{model_name}_{dataset_name}.pt")
        csv_dataset = dataset_utils.CSVDataset(dataset_file, label_column, chunk_size=3e+6)
        csv_dataset.load(balance_classes=True, rows_limit=250e+3)
        X, y, category_map, feature_names = csv_dataset.X, csv_dataset.y, csv_dataset.categories, csv_dataset.features
        num_classes = len(category_map)
        class_frequencies = y.bincount(minlength=num_classes)
        class_percentages = class_frequencies.float() / y.shape[0]
        print("Class percentages:", class_percentages)
        dataset = TensorDataset(X, y)
        num_features = X.shape[1]
        num_rows = X.shape[0]
        datatype = X.dtype
    else:
        model_filename = os.path.join("trained_models", f"best_model_{model_name}_TL.pt")
        rows_per_dataset = 77140
        if zero_shot:
            print("Running in zero-shot mode!")
            dataset_list = dataset_utils.load_datasets_from_dir(dataset_folder, label_column, rows_per_dataset=rows_per_dataset,
                                                                balance_classes=True, as_tensors_list=True)
            num_rows = sum([dataset.num_rows for dataset in dataset_list])
            num_classes = dataset_list[0].num_classes
            num_features = dataset_list[0].num_features
            category_map = dataset_list[0].categories
            feature_names = dataset_list[0].feature_names
            datatype = dataset_list[0].dtype
        else:
            loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_folder, label_column, rows_per_dataset=rows_per_dataset,
                                                                  balance_classes=True)
            dataset = loaded_dataset.dataset
            num_features = loaded_dataset.num_features
            num_rows = loaded_dataset.num_rows
            num_classes = loaded_dataset.num_classes
            datatype = loaded_dataset.dtype
            feature_names = loaded_dataset.feature_names
            category_map = loaded_dataset.categories
            X = dataset.tensors[0]
            y = dataset.tensors[1]
        dataset_name = "TL"

    print(f"# of rows: {num_rows}, # of features: {num_features}, # of classes: {num_classes}, datatype: {datatype}")
    metric_names = ["Run #","Fold #","Class", "Accuracy", "Precision", "Recall", "F1 Score"]
    df_list = []
    training_metric = MulticlassAccuracy(average='macro', num_classes=num_classes, device=DEVICE)
    for i in range(num_runs):
        print("#"*50,f"RUN {i}", "#"*50)
        if zero_shot:
            dataset_loo = LeaveOneOut()
            show_summary = True
            for fold, (_, testset_idx) in enumerate(dataset_loo.split(dataset_list)):
                print(f"Using dataset {testset_idx.item()+1} as test!")
                temp_dataset_list = dataset_list.copy()
                test_dataset = temp_dataset_list.pop(testset_idx.item()).dataset
                train_dataset_list = temp_dataset_list
                X = train_dataset_list[0].dataset.tensors[0]
                y = train_dataset_list[0].dataset.tensors[1]
                for j in range(1, len(train_dataset_list)):
                    X = torch.cat((X, train_dataset_list[i].dataset.tensors[0]), dim=0)
                    y = torch.cat((y, train_dataset_list[i].dataset.tensors[1]), dim=0)
                del train_dataset_list
                train_dataset = TensorDataset(X, y)
                del X, y
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=N_WORKERS)
                test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                metrics = prepare_test_metrics(num_classes)
                if use_transformer:
                    model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                if show_summary:
                    torchinfo.summary(model, input_size=(batch_size, num_features))
                    show_summary = False
                print("STARTING TRAINING SESSION!!!")
                train_model(model, model_filename, train_loader, None, training_metric, epochs=epochs, device=DEVICE,
                            learning_rate=learning_rate)
                print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                if use_transformer:
                    final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                final_model.load_state_dict(torch.load(model_filename, weights_only=False))
                multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_confusion_matrix = test_model(
                    final_model, test_loader, metrics, device=DEVICE)
                metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i + 1] * num_classes, [fold+1] * num_classes,
                                                                            category_map.values(), multiclass_accuracy,
                                                                            multiclass_precision, multiclass_recall,
                                                                            multiclass_f1_score])))
                del train_loader, test_loader
                df_list.append(metrics_df)
                print("Metrics:\n", metrics_df, sep='')
                plot_confusion_matrix(multiclass_confusion_matrix, os.path.join(images_dir, f"confusion_matrix_{model_name}_{dataset_name}_{i+1}_{fold+1}_TL.png"))
                plot_shap_values(final_model, test_dataset, num_classes, feature_names, os.path.join(images_dir, f"shap_values_{model_name}_{dataset_name}_{i+1}_{fold+1}_TL.png"))
        else:
            if folds == 0:
                print("Training and testing model with a random split of 80% train and 20% test!")
                metrics = prepare_test_metrics(num_classes)
                if use_transformer:
                    model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                torchinfo.summary(model, input_size=(batch_size, num_features))
                train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=N_WORKERS)
                test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                print("STARTING TRAINING SESSION!!!")
                train_model(model, model_filename, train_loader, None, training_metric, epochs=epochs,
                            device=DEVICE, learning_rate=learning_rate)
                print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                if use_transformer:
                    final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                final_model.load_state_dict(torch.load(model_filename, weights_only=False))
                multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_confusion_matrix = test_model(final_model, test_loader, metrics, device=DEVICE)
                metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1]*num_classes, [1]*num_classes, category_map.values(),
                                                                            multiclass_accuracy,multiclass_precision,
                                                                            multiclass_recall, multiclass_f1_score])))
                del train_loader, test_loader
                df_list.append(metrics_df)
                print("Metrics:\n", metrics_df, sep='')

            else:
                strat_kfold = StratifiedKFold(n_splits=folds, shuffle=True)
                final_model = None
                show_summary = True
                multiclass_confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.uint64)
                for fold, (train_index, test_index) in enumerate(strat_kfold.split(X, y)):
                    metrics = prepare_test_metrics(num_classes)
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
                    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=N_WORKERS)
                    test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                    print("STARTING TRAINING SESSION!!!")
                    train_model(model, model_filename, train_loader, None, metric=training_metric, epochs=epochs,
                                device=DEVICE, learning_rate=learning_rate)
                    print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                    if use_transformer:
                        final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                    else:
                        final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                    final_model.load_state_dict(torch.load(model_filename, weights_only=False))
                    multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, fold_confusion_matrix = test_model(final_model, test_loader, metrics, device=DEVICE)
                    multiclass_confusion_matrix += fold_confusion_matrix.astype(np.uint64)
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1]*num_classes, [fold+1]*num_classes, category_map.values(),
                                                                                multiclass_accuracy, multiclass_precision,
                                                                                multiclass_recall, multiclass_f1_score])))
                    del train_loader, test_loader
                    df_list.append(metrics_df)
                    print("Metrics:\n", metrics_df, sep='')
                    print("-"*50)
            print(f"[Run {i+1}] Multiclass confusion matrix:\n", multiclass_confusion_matrix, sep='')
            plot_confusion_matrix(multiclass_confusion_matrix, os.path.join(images_dir, f"confusion_matrix_{model_name}_{dataset_name}_{i+1}.png"))
            plot_shap_values(final_model, dataset, num_classes, feature_names, os.path.join(images_dir, f"shap_values_{model_name}_{dataset_name}_{i+1}.png"))
    results_df = pd.concat(df_list)
    results_df.to_csv(os.path.join(results_dir, f"results_{model_name}_{dataset_name}.csv"))
