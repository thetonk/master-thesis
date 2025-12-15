# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import sys
import os
import json
import tempfile
import argparse
import signal
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset, random_split
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from torcheval.metrics import BinaryAccuracy
from sklearn.model_selection import StratifiedKFold
from utils import dataset_utils
from utils import train_test_utils as ttutils
from utils.models import MyModel, MyLSTMClassifier
from utils.exceptions import handle_slurm_exception

N_WORKERS = 8

def train_test_model(pipe, model, model_config, train_dataset,
                     test_dataset, val_dataset, test_metrics, device=None, device_id=None) -> None | list[np.ndarray]:
    batch_size = model_config["batch_size"]
    learning_rate = model_config["learning_rate"]
    epochs = model_config["epochs"]
    if device is None:
        device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    train_metric = BinaryAccuracy(device=device)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, persistent_workers=True, num_workers=N_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, persistent_workers=True, num_workers=N_WORKERS, pin_memory=True)
    val_loader = None
    early_stopper = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size, persistent_workers=True, num_workers=N_WORKERS, pin_memory=True)
        early_stopper = ttutils.EarlyStopping(ttutils.get_patience(epochs), 2.5e-3)
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmpfile:
        ttutils.train_model(model, tmpfile.name, train_loader, val_loader, early_stopper=early_stopper,
                        metric=train_metric, epochs=epochs, learning_rate=learning_rate, device=device)
        model.load_state_dict(torch.load(tmpfile.name, weights_only=True, map_location=device))
    test_results = ttutils.test_model(model, test_loader, test_metrics, device=device)
    del train_loader, test_loader
    if pipe is None:
        return test_results
    else:
        pipe.send(test_results)
        pipe.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("label_column", type=str, help="The name of the column that will be used as class.", default="Label")
    parser.add_argument("runs", type=int, help="Number of runs. Must not be positive", default=1)
    parser.add_argument("folds", type=int, help="Number of folds. Must be at least 2", default=2)
    parser.add_argument("-te", "--transformer-epochs", type=int, help="Number of transformer training epochs. Must be positive", default=10, required=True)
    parser.add_argument("-le", "--lstm-epochs", type=int, help="Number of LSTM training epochs. Must be positive", default=5, required=True)
    parser.add_argument("-tc", "--transformer-config", type=str, help="Path to transformer hyperparameter configuration file", required=True)
    parser.add_argument("-lc", "--lstm-config", type=str, help="Path to LSTM hyperparameter configuration file", required=True)
    parser.add_argument("-r", action="store_true", help="Remove network specific features", dest="remove_features")
    parser.add_argument("-d", "--dataset-directory", type=str, help="Directory to look for datasets", dest="dataset_directory", required=True)
    parser.add_argument("-p", "--parallel", action='store_true', help="Train and test the models in parallel, utilizing multiple GPUs if possible")
    parser.add_argument("-e", "--early-stop", action='store_true', help="Add early stopping when training models")
    parser.add_argument("-dp", "--dataset-percentage", type=int, help="Percentage of dataset to use (0-100), default is 100", default=100)
    args = parser.parse_args()

    if "SLURM_JOB_ID" in os.environ:
        print("Running in SLURM environment!")
        signal.signal(signal.SIGTERM, handle_slurm_exception)

    try:
        use_parallel = args.parallel
        use_early_stop = args.early_stop
        label_column = args.label_column
        num_runs = args.runs
        num_folds = args.folds
        epochs_transformer = args.transformer_epochs
        epochs_lstm = args.lstm_epochs
        dataset_directory = args.dataset_directory
        dataset_percentage = args.dataset_percentage
        remove_features = args.remove_featues
        if num_folds < 2:
            raise ValueError("Number of folds must be at least 2.")
        if num_runs < 1 or epochs_transformer < 1 or epochs_lstm < 1:
            raise ValueError("Number of runs must be at least 1.")

        LSTM_DEVICE_ID = TRANSFORMER_DEVICE_ID = 0
        LSTM_DEVICE = TRANSFORMER_DEVICE = ttutils.get_device()

        if use_parallel:
            print("Running tests in parallel mode!")
            mp.set_start_method('spawn', force=True)
            if torch.cuda.device_count() > 1:
                print("Multiple GPUs detected! Running tests in separate GPUs!")
                device_id_gen = ttutils.get_all_cuda_devices()
                LSTM_DEVICE_ID = next(device_id_gen)
                TRANSFORMER_DEVICE_ID = next(device_id_gen)

        if remove_features:
            transformer_dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Idle Mean", "Idle Min", "Idle Max"]
            lstm_dropped_columns = ["Timestamp", "Fwd Seg Size Min"]
            print("For Transformer model the following features are being dropped:")
            print(*transformer_dropped_columns, sep='')
            print("For LSTM the following features are being dropped:")
            print(*lstm_dropped_columns, sep='')
        else:
            dropped_columns = ["Timestamp"]
            print("The following features will be ignored:")
            print(*dropped_columns, sep=',')
        model_data = {"transformer": {},
                      "lstm": {}}
        with open(args.transformer_config, "r") as f:
            json_data = json.load(f)
            config = json_data["config"]
        model_data["transformer"]["learning_rate"] = config.pop("lr")
        model_data["transformer"]["batch_size"] = config.pop("batch_size")
        model_data["transformer"]["hyperparameters"] = config
        model_data["transformer"]["epochs"] = epochs_transformer
        with open(args.lstm_config, "r") as f:
            json_data = json.load(f)
            config = json_data["config"]
        model_data["lstm"]["learning_rate"] = config.pop("lr")
        model_data["lstm"]["batch_size"] = config.pop("batch_size")
        model_data["lstm"]["hyperparameters"] = config
        model_data["lstm"]["epochs"] = epochs_lstm
        #del config 
    except ValueError as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    results_dir = "results"
    trained_models_dir = os.path.join(results_dir, "trained_models")
    images_dir = os.path.join(results_dir, "images")
    os.makedirs(trained_models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    dataset_name = "TL"
    rows_per_dataset = int(77140 * dataset_percentage / 100)
    if remove_features:
        dataset_name += "_removed_merged_ds"
        transformer_loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_directory, label_column, transformer_dropped_columns, rows_per_dataset, balance_classes=True)
        lstm_loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_directory, label_column, lstm_dropped_columns, rows_per_dataset, balance_classes=True)
        transformer_dataset = transformer_loaded_dataset.dataset
        lstm_dataset = lstm_loaded_dataset.dataset
        transformer_num_features = transformer_loaded_dataset.num_features
        num_rows = transformer_loaded_dataset.num_rows
        num_classes = transformer_loaded_dataset.num_classes
        datatype = transformer_loaded_dataset.dtype
        del transformer_loaded_dataset, lstm_loaded_dataset
        X = transformer_dataset.tensors[0]
        y = transformer_dataset.tensors[1]
    else:
        loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_directory, label_column, dropped_columns, rows_per_dataset, balance_classes=True)
        dataset = loaded_dataset.dataset
        transformer_num_features = loaded_dataset.num_features
        num_rows = loaded_dataset.num_rows
        num_classes = loaded_dataset.num_classes
        datatype = loaded_dataset.dtype
        del loaded_dataset
        X = dataset.tensors[0]
        y = dataset.tensors[1]
    class_frequencies = y.bincount(minlength=num_classes)
    class_percentages = class_frequencies.float() / y.shape[0]
    print("Class percentages:", class_percentages)
    print(f"# of rows: {num_rows}, # of classes: {num_classes}, datatype: {datatype}")
    metric_names = ["Run #","Fold #", "Accuracy", "Precision", "Recall", "F1 Score"]
    p_values_names = ["Metric", "T-test p-value", "Wilcoxon p-value"]
    transformer_metrics_df_list = []
    lstm_metrics_df_list = []
    try:
        for i in range(num_runs):
            print("#"*50,f"RUN {i+1}", "#"*50)
            strat_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
            for fold, (train_index, test_index) in enumerate(strat_kfold.split(X, y)):
                print("-"*50)
                print(f"Fold {fold+1}/{num_folds}")
                if remove_features:
                    transformer_val_dataset = None
                    lstm_val_dataset = None
                    transformer_train_dataset = Subset(transformer_dataset, train_index)
                    lstm_train_dataset = Subset(lstm_dataset, train_index)
                    transformer_test_dataset = Subset(transformer_dataset, test_index)
                    lstm_test_dataset = Subset(lstm_dataset, test_index)
                    if use_early_stop:
                        transformer_train_dataset, transformer_val_dataset = random_split(transformer_train_dataset, [0.8, 0.2])
                        lstm_train_dataset, lstm_val_dataset = random_split(lstm_train_dataset, [0.8, 0.2])
                else:
                    train_dataset = Subset(dataset, train_index)
                    test_dataset = Subset(dataset, test_index)
                    val_dataset = None
                    if use_early_stop:
                        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
                    transformer_train_dataset = train_dataset
                    transformer_val_dataset = val_dataset
                    transformer_test_dataset = test_dataset
                    lstm_train_dataset = train_dataset
                    lstm_val_dataset = val_dataset
                    lstm_test_dataset = test_dataset
                    del train_dataset, test_dataset, val_dataset
                if not use_parallel:
                    for model_name, model_config in model_data.items():
                        if model_name == "lstm":
                            model = MyLSTMClassifier(num_classes, **model_config["hyperparameters"])
                            device = LSTM_DEVICE
                            metrics = ttutils.prepare_test_metrics(num_classes, binary_class=True, confusion_matrix=False, device=device)
                            accuracy, precision, recall, f1_score = train_test_model(None, model, model_config, lstm_train_dataset, 
                                                                                     lstm_test_dataset, lstm_val_dataset, metrics, device)
                            metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                            lstm_metrics_df_list.append(metrics_df)
                        else:
                            model = MyModel(transformer_num_features, num_classes, **model_config["hyperparameters"])
                            device = TRANSFORMER_DEVICE
                            metrics = ttutils.prepare_test_metrics(num_classes, binary_class=True, confusion_matrix=False, device=device)
                            accuracy, precision, recall, f1_score = train_test_model(None, model, model_config, transformer_train_dataset, 
                                                                                     transformer_test_dataset, transformer_val_dataset, metrics, device)
                            metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                            transformer_metrics_df_list.append(metrics_df)                           
                        del model
                else:
                    transformer_config = model_data["transformer"]
                    lstm_config = model_data["lstm"]
                    transformer_metrics = ttutils.prepare_test_metrics(num_classes, binary_class=True, confusion_matrix=False,
                                                                       device=torch.device(f"cuda:{TRANSFORMER_DEVICE_ID}"))
                    lstm_metrics = ttutils.prepare_test_metrics(num_classes, binary_class=True, confusion_matrix=False,
                                                                device=torch.device(f"cuda:{LSTM_DEVICE_ID}"))
                    transformer_par_conn, transformer_child_conn = mp.Pipe()
                    lstm_par_conn, lstm_child_conn = mp.Pipe()
                    transformer_process = mp.Process(target=train_test_model, daemon=False, args=(transformer_child_conn,
                                                                                    MyModel(transformer_num_features, num_classes, **transformer_config["hyperparameters"]),
                                                                                    transformer_config, transformer_train_dataset, 
                                                                                    transformer_test_dataset, transformer_val_dataset, transformer_metrics,
                                                                                    None, TRANSFORMER_DEVICE_ID))
                    lstm_process = mp.Process(target=train_test_model, daemon=False, args=(lstm_child_conn,
                                                                                MyLSTMClassifier(num_classes, **lstm_config["hyperparameters"]),
                                                                                lstm_config, lstm_train_dataset, 
                                                                                lstm_test_dataset, lstm_val_dataset, lstm_metrics,
                                                                                None, LSTM_DEVICE_ID))
                    transformer_process.start()
                    lstm_process.start()
                    accuracy, precision, recall, f1_score = transformer_par_conn.recv()
                    transformer_process.join()
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                    transformer_metrics_df_list.append(metrics_df)
                    accuracy, precision, recall, f1_score = lstm_par_conn.recv()
                    lstm_process.join()
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                    lstm_metrics_df_list.append(metrics_df)
    except Exception as e:
        print("Exception occurred!", e, file=sys.stderr)
    finally:
        if len(transformer_metrics_df_list) > 0:
            print("Saving statistics...")
            transformer_results_df = pd.concat(transformer_metrics_df_list)
            print("Transformer results", transformer_results_df, sep='\n')
            lstm_results_df = pd.concat(lstm_metrics_df_list)
            print("LSTM results", lstm_results_df, sep='\n')
            del transformer_metrics_df_list, lstm_metrics_df_list
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            ttest_pvalues_list = []
            wilcoxon_pvalues_list = []
            for metric_name in metric_names:
                ttest_result = ttest_rel(transformer_results_df[metric_name], lstm_results_df[metric_name])
                wilcoxon_result = wilcoxon(transformer_results_df[metric_name], lstm_results_df[metric_name])
                ttest_pvalue = ttest_result.pvalue
                wilcoxon_pvalue = wilcoxon_result.pvalue
                ttest_pvalues_list.append(ttest_pvalue)
                wilcoxon_pvalues_list.append(wilcoxon_pvalue)
            pvalues_df = pd.DataFrame.from_dict(dict(zip(p_values_names, [metric_names, ttest_pvalues_list, wilcoxon_pvalues_list])))
            print("P-values", pvalues_df, sep='\n')
            transformer_results_filename = "results_transformer_comparator"
            lstm_results_filename = "results_lstm_comparator"
            pvalues_filename = f"pvalues_{dataset_name}"
            if args.remove_features:
                transformer_results_filename += "_removed"
                lstm_results_filename += "_removed"
                pvalues_filename += "_removed"
            if dataset_percentage < 100:
                transformer_results_filename += f"_{dataset_percentage}"
                lstm_results_filename += f"_{dataset_percentage}"
                pvalues_filename += f"{dataset_percentage}"
            transformer_results_df.to_csv(os.path.join(results_dir, f"{transformer_results_filename}.csv"))
            lstm_results_df.to_csv(os.path.join(results_dir, f"{lstm_results_filename}.csv"))
            pvalues_df.to_csv(os.path.join(results_dir, f"{pvalues_filename}.csv"))
