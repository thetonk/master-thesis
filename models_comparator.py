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
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from sklearn.model_selection import StratifiedKFold
from utils import dataset_utils
from utils import train_test_utils as ttutils
from utils.dataset_utils import get_dropped_columns
from utils.models import ModelTypes, get_model
from utils.exceptions import handle_slurm_exception

N_WORKERS = 8

def train_test_model(pipe, model, model_config, train_dataset,
                     test_dataset, val_dataset, test_metrics, device,
                     train_metric) -> None | list[np.ndarray]:
    batch_size = model_config["batch_size"]
    learning_rate = model_config["learning_rate"]
    epochs = model_config["epochs"]
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=N_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=N_WORKERS, pin_memory=True)
    val_loader = None
    early_stopper = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size, num_workers=N_WORKERS, pin_memory=True)
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
    parser.add_argument("first_model_type", choices=[model.value for model in ModelTypes], help="Type of first model")
    parser.add_argument("second_model_type", choices=[model.value for model in ModelTypes], help="Type of second model")
    parser.add_argument("label_column", type=str, help="The name of the column that will be used as class.", default="Label")
    parser.add_argument("runs", type=int, help="Number of runs. Must not be positive", default=1)
    parser.add_argument("folds", type=int, help="Number of folds. Must be at least 2", default=2)
    parser.add_argument("-fe", "--first-model-epochs", type=int, help="Number of first model training epochs. Must be positive", default=30, required=True)
    parser.add_argument("-se", "--second-model-epochs", type=int, help="Number of second model training epochs. Must be positive", default=10, required=True)
    parser.add_argument("-fc", "--first-model-config", type=str, help="Path to first model hyperparameter configuration file", required=True)
    parser.add_argument("-sc", "--second-model-config", type=str, help="Path to second hyperparameter configuration file", required=True)
    parser.add_argument("-r", action="store_true", help="Remove network specific features", dest="remove_features")
    parser.add_argument("-d", "--dataset-directory", type=str, help="Directory to look for datasets", dest="dataset_directory", required=True)
    parser.add_argument("-p", "--parallel", action='store_true', help="Train and test the models in parallel, utilizing multiple GPUs if possible")
    parser.add_argument("-e", "--early-stop", action='store_true', help="Add early stopping when training models")
    parser.add_argument("-dp", "--dataset-percentage", type=int, help="Percentage of dataset to use (0-100), default is 100", default=100)
    parser.add_argument("-b", "--binary-classification", action='store_true', help="Use binary classification")
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
        epochs_transformer = args.first_model_epochs
        epochs_lstm = args.second_model_epochs
        dataset_directory = args.dataset_directory
        dataset_percentage = args.dataset_percentage
        remove_features = args.remove_features
        use_binary_metrics = args.binary_classification
        model_1_type = args.first_model_type
        model_2_type = args.second_model_type
        if model_1_type == model_2_type:
            raise ValueError("First model must be different from the second!")
        if num_folds < 2:
            raise ValueError("Number of folds must be at least 2.")
        if num_runs < 1 or epochs_transformer < 1 or epochs_lstm < 1:
            raise ValueError("Number of runs must be at least 1.")

        model_1_device_id = model_2_device_id = 0
        MODEL_1_DEVICE = MODEL_2_DEVICE = ttutils.get_device()

        if use_parallel:
            print("Running tests in parallel mode!")
            mp.set_start_method('spawn', force=True)
            if torch.cuda.device_count() > 1:
                print("Multiple GPUs detected! Running tests in separate GPUs!")
                device_id_gen = ttutils.get_all_cuda_devices()
                model_1_device_id = next(device_id_gen)
                model_2_device_id = next(device_id_gen)
            MODEL_1_DEVICE = torch.device(f"cuda:{model_1_device_id}")
            MODEL_2_DEVICE = torch.device(f"cuda:{model_2_device_id}")

        model_1_dropped_columns = model_2_dropped_columns = dropped_columns = []
        if remove_features:
            if model_1_type == ModelTypes.TRANSFORMER:
                model_1_dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Idle Mean", "Idle Min", "Idle Max"]
            elif model_1_type == ModelTypes.LSTM:
                lstm_dropped_columns = ["Timestamp", "Fwd Seg Size Min"]
            else:
                raise NotImplementedError("Feature removal for CNN is currently not supported!")
            model_1_dropped_columns = get_dropped_columns(model_1_type)
            model_2_dropped_columns = get_dropped_columns(model_2_type)
            print(f"For the {model_1_type} model the following features are being dropped:")
            print(*model_1_dropped_columns, sep=',')
            print(f"For {model_2_type} model the following features are being dropped:")
            print(*model_2_dropped_columns, sep=',')
        else:
            dropped_columns = ["Timestamp"]
            print("The following features will be ignored:")
            print(*dropped_columns, sep=',')
        model_data = {model_1_type: {},
                      model_2_type: {}}
        with open(args.first_model_config, "r") as f:
            json_data = json.load(f)
            config = json_data["config"]
        model_data[model_1_type]["learning_rate"] = config.pop("lr")
        model_data[model_1_type]["batch_size"] = config.pop("batch_size")
        model_data[model_1_type]["hyperparameters"] = config
        model_data[model_1_type]["epochs"] = epochs_transformer
        with open(args.second_model_config, "r") as f:
            json_data = json.load(f)
            config = json_data["config"]
        model_data[model_2_type]["learning_rate"] = config.pop("lr")
        model_data[model_2_type]["batch_size"] = config.pop("batch_size")
        model_data[model_2_type]["hyperparameters"] = config
        model_data[model_2_type]["epochs"] = epochs_lstm
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
    rows_per_dataset = int((77140 if use_binary_metrics else 150903) * dataset_percentage / 100)
    if remove_features:
        dataset_name += "_removed_merged_ds"
        model_1_loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_directory, label_column, model_1_dropped_columns, rows_per_dataset, balance_classes=True)
        model_1_dataset = model_1_loaded_dataset.dataset
        model_1_num_features = model_1_loaded_dataset.num_features
        num_rows = model_1_loaded_dataset.num_rows
        num_classes = model_1_loaded_dataset.num_classes
        datatype = model_1_loaded_dataset.dtype
        del model_1_loaded_dataset
        model_2_loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_directory, label_column, model_2_dropped_columns, rows_per_dataset, balance_classes=True)
        model_2_dataset = model_2_loaded_dataset.dataset
        model_2_num_features = model_2_loaded_dataset.num_features
        del model_2_loaded_dataset
        X = model_1_dataset.tensors[0]
        y = model_1_dataset.tensors[1]
    else:
        loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_directory, label_column, dropped_columns, rows_per_dataset, balance_classes=True)
        dataset = loaded_dataset.dataset
        model_1_num_features = model_2_num_features = loaded_dataset.num_features
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
    model_1_metrics_df_list = []
    model_2_metrics_df_list = []
    if use_binary_metrics:
        model_1_train_metric = BinaryAccuracy(device=MODEL_1_DEVICE)
        model_2_train_metric = BinaryAccuracy(device=MODEL_2_DEVICE)
    else:
        model_1_train_metric = MulticlassAccuracy(average="macro", num_classes=num_classes, device=MODEL_1_DEVICE)
        model_2_train_metric = MulticlassAccuracy(average="macro", num_classes=num_classes, device=MODEL_2_DEVICE)
    try:
        for i in range(num_runs):
            print("#"*50,f"RUN {i+1}", "#"*50)
            strat_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
            for fold, (train_index, test_index) in enumerate(strat_kfold.split(X, y)):
                print("-"*50)
                print(f"Fold {fold+1}/{num_folds}")
                if remove_features:
                    model_1_val_dataset = None
                    model_2_val_dataset = None
                    model_1_train_dataset = Subset(model_1_dataset, train_index)
                    model_2_train_dataset = Subset(model_2_dataset, train_index)
                    model_1_test_dataset = Subset(model_1_dataset, test_index)
                    model_2_test_dataset = Subset(model_2_dataset, test_index)
                    if use_early_stop:
                        model_1_train_dataset, model_1_val_dataset = random_split(model_1_train_dataset, [0.8, 0.2])
                        model_2_train_dataset, model_2_val_dataset = random_split(model_2_train_dataset, [0.8, 0.2])
                else:
                    train_dataset = Subset(dataset, train_index)
                    test_dataset = Subset(dataset, test_index)
                    val_dataset = None
                    if use_early_stop:
                        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
                    model_1_train_dataset = train_dataset
                    model_1_val_dataset = val_dataset
                    model_1_test_dataset = test_dataset
                    model_2_train_dataset = train_dataset
                    model_2_val_dataset = val_dataset
                    model_2_test_dataset = test_dataset
                    del train_dataset, test_dataset, val_dataset
                if not use_parallel:
                    for i, (model_name, model_config) in enumerate(model_data.items()):
                        if i == 0:
                            num_features = model_1_num_features
                            train_dataset = model_1_train_dataset
                            test_dataset = model_1_test_dataset
                            val_dataset = model_1_val_dataset
                            train_metric = model_1_train_metric
                            device = MODEL_1_DEVICE
                            df_list = model_1_metrics_df_list
                        else:
                            num_features = model_2_num_features
                            train_dataset = model_2_train_dataset
                            test_dataset = model_2_test_dataset
                            val_dataset = model_2_val_dataset
                            train_metric = model_2_train_metric
                            device = MODEL_2_DEVICE
                            df_list = model_2_metrics_df_list
                        model = get_model(model_name, model_config["hyperparameters"], num_features, num_classes)
                        metrics = ttutils.prepare_test_metrics(num_classes, binary_class=use_binary_metrics, confusion_matrix=False, device=device)
                        accuracy, precision, recall, f1_score = train_test_model(None, model, model_config, train_dataset, test_dataset, 
                                                                                    val_dataset, metrics, device, train_metric)
                        metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                        df_list.append(metrics_df)                           
                        del model
                else:
                    model_1_config = model_data[model_1_type]
                    model_2_config = model_data[model_2_type]
                    model_1_metrics = ttutils.prepare_test_metrics(num_classes, binary_class=use_binary_metrics, confusion_matrix=False,
                                                                       device=MODEL_1_DEVICE)
                    model_2_metrics = ttutils.prepare_test_metrics(num_classes, binary_class=use_binary_metrics, confusion_matrix=False,
                                                                device=MODEL_2_DEVICE)
                    model_1_par_conn, model_1_child_conn = mp.Pipe()
                    model_2_par_conn, model_2_child_conn = mp.Pipe()
                    model_1_process = mp.Process(target=train_test_model, daemon=False, args=(model_1_child_conn,
                                                                                            get_model(model_1_type, model_1_config["hyperparameters"], 
                                                                                                      model_1_num_features, num_classes=num_classes),
                                                                                            model_1_config, model_1_train_dataset, 
                                                                                            model_1_test_dataset, model_1_val_dataset, model_1_metrics,
                                                                                            MODEL_1_DEVICE, model_1_train_metric))
                    model_2_process = mp.Process(target=train_test_model, daemon=False, args=(model_2_child_conn,
                                                                                        get_model(model_2_type, model_2_config["hyperparameters"], 
                                                                                                  model_2_num_features, num_classes ),
                                                                                        model_2_config, model_2_train_dataset,
                                                                                        model_2_test_dataset, model_2_val_dataset, model_2_metrics,
                                                                                        MODEL_2_DEVICE, model_2_train_metric))
                    model_1_process.start()
                    model_2_process.start()
                    accuracy, precision, recall, f1_score = model_1_par_conn.recv()
                    model_1_process.join()
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                    model_1_metrics_df_list.append(metrics_df)
                    accuracy, precision, recall, f1_score = model_2_par_conn.recv()
                    model_2_process.join()
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [fold+1], accuracy, precision, recall, f1_score])))
                    model_2_metrics_df_list.append(metrics_df)
    except Exception as e:
        print("Exception occurred!", e, file=sys.stderr)
    finally:
        if len(model_1_metrics_df_list) > 0:
            print("Saving statistics...")
            model_1_results_df = pd.concat(model_1_metrics_df_list)
            print("Transformer results", model_1_results_df, sep='\n')
            model_2_results_df = pd.concat(model_2_metrics_df_list)
            print("LSTM results", model_2_results_df, sep='\n')
            del model_1_metrics_df_list, model_2_metrics_df_list
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            ttest_pvalues_list = []
            wilcoxon_pvalues_list = []
            for metric_name in metric_names:
                ttest_result = ttest_rel(model_1_results_df[metric_name], model_2_results_df[metric_name])
                wilcoxon_result = wilcoxon(model_1_results_df[metric_name], model_2_results_df[metric_name])
                ttest_pvalue = ttest_result.pvalue
                wilcoxon_pvalue = wilcoxon_result.pvalue
                ttest_pvalues_list.append(ttest_pvalue)
                wilcoxon_pvalues_list.append(wilcoxon_pvalue)
            pvalues_df = pd.DataFrame.from_dict(dict(zip(p_values_names, [metric_names, ttest_pvalues_list, wilcoxon_pvalues_list])))
            print("P-values", pvalues_df, sep='\n')
            model_1_results_filename = f"results_{model_1_type}_comparator"
            model_2_results_filename = f"results_{model_2_type}_comparator"
            pvalues_filename = f"pvalues_{dataset_name}"
            if args.remove_features:
                model_1_results_filename += "_removed"
                model_2_results_filename += "_removed"
                pvalues_filename += "_removed"
            if dataset_percentage < 100:
                model_1_results_filename += f"_{dataset_percentage}"
                model_2_results_filename += f"_{dataset_percentage}"
                pvalues_filename += f"_{dataset_percentage}"
            model_1_results_df.to_csv(os.path.join(results_dir, f"{model_1_results_filename}.csv"))
            model_2_results_df.to_csv(os.path.join(results_dir, f"{model_2_results_filename}.csv"))
            pvalues_df.to_csv(os.path.join(results_dir, f"{pvalues_filename}.csv"))
