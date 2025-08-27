import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torcheval.metrics import MulticlassAccuracy, BinaryAccuracy
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import torchinfo
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import matplotlib
from utils import dataset_utils
from utils import train_test_utils as ttutils
from models import MyModel, MyLSTMClassifier


#SEED = 42

if __name__ == "__main__":
    #torch.manual_seed(SEED)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("model", choices=("lstm", "transformer"), help="The model type")
    parser.add_argument("label_column", type=str, help="The name of the column that will be used as class.", default="Label")
    parser.add_argument("runs", type=int, help="Number of runs. Must be positive", default=1)
    parser.add_argument("folds", type=int, help="Number of folds. Must not be negative. Ignored if used with either few-shot or zero-shot.",default=0)
    parser.add_argument("epochs", type=int, help="Number of training epochs. Must be positive", default=10)
    parser.add_argument("-c", "--config", type=str, help="Path to hyperparameter configuration file")
    parser.add_argument("-r", action="store_true", help="Remove network specific features", dest="remove_features")
    parser.add_argument("-e", "--early-stop", action="store_true", help="Use early stopping")
    parser.add_argument("-b", "--binary-metrics", action="store_true", help="Use binary metrics instead of multiclass")
    parser.add_argument("-p", "--dataset-percentage", type=int, help="Percentage of dataset to use (0-100), default is 100", default=100)
    dataset_args = parser.add_mutually_exclusive_group(required=True)
    dataset_args.add_argument("-f", "--file", type=str, help="Dataset CSV file", dest="dataset_file")
    dataset_args.add_argument("-d", "--directory", type=str, help="Dataset directory containing CSV files", dest="dataset_folder")
    shot_args = parser.add_mutually_exclusive_group(required=False)
    shot_args.add_argument("-z", "--zero-shot", action="store_true", help="Run zero-shot transfer learning")
    shot_args.add_argument("-fs", "--few-shot", type=int, metavar="SAMPLES_PER_CLASS", help="Run few-shot transfer learning with the specified samples per class")
    args = parser.parse_args()
    use_transformer = True
    load_directory = False
    zero_shot = False
    N_WORKERS = 8
    DEVICE = ttutils.get_device()
    matplotlib.use('Agg') # Use Agg engine for headless operation

    try:
        model_name = args.model
        config_file = args.config
        use_early_stop = args.early_stop
        use_binary_metrics = args.binary_metrics
        dataset_percentage = args.dataset_percentage
        raytune_results_dir = os.path.join("tests", "results", "raytune")
        if use_early_stop:
            print("Using early stopping!")
        if model_name == "lstm":
            use_transformer = False
            if config_file is None:
                config_file = os.path.join(raytune_results_dir, "test_raytune_lstm", "best_config.json")
                print(f"Config file not specified! Defaulting to {config_file}!")
        else:
            if config_file is None:
                config_file = os.path.join(raytune_results_dir, "test_raytune_transformer", "best_config.json")
                print(f"Config file not specified! Defaulting to {config_file}!")
        with open(config_file, "r") as file:
            json_data = json.load(file)
            config = json_data["config"]
        learning_rate = config.pop("lr")
        batch_size = config.pop("batch_size")
        model_hyperparameters = config
        if (args.zero_shot or args.few_shot) and args.dataset_folder is None:
            raise ValueError("Transfer learning requires a directory that contains datasets!")
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
        few_shot = bool(args.few_shot)
        few_shot_samples_per_class = args.few_shot
        if few_shot_samples_per_class is not None and few_shot_samples_per_class <= 0:
            raise ValueError("Invalid samples per class! Must be positive")
        if folds < 0 or epochs < 1 or num_runs < 1:
            raise ValueError("Please specify valid number of folds and epochs")
        if args.remove_features:
            #dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Fwd Seg Size Min", "Init Bwd Win Byts",
            #                   "Init Fwd Win Byts", "Dst Port", "Idle Min", "Idle Max"]
            #dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Fwd Seg Size Min", "Init Bwd Win Byts"]
            dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Fwd Seg Size Min", "Init Bwd Win Byts",
                               "Idle Mean", "Idle Min", "Idle Max"]
        else:
            dropped_columns = ["Timestamp"]
        print("The following features will be ignored:")
        print(*dropped_columns, sep=',')
    except ValueError as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    results_dir = os.path.join("tests", "results")
    trained_models_dir = os.path.join(results_dir, "trained_models")
    images_dir = os.path.join(results_dir, "images")
    os.makedirs(trained_models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    tl_type = None #needed for zero/few shot transfer learning as well
    PATIENCE = ttutils.get_patience(epochs) # Needed for early stopping
    DELTA = 2.5e-3 # Needed for early stopping

    if not load_directory:
        dataset_name = os.path.basename(dataset_file).split(".")[0]
        model_file = f"best_model_{model_name}_{dataset_name}"
        csv_dataset = dataset_utils.CSVDataset(dataset_file, label_column, columns_to_drop=dropped_columns, chunk_size=3e+6)
        rows_limit = int(250e+3 * dataset_percentage / 100)
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
        rows_per_dataset = 77140 * dataset_percentage / 100
        if zero_shot or few_shot:
            print(f"Running in {'zero-shot' if zero_shot else 'few-shot'} mode!")
            tl_type = f"{'zero_shot' if zero_shot else 'few_shot'}"
            model_file = f"best_model_{model_name}_{tl_type}_TL"
            dataset_list = dataset_utils.load_datasets_from_dir(dataset_folder, label_column, rows_per_dataset=rows_per_dataset,
                                                                drop_columns=dropped_columns, balance_classes=True, as_tensors_list=True)
            num_rows = sum([dataset.num_rows for dataset in dataset_list])
            num_classes = dataset_list[0].num_classes
            num_features = dataset_list[0].num_features
            category_map = dataset_list[0].categories
            feature_names = dataset_list[0].feature_names
            datatype = dataset_list[0].dtype
        else:
            model_file = f"best_model_{model_name}_merged_ds"
            loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_folder, label_column, rows_per_dataset=rows_per_dataset,
                                                                  drop_columns=dropped_columns, balance_classes=True)
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
    if args.remove_features:
        dataset_name += "_removed"
        model_file += "_removed"
    if dataset_percentage < 100:
        dataset_name += f"_{dataset_percentage}"
        model_file += f"f_{dataset_percentage}"

    model_filename = os.path.join(trained_models_dir, f"{model_file}.pt")
    print(f"# of rows: {num_rows}, # of features: {num_features}, # of classes: {num_classes}, datatype: {datatype}")
    df_list = []
    local_test_df_list = []  # needed for zero/few shot transfer learning, otherwise is unused
    if use_binary_metrics:
        metric_names = ["Run #", "Fold #", "Class", "Accuracy", "Precision", "Recall", "F1 Score"]
        training_metric = BinaryAccuracy(device=DEVICE)
    else:
        metric_names = ["Run #", "Fold #", "Class", "Accuracy", "Precision", "Recall", "F1 Score"]
        training_metric = MulticlassAccuracy(average='macro', num_classes=num_classes, device=DEVICE)
    for i in range(num_runs):
        print("#"*50,f"RUN {i+1}", "#"*50)
        if zero_shot or few_shot:
            dataset_loo = LeaveOneOut()
            show_summary = True
            for fold, (_, testset_idx) in enumerate(dataset_loo.split(dataset_list)):
                print(f"Using dataset {testset_idx.item()+1} as test!")
                # On the following datasets, tensor with index 0 are data rows with features, tensor index 1 is the corresponding labels
                temp_dataset_list = dataset_list.copy()
                test_dataset = temp_dataset_list.pop(testset_idx.item()).dataset
                train_dataset_list = temp_dataset_list
                X = train_dataset_list[0].dataset.tensors[0]
                y = train_dataset_list[0].dataset.tensors[1]
                for j in range(1, len(train_dataset_list)):
                    X = torch.cat((X, train_dataset_list[j].dataset.tensors[0]), dim=0)
                    y = torch.cat((y, train_dataset_list[j].dataset.tensors[1]), dim=0)
                del train_dataset_list
                train_dataset = TensorDataset(X, y)
                class_values = category_map.keys()
                if few_shot:
                    X_test_initial = test_dataset.tensors[0]
                    y_test_initial = test_dataset.tensors[1]
                    # shuffle test set so infused samples are random each time
                    random_indices = torch.randperm(X_test_initial.shape[0])
                    X_test_initial = X_test_initial[random_indices]
                    y_test_initial = y_test_initial[random_indices]
                    del random_indices
                    infused_indices_list = []
                    for class_value in class_values:
                        infused_samples_indexes = (y_test_initial == class_value).nonzero(as_tuple=True)[0][:few_shot_samples_per_class]
                        infused_indices_list += infused_samples_indexes
                        del infused_samples_indexes
                    X_infused = X_test_initial[infused_indices_list]
                    y_infused = y_test_initial[infused_indices_list]
                    test_indices_mask = torch.ones(y_test_initial.shape[0], dtype=torch.bool)
                    test_indices_mask[infused_indices_list] = False
                    X_test = X_test_initial[test_indices_mask]
                    y_test = y_test_initial[test_indices_mask]
                    X = torch.cat((X, X_infused), dim=0)
                    y = torch.cat((y, y_infused), dim=0)
                    test_dataset = TensorDataset(X_test, y_test)
                    del X_test, y_test, X_infused, y_infused, infused_indices_list, test_indices_mask
                    print(f"Final # of rows after infusion: {X.shape[0]}")
                    train_dataset = TensorDataset(X, y)
                    del X_test_initial, y_test_initial
                del X, y
                val_loader = None
                val_dataset = None
                early_stopper = None
                if use_early_stop:
                   train_dataset, val_dataset, local_test_dataset = random_split(train_dataset, [0.6, 0.2, 0.2])
                   early_stopper = ttutils.EarlyStopping(PATIENCE, DELTA)
                else:
                    train_dataset, local_test_dataset = random_split(train_dataset, [0.8, 0.2])
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=N_WORKERS)
                local_test_loader = DataLoader(local_test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                if val_dataset is not None:
                    val_loader = DataLoader(val_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                test_metrics = ttutils.prepare_test_metrics(num_classes, DEVICE, use_binary_metrics)
                local_test_metrics = ttutils.prepare_test_metrics(num_classes, DEVICE, use_binary_metrics)
                if use_transformer:
                    model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                if show_summary:
                    torchinfo.summary(model, input_size=(batch_size, num_features))
                    show_summary = False
                print("STARTING TRAINING SESSION!!!")
                ttutils.train_model(model, model_filename, train_loader, val_loader, training_metric, epochs=epochs, device=DEVICE,
                            learning_rate=learning_rate, early_stopper=early_stopper)
                print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                if use_transformer:
                    final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                final_model.load_state_dict(torch.load(model_filename, weights_only=True))
                multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_confusion_matrix = ttutils.test_model(
                    final_model, test_loader, test_metrics, device=DEVICE)
                local_test_multiclass_accuracy, local_test_multiclass_precision, local_test_multiclass_recall, local_test_multiclass_f1_score, _ = ttutils.test_model(
                    final_model, local_test_loader, local_test_metrics, device=DEVICE)
                if use_binary_metrics:
                    metrics_df = pd.DataFrame.from_dict(
                        dict(zip(metric_names, [[i + 1], [fold + 1], multiclass_accuracy,
                                                multiclass_precision, multiclass_recall,
                                                multiclass_f1_score])))
                    local_test_metrics_df = pd.DataFrame.from_dict(
                        dict(zip(metric_names, [[i + 1], [fold + 1], local_test_multiclass_accuracy,
                                                local_test_multiclass_precision, local_test_multiclass_recall,
                                                local_test_multiclass_f1_score])))
                else:
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i + 1] * num_classes, [fold+1] * num_classes,
                                                                                category_map.values(), multiclass_accuracy,
                                                                                multiclass_precision, multiclass_recall,
                                                                                multiclass_f1_score])))
                    local_test_metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i + 1] * num_classes, [fold+1] * num_classes,
                                                                                category_map.values(), local_test_multiclass_accuracy,
                                                                                local_test_multiclass_precision, local_test_multiclass_recall,
                                                                                local_test_multiclass_f1_score])))
                del train_loader, test_loader, local_test_loader
                df_list.append(metrics_df)
                local_test_df_list.append(local_test_metrics_df)
                print("Local Test Metrics:\n", local_test_metrics_df, sep='')
                print("Generalization Test Metrics:\n", metrics_df, sep='')
                confusion_matrix_filename = f"confusion_matrix_{model_name}_{dataset_name}_{i+1}_{fold+1}_{tl_type}.png"
                shap_values_filename = f"shap_values_{model_name}_{dataset_name}_{i+1}_{fold+1}_{tl_type}.png"
                ttutils.plot_confusion_matrix(multiclass_confusion_matrix, category_map, os.path.join(images_dir, confusion_matrix_filename))
                ttutils.plot_shap_values(final_model, test_dataset, category_map, num_classes, feature_names, os.path.join(images_dir, shap_values_filename))
        else:
            if folds == 0:
                print("Training and testing model with a random split of 80% train and 20% test!")
                metrics = ttutils.prepare_test_metrics(num_classes, DEVICE, use_binary_metrics)
                if use_transformer:
                    model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                torchinfo.summary(model, input_size=(batch_size, num_features))
                val_loader = None
                val_dataset = None
                early_stopper = None
                if use_early_stop:
                    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2])
                    early_stopper = ttutils.EarlyStopping(PATIENCE, DELTA)
                else:
                    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=N_WORKERS)
                test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                if val_dataset is not None:
                    val_loader = DataLoader(val_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                print("STARTING TRAINING SESSION!!!")
                ttutils.train_model(model, model_filename, train_loader, val_loader, training_metric, epochs=epochs,
                            device=DEVICE, learning_rate=learning_rate, early_stopper=early_stopper)
                print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                if use_transformer:
                    final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                else:
                    final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                final_model.load_state_dict(torch.load(model_filename, weights_only=True))
                multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_confusion_matrix = ttutils.test_model(final_model, test_loader, metrics, device=DEVICE)
                if use_binary_metrics:
                    metrics_df = pd.DataFrame.from_dict(
                        dict(zip(metric_names, [[i + 1], [1],
                                                multiclass_accuracy, multiclass_precision,
                                                multiclass_recall, multiclass_f1_score])))
                else:
                    metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1], [1],
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
                    metrics = ttutils.prepare_test_metrics(num_classes, DEVICE, use_binary_metrics)
                    if use_transformer:
                        model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                    else:
                        model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                    if show_summary:
                        torchinfo.summary(model, input_size=(batch_size, num_features))
                        show_summary = False
                    print("-"*50)
                    print(f"Fold {fold+1}/{folds}")
                    val_loader = None
                    val_dataset = None
                    early_stopper = None
                    train_dataset = Subset(dataset, train_index)
                    if use_early_stop:
                        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
                        early_stopper = ttutils.EarlyStopping(PATIENCE, DELTA)
                    test_dataset = Subset(dataset, test_index)
                    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=N_WORKERS)
                    test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                    if val_dataset is not None:
                        val_loader = DataLoader(val_dataset, batch_size, pin_memory=True, num_workers=N_WORKERS)
                    print("STARTING TRAINING SESSION!!!")
                    ttutils.train_model(model, model_filename, train_loader, val_loader, metric=training_metric, epochs=epochs,
                                device=DEVICE, learning_rate=learning_rate, early_stopper=early_stopper)
                    print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
                    if use_transformer:
                        final_model = MyModel(num_features, num_classes, **model_hyperparameters).to(DEVICE)
                    else:
                        final_model = MyLSTMClassifier(num_classes, **model_hyperparameters).to(DEVICE)
                    final_model.load_state_dict(torch.load(model_filename, weights_only=True))
                    multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, fold_confusion_matrix = ttutils.test_model(final_model, test_loader, metrics, device=DEVICE)
                    multiclass_confusion_matrix += fold_confusion_matrix.astype(np.uint64)
                    if use_binary_metrics:
                        metrics_df = pd.DataFrame.from_dict(dict(
                            zip(metric_names, [[i + 1], [fold + 1],
                                               multiclass_accuracy, multiclass_precision,
                                               multiclass_recall, multiclass_f1_score])))
                    else:
                        metrics_df = pd.DataFrame.from_dict(dict(zip(metric_names, [[i+1]*num_classes, [fold+1]*num_classes, category_map.values(),
                                                                                multiclass_accuracy, multiclass_precision,
                                                                                multiclass_recall, multiclass_f1_score])))
                    del train_loader, test_loader
                    df_list.append(metrics_df)
                    print("Metrics:\n", metrics_df, sep='')
                    print("-"*50)
            print(f"[Run {i+1}] Multiclass confusion matrix:\n", multiclass_confusion_matrix, sep='')
            ttutils.plot_confusion_matrix(multiclass_confusion_matrix, category_map, os.path.join(images_dir, f"confusion_matrix_{model_name}_{dataset_name}_{i+1}.png"))
            ttutils.plot_shap_values(final_model, dataset, category_map, num_classes, feature_names, os.path.join(images_dir, f"shap_values_{model_name}_{dataset_name}_{i+1}.png"))
    results_df = pd.concat(df_list)
    if zero_shot or few_shot:
        local_test_results_df = pd.concat(local_test_df_list)
        local_test_results_df.to_csv(os.path.join(results_dir, f"local_test_results_{model_name}_{dataset_name}_{tl_type}.csv"))
        results_df.to_csv(os.path.join(results_dir, f"results_{model_name}_{dataset_name}_{tl_type}.csv"))
    else:
        results_df.to_csv(os.path.join(results_dir, f"results_{model_name}_{dataset_name}.csv"))
