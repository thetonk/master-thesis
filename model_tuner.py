# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import os
os.environ["RAY_DEDUP_LOGS"] = '0'
os.environ["RAY_USAGE_STATS_ENABLED"] = '0'
import argparse
import tempfile
import json
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torcheval.metrics import MulticlassAccuracy
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
#from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
from utils import dataset_utils
from utils.train_test_utils import train_model, get_device
from utils.models import ModelTypes, get_model
from utils.exceptions import InvalidArgumentException


def prepare_tunable_training(dataset_id, epochs:int, n_features:int, n_classes: int, model_type: ModelTypes, device = torch.device("cuda")):
    def tunable_training(config: dict):
        dataset: TensorDataset = ray.get(dataset_id)
        config_copy = config.copy()
        batch_size = config_copy.pop("batch_size")
        learning_rate = config_copy.pop("lr")
        train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])
        del dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=True)
        metric = MulticlassAccuracy(average='macro', num_classes=n_classes, device=device)
        model = get_model(model_type, config_copy, n_features, n_classes).to(device)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmpfile:
            tmpfilename = tmpfile.name
            train_model(model, tmpfilename, train_loader, validation_loader, metric=metric, epochs=epochs,
                        learning_rate=learning_rate, train_tune=True, device=device)
    return tunable_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("run_mode", choices=("slurm", "local"), default="local", help="Run on SLURM or locally. Required for Ray")
    parser.add_argument("model", choices=[model.value for model in ModelTypes], help="Model type")
    parser.add_argument("label_column", type=str, help="The name of the column to be used as class", default="Label")
    parser.add_argument("-r", action="store_true", help="Remove network specific features according to model type", dest="remove_features")
    parser.add_argument("-u", "--unified-removal", action="store_true", help="Reemove network specific features regardless of model type")
    dataset_args = parser.add_mutually_exclusive_group(required=False)
    dataset_args.add_argument("-f", "--file", type=str, help="Dataset CSV file", dest="dataset_file")
    dataset_args.add_argument("-d", "--directory", type=str, help="Dataset directory containing CSV files", dest="dataset_folder")
    args = parser.parse_args()
    DEVICE = get_device()
    use_directory = False
    if args.dataset_folder is not None:
        use_directory = True
    use_slurm = False
    run_mode = args.run_mode
    model_name = args.model
    label_column = args.label_column

    if run_mode.lower() == "slurm":
        # running on aristotle HPC
        use_slurm = True
        if model_name == ModelTypes.TRANSFORMER:
            tune_resources = {"cpu": 8, "gpu": 0.25}
        elif model_name == ModelTypes.LSTM:
            tune_resources = {"cpu": 8, "gpu": 0.5}
        else:
            tune_resources = {"cpu": 8, "gpu": 0.2}
    else:
        if model_name == ModelTypes.TRANSFORMER:
            tune_resources = {"cpu": 8, "gpu": 0.5}
        elif model_name == ModelTypes.LSTM:
            tune_resources = {"cpu": 8, "gpu": 1}
        else:
            tune_resources = {"cpu": 8, "gpu": 0.25}

    if args.remove_features:
        if args.unified_removal:
            dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Fwd Seg Size Min", "Init Bwd Win Byts",
                                "Init Fwd Win Byts", "Dst Port", "Idle Min", "Idle Max"]
        else:
            if model_name == ModelTypes.TRANSFORMER:
                dropped_columns = ["Timestamp", "Src IP", "Dst IP", "Idle Mean", "Idle Min", "Idle Max"]
            elif model_name == ModelTypes.LSTM:
                dropped_columns = ["Timestamp", "Fwd Seg Size Min"]
            else:
                # for now
                raise NotImplementedError("Feature removal for CNN is currently not supported!")
        experiment_name = f"test_raytune_removed_features_{model_name}"
        best_config_file = "removed_features_best_config.json"
        
    else:
        if args.unified_removal:
            raise InvalidArgumentException("You must enable feature removal in order to use the unified feature removal variant!")
        dropped_columns = ["Timestamp"]
        experiment_name = f"test_raytune_{model_name}"
        best_config_file = "best_config.json"

    print("The following features will be ignored:")
    print(*dropped_columns, sep=',')

    raytune_dir = os.path.realpath(os.path.join("results", "raytune"))
    experiment_dir = os.path.join(raytune_dir, experiment_name)
    best_config_file_path = os.path.join(experiment_dir, best_config_file)
    rows_limit = int(400e+3)
    os.makedirs(raytune_dir, exist_ok=True)
    if use_directory:
        loaded_dataset = dataset_utils.load_datasets_from_dir(args.dataset_folder, label_column,
                                                          drop_columns=dropped_columns, total_rows_limit=rows_limit, balance_classes=True)
        dataset, num_rows, num_features, num_classes = loaded_dataset.dataset, loaded_dataset.num_rows, loaded_dataset.num_features, loaded_dataset.num_classes
        del loaded_dataset
    else:
        csv_dataset = dataset_utils.CSVDataset(args.dataset_file, label_column, columns_to_drop=dropped_columns, chunk_size=3e+6)
        csv_dataset.load(balance_classes=True, rows_limit=rows_limit)
        X, y, num_rows, num_features, num_classes = csv_dataset.X, csv_dataset.y, csv_dataset.n_rows, csv_dataset.n_features, csv_dataset.n_classes
        dataset = TensorDataset(X, y)
        del csv_dataset
    print(f"# of rows: {num_rows}, # of features: {num_features}, # of classes: {num_classes}")
    if use_slurm:
        slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        slurm_gpus = int(os.getenv("SLURM_GPUS", 0))
        ray.init(include_dashboard=False, num_cpus=slurm_cpus, num_gpus=slurm_gpus)
    else:
        ray.init(include_dashboard=False)
    print("Ray will use the following resources:", ray.available_resources())
    if model_name == ModelTypes.TRANSFORMER:
        search_space = {"lr": tune.loguniform(1e-4, 5e-3),
                        "batch_size": tune.choice( [32, 64, 128, 256]),
                        "enc_embedding_dim": tune.choice([32, 64, 128, 256]),
                        "enc_num_heads": tune.choice([4,8,16,32]),
                        "enc_ff_neurons": tune.choice([64, 128, 256, 512]),
                        "enc_ff_dropout": tune.choice([0, 0.1, 0.2, 0.3]),
                        "enc_attn_dropout": tune.choice([0, 0.1, 0.2]),
                        "mlp_hidden_neurons": tune.choice([128, 256, 512, 1024]),
                        "num_encoders": tune.choice([1,2,3]),
                        "num_mlps": tune.choice([1,2,3,4,5]),
                        "mlp_dropout": tune.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
                        }
        initial_config = [{
            'lr': 0.0001, 'batch_size': 64,
            'enc_embedding_dim': 128, 'enc_num_heads': 8, 'enc_ff_neurons': 256, 'enc_ff_dropout': 0, 'enc_attn_dropout': 0,
            'mlp_hidden_neurons': 256, 'num_encoders': 1, 'num_mlps': 1, 'mlp_dropout': 0.1
        }]
        epochs = 30
        num_samples = 500
    elif model_name == ModelTypes.LSTM:
        search_space = {"lr": tune.loguniform(1e-4, 5e-3),
                        "batch_size": tune.choice([32, 64, 128, 256]),
                        "hidden_lstm_states": tune.choice([128, 256, 512]),
                        "hidden_mlp_neurons": tune.choice([128, 256, 512, 1024]),
                        "mlp_dropout": tune.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])}
        initial_config = [{
            "lr": 0.0001, "batch_size": 64, "hidden_lstm_states": 512, "hidden_mlp_neurons": 1024, "mlp_dropout": 0.1
        }]
        epochs = 5
        num_samples = 150
    else:
        search_space = {"lr": tune.loguniform(5e-4, 5e-3),
                        "batch_size": tune.choice([32, 64, 128, 256]),
                        "hidden_mlp_neurons": tune.choice([64, 128, 256, 512, 1024]),
                        "mlp_dropout": tune.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
                        "conv_layers": tune.randint(4, 16),
                        "kernel_size": tune.randint(3, 11),
                        "padding": tune.randint(0, 4)
                    }
        initial_config = [{"lr": 1e-3, "batch_size": 128, "hidden_mlp_neurons": 64, "mlp_dropout": 0.1, "conv_layers": 6, 
                           "kernel_size": 5, "padding": 1}]
        num_samples = 1000
        epochs = 40

    #hyperband = HyperBandScheduler(time_attr="training_iteration", max_t=epochs, reduction_factor=2)
    asha = ASHAScheduler(time_attr="training_iteration", max_t=epochs, reduction_factor=2)
    #hyperopt_search = HyperOptSearch()
    # Hybrid genetic / differential evolution algorithm
    #nevergrad_search = NevergradSearch(optimizer=ng.optimizers.GeneticDE, points_to_evaluate=initial_config)
    nevergrad_search = NevergradSearch(optimizer=ng.optimizers.DiscreteOnePlusOne, points_to_evaluate=initial_config)
    dataset_object_id = ray.put(dataset)
    tune_with_resources = tune.with_resources(
        prepare_tunable_training(dataset_object_id, epochs, num_features, num_classes, model_name, DEVICE), 
        resources=tune_resources)
    tuner = tune.Tuner(
        tune_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_accuracy",
            mode="max",
            scheduler=asha,
            search_alg=nevergrad_search
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=raytune_dir,
            log_to_file=os.path.join(experiment_dir, "raytune_output_combined.log")
        )
    )
    result = tuner.fit().get_best_result()
    with open(best_config_file_path, "w") as f:
        json_content = {
            "config": result.config,
            "metrics": result.metrics
        }
        json.dump(json_content, f, sort_keys=True, indent=4)
