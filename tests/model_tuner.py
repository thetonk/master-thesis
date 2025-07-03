import os
os.environ["RAY_DEDUP_LOGS"] = '0'
os.environ["RAY_USAGE_STATS_ENABLED"] = '0'
import sys
import tempfile
import json
import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import MulticlassAccuracy
import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
#from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
import dataset_utils
from models import MyModel, MyLSTMClassifier, train_model

def prepare_tunable_training(dataset_id, epochs:int, n_features:int, n_classes: int, use_transformer: bool = True, device = "cuda"):
    def tunable_training(config):
        dataset = ray.get(dataset_id)
        batch_size = config["batch_size"]
        train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])
        del dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
        metric = MulticlassAccuracy(average='macro', num_classes=n_classes, device=device)
        if use_transformer:
            model = MyModel(n_features, n_classes, config["num_encoders"], config["num_mlps"], config["enc_embedding_dim"],
                            config["enc_num_heads"], config["enc_ff_neurons"], config["mlp_hidden_neurons"]).to(device)
        else:
            model = MyLSTMClassifier(n_classes, config["hidden_lstm_states"], config["hidden_mlp_neurons"]).to(device)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmpfile:
            tmpfilename = tmpfile.name
            train_model(model, tmpfilename, train_loader, validation_loader, metric=metric, epochs=epochs,
                        learning_rate=config["lr"], train_tune=True, device=device)
    return tunable_training


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
        DEVICE = "cuda"
    else:
        print("CUDA is not available")
        DEVICE = "cpu"
    HELPTEXT = f"Usage: {sys.argv[0]} RUN_MODE MODEL DATASET_FOLDER LABEL_COLUMN"
    use_transformer = True
    use_slurm = False
    if len(sys.argv) < 5:
        print("Error! You must specify model, dataset folder and label column! Exiting!", file=sys.stderr)
        print(HELPTEXT)
        sys.exit(1)
    else:
        run_mode = sys.argv[1]
        model_name = sys.argv[2]
        if model_name.lower() == "lstm":
            use_transformer = False
        if run_mode.lower() == "slurm":
            use_slurm = True
            # running on aristotle HPC
            if use_transformer:
                tune_resources = {"cpu": 4, "gpu": 0.25}
            else:
                tune_resources = {"cpu": 4, "gpu": 0.5}
        else:
            if use_transformer:
                tune_resources = {"cpu": 4, "gpu": 0.5}
            else:
                tune_resources = {"cpu": 4, "gpu": 1}
        dataset_folder_path = sys.argv[3]
        label_column = sys.argv[4]
        raytune_dir = os.path.realpath(os.path.join("tests", "results", "raytune"))
        rows_limit = int(400e+3)
        os.makedirs(raytune_dir, exist_ok=True)

    loaded_dataset = dataset_utils.load_datasets_from_dir(dataset_folder_path, label_column, total_rows_limit=rows_limit)
    dataset, num_rows, num_features, num_classes = loaded_dataset.dataset, loaded_dataset.num_rows, loaded_dataset.num_features, loaded_dataset.num_classes
    del loaded_dataset
    print(f"# of rows: {num_rows}, # of features: {num_features}, # of classes: {num_classes}")
    epochs = 10 if use_transformer else 5
    num_samples = 300 if use_transformer else 100
    if use_slurm:
        ray.init(include_dashboard=False, address=os.getenv("TUNER_HEAD_IP_ADDRESS"), _redis_password=os.getenv("TUNER_REDIS_PASSWORD"))
    else:
        ray.init(include_dashboard=False)
    if use_transformer:
        search_space = {"lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
                        "batch_size": tune.choice( [32, 64, 128, 256]),
                        "enc_embedding_dim": tune.choice([32, 64, 128, 256]),
                        "enc_num_heads": tune.choice([4,8,16,32]),
                        "enc_ff_neurons": tune.choice([64, 128, 256, 512]),
                        "mlp_hidden_neurons": tune.choice([128, 256, 512, 1024]),
                        "num_encoders": tune.choice([1,2,3,4]),
                        "num_mlps": tune.choice([1,2,3,4])}
    else:
        search_space = {"lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
                        "batch_size": tune.choice([32, 64, 128, 256]),
                        "hidden_lstm_states": tune.choice([128, 256, 512]),
                        "hidden_mlp_neurons": tune.choice([128, 256, 512, 1024])}
    hyperband = HyperBandScheduler(time_attr="training_iteration", max_t=epochs, reduction_factor=2)
    #hyperopt_search = HyperOptSearch()
    initial_config = [{
        'lr': 0.0001, 'batch_size': 64, 'enc_embedding_dim': 128, 'enc_num_heads': 8, 'enc_ff_neurons': 256,
        'mlp_hidden_neurons': 256, 'num_encoders': 4, 'num_mlps': 1
    }]
    nevergrad_search = NevergradSearch(optimizer=ng.optimizers.DiscreteOnePlusOne, points_to_evaluate=initial_config)
    dataset_object_id = ray.put(dataset)
    tune_with_resources = tune.with_resources(
        prepare_tunable_training(dataset_object_id, epochs, num_features, num_classes, use_transformer, DEVICE), 
        resources=tune_resources)
    experiment_name = f"test_raytune_{model_name}"
    experiment_dir = os.path.join(raytune_dir, experiment_name)
    tuner = tune.Tuner(
        tune_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_accuracy",
            mode="max",
            scheduler=hyperband,
            search_alg=nevergrad_search
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=raytune_dir,
            log_to_file=os.path.join(experiment_dir, "raytune_output_combined.log")
        )
    )
    result = tuner.fit().get_best_result()
    best_config_file = os.path.join(experiment_dir, "best_config_new.json")
    with open(best_config_file, "w") as f:
        json_content = {
            "config": result.config,
            "metrics": result.metrics
        }
        json.dump(json_content, f, sort_keys=True, indent=4)
