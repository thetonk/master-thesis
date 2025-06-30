import os
os.environ["RAY_DEDUP_LOGS"] = '0'
import sys
import tempfile
import torch
from torch.utils.data import TensorDataset, DataLoader
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
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        metric = MulticlassAccuracy(average='macro', num_classes=n_classes, device=device)
        if use_transformer:
            model = MyModel(n_features, n_classes, config["num_encoders"], config["num_mlps"], config["enc_embedding_dim"],
                            config["enc_num_heads"], config["enc_ff_neurons"], config["mlp_hidden_neurons"]).to(device)
        else:
            model = MyLSTMClassifier(n_classes, config["hidden_lstm_states"], config["hidden_mlp_neurons"]).to(device)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmpfile:
            tmpfilename = tmpfile.name
            train_model(model, tmpfilename, train_loader, metric, epochs=epochs, learning_rate=config["lr"], train_tune=True, device=device)
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
    
    # assume that all datasets have same amount of classes and have same label column and features
    # first pass, discover num of classes and feature names
    dataset_list = []
    for root, _, files in os.walk(dataset_folder_path):
        for file in files:
            filename_path = os.path.join(root, file)
            dataset_list.append(filename_path)
    dataset_list.sort(key=lambda filename: os.path.getsize(filename))
    num_datasets = len(dataset_list)
    print("Number of datasets found: ", num_datasets)
    csv_dataset = dataset_utils.CSVDataset(dataset_list[0], label_column, chunk_size=3e+6)
    csv_dataset.load(balance_classes=False, rows_limit=10)
    num_classes = len(csv_dataset.categories)
    feature_names = csv_dataset.features
    num_features = csv_dataset.X.shape[1]
    # second pass, merge datasets into a large single dataset
    rows_per_dataset = int(rows_limit / num_datasets)
    X = None
    y = None
    for dataset_path in dataset_list:
        print(f"Loading {dataset_path}...")
        csv_dataset = dataset_utils.CSVDataset(dataset_path, label_column, chunk_size=3e+6)
        csv_dataset.load(balance_classes=True, rows_limit=rows_per_dataset)
        if X is None:
            X, y = csv_dataset.X, csv_dataset.y
        else:
            X = torch.cat((X, csv_dataset.X), dim=0)
            y = torch.cat((y, csv_dataset.y), dim=0)
        del csv_dataset
        print("Done!")
    dataset = TensorDataset(X, y)
    print(f"# of rows: {X.shape[0]}, # of features: {num_features}, # of classes: {num_classes}, datatype: {X.dtype}")
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
    tuner = tune.Tuner(
        tune_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="mean_accuracy",
            mode="max",
            scheduler=hyperband,
            search_alg=nevergrad_search
        ),
        run_config=tune.RunConfig(
            name=f"test_raytune_{model_name}", 
            storage_path=raytune_dir
        )
    )
    result = tuner.fit().get_best_result()
    best_config_file = os.path.join(raytune_dir, f"test_raytune_{model_name}", "best_config.log")
    with open(best_config_file, "w") as f:
        print("Best config:", result.config, "Best macro accuracy average:", result.metrics["mean_accuracy"], file=f)
