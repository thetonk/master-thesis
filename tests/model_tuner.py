import os
import sys
import tempfile
import torch
from torch.utils.data import TensorDataset, DataLoader
#import torchinfo
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
#from ray.tune.search.hyperopt import HyperOptSearch
import dataset_utils
from models import MyModel, MyLSTMClassifier, train_model

def prepare_tunable_training(dataset: torch.utils.data.Dataset, epochs:int, n_features:int, n_classes: int, use_transformer: bool = True):
    def tunable_training(config):
        batch_size = config["batch_size"]
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        if use_transformer:
            model = MyModel(n_features, n_classes, config["n_encoders"], config["n_mlp"], config["enc_embedding_dim"],
                            config["enc_num_heads"], config["enc_ff_neurons"], config["mlp_hidden_neurons"]).to("cuda")
        else:
            model = MyLSTMClassifier(n_classes, config["hidden_lstm_states"], config["hidden_mlp_neurons"]).to("cuda")
        #torchinfo.summary(model, input_size=(batch_size, n_features))
        train_model(model, "test_tune.pt", train_loader, epochs=epochs, learning_rate=config["lr"], train_tune=True)
    return tunable_training


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
        DEVICE = "cuda"
    else:
        print("CUDA is not available")
        DEVICE = "cpu"
    HELPTEXT = f"Usage: {sys.argv[0]} MODEL DATASET_PATH LABEL_COLUMN"
    use_transformer = True
    if len(sys.argv) < 4:
        print("Error! You must specify model, dataset path and label column! Exiting!", file=sys.stderr)
        print(HELPTEXT)
        sys.exit(1)
    else:
        model_name = sys.argv[1]
        if model_name.lower() == "lstm":
            use_transformer = False
        dataset_path = sys.argv[2]
        label_column = sys.argv[3]
        os.makedirs(os.path.join("results", "raytune"), exist_ok=True)
    
    csv_dataset = dataset_utils.CSVDataset(dataset_path, label_column, chunk_size=3e+6)
    csv_dataset.load(balance_classes=True)
    X, y, category_map, feature_names = csv_dataset.X, csv_dataset.y, csv_dataset.categories, csv_dataset.features
    num_classes = len(category_map)
    class_frequencies = y.bincount(minlength=num_classes)
    class_percentages = class_frequencies.float() / y.shape[0]
    print("Class percentages:", class_percentages)
    dataset = TensorDataset(X, y)
    num_features = X.shape[1]
    print(f"# of rows: {X.shape[0]}, # of features: {num_features}, # of classes: {num_classes}, datatype: {X.dtype}")
    epochs = 5
    if use_transformer:
        search_space = {"lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
                        "batch_size": tune.choice([32,64,128, 256]),
                        "enc_embedding_dim": tune.choice([32, 64, 128, 256]),
                        "enc_num_heads": tune.choice([4,8,16,32]),
                        "enc_ff_neurons": tune.choice([64, 128, 256, 512]),
                        "mlp_hidden_neurons": tune.choice([128, 256, 512, 1024]),
                        "n_encoders": tune.choice([1,2,3,4]),
                        "n_mlp": tune.choice([1,2,3,4])}
    else:
        search_space = {"lr": tune.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                        "batch_size": tune.choice([32,64,128,256]),
                        "hidden_lstm_states": tune.choice([128, 256, 512]),
                        "hidden_mlp_neurons": tune.choice([128, 256, 512, 1024])}
    hyperband = HyperBandScheduler(time_attr="training_iteration", max_t=30)
    tune_with_resources = tune.with_resources(
        prepare_tunable_training(dataset, epochs, num_features, num_classes, use_transformer), 
        {"cpu":2, "gpu": 0.5})
    with tempfile.TemporaryDirectory(prefix="_raytune") as tmpdir:
        tuner = tune.Tuner(
            tune_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=20,
                metric="mean_accuracy",
                mode="max",
                scheduler=hyperband
            ),
            run_config=tune.RunConfig(
                name="test_raytune", 
                storage_path=tmpdir,)
        )
        results = tuner.fit()
    print(results.get_best_result().config)