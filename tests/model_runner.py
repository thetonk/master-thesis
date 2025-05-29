import sys
import torch
from torcheval.metrics import MulticlassAccuracy
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
#import torchinfo
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import shap
from shap.plots import bar
import dataset_utils
from models import MyModel, train_model, test_model

SEED = 42

if __name__ == "__main__":
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        print("CUDA available! GPU device name is:", torch.cuda.get_device_name())
        DEVICE = "cuda"
    else:
        print("CUDA is not available")
        DEVICE = "cpu"

    HELPTEXT = f"Usage: {sys.argv[0]} DATASET_PATH LABEL_COLUMN N_FOLDS N_EPOCHS"

    if len(sys.argv) < 5:
        print("Error! You must specify label column name, number of folds and number of epochs. Exiting!", file=sys.stderr)
        print(HELPTEXT)
        sys.exit(1)
    else:
        try:
            dataset_path = sys.argv[1]
            label_column = sys.argv[2]
            folds = int(sys.argv[3])
            epochs = int(sys.argv[4])
            if folds < 0 or epochs < 1:
                raise ValueError
        except ValueError:
            print("Please specify valid number of folds and epochs", file=sys.stderr)
            print(HELPTEXT)
            sys.exit(1)

    csv_dataset = dataset_utils.CSVDataset(dataset_path, label_column, chunk_size=3e+6)
    csv_dataset.load()
    X, y, category_map, feature_names = csv_dataset.X, csv_dataset.y, csv_dataset.categories, csv_dataset.features
    dataset = TensorDataset(X, y)
    batch_size = 1500
    num_features = X.shape[1]
    num_classes = len(category_map)
    metric = MulticlassAccuracy(average=None, num_classes=num_classes, device=DEVICE)
    print(f"# of rows: {X.shape[0]}, # of features: {num_features}, # of classes: {num_classes}, datatype: {X.dtype}")
    if folds == 0:
        print("Training and testing model with a random split of 80% train and 20% test!")
        model = MyModel(num_features, num_classes).to(DEVICE)
        #torchinfo.summary(model, (batch_size, num_features))
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=10)
        test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=10)
        print("STARTING TRAINING SESSION!!!")
        train_model(model, train_loader, epochs=epochs, device=DEVICE)
        print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
        final_model = MyModel(num_features, num_classes).to(DEVICE)
        final_model.load_state_dict(torch.load("trained_models/best_model.pt", weights_only=False))
        multiclass_accuracy = test_model(final_model, test_loader, metric, device=DEVICE)
        print("Accuracy per class:")
        for i, class_accuracy in enumerate(multiclass_accuracy):
            print(f"{category_map[i]}: {class_accuracy*100} %")
    else:
        strat_kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        final_model = None
        for fold, (train_index, test_index) in enumerate(strat_kfold.split(X, y)):
            model = MyModel(num_features, num_classes).to(DEVICE)
            print("-"*50)
            print(f"Fold {fold+1}/{folds}")
            train_dataset = Subset(dataset, train_index)
            test_dataset = Subset(dataset, test_index)
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=10)
            test_loader = DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=10)
            print("STARTING TRAINING SESSION!!!")
            train_model(model, train_loader, epochs=epochs)
            print("TRAINING COMPLETE. STARTING TESTING SESSION!!!")
            final_model = MyModel(num_features, num_classes).to(DEVICE)
            final_model.load_state_dict(torch.load("trained_models/best_model.pt", weights_only=False))
            multiclass_accuracy = test_model(final_model, test_loader, metric)
            print("Accuracy per class:")
            for i, class_accuracy in enumerate(multiclass_accuracy):
                print(f"{category_map[i]}: {class_accuracy*100} %")
            print("-"*50)
        
    final_model = final_model.to("cpu")
    shap_batch_loader = DataLoader(dataset, 110, shuffle=True)
    features, _ = next(iter(shap_batch_loader))
    features = features.detach().cpu()
    final_model.eval()
    background = features[:100]
    test_values = features[100:]
    with torch.no_grad():
        base_values = final_model(background).mean(dim=0).numpy()
    explainer = shap.GradientExplainer(final_model, background)
    shap_values = explainer.shap_values(test_values)
    print("base values", base_values)
    print("features", features.shape[1])
    print("test values shape", test_values.shape)
    charts_per_row = 2
    rows = num_classes // charts_per_row + ((num_classes % charts_per_row) != 0)
    fig, axes = plt.subplots(rows, charts_per_row, dpi=300, figsize=(charts_per_row*4, rows*3), constrained_layout=True)
    axes = axes.ravel()
    for i in range(num_classes):
        # Create Explanation object for class 0 (you can loop for others)
        shap_explanation = shap.Explanation(
            values=shap_values[i].T,                           # SHAP values for class i
            base_values=base_values[i],                      # base value for class i
            data=features[100:].numpy(),                  # input data
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
    plt.show()