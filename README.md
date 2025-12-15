# Master thesis

The code and results repository of my thesis "Cyberattack detection on network level using state-of-the-art deep learning models" for my Electrical and Computer Engineering diploma.

## Licensing

All source code files (Python scripts and Jupyter notebooks) are licensed under the [GNU General Public License v3.0](./LICENSE). All diagrams, figures and result reports in the `results` directory are licensed under the [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0). You may share and adapt these materials for **non-commercial purposes**, provided you give appropriate credit and distribute any derivatives under the same license. For **commercial use or redistribution**, please contact the author to obtain written permission.

## Project setup
In order to run the experiments there are some python packages that need to be installed first. There are 2 ways to install them on project level and not system-wide.

- Installing via `uv` project manager (**RECOMMENDED**). After cloning the repository, run the command `uv sync` to download all required packages. `uv` will take care of virtual environment creation, management and correct python versioning. 
- Installing via `pip` directly. Tested on Linux systems with minimum python version 3.11. For this method a few more steps are required.
    1. Set up a python virtual environment by using the following commands;
        ```
       $ python3 -m venv .venv
       $ source .venv/bin/activate
       ```
    2. Install the packages using the command `pip install -r requirements.txt`.

## Running the experiments
On the top level of this project there are 4 python scripts. You may either run them by running `uv run <scriptfile>` if you use `uv`, or by running `python3 <scriptfile>` while having the virtual environment you created activated.
The following table matches the script files with their designed experiment usages.

| Script                                        | Experiment usage                                                                  |
|-----------------------------------------------|-----------------------------------------------------------------------------------|
| [model_runner.py](./model_runner.py)          | Transfer learning and cross domain experiments.                                   |
| [model_tuner.py](./model_tuner.py)            | Model hyperparameter tuning                                                       |
| [model_comparator.py](./models_comparator.py) | Model scaling experiments, compares the two models.                               |
| [model_exporter.py](./model_exporter.py)      | Converts the pytorch trained model files (.pt, .pth) to the standard ONNX format. |

Each script is flexible and has multiple options available. To view the available options of each script, you may consult their available help texts. You may read them by running `uv run <scriptfile> --help`, or `python3 <scriptfile> --help`.
Please note that for the model hyperparameter tuning, search space is limited. The possible values of the hyperparameters of each model are available on the following tables.

### Transformer-based model
| Hyperparameter name  | Values                               |
|----------------------|--------------------------------------|
| `lr`                 | $10^{-4}, 10^{-3}, 10^{-2}$          |
| `batch_size`         | 32, 64, 128, 256                     |
| `enc_embedding_dim`  | 32, 64, 128, 256                     |
| `ecn_num_heads`      | 4, 8, 16, 32                         |
| `enc_ff_neurons`     | 64, 128, 256, 512                    |
| `enc_ff_dropout`     | 0, 0.1, 0.2, 0.3                     |
| `enc_attn_dropout`   | 0, 0.1, 0.2                          |
| `mlp_hidden_neurons` | 128, 256, 512, 1024                  |
| `num_encoders`       | 1, 2, 3, 4                           |
| `num_mlps`           | 1, 2, 3, 4                           |
| `mlp_dropout`        | 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 |

### LSTM-based model
| Hyperparameter name  | Values                               |
|----------------------|--------------------------------------|
| `lr`                 | $10^{-4}, 10^{-3}, 10^{-2}$          |
| `batch_size`         | 32, 64, 128, 256                     |
| `hidden_lstm_states` | 128, 256, 512                        |
| `hidden_mlp_neurons` | 128, 256, 512, 1024                  |
| `mlp_dropout`        | 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 |

## Experiment results
Results are located under `results` folder and contains the following directories:
- `images/`: contains the generated plots and diagrams for all experiments. For thesis results presentation purposes the subdirectories that were uses are `with_network_specific_features/` and `without_network_specific_features_improved/`.

| Directory name              | Results type                                                                      |
|-----------------------------|-----------------------------------------------------------------------------------|
| `few_shot`                  | Few-shot TL experiment results. Contains cross-domain experiment results as well  |
| `zero_shot`                 | Zero-shot TL experiment results. Contains cross-domain experiment results as well |
| `model_scaling_plots`       | Model scaling - statistical importance experiment results                         |
| `overall_performance_plots` | Few general model performance plots                                               |

- `raytune/`: contains various hyperparameter configurations obtained from Ray Tune.
- `reports/`: contains the csv files of the experiment results
- `logs/`: contains various training logs. Can be ignored.
