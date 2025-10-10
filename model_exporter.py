# Copyright (C) 2025 Spyridon Baltsas
# This file is part of the research project "Cyberattack detection on network level using state-of-the-art deep learning models"
# Licensed under the GNU General Public License v3.0 (GPLv3)
# See the LICENSE file in the project root for full license text.

import argparse
import os
import torch
import json
from utils.models import MyLSTMClassifier, MyModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="Convert pytorch models to ONNX format")
    parser.add_argument("-c", "--config", type=str, required=True, help="Model hyperparameters configuration file")
    parser.add_argument("-o", "--output-file", type=str, required=True, help="Path of output file")
    parser.add_argument("-r", action='store_true', help="Do not include network specific features", dest="remove_features")
    parser.add_argument("model", choices=("transformer", "lstm"), help="Model type")
    args = parser.parse_args()
    output_file = args.output_file
    config_file = args.config
    model_type = args.model
    NUM_FEATURES = 74 if args.remove_features else 81
    NUM_CLASSES = 2
    if os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(config_file, "r") as file:
        json_data = json.load(file)
        config = json_data["config"]
        del config["lr"], config["batch_size"]
    model_hyperparameters = config
    if model_type == "transformer":
        model = MyModel(NUM_FEATURES, NUM_CLASSES, **model_hyperparameters)
    else:
        model = MyLSTMClassifier(NUM_CLASSES, **model_hyperparameters, device="cpu")
    dummy_input = (torch.randn(256, NUM_FEATURES),)
    onnx_program = torch.onnx.export(model, dummy_input, dynamo=True)
    onnx_program.save(output_file)