#!/bin/bash

#SBATCH --job-name=cyber_transformer_classifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_%x_%j.log
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <run_mode> <config-file> <model> <dataset> <label_column>"
    exit 1
fi

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

run_mode="$1"

shopt -s nocasematch

if [[ "$run_mode" =~ ^few_shot.* ]]; then
    samples_per_class="$2"
    config_file="$3"
    model="$4"
    dataset="$5"
    label_column="$6"
else
    config_file="$2"
    model="$3"
    dataset="$4"
    label_column="$5"
fi

if [[ "$model" == "transformer" ]]; then
    epochs=30
else
    epochs=5
fi

if [[ "$run_mode" == "zero_shot" ]]; then
    python3 -u model_runner.py -c "$config_file" -e --zero-shot "$model" -d "$dataset" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "zero_shot_remove_features" ]]; then
    python3 -u model_runner.py -c "$config_file" -r -e --zero-shot "$model" -d "$dataset" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "normal" ]]; then
    python3 -u model_runner.py -c "$config_file" -e "$model" -f "$dataset" "$label_column" 10 10 $epochs
elif [[ "$run_mode" == "normal_remove_features" ]]; then
    python3 -u model_runner.py -c "$config_file" -r -e "$model" -f "$dataset" "$label_column" 10 10 $epochs
elif [[ "$run_mode" == "few_shot" ]]; then
    python3 -u model_runner.py -c "$config_file" -e --few-shot "$samples_per_class" "$model" -d "$dataset" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "few_shot_remove_features" ]]; then
    python3 -u model_runner.py -c "$config_file" -r -e --few-shot "$samples_per_class" "$model" -d "$dataset" "$label_column" 10 1 $epochs
else
    echo "Invalid run mode $run_mode! Valid options are: normal, normal_remove_features, zero_shot, zero_shot_remove_features, few_shot, few_shot_remove_features"
    exit 1
fi
