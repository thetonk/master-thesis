#!/bin/bash

#SBATCH --job-name=cyber_classifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_model_runner_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_model_runner_%x_%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

run_mode="$1"

shopt -s nocasematch

if [[ "$run_mode" =~ ^few_shot.* ]]; then
    samples_per_class="$2"
    config_file="$3"
    model="$4"
    if [[ "$run_mode" == *custom ]]; then
      train_dataset="$5"
      test_dataset="$6"
      label_column="$7"
      shift 7
    else
      dataset="$5"
      label_column="$6"
      shift 6
    fi
else
    config_file="$2"
    model="$3"
    dataset="$4"
    label_column="$5"
    shift 5
fi

if [[ "$model" == "transformer" ]]; then
    epochs=30
else
    epochs=5
fi

if [[ "$run_mode" == "zero_shot" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -e --zero-shot -d "$dataset" "$@" default_tt \
     "$model" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "zero_shot_remove_features" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -r -e --zero-shot -d "$dataset" "$@" default_tt \
     "$model" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "normal" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -e -f "$dataset" "$@" default_tt \
     "$model" "$label_column" 10 10 $epochs
elif [[ "$run_mode" == "normal_remove_features" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -r -e -f "$dataset" "$@" default_tt \
     "$model" "$label_column" 10 10 $epochs
elif [[ "$run_mode" == "few_shot" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -e --few-shot "$samples_per_class" -d "$dataset" "$@" default_tt \
     "$model" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "few_shot_remove_features" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -r -e --few-shot "$samples_per_class" -d "$dataset" "$@" default_tt \
     "$model" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "zero_shot_custom" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -e --zero-shot "$@" custom_tt \
     --train-dir "$train_dataset" --test-dir "$test_dataset" "$model" "$label_colun" 10 1 $epochs
elif [[ "$run_mode" == "zero_shot_remove_features_custom" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -r -e --zero-shot "$@" custom_tt \
     --train-dir "$train_dataset" --test-dir "$test_dataset" "$model" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "few_shot_custom" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -e --few-shot "$samples_per_class" "$@" custom_tt \
     --train-dir "$train_dataset" --test-dir "$test_dataset" "$model" "$label_column" 10 1 $epochs
elif [[ "$run_mode" == "few_shot_remove_features_custom" ]]; then
    srun python3 -u model_runner.py -c "$config_file" -b -r -e --few-shot "$samples_per_class" "$@" custom_tt \
     --train-dir "$train_dataset" --test-dir "$test_dataset" "$model" "$label_column" 10 1 $epochs
else
    echo "Invalid run mode $run_mode!"
    exit 1
fi
