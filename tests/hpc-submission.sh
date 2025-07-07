#!/bin/bash

#SBATCH --job-name=cyber_transformer_classifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_%x_%j.log
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <run_mode> <model> <dataset> <label_column>"
    exit 1
fi

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

run_mode="$1"
model="$2"
dataset="$3"
label_column="$4"

shopt -s nocasematch

if [[ "$model" == "transformer" ]]; then
    epochs=10
else
    epochs=5
fi

if [[ "$run_mode" == "zero_shot" ]]; then
    python3 -u model_runner.py --zero-shot "$model" -d "$dataset" "$label_column" 3 1 $epochs
elif [[ "$run_mode" == "normal" ]]; then
    python3 -u model_runner.py "$model" -f "$dataset" "$label_column" 10 10 $epochs
else
    echo "Invalid run mode $run_mode! Valid options are: normal, zero_shot"
    exit 1
fi
