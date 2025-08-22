#!/bin/bash

#SBATCH --job-name=cyber_model_comparator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_model_comparator_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_model_comparator_%x_%j.log
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

if [[ $# -ne 9 ]]; then
  echo "Invalid number of arguments! You must enter 8 arguments!"
  echo "Usage: $0 <run_mode> <dataset_directory> <transformer_config> <transformer_epochs> <lstm_config> <lstm_epochs> <label_column> <num_runs> <num_folds>"
  echo "Run mode can be either 'sequential' or 'parallel'"
fi

run_mode="$1"
dataset_dir="$2"
transformer_config="$3"
transformer_epochs="$4"
lstm_config="$5"
lstm_epochs="$6"
label_column="$7"
num_runs="$8"
num_folds="$9"

shopt -s nocasematch

if [[ "$run_mode" == "parallel" ]]; then
  python3 -u models_comparator.py -r -p -d "$dataset_dir" -tc "$transformer_config" -te "$transformer_epochs" -lc "$lstm_config" -le "$lstm_epochs" "$label_column" "$num_runs" "$num_folds"
else
  python3 -u models_comparator.py -r -d "$dataset_dir" -tc "$transformer_config" -te "$transformer_epochs" -lc "$lstm_config" -le "$lstm_epochs" "$label_column" "$num_runs" "$num_folds"
fi