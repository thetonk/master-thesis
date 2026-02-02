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

if [[ $# -lt 9 ]]; then
  echo "Invalid number of arguments! You must enter at least 8 arguments!"
  echo "Usage: $0 <run_mode> <dataset_directory> <model1_config> <model1_epochs> <model2_config> <model2_epochs> <model1_type> <model2_type> <label_column> <num_runs> <num_folds>"
  echo "Run mode can be either 'sequential' or 'parallel'"
fi

run_mode="$1"
dataset_dir="$2"
model1_config="$3"
model1_epochs="$4"
model2_config="$5"
model2_epochs="$6"
model1_type="$7"
model2_type="$8"
label_column="$9"
num_runs="${10}"
num_folds="${11}"
shift 11

shopt -s nocasematch

if [[ "$run_mode" == "parallel" ]]; then
  srun python3 -u models_comparator.py -r -e -p -d "$dataset_dir" -fc "$model1_config" -fe "$model1_epochs" -sc "$model2_config" -se "$model2_epochs" "$@" \
    "$model1_type" "$model2_type" "$label_column" "$num_runs" "$num_folds"
else
  srun python3 -u models_comparator.py -r -e -d "$dataset_dir" -fc "$model1_config" -fe "$model1_epochs" -sc "$model2_config" -se "$model2_epochs" "$@" \
    "$model1_type" "$model2_type" "$label_column" "$num_runs" "$num_folds"
fi
