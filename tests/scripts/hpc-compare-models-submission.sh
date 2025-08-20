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
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

dataset_file="$1"
transformer_config="$2"
transformer_epochs="$3"
lstm_config="$4"
lstm_epochs="$5"
label_column="$6"
num_runs="$7"
num_folds="$8"

python3 -u models_comparator.py -r -f "$dataset_file" -tc "$transformer_config" -te "$transformer_epochs" -lc "$lstm_config" -le "$lstm_epochs" "$label_column" "$num_runs" "$num_folds"
