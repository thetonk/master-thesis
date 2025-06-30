#!/bin/bash

#SBATCH --job-name=cyber_transformer_classifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_%x_%j.log
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

model="$1"
dataset="$2"
label_column="$3"

python3 -u model_runner.py "$model" "$dataset" "$label_column" 10 10 5
