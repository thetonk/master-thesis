#!/bin/bash

#SBATCH --job-name=cyber_transformer_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_%x_%j.log
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

# run hyperparameter tuning with ray tune using apptainer to avoid permission errors.

cd ~/thesis-task

model="$1"
dataset_directory="$2"
label_column="$3"

module load gcc/14.2.0 apptainer/1.3.4

apptainer exec --nv --pid --writable-tmpfs /scratch/m/mspyrido/my-thesis-image.sif python3 -r local "$model" "$dataset_directory" "$label_column"

exit
