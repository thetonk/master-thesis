#!/bin/bash

#SBATCH --job-name=mpl_net_transformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --output=/home/m/mspyrido/thesis-task/out.log
#SBATCH --error=/home/m/mspyrido/thesis-task/error.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

python3 models.py