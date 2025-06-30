#!/bin/bash

#SBATCH --job-name=cyber_transformer_tuning
#SBATCH --nodes=1
#SBATCH --partition=ampere
#SBATCH --output=/home/m/mspyrido/thesis-task/logs/out_%x_%j.log
#SBATCH --error=/home/m/mspyrido/thesis-task/logs/error_%x_%j.log
#SBATCH --mem=36G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mspyrido@ece.auth.gr

module load gcc/13.2.0 python/3.11 cuda/12.4.0

cd ~/thesis-task
source .venv/bin/activate

model="$1"
dataset_directory="$2"
label_column="$3"
ray_temp_dir="/mnt/cn50_nvme/tmp/mspyrido/ray"
port='6369'

mkdir -p "$ray_temp_dir"

export RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0

ip_head="127.0.0.1:$port"
redis_password=$(uuidgen)

export TUNER_HEAD_IP_ADDRESS="$ip_head" TUNER_REDIS_PASSWORD="$redis_password"

ray start --block --head --port=$port --redis-password="$redis_password" --temp-dir="$ray_temp_dir" \
 --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS}" --include-dashboard=False & # Starting the head

echo "Waiting for ray head node to start..."
sleep 10

python3 -u model_tuner.py SLURM "$model" "$dataset_directory" "$label_column"

ray stop -g 15
deactivate
rm -r "$ray_temp_dir"
exit