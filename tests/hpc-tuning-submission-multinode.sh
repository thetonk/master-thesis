#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=12:00:00
#SBATCH -p ampere #Run on partition
#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=mspyrido@ece.auth.gr #email recipient

let "worker_num=(${SLURM_NTASKS} - 1)"

# Define the total number of CPU cores available to ray
let "total_cores=${worker_num} * ${SLURM_CPUS_PER_TASK}"

module load gcc/13.2.0 python/3.11 cuda/12.4.0
cd ~/thesis-task
source .venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head

sleep 5

# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for (( i=1; i<=$worker_num; i++ ))
  do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers

  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

python -u trainer.py $redis_password ${total_cores} # Pass the total number of allocated CPUs
