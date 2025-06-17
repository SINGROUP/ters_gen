#!/bin/bash
#SBATCH --time=61:00:00
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --partition=gpu-h100-80g-short
# SBATCH --partition=gpu-debug
#SBATCH -o /home/sethih1/masque_new/ters_gen/log_file/slurm_%x.out

arg1=$1

# Load Environments
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

# Start GPU monitoring: Log GPU usage every 2 seconds into gpu_usage.log
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,temperature.gpu \
         --format=csv -l 2 > gpu_usage.log &
GPU_MONITOR_PID=$!

# Optionally, start system resource monitoring: Log CPU, memory, etc. every 1 second
vmstat -n 1 > resource_usage.log &
RESOURCE_MONITOR_PID=$!

# Run your main Python job
# python parameter_search.py --config $arg1
python /home/sethih1/masque_new/ters_gen/hyperopt.py --config $arg1
# python check_train.py --config $arg1

# After your job finishes, stop the monitoring processes
kill $GPU_MONITOR_PID
kill $RESOURCE_MONITOR_PID
