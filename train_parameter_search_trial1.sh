#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH -c 16
# SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu-h100-80g-short
#SBATCH --partition=gpu-a100-80g
# SBATCH --partition=gpu-debug
# SBATCH --partition=gpu-h200-141g-short
# SBATCH --partition=gpu-h200-141g-ellis

# SBATCH --partition=gpu-amd
#SBATCH -o /scratch/work/sethih1/slurm_logs_planar_again/norm_slurm_0.1.out

arg1=$1


echo $CUDA_VISIBLE_DEVICES


# Load Environments
source /scratch/phys/sin/sethih1/venv/masque_env/bin/activate

# Start GPU monitoring: Log GPU usage every 2 seconds into gpu_usage.log
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,temperature.gpu \
         --format=csv -l 2 > /scratch/work/sethih1/slurm_logs_0.1/gpu_usage.log &
GPU_MONITOR_PID=$!

# Optionally, start system resource monitoring: Log CPU, memory, etc. every 1 second
vmstat -n 1 > /scratch/work/sethih1/slurm_logs_0.1/resource_usage.log &
RESOURCE_MONITOR_PID=$!

# Run your main Python job
# python parameter_search.py --config $arg1

export WANDB_API_KEY=8e4e0db2307a46c329b7d30d5f7ab11a176ba158
python /home/sethih1/masque_new/ters_gen/hyperopt.py --config $arg1 --use_wandb 
# python check_train.py --config $arg1

# After your job finishes, stop the monitoring processes
kill $GPU_MONITOR_PID
kill $RESOURCE_MONITOR_PID
