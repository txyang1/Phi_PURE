#!/bin/bash
#SBATCH --job-name=phi_pure_1.5b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1_%j.out
#SBATCH --error=logs/run1_%j.err

source ~/.bashrc
conda activate pure

mkdir -p logs
export OMP_NUM_THREADS=32

export WANDB_MODE=online
export WANDB_PROJECT=phi_pure_rloo_1,5b
export WANDB_NAME=phi_pure_rloo_1.5b_run1

python verl/trainer/main_ppo.py 