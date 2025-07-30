#!/bin/bash
#SBATCH --job-name=phi_pure_prm_new_1.5b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --time=06:00:00
#SBATCH --output=logs/run1_%j.out
#SBATCH --error=logs/run1_%j.err

source ~/.bashrc
conda activate verl

mkdir -p logs
export OMP_NUM_THREADS=32

export WANDB_MODE=online
export WANDB_PROJECT=phi_pure_rloo_prm_new_1,5b
export WANDB_NAME=phi_pure_rloo_prm_new_1.5b_run1

python verl/trainer/main_ppo_tx.py 