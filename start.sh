#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -n 32
#SBATCH --gpus=a100:1
#SBATCH -p gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -A brehm-prj-eng
###SBATCH --mail-type=ALL
###SBATCH --mail-user=jamcq@umd.edu
#SBATCH --exclusive
####SBATCH --exclude=gpu-b11-5,gpu-b11-2,gpu-b10-3
###source ~/.bashrc
###source ~/switch-modules.x

python main.py > ./output/trace.log