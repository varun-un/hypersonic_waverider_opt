#!/bin/bash
#SBATCH -t 24:00:00
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

module load python/3.10.10
python -mvenv --system-site-packages ~/my-venv
source ~/venv/bin/activate
pip install -r requirements.txt
python main.py | tee ./output/trace.log