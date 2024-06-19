#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out

# exit when any command fails
set -e

source /srv/beegfs-benderdata/scratch/$USER/data/conda/etc/profile.d/conda.sh
conda activate py39
cd /srv/beegfs-benderdata/scratch/$USER/data/cvaiac2023-project-2/problem2_and_3/

wandb login e60a5617bf29d3ef36de0a7c3d35a7d67222f588

python train.py
