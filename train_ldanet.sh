#!/bin/bash
#SBATCH --array=0-4
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-23:59
#SBATCH --output=%N-%j.out


module load python/3.6 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env_pytorch
source $SLURM_TMPDIR/env_pytorch/bin/activate
pip install numpy --no-index

pip install scipy --no-index
pip install scikit-learn --no-index
pip install seaborn --no-index
pip install torch torchvision --no-index

python ldanet.py --num_classes 15 --gamma 0.01 --gamma2 0.01 --split_k $SLURM_ARRAY_TASK_ID >results/ldanet_gamma001_001_$SLURM_ARRAY_TASK_ID.txt 2>&1 

