#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=12000M
#SBATCH --mail-type=ALL,TIME_LIMIT,BEGIN,END,FAIL
#SBATCH --mail-user=arorakus@mila.quebec
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.out
###########################

set -x
module load python/3.7
module load cuda/11.2
export NUM_GPUS=${NUM_GPUS:=2}
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.:${PYTHONPATH}
source ~/envs/ews/bin/activate
$@
