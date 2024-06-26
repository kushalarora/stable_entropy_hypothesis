#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --mem=48000M
#SBATCH --mail-type=ALL,TIME_LIMIT,BEGIN,END,FAIL
#SBATCH --mail-user=arorakus@mila.quebec
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:3g.39gb:1
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
###########################

set -x
module load cuda
module load python/3.8 
module load libffi

export NUM_GPUS=${NUM_GPUS:=2}
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.:${PYTHONPATH}
source  ${HOME}/scratch/envs/ead/bin/activate
$@

