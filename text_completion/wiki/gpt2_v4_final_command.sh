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
source  ${HOME}/scratch/envs/ews2/bin/activate

python text_completion/wiki/generate_from_gpt2.py --fp16 --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.0 -0.00215 2.87399 --ea_human_std_coeffs 2e-05 -0.00373 0.97922 --batch_size 4 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} $@

python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}
